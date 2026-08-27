# Qwen2.5-VL adapter for the jlens LensModel protocol.
#
# Design: the Jacobian lens is defined over the TEXT DECODER's residual stream
# (h_l -> h_final), so the vision tower needs no gradients. encode() runs the
# HF processor (image + text) once per prompt and caches the merged
# inputs_embeds keyed by the input_ids; forward() replays the cached embeds
# through the decoder, expanding along the batch axis when the fitting
# estimator replicates the prompt.
#
# Prompt convention (keeps jlens.fit(prompts=[str]) unchanged):
#   "IMG::<abs_image_path>::<instruction text>"   -> multimodal
#   any other string                              -> text-only
#
# unembed(): Nanonets-OCR2-3B (revision c3886ff0) ships NO lm_head.weight and
# its top-level config mis-sets tie_word_embeddings=False, so HF random-inits
# lm_head. We detect missing/untied heads and re-project through embed_tokens
# (the intended tied projection). Pin the model revision (see README) so a
# future config fix upstream cannot silently change which head loads.
from __future__ import annotations
import os
import torch
from torch import nn


def _find_decoder(hf_model):
    """Locate (text_decoder_layers_container, final_norm, embed_tokens)."""
    cands = []
    for name, mod in hf_model.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            parent_name = name.rsplit(".", 1)[0]
            parent = hf_model.get_submodule(parent_name)
            if hasattr(parent, "norm") and hasattr(parent, "embed_tokens"):
                cands.append((name, parent))
    # prefer the language decoder over any vision stack (vision has no embed_tokens,
    # so surviving candidates are text decoders; take the deepest/last match)
    if not cands:
        raise RuntimeError("no decoder layers found")
    name, parent = cands[-1]
    return parent.layers, parent.norm, parent.embed_tokens


def _resolve_unembed(hf_model, embed_tokens):
    """Return a weight matrix [vocab, d] that is REAL, never random-init."""
    head = getattr(hf_model, "lm_head", None)
    if head is not None and isinstance(head, nn.Linear):
        w = head.weight
        # tied or genuinely trained -> cosine to embed rows is high for tied,
        # but a random-init head has near-zero overlap with embeddings AND
        # near-uniform singular structure. Cheap test: max |cos| of a few rows.
        with torch.no_grad():
            idx = torch.arange(0, min(1000, w.shape[0]), 97)
            a = torch.nn.functional.normalize(w[idx].float(), dim=-1)
            b = torch.nn.functional.normalize(embed_tokens.weight[idx].float(), dim=-1)
            tied_cos = (a * b).sum(-1).mean().item()
        if tied_cos > 0.5:
            return w, f"lm_head (tied-check cos={tied_cos:.3f})"
        # untied but plausibly trained head: accept only if config says untied on purpose
        cfg = getattr(hf_model, "config", None)
        txt = getattr(cfg, "text_config", cfg)
        if txt is not None and getattr(txt, "tie_word_embeddings", False):
            return embed_tokens.weight, f"embed_tokens.T (config tied; lm_head cos={tied_cos:.3f} -> random-init suspected)"
        return w, f"lm_head (untied by config; cos={tied_cos:.3f}) -- VERIFY for this checkpoint"
    return embed_tokens.weight, "embed_tokens.T (no lm_head module)"


class QwenVLLensModel:
    """LensModel over the text decoder of a Qwen2.5-VL-family checkpoint."""

    def __init__(self, hf_model, processor, device=None):
        self.hf = hf_model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device or next(hf_model.parameters()).device
        self.layers, self._norm, self._embed = _find_decoder(hf_model)
        self._wu, self._wu_source = _resolve_unembed(hf_model, self._embed)
        self.n_layers = len(self.layers)
        self.d_model = self._embed.weight.shape[1]
        self._cache: dict[tuple, torch.Tensor] = {}   # ids-key -> inputs_embeds [1,T,d]
        print(f"[vl-adapter] n_layers={self.n_layers} d_model={self.d_model} unembed={self._wu_source}", flush=True)

    # -- protocol --------------------------------------------------------
    def encode(self, text: str, *, max_length: int = 8192) -> torch.Tensor:
        if text.startswith("IMG::"):
            _, img_path, instr = text.split("::", 2)
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            msgs = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
            chat = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            enc = self.processor(text=[chat], images=[img], return_tensors="pt").to(self.device)
            ids = enc["input_ids"][:, :max_length]
            with torch.no_grad():
                embeds = self._merged_embeds(enc)[:, :max_length]
        else:
            ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=max_length)["input_ids"].to(self.device)
            with torch.no_grad():
                embeds = self._embed(ids)
        # single-entry cache: jlens.fit processes prompts strictly sequentially,
        # and an unbounded cache accumulates every document's embeds on-GPU
        # (observed to OOM a 141GB H200 mid-shard before this was bounded).
        self._cache.clear()
        self._cache[self._key(ids)] = embeds.detach()
        return ids

    def forward(self, input_ids: torch.Tensor):
        embeds = self._cache.get(self._key(input_ids[:1]))
        if embeds is None:                       # text-only fallback path
            embeds = self._embed(input_ids)
        elif input_ids.shape[0] > 1:             # fitting replicates along batch
            embeds = embeds.expand(input_ids.shape[0], -1, -1)
        return self._run_decoder(embeds)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(self._norm(residual), self._wu.to(residual.dtype))

    # -- internals -------------------------------------------------------
    def _key(self, ids):  # first row identifies the prompt
        t = ids[0]
        return (t.shape[0], int(t.sum().item()), int(t[:64].prod(dtype=torch.long).item() & 0x7FFFFFFF))

    def _merged_embeds(self, enc):
        """Vision tower + embedding merge, exactly as the HF forward does it."""
        m = self.hf
        ids = enc["input_ids"]
        embeds = self._embed(ids)
        pv = enc.get("pixel_values")
        if pv is not None:
            grid = enc.get("image_grid_thw")
            # version-stable public API (transformers>=5 keeps the tower at
            # m.model.visual; get_image_features abstracts the path)
            owner = m if hasattr(m, "get_image_features") else m.model
            vis = owner.get_image_features(pixel_values=pv, image_grid_thw=grid)
            if isinstance(vis, (list, tuple)):
                vis = torch.cat([v for v in vis], dim=0)
            img_token = int(getattr(m.config, "image_token_id", 151655))
            mask = (ids == img_token).unsqueeze(-1).expand_as(embeds)
            embeds = embeds.masked_scatter(mask, vis.to(embeds.dtype))
        return embeds

    def _run_decoder(self, embeds):
        """Minimal decoder replay: rotary + blocks + (no head)."""
        # Use the HF text model's own forward with inputs_embeds to inherit
        # correct rope/position handling across transformers versions.
        parent = self.layers[0]
        # find the module that owns `layers` again (parent of block 0)
        for name, mod in self.hf.named_modules():
            if getattr(mod, "layers", None) is not None and len(getattr(mod, "layers")) == self.n_layers \
               and hasattr(mod, "norm") and hasattr(mod, "embed_tokens"):
                return mod(inputs_embeds=embeds, use_cache=False).last_hidden_state
        raise RuntimeError("decoder module not found for forward")
