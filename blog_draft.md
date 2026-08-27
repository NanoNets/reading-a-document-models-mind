# BLOG DRAFT v1 — "What is your document AI thinking before it types?"

*Non-technical piece. Spine: scrubber (lite) → schema tapestry → the crossing. One stat tile: 88%. No stop-token content, no shortcoming framing. ~800 words.*

---

**Standfirst:** AI models are famously black boxes — even their makers can't open them up and read the code, because there is no code inside, only billions of numbers. This summer, Anthropic published an instrument that translates some of those numbers into plain words. We built the first fully-validated version of it for a model that can *see* — and pointed it at our own document AI. Here's what it's thinking about your invoice before it writes a single character.

## The strangest thing about AI models

If you've worked in software, you know that when a program misbehaves, someone can open the code and find the reason. AI models broke that deal. A model that reads your invoices is not a program anyone wrote — it's billions of numbers that arranged themselves during training. It plainly *knows* things. But nothing inside it is written in any language a human can read.

A young scientific field — AI interpretability — exists to fix that, by building instruments the way biology built microscopes. In July, [Anthropic published one of the best yet](https://www.anthropic.com/research/global-workspace): a "lens" that translates a model's internal states into ordinary words, revealing thoughts that never appear in its output. It only worked on text models. Our models look at pixels. So we built the missing version — and turned it on Nanonets-OCR2.

## Watch it think

`[VISUAL 1 — the scrubber (lite, one document: the invoice). Caption: "Drag through the model's 35 thinking steps. It hasn't typed anything yet — every word here is read directly from its internal state."]`

Give the model an invoice and freeze it before it responds. Then slide through its processing stages. For the first few, almost nothing — it's still *looking*. Then, around a third of the way in, the readings snap into focus: **Date. Name. Address. Number. Page.** The model has recognized what it's holding and pulled up the vocabulary of invoices — fields it expects to find, most of them not yet read off the page. By the final stages the readings change character again: now it's arranging the exact table format it will type. Looking, understanding, writing — three phases you can watch.

## Every document type summons its own mind

`[VISUAL 2 — schema tapestry: 3-4 document genres × their signature inner vocabulary as token chips. Caption: from real measurements across our test documents.]`

Show it a page of equations instead, and the inner vocabulary changes completely — mathematical scaffolding, LaTeX symbols. Handwriting summons prose words. A form summons field labels. The model doesn't process all documents the same way and then notice differences; it *becomes a different specialist* for each genre, before writing anything.

And one more thing we didn't expect: on an English receipt, among Date and Total, the readings also surface **金额** — Chinese for "amount." The model's inner voice is multilingual even when the document isn't. It learned "amount-ness" as an idea bigger than any one language.

**STAT TILE: 88% — the share of the model's most influential internal signals that have no name in its vocabulary at the strongest thinking layer. Models know far more than they say; words are just the export format. That hidden layer is where our reliability research lives.**

## Fine-tuning, X-rayed

`[VISUAL 3 — the crossing. Caption: blue = how similar the two models' translation machinery is, layer by layer. Orange = how similar their actual thoughts are. New reader, same typewriter.]`

Nanonets-OCR2 is built from an open base model plus our document training — which let us do something new: point the lens at *both* and compare minds. The result is one of the cleanest pictures of what fine-tuning actually does that we've seen anywhere. Our training rewired the model's early *reading* machinery — where document expertise lives — while leaving its *writing* machinery essentially untouched. Expertise went exactly where it should.

That's what buying a specialized model gets you, visible for the first time at layer resolution: not a generic model with a prompt taped on, but one whose reading apparatus was rebuilt around documents.

## Why we do this

Understanding *why* a model behaves as it does is how you make it reliable — and it's a lot more fun than treating models as appliances. The full study, with methods, all measurements, and an interactive version of everything above, is in our research note. The instruments themselves — the first validated lenses for any vision-language model — are open on HuggingFace and GitHub. Point them at your own documents.

`[CTA row: Read the research note → · Get the lenses → · Nanonets-OCR2 on HuggingFace →]`

---
*Built on the Jacobian-lens method of Gurnee, Sofroniew, Lindsey et al. (Anthropic, 2026), with thanks.*
