# Figure registry — every paper figure is a script reading only evidence tables

No GPU or model download needed: each script reads exclusively from
`artifacts/evidence/tables/*.json` (shipped) and writes `.png` + `.svg` next to
itself. Run from anywhere; paths are resolved relative to the script.

| id | script | data (tables/) | regenerate |
|---|---|---|---|
| fig1 | `fig1_band_effdim.py` | `E1_effdim_band.json`, `E1_merged_lens_anatomy.json` | `python artifacts/evidence/figures/fig1_band_effdim.py` |
| fig2 | `fig2_b_vs_loading.py` | `E2_B_kappa.json`, `E2_H1_stats_v4.json` | `python artifacts/evidence/figures/fig2_b_vs_loading.py` |
| fig3 | `fig3_intervention_battery.py` | `E7_dose_ladder.json`, `E6_text_control_v2.json`, `E4_discriminator.json`, `E8_nameless_a8.json`, `E8_nameless_a16.json` | `python artifacts/evidence/figures/fig3_intervention_battery.py` |
| fig4 | `fig4_twin_timeline.py` | `E1_twin_vs_base.json`, `e1merged_diag.json`, `twinmerged_diag.json` | `python artifacts/evidence/figures/fig4_twin_timeline.py` |
| fig5 | `fig5_chiasmus.py` | `E1_twin_vs_base.json`, `exploratory/chiasmus_state_cos.json` | `python artifacts/evidence/figures/fig5_chiasmus.py` |

The H1 statistics table (`E2_H1_stats_v4.json`, read by fig2) is itself
regenerable from the raw E2 arrays:
`python code/compute_h1_stats.py artifacts/evidence/tables/E2_B_kappa.json <out.json>`.
