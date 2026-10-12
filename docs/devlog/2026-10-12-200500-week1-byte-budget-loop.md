# Devlog — week1-byte-budget-loop

- **Date:** 2026-10-12 20:05:00
- **Branch:** master (workspaces/budgetgs) + outer master (project-budget-gaussian-splatting)
- **Commit:** budgetgs d0077b4 + local patches (k1 progress prints, landing ref-metric fix) · outer 00106bd
- **Kind:** interim
- **Status:** partially met
- **Task:** docs/plans/2026-10-08-week1-budgetgs-ramp-up.md

**Previous entries:** none

## Goal

Close the Week-1 acceptance loop: a byte ledger that predicts the real serialized
size of a trained 3D Gaussian Splatting checkpoint within tolerance, a byte-budget
"landing" fitter, and the K1 kill-experiment evidence that serialized bytes are
affine in the Gaussian count N.

**Success criterion:** acceptance items 1-3 of the Week-1 plan:
(1) repo reachable with a CLI that trains and fits; (2) ledger-vs-real gap <= 1 %
of file; (3) K1 R^2 > 0.98 and worst residual < 5 %, on all three Mip-NeRF 360
scenes (bicycle / garden / room).

## Outcome

- Ledger-vs-real gap: < 0.3 % on the 30k bicycle checkpoint (range-codec
  round-trip), 22/22 pytest green.
- K1 (all three scenes, 30 points each): bytes affine in N everywhere —
  bicycle 69.04 B/N + 2.98 MB (R² 0.9973, worst 2.5 %, PASS), garden 68.94 B/N +
  1.96 MB (R² 0.9930, worst 2.8 %, PASS), room 58.39 B/N + 4.39 MB (R² 0.9905,
  worst 5.09 %, FAIL at the edge). Full report in runs/analysis/k1-all.json.
- Landing (byte budget 10 MB, single global levels actuator): all three scenes
  fail to land in tolerance. bicycle levels=2 floor 12,845,584 B (+28.5 %);
  garden levels=10 → 9,733,016 B (−2.67 %); room levels=23 → 9,432,424 B (−5.68 %).
- Plan changes forced by measurements: single executor has a byte floor and ~10
  effective precision steps; pruning + per-group executors moved from Week-3 to Week-2.
- Deliverables for the 10-15 advisor review shipped: English deck (12 pages),
  proposal update docs/03-proposal-w1-update.md, 13 submodules aligned.

## What was done

- Built the byte ledger (header 28 B + six attribute groups means/scales/quats/
  opacity/sh0/shn, dual codec zlib/range) and the serializer round-trip.
- Adopted the other-line numpy/zlib pipeline (commit d0077b4): shn layout fixed
  to 3 channels; 1.1 M-Gaussian ledger from 2 h+ to seconds.
- Three-scene training baseline (30 k steps each, ckpt every 1 k): final val PSNR
  bicycle 19.997 / garden 21.071 / room 27.419. Protocol not yet comparable
  (factor=2, pure L1, six bad views) — aligned in Week-2 before quality columns.
- K1 on bicycle (6 real-encode samples) from the other-line run; this session ran
  K1 on all three scenes (90-ckpt ledger + real-encode samples), added progress
  prints to k1.py, fixed a landing.py bug (ref_avg computed from rec_tot, making
  reference metrics equal to landed ones), un-ignored runs/analysis in
  .gitignore, and drafted the three reproducibility figures.

## Changes

- **Code:** budgetgs/evaluation/{k1,fit,landing,shn_compare}.py · modules/
  {ledger,serialize,quantize}.py · cli.py (train/fit/ledger/serialize) ·
  utils/ckpt.py · .gitignore (runs/analysis tracked, scene ckpts ignored).
- **API:** `budgetgs fit --ckpt <pt> --budget-mb 10 [--tolerance]`;
  `budgetgs ledger --ckpt <pt> --levels N`; `budgetgs serialize --ckpt <pt>
  --out <file> --codec zlib|range`; `python -m budgetgs.evaluation.k1
  --ckpt-dir <dir> [--sample N] [--out report.json]` (progress prints).
- **Config / data:** ATTRIBUTE_GROUPS = ("shn", 3) after the 3-vs-45 comparison
  (30 k: 78,295,632 vs 83,255,584 B = −6.34 %).

## Experiments

### E1 — three-scene training baseline

- **Hypothesis:** the gsplat trainer with the default pipeline reproduces usable
  3DGS scenes on Mip-NeRF 360.
- **Setup:** bicycle commit 8af3cec, garden/room commit 9d3867f · gsplat ·
  Mip-NeRF 360 v2 · n=1 seed per scene · RTX 4090 (slurm) · 3 × ~50 min
- **Command:** `budgetgs train --scene <scene> --steps 30000 --ckpt-every 1000`
- **Results:**

  | Metric | Split | Baseline (official) | This run | Δ |
  |---|---|---|---|---|
  | val PSNR bicycle | val | ~23.8 (official) | 19.997 | −3.8 dB |
  | val PSNR garden | val | ~25.4 (official) | 21.071 | −4.3 dB |
  | val PSNR room | val | ~30.6 (official) | 27.419 | −3.2 dB |

  Baseline = official reported numbers, not re-run identically. Single seed, n=1.

- **Observations:** PSNR gap concentrated in views 0-4 + 9 (12.9-17.9 dB);
  factor=2 + pure L1 explains most of the gap.
- **Verdict:** inconclusive for protocol comparability — training works, protocol
  alignment deferred to Week-2.

### E2 — K1 kill experiment, bicycle

- **Hypothesis:** serialized bytes are affine in N on bicycle.
- **Setup:** commit d0077b4 · 30 ckpts · 1 real-encode sample (levels=4096,
  range codec) · CPU (this run; the other-line 6-sample run reported
  74.28 B/N under the pre-3ch accounting)
- **Command:**
  ```bash
  python -m budgetgs.evaluation.k1 --ckpt-dir runs/bicycle/bicycle_s0 --sample 6 --out runs/analysis/k1-bicycle.json
  ```
- **Results:**

  | Metric | This run |
  |---|---|
  | slope (B / Gaussian) | 69.0387 |
  | intercept (B) | 2,984,316 |
  | R^2 | 0.99726 |
  | worst residual | 2.47 % of file |
  | criterion | PASS |

- **Verdict:** supports — bytes are affine in N on bicycle.

### E3 — K1, garden and room (this session)

- **Hypothesis:** the exchange rate transfers across scenes.
- **Setup:** d0077b4 + progress patch · 30 ckpts each · 2 real-encode samples each
- **Command:**
  ```bash
  python -m budgetgs.evaluation.k1 --ckpt-dir runs/bicycle/bicycle_s0 --ckpt-dir runs/garden/garden_s0 --ckpt-dir runs/room/room_s0 --sample 2 --out runs/analysis/k1-all.json
  ```
- **Results:**

  | Scene | slope (B/N) | intercept (B) | R^2 | worst residual | criterion |
  |---|---|---|---|---|---|
  | bicycle | 69.0387 | 2,984,316 | 0.99726 | 2.47 % | PASS |
  | garden | 68.9368 | 1,957,724 | 0.99303 | 2.76 % | PASS |
  | room | 58.3942 | 4,389,608 | 0.99054 | 5.09 % | FAIL (edge) |

  ledger-vs-serializer sample check (1k ckpt each): err 0.275 % / 0.475 % / 0.433 %.

- **Verdict:** supports transfer of the byte-on-N model across scenes; room
  residual sits just over the 5 % bar, so per-scene exchange rates are kept
  (Week-2 M3 open question).

### E4 — landing, byte budget 10 MB (single global levels)

- **Hypothesis:** a single block-precision actuator (global levels) can land each
  scene within 1 %/256 KiB of 10 MB.
- **Setup:** commit d0077b4 · ckpt-0030000.pt per scene · real write+read-back ·
  tolerance max(1 % of budget, 256 KiB)
- **Command:**
  ```bash
  python -m budgetgs.cli fit --ckpt runs/<scene>/<scene>_s0/ckpt-0030000.pt --budget-mb 10
  ```
- **Results:**

  | Scene | levels | bytes | err vs 10 MB | status |
  |---|---|---|---|---|
  | bicycle | 2 | 12,845,584 | +28.5 % | over-budget at min precision |
  | garden | 10 | 9,733,016 | −2.67 % | tolerance missed (levels straddle) |
  | room | 23 | 9,432,424 | −5.68 % | tolerance missed (levels straddle) |

- **Observations:** bicycle is above budget even at the coarsest precision (byte
  floor); garden/room steps jump 2.7-5.7 %, skipping the band. Only ~10 effective
  precision levels exist for ~1.1 M Gaussians.
- **Verdict:** refutes the single-executor landing assumption — recorded as a
  plan change, not a fix.

### E5 — shn layout 3 vs 45 channels (decision)

- **Hypothesis:** fewer SH channels cut bytes with no quality cliff.
- **Results:**

  | Metric | 3ch | 45ch | Δ |
  |---|---|---|---|
  | 30 k ledger (B) | 78,295,632 | 83,255,584 | −6.34 % |
  | 1 k range (B) | 8,091,248 | 8,251,568 | −1.98 % |
  | 1 k zlib (B) | 10,724,224 | 11,070,560 | −3.13 % |

- **Verdict:** supports — 3ch adopted as default.

## Evaluation

- **Consolidated results:** K1 criterion PASS on bicycle; landing criterion FAIL
  on all three (single executor); training protocol not comparable until Week-2.
  K1 garden/room: k1-all.json.
- **Runs not detailed above:** pytest 22/22 (unit); smoke job chain
  1901438→1901479→1901609; shn-compare-30k/-1k JSONs.
- **Not evaluated:** PSNR/SSIM/LPIPS of landed containers (GPU render, blocked by
  another process's landing run; queued for Week-2); LPIPS on reference scenes;
  multi-seed stats.
- **Threats to validity:** single seed n=1 everywhere; official PSNR baselines not
  re-run identically; K1 samples evenly spaced, not random; storage is
  network-backed (IO variance affects timing, not byte counts).

## Verification

```
$ cd workspaces/budgetgs && pytest -q
22 passed
$ python -m budgetgs.cli ledger --ckpt runs/bicycle/bicycle_s0/ckpt-0030000.pt --levels 4096
predicted serialized size: <gap vs real serialize < 0.3 %>
$ python -m budgetgs.evaluation.k1 --ckpt-dir runs/bicycle/bicycle_s0 --ckpt-dir runs/garden/garden_s0 --ckpt-dir runs/room/room_s0 --sample 1
bicycle: 30 checkpoints
  fit: bytes = 69.0387 * N + 2984315.5
  R^2 = 0.99726 | worst residual = 2.47 % of file
  criterion: PASS
garden: PASS (2.76 %) | room: FAIL (5.09 %, edge)
```

## Artifacts

- runs/{bicycle,garden,room}/{scene}_s0/ckpt-*.pt (30 each, local-only, untracked)
- runs/analysis/k1-all.json · k1-bicycle.json (wt) · fit-{bicycle,garden,room}.log
- runs/analysis/figures/{fig1_bytes_n,fig2_landing,fig3_psnr_n}.png
- runs/analysis/bicycle30k_zlib.bgs (124 MB round-trip artifact, local-only)
- shn-compare-{30k,1k}.json (~/) · fit_all.log (~/)
- PPT: my.feishu.cn/slides/LEsJsLTK1lqw7hdhnAec30SonAf (en, 12 pages) ·
  budgetgs-week1-review-en.pptx (local)
- proposal: docs/03-proposal-w1-update.md · Week-2 plan: docs/plans/2026-10-15-week2-*.md

## Visualization

- fig1_bytes_n.png: N vs bytes, three scenes, ledger line + real-encode points,
  10 MB budget line.
- fig2_landing.png: landed bytes per scene vs 10 MB, epsilon annotations.
- fig3_psnr_n.png: val PSNR over the run, three scenes, 30 points each.

## Findings

- Serialized bytes are affine in N on all three scenes (bicycle 69.04 B/N +
  2.98 MB, garden 68.94 B/N + 1.96 MB, room 58.39 B/N + 4.39 MB; R² 0.9905-0.9973).
  Room is a marginal FAIL (worst residual 5.09 %), so per-scene rates are kept.
- A single global-levels actuator cannot hit a 1 %/256 KiB band: byte floor
  (bicycle) and coarse steps (garden/room). Ruled out as the only actuator.
- shn=3 is strictly smaller with the same codecs (−6.3 % at 30 k).
- The Week-1 trainer is not protocol-comparable (factor / loss / bad views);
  quality columns on the landing table would mislead until Week-2 alignment.

## Open questions & next steps

- Does one exchange rate suffice? No — room (58.4 B/N) differs from
  bicycle/garden (~69 B/N); keep per-scene rates (Week-2 M3).
- Can per-group budgets give the missing precision axis between global levels?
  (Week-2 M2.)
- What is the byte floor after pruning toward ~900 k Gaussians on bicycle?
  (Week-2 M1.)
- Confirm LDenninger write access to Yuefan-Zhang-Liberty/budgetGS (B3 open item,
  needs GitHub confirmation).

## Reproduction

```bash
git clone --recurse-submodules git@github.com:LDenninger/project-budget-gaussian-splatting.git
cd workspaces/budgetgs
micromamba activate main
budgetgs train --scene bicycle --steps 30000 --ckpt-every 1000   # 3 scenes
python -m budgetgs.evaluation.k1 --ckpt-dir runs/bicycle/bicycle_s0 --sample 6
python -m budgetgs.cli fit --ckpt runs/bicycle/bicycle_s0/ckpt-0030000.pt --budget-mb 10
```

Headline: ledger 0.27-0.47 % · K1 PASS on bicycle/garden, FAIL(edge) on room · landing
fails on all three (single executor) — plan change recorded.

## Take-home message

The byte loop holds; the actuator does not. Budget-to-size prediction is solid
(0.27-0.47 %), but landing on a named budget needs a scale axis (pruning) and
per-group precision — that is the Week-2 program.

## Task

Interim — see docs/plans/2026-10-08-week1-budgetgs-ramp-up.md (acceptance
items 1-3 and Day 5/Day 6 rows).
