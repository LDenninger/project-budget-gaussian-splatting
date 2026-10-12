# Week 2 — BudgetGS ramp-up (10-15 → 10-21)

- **Date:** 2026-10-12 (drafted; numbers filled as they land)
- **Branch:** master (workspaces/budgetgs)
- **Commit:** budgetgs d0077b4 · outer 00106bd
- **Kind:** plan
- **Status:** draft — to be finalized 10-15 with the advisor

## Context

Week-1 closed the byte loop (ledger-vs-real < 0.3 %, K1 PASS on bicycle:
74.28 B/N + 1.85 MB, R² 0.9994) and **falsified the single-executor landing
assumption** on all three scenes (bicycle byte floor 12.85 MB; garden/room steps
jump 2.7–5.7 % outside the 1 %/256 KiB band). The Week-2 program replaces the
single actuator with two axes: **scale (pruning)** and **per-group precision
(multi-executors)**, and aligns the training protocol before any quality column
is trusted.

## Goals (M1–M5)

### M1 · Pruning executor — closes the 10 MB reachability loop

- Goal: a pruning executor that removes Gaussians by a budget-derived target N
  (from the K1 rate), so bicycle lands at or under 10 MB with tolerance.
- Bar: `budgetgs fit` on pruned bicycle lands within 1 %/256 KiB of 10 MB, and
  val PSNR is reported (expected some loss vs the 30k baseline).
- Steps:
  1. N_target = (10 MB − intercept)/slope ≈ (10,000,000 − 1,849,626)/74.2843 ≈ 109.7 k
     — but bicycle is already 1.1 M; check whether intercept re-fits at small N.
  2. Implement pruning (importance by opacity × scale, or spatial thinning),
     re-fit K1 on pruned runs to see whether the exchange rate holds.
  3. Wire pruning into the budgetgs CLI as an executor alongside precision.
- Risk: pruning changes the byte-N curve (intercept shrinks); re-measure, do not
  reuse the Week-1 rate blindly.

### M2 · Per-group multi-executors — the precision axis between global levels

- Goal: independent budget per attribute group (means/scales/quats/opacity/sh0/shn)
  plus block precision, giving fine-grained byte steps that a bisection can land on.
- Bar: garden/room land within the 1 %/256 KiB band with per-group levels.
- Steps:
  1. Extend the ledger config to per-group `levels` (LedgerConfig already
     supports levels; add a dict form).
  2. Extend `fit` to search per-group levels (coordinate descent or
     sequential bisection) and keep real write+read-back as the feedback signal.
  3. Report bytes + PSNR/SSIM/LPIPS on all three scenes.
- Note: Week-1 showed only ~10 effective global levels; per-group multiplies
  the grid (6 groups × levels) — this is the missing precision axis.

### M3 · K1 completion — does the rate transfer across scenes?

- Goal: K1 on garden and room, 30 points each, real-encode sample.
- Bar: R² > 0.98 and worst residual < 5 % per scene (same criterion).
- Status: DONE 2026-10-12. bicycle 69.04 B/N (R² 0.9973, 2.5 %, PASS);
  garden 68.94 B/N (R² 0.9930, 2.8 %, PASS); room 58.39 B/N (R² 0.9905,
  5.09 %, FAIL at the edge). Rates differ per scene (room vs bicycle/garden),
  so the controller keeps a per-scene exchange rate (no single global rate).

### M4 · Protocol alignment — make the quality columns trustworthy

- Goal: retrain (or re-evaluate) with the official protocol: factor=4,
  L1 + 0.2·SSIM, bad-view triage, so PSNR/SSIM/LPIPS on the landing table are
  comparable to the field (3DGS.zip leaderboard).
- Bar: PSNR within ~1 dB of the official per-scene numbers on un-landed ckpts.
- Steps:
  1. Confirm the trainer flags (factor, loss weights) and rerun one scene;
     compare against official.
  2. Triage the 6 bad views (views 0–4 + 9) seen in Week-1 bicycle.
  3. Fill PSNR/SSIM/LPIPS columns on the landing table for all three scenes.
- Note: do not mix pre/post-alignment numbers in one table; mark the protocol.

### M5 · Reproducible materials — devlog, figures, viewer, README

- Goal: a clean checkout reproduces the Week-1 numbers in ~4 commands.
- Bar: README section "Reproduce the Week-1 numbers" runs end-to-end.
- Steps:
  1. devlog entry (docs/devlog/, kind interim) — this week's record.
  2. Three figures (fig1_bytes_n, fig2_landing, fig3_psnr_n) in
     workspaces/budgetgs/runs/analysis/figures/.
  3. Viewer checkpoint: open-splatting-viewer pointing at one ckpt, screenshot.
  4. README: install → train → K1 → fit → landing quality.

## Deliverables / milestone

- 10-15 advisor review: deck (Week-1 review) + proposal update + submodule code
  (all three shipped 10-12). This plan is the Week-2 commitment. Reproducible
  materials (figures, devlog, README one-command reproduction) shipped 10-12 too.
- 10-21 (end of Week-2): M1–M5 with measured numbers; updated proposal;
  decision on the controller architecture (two-axis executor).

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pruning changes the K1 rate | medium | re-fit K1 after each pruning level; do not reuse Week-1 rate |
| Per-group search is slow (6 dims × real serialize) | medium | coordinate descent, 2–3 passes; cap evals |
| Protocol alignment reveals trainer bugs | low | isolate loss/factor first, then views |
| Storage/network flakiness (seen Week-1) | high | sbatch with restart count; logs to /storage/slurm/zyuef |
| Advisor asks for a 5 MB target | low | N_target formula is linear in budget; re-derive, don't guess |
