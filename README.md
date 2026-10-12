# BudgetGS

A training-time controller that drives a 3D Gaussian Splatting scene onto a stated resource budget.

The user names a budget once, before training starts, and a single training run produces a scene that
fits it: bytes on disk first, then bytes of device memory at render time, then milliseconds per frame
on an edge device. The controller measures the budget with a real encoder during training and decides
how much of it goes into more primitives, how much into attribute precision, and where in the scene
either is spent. This repository holds the research behind that controller: the project proposal,
the research program, the literature library and the reference implementations.

## 📖 Background

The representation is [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) trained with
[gsplat](https://github.com/nerfstudio-project/gsplat). The entropy-coded rate term comes from the
3DGS compression line around [HAC++](https://arxiv.org/abs/2501.12255), and the budget-as-input
baselines are [SizeGS](https://arxiv.org/abs/2412.05808) and
[MesonGS++](https://arxiv.org/abs/2604.26799), which reach a size after training rather than during it.

## 📦 Install

```bash
git clone --recurse-submodules git@github.com:LDenninger/project-budget-gaussian-splatting.git budgetgs
cd budgetgs
```

Download the 67 reference PDFs, which are not tracked:

```bash
while IFS='|' read -r dir slug id; do
  curl -sL -o "references/$dir/${slug}_arxiv$id.pdf" "https://arxiv.org/pdf/$id"
done < references/manifest.txt
```

## 🚀 Usage

Read the documents in this order:

1. `docs/01-proposal.md`, the problem, the goals and the starting reading list.
2. `docs/02-research-program.md`, the claim, the gaps in the literature, the research directions
   D1 to D6, the kill experiments K1 to K5, the roadmap and the decisions taken so far.
3. `references/README.md`, the 67 papers grouped by role, why each one is here, and the evaluation
   protocol the field reports under.
4. `scratchpad/08_09_26_14_35_literature_notes/`, full-text notes per paper with page-cited quotes,
   and the 3DGS.zip leaderboard CSVs under `leaderboard/`.

The project is set up for Claude Code. `.claude/CLAUDE.md` lists the skills and the rules, and the
rules in `.claude/rules/` bind human contributors too.

## 📊 Reproduce the Week-1 numbers

The Week-1 acceptance loop is: train → byte ledger → real serialize → fit onto a budget.
From a clean checkout with `workspaces/budgetgs` installed and the `main` env active:

```bash
# 1. Train a scene (30k steps, one checkpoint every 1k -> runs/<scene>/<scene>_s0/)
cd workspaces/budgetgs
budgetgs train --scene bicycle --steps 30000 --ckpt-every 1000

# 2. K1: bytes-on-N regression + real-encode verification, one command
python -m budgetgs.evaluation.k1 --ckpt-dir runs/bicycle/bicycle_s0 --sample 6 --out runs/analysis/k1-bicycle.json

# 3. Landing: land a checkpoint on a 10 MB byte budget (real write + read-back)
budgetgs fit --ckpt runs/bicycle/bicycle_s0/ckpt-0030000.pt --budget-mb 10

# 4. Landing quality (PSNR / SSIM / LPIPS of the landed container; needs a GPU)
python -m budgetgs.evaluation.landing --ckpt runs/bicycle/bicycle_s0/ckpt-0030000.pt \
  --scene <mipnerf-360 scene dir> --budget-mb 10 --out runs/analysis/landing-bicycle.json
```

Recorded Week-1 results (bicycle / garden / room, 30k steps, ckpt-0030000.pt):

| Check | Result |
|---|---|
| ledger vs real serialize | 0.27–0.47 % gap (range round-trip, 3 scenes) |
| K1 bytes-on-N | bicycle 69.04 B/N + 2.98 MB, R² 0.9973, 2.5 % PASS · garden 68.94 B/N + 1.96 MB, R² 0.9930, 2.8 % PASS · room 58.39 B/N + 4.39 MB, R² 0.9905, 5.09 % FAIL (edge) |
| landing @ 10 MB | bicycle 12.85 MB (floor, +28.5 %), garden 9.73 MB (−2.67 %), room 9.43 MB (−5.68 %) — all outside the 1 %/256 KiB band |

The landing miss is intentional evidence: a single global block-precision actuator
has a byte floor and only ~10 effective steps; pruning + per-group executors are
the Week-2 program (see `docs/03-proposal-w1-update.md`). Figures:
`workspaces/budgetgs/runs/analysis/figures/` (fig1_bytes_n, fig2_landing, fig3_psnr_n).

## 📁 Repository layout

```
.
├── docs
│   ├── 01-proposal.md                    # Problem, goals, reading list
│   └── 02-research-program.md            # Claim, gaps, directions, kill experiments, roadmap
├── references
│   ├── 00_foundations … 07_runtime_cost  # PDFs by role, ignored by git
│   ├── results                           # Result tables transcribed from the papers, CSV
│   ├── manifest.txt                      # dir|slug|arxiv-id for every PDF
│   └── README.md                         # Annotated bibliography, gap table, evaluation protocol
├── scratchpad
│   └── 08_09_26_14_35_literature_notes   # Per-paper reading notes and the leaderboard
├── submodules                            # Reference implementations, read-only
├── workspaces
│   ├── ldtrain                           # Run logger and web viewer, submodule
│   └── open-splatting-viewer             # Browser viewer for splatting representations, submodule
├── .claude                               # Claude Code rules, skills and hooks
└── README.md
```

`submodules/` holds gsplat, HAC++, SizeGS, MesonGS++, ffsplat, Taming 3DGS, GaussianSpa, PUP 3D-GS,
SALVQ and the G-PCC reference encoder `tmc13`. They are called from new code and never edited. Code
goes into its own repository under `workspaces/`, and datasets go under `datasets/`, which git ignores.

## 🤝 Contributing

Commit and push directly to `master` of
[LDenninger/project-budget-gaussian-splatting](https://github.com/LDenninger/project-budget-gaussian-splatting),
with no pull request or review. Commit messages follow `.github/commit_convention.md`, and documents
follow `.claude/rules/docs-rules.md`.

## 📄 License

No license file is present. The PDFs under `references/` belong to their authors and are not
redistributed.
