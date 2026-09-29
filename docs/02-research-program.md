# BudgetGS: literature gaps, research directions and the road to a paper

- **Date:** 2026-09-08
- **Status:** research program, written before any code exists in this repository
- **Inputs:** 67 references under `references/` (36 from a first survey plus 31 added later), the
  3DGS.zip leaderboard CSVs pulled from `w-m/3dgs-compression-survey` (updated 2026-07-14), and
  full-text reading notes for every reference under
  `scratchpad/08_09_26_14_35_literature_notes/`

Every number in this document is **reported** by the cited paper or leaderboard unless it is
marked **derived** (follows from an argument given beside it) or **assumed**. Nothing was measured
in this repository yet, and no GPU run has happened.

## 0. The answer

**The gap claimed by the first survey survives, but only in a narrower form.** The first survey wrote "no
published method binds a byte target during training". Two papers touch that cell:

- GETA-3DGS (arXiv 2605.02086) accepts a storage budget B in MB as a training input, and reports
  that B is "a sparsity penalty rather than a hard byte cap": every requested budget from 3 MB to
  120 MB lands at 4.0 to 4.2 MB on Mip-NeRF 360 (its Fig. 2 and Fig. 6, reported). It names "a
  standard sub-gradient ascent on the Lagrangian dual of the byte-counter constraint" as its own
  future work (p. 11).
- HybridGS (ICML 2025, arXiv 2505.01938) accepts a target size B in MB inside its training run,
  but only after the top-quality point at iteration 36 000 of 70 000, and 21 000 iterations after
  densification has ended. Its two rate-control methods are mutually exclusive, prune primitives
  or lower bit depths, never both and never grow. Its size model is n · P_bit / L with the G-PCC
  ratio L = 1.3 hand-set, and across the 15 usable pairs of its Table 4 the achieved size misses
  the target by −45 % to +8 % (derived from the reported pairs), overshooting on "dance" at all
  four rate points and undershooting by 14 % on "train" with pruning alone.
- MesonGS++ (arXiv 2604.26799), SizeGS's successor, lands within 0.3 to 0.9 % of the budget
  (reported) by measuring the real compressed size inside its search loop, on a frozen
  30 000-iteration checkpoint, coupling the reserve ratio and the per-group bit widths under one
  byte budget. Nothing in it touches a gradient step.

The defensible claim is therefore:

> No published method makes an entropy-coded byte budget bind on both primitive growth and
> attribute precision jointly, from the first iteration of training, and demonstrates that it
> hits the budget. The closest work either does not bind (GETA-3DGS), binds late and only by
> shrinking, one lever at a time (HybridGS), or searches a frozen model after training (SizeGS,
> MesonGS++, FlexGaussian).

Each clause survives a named counter. "From the first iteration" and "jointly" exclude HybridGS.
"During training" excludes MesonGS++, which does couple count and precision and does hit its
budget. The asymmetry worth naming is the actual technical content: a post-hoc method can run
the real encoder inside its search loop on a model that does not move, whereas a training-time
controller has to hold the budget on a model that changes under it, so it must predict between
encodes and correct on each real encode. That is rate control in the video-coding sense, and it
does not exist for 3DGS.

Three further cells are empty and belong to phases 3 and 4 of this project (Section 9):

- no method takes a **runtime memory target in bytes** (VRAM of the decoded model plus render
  buffers). MEGS² (ICLR 2026) is the only training method that targets rendering VRAM, and it does
  so through sparsity targets, not a byte number
- no method takes a **frame-time target in milliseconds on a named device**. MetaSapiens (ASPLOS
  2025) has the only per-primitive cost proxy validated against measured latency, used post hoc
- no 3DGS paper builds or cites a **device latency predictor** in the sense of hardware-aware
  neural architecture search

**What the literature does establish, and the program has to build on rather than repeat:**

1. λ is not a size handle. At one fixed λ the encoded size spreads 3.2× to 5.5× across the nine
   Mip-NeRF 360 scenes (HAC λ_e = 0.004: bicycle 27.54 MB, room 5.53 MB. ContextGS: 4.50 to
   21.82 MB. CAT-3DGS: 4.11 to 21.42 MB. All reported). SizeGS says it in one sentence: "it is
   difficult to predict the final size given a certain λ" (p. 2).
2. Hitting a size post hoc is cheap. SizeGS reaches a target within 1 % in about a minute of
   search, FlexGaussian adjusts in 1 to 2 s. A training-time method cannot win on hit accuracy or
   on time. It has to win on quality at the target, or on a target that post-hoc methods cannot
   reach at all (a runtime-memory or frame-time bound).
3. Knowing the budget from iteration 0 does not by itself win. At matched primitive count, Taming
   3DGS (budget from the start) is below GaussianSpa and Mini-Splatting (over-densify, then
   sparsify gradually): 27.31 dB at 0.630 M against 27.85 dB at 0.547 M on Mip-NeRF 360
   (GaussianSpa Table 1, reported). What does win is smooth tracking over an abrupt cap (Smart
   Target Point Control, +0.12 dB on Mip-NeRF 360 and +2.18 dB on NeRF-Synthetic, reported) and
   gradual over one-shot sparsification (GaussianSpa). The untested design is a byte trajectory
   with a bounded, scheduled overshoot.
4. The PSNR frontier is flat above about 6 MB on Mip-NeRF 360: 27.76 dB at 5.9 MB (Smol-GS) to
   27.93 dB at 21 MB (HEMGS), 0.17 dB over a 3.6× size range (leaderboard, reported). Byte
   targets are interesting below 6 MB, on Tanks and Temples and Deep Blending below 5 MB, and in
   LPIPS rather than PSNR.

**The recommended program, in one paragraph.** Build a closed-loop rate controller for per-scene
3DGS training whose state is the measured encoded size, whose actuators are the rate multiplier λ,
the per-attribute quantization steps and the densification allowance, and whose set point is a
byte trajectory ending at S★. Validate it first on the storage budget, where the baselines
(SizeGS, MesonGS++, HybridGS, HAC++ with an oracle λ sweep, Smol-GS) and the protocol (3DGS.zip)
exist, in the low-rate regime where the frontier is steep. Then swap the byte counter for a
device cost model (VRAM bytes, then ms per frame from a fitted per-device predictor) and show that
the same controller lands a model on a Jetson-class or iGPU-class bound. Section 8 gives the kill
experiments that decide within two weeks whether the first half is worth a paper.

**Decisions taken by the user on 2026-09-08 (Section 12):** storage size is the first bound, the
substrate is gsplat (option A), a Jetson-class board is available for the device phases, no venue
is targeted and the work is pushed as far and as fast as possible, and the sibling project's
controller is a design reference only.

## 1. The problem and its three budgets

A deployment constraint on an edge device is one of three numbers, and the literature measures
only the first:

| Budget | Symbol | What produces it | Who measures it today |
|---|---|---|---|
| storage or transmission size | S★, target for R(θ) | quantizer plus entropy coder, or an image or point-cloud codec | every paper in `01`, `02`, `04` |
| memory at render time | M★, target for M(θ) | decoded attribute layout × N, plus the rasterizer's per-tile buffers, plus any MLP or hash grid | MEGS², FLoD, GETA (peak GPU column), Gaussians on a Diet (training peak) |
| frame time on the device | T★, target for T(θ) | preprocess, Gaussian-tile pair generation, sort, blend, dominated by pairs and per-pixel list length | MetaSapiens, Splats under Pressure, HiGS, RoofGS, the accelerator papers |

Symbol bindings. No code exists in this repository, so each symbol is bound to the place it lives
in the reference implementations or in the sibling project, and marked when it does not exist.
The sibling project is a design reference only, none of its code is reused (Section 12):

    N     | primitive count                       | `num_triangles`   | sibling `textured_triangle_splatting/model/budget.py:59`
    P     | learnable parameter count             | `params_now`      | sibling `budget.py:39`
    P★    | parameter target of the sibling       | `target_params`   | sibling `configs/experiment/lego.yaml:8`
    θ     | full model state                      | -                 | does not exist yet
    S★    | byte target on disk                   | -                 | does not exist yet
    R(θ)  | encoded bytes on disk                 | -                 | does not exist yet
    R̂(θ)  | differentiable rate estimate          | HAC `entropy_loss` | reference repo, not yet in `submodules/`
    λ     | rate multiplier in D + λ·R̂             | HAC `lmbda`       | reference repo, not yet in `submodules/`
    M(θ)  | bytes in device memory at render time | -                 | does not exist yet
    T(θ)  | ms per frame on the target device     | -                 | does not exist yet

**Why the three differ (all reported).** 3DGS.zip observes on p. 8 that for structured methods
"their memory usage is constant, meaning that reducing the number of primitives does not further
decrease VRAM consumption". The original 3DGS paper puts the rasterizer's own working memory at
"30–500 MB" beside a model of "several hundred megabytes" (p. 11). GETA-3DGS reports a 4.86 MB
file that needs 261 MB of peak GPU memory to render on an A100 (its Table III), a 54× ratio
(derived from those two numbers). The HAC family decodes in 1 to 31 s of CPU arithmetic coding and
then runs Scaffold-GS MLPs per anchor per view. Ali et al. state the consequence on p. 12:
"structured methods do not inherently translate compression efficiency into rendering
efficiency." A byte target on disk therefore says little about M(θ) or T(θ), and a project whose
end goal is an edge device has to model all three, in that order.

**Why bytes are not a rescaled count (derived).** The sibling controller's currency is P, which is
affine in N under a fixed per-primitive payload, so its ramp closes exactly against the model's
own parameter total. R(θ) is the output of a quantizer and an entropy coder, so two models with
the same N differ in bytes whenever their attributes differ in entropy. The literature gives the
magnitude: an uncompressed float32 ply costs a constant 242 to 245 B per Gaussian (derived from
Taming 3DGS's Fig. 1 labels), GETA-3DGS reports 63.7 B per Gaussian after quantization, KISS-GS
reports 129 bits per Gaussian at 256 k primitives, and Smol-GS reaches 5.86 MB with 514 835
primitives on Mip-NeRF 360 (leaderboard), which is 91 bits per Gaussian (derived). A count target
is a byte target only for the uncompressed format, which is the one nobody ships to an edge
device.

## 2. The scoreboard

The field compares on four datasets with the 3DGS protocol (9 Mip-NeRF 360 scenes at up to 1600 px
on the long side, `train` and `truck`, `drjohnson` and `playroom`, every 8th image held out, and
the Synthetic NeRF split), and reports PSNR, SSIM, LPIPS and size in MB with 1 MB = 10⁶ bytes.
The maintained leaderboard is the 3DGS.zip survey's CSVs. Neither survey tabulates training time,
decode time, peak VRAM or FPS, and neither states whether MLP, hash-grid or codebook bytes are
inside "size", which is the first evaluation hole this project has to close for its own tables.

**Pareto frontier of PSNR against size, from the leaderboard CSVs (reported), methods under
200 MB.** The F-3DGS Tanks and Temples row is excluded because the CSV flags it as off-protocol.

Mip-NeRF 360:

| Method | Size MB | PSNR | SSIM | LPIPS | N |
|---|---|---|---|---|---|
| Smol-GS base | 5.86 | 27.76 | 0.807 | 0.248 | 514 835 |
| Smol-GS large | 10.40 | 27.86 | 0.812 | 0.230 | 934 185 |
| HEMGS high rate | 21.00 | 27.93 | 0.813 | 0.230 | not listed |

Below 5.86 MB no method is listed on Mip-NeRF 360. The nearest smaller points are RDO-Gaussian at
6.16 MB and 26.03 dB and gsplat-compression at 6.92 MB and 26.64 dB, which is a 1.7 dB drop from
the frontier over a 1 MB step and the regime where a byte controller has something to allocate.

Tanks and Temples:

| Method | Size MB | PSNR | SSIM | LPIPS | N |
|---|---|---|---|---|---|
| RDO-Gaussian | 3.74 | 22.98 | 0.812 | 0.234 | 263 067 |
| Smol-GS base | 4.78 | 24.25 | 0.843 | 0.199 | 426 744 |
| HEMGS low rate | 6.03 | 24.42 | 0.848 | 0.192 | not listed |
| HEMGS high rate | 10.14 | 24.58 | 0.856 | 0.176 | not listed |

Deep Blending:

| Method | Size MB | PSNR | SSIM | LPIPS | N |
|---|---|---|---|---|---|
| Smol-GS base | 2.86 | 30.12 | 0.905 | 0.262 | 254 500 |
| HEMGS low rate | 2.99 | 30.24 | 0.908 | 0.266 | not listed |
| HAC++ high rate | 5.54 | 30.34 | 0.911 | 0.254 | 642 515 |
| HEMGS high rate | 6.70 | 30.37 | 0.911 | 0.253 | not listed |
| ContextGS high rate | 6.86 | 30.41 | 0.909 | 0.259 | not listed |

Two consequences. First, the size-targeted ancestor sits below the frontier at its own operating
point: SizeGS reports 27.48 dB at 18.17 MB on Mip-NeRF 360 (its Table 1), where the frontier is
27.86 dB at 10.40 MB, and its HAC baseline in that table is the 2024 HAC, not HAC++ or HEMGS.
Second, every frontier method above is anchor based (Scaffold-GS lineage) or feature based
(Smol-GS decodes splat-wise features), and every one needs a network at render time. The raw
3DGS methods that render with the vanilla rasterizer (RDO-Gaussian, KISS-GS, SOG, Compressed 3DGS,
gsplat compression) sit 0.5 to 1.5 dB lower at equal size. That is the representation fork of
Section 6.

## 3. The families and the corrected gap table

The first survey split the literature into three families by what the user supplies. With 67 papers
read, the split has eight rows, and two of its cells were wrong.

| Family | Representatives | User supplies | Binds | Size model | Hits an exact byte target |
|---|---|---|---|---|---|
| post-hoc size search | SizeGS, MesonGS++, FlexGaussian (ratio or ΔPSNR), KISS-GS (count) | MB, ratio, or count | after training, frozen model | real encode plus linear estimator S(Q) = Σ P_ij Q_ij + C + S_Δ, calibrated by re-encoding, 0-1 ILP over bit widths | yes, within 1 % (SizeGS), in about 1 min |
| byte budget as training input | GETA-3DGS | MB, count or PSNR floor | during training, 25 k warm-up plus 10 k | raw quantized bytes b̄ × 59 / 8, no entropy coder | no, non-binding, 3 to 120 MB all land at 4.0 to 4.2 MB |
| byte target in a continued-training shrink stage | HybridGS | MB, given at iteration 36 000 of 70 000 | second stage, prune or lower bit depths, one lever at a time, count cannot grow | R = n · P_bit / L with L = 1.3 hand-set | no, −45 % to +8 % across its Table 4 (derived) |
| rate-distortion training with λ | HAC, HAC++, ContextGS, CAT-3DGS, CompGS, RDO-Gaussian, CodecGS, Voxel-GS, SpeedyGS | λ | during training, entropy term from 3 k to 15 k iterations | entropy model, per-parameter bit average | no, per-scene spread 3.2× to 5.5× at fixed λ |
| variable rate, one model | HEMGS, SALVQ, PCGS, RAVE, ProGS | λ index, gain index, level count, interpolation weight | during training, rate chosen at encode time | entropy model | no, discrete points, no map from bytes to λ or gain, up to +18.89 % BD-rate versus dedicated models (SALVQ) |
| count budget during training | Taming 3DGS, 3DGS-MCMC, GaussianSpa, Smart TPC, Constrained Dynamic GS, YOGO, ConeGS, ControlGS, Gaussians on a Diet | count, trajectory, or one scene-agnostic λ | during training | none, count only | no, count is not bytes under compression |
| anytime and elastic | Matryoshka GS, FlexGS, CLoD-GS, FLoD, GoDe, PCGS | count fraction or level at inference | during training, all budgets | none | no, and no byte or ms input |
| runtime resources | MEGS² (VRAM), MetaSapiens and Speedy-Splat (render time), Splats under Pressure and HiGS (measurement) | sparsity targets or pruning fraction | training (MEGS², Speedy-Splat) or post hoc | tile intersections (MetaSapiens), count (HiGS, Splats under Pressure) | no, none takes VRAM bytes or ms as input |

The three empty cells: **(a)** an entropy-coded byte target bound from the start of training,
with growth, **(b)** a runtime-memory target in bytes, **(c)** a frame-time target on a named
device. The order of the program follows the order of the cells, because (a) has baselines and a
protocol, (b) needs a memory model that (a) produces as a by-product, and (c) needs a device.

**Corrections to the first survey's table.** "Every method accepting a size in MB is post-hoc" was
wrong for GETA-3DGS and HybridGS. "Every method binding a budget during training denominates it
in primitive count" was wrong for HybridGS's shrink stage. Its claim that GETA-3DGS's
budget "is currently non-binding" and saturates within 4.0 to 4.9 MB, which it could not attribute
to a source, is GETA-3DGS's own p. 8 sentence.

## 4. Holes and shortcomings in the existing literature

Each hole names the evidence, the paper and page it comes from, and what the program does about
it. Numbers are reported unless marked.

**H1. λ is a knob, not a size handle, and no paper maps a requested size to λ.** Per-scene sizes at
one fixed λ on Mip-NeRF 360: HAC λ_e = 0.004 gives bicycle 27.54 MB and room 5.53 MB (5.0×), HAC++
λ = 4×10⁻³ gives 12.84 and 3.41 MB (3.8×), ContextGS 21.82 and 4.50 MB (4.8×), CAT-3DGS 21.42 and
4.11 MB (5.2×), CompGS 28.66 (garden) and 8.85 MB (room), RDO-Gaussian 43.14 (bicycle) and
7.91 MB (counter), Compressed 3DGS 47.15 (bicycle) and 12.79 MB (bonsai). HAC and HAC++ pick
different λ per dataset for their "low rate" and "high rate" rows, and PCGS uses different λ sets
per dataset. SizeGS's HAC baseline needed a λ binary search with 10 to 20 min of retraining per
step, 16 627 s in total on Mip-NeRF 360, stopping inside a 5 % band. Program: the controller makes
λ a state driven by the measured gap R(θ) − S★, so a scene never needs a sweep.

**H2. The rate loss never sees the total.** HAC, HAC++ and PCGS divide the entropy loss by
N(D_a + 6 + 3K), a per-parameter bit average, so the objective is indifferent to N and to the
fixed costs. RDO-Gaussian's rate covers six VQ index streams, while positions (43.9 % of its
high-rate file) and opacity sit outside the RD objective, and its count is driven by a mask-mean
proxy. Every anchor-based method carries MLP weights (HAC++ 0.33 MB), a hash grid or triplane, and
a header that do not shrink with the budget, so at 1 to 3 MB they are a large fraction of the
file. Program: R̂ is the total estimated code length of the shipped file, fixed costs included,
and N enters R̂ explicitly through the densification allowance.

**H3. The estimate-to-encoded gap is unreported.** None of HAC, HAC++, HEMGS, SALVQ, PCGS,
ContextGS, CAT-3DGS, CompGS or RDO-Gaussian states the difference between the noise-relaxed
entropy estimate used in training and the arithmetic-coded bytes. SizeGS avoids the question by
re-encoding inside its search loop and calibrating an additive offset S_Δ until the encoded size
is within 5 % of the target (its Algorithm 1), and MesonGS++ tightens that to 0.3 to 0.9 %.
HybridGS uses a linear model whose G-PCC ratio L = 1.3 is a hand-set constant, and its own
explanation for a 14 % miss is that "different primitive densities will influence the lossless
compression ratio" (p. 8). Program: measure the gap along training (kill experiment K4) and close
the loop on real encodes at a fixed cadence, as video rate control does.

**H4. Variable rate is not target rate.** HEMGS trains four λ values and shows a 16-point sweep as
a plot, SALVQ trains four gains and states that a single variable-rate model "is not yet
sufficient to cover a broad operating range" (p. 12) with up to +18.89 % BD-rate against
single-rate models, PCGS decodes 3 to 4 prefix levels, RAVE interpolates between two bounds. None
maps a requested byte count to a λ, a gain or a level. Choi's image-codec result that a continuous
λ input "did not produce good results" (p. 4) and Cui's gain interpolation are the ancestors. The
cheapest exact knob the literature names is the per-attribute quantization step (Cui's gains,
SALVQ's lattice scale, Choi's bin size Δ), and bisection on it against the measured size is an
exact fit after training that no 3DGS paper performs. Program: use the step vector as the fast
actuator inside the loop, and λ and N as the slow ones.

**H5. Count is not bytes, and the non-affinity is unmeasured.** Bytes per Gaussian: 242 to 245
uncompressed (derived from Taming 3DGS), 63.7 quantized without entropy coding (GETA), 129 bits at
256 k (KISS-GS), 91 bits (Smol-GS base, derived from the leaderboard), and inside one method the
valid-anchor ratio moves from 0.865 to 0.596 as λ rises (HAC++). No paper plots encoded bytes
against N under a fixed codec across a training run or a pruning sweep. Program: kill experiment
K1 measures it first, because a near-affine result would let the sibling controller port after
all.

**H6. "Budget known from the start" is not established to beat "explore then sparsify".** No paper
runs the clean test, train toward N against prune a full model to N with both fine-tuned. The
indirect evidence cuts both ways: SPARE-GS-trained models take post-hoc pruning 1.7 dB better than
vanilla-trained ones without fine-tuning, MaskGaussian's learned mask beats one-shot LightGaussian
pruning by 0.34 dB at about 10 % count difference, but with 5 k fine-tuning iterations the two tie
(27.49 against 27.50 dB), PUP 3D-GS loses only 0.50 dB at 80 % post-hoc pruning after fine-tuning,
and Taming 3DGS sits below GaussianSpa and Mini-Splatting at matched count (Section 0). Program:
kill experiment K3 runs the clean test on counts, then on bytes, before any controller is built.
The design consequence is a trajectory with a scheduled overshoot rather than a monotone ramp.

**H7. Post-hoc methods can only remove.** SizeGS holds the size during its fine-tuning only by
freezing the reserve ratio, the bit widths and the coordinates, which rules out densification.
HybridGS's rate-control stage can prune or lower bit depths, one lever at a time, never grow, and
it pays 0.13 to 0.26 dB against the same model compressed without a target (derived from its
Table 4 pairs). Compressed 3DGS reports
that it "embedded these positional constraints into the Gaussian splatting training process" and
"were not able to further compress the positions without introducing a significant error" (p. 8),
and SOG says "it would be interesting to perform the quantization during training" (p. 14). Placing
capacity under a known budget is the hypothesis every post-hoc paper leaves open. Program: this
is the paper's central claim, and K3 and K5 test it before it is claimed.

**H8. Storage size is not runtime memory.** The frontier methods decode in 1 to 31 s of CPU
arithmetic coding, run Scaffold-GS MLPs per anchor per view, and a 4.86 MB GETA file needs 261 MB
of GPU memory to render (Section 1). MEGS² is the only training method that targets rendering VRAM,
through spherical-Gaussian lobe pruning and count pruning under an ADMM constraint
ρ_o‖o‖₀ + ρ_s‖s‖₀ ≤ κ with ρ_o = 11 and ρ_s = 7 counted in parameters, not bytes, and it gives no
map from κ to megabytes and no requested-against-achieved VRAM table. Its device numbers show
what the bound buys: 91.0 FPS on a Dimensity 9400+ phone against 6.6 for 3DGS, and 60.1 FPS on a
Snapdragon 888 where the baselines fail (reported). FLoD builds levels for "a broad range of GPU
settings" but the user picks a level, not a byte number, and its own Table 9 shows an MX250
laptop with 2 GB of VRAM running out of memory at the finest level on three scenes. On the web
the memory bound is a cliff: 5.8 M splats need 1.20 to 2.80 GB of VRAM and the baseline viewers
crash outright on an Oppo Find X2, an iPhone 15 Pro Max and a Redmi K70 Pro (WebSplatter). HiGS
gives the only closed-form buffer arithmetic (16 · N_R KB for the key buffers, 2 · N_B · N_M B for
the bins). Program: phase 3 replaces the byte counter by a VRAM model M(θ) = N × (decoded bytes
per primitive) + buffer bytes(tile pairs, resolution) + network bytes, and Section 7 chooses a
representation whose decoded layout the vanilla rasterizer can consume.

**H9. Render time has no predictor and no target.** The cost drivers the literature names are
Gaussian-tile pairs (AdaGScale measures pair generation, sort and rasterisation at 88.1 % of the
frame, Speedy-Splat, AdR-Gaussian, GSCore), per-pixel list length (Shorter Gaussian Lists),
sorting bandwidth (Neo, RoofGS) and tile load imbalance (LS-Gaussian). HiGS measures frame time
near-linear in count on an RTX PRO 6000 (1.25 to 9.97 ms for 5 M to 75 M at 1080p). Splats under
Pressure measures FPS against splat count per GPU tier (RTX 3050: 45.8 FPS at 0.58 M to 19.7 FPS
at 3.45 M at 1080p) and finds 60 FPS needs under about 600 k visible splats. MetaSapiens' per-point
efficiency CE_i = Val_i / Comp_i, with Comp_i the number of tiles the ellipse intersects, is the
only per-primitive proxy validated against measured latency, and it is applied post hoc. RoofGS
gives the only analytic skeleton, a roofline per stage with 236 B of preprocess traffic per
Gaussian and a knee of 81.9 FLOPs per byte on an RTX 4090, unvalidated as a predictor. No paper
regresses frame time on model features and reports an error, and no 3DGS paper cites
hardware-aware NAS. Three device facts make a single global model wrong: the bottleneck stage
moves with the device (WebSplatter: render-bound at 52 to 63 % on an RTX 3070, an M-series chip
and a Redmi K70 Pro, preprocess-bound at 40 to 47 % on an Intel iGPU and an MX350. Mobile-GS: the
sort takes 3.1 to 4.9 ms per frame on a Snapdragon 8 Gen 3. GBU: blending takes 48 to 78 % on a
Jetson Orin NX), frame time is not monotone in count (bonsai at 1.24 M renders faster than train
at 1.03 M on three devices, WebSplatter), and weak devices pay per fragment, not per Gaussian
(GBU: 541, 161 and 688 fragments per Gaussian with 7.6 to 13.7 % contributing, and the exponent
alone needs 1.1 TFLOPs for 60 FPS on the Orin NX). Steady-state matters too: Mobile-GS drops from
127 to 74 FPS at thermal equilibrium. Program: phase 4 fits T(θ) per device from (visible N, tile
pairs, list length, resolution) at steady state, in the style of the NAS latency predictors, and
constrains it through the same controller.

**H10. The evaluation protocol hides what an edge deployment needs.** The leaderboard reports MB,
PSNR, SSIM, LPIPS and count. Training time, encode time, decode time, peak VRAM and FPS appear in
individual papers with different hardware and never in one table. Sizes disagree between the two
surveys by a factor of 1.045 (MiB against MB), and Scaffold-GS's Mip-NeRF 360 PSNR differs by
1.34 dB between them. Above 6 MB the PSNR frontier moves 0.17 dB over a 3.6× size range, so PSNR
cannot discriminate methods at mid budgets. Program: the paper's tables carry size in MB
(10⁶ bytes) with the fixed costs itemised, decode time, peak render VRAM, FPS on two named
devices, training time, and LPIPS beside PSNR, at budgets on the steep part of the curve.

**H11. Licensing narrows the substrate.** 22 of the 31 confirmed reference repositories inherit the
Inria non-commercial licence through the vendored rasterizer, including SizeGS, HAC, HAC++, PCGS,
Taming 3DGS and MesonGS. gsplat (Apache-2.0) is the only base with an OSI licence, an MCMC
densifier, a compression benchmark on the leaderboard and the KISS-GS code link. GETA-3DGS, HEMGS
and Smart Target Point Control have no code. Program: build on gsplat, call the Inria-licensed
repositories only as baselines run from `submodules/`.

**H12. Nobody answers KISS-GS.** KISS-GS argues that "training-format coupling effectively locks
the codec to a particular 3DGS training pipeline" (p. 2 to 3) and shows a training-free encoder
beating HAC++ on Tanks and Temples and Mip-NeRF 360 at INRIA quality (227× and 104× against 77×
and 70×), while conceding Deep Blending to the anchor method. A training-time byte-budget method
has to say what coupling buys. The answer this program proposes is not a better codec but a
resource bound the codec cannot see: the controller places capacity where the budget allows,
and for M★ and T★ there is no post-hoc equivalent, because pruning after the fact is exactly the
"remove only" operation of H7.

## 5. Where the mechanisms come from

The idea has named ancestors in four fields, none of them 3DGS. Each entry names what the paper
supplies, and Section 6 says how it is used. All verified against arXiv or a DOI on 2026-09-08.

**(a) λ as a controlled variable driven by measured rate.** Li, Li, Li and Zhang, "λ Domain Rate
Control Algorithm for High Efficiency Video Coding", IEEE TIP 2014, DOI 10.1109/TIP.2014.2336550.
Rate control in HEVC fits a one-parameter model λ = α · bpp^β between the multiplier and the bits,
and re-fits α and β from the bits each picture actually produced (update law as implemented in
the HM reference software, the paper text is paywalled). The learned-codec descendant is Xu et al.,
"Feedback-Driven Rate Control for Learned Video Compression", arXiv 2604.20104 (2026): a log-domain
PI controller on λ,

    e_t = log(r̂_t / r_t)
    I_t = clip(I_{t−1} + e_t, −I_max, I_max)
    Δlog λ_t = −(k_p · e_t + k_i · I_t)            clipped to ±Δ_max
    λ_{t+1} = clip(λ_t · exp(Δlog λ_t), λ_min, λ_max)

with k_p = 0.9 and k_i = 0.05, a derivative term dropped because it "amplifies instability", and
an average bitrate error of 2.88 % on DCVC (reported). This is the closest published mechanism to
the project. The difference in our favour: their feedback is the entropy-model estimate, ours can
be the exact encoder byte count of the one scene being shipped.

**(b) One model, many rates.** Choi, El-Khamy and Lee, ICCV 2019, arXiv 1909.04802 (λ and bin size
as network inputs, a continuous λ input "did not produce good results"). Cui et al., CVPR 2021,
arXiv 2003.02012 (per-channel gain vectors with exponential interpolation between trained pairs,
0.04 to 0.08 % extra parameters). Takikawa et al., "Variable Bitrate Neural Fields", SIGGRAPH
2022, arXiv 2206.07707 (rate by prefix truncation of a codebook hierarchy). Cai et al.,
"Once-for-All", ICLR 2020, arXiv 1908.09791 (one supernet, many sub-networks). The 3DGS
descendants are HEMGS, SALVQ, PCGS, RAVE, Matryoshka GS and FlexGS.

**(c) Latency predictor as a differentiable resource constraint.** Cai, Zhu and Han,
"ProxylessNAS", ICLR 2019, arXiv 1812.00332: expected latency E[lat] = Σ_i Σ_j p_j^i · F(o_j^i)
from a per-operation lookup table F, entering the loss as λ₂ · E[lat]. Wu et al., "FBNet", CVPR
2019, arXiv 1812.03443: the same lookup table with loss CE · α · log(LAT)^β. Later predictors are
learned (HAT, nn-Meter, BRP-NAS). No 3DGS paper cites any of these or fits a predictor to a device.
The adaptation needed: a lookup table is additive in operations, and splatting time is not additive
in N (H9), so the 3DGS predictor has to be fitted on (N, tile pairs, list length, resolution) per
device.

**(d) Budget-aware regularisation for pruning.** Lemaire, Achkar and Jodoin, "Structured Pruning
of Neural Networks with Budget-Aware Regularization", CVPR 2019, arXiv 1811.09332: a barrier on the
pruned activation volume that is infinite above the budget and zero below a margin, with the budget
itself lowered on a schedule instead of hardening the barrier. GaussianSpa's alternating penalty
and projection (CVPR 2025) is the 3DGS form of the same idea on counts.

**(e) A memory budget in bytes inside a training loss.** Uhlich et al., "Mixed Precision DNNs: All
you need is a good parametrization", ICLR 2020, arXiv 1905.11452: an analytic bit count
S^w = Σ_l M_l (M_{l−1} + 1) b_l with penalty Σ_j λ_j · max(0, S^w − S^w₀)², and the statement
that the constraint is not guaranteed and λ_j is hand-set. Veniat and Denoyer, "Learning
Time/Memory-Efficient Deep Architectures with Budgeted Super Networks", CVPR 2018, arXiv 1706.00046:
λ · max(0, C(H) − C̄) with C measured in ms or Mb. GETA-3DGS's non-binding B is this family's known
failure mode reproduced in 3DGS, and both papers say why: a fixed penalty weight does not enforce
a level.

**(f) Dual ascent on a Lagrange multiplier during SGD.** Gallego-Posada et al., "Controlled
Sparsity via Constrained Optimization", NeurIPS 2022, arXiv 2208.04425: a density target as a
constraint, simultaneous descent on the model and ascent on the multiplier, and "dual restarts"
that reset a multiplier once its constraint is met, because accumulated violations otherwise keep
acting after satisfaction. Cotter, Jiang and Sridharan, "Two-Player Games for Efficient Non-Convex
Constrained Optimization", ALT 2019, arXiv 1804.06500: the proxy-Lagrangian, in which the primal
step uses a differentiable surrogate of the constraint and the dual step uses the true,
non-differentiable one. That split is exactly ours, entropy estimate for the primal step and
encoder bytes for the dual step. Platt and Barr, "Constrained Differential Optimization", NIPS 1987
(the origin, λ̇ = g(x)). Stooke, Achiam and Abbeel, ICML 2020, arXiv 2007.03964 (multiplier ascent
is integral control, PI and PID forms damp it). Sohrabi et al., ICML 2024, arXiv 2406.04558 (νPI
multiplier updates). Ramirez and Lacoste-Julien, ICLR 2026, arXiv 2509.22500 (dual optimistic
ascent equals the augmented Lagrangian). Implementation home: the Cooper library, arXiv 2504.01212.

**In-field ancestors.** GaussianSpa (penalty plus projection alternation on counts), Smart Target
Point Control (quota governor on the native densification and pruning thresholds, log-domain
multiplicative updates with a deadband and a prune lockout after opacity resets), Taming 3DGS
(per-event allowance spent by score-ranked sampling), SPARE-GS (marginal utility of the budget
equalised across regions, ∂U_r/∂n_r = λ(t) for every active region), Constrained Dynamic GS
(a differentiable count controller meeting its target within 2 %), HybridGS (a byte-denominated
size model inside a continued-training loop), and SizeGS (encode-in-the-loop calibration of a
linear size estimator).

**What transfers unchanged (derived from the mechanisms above):** the log-domain multiplicative
update of λ from measured bytes, the proxy split between estimate and encoder, dual restarts and
integral clipping as anti-windup, PI damping, and a final exact fit by bisection on the
quantization step (Cui's interpolation) or by prefix truncation (VBNF, PCGS).

**What needs adaptation because N changes during training:** every rate model above assumes a
fixed unit count (pixels, weights, operations), so the fit has to be on r = R/N with N tracked and
re-fitted after each densification event, as HEVC re-fits after a scene cut. Budget schedules
(Lemaire, TM5) assume the resource only falls, so they need explicit growth phases. Anti-windup is
mandatory because densification makes the violation transient and large. And latency tables are
additive where splatting time is not.

## 6. Research directions

Each direction names its ancestor, the baselines it must beat, and the measurement that separates
them, in that order. D1 to D3 are the first paper (phases 1 and 2). D4 and D5 are the device phases the user named
(phases 3 and 4), and D6 is optional.

### D1. Rate control for per-scene 3DGS training (the core)

**The quantity.** Minimise the photometric loss D(θ) subject to R(θ) ≤ S★, where R(θ) is the byte
count of the shipped file: entropy-coded attributes, coded positions, the primitive count, every
network or grid the decoder needs, and headers. A scene is one sample, so R(θ) is exactly
measurable at any iteration by running the encoder, and the problem is a per-object constrained
problem, not a dataset-level surrogate (Section 5, F2 notes).

**The mechanism (derived from the ancestors in Section 5).** A proxy-Lagrangian loop:

    primal step   θ ← θ − η ∇_θ [ D(θ) + λ_t · R̂(θ) ]                 R̂ the noise-relaxed entropy estimate
    dual step     e_t = log( R_enc(θ) / S_t )                          R_enc a real encode, every C iterations
                  I_t = clip( I_{t−1} + e_t, −I_max, I_max )
                  λ_{t+1} = clip( λ_t · exp( k_p · e_t + k_i · I_t ), λ_min, λ_max )

with S_t a set-point trajectory that ends at S★, and a dual restart (I_t ← 0) when the
constraint flips sign after a densification event. The primal step uses the estimate because it
has a gradient, the dual step uses the encoder because it is exact, which is Cotter's proxy split
and Xu et al.'s log-domain PI update with the estimate replaced by the real byte count.

**Three actuators at three speeds.**

1. λ_t (slow): moves the whole rate-distortion operating point, as above.
2. Per-attribute quantization steps Δ_k (fast, exact): Cui's gain vectors, SALVQ's lattice scale,
   Choi's bin size. Learned under the same λ during training, and used at the end for an exact
   fit by bisection on a global step multiplier against the encoded size, so the final miss is
   set by the encoder's granularity rather than by the controller.
3. The densification allowance A_t in bytes (event-rate): at each densification event the
   controller converts the byte headroom S_t − R̂_t into a number of primitives through the
   current marginal cost r̂_t = R̂_t / N_t, re-fitted after every event as HEVC re-fits after a
   scene cut, and spends it on score-ranked candidates. This has the shape of the sibling
   project's ranked-walk spend (`compute_spend`, `plan_moves`), which serves as the design
   reference only. The byte controller is written fresh in this repository on gsplat, and r̂_t is
   the exchange rate the parameter-count design never needed (Section 1).

**The trajectory S_t.** Not a monotone ramp. The literature says a from-the-start cap loses to
explore-then-sparsify at matched count (Section 0, item 3), CDGS loses 1.24 dB when its
500-iteration unconstrained warm-up is removed (reported), and ControlGS reaches its quality
through explicit overshoot rounds of up to 8× before sparsifying (reported). So S_t follows the
Taming and Smart TPC quadratic to a peak ω · S★ with ω ∈ [1, 2] scheduled, then descends to S★ at
the end of the densification window, followed by a structure-frozen fine-tune. The peak is the
one number the user may set beside S★, and it bounds training memory the way Taming's schedule
does. Whether ω > 1 buys quality is kill experiment K4.

**Ancestors.** HEVC R-λ rate control (Li et al. 2014), feedback rate control for learned video
coding (Xu et al. 2026), the proxy-Lagrangian (Cotter et al. 2019), controlled sparsity via dual
ascent (Gallego-Posada et al. 2022), GaussianSpa's penalty-projection alternation, Smart Target
Point Control's quota governor, Taming 3DGS's per-event allowance, Constrained Dynamic GS's soft
count gate, and GETA-3DGS's own named future work.

**Baselines, on the same base representation and the same encoder.** (i) SizeGS and MesonGS++
post hoc at S★ on the frozen 30 k checkpoint, the direct ancestors. (ii) HybridGS, the only other
in-training byte target. (iii) An oracle λ sweep: bisection over full retrains of the same RD
model until the encoded size is within 1 % of S★, which upper-bounds the λ family's quality at
S★ and costs k runs. (iv) HEMGS or SALVQ at their nearest variable-rate point, where code exists
(SALVQ only). (v) Smol-GS and HAC++ at their published points, to place the result on the
leaderboard. (vi) KISS-GS at matched count, the decoupling counter-argument.

**Measurement.** On Mip-NeRF 360, Tanks and Temples and Deep Blending under the 3DGS.zip
protocol, at budgets on the steep part of each curve (Mip-NeRF 360 at 2, 3, 4 and 6 MB, Tanks
and Temples at 2, 3 and 4 MB, Deep Blending at 1, 2 and 3 MB): hit error ε = |R_enc − S★| / S★,
PSNR, SSIM and LPIPS at S★, the number of training runs needed to land (1 against k for the
sweep), wall-clock, and, itemised, the bytes of positions, attributes, networks and headers.
Three seeds, mean and standard deviation, the seed recorded. The bar that makes it a result: ε ≤
1 % in one run on every scene (MesonGS++ reaches 0.3 to 0.9 % post hoc), with quality at S★ at
least equal to the oracle sweep and above the post-hoc baselines by a margin that survives the
seed spread.

### D2. Byte-priced density control

**The quantity.** Each clone, split or prune changes R(θ) by a marginal number of bytes that the
entropy model can price before the operation is applied (the code length of the new primitive's
attributes under the current context, plus its coded position), and changes D(θ) by a marginal
utility that the existing scores estimate (Taming 3DGS's score, SPARE-GS's regional marginal
utility, PUP's Fisher sensitivity). The allocation rule follows SPARE-GS's optimality condition
in bytes: spend the allowance A_t on the candidates with the highest utility per marginal byte
until ∂U/∂byte is equalised across regions.

**Ancestors.** SPARE-GS (∂U_r/∂n_r = λ(t) across active regions, on counts), Taming 3DGS
(score-ranked spending of an allowance), MetaSapiens (utility per cost, with cost in tiles),
HAC++'s mask-aware rate (count control through λ, open loop).

**Baselines.** The same controller with count-priced spending (Taming's score, every primitive
costs one), and with uniform random spending.

**Measurement.** At fixed S★ on the same scenes, PSNR and LPIPS at S★, and the distribution of
bits per primitive across regions. The bar: byte-priced beats count-priced by more than the seed
spread at the two smallest budgets, where the frontier is steep.

### D3. Joint count-precision allocation under one budget

**The question.** At a given S★, is the optimum "more primitives at fewer bits" or "fewer at more
bits", and does it move with S★? GETA-3DGS concludes "whether enough bit budget is allocated
matters far more than how it is allocated across attributes" (p. 13), SizeGS finds "under
different size budgets, the choices of bit-widths are generally the same" (p. 7), and HAC++'s
valid-anchor ratio falls from 0.865 to 0.596 as λ rises. None of them varies N and Δ jointly
under a fixed byte total inside training. MesonGS++ does it post hoc.

**Mechanism.** D1 already carries both levers (A_t for N, Δ_k for bits). The experiment freezes
one and lets the controller move the other, then frees both, at each S★.

**Baseline.** MesonGS++'s post-hoc joint allocation at the same S★ on the same checkpoint.

**Measurement.** PSNR and LPIPS at S★ against (N, mean bits per primitive), on three scenes and
three seeds. The bar: the joint controller lands on or above MesonGS++'s point at every S★, and
the optimal N/bits ratio changes with S★ by a factor that the plot makes visible.

### D4. A runtime-memory bound (phase 3)

**The quantity.** M(θ) = N · b_dec + B_buf(pairs, resolution) + b_net, with b_dec the bytes per
primitive in the layout the rasterizer reads, B_buf the per-tile key and list buffers that depend
on the number of Gaussian-tile pairs, and b_net the bytes of any decoder network. A representation
that renders directly from quantized attributes (Compressed 3DGS's renderer, Mobile-GS's VQ
layout) makes b_dec the quantized size. Anchor methods have b_dec equal to the full decoded
Gaussian plus the MLPs, which is why 3DGS.zip says their VRAM does not fall with count.

**Mechanism.** The D1 controller with R replaced by M: the fast actuator becomes the stored
precision of the render layout, the event-rate actuator stays the allowance, and the measurement
is the peak allocation of the rasterizer on the device.

**Ancestors.** MEGS² (rendering VRAM as an ADMM-constrained optimisation over count and lobe
count, whose proximal loop carries over once the parameter constraint is replaced by a
calibrated byte or time model), FLoD (memory-tiered levels), Gaussians on a Diet (count cap as a
training-memory proxy), Compressed 3DGS (direct rendering of the quantized layout on an Intel UHD
iGPU at 16 FPS), HiGS (closed-form buffer arithmetic).

**Baselines.** MEGS² at its published VRAM reduction, FLoD at the level that fits the device,
and post-hoc pruning to the count that fits.

**Measurement.** Peak VRAM and PSNR on a named 2 GB to 8 GB device (an MX250-class laptop GPU as
in FLoD, or a Jetson Orin Nano), plus the same quantities on the A6000, at M★ ∈ {1, 2, 4} GB. The
bar: the model fits and renders where FLoD reports out-of-memory, at higher PSNR than pruning to
the fitting count.

**Device candidates (from the edge-device reading, all reported).** A Snapdragon 8 Gen 3 phone is
the shared phone reference in Mobile-GS and MEGS² and must be measured at steady state (127 FPS
cold, 74 FPS at thermal equilibrium for Mobile-GS). A Jetson Orin NX 16 GB is the accelerator
reference, where stock 3DGS runs at 7 to 17 FPS against a 60 FPS bar (GBU), and the AGX Orin
carries LS-Gaussian's 90 FPS bar. A 2 GB laptop GPU is where FLoD's out-of-memory rows live.
Decision (user, 2026-09-08): a Jetson-class board is available beside the four A6000s and is the
named device for D4 and D5, its exact model to be recorded when the board is in hand. A phone or
a laptop iGPU is a second device if one is at hand, not a requirement.

### D5. A frame-time bound on a named device (phase 4)

**The quantity.** T(θ) in ms per frame on the device, which the literature says depends on the
number of Gaussian-tile pairs, the per-pixel list length or fragment count, the visible count and
the resolution, not on N alone, and whose dominant stage changes with the device (H9). No fitted
predictor exists. The plan is the hardware-aware NAS recipe: profile the device's rasterizer over
(N_visible, pairs, list length, resolution) on rendered views at steady state, fit a small
predictor T̂ per device with RoofGS's per-stage roofline as the functional form and HiGS's
measured grid as validation data, and use T̂ as R̂ in the D1 loop with the real frame time as the
dual-step measurement. MetaSapiens' CE_i supplies the per-primitive utility per cost for the
allowance, and the paper shows its γ · WS loss term already pushes a compute cost through the
gradient.

**Ancestors.** ProxylessNAS and FBNet (latency lookup tables as constraints), MetaSapiens
(utility per tile intersection, validated against latency on a Jetson AGX Xavier), Speedy-Splat
(training-time pruning for speed), Splats under Pressure (FPS against count per GPU tier),
LS-Gaussian (per-tile workload prediction from viewpoint change).

**Baselines.** MetaSapiens' post-hoc CE pruning to the same FPS, Speedy-Splat, and ControlGS's
hand-mapped λ tiers (its Table I maps three λ values to three iGPU tiers at ≥ 25 FPS, reported).

**Measurement.** Measured ms per frame on the device against T★, PSNR and LPIPS at T★, and the
predictor's error. The bar: within 10 % of T★ on every test view at higher quality than post-hoc
pruning to the same frame time.

### D6. Nested byte budgets from one run (optional)

Matryoshka GS gives any prefix count from one model at a 0.20 dB cost against its own backbone
(reported), PCGS gives 3 to 4 prefix-decodable rate levels, and nobody measures the elastic
against dedicated gap in bytes. A byte-ordered prefix trained under D1's controller would give a
family of budgets from one run. It is a follow-up, not the first paper.

## 7. The substrate decision

This is a fork, not a detail, because it decides which prior art is comparable and whether phases
3 and 4 are possible at all.

| Option | Base | Licence | Render-time network | Quality at 6 MB on Mip-NeRF 360 (reported) | Phases 3 and 4 |
|---|---|---|---|---|---|
| A | gsplat raw 3DGS with the MCMC densifier, plus an entropy layer on raw attributes (learned per-attribute steps, factorized or hyperprior model conditioned on a hash grid of position, positions via G-PCC or a sorted-grid codec) | Apache-2.0 | none, vanilla rasterizer | about 26.6 to 27.0 dB (gsplat compression 26.64 at 6.9 MB, RDO-Gaussian 26.03 at 6.2 MB) | yes, M(θ) and T(θ) are functions of a plain point set |
| B | Scaffold-GS anchors with HAC++'s entropy model and mask, controller on its λ and mask | Inria non-commercial | yes, MLPs per anchor per view, 15 to 31 s decode | 27.60 at 8.7 MB | no for M(θ), VRAM is constant in count |
| C | Smol-GS splat-wise features with its octree coding, controller on λ_o and λ_q | not checked | yes, four MLPs per frame, 210 FPS on an H200 | 27.76 at 5.9 MB | partial |

**Decision (user, 2026-09-08): A for the paper's main results, with B as a transfer
demonstration.** The claim is about control, not about the codec, and the comparison that matters is "controller on
base X against post-hoc on base X", which SizeGS itself set up by reporting on both 3DGS and
Scaffold-GS bases. Option A is the only one whose decoded layout the vanilla rasterizer consumes,
which phases 3 and 4 need, and the only one with a clean licence. Its cost is 0.8 to 1.2 dB
against the anchor frontier at equal bytes, which reviewers will see, so the paper also plugs the
controller into HAC++'s λ and mask (option B) for one table, to show the loop is
representation-agnostic. Option C becomes attractive if its code and licence check out, because
it already carries both levers.

**What to pull into `submodules/` before writing anything**, all confirmed to exist on 2026-09-08:
`nerfstudio-project/gsplat` (base), `YihangChen-ee/HAC-plus` (option B and baseline),
`mmlab-sigs/SizeGS` and `mmlab-sigs/mesongs_plus` (post-hoc baselines and the calibrated
estimator), `w-m/ffsplat` (KISS-GS encoder), `humansensinglab/taming-3dgs` (score-ranked
allowance), `noodle-lab/GaussianSpa` (penalty-projection alternation), `j-alex-hanson/gaussian-splatting-pup`
(post-hoc pruning baseline for K3), `hxu160/SALVQ` (variable-rate baseline). GETA-3DGS, HEMGS and
Smart Target Point Control have no code and can only be cited.

## 8. Kill experiments (phase 0)

Ordered by how cheaply each can end the project. All run on this machine's four RTX A6000s
through `dschedule`, one variable per run, three seeds where a number is compared, commit and
config recorded per run. Two weeks.

**K1. Is R affine in N?** Train gsplat 3DGS on bicycle, garden and room, checkpoint every 1 000
iterations, and encode every checkpoint with one fixed codec (the gsplat compression pipeline,
and the KISS-GS encoder as a second codec). Plot encoded bytes against N across the run, and
bytes against iteration at fixed N after densification stops. **Bar:** if a linear fit of bytes
on N has R² > 0.98 and a worst residual under 5 % of the file, bytes are affine enough that the
sibling's parameter controller ports with one exchange rate, and D1 collapses to a count
controller plus a final exact fit. Otherwise the premise holds and the exchange rate has to be
tracked.

**K2. How wrong is the estimate?** On the option-A substrate with the entropy layer active,
record the noise-relaxed estimate R̂ and the encoded R at every 500 iterations. **Bar:** if the
ratio R / R̂ has a standard deviation under 3 % over the run, a real encode every 500 iterations
suffices for the dual step. If it drifts with N or with iteration, the cadence and the
exchange-rate re-fit are set from this plot.

**K3. Does placing beat removing, on counts?** Train toward N with the MCMC cap and with Taming's
allowance, against training free and pruning to N with PUP's score followed by 5 000 fine-tuning
iterations, at N ∈ {250 k, 500 k, 1 M} on three scenes, three seeds. This is the clean test no
paper runs (H6). **Bar:** train-to-N wins by more than the seed spread in PSNR or LPIPS at the two
smaller N. If it loses, the placement hypothesis is dead for counts, and the paper's claim
narrows to one-run hit accuracy plus the runtime bounds of D4 and D5, which post-hoc methods
cannot address.

**K4. Does the trajectory shape matter?** Same as K3 at N = 500 k, with the budget trajectory
monotone (ω = 1), with a scheduled overshoot (ω ∈ {1.5, 2}), and with explore-then-sparsify
(GaussianSpa's alternation from a 15 k checkpoint). **Bar:** the ω = 1.5 variant is within
0.1 dB of explore-then-sparsify while its peak count never exceeds ω · N, which is what makes it
usable under a training-memory bound.

**K5. Does the loop hit a byte target in one run?** The D1 loop with λ as the only actuator (Δ_k
fixed, densification allowance off) on the option-A substrate and, in parallel, wrapped around
HAC++'s λ, at S★ ∈ {3, 6, 12} MB on three scenes. **Bar:** ε ≤ 2 % in one run on every scene,
and PSNR within 0.1 dB of the oracle λ sweep. This is the minimum viable result of the paper. If
it fails, the failure mode (oscillation, windup, estimator drift) is the next design problem, and
K2 says which.

**What K1 to K5 cost (assumed, not measured):** about 120 training runs of 20 to 40 minutes on
one A6000 each, feasible on four GPUs in under a week of wall-clock, plus the baseline
reproductions.

## 9. Roadmap: the fastest path to a result

The tracks below map onto five phases, numbered 0 to 4: track 0 is phase 0, track 1 is phase 1,
track 2 is split between phases 0 and 2, tracks 3, 4 and 6 are phase 2, track 5 is phase 3, and
D5 is phase 4.

The user's instruction is to push the work as far and as fast as possible rather than scope it to
a venue, so the plan is ordered by dependency, not by deadline, and every track that does not
depend on another's result runs in parallel on the four A6000s from day one.

| Track | Starts | Depends on | Deliverable | Stop or branch rule |
|---|---|---|---|---|
| 0, kill experiments K1 to K5 | now | submodules pulled, three scenes per dataset prepared | five devlogs with numbers | K1 affine: D1 collapses to a count controller plus an exact fit. K3 lost: the claim narrows to one-run hit accuracy plus the runtime bounds. K5 failed: fix the loop before anything else, K2 says where |
| 1, controller v1 (D1, three actuators) on option A | now, in parallel with track 0 | K2's encode cadence and K5's loop once they land | the controller landing S★ on all nine Mip-NeRF 360 scenes | ε > 2 % on any scene: back to track 0's diagnosis |
| 2, baselines | now, in parallel with track 0 | submodules | SizeGS, MesonGS++, the HAC++ oracle sweep, KISS-GS and Smol-GS reproduced on the same scenes | a baseline more than 0.2 dB off its paper: protocol check before any comparison |
| 3, allocation (D2, D3) | after track 1 lands | controller v1 | the byte-priced and joint-allocation ablation tables | none |
| 4, full protocol | after tracks 1 to 3 | | four datasets × three budgets × three seeds, itemised bytes, timing tables | |
| 5, device bound (D4) on the Jetson | after track 1, in parallel with track 4 | a plain-point-set model from track 1, the board in hand | the VRAM-bound table on the Jetson, then the D5 predictor | |
| 6, writing | continuous from track 1's first landing | | an arXiv preprint as soon as tracks 1, 2 and 4 hold | |

**Calendar facts, not targets** (as listed by the conference pages and trackers on 2026-09-08, to be
confirmed on the official calls): CVPR 2027 closes 2026-11-16, SIGGRAPH 2027 is listed at
2027-01-22, ICCV 2027 tentatively at 2027-03-05. The paper goes to whichever deadline comes next
once tracks 1, 2 and 4 hold, and the preprint goes out before that.

**Estimated wall-clock (assumed, not measured):** tracks 0 to 2 in two to three weeks, tracks 3 to 5
in four to five weeks more, so a complete draft in about eight weeks if nothing kills the idea,
and an earlier stop if K1 or K3 does.

## 10. The paper, sketched

**Working title.** "Rate Control for 3D Gaussian Splatting: Training onto a Byte Budget".

**Claims, each tied to a table.**

1. One training run lands a scene within 1 % of a user-given byte budget on every scene of four
   datasets, where the λ methods need a per-scene sweep and the post-hoc methods need a frozen
   model (D1 table, hit-error column).
2. At budgets on the steep part of the curve, quality at the budget is above the post-hoc search
   on the same base and at least equal to the oracle λ sweep at 1/k of its cost (D1 table, PSNR
   and LPIPS columns, and the runs column).
3. Pricing densification in bytes beats pricing it in primitives at the two smallest budgets (D2
   table).
4. The loop is representation-agnostic: the same controller lands HAC++ on a budget (option B
   table).
5. If track 5 lands in time: the same loop lands a model on the Jetson under a VRAM bound, and
   on a 2 GB-class device where FLoD reports out-of-memory (D4 table).

**Figures.** Rate trajectories R_t against S_t for nine scenes at one S★ on one axis. The
rate-distortion curve at low rates against the leaderboard frontier. A histogram of per-scene hit
errors against HybridGS's and the λ family's spread. The (N, bits per primitive) allocation as a
function of S★. Peak VRAM and FPS on two devices.

**Tables.** Main results at three budgets per dataset with itemised bytes. Ablations: actuators,
trajectory shape ω, encode cadence C, anti-windup. Timing: training, encode, decode, FPS on the
A6000 and on one edge device, peak VRAM. All with three seeds.

## 11. Risks and the counter-arguments to answer

- **The flat frontier.** Above 6 MB PSNR moves 0.17 dB over 3.6× in size, so a reviewer will ask
  what a byte target buys there. Answer: the paper's budgets sit below 6 MB, where the frontier
  drops 1.7 dB per MB, and the tables lead with LPIPS.
- **KISS-GS's decoupling argument.** Answer: the controller is not a codec, and for M★ and T★
  there is no post-hoc equivalent because post-hoc means remove-only (H7, H12). The paper also
  reports the controller on two representations.
- **Post-hoc is cheap.** MesonGS++ hits 0.3 to 0.9 % in a minute. Answer: the paper never claims
  to win on hit accuracy or time, it claims quality at the budget (claim 2) and one run instead
  of a sweep (claim 1). If K3 and D1 do not show a quality margin, the storage-bound paper is not
  worth writing and the program leads with D4 and D5.
- **GETA-3DGS and HybridGS.** Both will be cited by reviewers as prior byte targets in training.
  Answer: Section 0's claim wording, with GETA's own non-binding disclosure and HybridGS's late,
  one-lever, shrink-only stage and its −45 % to +8 % misses quoted from their pages.
- **Raw 3DGS is below the anchor frontier.** Answer: matched-base comparisons plus the HAC++
  transfer table, and the honest statement that phases 3 and 4 need a plain point set.
- **Licences.** Baselines under the Inria licence are run, never redistributed, and the method
  code is on gsplat.
- **HEMGS could be searched post hoc over its continuous λ input.** Its authors never do it and
  its code is unreleased, so it can only be discussed, not run.

## 12. Decisions taken (user, 2026-09-08)

The five open questions of the first draft were answered in the file by the user, and the answers
are applied throughout this document.

1. **The storage size is the primary bound for the first paper**, because it has baselines and a
   protocol. Runtime VRAM (D4) and frame time (D5) follow as track 5 and the second paper.
2. **A Jetson-class board is available** beside the four A6000s here, and is the named device for
   D4 and D5. Its model is recorded when the board is in hand. The literature's reference points
   on that family are the AGX Orin (LS-Gaussian's 90 FPS bar) and the Orin NX 16 GB (stock 3DGS
   at 7 to 17 FPS against a 60 FPS bar, GBU).
3. **Substrate A**: gsplat with the MCMC densifier and an entropy layer on raw attributes. HAC++ is
   a baseline and a transfer table only.
4. **No venue is targeted.** The work is pushed as far and as fast as possible, the plan is ordered
   by dependency (Section 9), and the paper goes to whichever deadline comes next once the results
   hold.
5. **The sibling project's controller is a design reference only.** Its ranked-walk spend is the
   reference for the shape of the byte allowance in D1, none of its code is reused, and the byte
   controller is written in this repository.
