# C1. Count-budget family: reading notes

Four papers from `references/03_count_budget/`, every page read via pymupdf text extraction plus
rendered crops of the figures that carry numbers only graphically. Conventions used throughout:

- Every number is copied from the PDF and marked "as reported". A number read off a plot is marked
  "approx. plot reading" and is mine, not the paper's.
- Page numbers are PDF page indices. They coincide with the printed page numbers for Taming 3DGS,
  Mini-Splatting and Smart Target Point Control. GaussianSpa prints no page numbers, so PDF indices
  are used (main paper 1 to 8, references 9 to 10, supplementary 11 to 14).
- Quotes are verbatim, including the papers' own em dashes.
- Tables in this file use `/` to separate the columns of one row, with the column order named above
  each table.

---

## A. Taming 3DGS

### 1. Citation
- Title: Taming 3DGS: High-Quality Radiance Fields with Limited Resources
- First author: Saswat Subhajyoti Mallick (equal contribution with Rahul Goel), CMU, TU Graz, IIIT
  Hyderabad
- Venue as stated in the PDF: not stated. The PDF is `arXiv:2406.15643v1 [cs.CV] 21 Jun 2024`. The
  file name says SIGGRAPH Asia 2024, and the Smart Target Point Control PDF cites it as "In SIGGRAPH
  Asia 2024 Conference Papers, pages 1–11, 2024" (TPC p. 9, ref. [8]).
- arXiv id: 2406.15643

### 2. Base representation and renderer
Vanilla 3DGS (Kerbl et al.) with SH order 3, tile-based differentiable rasterizer, L1 plus SSIM
loss. Implementation "is based on top of the original 3DGS codebase" (p. 8) with modified CUDA
kernels: per-splat parallel backward, separate SH tensors, separable fused SSIM kernels, tighter
culling from StopThePop. Batched SH updates (one ADAM step every 16 iterations for bands above the
first), SH learning rate 0.001, opacity learning rate 0.025 (as reported, p. 8). After 15K iterations
Gaussians become "high-opacity Gaussians": opacity activation replaced by abs, blending weights
clamped to 1 (p. 6).

### 3. Budget handle
- User supplies the final Gaussian count B, "the final count (budget)" (p. 4). In the experiments
  it is set per scene as a multiplier of the SfM point count (2×, 5×, 15×, see field 8), or set
  equal to the 3DGS count for the second scenario.
- TARGET, not a knob: "Given a user-defined model size, we ensure a deterministic training schedule
  that can yield the exact number of desired Gaussians." (p. 2)
- Achieved equals requested: yes. Table 1 reports final `#G` equal to `Peak #G` for Ours in every
  dataset (0.29 / 0.29 on T&T, 0.63 / 0.63 on M360, 0.27 / 0.27 on DB, in 10⁶, as reported), and
  "Ours (#3DGS)" reaches exactly the 3DGS counts 1.84, 3.31, 2.81 (10⁶, as reported): "we
  demonstrate that our budgeting mechanism allows to match their model size exactly." (p. 9). Table 2
  caption: "Note that all configurations yield the same number of Gaussians." (p. 9)

### 4. When and how it binds
- Mechanism: purely constructive, score-based sampling with a per-event allowance. Densification
  runs every 500 iterations, "one-fifth of 3DGS" (p. 5). At each event the allowance is
  B_add = T − |G| (Alg. 1 line 19, p. 6), where T is the accumulated target count at the current
  iteration and |G| the current count. B_add Gaussians are drawn by weighted random sampling from
  all Gaussians with the score vector S_G as weights: "we perform densification by randomly
  resampling B primitives from all Gaussians using S_G as sampling weights" (p. 6). Each sampled
  Gaussian is split if ∇g > Gt and radius > Rt, cloned if ∇g > Gt and radius ≤ Rt (Alg. 1 lines
  22 to 26). Low-opacity pruning of 3DGS still happens, and the "accumulated target" difference
  compensates for it: "Since 3DGS prunes low-opacity Gaussians over time, following an additive
  schedule directly may produce fewer primitives than the given target. To avoid this, we instead
  compute the difference between our current and accumulated target count and densify the
  corresponding number of primitives." (p. 4)
- Budget schedule (Eq. 2, p. 4), printed exactly as:

  A(x) = (B − S − 2N)/N² · x² + 2x + B

  "where N is the number of densification steps, B is the final count (budget), and S is the number
  of SfM points at initialization." The curve is described as "a parabolic curve that starts from the
  SfM initialization and peaks precisely at the user-defined budget" (p. 4). My observation, not
  the paper's: with the constant term B as printed, A(0) = B and A(N) = 2B − S, whereas with a
  constant term S one gets A(0) = S and A(N) = B, matching the prose. Treat the printed constant as
  a possible typo and check the code before reusing the formula. Fig. 2b (p. 3) shows the
  normalized schedule rising from 0 at iteration 0 to 1.0 at 15K iterations, fitted to the 3DGS
  growth curves of the M360 outdoor scenes ("the number of Gaussians added in each step follows a
  trend of quadratic decrease", p. 4). Target reached at the end of the densification window, 15K
  iterations (Fig. 2b), out of 30K total ("midpoint of our training (15K iterations)", p. 6). The
  value of N is not stated explicitly.
- Score (Eq. 3 to 5, p. 5): per-view saliency Sᵥ = 1_ROI ⊙ (λ₁ L1(v, rᵥ) + λ₂ E(v)) with E a
  Laplacian filter, λ₁ = λ₂ = 0.5 (as reported). Per-Gaussian score S_g = ∑[i=1..N] F(∇g, cᵢ_g,
  1ᵢ_g, Dᵢ_g, Sᵢᵥ, Bᵢ_g, zᵢ_g, o_g, s_g), summed over N = 10 uniformly sampled training views
  (as reported, p. 6). Each term is median-scaled, multiplied by the view's photometric loss, then
  weighted: ∇g (50), pixel count c (0.1), pixel-to-center distance D (50), saliency S (10),
  blending weight B (50), depth z (5), opacity o (100), scale s (25) (weights as reported, p. 5 to
  6). Alg. 1 line 16: S_g = S_g + Pᵢ · F(...), with Pᵢ the photometric loss of view i.
- Training memory never peaks above the target because there is no over-densification: "we avoid
  unnecessary peaks in the optimization that could violate the user's hardware or budget
  constraints." (p. 2)

### 5. Meaning of "size" or "count" in tables
Table 1 columns are `#G (10⁶)` (final number of Gaussians) and `Peak #G` (peak during training).
"Model size" in the prose means the Gaussian count. A storage size in MB is given only in Fig. 1
(p. 2), Garden: 3DGS "Size:1,450 MB, #Gaussians: 6 M, PSNR: 25.2 dB, 47 mins" versus Ours
"#Gaussians: 0.8 M, PSNR: 24.97 dB, Size:196 MB, 9 mins" (as reported). My computation from those
labels: 196 MB / 0.8 M ≈ 245 B per Gaussian, 1 450 MB / 6 M ≈ 242 B per Gaussian, consistent
with an uncompressed float32 ply. No MB column in any table.

### 6. Runtime cost reported
- Hardware: "The evaluation was conducted using an NVIDIA RTX A4500 GPU." (p. 8)
- Train time (Table 1, as reported): Ours 7 m (T&T), 11 m (M360), 7 m (DB). Ours (#3DGS) 20 m,
  32 m, 22 m. 3DGS 28 m, 43 m, 39 m. Mini-Splatting 20 m, 30 m, 24 m. [26] 18 m, 25 m, 22 m.
  INGP-Big 7 m, 8 m, 8 m.
- Training VRAM: "our first budgeted scenario allows training with consistently less than 10 GB
  VRAM—compact enough for a mid-range NVIDIA RTX 3080." (p. 10)
- Render FPS: not stated. Rendering GPU memory: not stated.
- Backward-pass timing on Bicycle in Fig. 4 (p. 8), values only graphical. Speed versus gsplat 1.0:
  "can outperform the recently released version 1.0 of gsplat [38] by 1.5×–2×" (p. 8).
- Ablation timings (Table 2, p. 9): removing the per-splat backward raises time from 7 m to 14 m
  (Truck) and 9 m to 17 m (Train), as reported.

### 7. Evidence on training-time budget versus post-hoc pruning
- No table compares against a post-hoc pruning method at the same final count. Table 1 top half
  compares at each method's own count (T&T / M360 / DB, columns SSIM / PSNR / LPIPS / Train time /
  #G / Peak #G, as reported):
  - C3DGS: 0.843 / 23.57 / 0.182 / 28 m / 1.53 / 1.84 ‖ 0.811 / 27.34 / 0.221 / 43 m / 2.44 / 2.94 ‖ 0.900 / 29.54 / 0.252 / 39 m / 2.43 / 2.81
  - RVQ: 0.831 / 23.30 / 0.202 / 27 m / 0.83 / 1.46 ‖ 0.797 / 26.99 / 0.245 / 48 m / 1.41 / 2.57 ‖ 0.901 / 29.75 / 0.260 / 38 m / 1.04 / 2.25
  - [26] Papantonakis: 0.844 / 23.66 / 0.178 / 18 m / 0.71 / 0.71 ‖ 0.814 / 27.43 / 0.220 / 25 m / 0.83 / 0.83 ‖ 0.902 / 29.57 / 0.247 / 22 m / 0.97 / 0.97
  - Mini-Splatting: 0.847 / 23.42 / 0.181 / 20 m / 0.31 / 4.32 ‖ 0.822 / 27.26 / 0.217 / 30 m / 0.50 / 4.32 ‖ 0.909 / 30.04 / 0.244 / 24 m / 0.56 / 4.51
  - Ours: 0.837 / 23.95 / 0.201 / 7 m / 0.29 / 0.29 ‖ 0.801 / 27.31 / 0.252 / 11 m / 0.63 / 0.63 ‖ 0.904 / 29.82 / 0.260 / 7 m / 0.27 / 0.27
  Counts are close but not matched (0.29 vs 0.31 on T&T, 0.63 vs 0.50 on M360, 0.27 vs 0.56 on
  DB). At those counts Ours beats Mini-Splatting in PSNR on T&T and M360 and loses on DB, and loses
  in LPIPS on all three.
- Matched count against unconstrained 3DGS (not a pruning method), Table 1 bottom half:
  - 3DGS: 0.847 / 23.65 / 0.176 / 28 m / 1.84 / 1.84 ‖ 0.815 / 27.46 / 0.215 / 43 m / 3.31 / 3.31 ‖ 0.904 / 29.64 / 0.243 / 39 m / 2.81 / 2.81
  - Ours (#3DGS): 0.851 / 24.04 / 0.170 / 20 m / 1.84 / 1.84 ‖ 0.822 / 27.79 / 0.205 / 32 m / 3.31 / 3.31 ‖ 0.907 / 30.14 / 0.235 / 22 m / 2.81 / 2.81
  This is a matched-count comparison. It shows budgeted score-based densification beating the 3DGS
  heuristic at identical count, not beating post-hoc pruning.
- Matched-count ablation (Table 2, p. 9, T&T, PSNR / LPIPS / Time for Truck then Train, as
  reported, all rows have the same Gaussian count):
  - Ours: 25.20 / 0.165 / 7 m ‖ 22.69 / 0.238 / 9 m
  - –score-based sampling: 24.92 / 0.189 / 6 m ‖ 22.24 / 0.246 / 8 m
  - –image loss: 24.94 / 0.187 / 7 m ‖ 22.08 / 0.242 / 9 m
  - –high opacity: 25.01 / 0.174 / 7 m ‖ 22.29 / 0.239 / 9 m
  - –reduce SH frequency: 25.39 / 0.161 / 9 m ‖ 22.75 / 0.235 / 12 m
  - –per splat backward: 25.20 / 0.165 / 14 m ‖ 22.69 / 0.238 / 17 m
- Fig. 1 right (p. 2), Garden PSNR versus #Gaussians (approx. plot reading): Ours curve ≈ 25.3 dB at
  0.2 M, 25.75 at 0.3 M, 26.2 at 0.4 M, 26.8 at 0.7 M, 27.3 at 1.35 M, 27.5 at 2.1 M, 27.63 at
  2.75 M, 27.68 at 4.05 M. Single points: Mini-Splatting* ≈ 26.7 at 0.55 M, R-VQ ≈ 26.78 at 2.2 M,
  Papantonakis ≈ 27.28 at 2.35 M, C3DGS ≈ 27.35 at 5.75 M, 3DGS ≈ 27.28 at 5.75 M. Read this way:
  Mini-Splatting* sits on or slightly above the Ours curve at its count, R-VQ and Papantonakis sit
  below the Ours curve at theirs.
- The paper's own framing of the difference: "Mini-Splatting—similar to 3DGS—relies on heavily
  oversampling the scene before pruning, creating a vast gap of up to 10× between their peak and
  final model size. In contrast, our method uses a purely constructive optimization that only adds
  Gaussians towards an exact target budget." (p. 9)

### 8. Scene dependence
- Budget set per dataset class via multipliers of the SfM count: "For the small-scale indoor scenes
  in MipNeRF-360, we set the budget to 2× the SfM points. For the larger, full-room indoor captures
  of Deep Blending, we use 5×, and for unbounded outdoor scenes, we use 15×. For the outdoor
  Tanks&Temples, the initial SfM point count is significantly higher, thus we set the budget to 2×
  here as well. Note that this parameterization could be automatized by providing scenes in
  real-world coordinates or a corresponding multiplier." (p. 9). Per-scene absolute budgets are not
  listed.
- On 3DGS variability: "even when starting from the same number of input points, the difference
  between two reconstructed scenes w.r.t. the number of Gaussians (and thus required storage) can be
  as much as one order of magnitude." (p. 2)
- Fig. 1 right shows quality rising monotonically with budget on Garden: "We see a consistent
  improvement as budget increases, showing a clear correlation between provided budget and achieved
  image quality." (p. 10)

### 9. Main quantitative results
See the Table 1 rows copied in field 7 (Ours, Ours (#3DGS), 3DGS, Mini-Splatting, [26], C3DGS,
RVQ). Additional Table 1 rows, as reported: INGP-Big 0.745 / 21.92 / 0.305 / 7 m ‖ 0.699 / 25.59 /
0.331 / 8 m ‖ 0.817 / 24.96 / 0.390 / 8 m. Plenoxels 0.719 / 21.08 / 0.379 / 25 m ‖ 0.626 / 23.08 /
0.463 / 26 m ‖ 0.795 / 23.06 / 0.51 / 28 m. MipNeRF360 0.759 / 22.22 / 0.257 / 48 h ‖ 0.792 /
27.69 / 0.237 / 48 h ‖ 0.901 / 29.4 / 0.245 / 48 h. Zip-NeRF (M360 only) 0.828 / 28.54 / 0.189 /
1.5 h. No MB in tables. Strongest baselines at low count: Mini-Splatting and [26].

### 10. Densification details
The 3DGS split/clone operators are kept, and the gradient threshold Gt and radius threshold Rt still
decide split versus clone. What changes: (a) candidates are chosen by weighted random sampling with
the score S_G instead of thresholding the positional gradient, (b) the number of candidates per
event is fixed by the schedule as T − |G|, (c) events happen every 500 instead of every 100
iterations, (d) there is no reset-and-prune oversampling cycle beyond 3DGS's own low-opacity
removal. Region-of-interest masks can bias the saliency (Fig. 6, weight raised to 10³ on a face
mask, p. 11).

### 11. Stated limitations and future work (verbatim)
- p. 11: "While our approach is an important step towards low-cost, high-quality radiance fields,
  achieving optimal quality still requires a substantial sample count and meandering exploration as
  Gaussians move across the scene. We consider efficient search paths, occupancy predictions, and
  resolution of blind spots in scene reconstructions as exciting avenues for future work."
- p. 9 (implicit, budget setting): "Note that this parameterization could be automatized by
  providing scenes in real-world coordinates or a corresponding multiplier."

### 12. Sentences on size, byte, bitrate, memory or FPS targets, rate control, or edge deployment (verbatim)
- p. 1: "Especially on constrained devices, training performance degrades quickly and often cannot
  complete due to excessive memory consumption of the model."
- p. 1: "To address these issues, we tackle the challenges of training and rendering 3DGS models on
  a budget."
- p. 1: "Model size continuously increases in a controlled manner towards an exact budget, using
  score-based densification of Gaussians with training-time priors that measure their contribution."
- p. 1: "Our evaluation shows that in a budgeted setting, we obtain competitive quality metrics with
  3DGS while achieving a 4–5× reduction in both model size and training time. With more generous
  budgets, our measured quality surpasses theirs. These advances open the door for novel-view
  synthesis in constrained environments, e.g., mobile devices."
- p. 2 (Fig. 1 caption): "Right: Our method produces models with an exact, user-specified target
  size, surpassing 3DGS quality as the target increases."
- p. 2: "A typical 3DGS model can yield several millions of Gaussians for a single unbounded scene
  and require more than one gigabyte of disk space. Such substantial memory usage and geometry
  workload complicate real-time rendering on low-end devices, preventing application in constrained
  settings like network streaming or AR/VR on embedded systems."
- p. 2: "In addition to being excessive, the memory consumption of 3DGS is also hard to predict:
  even when starting from the same number of input points, the difference between two reconstructed
  scenes w.r.t. the number of Gaussians (and thus required storage) can be as much as one order of
  magnitude."
- p. 2: "Given a user-defined model size, we ensure a deterministic training schedule that can yield
  the exact number of desired Gaussians."
- p. 2: "Therefore, we avoid unnecessary peaks in the optimization that could violate the user's
  hardware or budget constraints."
- p. 2: "1. A purely constructive, budget-constrained optimization, enabling full control over model
  size and resources."
- p. 3: "Other aspects of previously proposed on-disk compression techniques, such as code-booking
  or entropy minimization, are directly compatible with our method, which would lead to even smaller
  file sizes due to our higher primitive reduction."
- p. 4: "The maximum number of new Gaussians added at every stage is pre-determined: although our
  method mimics the original 3DGS growth curve, the peak (and final) number of Gaussians is fully
  controllable by the user who provides the limits for model size (Fig. 2b)."
- p. 10: "While our approach does not target the peculiarities of PyTorch, we note that our first
  budgeted scenario allows training with consistently less than 10 GB VRAM—compact enough for a
  mid-range NVIDIA RTX 3080."
- p. 11 (Fig. 6 caption): "This demonstrates the potential of our 3DGS budgeting for
  latency-constrained live scenarios: In a telepresence setting, we could prioritize the quality of
  the most frequently observed image regions—e.g., faces—and leave others under-sampled, without
  significantly degrading the user experience."
- p. 11: "These properties generate new opportunities for optimizing novel-view synthesis in various
  environments, e.g., hardware-constrained and edge devices. Other potential applications include
  latency-constrained streaming services, where on-the-fly, interactive 3D reconstructions could be
  steered towards prioritizing salient regions of interest, such as faces (see Fig. 6)."
- p. 11: "Our contributions are complementary to ongoing 3DGS compression efforts, many of which
  could be applied to our reduced-size models to even greater effect."
- Reading: every "size" and "budget" in the paper is a primitive count. Bytes appear only as
  reported outcomes (Fig. 1). No byte, bitrate or FPS target is bound during training.

### 13. Code
"We provide the source code for our optimizations at https://github.com/nullptr81/3dgs-accel." (p. 8).
Licence: not stated.

---

## B. Mini-Splatting

### 1. Citation
- Title: Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians
- First author: Guangchi Fang (with Bing Wang, corresponding), Spatial Intelligence Group, The Hong
  Kong Polytechnic University
- Venue as stated in the PDF: not stated. The PDF is `arXiv:2403.14166v3 [cs.CV] 16 Oct 2024` in
  LNCS layout. The file name says ECCV 2024.
- arXiv id: 2403.14166

### 2. Base representation and renderer
Vanilla 3DGS, PyTorch, with the rasterization module modified "to render Gaussian indexes and depth
points" (p. 11), i.e. the index i_max(x) of the Gaussian with maximal blending weight per pixel and a
mid-point ellipsoid depth. Three variants: Mini-Splatting (densify then simplify), Mini-Splatting-D
(densify only), Mini-Splatting-C (Mini-Splatting plus post-hoc RAHT transform coding and zip).
SH coefficients are enabled only after simplification at 15K: "we only enable SH coefficients and
increase the SH level after simplification (15K)." (p. 10)

### 3. Budget handle
- KNOB: a sampling ratio in the importance-weighted sampling step. "The curve for our Mini-Splatting
  is generated by controlling the sampling ratio during the importance-weighted sampling process."
  (p. 12) and "the number of Gaussians in our Mini-Splatting is manually controlled by the sampling
  ratio." (p. 21). The ratio value used for the main results is not stated.
- No target count is supplied. The achieved count is an outcome and varies per dataset (Table 1 Num,
  in millions, as reported: 0.49 on M360, 0.20 on T&T, 0.35 on DB) and per scene (Fig. 8 labels,
  as reported: bicycle 0.54 M, truck 0.36 M).
- Achieved equals requested: not applicable, nothing is requested in count units.

### 4. When and how it binds
Two stages inside a 30K-iteration run (Alg. 1, p. 22, and Appendix F, p. 21):
- Densification, iterations 0 to 15K ("DensificationIteration is set as 15K", p. 21). At each
  refinement iteration: BlurSplit() then the 3DGS SplitAndClone(). Blur split (Eq. 2, p. 7):
  G^blur = { Gᵢ | Sᵢ > T_blur }, T_blur = θ_blur · H · W, with Sᵢ the number of pixels where Gᵢ is
  the max-contribution Gaussian and θ_blur = 2×10⁻⁴ (as reported). Those Gaussians are split with
  the 3DGS split rule. Depth reinitialization every 5K iterations ("our depth reinitialization is
  enabled every 5K iterations", p. 21): render the mid-point depth d^mid of the max-contribution
  Gaussian (ray–ellipsoid intersection midpoint, Appendix D), reproject all views, randomly sample
  "around 3.5 million" points per scene with ground-truth colours (as reported, p. 8), and
  reinitialize the Gaussian set from them.
- Simplification at 15K (SimplificationIteration1) and 20K (SimplificationIteration2), as reported
  p. 21. At 15K: intersection preserving (Eq. 3, p. 9, keep only Gaussians that are i_max for at
  least one pixel in at least one view), then importance-weighted sampling with probability
  Pᵢ = Iᵢ / ∑[i=1..N] Iᵢ (p. 10) where Iᵢ is the accumulated blending weight I¹ᵢ = ∑[j=1..K] wᵢⱼ
  indoors, or I²ᵢ = ∑[m=1..M] I⁽ᵐ⁾ᵢ · δ(i ∈ I⁽ᵐ⁾_max) with I⁽ᵐ⁾ᵢ = ∑ w⁽ᵐ⁾ᵢⱼ / S⁽ᵐ⁾ᵢ outdoors (Eq.
  12, p. 20), followed by "Gaussians ← Gaussian Centers" (re-initialization of attributes from the
  sampled centers). At 20K: intersection preserving again plus "Directly Prune a Few Gaussians".
- Sampling rather than thresholding is the core claim: "Compared to deterministic pruning, a
  stochastic sampling strategy can better maintain overall geometry." (p. 10)
- The count is therefore fixed at 15K by a ratio applied to whatever the densified set is, and
  optimisation continues to 30K.

### 5. Meaning of "size" or "count" in tables
Table 1 column `Num` is the number of Gaussians in millions. Table 2 `Num` likewise. Storage in MB
appears only in Fig. 9 (p. 13), a "Rate-distribution" plot for Mini-Splatting-C on M360 (approx.
plot reading: Mini-Splatting-C ≈ 26.8 dB at ≈ 17 MB, ≈ 27.15 dB at ≈ 27 MB, ≈ 27.4 dB at ≈ 48 MB,
Niedermayr et al. ≈ 26.98 dB at ≈ 28 MB, Lee et al. ≈ 27.03 dB at ≈ 28 MB). Mini-Splatting-C stores
centers as float32 and codes the other attributes with RAHT (depth level 16, quantization step 0.02)
plus `numpy.savez_compressed()` (as reported, p. 21).

### 6. Runtime cost reported
Table 2 (p. 13), M360, "all other models are trained with a RTX 3090 GPU", Mini-Splatting* on a
"GTX 1060 6G GPU", peak memory via `torch.cuda.max_memory_allocated()`. Columns Num / training time
/ training memory / rendering FPS / rendering memory, outdoor then indoor, as reported:
- 3DGS: 4.86 / 30m8s / 7.45GB / 98 / 2.79GB ‖ 1.46 / 24m41s / 2.75GB / 151 / 1.07GB
- Mini-Splatting-D: 5.40 / 31m48s / 7.45GB / 83 / 3.12GB ‖ 3.80 / 40m13s / 5.55GB / 83 / 2.46GB
- Mini-Splatting: 0.57 / 17m56s / 2.61GB / 410 / 0.40GB ‖ 0.40 / 27m2s / 2.77GB / 362 / 0.35GB
- Mini-Splatting*: 0.57 / 101m11s / 2.61GB / 64 / 0.40GB ‖ 0.40 / 154m / 2.82GB / 40 / 0.35GB
Fig. 1 (bicycle, as reported): Mini-Splatting-D 6.0 mil, 75 FPS, 33 min, 25.5 dB. 3DGS 6.1 mil,
66 FPS, 35 min, 25.2 dB. Mini-Splatting 0.6 mil, 430 FPS, 17 min, 25.2 dB. Note that the
training-time saving is small (peak memory during densification equals 3DGS's, 7.45 GB outdoors)
because simplification happens at 15K, halfway.

### 7. Evidence on training-time budget versus post-hoc pruning
Mini-Splatting is itself a simplify-at-15K method, so its comparisons are "sampling plus
reorganisation" versus "direct pruning", both applied mid-training to a densified model.
- Fig. 2 (p. 5, bicycle, labels as reported, format (PSNR, millions)): 3DGS (25.2, 6.1), Pruning
  (24.9, 0.6), Random (24.8, 0.6), Grid (23.5, 1.1), Density (24.7, 0.6). Text: "The pruning
  strategy can reduce the number of Gaussians from 6.1 million to 0.6 million, but it also degrades
  PSNR from 25.2 dB to 24.9 dB, and the phenomena of 'overlapping' and 'under-reconstruction' still
  exist." Matched count 0.6 M between pruning, random and density sampling.
- Fig. 6 (p. 9, bicycle, Ratio vs chamfer distance, PSNR labels as reported): Pruning 23.21, 24.20,
  25.01, 25.28 at preserving ratios ≈ 8, 18, 32, 55 % (ratios approx. plot reading). Sampling
  23.28, 24.41, 24.99, 25.28 at ≈ 5, 16, 31, 50 %. Chamfer distance of pruning is far higher at low
  ratios (≈ 2.4 vs ≈ 0.6 at the lowest ratio, approx. plot reading).
- Fig. 7 (p. 11, M360 average, approx. plot readings on a log axis): Mini-Splatting curve ≈ 27.05 dB
  at 0.3 M, ≈ 27.3 at 0.5 M, ≈ 27.5 at 0.85 M. VQ (pruning of [21] applied to 3DGS) ≈ 27.27 at
  0.75 M. LightGaussian (re-implemented prune and recovery) ≈ 27.05 at 1.1 M. Lee et al. [20]
  (from their paper) ≈ 27.08 at 1.35 M. 3DGS ≈ 27.47 at 3.35 M. Mini-Splatting-D ≈ 27.54 at 4.7 M.
  LPIPS panel: Mini-Splatting ≈ 0.24 / 0.218 / 0.196 at the same three counts, VQ ≈ 0.228,
  LightGaussian ≈ 0.238, Lee ≈ 0.246. Paper's statement: "it is clear that our Mini-Splatting
  outperforms all its counterparts at similar numbers of Gaussians." (p. 12). Not exactly matched
  counts, but overlapping ranges.
- Fig. 10 (p. 14, M360 ablation of simplification, approx. plot readings): Baseline 3DGS ≈ 27.47 at
  3.35 M, Add Densification ≈ 27.54 at 4.7 M, Add Pruning (Mini-Splatting-D plus direct pruning at
  tuned ratios) ≈ 26.7 at 0.4 M, ≈ 27.05 at 0.65 M, ≈ 27.48 at 1.1 M. Add Intersection ≈ 27.48 at
  0.9 M. Add Sampling ≈ 27.05 at 0.2 M, ≈ 27.3 at 0.45 M, ≈ 27.47 at 0.9 M. Paper's statement: "our
  simplification algorithm outperforms direct pruning, achieving approximately a halved reduction in
  the number of Gaussians while maintaining comparable rendering quality." (p. 14)
- Nothing in the paper knows the count from the start of training. The evidence supports
  "densify widely, then sample with geometry awareness" over "prune by importance" at the same count.

### 8. Scene dependence
- Counts differ by dataset and scene under the same ratio (Table 1: 0.49 / 0.20 / 0.35 M, Table 2
  outdoor 0.57 M vs indoor 0.40 M, Fig. 8 bicycle 0.54 M, truck 0.36 M, all as reported).
- The importance metric is chosen per scene type: "It is noteworthy that this importance I¹ᵢ
  demonstrates superior performance within indoor settings ... However, outdoor scans often feature
  additional Gaussians representing background elements such as the sky or distant objects." (p. 20)
  and "We posit that the design of importance metrics is case-dependent and hand-crafted. Thus,
  this part is presented as an experimental trick in the appendix." (p. 21)
- "In indoor scenes, both our Mini-Splatting-D and Mini-Splatting learn denser Gaussian
  distributions" (p. 12).
- T&T: "Our algorithms exhibit a decreased PSNR on the Tanks&Temples dataset. This can be
  attributed to large sky areas in the scans (i.e., train) within Tanks&Temples, which our
  depth-based strategy does not accurately model." (p. 12)
- Per-scene budgets: none set.

### 9. Main quantitative results
Table 1 (p. 11), columns SSIM / PSNR / LPIPS / Num (millions), M360 ‖ T&T ‖ DB, as reported.
"3DGS* indicates the retrained model from the official implementation."
- 3DGS [17]: 0.815 / 27.21 / 0.214 / 3.36 ‖ 0.841 / 23.14 / 0.183 / 1.78 ‖ 0.903 / 29.41 / 0.243 / 2.98
- 3DGS [17]*: 0.815 / 27.47 / 0.216 / 3.35 ‖ 0.848 / 23.66 / 0.176 / 1.84 ‖ 0.904 / 29.54 / 0.244 / 2.82
- Zip-NeRF: 0.828 / 28.54 / 0.189 / - ‖ - ‖ -
- mip-NeRF 360: 0.792 / 27.69 / 0.237 ‖ 0.759 / 22.22 / 0.257 ‖ 0.901 / 29.40 / 0.245
- Mini-Splatting-D: 0.831 / 27.51 / 0.176 / 4.69 ‖ 0.853 / 23.23 / 0.140 / 4.28 ‖ 0.906 / 29.88 / 0.211 / 4.63
- Mini-Splatting: 0.822 / 27.34 / 0.217 / 0.49 ‖ 0.835 / 23.18 / 0.202 / 0.20 ‖ 0.908 / 29.98 / 0.253 / 0.35
Table 3 (p. 14, M360 densification ablation, SSIM / PSNR / LPIPS / Num): Baseline 0.815 / 27.47 /
0.216 / 3.35, + Blur Split 0.819 / 27.47 / 0.195 / 3.74, + Depth Reinit 0.832 / 27.54 / 0.175 / 4.32.
Table 5 (p. 17, sparse vs dense init, M360): 3DGS 0.815 / 27.47 / 0.216 / 3.35 vs 0.831 / 27.78 /
0.180 / 4.40. Mini-Splatting-D 0.832 / 27.54 / 0.175 / 4.69 vs 0.838 / 27.76 / 0.171 / 4.71.
Mini-Splatting 0.822 / 27.22 / 0.218 / 0.48 vs 0.825 / 27.31 / 0.215 / 0.48. Per-scene values in
Table 6 (p. 23). No MB in tables.

### 10. Densification details
3DGS gradient-thresholded split/clone is kept unchanged and augmented by blur split (screen-space
max-contribution area criterion) and periodic depth reinitialization from mid-point depths. No
budget modifies densification. The count is set afterwards by simplification.

### 11. Stated limitations and future work (verbatim)
- p. 21 (Appendix G): "Our method employs a depth-based strategy to reinitialize Gaussian models.
  This strategy fails in areas without a certain depth value, such as the sky in train, as shown in
  Fig. 15 (a). This issue could potentially be addressed by considering multiview consistency and
  background removal. Moreover, the number of Gaussians in our Mini-Splatting is manually controlled
  by the sampling ratio. However, finding the minimal number of Gaussians while maintaining
  high-quality rendering remains a challenge. As illustrated in Fig. 15 (b), a high sampling ratio
  leads to distortion in the background. This problem may be alleviated by considering uncertainty
  for each image, which we leave for future study."
- p. 21 (Appendix E): "We posit that the design of importance metrics is case-dependent and
  hand-crafted. Thus, this part is presented as an experimental trick in the appendix."

### 12. Sentences on size, byte, bitrate, memory or FPS targets, rate control, or edge deployment (verbatim)
- p. 10: "Mini-Splatting-C. Our Mini-Splatting-C is designed for scenarios requiring storage
  compression."
- p. 13: "As depicted in Fig. 9, we provide rate-distribution curves comparing our Mini-Splatting-C
  with two compression-oriented methods [20,27] applied to 3DGS."
- p. 13: "Additionally, we successfully train and evaluate our Mini-Splatting* on a low-cost
  graphics card (a GTX 1060 6G GPU). This highlights the potential for on-device neural rendering
  and reconstruction on consumer-grade machines."
- p. 21: "Moreover, the number of Gaussians in our Mini-Splatting is manually controlled by the
  sampling ratio."
- Reading: no byte, bitrate, memory or FPS target during training. Storage size and FPS are
  post-hoc outcomes only.

### 13. Code
"Code is available." (abstract, p. 1). No repository URL is printed anywhere in the PDF. Licence:
not stated.

---

## C. GaussianSpa

### 1. Citation
- Title: GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and
  High-Quality 3D Gaussian Splatting
- First author: Yangming Zhang ("leading co-first author") with Wenqi Jia, University of Texas at
  Arlington. Wei Niu (University of Georgia), Miao Yin (corresponding).
- Venue as stated in the PDF: not stated. The PDF is `arXiv:2411.06019v3 [cs.CV] 10 Apr 2025` in
  CVPR layout with a "Supplementary Material" section. The file name says CVPR 2025.
- arXiv id: 2411.06019

### 2. Base representation and renderer
Vanilla 3DGS in PyTorch, "under the same environment specified in the original 3DGS [22]" (p. 6 to
7). The starting point is not vanilla 3DGS: "We use the same starting checkpoint files as
Mini-Splatting [17] before our 'optimizing-sparsifying' process to ensure fair comparisons." (p. 7),
i.e. the Mini-Splatting-densified model at 15K (blur split and depth reinitialization included).

### 3. Budget handle
- User supplies a target number of Gaussians κ: constraint N(G) ≤ κ (Eq. 5, p. 4), rewritten as
  ‖a‖₀ ≤ κ on the opacity vector a (Eq. 6, p. 5). Alg. 1 input: "target number of Gaussians κ".
- TARGET as an upper bound: the projection "keeps top-κ elements and sets the rest to zeros" (p. 6),
  so after removing the "zero" Gaussians at 25K the count is at most κ.
- Achieved equals requested: not stated. κ values are never printed. Reported counts (Table 1,
  `#G/M`, as reported): 0.547 on M360, 0.269 on T&T, 0.256 and 0.335 (two settings) on DB. Per-scene
  counts vary (Table 3, 4, 5), which is consistent with a per-scene κ but the paper does not say
  how κ was chosen.

### 4. When and how it binds
- Formulation (p. 4 to 6): min L s.t. N(G) ≤ κ (Eq. 5) → min_{a,Θ} L(a, Θ) s.t. ‖a‖₀ ≤ κ (Eq. 6),
  with Θ all non-opacity parameters. Indicator h(a) = 0 if ‖a‖₀ ≤ κ, +∞ otherwise (Eq. 7).
  Auxiliary z with a = z (Eq. 9). Augmented Lagrangian (Eq. 10):

  L(a, z, Θ, λ; δ) = L(a, Θ) + h(z) + (δ/2) ‖a − z + λ‖² + (δ/2) ‖λ‖²

- Alternation (Alg. 1, p. 5, and Fig. 3, p. 3):
  - Optimizing step (Eq. 11): min_{a,Θ} L(a, Θ) + (δ/2) ‖a − z + λ‖², solved by gradient descent
    with ∂L/∂a = ∂L(a,Θ)/∂a + δ(a − z + λ) and ∂L/∂Θ = ∂L(a,Θ)/∂Θ (Eq. 12, 13), updates
    a ← a − η ∂L/∂a, Θ ← Θ − η ∂L/∂Θ (Eq. 14).
  - Sparsifying step (Eq. 15, 16): z ← prox_h(a + λ), the projection onto {‖z‖₀ ≤ κ}: keep the
    top-κ entries of (a + λ), zero the rest. Any importance criterion can replace the plain opacity
    ranking here: "Existing importance criteria can be applied to project the auxiliary variable z
    onto the sparse space in the 'sparsifying' step" (p. 8), demonstrated with LightGaussian's
    criterion.
  - Multiplier update (Eq. 17): λ ← λ + a − z.
  - Loop while ‖a − z‖₂ > ϵ and t ≤ T.
- Schedule: "GaussianSpa starts 'optimizing-sparsifying' at iteration 15K and removes 'zero'
  Gaussians at iteration 25K." (p. 7), then "a light tuning" (p. 6) to the end. Fig. 5 (p. 5, Room)
  shows the opacity histogram splitting into a "zero" cluster and the rest between 16K and 28K.
  Sparsifying interval and δ: only the ranges tested in the supplementary loss curves (Fig. 9,
  p. 11) are given, δ ∈ {0.0001, 0.0003, 0.0005, 0.0009, 0.0013, 0.0017} and interval ∈ {30, 40,
  50, 60, 70, 80} iterations (as reported), described as "similar convergence rates". The values
  used for the main results are not stated.
- Why gradual: "Instead of permanently removing a certain number of Gaussians, GaussianSpa
  incorporates the 'optimizing-sparsifying' algorithm into the training process, gradually imposing
  a substantial sparse property onto the trained Gaussians." (p. 2)

### 5. Meaning of "size" or "count" in tables
Table 1 column `#G/M` "represents number of million Gaussians" (p. 6). Storage in MB only in the
supplementary Table 2 (p. 11, M360, PSNR / SSIM / LPIPS / Storage, as reported): EfficientGS 27.38 /
0.817 / 0.216 / 98 MB, LightGaussian 27.28 / 0.805 / 0.243 / 42 MB, GaussianSpa 27.85 / 0.825 /
0.214 / 25 MB, with the caption "GaussianSpa's storage cost is reported based on the add-on
compression methods (i.e., SH distillation and vector quantization) from LightGaussian [16]." The
MB figure is post-hoc, not a training target.

### 6. Runtime cost reported
Hardware: "Our experimental server has two AMD EPYC 9254 CPUs and eight NVIDIA GTX 6000 Ada GPUs."
(p. 7, as printed). Training time: not stated. Render FPS: not stated. GPU memory: not stated. Cost
claim without a number: "integrated into the 3DGS training with negligible costs" (p. 2).

### 7. Evidence on training-time budget versus post-hoc pruning
This paper has the most direct matched-count comparisons of the four, all between in-training
gradual sparsification and one-shot pruning or sampling applied from the same 15K checkpoint.
- Fig. 2 (p. 2, scene not stated, "Gaussians are removed by 85% at iteration 25K", approx. plot
  reading): Original 3DGS ≈ 21.4 dB throughout. Opacity-based one-shot pruning drops to ≈ 20.4 at
  removal and recovers only to ≈ 20.9 by 35K, annotated "−0.5dB". Importance-based drops to ≈ 20.75
  and recovers to ≈ 21.25, annotated "−0.2dB". GaussianSpa stays at ≈ 21.3 to 21.35 with no drop.
  Text: "the sudden one-shot removal may cause permanent loss of Gaussians crucial to visual
  synthesis, making it challenging to recover the original performance after even long-term
  training, as shown in Figure 2." (p. 2). Matched count by construction (same 85 % removal).
- Table 1 (p. 6), rows at roughly matched counts (PSNR / SSIM / LPIPS / #G/M, M360 ‖ T&T ‖ DB, as
  reported):
  - Mini-Splatting [17]: 27.40 / 0.821 / 0.219 / 0.559 ‖ 23.45 / 0.841 / 0.186 / 0.319 ‖ 30.05 / 0.909 / 0.254 / 0.397
  - Taming 3DGS [31]: 27.31 / 0.801 / 0.252 / 0.630 ‖ 23.95 / 0.837 / 0.201 / 0.290 ‖ 29.82 / 0.904 / 0.260 / 0.270
  - GaussianSpa: 27.85 / 0.825 / 0.214 / 0.547 ‖ 23.98 / 0.852 / 0.180 / 0.269 ‖ 30.33 / 0.912 / 0.254 / 0.256
  - GaussianSpa (second DB row, raw extraction "30.37 0.914 0.249 0.335"): DB 30.37 / 0.914 / 0.249 / 0.335
  "Mini-Splatting [17] results are replicated using official code." Counts are within ±0.1 M of
  each other on M360 and within ±0.05 M on T&T, so this is close to matched count.
- Per-scene matched-count rows versus Mini-Splatting (Table 3, p. 11, M360, PSNR / SSIM / LPIPS /
  #G (M), as reported, 3DGS numbers "reported from [19]"):
  - Bicycle: 3DGS 25.13 / 0.750 / 0.240 / 5.310, Mini-Splatting 25.21 / 0.760 / 0.247 / 0.646, GaussianSpa 25.44 / 0.769 / 0.246 / 0.656
  - Bonsai: 32.19 / 0.950 / 0.180 / 1.250, 31.73 / 0.945 / 0.180 / 0.360, 32.40 / 0.947 / 0.174 / 0.372
  - Counter: 29.11 / 0.910 / 0.180 / 1.170, 28.53 / 0.911 / 0.184 / 0.408, 29.23 / 0.919 / 0.176 / 0.392
  - Flowers: 21.37 / 0.590 / 0.360 / 3.470, 21.42 / 0.616 / 0.336 / 0.670, 21.75 / 0.610 / 0.329 / 0.674
  - Garden: 27.32 / 0.860 / 0.120 / 5.690, 26.99 / 0.842 / 0.156 / 0.738, 27.26 / 0.848 / 0.151 / 0.728
  - Kitchen: 31.53 / 0.930 / 0.120 / 1.770, 31.24 / 0.929 / 0.122 / 0.438, 32.03 / 0.934 / 0.117 / 0.423
  - Room: 31.59 / 0.920 / 0.200 / 1.500, 31.44 / 0.929 / 0.193 / 0.394, 32.04 / 0.933 / 0.188 / 0.355
  - Stump: 26.73 / 0.770 / 0.240 / 4.420, 27.35 / 0.803 / 0.219 / 0.717, 27.56 / 0.808 / 0.218 / 0.690
  - Treehill: 22.61 / 0.640 / 0.350 / 3.420, 22.69 / 0.652 / 0.332 / 0.663, 22.94 / 0.660 / 0.329 / 0.637
  - Average: 27.45 / 0.810 / 0.220 / 3.110, 27.40 / 0.821 / 0.219 / 0.559, 27.85 / 0.825 / 0.214 / 0.547
  Table 4 (p. 14, T&T): Train 3DGS 21.94 / 0.810 / 0.200 / 1.110, Mini-Splatting 21.78 / 0.805 /
  0.231 / 0.287, GaussianSpa 22.17 / 0.815 / 0.228 / 0.199. Truck 25.31 / 0.880 / 0.150 / 2.540,
  25.13 / 0.878 / 0.141 / 0.352, 25.79 / 0.888 / 0.132 / 0.338.
  Table 5 (p. 14, DB): Drjohnson 3DGS 28.77 / 0.900 / 0.250 / 3.260, Mini-Splatting 29.37 / 0.904 /
  0.261 / 0.377, GaussianSpa 29.89 / 0.913 / 0.243 / 0.450 and 29.82 / 0.909 / 0.254 / 0.293.
  Playroom 30.07 / 0.900 / 0.250 / 2.290, 30.72 / 0.914 / 0.248 / 0.417, 30.84 / 0.916 / 0.254 /
  0.219 (printed twice).
- Curves at matched count: Fig. 7 (p. 7, Kitchen and Room, PSNR vs reduction rate 95 % to 50 %,
  approx. plot reading): Kitchen GaussianSpa ≈ 29.9 → 32.3, Mini-Splatting ≈ 29.6 → 31.6, gap ≈ 0.5
  to 0.7 dB at every rate. Room GaussianSpa ≈ 30.9 → 32.1, Mini-Splatting ≈ 30.5 → 31.6. Fig. 14
  (p. 14): 13 scene panels GaussianSpa vs Mini-Splatting at #G from ≈ 0.05 M to ≈ 1.0 M, GaussianSpa
  above on every panel and every count (approx. plot reading, gaps ≈ 0.2 to 0.9 dB), plus Kitchen-L
  and Playroom-L panels versus LightGaussian with LightGaussian's criterion used inside the
  projection. Paper's statements: "GaussianSpa outperforms the state-of-the-art method,
  Mini-Splatting [17], by an average of 0.5 dB PSNR improvement across multiple Gaussian reduction
  rates." (p. 7) and "GaussianSpa consistently outperforms LightGaussian [16] with an average of 0.4
  dB improvement for the Playroom and Kitchen scenes." (p. 8) and "with the same number of
  Gaussians, our GaussianSpa shows superior rendering outcomes." (p. 11)
- Caveat for the hypothesis: the budget κ enters at 15K, from a checkpoint densified without any
  budget. GaussianSpa's evidence is for "gradual, optimisation-integrated sparsification beats
  one-shot removal at the same count", not for "budget known from iteration 0".

### 8. Scene dependence
- Per-scene final counts vary under the paper's settings (Table 3: 0.355 M on Room to 0.728 M on
  Garden, as reported), and DB is reported at two operating points (0.335 M and 0.256 M average).
  How κ is chosen per scene is not stated.
- Qualitative statement about adaptive allocation inside a scene: "GaussianSpa uses fewer but
  larger Gaussians to represent the blue sky and uses dense Gaussians to render complex carpet
  textures" (p. 7), "GaussianSpa adaptively reduce the Gaussians in the low-frequency areas and
  preserve more Gaussians in high-frequency areas" (p. 7).

### 9. Main quantitative results
Table 1 (p. 6), PSNR / SSIM / LPIPS / #G/M, M360 ‖ T&T ‖ DB, as reported. "3DGS results are
reported from [19]."
- 3DGS [22]: 27.45 / 0.810 / 0.220 / 3.110 ‖ 23.63 / 0.850 / 0.180 / 1.830 ‖ 29.42 / 0.900 / 0.250 / 2.780
- CompactGaussian [26]: 27.08 / 0.798 / 0.247 / 1.388 ‖ 23.32 / 0.831 / 0.201 / 0.836 ‖ 29.79 / 0.901 / 0.258 / 1.060
- LP-3DGS-R [51]: 27.47 / 0.812 / 0.227 / 1.959 ‖ 23.60 / 0.842 / 0.188 / 1.244 ‖ -
- LP-3DGS-M [51]: 27.12 / 0.805 / 0.239 / 1.866 ‖ 23.41 / 0.834 / 0.198 / 1.116 ‖ -
- EAGLES [19]: 27.23 / 0.809 / 0.238 / 1.330 ‖ 23.37 / 0.840 / 0.200 / 0.650 ‖ 29.86 / 0.910 / 0.250 / 1.190
- Mini-Splatting [17]: 27.40 / 0.821 / 0.219 / 0.559 ‖ 23.45 / 0.841 / 0.186 / 0.319 ‖ 30.05 / 0.909 / 0.254 / 0.397
- Taming 3DGS [31]: 27.31 / 0.801 / 0.252 / 0.630 ‖ 23.95 / 0.837 / 0.201 / 0.290 ‖ 29.82 / 0.904 / 0.260 / 0.270
- CompGS [35]: 27.12 / 0.806 / 0.240 / 0.845 ‖ 23.44 / 0.838 / 0.198 / 0.520 ‖ 29.90 / 0.907 / 0.251 / 0.550
- GaussianSpa: 27.85 / 0.825 / 0.214 / 0.547 ‖ 23.98 / 0.852 / 0.180 / 0.269 ‖ 30.33 / 0.912 / 0.254 / 0.256, plus DB 30.37 / 0.914 / 0.249 / 0.335
Strongest baselines: Mini-Splatting and Taming 3DGS (rows above). MB only in supplementary Table 2
(field 5).

### 10. Densification details
GaussianSpa adds no densification of its own. Densification up to 15K is inherited from the
Mini-Splatting checkpoint (3DGS split/clone plus blur split plus depth reinitialization). The budget
acts only through opacity sparsity after 15K, so it does not modify densification.

### 11. Stated limitations and future work
Not stated. The paper has no limitations or future-work paragraph. The closest statement is the
generality remark on p. 8: "The proposed GaussianSpa is a general Gaussian simplification framework.
Existing importance criteria can be applied to project the auxiliary variable z onto the sparse space
in the 'sparsifying' step".

### 12. Sentences on size, byte, bitrate, memory or FPS targets, rate control, or edge deployment (verbatim)
- p. 1: "However, 3DGS suffers from substantial memory requirements to store the large amount of
  Gaussians, hindering its efficiency and practicality."
- p. 2: "In densely sampled scenes, the sheer volume of Gaussians leads to memory usage that exceeds
  the capacity of typical hardware, making it challenging to handle higher-resolution scenes and
  limiting its applicability in resource-constrained environments."
- p. 2: "In the proposed framework, we formulate 3DGS simplification as a constrained optimization
  problem under a target number of Gaussians."
- p. 2: "Hence, our GaussianSpa can simultaneously enjoy maximum information preservation from the
  original Gaussians and a desired number of reduced Gaussians, providing compact 3DGS models with
  high-quality rendering."
- p. 8: "While GaussianSpa focuses on simplifying and optimizing the number of Gaussians for a
  sparse 3DGS representation, existing compression techniques such as SH optimization and vector
  quantization can be applied as add-ons to the simplified 3DGS model. Table 2 (in Supplementary
  Material) shows the final storage cost for the Mip-NeRF 360 dataset by incorporating the SH
  distillation and vector quantization methods proposed by LightGaussian [16]."
- p. 11 (Table 2 caption): "GaussianSpa's storage cost is reported based on the add-on compression
  methods (i.e., SH distillation and vector quantization) from LightGaussian [16]."
- Reading: the target is a count κ. Bytes appear only as a post-hoc outcome with borrowed
  compression. No FPS, bitrate or memory target during training.

### 13. Code
"Our project page is available at https://noodle-lab.github.io/gaussianspa." (p. 1). Licence: not
stated.

---

## D. Smart Target Point Control (TPC)

### 1. Citation
- Title: Smart Target Point Control for Gaussian Splatting Methods
- First author: Pratik Bisht (with Andreas Kolb), Computer Graphics, University of Siegen
- Venue as stated in the PDF: "A PREPRINT - MAY 18, 2026", `arXiv:2605.16158v1 [cs.GR] 15 May 2026`
- arXiv id: 2605.16158

### 2. Base representation and renderer
Two pipelines treated as black boxes: 3DGS and 2DGS ("We treat the standard splatting pipeline
(rendering, photometric loss, densification, pruning, opacity reset) as a black box", p. 3). Their
own rasterizers. No implementation detail beyond that.

### 3. Budget handle
- User supplies K, "the final desired population at t_stop" (p. 4), plus implicitly the trajectory
  shape. TARGET trajectory N*(t) with terminal value K, reached approximately: "guides the primitive
  population size to meet N(t_stop) ≈ K" (p. 3), "reach the approximate desired budget at the end of
  the densification window" (p. 2).
- Achieved equals requested: approximately, with a deadband of about 1 % of N*(t) and a quota floor
  of about K/100 (p. 4). The achieved counts are not printed. Tables 2 and 3 state only the target,
  "the same target point average budget of 0.785M points" (M360) and "0.0912M points"
  (NeRF-Synthetic). Fig. 1b (p. 4, Lego): actual count tracks N*(t) to the 140 000 target by about
  15K iterations (approx. plot reading).

### 4. When and how it binds
- Target trajectory (Eq. 3 to 5, p. 4): N*(t) = N₀ + s(x)(K − N₀), x = (t − t_start)/(t_stop −
  t_start) ∈ [0, 1], s(x) = 1 − (1 − x)² = 2x − x² ("quadratic fast-start easing"), piecewise N₀ for
  t ≤ t_start, K for t ≥ t_stop. "s(0.5) = 0.75" (as reported). t_start = densify_from_iter,
  t_stop = densify_until_iter "typically 15k iterations", cadence c = densification_interval
  "typically 100 iterations" (p. 3, 6).
- Controller law (Sec. 3.3 to 3.5, p. 4 to 5), evaluated only at densify/prune events:
  - gap g(t) = N*(t) − N(t) (Eq. 6)
  - remaining actuations A(t) = 1 + floor((t_stop − t)/c) (Eq. 7, floor brackets lost in extraction)
  - quota q(t) = round(g(t)/A(t)) (Eq. 8), with deadband |g(t)| < δ(t), δ(t) ∝ N*(t) "(e.g., 1%)"
    lower bounded by a constant, and quota floor |q(t)| < q_min, q_min ∝ K "(e.g., K/100)"
  - observed actuation ΔN(t) = N(t) − N(t − 1) (Eq. 9)
  - multiplicative bounded update τ ← τ exp(Δ), |Δ| ≤ Δ_max (Eq. 10), "e.g., Δ_max = 0.12
    corresponds to at most ≈12.7% change", then clamped to fixed multiples of the defaults τ_den,0
    and τ_prune,0 (Eq. 2)
  - under-target (g > δ): τ_prune,eff ← τ_prune,min, e_u = q − ΔN, Δ_den = −β_den e_u,
    τ_den,eff ← τ_den exp(clamp(Δ_den, −Δ_max, Δ_max)) (Eq. 11)
  - over-target (g < −δ): τ_den,eff ← τ_den,max, e_o = (−q) − (−ΔN), Δ_pru = +β_pru e_o,
    τ_prune,eff ← τ_prune exp(clamp(Δ_pru, −Δ_max, Δ_max)) (Eq. 12)
  - deadband: thresholds unchanged
  - prune lockout after each opacity reset: τ_prune,eff ← τ_prune,min for t ∈ [t_reset, t_reset +
    T_lock) (Eq. 13)
- What it moves: only the densification gradient threshold τ_den and the opacity-culling threshold
  τ_prune. Cadence, window start/stop and opacity-reset schedule are untouched (p. 3).
- Values of β_den, β_pru, T_lock, the clamp multiples and the actual Δ_max used: not stated beyond
  the "e.g." examples.
- Target reached: at t_stop, the end of the densification window, 15K of 30K iterations.

### 5. Meaning of "size" or "count" in tables
`Pts (M)` is the number of primitives in millions (Table 1). Tables 2 and 3 print no count column,
only the target in the caption. No MB anywhere.

### 6. Runtime cost reported
Hardware: "All the experiments are run on a single RTX 3090 (24GB) with a 24-core CPU and 64GB RAM."
(p. 6). Training time: not stated. FPS: not stated. GPU memory: not stated.

### 7. Evidence on training-time budget versus post-hoc pruning
No post-hoc pruning baseline. The matched-count comparison is trajectory tracking versus an abrupt
hard cap during training, both at the same target.
- Table 1 (p. 6, unconstrained, MS-SSIM / LPIPS / L1 / PSNR / Pts (M), M360 ‖ NeRF-Synthetic, as
  reported): 2DGS 0.894 / 0.074 / 0.021 / 30.01 / 0.833 ‖ 0.965 / 0.030 / 0.006 / 32.92 / 0.096.
  3DGS 0.903 / 0.062 / 0.020 / 30.47 / 1.582 ‖ 0.969 / 0.023 / 0.006 / 33.88 / 0.288.
- Table 2 (p. 6, M360, target 0.785 M, MS-SSIM / LPIPS / L1 / PSNR, as reported):
  - Hard cutoff: 2DGS 0.892 / 0.077 / 0.022 / 29.96, 3DGS 0.897 / 0.073 / 0.021 / 30.33
  - TPC: 2DGS 0.893 / 0.076 / 0.0215 / 30.054, 3DGS 0.901 / 0.068 / 0.0205 / 30.447
- Table 3 (p. 7, NeRF-Synthetic, target 0.0912 M):
  - Hard cutoff: 2DGS 0.949 / 0.056 / 0.008 / 30.84, 3DGS 0.952 / 0.054 / 0.008 / 31.32
  - TPC: 2DGS 0.966 / 0.030 / 0.006 / 32.94, 3DGS 0.967 / 0.027 / 0.006 / 33.50
- Matched count: yes by target (achieved counts not printed). Notable: TPC 3DGS at 0.785 M
  (30.447) is within 0.03 dB of unconstrained 3DGS at 1.582 M (30.47), and TPC 2DGS at 0.0912 M
  (32.94) matches unconstrained 2DGS at 0.096 M (32.92).
- Mechanism claim: "hitting the cap early can freeze the primitive distribution before
  under-reconstructed regions have received their intended share of point churn, whereas
  over-reconstructed regions may already have consumed the available budget." (p. 7)
- The M360 protocol uses 7 scenes and an unspecified resolution, so its absolute PSNR (≈ 30 dB) is
  not comparable to the ≈ 27.4 dB M360 numbers in the other three papers.

### 8. Scene dependence
- "the final number of primitives is not fixed: it emerges from the interaction between scene
  content, view sampling, and the chosen densify/prune hyper-parameters." (p. 1)
- The budget is a per-dataset "target point average budget" (0.785 M on M360, 0.0912 M on
  NeRF-Synthetic). Whether K differs per scene is not stated. Per-scene budgets: not listed.
- Fig. 1 uses K = 140 000 on Lego (as printed in the legend "Max points (140000)").

### 9. Main quantitative results
Only M360 (7 scenes: Bicycle, Bonsai, Counter, Garden, Kitchen, Room, Stump) and NeRF-Synthetic (8
scenes). No Tanks and Temples, no Deep Blending. Rows are in field 7 (Tables 1 to 3). Metrics are
MS-SSIM, LPIPS, L1, PSNR. No count in MB.

### 10. Densification details
Standard ADC of 3DGS and 2DGS: gradient-driven clone/split (τ_den), opacity cull (τ_prune), periodic
opacity resets. The budget modifies only the two thresholds via the controller, keeping cadence,
window and reset schedule.

### 11. Stated limitations and future work (verbatim)
- p. 8: "In future work, we plan to explore alternative target trajectories and controller designs
  that further reduce run-to-run variance while keeping the underlying point-churn heuristics
  intact."
- p. 2 (scope statement): "Our goal is not to improve reconstruction quality directly, but to
  provide a fairer evaluation protocol: methods can be compared under matched capacity while
  preserving the original point-churn behavior of classical splatting pipelines."
- No explicit limitations paragraph.

### 12. Sentences on size, byte, bitrate, memory or FPS targets, rate control, or edge deployment (verbatim)
- p. 1: "Standard Gaussian splatting methods rely on heuristic densification and pruning to
  adaptively allocate primitives during training, and the resulting Gaussian count strongly
  influences both reconstruction quality and runtime."
- p. 1: "A common and naive workaround for this is hard-stopping or budgeting densification/pruning
  once a target count is reached, which biases training because different methods hit the cap at
  different times, yielding non-uniform densify/prune exposure across views and uneven point
  distributions."
- p. 1: "Some impose a hard cap, e.g., explicitly capping the maximum number of primitives and
  effectively preventing further growth once the limit is reached [6] [7]. Other approaches enforce
  a budgeted densification policy [8]: densification continues, but when the budget is exceeded,
  only the top-ranked candidates are kept and the rest are discarded [9]. However, often these
  strategies are motivated primarily by better primitive placement or memory constraints rather than
  evaluation and comparison fairness."
- p. 2: "More recent methods explicitly incorporate primitive budgeting or memory constraints into
  training, in order to control maximum memory growth from densification and avoid uncontrolled
  early expansion [8] [9]. These methods demonstrate that controlling primitive growth is
  practically important, but their objectives are typically deployment-oriented (e.g.,
  memory-bounded training or compression) rather than providing a general fair comparison protocol
  that preserves a fixed densification/pruning window and cadence across diverse GS variants."
- p. 2: "Community efforts such as standardized NeRF evaluation frameworks highlight how
  inconsistent settings (e.g., different training-time budgets, hyperparameter sweeps, or data
  processing) can obscure scientific conclusions [11]." (here "training-time budgets" means
  wall-clock time, not bytes)
- p. 3: "In contrast to approaches that redesign ADC, introduce new candidate scoring signals, or
  enforce budgets via abrupt stopping or memory-centric schedules, our goal is purely evaluation
  fairness."
- p. 8: "These results suggest that fair benchmarking for Gaussian splatting should treat the
  primitive budget as a controlled variable rather than an emergent outcome or a late-stage cap."
- Reading: count target only. [8] is Taming 3DGS and [9] is Bulò et al., "Revising densification in
  gaussian splatting" (ECCV 2024), which this paper characterises as a budgeted densification
  policy that keeps only top-ranked candidates when the budget is exceeded. No byte or FPS target.

### 13. Code
Not stated. No repository URL in the PDF. Licence: not stated.

---

## Cross-paper synthesis

### Classification

| paper | handle | mechanism | target or knob | achieved = requested |
|---|---|---|---|---|
| Taming 3DGS | final count B (set as SfM multiplier or as the 3DGS count) | parabolic count schedule, per-event allowance T − |G|, score-weighted random sampling of clone/split candidates every 500 it, purely constructive | target, exact | yes, Table 1 #G = Peak #G, and 1.84 / 3.31 / 2.81 M matched to 3DGS exactly |
| Mini-Splatting | sampling ratio | densify to 15K (blur split + depth reinit), then intersection preserving + importance-weighted sampling at 15K, prune at 20K | knob | not applicable, count is an outcome (0.49 / 0.20 / 0.35 M) |
| GaussianSpa | count κ | ADMM-style alternation from 15K: penalised gradient step, top-κ projection of opacity, multiplier update, zero-removal at 25K | target (upper bound via projection) | not stated, κ never printed, per-scene counts vary |
| Smart TPC | terminal count K plus quadratic trajectory N*(t) | quota governor moving τ_den and τ_prune multiplicatively in log space at the native cadence, deadband 1 %, quota floor K/100, prune lockout after resets | target, approximate | approximately, achieved counts not printed |

### Matched-count evidence, assembled

1. Trajectory tracking versus abrupt cap at the same count (TPC Tables 2 and 3): +0.12 dB PSNR
   and −0.005 LPIPS on M360 for 3DGS, +2.18 dB on NeRF-Synthetic for 3DGS, +2.10 dB for 2DGS
   (differences computed from the reported rows). Supports "let the budget shape the whole
   densification window" over "cap when reached".
2. Budgeted constructive densification versus 3DGS heuristic at the identical count (Taming Table 1
   bottom): +0.39 / +0.33 / +0.50 dB PSNR on T&T / M360 / DB (computed from reported rows). The
   fixed-count ablation (Taming Table 2) attributes +0.28 dB (Truck) and +0.45 dB (Train) to the
   score-based sampling itself.
3. Gradual in-training sparsification versus one-shot removal at the same count (GaussianSpa):
   Fig. 2 one-shot opacity pruning −0.5 dB and importance pruning −0.2 dB after 10K iterations of
   recovery, GaussianSpa without a visible drop. Versus Mini-Splatting at the same #G: "an average of
   0.5 dB" (p. 7) and Table 3 per scene, every scene above except Garden (27.26 vs 26.99, both
   below 3DGS 27.32). Versus LightGaussian with its own criterion: "an average of 0.4 dB" (p. 8).
4. Geometry-aware sampling versus importance pruning at the same count, mid-training
   (Mini-Splatting Fig. 6, Fig. 10): about half the Gaussians for the same PSNR on M360
   (approx. plot reading), and lower chamfer distance at every ratio.

### Evidence against the strong form of the hypothesis

The hypothesis is "training with the budget known from the start places capacity better than
post-hoc pruning to the same count". The only method here that knows the count from iteration 0 is
Taming 3DGS. In the one table that puts all three on one protocol (GaussianSpa Table 1, counts
within ±0.1 M):

- M360: Taming 27.31 dB at 0.630 M, Mini-Splatting 27.40 at 0.559 M, GaussianSpa 27.85 at 0.547 M
- DB: Taming 29.82 at 0.270 M, Mini-Splatting 30.05 at 0.397 M, GaussianSpa 30.33 at 0.256 M
- T&T: Taming 23.95 at 0.290 M, Mini-Splatting 23.45 at 0.319 M, GaussianSpa 23.98 at 0.269 M

Taming's own Table 1 gives the same ordering versus Mini-Splatting in LPIPS on all three datasets
(0.201 vs 0.181, 0.252 vs 0.217, 0.260 vs 0.244) and in PSNR on DB, and Taming Fig. 1 right places
Mini-Splatting* on or slightly above the Taming curve at 0.55 M on Garden (approx. plot reading).
So the methods that over-densify to 4 to 5 M Gaussians and then sparsify (Mini-Splatting peak
4.32 to 4.51 M per Taming Table 1, GaussianSpa starting from the same checkpoint) match or beat the
constructive budgeted method at equal final count in quality, and Taming's clear wins are training
time (7 to 11 min versus 20 to 30 min) and peak memory (no over-densification, under 10 GB VRAM).
Taming itself concedes the mechanism: "achieving optimal quality still requires a substantial
sample count and meandering exploration as Gaussians move across the scene." (p. 11)

What the literature does support: (a) knowing the budget throughout the densification window and
tracking it smoothly beats an abrupt cap (TPC), (b) sparsifying gradually inside the optimisation
beats one-shot removal followed by fine-tuning (GaussianSpa), (c) reorganising positions beats
pruning positions (Mini-Splatting). None of these shows a from-the-start budget beating an
explore-then-gradually-sparsify schedule at matched count. A BudgetGS method that wants both the
quality of (b) and the resource predictability of Taming would need a budget trajectory that permits
a bounded, scheduled overshoot.

### On the claimed gap

None of the four binds bytes, bitrate, memory in bytes, or FPS during training. All four bind or
tune a primitive count. Two caveats for how the gap is phrased:

- Taming calls the count a "model size" and reports MB only as an outcome (196 MB at 0.8 M, 1 450 MB
  at 6 M on Garden). For an uncompressed float32 ply the bytes-per-Gaussian are constant (my
  computation from Fig. 1: ≈ 242 to 245 B per Gaussian), so a count target is already a byte target
  for the uncompressed format. The gap claim survives only when phrased as a byte target under a
  compressed or variable-rate representation (pruned SH bands, quantised or entropy-coded
  attributes), or as a byte target reached by a rate-adaptive mechanism during training.
- TPC (May 2026) surveys budgeted and memory-constrained training and names only Taming 3DGS and
  Bulò et al. as "deployment-oriented (e.g., memory-bounded training or compression)" budget
  methods. It does not name any byte-target method, which is weak corroboration of the gap as of its
  date, and not proof.
