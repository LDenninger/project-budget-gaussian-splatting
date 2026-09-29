# B2. Rate-distortion 3DGS compression: RDO-Gaussian, ContextGS, CAT-3DGS, FCGS, CompGS

Reading notes for the BudgetGS project. Every page of each PDF was read from a pymupdf text dump.
Page numbers are PDF page indices (page 1 = title page). Every number below is as reported in the
respective paper. Tables were extracted as flattened columns, so a raw extracted row is quoted wherever
the reading is not certain. "not stated" means the PDF does not say it.

Conventions in this file: MB values are the paper's own "size" column. Tuples for rate points are
written as (size MB, PSNR, SSIM, LPIPS) unless a table lists them differently.

---

## 1. RDO-Gaussian

### 1.1 Citation
- Title: End-to-End Rate-Distortion Optimized 3D Gaussian Representation
- First author: Henan Wang (University of Science and Technology of China), with Hanxin Zhu, Tianyu He,
  Runsen Feng, Jiajun Deng, Jiang Bian, Zhibo Chen
- Venue as stated in PDF: not stated (no conference header in the PDF text). The file name says ECCV 2024.
- arXiv id: 2406.01597 (v2, 21 Oct 2024)

### 1.2 Base representation
- Raw 3DGS Gaussians, "We build our method on 3D Gaussian Splatting (3DGS) [16]" (p.4). Each Gaussian
  keeps position, opacity, scale, rotation, DC colour and SH degrees 1 to 3, with per-Gaussian SH degree
  chosen by learned masks.
- Render-time cost: no MLP, no hash grid, no triplane. Decoding is a one-time arithmetic decode of codebook
  indexes plus codebook lookup. After that the model is a plain 3DGS point set. Pruned SH degrees are set
  to zero (Eq. 7, p.7), and Gaussians are re-sorted by their 3-bit SH mask so that only "8 additional
  indexes need to be stored" (p.9). A vanilla rasterizer can render it once the missing SH coefficients
  are zero-filled. Positions are float16, opacity is an 8-bit scalar (p.9).

### 1.3 Budget handle
- Knobs: λGSprune and λSHprune in the total loss (Eq. 13, p.8). The ECVQ λ values are fixed across
  rate points: λ(s) = 32768, λ(r) = λ(DC) = λ(SH1) = λ(SH2) = λ(SH3) = 256 (p.17). "By modifying the
  λSHprune and λGSprune in Eq. 13, we obtain a series of compressed Gaussians with different sizes." (p.10)
- Rate points, real scenes (Tables A.1 to A.3, p.18 to p.19): λGSprune ∈ {0.05, 0.02, 0.01, 0.005, 0.002,
  0.0005}, paired with λSHprune ∈ {0.5, 0.2, 0.1, 0.05, 0.02, 0.005}.
- Rate points, NeRF-Synthetic (Table A.4, p.19): λGSprune ∈ {0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001},
  λSHprune ∈ {0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005}.
- Formulation: "a joint optimization of rate (R) and distortion (D), i.e., R+λD, where λ governs the
  trade-off between them" (p.2).
- Target versus knob: knob only. No number to hit appears anywhere. The paper's phrase "flexible and
  continuous rate control" (p.1, p.2, p.4, p.14) means the λ pair sweeps a continuous RD curve.

### 1.4 When it binds
- Stage 1: plain 3DGS training "up to 15k iterations" (p.10), with standard adaptive density control.
- Stage 2: "Subsequently, we introduce our rate-distortion optimization, incorporating Gaussian pruning,
  adaptive SHs pruning and ECVQ into the optimization pipeline. During this process, parameters generated
  from the aforementioned operations are jointly optimized with classical Gaussian parameters ... until
  the end of training." (p.10). Total iteration count: not stated. Wall-clock training time: not stated.
- Stage 3: post-training packing (Sec. 3.3, p.9): removal of masked Gaussians, re-sorting by SH mask,
  arithmetic coding of indexes with the learned logits, float16 positions, 8-bit opacity.
- Rate term coverage: the entropy-estimated rate L_rate (Eq. 11, p.8) covers only the six ECVQ index
  streams: scale, rotation, DC colour, SH1, SH2, SH3. Position and opacity carry no rate term ("Two
  attributes, position and opacity, are not quantized" by VQ, p.9). Positions are 43.9 % of the high-rate
  file (Table 2, p.14) and are outside the RD objective.
- How pruning is driven: not by the entropy model. A learned scalar ϕᵢ per Gaussian, soft mask
  sigmoid(ϕᵢ), hard mask via STE with threshold ϕthres = 0.1 (Eq. 2 to 4, p.6, p.17). The count pressure
  is L_GSprune = (1/N) Σᵢ ϕᵢ^soft (Eq. 5, p.6), weighted by λGSprune. It is a mask-mean proxy for count,
  not a bit estimate. Masks are dynamic: "The masks are dynamic, meaning that their values are
  continuously optimized until training ends. Once they surpass the threshold, the corresponding
  Gaussians will respawn. Upon completion of training, these Gaussians are finally eliminated" (p.6).
  SH degree pruning uses three masks per Gaussian and L_SHprune weighted by coefficient count per degree
  (Eq. 8, p.7).
- Entropy model: "a discrete unconditional entropy model P" per codebook, p_j = softmax of learned
  logits w (Eq. 9, p.8). Codeword selection is ECVQ, argmin over m of −log p_m / λ + ‖s_i − CB[m]‖²
  (Eq. 10, p.8). Codebook sizes: 8192 for scale, rotation, DC, 4096 for SH1 to SH3 (p.17). Learning rates:
  masks 0.01, SH masks 0.05 real and 0.005 synthetic, codebooks 0.0002, logits 0.002 (p.17).

### 1.5 What "size" means
- On-disk bytes after arithmetic coding. Table 2 (p.14, scene Truck) lists the components: header,
  indexes, codebooks, logits, positions.
  - High rate: header 4 × 10⁻⁵ MB, indexes 7.23 MB (48.7 %), codebooks 0.98 MB (6.6 %), logits 0.12 MB
    (0.8 %), positions 6.53 MB (43.9 %), total 14.86 MB.
  - Low rate: indexes 0.54 MB (36.2 %), codebooks 0.32 MB (21.8 %), logits 0.09 MB (5.8 %), positions
    0.53 MB (36.1 %), total 1.48 MB.
- Header holds SH-mask cluster boundaries plus the opacity quantization step and minimum (p.14). No MLP,
  no grid. Unused codewords and their logits are dropped (p.9).
- Index breakdown (Table A.10, p.22, Truck high rate): scales 1.83, rotations 1.89, base colours 1.84,
  SH1 0.14, SH2 0.20, SH3 0.26, opacities 1.04 MB, total 7.23 MB.

### 1.6 Runtime cost reported
- Render FPS on an NVIDIA A100 (Table A.9, p.22):
  - 3DGS: 141 (Mip-NeRF 360), 176 (Tanks and Temples), 143 (Deep Blending).
  - Ours Rate 1 to Rate 6 (high to low): Mip-NeRF 360 191, 244, 308, 361, 445, 509. Tanks and Temples
    269, 367, 444, 531, 636, 732. Deep Blending 207, 302, 395, 485, 572, 710.
- Encode time, decode time, GPU memory: not stated.

### 1.7 Rate model
- Entropy model type: unconditional categorical per codebook (softmax logits), six codebooks.
- Quantization relaxation: none described. ECVQ makes a hard codeword choice, the paper does not describe
  how gradients pass to the continuous attribute (not stated). Masks use STE.
- Step sizes: no scalar steps except opacity mapped to 8-bit (p.9). Not learned per attribute.
- Gap between entropy estimate and encoded bytes: not stated.

### 1.8 Scene dependence
- Per-scene sizes at the single highest-rate configuration (λGSprune = 0.0005, λSHprune = 0.005),
  Tables A.5 to A.7 (p.19 to p.20): Truck 14.87, Train 9.18, DrJohnson 22.89, Playroom 13.11, Bicycle 43.14,
  Flowers 26.52, Garden 38.75, Stump 36.17, Treehill 28.76, Room 8.11, Counter 7.91, Kitchen 13.01,
  Bonsai 8.77 MB. On Mip-NeRF 360 one λ pair yields 7.91 MB to 43.14 MB, a 5.5× spread.
- Retuning statement: separate λ sets and SH-mask learning rates for synthetic versus real scenes (p.17).
  No per-scene tuning claim.

### 1.9 Main quantitative results
- Mip-NeRF 360, all rate points (Table A.3, p.19), as reported: (1.71, 24.43, 0.683, 0.406),
  (3.54, 25.38, 0.734, 0.343), (6.16, 26.03, 0.764, 0.299), (9.76, 26.50, 0.784, 0.268),
  (15.35, 26.87, 0.796, 0.248), (23.46, 27.05, 0.802, 0.239). Gaussian prune ratios 0.965, 0.922,
  0.862, 0.776, 0.636, 0.437. SH prune ratios 0.999, 0.996, 0.989, 0.970, 0.922, 0.809.
- Tanks and Temples (Table A.1, p.18): (1.32, 22.09, 0.755, 0.318), (2.39, 22.69, 0.793, 0.264),
  (3.74, 22.98, 0.812, 0.233), (5.49, 23.14, 0.823, 0.214), (8.02, 23.28, 0.831, 0.202),
  (12.02, 23.34, 0.835, 0.195).
- Deep Blending (Table A.2, p.18): (1.22, 28.38, 0.872, 0.331), (2.40, 29.07, 0.887, 0.296),
  (4.14, 29.35, 0.895, 0.277), (6.71, 29.48, 0.899, 0.265), (10.96, 29.59, 0.901, 0.256),
  (18.00, 29.63, 0.902, 0.252).
- Baselines in the same tables: only the authors' own 3DGS reimplementation, per scene (Tables A.5 to
  A.7). Examples as reported: Truck 3DGS 609.32 MB / 25.36 dB versus Ours 14.87 MB / 25.04 dB. Bicycle
  3DGS 1335.48 MB / 25.11 dB versus Ours 43.14 MB / 24.88 dB. Room 3DGS 349.48 MB / 31.62 dB versus Ours
  8.11 MB / 31.11 dB. CompGS [24] (Navaneet et al.), Compact-3DGS and LightGaussian appear only as points
  on the RD curves of Fig. 3 (p.11), without tabulated numbers.
- Ablation Table 1 (p.13), Truck: 3DGS 609.3 MB / 25.36, +GS prune 275.2 MB / 25.34, +Ada. SHs prune
  94.1 MB / 25.32, +ECVQ 33.9 MB / 25.06, +Param comp 14.9 MB / 25.04.

### 1.10 Densification and count control
- First 15k iterations: standard 3DGS adaptive density control (clone, split, opacity reset pruning).
- After 15k: count is decided by the learned Gaussian masks under L_GSprune. The entropy rate term has no
  influence on count or on SH degree. λGSprune and λSHprune are swept together (Tables A.1 to A.4).
- Fig. A.1 (p.21): "our method can attain the quality of 3DGS when approximately 50% Gaussians are
  pruned."

### 1.11 Stated limitations and future work (verbatim)
- p.14: "One of the limitations of our work is that representations are trained individually for
  different rates without shared information. This introduces additional storage overhead when the same
  scene is represented by multiple Gaussian representations of different bitrates. In future work, we aim
  to design a variable-rate Gaussian representation that supports multiple bitrates in a single
  representation."

### 1.12 Sentences touching a target, rate control, budget or desired size (verbatim)
- p.1 (abstract): "we formulate the compact 3D Gaussian learning as an end-to-end Rate-Distortion
  Optimization (RDO) problem and propose RDO-Gaussian that can achieve flexible and continuous rate
  control."
- p.2: "we formulate the 3D Gaussian representation learning as a joint optimization of rate (R) and
  distortion (D), i.e., R+λD, where λ governs the trade-off between them."
- p.2: "To realize rate-distortion optimization, we control the rates by introducing dynamic pruning and
  entropy-constrained vector quantization (ECVQ) that optimize the rate and distortion at the same time."
- p.10: "By modifying the λSHprune and λGSprune in Eq. 13, we obtain a series of compressed Gaussians
  with different sizes."
- p.10: "Such an observation indicates that our method can compress Gaussian to a size less than an image
  with adequate scene information, which is important in extremely band-limited circumstances."
- p.14: "Our presented RDO-Gaussian formulates the learning of 3D Gaussian representations as a joint
  rate-distortion optimization process, achieving flexible and continuous rate control."
- No sentence names a byte, MB, bitrate, memory or FPS target, a budget, or a desired size to hit.

### 1.13 Code
- https://github.com/USTC-IMCL/RDO-Gaussian (p.1). Licence: not stated.

---

## 2. ContextGS

### 2.1 Citation
- Title: ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model
- First author: Yufei Wang (Nanyang Technological University), with Zhihao Li, Lanqing Guo, Wenhan Yang,
  Alex C. Kot, Bihan Wen
- Venue as stated in PDF: "Preprint. Under review." (p.1). No conference named. (FCGS's reference list
  cites it as NeurIPS 2024, this PDF does not.)
- arXiv id: 2405.20721 (v1, 31 May 2024)

### 2.2 Base representation
- Scaffold-GS anchors plus MLP. Each anchor has feature f (dimension 50 in the experiments, p.7),
  location x, scaling l and k offsets. Neural Gaussians are predicted per view by an MLP F(f, σ_c, d_c)
  (Eq. 2, p.4).
- Render-time cost: the Scaffold-GS MLP runs per view. Entropy decoding happens once at load: "For
  testing, after we decode anchor features from the bit stream, the rendering is exactly the same with
  Scaffold-GS [20] without introducing overhead." (p.5). The decoded model is a Scaffold-GS anchor set,
  not a vanilla 3DGS point set.

### 2.3 Budget handle
- Knob λe in L = L_scaffold + λe L_entropy + λm L_m (Eq. 10, p.6), with λm = 5e−4 (p.7). "a larger λe
  leads to a smaller model size" (p.12). λe is normalised by the anchor feature dimension as in HAC (p.12).
- Rate points in tables: low-rate λ = 0.004 and high-rate λ = 0.0005 for Mip-NeRF360, Tanks and Temples
  and Deep Blending (Tables 7 to 9, p.13 to p.14). BungeeNeRF high-rate λ = 0.001 (Table 6, p.13).
- A second, different "target": the anchor-level partition uses a target count ratio τ = 0.2 between
  adjacent levels, found by a one-time binary search on the voxel scales κᵢ (p.5 to p.6). This targets a
  level-size ratio for the coding structure, not bytes.
- Target versus knob for size: knob only.

### 2.4 When it binds
- During training. "we use the same training iterations with Scaffold-GS [20] and HAC [5], i.e., 30000
  iterations." (p.7). Whether the entropy term is active from iteration 0 or switched on later: not
  stated. Wall-clock training time: not stated.
- Encoding after training with arithmetic coding, decoding before rendering.

### 2.5 What "size" means
- Storage in MB of the bitstream plus uncompressed parts. Table 4 (p.9, scene rome, BungeeNeRF) lists
  Hyper, Position, Feature, Scaling, Offset, Mask, MLPs, Total. Ours: 0.778, 2.543, 5.808, 1.586, 2.563,
  0.452, 0.316, total 14.06 MB with 52.5K anchors. Scaffold-GS: 7.08 (position), 75.16, 14.18, 70.88,
  2.362, 0.047 (MLPs), total 186.7 MB with 61.9K anchors. Ours with anchor position coding (APC): 1.026,
  1.954, 5.708, 1.603, 2.556, 0.452, 0.320, total 13.62 MB.
- Positions are not entropy coded in the main results: "Since the anchor position only occupies a small
  portion of bitstreams as shown in Table 4, we do not encode anchors into bitstreams in all the
  experiments." (p.9). MLP weights are included.

### 2.6 Runtime cost reported
- Encode 20.40 s, decode 17.85 s (Ours), 41.33 s and 51.58 s with APC, "measured on an RTX3090" (Table 4
  caption, p.9), scene rome.
- Render FPS: no number. "The rendering speed after decompression is the same as or even faster than
  Scaffold-GS [20] when the number of anchors is the same since we use the same data structure." (p.9).
- GPU memory at render time: not stated.

### 2.7 Rate model
- Anchor attributes f, l, O per level k: Gaussian N(μ, σ) convolved with U(−Δ/2, Δ/2), with (μ, σ, Δ)
  predicted by a per-level MLP F^k from the prior ψ (Eq. 7, p.6). Prior: position only at the coarsest
  level, otherwise the decoded feature and scaling of the adjacent coarser-level anchor plus position
  (Eq. 8). A learnable hyperprior zᵢ ∈ R^(50//hc), hc = 4, coded with the non-parametric fully factorized
  density of Ballé et al. (Eq. 9, p.6).
- Quantization relaxation: additive uniform noise ("with quantization noise", p.6). Step Δ is predicted
  per anchor and attribute ("Adaptive quantization" in Fig. 3, p.5), so the step is learned per attribute
  and context.
- Gap between entropy estimate and encoded bytes: not stated.
- Three levels (K = 3), autoregressive over levels only ("a loop of 3 iterations", p.9).

### 2.8 Scene dependence
- Table 9 (p.14), λ = 0.004: bicycle 21.82, bonsai 7.17, counter 6.30, flowers 16.71, garden 18.78,
  kitchen 7.00, room 4.50, stump 14.86, treehill 17.00 MB (average 12.68). A single λ spans 4.50 MB to
  21.82 MB, a 4.8× spread.
- Retuning statement (about the level partition, not λ): "Since different scenes have different initial
  voxel sizes and anchor point distributions, using a fixed set of voxel scaling parameters {κi} leads to
  a suboptimal performance. To avoid finetuning hyper-parameters for each scene, we propose to conduct a
  one-time parameter search after initializing anchors." (p.5)

### 2.9 Main quantitative results (Table 1, p.7, as reported)
- Mip-NeRF360: Ours (low-rate) 27.62 / 0.808 / 0.237 / 12.68 MB. Ours (high-rate) 27.75 / 0.811 / 0.231 /
  18.41 MB. HAC 27.53 / 0.807 / 0.238 / 15.26 MB. Scaffold-GS 27.50 / 0.806 / 0.252 / 253.9 MB. 3DGS
  27.49 / 0.813 / 0.222 / 744.7 MB.
- Tanks and Temples: Ours (low) 24.20 / 0.852 / 0.184 / 7.05. Ours (high) 24.29 / 0.855 / 0.176 / 11.80.
  HAC 24.04 / 0.846 / 0.187 / 8.10. Scaffold-GS 23.96 / 0.853 / 0.177 / 86.50.
- Deep Blending: Ours (low) 30.11 / 0.907 / 0.265 / 3.45. Ours (high) 30.39 / 0.909 / 0.258 / 6.60.
  HAC 29.98 / 0.902 / 0.269 / 4.35. Scaffold-GS 30.21 / 0.906 / 0.254 / 66.00.
- Mip-NeRF360 rate points with numbers: 12.68 MB (λ = 0.004) and 18.41 MB (λ = 0.0005) in Table 1, plus
  the APC variant 11.32 MB / 27.61 dB and 21.07 MB / 27.86 dB in Table 5 (p.12). Fig. 4 (p.7) draws more
  RD points without numbers. Note a discrepancy as reported: Table 9 (p.14) gives the high-rate average as
  21.58 MB / 27.72 dB while Table 1 gives 18.41 MB / 27.75 dB.
- BungeeNeRF ablation (Table 2, p.8): Scaffold-GS 183.0 MB / 26.62, Ours w/o HP w/o CM 18.67 / 26.93,
  Ours 14.00 / 26.90.

### 2.10 Densification and count control
- Scaffold-GS anchor growing, unchanged: "we use the same hyperparameters for anchor growing as
  Scaffold-GS [20] so that the final model has a similar or even smaller number of anchors, leading to
  faster rendering speed." (p.7). Plus the Compact-3DGS masking loss L_m on neural Gaussian offsets with
  λm = 5e−4 (p.6 to p.7).
- The rate term does not enter anchor growing. The mask loss weight is fixed across rate points.

### 2.11 Stated limitations and future work (verbatim)
- p.12 (A.2): "A main and inevitable limitation of the proposed method is that the entropy coding process
  introduces extra computational costs to estimate the entropy of the anchor features during training
  encoding/decoding when saving/loading the 3D scene. For example, it requires extra time to decode the
  data of 3D scenes from the bitstream, making it challenging to start rendering at once when clicking
  the file."
- No future-work paragraph.

### 2.12 Sentences touching a target, rate control, budget or desired size (verbatim)
- p.5: "Instead of directly setting the scale κi, we set a target ratio between level i and i + 1 and
  expect |Vi+1|/|Vi| ≈ τ." (a count-ratio target for the level partition, resolved once by binary search,
  not a size target)
- p.7 (Table 1 caption): "Our methodology showcases two results representing varying size and fidelity
  tradeoffs, achieved through adjustment of λe."
- p.12: "To evaluate the performance among different rate-distortion (RD) tradeoffs, we utilize different
  λe, i.e., a larger λe leads to a smaller model size."
- p.12 (Table 5 caption): "A smaller λe results in a larger size but improved fidelity, and vice versa."
- No sentence names a byte, MB, bitrate, memory or FPS target during training, a budget, or a desired size.

### 2.13 Code
- https://github.com/wyf0912/ContextGS (p.1, "Homepage"). Licence: not stated.

---

## 3. CAT-3DGS

### 3.1 Citation
- Title: CAT-3DGS: A Context-Adaptive Triplane Approach to Rate-Distortion-Optimized 3DGS Compression
- First authors: Yu-Ting Zhan and Cheng-Yuan Ho (equal contribution), National Yang Ming Chiao Tung
  University, with Hebi Yang, Yi-Hsin Chen, Jui Chiu Chiang, Yu-Lun Liu, Wen-Hsiao Peng
- Venue as stated in PDF: "Published as a conference paper at ICLR 2025"
- arXiv id: 2503.00357 (v2, 7 Mar 2025)

### 3.2 Base representation
- ScaffoldGS anchors plus MLP: anchor position x, latent feature f ∈ R⁵⁰, scaling l ∈ R⁶, K offsets
  Oᵢ ∈ R³ (p.4). The MLP decoder F_S produces colour, opacity, rotation, scale of the K Gaussians per view
  (Eq. 1, p.4).
- Render-time cost: the ScaffoldGS MLP F_S runs per view. Triplane, anchor attributes and masks are
  entropy decoded once before rendering (Table 2, p.10). "The rendering of a 2D image proceeds in much the
  same way as Scaffold-GS" (p.5). Not a vanilla 3DGS point set.

### 3.3 Budget handle
- Knob λr in L = L_Scaffold + λr L_rate + λm L_m (Eq. 10, p.7). "The rate parameter λr ranges from 0.002
  to 0.04, and from 0.001 to 0.02 for BungeeNeRF." (p.8). The mask weight is tied to it: λm =
  max(10⁻³, 0.3 · λr) (p.8, p.13). λtri inside L_rate (Eq. 11): value not stated. ϵ = 0.01 (0.0004 for
  BungeeNeRF) (p.8).
- Rate points in tables: Mip-NeRF 360, Tanks and Temples, Deep Blending λr ∈ {0.002, 0.004, 0.01, 0.015,
  0.03, 0.04} (Tables 7, 9, 10, p.17, p.19). BungeeNeRF λr ∈ {0.001, 0.002, 0.003, 0.006, 0.01, 0.02}
  (Table 8, p.18).
- Target versus knob: knob only.

### 3.4 When it binds
- During training, in stages (Appendix A, p.13): "we adopt ScaffoldGS to ensure a stable start of both
  anchor attribute training and anchor spawning. However, to simulate the quantization effect, we
  introduce noise to the attributes of the anchor points starting at iteration 3000. To prevent the
  generation of an excessive number of anchors, we disable anchor growing during iterations 3000 to 4000."
  Then "we begin using the triplane as a hyperprior to learn the distribution of anchors' attributes
  starting from the 10,000th iteration." and "We start triplane coding at iteration 15,000. Specifically,
  we warm up the spatial autoregressive model while freezing the other learnable parameters between
  iterations 15,000 and 16,000."
- The triplane base resolution B "is determined in proportional to the number of anchor points obtained
  after 10k training iterations" (p.8).
- Total iteration count: not stated in the text (Fig. 8 is an image). Wall-clock training time: not stated.

### 3.5 What "size" means
- "The bit rate is the file size of the compressed bitstream obtained by performing entropy encoding."
  (p.9). Contents (p.5, p.14): triplanes, anchor attributes and positions, binary mask, and the network
  weights of F_S, F_ARM, F_tri, F_ch. "The anchors' positions x and the network weights are signaled in
  16-bit and 32-bit floating-point formats, respectively. The binary mask is entropy encoded." (p.5)
- Table 4 (p.15, treehill): high-rate 516.3K anchors, Position 3.10, Feature 7.75, Scaling 2.47, Offsets
  3.19, Masks 0.53, MLPs 0.35, Triplane 0.11, total 17.5 MB, PSNR 23.28. Low-rate 85.1K anchors, 0.51,
  0.33, 0.28, 0.18, 0.06, 0.35, 0.04, total 1.74 MB, PSNR 22.71.

### 3.6 Runtime cost reported
- Table 2 (p.10), "collected on one NVIDIA V100": room (36.5K anchors, B = 64) decoding 11.4 s triplane
  + 2.2 s anchor attributes = 13.6 s total, 127.0 FPS. amsterdam (533.0K anchors, B = 128) 47.4 + 17.0 =
  64.4 s, 83.4 FPS. HAC: room 228.8K anchors, 6.6 s, 103.1 FPS. amsterdam 483.5K anchors, 11.3 s, 77.9 FPS.
- Encode time: not stated. GPU memory: not stated.
- "CAT-3DGS has much higher decoding time than HAC due to the use of autoregressive models." (p.10)

### 3.7 Rate model
- Hyperprior: multi-scale dense triplanes (ch = 72, scales r ∈ {1, 2}) oriented by PCA of anchor
  positions with a contraction function (Eq. 4 to 5, p.6). An MLP F_tri maps the interpolated triplane
  feature to (μ, σ, q) for each anchor attribute, Gaussian with per-attribute step q (Eq. 6, p.6). The
  feature f is coded in M = 4 uneven channel slices (5, 10, 15, 20) with a channel-wise autoregressive
  model F_ch adding (μ_ch, σ_ch) (Eq. 8, p.7).
- Triplanes themselves are coded by a spatial autoregressive model F_ARM (one per plane orientation) with
  a Laplace distribution and a fixed step Q = 1/16, "chosen to make the training more stable when the
  training process transitions from the non-quantization-aware training to the quantization-aware
  training" (p.6).
- Relaxation: additive noise on anchor attributes from iteration 3000 (p.13). Step q is predicted, so
  learned per anchor and attribute. Gap between estimate and encoded bytes: not stated.

### 3.8 Scene dependence
- Table 7 (p.17), λr = 0.002: bicycle 21.42, bonsai 6.06, counter 6.33, flower 18.92, garden 18.64,
  kitchen 6.92, room 4.11, stump 11.23, treehill 17.50 MB (average 12.35). One λr spans 4.11 MB to
  21.42 MB, a 5.2× spread.
- Retuning: different λr range and ϵ for BungeeNeRF (p.8), and B scales with the anchor count (p.8).
  No per-scene tuning statement beyond that.

### 3.9 Main quantitative results (Table 6, p.16, as reported)
- Mip-NeRF360: Ours (high-rate) 27.77 / 0.809 / 0.241 / 12.35. Ours (low-rate) 25.82 / 0.730 / 0.362 /
  1.72. HAC (high) 27.77 / 0.811 / 0.230 / 21.87. HAC (low) 26.11 / 0.759 / 0.312 / 5.96. RDO-Gaussian
  (high) 27.05 / 0.802 / 0.239 / 23.46. CompGS (high) 27.26 / 0.802 / 0.239 / 16.5. ScaffoldGS 27.50 /
  0.806 / 0.252 / 253.9. ContextGS appears only in Fig. 4.
- Tanks and Temples: Ours (high) 24.41 / 0.853 / 0.189 / 6.93. Ours (low) 22.97 / 0.786 / 0.293 / 1.42.
  HAC (high) 24.40 / 0.853 / 0.177 / 11.24. HAC (low) 23.11 / 0.809 / 0.238 / 3.68. RDO-Gaussian (high)
  23.34 / 0.835 / 0.195 / 12.02.
- Deep Blending: Ours (high) 30.29 / 0.909 / 0.269 / 3.56. Ours (low) 28.53 / 0.878 / 0.336 / 0.93.
  HAC (high) 30.34 / 0.906 / 0.258 / 6.35. HAC (low) 28.62 / 0.888 / 0.302 / 2.64. RDO-Gaussian (high)
  29.63 / 0.902 / 0.252 / 18.00.
- All Mip-NeRF 360 rate points (Table 7 AVG rows, p.17): λr 0.002 (12.35, 27.77, 0.809, 0.241),
  0.004 (9.49, 27.66, 0.807, 0.249), 0.01 (5.47, 27.24, 0.793, 0.274), 0.015 (4.10, 26.94, 0.782, 0.293),
  0.03 (2.27, 26.22, 0.750, 0.338), 0.04 (1.72, 25.82, 0.730, 0.362).
- Text claim (p.9): "On the Mip-NeRF 360 dataset, our CAT-3DGS achieves (at its second highest rate
  point) 78× and 26x rate reductions than 3DGS and ScaffoldGS, respectively, while achieving slightly
  higher PSNR by 0.16 dB."

### 3.10 Densification and count control
- ScaffoldGS anchor growing, paused between iterations 3000 and 4000 (p.13).
- View frequency-aware masking: M_{n,k} = 1(sigmoid(m_{n,k}) · p_{n,k} > ϵ) where p is the relative
  frequency of use in training views (Eq. 9, p.7), with mask loss L_m = Σ sigmoid(m) (Eq. 12, p.7).
- The rate knob drives pruning through λm = max(10⁻³, 0.3 · λr) ("Rate-aware Mask Trade-off", p.13).
  Treehill anchors drop from 516.3K (high rate) to 85.1K (low rate) (Table 4, p.15). "at the lowest rate
  point, the number of anchor points is only one-third of that 'w/o RMT', and the total size achieves a
  60% reduction, while maintaining similar PSNR." (p.14)

### 3.11 Stated limitations and future work (verbatim)
- No limitations or future-work section exists. The closest statements:
- p.7: "For this scheme to work well, the basic premise is that the distribution of training views should
  be similar to that of test views. We argue that this is true to some extent because when their
  distributions differ significantly, there is little guarantee of the rendering quality in those test
  views."
- p.10: "CAT-3DGS has much higher decoding time than HAC due to the use of autoregressive models."
- p.10: "The decoding time of CAT-3DGS can be further improved by making full use of the parallelism in
  decoding triplanes."

### 3.12 Sentences touching a target, rate control, budget or desired size (verbatim)
- p.1 (abstract): "However, the needs to compress and transmit the 3DGS representation to the remote side
  are overlooked. This new application calls for rate-distortion-optimized 3DGS compression."
- p.8: "The rate parameter λr ranges from 0.002 to 0.04, and from 0.001 to 0.02 for BungeeNeRF."
- p.13: "we accordingly relate the mask hyperparameter λm to the rate hyperparameter λr using the
  relationship λm = max(10−3, 0.3 · λr). This implies that, at higher bit rates, λr decreases and more
  offsets (i.e. Gaussian primitives) and anchors are kept. Conversely, at lower bit rates, λr increases
  and more offsets and anchors are removed."
- p.16 (Table 6 caption): "For comparison, we also provide two results of different size and fidelity
  tradeoffs by adjusting λr."
- No sentence names a byte, MB, bitrate, memory or FPS target during training, a budget, or a desired size.

### 3.13 Code
- No repository URL is printed in the PDF. Licence: not stated.

---

## 4. FCGS

### 4.1 Citation
- Title: Fast Feedforward 3D Gaussian Splatting Compression
- First author: Yihang Chen (Shanghai Jiao Tong University and Monash University), with Qianyi Wu, Mengyao
  Li, Weiyao Lin, Mehrtash Harandi, Jianfei Cai
- Venue as stated in PDF: "Published as a conference paper at ICLR 2025"
- arXiv id: 2410.08017 (v3, 12 Mar 2025)

### 4.2 Base representation
- Raw 3DGS with degree-3 SH: geometry attributes f_geo ∈ R⁸ (opacity, scale, quaternion), colour f_col ∈
  R⁴⁸, positions μ_g (p.4). Agnostic to where the 3DGS came from (optimisation or feed-forward models).
- Render-time cost: none beyond vanilla 3DGS. The synthesis MLP g_s (for colour attributes routed through
  the autoencoder path) runs once at decode. "The rendering time of the decoded 3DGS is consistent with
  that before compression since FCGS does not alter the number or structure of the Gaussians." (p.10).
  The decoded model is a standard 3DGS point set with the same Gaussian count.

### 4.3 Budget handle
- Knob λ in L = L_fidelity + λ · L_entropy / (N_g × 56) (Eq. 8, p.7). "We adjust λ from 1e−4 to 16e−4 to
  achieve variable bitrates." (p.7). Values in tables: λ ∈ {1e−4, 2e−4, 4e−4, 8e−4, 16e−4} (Tables B, F,
  G, H, I, p.19 to p.22). Since FCGS is a generalisable model, λ selects which trained model is applied.
- No mask-rate weight λm: "we integrate the mask information directly into the bit consumption calculation,
  allowing the model to adaptively learn the optimal mask rate, thereby eliminating λm" (p.5).
- Target versus knob: knob only.

### 4.4 When it binds
- Post-hoc without per-scene training: "an optimization-free model that can compress 3DGS representations
  rapidly in a single feed-forward pass" (p.1). Input is a frozen, already trained 3DGS.
- The FCGS network itself is trained once on DL3DV-GS: 6770 3DGS scenes generated from DL3DV at 960P,
  "which takes about 60 GPU days on NVIDIA L40s", 100 held out for testing (p.8). Training stages (p.7):
  first g_a and g_s of the m = 1 path with fidelity loss only, then the colour context models with m = 1,
  then the whole model end-to-end. Iteration counts: not stated. Batch size 1 scene per step.
- Per-scene compression time versus per-scene optimisation baselines (Tables C to E, p.20, "multiple /
  single GPUs" for FCGS):
  - DL3DV-GS (Table C): 3DGS training 751 s. Light* (751 +) 227 s, Navaneet* (751 +) 546 s, Simon*
    (751 +) 122 s finetuned. SOG** 1068 s, EAGLES** 518 s, Lee** 938 s from scratch. Ours-lowrate
    (751 +) 9 / 16 s, Ours-highrate (751 +) 11 / 20 s.
  - MipNeRF360 (Table D): 3DGS 1583 s. Light* 422, Navaneet* 973, Simon* 195, SOG** 2391, EAGLES**
    1245, Lee** 1885. Ours-lowrate 10 / 31 s, Ours-highrate 14 / 41 s.
  - Tanks and Temples (Table E): 3DGS 814 s. Light* 241, Navaneet* 587, Simon* 120, SOG** 1219,
    EAGLES** 604, Lee** 1007. Ours-lowrate 10 / 16 s, Ours-highrate 13 / 24 s.
  - Fig. 1 (p.2): LightGaussian 227 s / 28.9 dB / 24.6 MB versus Ours 18 s / 29.2 dB / 21.1 MB on
    DL3DV-GS.

### 4.5 What "size" means
- Compressed bitstream in MB: positions "16-bit quantized and encoded losslessly using GPCC", attributes
  f_geo and f_col arithmetic coded in the ŷ and ẑ spaces, masks m arithmetic coded (p.7).
- Table F (p.21, DL3DV-GS): λ = 1e−4 total 27.44 MB = μ_g 3.25 + f_col(m=1) 2.61 + f_col(m=0) 10.92 +
  f_geo 10.50 + mask 0.17, mask rate 0.62. λ = 16e−4 total 15.83 = 3.25 + 2.94 + 1.76 + 7.80 + 0.08,
  mask rate 0.89. The shared FCGS network ("a total model size of less than 10 MB", p.25) is not listed
  as a bitstream component.
- Table G (p.21) per-parameter bits at λ = 1e−4: weighted average 2.51, μ_g 5.92, f_col(m=1) 0.45,
  f_col(m=0) 3.29, f_geo 6.94.

### 4.6 Runtime cost reported
- "On average, FCGS takes about 1 second to encode 100K Gaussians when running on a single GPU." (p.10)
- Table H (p.22), encoding, DL3DV-GS test set with "over 1.57 million Gaussians on average", single GPU:
  total 20.47 s at λ = 1e−4 (GPCC 9.48 s, 46 %) down to 15.86 s at 16e−4 (GPCC 9.44 s, 60 %).
- Table I (p.22), decoding: 15.93 s at λ = 1e−4 down to 11.47 s at 16e−4.
- FPS: "the average FPS is 102 and 91 before and after compression (λ = 1e−4) on the MipNeRF360 dataset."
  (p.10). GPU for this FPS: not named (the model is "trained on a single NVIDIA L40s GPU", p.7).
- GPU memory at render time: not stated.

### 4.7 Rate model
- Hyperprior autoencoder in the style of Ballé et al. 2018 (ẑ with a factorized prior), plus an
  inter-Gaussian context model (grids built from already decoded batches, N_s = 4 batches with ratios
  1/6, 1/6, 1/3, 1/3, 3D grid resolutions {70, 80, 90}, 2D grids {300, 400, 500}) and an intra-Gaussian
  context model (N_c = 4 channel chunks for the m = 1 latent, 3 RGB chunks for m = 0). The three parameter
  sets are mixed as a Gaussian mixture (Eq. 7, p.7).
- Quantization: "During training, uniform noise is added, while during testing, quantization is applied."
  (p.5). Step q = 1 in the latent (m = 1), "otherwise (mi = 0), q becomes a trainable decimal parameter"
  (p.5), so learned for the direct-quantization path (m = 0 colour and geometry).
- Geometry attributes bypass the autoencoder entirely (y = x) because "Geometry attributes are the most
  sensitive to such deviations" (p.5).
- Hyperprior ablation (Table M, p.25): removing it changes almost nothing (29.25 / 39.41 versus 29.26 /
  38.55 without intra and inter contexts).
- Gap between entropy estimate and encoded bytes: not stated. Random-seed variation of the split: standard
  deviation 8e−4 MB at λ = 1e−4 (p.18).

### 4.8 Scene dependence
- No per-scene table with scene names. Fig. E (p.27) shows eight test images at λ = 1e−4 with PSNR / size:
  25.33 / 42.65, 21.92 / 21.39, 26.31 / 93.94, 31.57 / 24.51, 23.40 / 55.75, 32.97 / 11.27,
  27.95 / 48.60, 25.78 / 26.18 (dB / MB), scenes not named in the text.
- No per-scene retuning by construction. The domain-gap limitation (Appendix B) shows that a 3DGS with
  unusual value statistics needs the mask forced to m = 0.

### 4.9 Main quantitative results (as reported)
- MipNeRF360 (Table D, p.20): 3DGS 27.52 / 0.813 / 0.221 / 741.12 MB. Light* 27.31 / 0.808 / 0.235 /
  48.61. Simon* 27.15 / 0.802 / 0.242 / 27.71. Navaneet* 26.80 / 0.796 / 0.256 / 20.66. Ours-lowrate
  27.05 / 0.798 / 0.237 / 34.64. Ours-highrate 27.39 / 0.806 / 0.226 / 64.05.
- Tanks and Temples (Table E, p.20): 3DGS 23.71 / 0.845 / 0.179 / 432.03. Simon* 23.63 / 0.842 / 0.187 /
  17.65. Light* 23.62 / 0.837 / 0.197 / 28.60. Ours-lowrate 23.48 / 0.832 / 0.193 / 17.89. Ours-highrate
  23.62 / 0.839 / 0.184 / 32.02.
- Deep Blending: only Fig. 8 (p.10), "3DGS (664MB)", no tabulated numbers. Text: "The size of the
  compressed 3DGS using FCGS (at the highest rate) is only 40% that of MSC with the same fidelity."
- DL3DV-GS (Table C, p.20): 3DGS 29.34 / 0.898 / 0.146 / 372.50. Ours-lowrate 28.86 / 0.891 / 0.156 /
  15.83. Ours-highrate 29.26 / 0.897 / 0.148 / 27.44. Light* 28.89 / 0.890 / 0.159 / 24.61.
- Rate points on MipNeRF360: only the two rows above are tabulated. Fig. 4 (p.8) plots five λ points
  without numbers. The five-point sweep with numbers exists only for DL3DV-GS (Table L, p.25): 29.26 /
  27.44, 29.22 / 24.15, 29.17 / 21.12, 29.05 / 18.14, 28.86 / 15.83 (dB / MB) for λ = 1e−4 to 16e−4.

### 4.10 Densification and count control
- None. The Gaussian count is whatever the input 3DGS has. The MEM mask m routes each Gaussian's colour
  attributes through the autoencoder (m = 1) or direct quantization (m = 0), it does not prune (Eq. 3,
  p.5). Mask rate is learned through the bit term (Table B, p.19: 0.62 to 0.89 on DL3DV-GS as λ grows).
- FCGS can be stacked on pruning methods (Mini-Splatting, Trimming), reaching "a compression ratio of
  100× over the vanilla 3DGS" on Deep Blending (Fig. 8 caption, p.10).

### 4.11 Stated limitations and future work (verbatim)
- p.18 (Appendix B, printed appendix page 4): "Failures mainly arise from incorrect selections of m by MEM
  due to the domain gap: a Gaussian that should have been assigned to the m = 0 path for fidelity
  preservation may be mistakenly assigned to m = 1, and the autoencoder in this path fails to recover
  these Gaussian values properly from their latent space, leading to significant fidelity degradation."
  and "Overall, this is our major limitation, which points out an interesting problem for future work:
  how to design optimization-free compression models that can be well generalized to 3DGS with different
  value characteristics."
- p.19 (Appendix D): "indeed, there is a performance gap compared to these two methods (i.e., HAC and
  ContextGS)." and "It is also promising to extend FCGS to anchor-based 3DGS variants (Lu et al., 2024)
  to achieve better RD performance, which we leave for future work."
- p.25 (Appendix L): "This finding suggests a promising direction for future work: training FCGS with a
  larger and more diverse dataset to further enhance its capability."
- p.2: "Although the absence of per-scene finetuning naturally limits our RD performance, we have still
  achieved a compression ratio exceeding 20×"

### 4.12 Sentences touching a target, rate control, budget or desired size (verbatim)
- p.5: "Balancing the mask rate is crucial, as it directly impacts the RD trade-off. Previous works like
  Lee et al. (2024) incorporate an additional loss term with a hyperparameter λm to regulate the mask
  rate. However, given that the RD loss already includes a trade-off hyperparameter λ to balance bits and
  fidelity, introducing another parameter, λm, would unnecessarily complicate the process of finding the
  optimal RD trade-off."
- p.7: "λ is a hyperparameter controlling the trade-off between fidelity and entropy."
- p.7: "We adjust λ from 1e −4 to 16e −4 to achieve variable bitrates."
- p.2: "In contrast, our pipeline offers a convenient and hassle-free solution, where a single model can
  directly and rapidly compress various 3DGS without the need for finetuning, making it well-suited for
  time-sensitive applications."
- No sentence names a byte, MB, bitrate, memory or FPS target, a budget, or a desired size.

### 4.13 Code
- github.com/YihangChen-ee/FCGS (p.1). Licence: not stated.

---

## 5. CompGS

### 5.1 Citation
- Title: CompGS: Efficient 3D Scene Representation via Compressed Gaussian Splatting
- First authors: Xiangrui Liu and Xinju Wu (equal contribution), City University of Hong Kong, with
  Pingping Zhang, Shiqi Wang (corresponding), Zhu Li, Sam Kwong
- Venue as stated in PDF: not stated. (FCGS's reference list cites it as ACM Multimedia 2024, this PDF
  does not.)
- arXiv id: 2404.09458 (v1, 15 Apr 2024)

### 5.2 Base representation
- A "hybrid primitive structure" (p.3): anchor primitives ω with location μ_ω, covariance Σ_ω and
  reference embedding f_ω (dimension 32), each associated with K = 10 coupled primitives γ_k that hold
  only residual embeddings g_k (dimension 8) (p.3, p.6). Anchors are initialised from voxel-downsampled
  SfM points (p.6).
- Render-time cost: every coupled primitive's Gaussian is predicted by neural networks per view. Geometry
  by affine warping of the anchor with translation, scale and rotation from networks ϕ, ψ, φ on
  h_k = f_ω ⊕ g_k (Eq. 3 to 4, p.4), opacity and colour by networks κ, ζ on view embeddings ε ⊕ h_k
  (Eq. 5, p.5). Networks are "two residual multi-layer perceptrons" (p.6). Entropy decoding once at load
  (4.46 s, Table 7). Not a vanilla 3DGS point set (view-dependent MLP prediction as in Scaffold-GS [27]).

### 5.3 Budget handle
- Knob λ, "the Lagrange multiplier to control the trade-off between rate and distortion" in
  L = λR + D (Eq. 1, p.4). "The Lagrange multiplier λ in Equation 1 is set to {0.001, 0.005, 0.01} to
  obtain multiple bitrate points." (p.6). Three rows per dataset in Tables 1 to 3, unlabelled by λ.
- Target versus knob: knob only. "Rate-constrained optimization" in this paper means the R term in the
  loss, not a constraint to satisfy.

### 5.4 When it binds
- During training, end-to-end: "all primitives within the proposed CompGS are jointly optimized via
  rate-distortion cost minimization" (p.4), "the rate-distortion cost is used to perform end-to-end
  optimization of primitives and neural networks" (p.6). Whether the rate term is active from iteration 0:
  not stated. Iteration count: not stated.
- Wall-clock: "the proposed method requires an average of 37.83 minutes for training" on Tanks and
  Temples (Table 7, p.9). Encoding after training with arithmetic coding and G-PCC (p.6).

### 5.5 What "size" means
- Bitstream in MB: arithmetic-coded anchor reference embeddings and covariances plus coupled residual
  embeddings, with G-PCC for anchor locations (p.6). Fig. 8 (p.9) shows three bitstream components:
  anchor primitives, coupled primitives and "Network Weights", so network weights are inside the reported
  size. Raw extracted percentage row for λ = 0.001, 0.005, 0.01: "32.05% 38.39% 29.56% 32.89% 29.33%
  37.78% 29.87% 24.14% 45.98%". The assignment of percentages to the three components cannot be recovered
  from the text dump.
- Bits per primitive (Fig. 8 lower, raw row "128.26 15.36 Anchor Primitives 102.1 9.11 80.65 6.52 Coupled
  Primitives"): reading as anchor 128.26 / 102.1 / 80.65 and coupled 15.36 / 9.11 / 6.52 bits at
  λ = 0.001 / 0.005 / 0.01. Text: "the average bit consumption of coupled primitives is demonstrably lower
  than that of anchor primitives" (p.8).

### 5.6 Runtime cost reported
- Table 7 (p.9, Tanks and Temples), hardware not stated: Proposed training 37.83 min, encoding 6.27 s,
  decoding 4.46 s, rendering 5.32 ms per view. Navaneet et al. 14.38 min / 68.29 s / 12.32 s / 9.88 ms.
  Niedermayr et al. 15.50 / 2.23 / 0.25 / 9.74. Lee et al. 44.70 / 1.96 / 0.18 / 6.60. Girish et al.
  8.95 / 0.54 / 0.64 / 6.96.
- GPU memory: not stated.

### 5.7 Rate model
- Scalar quantization of Σ_ω, f_ω, g_k with steps s_Σ, s_f, s_g (Eq. 6, p.5). "Quantization steps
  {sf, sg} are fixed to 1, whereas sΣ is a learnable parameter with an initial value of 0.01." (p.6).
- Relaxation: additive uniform quantization noise (Eq. 7, p.5).
- Entropy models: p(f̃_ω) Gaussian with (τ_f, ρ_f) predicted from hyperpriors η_f, which are coded with a
  factorized entropy bottleneck (Eq. 8 to 9, p.5). p(Σ̃_ω) Gaussian conditioned on f̃_ω (Eq. 10).
  p(g̃_k) Gaussian conditioned on f̃_ω ⊕ η_g (Eq. 11). Implemented with CompressAI (p.6).
- Gap between entropy estimate and encoded bytes: not stated.

### 5.8 Scene dependence
- Table 10 (p.13), first Proposed row per scene (the largest-size row, the three rows are not labelled
  with λ): Bicycle 21.74, Flowers 25.44, Garden 28.66, Stump 12.02, Tree Hill 20.02, Room 8.85, Counter
  9.61, Kitchen 10.39, Bonsai 11.78 MB. One setting spans 8.85 MB to 28.66 MB, a 3.2× spread.
- Tanks and Temples (Table 8, p.12): Train 8.60 / 6.72 / 5.51 MB, Truck 10.61 / 7.82 / 6.27 MB.
- Per-scene retuning statement: none.

### 5.9 Main quantitative results (as reported)
- Mip-NeRF 360 (Table 3, p.6): Kerbl et al. 27.46 / 0.82 / 0.22 / 788.98. Navaneet et al. 27.04 / 0.81 /
  0.23 / 86.10. Niedermayr et al. 27.12 / 0.80 / 0.23 / 28.61. Lee et al. 27.05 / 0.80 / 0.24 / 49.60.
  Girish et al. 27.04 / 0.80 / 0.24 / 65.09. Proposed 27.26 / 0.80 / 0.24 / 16.50, 26.78 / 0.79 / 0.26 /
  11.02, 26.37 / 0.78 / 0.28 / 8.83.
- Tanks and Temples (Table 1, p.6): Kerbl 23.72 / 0.85 / 0.18 / 434.38. Niedermayr 23.58 / 0.85 / 0.19 /
  17.65. Lee 23.40 / 0.84 / 0.20 / 39.47. Proposed 23.70 / 0.84 / 0.21 / 9.60, 23.39 / 0.83 / 0.22 / 7.27,
  23.11 / 0.81 / 0.24 / 5.89.
- Deep Blending (Table 2, p.6): Kerbl 29.54 / 0.91 / 0.24 / 665.99. Girish 29.90 / 0.91 / 0.25 / 61.69.
  Navaneet 29.89 / 0.91 / 0.25 / 72.46. Niedermayr 29.45 / 0.91 / 0.25 / 23.87. Proposed 29.69 / 0.90 /
  0.28 / 8.77, 29.40 / 0.90 / 0.29 / 6.82, 29.30 / 0.90 / 0.29 / 6.03.
- Ablation (Table 4, p.8), Train scene: baseline 3DGS 257.44 MB / 22.02, hybrid structure only 48.58 MB /
  22.15, hybrid plus rate-constrained optimisation 8.60 MB / 22.12. Truck: 611.31, 30.38, 10.61 MB.

### 5.10 Densification and count control
- "adaptive control [27] is applied to manage the number of anchor primitives" (p.6), i.e. the Scaffold-GS
  anchor growing and pruning rule. K = 10 coupled primitives per anchor, fixed (Table 6, p.9). No learned
  pruning mask. Whether the rate term feeds back into anchor count: not stated.

### 5.11 Stated limitations and future work (verbatim)
- No limitations or future-work section exists. The closest statements:
- p.9: "the proposed method requires an average of 37.83 minutes for training, which is shorter than the
  method proposed by Lee et al. [20] and longer than other methods. This might be attributed to that the
  proposed method needs to optimize both primitives and neural networks."
- p.9: "the increase of K from 10 to 15 leads to a rendering quality degradation of 0.22 dB. This might be
  because excessive coupled primitives could lead to an inaccurate prediction."

### 5.12 Sentences touching a target, rate control, budget or desired size (verbatim)
- p.1 (abstract): "we develop a rate-constrained optimization scheme to eliminate redundancies within such
  hybrid primitives, steering our CompGS towards an optimal trade-off between bitrate consumption and
  representation efficacy."
- p.4: "where λ denotes the Lagrange multiplier to control the trade-off between rate and distortion"
- p.6: "The Lagrange multiplier λ in Equation 1 is set to {0.001, 0.005, 0.01} to obtain multiple bitrate
  points."
- p.7: "Notably, extant compression methods [10, 20, 33, 34] only provide the configuration for a single
  bitrate point."
- No sentence names a byte, MB, bitrate, memory or FPS target, a budget, or a desired size.

### 5.13 Code
- "Our code will be released on GitHub for further research." (p.1). No URL printed. Licence: not stated.

---

## 6. Cross-paper synthesis for the gap question

| Paper | Handle | Binds when | Size meaning | Render-time MLP |
|---|---|---|---|---|
| RDO-Gaussian | λGSprune, λSHprune knobs (ECVQ λ fixed) | training, after 15k plain 3DGS iterations | arithmetic-coded indexes + codebooks + logits + float16 positions | no |
| ContextGS | λe knob | training, 30000 iterations | bitstream + uncoded positions + MLPs | yes (Scaffold-GS) |
| CAT-3DGS | λr knob, λm tied to λr | training, staged from iteration 3000 / 10000 / 15000 | entropy-coded bitstream incl. triplanes, mask, 16-bit positions, 32-bit MLPs | yes (ScaffoldGS) |
| FCGS | λ selects one of five trained models | post-hoc, no per-scene training | GPCC positions + arithmetic-coded attributes and masks, shared network excluded | no |
| CompGS | λ ∈ {0.001, 0.005, 0.01} knob | training, end-to-end | arithmetic-coded anchors and residuals + G-PCC positions + network weights | yes (own MLPs) |

- None of the five takes a size, byte count, bitrate, memory or FPS number as input, and none adjusts
  anything during training to meet one. Every method produces a rate point per λ and the user reads the
  size off afterwards.
- Evidence that λ does not pin a size: at one fixed setting, Mip-NeRF 360 per-scene sizes span 7.91 to
  43.14 MB (RDO-Gaussian, 5.5×), 4.50 to 21.82 MB (ContextGS, 4.8×), 4.11 to 21.42 MB (CAT-3DGS, 5.2×),
  8.85 to 28.66 MB (CompGS, 3.2×). Hitting a byte target with these methods needs a per-scene λ sweep,
  which none of the papers performs or discusses.
- RDO-Gaussian is the raw-3DGS method with the rate term in training, but the entropy-estimated rate
  covers only the six VQ index streams. Count is driven by a mask-mean proxy, and positions (43.9 % of the
  high-rate file) plus opacity sit outside the RD objective. The rate term switches on at 15k iterations,
  not at iteration 0.
- The only place a rate knob drives primitive count is CAT-3DGS's coupling λm = max(10⁻³, 0.3 λr). That
  is still open loop.
- Wording that could be mistaken for the gap being filled: RDO-Gaussian's "flexible and continuous rate
  control" (a continuous λ knob, not a control loop) and ContextGS's "target ratio" τ (a count ratio
  between anchor levels for the coding structure, found once by binary search after initialisation).
  Neither is a byte target. A gap statement should say "no method takes a byte target as input and steers
  training to it", not "no method mentions rate control".
- FCGS is the outlier in pipeline shape (seconds, post-hoc, generalisable) but is still a λ family. It
  also states that its size follows the input 3DGS and that its network cannot change the count.
