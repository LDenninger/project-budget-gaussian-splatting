# Literature notes B1: HAC family (rate-distortion, anchor based)

Source PDFs: `references/02_rate_distortion/`. Text extracted with pymupdf, every page read. Page numbers below are PDF page numbers as extracted (`=== page N ===`). For HAC the supplementary material starts at PDF page 18 (its own numbering "Supp. 1"). Every number in these notes is copied from the paper and is "as reported" by its authors, none was recomputed. Quotes in fields 11 and 12 are verbatim, including any typos.

Reading key for the gap question: none of the five papers accepts a byte, MB, bitrate, memory or FPS number as input. Every one of them exposes a Lagrange multiplier λ (or a set of λ values) as the handle, and the resulting size is only known after training and encoding. HEMGS, SALVQ and PCGS reduce the number of trainings needed to cover several rate points, but the achievable points are still indexed by λ, not by a target size.

---

## Paper 1: HAC (arXiv 2403.14530)

### 1. Citation
"HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression". Yihang Chen (Shanghai Jiao Tong University and Monash University), Qianyi Wu, Weiyao Lin, Mehrtash Harandi, Jianfei Cai. Venue not printed in the PDF (Springer LNCS style, file name says ECCV 2024, HAC++ cites it as "European Conference on Computer Vision, 2024"). arXiv:2403.14530v3 [cs.CV] 12 Jul 2024.

### 2. Base representation
Scaffold-GS anchors plus MLPs. Each anchor holds location xᵃ ∈ ℝ³ and attributes A = {fᵃ ∈ ℝ^Da, l ∈ ℝ⁶, o ∈ ℝ^3K}. Dₐ raised to 50 and the Scaffold-GS feature bank disabled (p.10). A binary mixed 3D-2D hash grid (12 levels 3D, 16 to 512 resolution, 4 levels 2D, 128 to 1024, table sizes 2¹³ and 2¹⁵, Dₕ = 4) is used only as context for entropy coding, not as part of the renderer.
Render-time cost: the Scaffold-GS MLPs must run per view (fᵃ is fed into MLPs to produce Gaussian attributes, p.6). The hash grid is not needed at render time: "The inference process benefits from the design of context modeling, allowing for the removal of the hash grid once A is decoded. Consequently, no additional operations are required during rendering, resulting in a similar FPS with Scaffold-GS." (p.14). Entropy decoding happens once when loading (0.87 s to 26.7 s, see field 6). The decoded model is a Scaffold-GS model (anchors plus MLPs), not a vanilla 3DGS point set renderable by the plain rasterizer.

### 3. Budget handle
A knob, λₑ, in the loss (Eq. 9, p.10):
Loss = L_Scaffold + λₑ · 1/(N(Dₐ + 6 + 3K)) · (L_entropy + L_hash) + λₘ L_m
with λₘ = 5e−4 fixed and "change λe from 5e −4 to 4e −3 for variable bitrates" (p.10). Rate points in the per-scene tables: λₑ ∈ {0.004, 0.003, 0.002, 0.001, 0.0005} (Tables B to F, p.21 to 24). Base quantization steps Q₀ = 1, 0.001, 0.2 for fᵃ, l, o (p.10).
No target notion. The entropy term is normalised by the number of anchors N and the per-anchor dimension, so it is a per-parameter bit average, not a total byte count. The "Ours-lowrate" and "Ours-highrate" rows of Table 1 (p.11) match different λₑ per dataset: Synthetic-NeRF lowrate = λₑ 0.002 row of Table B, highrate = 0.0005. Mip-NeRF360, Tanks and Temples, DeepBlending lowrate = 0.004, highrate = 0.001 (Tables C, D, E). BungeeNeRF lowrate = 0.004, highrate = 0.0005 (Table F). The paper does not explain the per-dataset choice.

### 4. When it binds
During training, in stages (Fig. A, p.18, and Sec. A.1, p.18 to 19), total 30 000 iterations:
- 0 to 3000: plain Scaffold-GS, "no additional techniques are applied".
- 3000 to 10000: "adding noise" to A with Q₀ only, no hash grid, no r refinement. Anchor spawning paused 3000 to 4000, re-enabled after 4000.
- after 10000: full HAC, L_entropy, L_hash, L_m applied, hash grid bound set from anchor extent at iteration 10000. Anchor spawning ends at 15000 (Scaffold-GS milestone).
- entropy training on a random 5 % of rendered anchors per iteration (Sec. A.2, p.19).
Wall clock (single NVIDIA RTX 4090, p.10, p.14): BungeeNeRF 38.2 min 3DGS, 15.1 min Scaffold-GS, 27.6 min HAC. Synthetic-NeRF 3.4 min, 4.4 min, 9.0 min. "approximately 0.9× longer than Scaffold-GS" (p.14).

### 5. What "size" means
On-disk bitstream in MB after entropy coding (Table 1 caption "The size is measured in MB", p.11). Five components (p.12): anchor attributes A (fᵃ, l, o) coded by arithmetic coding with HAC probabilities, binary hash grid H (AE with frequency h_f), offset masks (binary, AE), anchor locations xᵃ stored in 16 bits, MLPs stored in 32 bits. Breakdown on BungeeNeRF at λₑ = 4e−3: A 14.90 MB (fᵃ 8.76, l 2.52, o 3.62), H 0.15 MB, masks 0.52 MB, xᵃ 2.77 MB, MLPs 0.16 MB (p.12). Table A (p.19): per-parameter bits at λₑ = 4e−3 on BungeeNeRF fᵃ 3.03, l 7.27, o 4.56 bit, on Synthetic-NeRF 1.33, 3.58, 3.76 bit.

### 6. Runtime cost reported
Hardware: single NVIDIA RTX 4090 (p.10).
- Coding: "The encoding/decoding process takes approximately 0.87 seconds and 26.7 seconds on Synthetic-NeRF and BungeeNeRF dataset under λe = 4e−3, respectively. The dominant time consumption occurs during Codec execution of AE on the CPU (over 90%), as we only use a single thread." (p.14).
- FPS: "The rendering FPS are 75, 232 and 283 for 3DGS, Scaffold-GS and HAC on BungeeNeRF, and 401, 326 and 341 on Synthetic-NeRF." (p.14).
- GPU memory at render time: not stated.

### 7. Rate model
- Entropy model: hash-grid context. MLP_c maps interpolated hash feature fʰ to per-element Gaussian parameters (μ, σ) for each attribute vector, attributes assumed independent (Eq. 6, p.9). Probability of a quantized value is the Gaussian CDF difference over the bin of width q. Loss L_entropy = Σ over attributes, anchors, dimensions of −log₂ p (Eq. 7, p.9). Hash grid binarised to {−1, +1} with STE and its bits estimated from the frequency of +1 (Eq. 8, p.9).
- Quantization relaxation: "we utilize the 'adding noise' operation during training and 'rounding' during testing, as described in [1]" (Ballé et al.), Eq. 4: f̂ = f + U(−½, ½) × q in training, Round(f/q) × q in testing (p.8).
- Step sizes: Adaptive Quantization Module, qᵢ = Q₀ × (1 + Tanh(rᵢ)), rᵢ = MLP_q(fʰᵢ) (Eq. 5, p.8), so a learned per-anchor refinement within (0, 2Q₀), with a fixed per-attribute-type Q₀ (1, 0.001, 0.2). MLP_q and MLP_c share one 3-layer ReLU MLP.
- Gap between entropy estimate and encoded bytes: not stated.
- Variable rate: no, one model per λₑ.

### 8. Scene dependence
Table C (p.22), sizes in MB at fixed λₑ = 0.004: bicycle 27.54, garden 22.69, stump 18.11, room 5.53, counter 7.26, kitchen 8.05, bonsai 8.56, flower 19.59, treehill 20.04 (AVG 15.26). At λₑ = 0.0005: bicycle 44.01, room 9.16, AVG 25.01. The spread across scenes at one λ is about 5× (bicycle versus room), and the same λ gives 0.51 MB (hotdog) to 1.82 MB (ship) on Synthetic-NeRF (Table B, p.21) and 22.49 MB (amsterdam) on BungeeNeRF (Table F, p.24).
Statement on scene simplicity: "With scenes become simpler, the storage share of A decreases as the value distribution become easier to predict." (p.12). No statement about retuning λ per scene, but the headline rows use different λₑ per dataset (field 3).

### 9. Main quantitative results (as reported, Table 1, p.11, size in MB)
Mip-NeRF360 (PSNR / SSIM / LPIPS / size):
- 3DGS 27.49 / 0.813 / 0.222 / 744.7
- Scaffold-GS 27.50 / 0.806 / 0.252 / 253.9
- Compressed3D 26.98 / 0.801 / 0.238 / 28.80
- Morgenstern et al. 26.01 / 0.772 / 0.259 / 23.90
- Ours-lowrate 27.53 / 0.807 / 0.238 / 15.26
- Ours-highrate 27.77 / 0.811 / 0.230 / 21.87
Tanks and Temples:
- 3DGS 23.69 / 0.844 / 0.178 / 431.0
- Scaffold-GS 23.96 / 0.853 / 0.177 / 86.50
- Compressed3D 23.32 / 0.832 / 0.194 / 17.28
- Morgenstern et al. 22.78 / 0.817 / 0.211 / 13.05
- Ours-lowrate 24.04 / 0.846 / 0.187 / 8.10
- Ours-highrate 24.40 / 0.853 / 0.177 / 11.24
DeepBlending:
- 3DGS 29.42 / 0.899 / 0.247 / 663.9
- Scaffold-GS 30.21 / 0.906 / 0.254 / 66.00
- Compressed3D 29.38 / 0.898 / 0.253 / 25.30
- Morgenstern et al. 28.92 / 0.891 / 0.276 / 8.40
- Ours-lowrate 29.98 / 0.902 / 0.269 / 4.35
- Ours-highrate 30.34 / 0.906 / 0.258 / 6.35
Full Mip-NeRF360 RD curve (Table C AVG rows, p.22): λₑ 0.004: 27.53 / 0.807 / 0.238 / 15.26. 0.003: 27.62 / 0.808 / 0.236 / 16.57. 0.002: 27.72 / 0.809 / 0.233 / 18.54. 0.001: 27.77 / 0.811 / 0.230 / 21.87. 0.0005: 27.83 / 0.811 / 0.229 / 25.01.
Other baselines in Table 1 on Mip-NeRF360: Lee et al. 27.08 / 0.798 / 0.247 / 48.80, EAGLES 27.15 / 0.808 / 0.238 / 68.89, LightGaussian 27.00 / 0.799 / 0.249 / 44.54, Navaneet et al. 27.16 / 0.808 / 0.228 / 50.30.

### 10. Densification and count control
Anchor count comes from Scaffold-GS anchor spawning (growing), enabled from 1600 to 15000 iterations except the 3000 to 4000 pause (Fig. A, p.18). Pruning: "Adaptive Offset Masking" adopted from Lee et al. [22], straight-through binary masks on offsets with a masking loss L_m weighted by λₘ = 5e−4 "to encourage masking as many Gaussians as possible" (p.9). "if all the attached o are pruned on an anchor, then this anchor no longer contributes to rendering and should be pruned entirely (including its xᵃ and A)" (p.9). The rate term itself does not enter the mask decision in HAC: the mask is driven by the separate L_m with its own fixed weight. The entropy loss is per-parameter normalised (division by N), so it does not directly penalise the anchor count. The offset o distribution "exhibits an impulse at zero, suggesting the occurrence of substantial unnecessary Gaussians" (p.9).

### 11. Stated limitations and future work (verbatim)
- p.14: "This increase of training time in our model over Scaffold-GS is our main limitation, but it is still fast."
- Future work: not stated (the conclusion on p.14 names none).

### 12. Sentences touching a target, rate control, budget or desired size (verbatim)
- p.9 to 10: "During training, we incorporate both the rendering fidelity loss and the entropy loss to ensure the model improves rendering quality while controlling total bitrate consumption in a differentiable manner."
- p.10: "The second part in Eq. (9) is the estimated controllable bit consumption, including the estimated bits Lentropy for anchor attributes and Lhash for the hash grid."
- p.10: "λe and λm are trade-off hyperparameters used to balance the loss components."
- p.10: "We set λm to 5e −4, and change λe from 5e −4 to 4e −3 for variable bitrates."
- p.11: "Note that BD-rate [4] is incalculable as other methods can typically only output a single rate, while four are needed for its calculation."
- p.11 (Table 1 caption): "For our approach, we give two results of different size and fidelity tradeoffs by adjusting λe. A smaller λe results in a larger size but improved fidelity, and vice versa."
- p.12 (Fig. 4 caption): "We vary λe to achieve variable bitrates."
- p.27 (Table I): "λe Tradeoff parameter to achieve variable birate"
No sentence names a byte, MB, memory or FPS target, and the words "budget" and "desired" do not occur.

### 13. Code
"Our code is available here." (p.2), the hyperlink target is https://github.com/YihangChen-ee/HAC (extracted from the PDF link annotation on page 2, the URL is not printed as text). Licence: not stated.

---

## Paper 2: HAC++ (arXiv 2501.12255)

### 1. Citation
"HAC++: Towards 100X Compression of 3D Gaussian Splatting". Yihang Chen (Shanghai Jiao Tong University and Monash University), Qianyi Wu, Weiyao Lin, Mehrtash Harandi, Jianfei Cai. Venue not printed in the PDF (IEEE journal template header "JOURNAL OF LATEX CLASS FILES", file name says TPAMI 2025, SALVQ ref [17] cites it as IEEE TPAMI 2025). arXiv:2501.12255v4 [cs.CV] 11 Feb 2025.

### 2. Base representation
Scaffold-GS anchors plus MLPs, Dₐ = 50, K = 10 offsets per anchor, feature bank disabled (p.8). Binary hash grid as in HAC (same configuration) plus an intra-anchor context model (MLP_a) and a GMM to fuse both contexts. Render-time: the Scaffold-GS MLPs run per view. Hash grid dropped after decoding: "HAC++ benefits from its context modeling design, enabling the removal of the hash grid after decoding A. This design eliminates the need for additional operations during rendering." (p.13). Decoding is sequential for fᵃ: "it is encoded/decoded sequentially chunk by chunk" (p.7) with N_c = 5 chunks, anchor locations decoded with GPCC. The decoded model is a Scaffold-GS model, not a vanilla 3DGS point set.

### 3. Budget handle
A knob, λ, in the loss (Eq. 13, p.7):
Loss = L_Scaffold + λ · 1/(N(Dₐ + 6 + 3K)) · (L_entropy + L_hash)
No separate mask loss (mask is inside the rate term, field 10). "We change λ from 0.5e −3 to 4e −3 to achieve variable bitrates." (p.8). Rate points: λ ∈ {0.5e−3, 1e−3, 2e−3, 3e−3, 4e−3} (Tables IV to VIII, p.12, and Appendix Tables A to E, p.17 to 18). Q₀ = 1, 0.001, 0.2 for fᵃ, l, o (p.8).
No target notion. Table II "lowrate" rows match λ = 4e−3 for Mip-NeRF360, Tanks and Temples, DeepBlending, BungeeNeRF and λ = 3e−3 for Synthetic-NeRF (Appendix Table A). "highrate" rows match λ = 0.5e−3 for Synthetic-NeRF and Mip-NeRF360 and λ = 1e−3 for Tanks and Temples, DeepBlending and BungeeNeRF (Tables B, C, E only list four λ values, no 0.5e−3).

### 4. When it binds
During training in stages (Fig. 4 and "Training Process", p.8), 30k iterations total "consistent with Scaffold-GS and 3DGS methods to ensure a fair comparison" (p.8):
- 0 to 3k: original Scaffold-GS.
- 3k to 10k: noise added to A with Q₀ only, no hash grid, spawning paused 3k to 4k.
- after 10k: full HAC++ (L_entropy, L_hash), spawning enabled until 15k.
- 5 % of all anchors entropy-trained per iteration.
Wall clock on Mip-NeRF360 (Table X, p.13, NVIDIA L40s 48 GB): 3DGS 1590 s, Scaffold-GS 1286 s, HAC++ 2384 s (lowrate) and 2278 s (highrate), "an 81% increase in training time compared to Scaffold-GS" (p.13). Peak GPU memory during training 12.00 / 9.69 / 10.87 / 11.66 GB. Longer schedules (Table XII, p.14): 40k 3307 s BD-rate −17.5 %, 50k 4224 s −25.6 %, 60k 5294 s −32.9 %, relative to 30k at 2292 s.

### 5. What "size" means
On-disk bitstream in MB (Table II "THE SIZES ARE MEASURED IN MB", p.9). Components (p.8 to 9): A entropy coded with AE, H and masks m coded by AE from occurrence frequencies, "Anchor locations xa are 16-bit quantized and losslessly compressed using GPCC [49]. The MLP parameters are stored directly using 32-bit precision." Only valid anchors are encoded (p.11). Table VI (p.12), Mip-NeRF360, total = xᵃ + fᵃ + l + o + H + m + MLP in MB:
- λ 0.5e−3: 18.48 = 0.91 + 9.54 + 2.52 + 4.49 + 0.12 + 0.56 + 0.33
- λ 1e−3: 14.73 = 0.83 + 7.52 + 2.18 + 3.27 + 0.12 + 0.48 + 0.33
- λ 2e−3: 11.18 = 0.75 + 5.61 + 1.81 + 2.19 + 0.11 + 0.38 + 0.33
- λ 3e−3: 9.35 = 0.70 + 4.61 + 1.60 + 1.69 + 0.10 + 0.32 + 0.33
- λ 4e−3: 8.34 = 0.67 + 4.00 + 1.48 + 1.47 + 0.09 + 0.30 + 0.33
Per-parameter bits (Table VII, p.12) at λ 0.5e−3 / 4e−3: xᵃ 5.19 / 5.53, fᵃ 3.23 / 1.90, l 7.15 / 6.00, o 6.72 / 5.93, m 0.94 / 0.72.

### 6. Runtime cost reported
Hardware: single NVIDIA L40s GPU with 48 GB (p.8).
- Encoding time Mip-NeRF360 (Table IV, p.12): 18.10 s (λ 0.5e−3, 491852 valid anchors), 15.19 s, 12.17 s, 10.59 s, 9.80 s (λ 4e−3, 342049 valid anchors). fᵃ takes about 46 %.
- Decoding time Mip-NeRF360 (Table V, p.12): 30.86 s, 25.62 s, 20.05 s, 17.22 s, 15.77 s. fᵃ about 49 to 51 %.
- Table IX (p.12) enc / dec, lowrate then highrate: Synthetic-NeRF 0.80, 1.57 / 1.18, 2.36 s. Mip-NeRF360 9.80, 18.10 / 15.77, 30.86 s. Tanks and Temples 6.01, 9.01 / 9.58, 14.20 s. DeepBlending 3.35, 5.17 / 4.92, 7.86 s. BungeeNeRF 12.44, 18.62 / 18.67, 28.58 s.
- FPS on Mip-NeRF360 (Table XI, p.13): 3DGS 99 FPS (3175k Gaussians), Scaffold-GS 135 FPS (5674k), HAC++ 151 FPS lowrate (682k valid Gaussians), 130 FPS highrate (1853k). Ablation Table III (p.11): HAC++ 141 FPS.
- GPU memory at render time: not stated (only training peak, field 4).

### 7. Rate model
- Entropy model: two-component Gaussian Mixture Model (Eq. 8, p.6) fusing (a) the hash-grid context, MLP_c(fʰ) → μˢ, σˢ, πˢ (Eq. 6, p.6), and (b) an intra-anchor channel-wise autoregressive context, MLP_a over N_c = 5 chunks of fᵃ conditioned on previously decoded chunks and on μˢ, σˢ, πˢ (Eq. 7, p.6). Mixture weights are softmax of πˢ, πᶜ. Intra context applies only to fᵃ (l and o gain nothing, Table III: −0.1 % and +0.9 % BD-rate). Removing the hash context costs +63.3 % BD-rate, removing intra context +14.7 %, replacing the GMM by concatenation +5.7 % (Table III, p.11).
- Quantization relaxation: additive uniform noise in training, rounding at test (Eq. 4, p.5).
- Step sizes: same AQM as HAC, qᵢ = Q₀ × (1 + Tanh(rᵢ)), rᵢ = MLP_q(fʰᵢ) (Eq. 5, p.5), per-anchor refinement of a per-attribute-type Q₀. Without AQM the fidelity collapses (Table III "N/A").
- Mask-aware rate: bits of anchor i are weighted by the anchor mask mᵃᵢ and the per-offset masks mᵢ,ₖ (Eq. 11, p.7).
- Gap between entropy estimate and encoded bytes: not stated.
- Variable rate: no, one model per λ.

### 8. Scene dependence
Appendix Table D (p.18), Mip-NeRF360 sizes in MB at λ = 4e−3: bicycle 12.84, garden 12.83, stump 9.40, room 3.41, counter 4.69, kitchen 5.26, bonsai 5.35, flower 10.68, treehill 10.57 (AVG 8.34). At λ = 0.5e−3: bicycle 29.25, garden 27.22, room 7.55, bonsai 11.15 (AVG 18.48). Spread across scenes at one λ about 3.8×. Table VIII (p.12): total anchors 560425 to 571923 across λ, valid anchor ratio 0.865 (0.5e−3) to 0.596 (4e−3), valid Gaussian ratio 0.314 to 0.118.
Retuning statement: none about per-scene retuning. The paper does state that the HAC-style mask loss weight had to be retuned per rate point: "Specifically, the weight of Lm must be manually adjusted across different RD trade-off points in Eq. 13 as λ varies to achieve optimal mask ratios. This process is tedious, and difficult to optimize effectively." (p.6 to 7).

### 9. Main quantitative results (as reported, Table II, p.9, size in MB)
Mip-NeRF360:
- 3DGS 27.46 / 0.812 / 0.222 / 750.9
- Scaffold-GS 27.50 / 0.806 / 0.252 / 253.9
- HAC (lowrate) 27.53 / 0.807 / 0.238 / 15.26, HAC (highrate) 27.77 / 0.811 / 0.230 / 21.87
- ContextGS (lowrate) 27.62 / 0.808 / 0.237 / 12.68, ContextGS (highrate) 27.75 / 0.811 / 0.231 / 18.41
- CompGS (lowrate) 26.37 / 0.778 / 0.276 / 8.83, CompGS (highrate) 27.26 / 0.803 / 0.239 / 16.50
- HAC++ (lowrate) 27.60 / 0.803 / 0.253 / 8.34, HAC++ (highrate) 27.82 / 0.811 / 0.231 / 18.48
Tanks and Temples:
- 3DGS 23.69 / 0.844 / 0.178 / 431.0, Scaffold-GS 23.96 / 0.853 / 0.177 / 86.50
- HAC 24.04 / 0.846 / 0.187 / 8.10 and 24.40 / 0.853 / 0.177 / 11.24
- ContextGS 24.20 / 0.852 / 0.184 / 7.05 and 24.29 / 0.855 / 0.176 / 11.80
- HAC++ 24.22 / 0.849 / 0.190 / 5.18 and 24.32 / 0.854 / 0.178 / 8.63
DeepBlending:
- 3DGS 29.42 / 0.899 / 0.247 / 663.9, Scaffold-GS 30.21 / 0.906 / 0.254 / 66.00
- HAC 29.98 / 0.902 / 0.269 / 4.35 and 30.34 / 0.906 / 0.258 / 6.35
- ContextGS 30.11 / 0.907 / 0.265 / 3.45 and 30.39 / 0.909 / 0.258 / 6.60
- HAC++ 30.16 / 0.907 / 0.266 / 2.91 and 30.34 / 0.911 / 0.254 / 5.28
Full Mip-NeRF360 RD curve (Appendix Table D AVG rows, p.18): λ 4e−3: 27.60 / 0.803 / 0.253 / 8.34. 3e−3: 27.65 / 0.805 / 0.249 / 9.35. 2e−3: 27.74 / 0.808 / 0.242 / 11.18. 1e−3: 27.80 / 0.810 / 0.234 / 14.73. 0.5e−3: 27.82 / 0.811 / 0.231 / 18.48.

### 10. Densification and count control
Anchor growth is Scaffold-GS spawning (enabled 1600 to 15k, paused 3k to 4k, Fig. 4 p.8). Pruning is the adaptive offset masking with the rate term in the loop: Gaussian-level mask mᵢ = sg(1[Sig(fᵐᵢ) > εₘ] − Sig(fᵐᵢ)) + Sig(fᵐᵢ) (Eq. 9, p.7), anchor-level mask mᵃᵢ from the mean offset mask (Eq. 10), masks applied to opacity and scale in rendering and to the bit count in L_entropy (Eq. 11). "we incorporate the mask into the rate calculation, allowing the mask ratio to be adaptively adjusted according to the rate via backpropagation" (p.7) and "Consequently, the mask update process dynamically identifies the optimal pruning ratio." (p.7). Effect measured in Table VIII (p.12): valid anchor ratio falls from 0.865 to 0.596 and valid Gaussian ratio from 0.314 to 0.118 as λ rises from 0.5e−3 to 4e−3. Ablation Table III (p.11): no adaptive masking +31.4 % BD-rate and FPS 125 versus 141, no anchor-level mask in the entropy loss +9.3 %, extra mask loss instead of mask-aware rate +9.6 %, no GPCC +14.0 %. This is the one place in the family where the rate term influences the primitive count directly, but only through λ, never through a count or byte target.

### 11. Stated limitations and future work (verbatim)
- p.14: "Limitation. The main limitation of HAC++ lies in its increased training time compared to the base method, Scaffold-GS, due to the additional loss term and the incorporation of context models. Future work could explore lightweight context model designs to alleviate this issue. Furthermore, HAC++ establishes relationships among anchors indirectly through an intermediate hash grid. Investigating approaches that directly model relationships among anchors could provide an alternative strategy for redundancy elimination."
- p.13: "Overall, the coding process is efficient and could be further optimized through advanced codec techniques, which we consider an engineering task for future work."
- p.14: "Overall, selecting appropriate training iterations is crucial for achieving a desirable trade-off between training time and compression performance, which can be tailored to specific applications or computational resources."

### 12. Sentences touching a target, rate control, budget or desired size (verbatim)
- p.2: "This approach adaptively determines the optimal mask ratios across different bit rate points."
- p.4: "This design enables the framework to achieve an optimal masking ratio for different rate points."
- p.7: "During training, we incorporate both the rendering fidelity loss and the entropy loss to ensure the model improves rendering quality while controlling total bitrate consumption in a differentiable manner."
- p.7: "The second part in Eq. 13 is the estimated controllable bit consumption, including the estimated bits Lentropy for anchor attributes and Lhash for the hash grid. λ is the trade-off hyperparameters used to balance the rate and fidelity."
- p.7: "By incorporating mask information into both paths' gradient chain, this approach ensures adaptive updates across different λ constraints in Eq. 13, eliminating the need for additional loss terms."
- p.8: "We change λ from 0.5e −3 to 4e −3 to achieve variable bitrates."
- p.9 (Table II caption): "FOR OUR APPROACH, WE PROVIDE TWO RESULTS WITH DIFFERENT SIZE AND FIDELITY TRADE-OFFS BY ADJUSTING λ. A SMALLER λ RESULTS IN A LARGER SIZE BUT IMPROVED FIDELITY, AND VICE VERSA."
- p.10 (Fig. 5 caption): "We vary λ to achieve variable bitrates."
- p.11: "As λ increases, stricter rate constraints lead to a decrease in the mask ratio."
- p.12: "As λ increases, the rate constraint is stricter, leading to an overall reduction in storage size."
No byte, MB, memory or FPS target, and the words "budget" and "desired" do not occur in a rate sense (only "desirable trade-off" in field 11).

### 13. Code
"Our code is available at https://github.com/YihangChen-ee/HAC-plus." (p.1). Licence: not stated.

---

## Paper 3: HEMGS (arXiv 2411.18473)

### 1. Citation
"HEMGS: A Hybrid Entropy Model for 3D Gaussian Splatting Data Compression". Lei Liu (The University of Hong Kong), Zhenghao Chen (The University of Newcastle), Wei Jiang and Wei Wang (Futurewei Technologies Inc.), Dong Xu (The University of Hong Kong). Venue not stated in the PDF (CVPR-style two-column template). arXiv:2411.18473v2 [cs.CV] 22 Apr 2025.

### 2. Base representation
Anchor-based 3DGS (Scaffold-GS anchors with location xᵃ and attributes fᵃ ∈ ℝ^Da, l ∈ ℝ⁶, o ∈ ℝ^3K, p.3). Baseline and code base is HAC ("Compared to our baseline method, HAC [3]", p.6). Rendering: "We will use all compressed anchors to render the novel-view image using the standard operations [3, 29, 39]." (p.4), i.e. the Scaffold-GS MLP renderer. Whether the hash grid or PointNet++ must be kept at render time is not stated (they are parts of the entropy model, needed for decoding). Decoding needs autoregressive arithmetic decoding in raster voxel order. The decoded model is a Scaffold-GS model, not a vanilla 3DGS point set.

### 3. Budget handle
A knob, λ, used in two roles: (a) as an input to the Variable-rate Predictor, "HEMGS takes the compressed location feature x̄ᵃ and a hyperparameter λ as inputs to generate a learned quantization step s_f via the Variable-rate Predictor" (p.3), and (b) as the loss weight (Eq. 2, p.5): L = 1/K Σₖ (L_Rendering + λₖ L_anchor). "To achieve variable-rate compression, we predefine four λ values, including 1e-3, 2e-3, 3e-3, 4e-3, to optimize the entire compression framework within a single model." (p.6). The λ values behind the Table 1 "low-rate" and "high-rate" rows are not stated. No target notion, λ is described as adjusting storage, never as hitting a size.

### 4. When it binds
During training, single model: "We use the Adam optimizer [19] and train our HEMGS for 60,000 iterations." (p.6). Staging of the losses: not stated. Wall clock (Table 6, p.8, DeepBlending, NVIDIA RTX3090): training time "1 × 1" hours for HEMGS versus "0.5 × N" hours for HAC and "5 × N" for Context-GS, N being the number of rate points, model size 0.236 MB versus 0.157 × N and 0.316 × N. The rate is selected at encode time by the λ input, after training.

### 5. What "size" means
Compressed storage in MB ("We measure the storage used for compressed 3DGS data using the metric of storage size in megabytes (MB)", p.6). Table 5 (p.8), DeepBlending, MB for Location / Feature / Scaling / Offsets / Others / Total: HAC 0.897 / 1.366 / 0.716 / 0.974 / 0.362 / 4.318 (PSNR 30.22, SSIM 0.908), Ours 0.736 / 1.244 / 0.648 / 0.640 / 0.402 / 3.670 (PSNR 30.24, SSIM 0.909). "Others" refers to additional storage costs (e.g., the parameters of MLPs), detailed in the supplementary materials (not in this PDF). Locations are 16-bit quantized then losslessly coded (p.3). The pre-trained PointNet++ is described as "introducing no additional storage overhead" (p.5), so it is apparently not counted, the paper does not say explicitly.

### 6. Runtime cost reported
Hardware: NVIDIA RTX3090 GPU (24GB memory) (p.6). Encode time: not stated. Decode time: not stated. Render FPS: not stated. GPU memory at render time: not stated. Training time: 1 hour on DeepBlending (Table 6, p.8).

### 7. Rate model, with the variable-rate mechanism in full
- Entropy model type: joint hyperprior plus autoregressive (Minnen et al. style), one Gaussian per element, estimated from concatenated hyperprior and context features (Fig. 2a, p.5). Hyperprior network: a frozen pre-trained PointNet++ (scene-agnostic) plus a scene-specific network "(e.g., hash-grid [3])" (p.5), fed with the compressed location x̄ᵃ (and the compressed feature f̄ᵃ when coding l and o). Autoregressive network: anchors coded in raster order over voxels, adaptive context selection with a 25 × 25 × 25 voxel receptive field, all elements if fewer than n, else the nearest n = 20 (p.5 to 6). Progressive coding order: location, then fᵃ, then (l, o) concatenated. MLPs are a single 2-layer ReLU MLP (p.6).
- Quantization: "adaptive quantization, as described in HAC [3]", f̂ᵃ = Round(fᵃ / s_f) × s_f with s_f the learned step (p.3). How the rounding is relaxed during training is not stated in this paper (HAC uses additive uniform noise).
- Step sizes: learned per element by the Variable-rate Predictor, separately for fᵃ and for (l, o) (p.4).
- Gap between entropy estimate and encoded bytes: not stated.
- VARIABLE-RATE mechanism: "we first expand λ by duplicating its values to match the dimensionality of the prior feature. Next, we concatenate the expanded λ with the prior feature and use MLPs to generate the learned quantization step feature s." (p.4). "a larger λ induces greater quantization, leading to reduced compressed attribute storage, whereas a smaller λ results in less quantization and increased storage." (p.4). Training: "K different values of λ are predefined as inputs to HEMGS, while each λ also serves to balance the various loss components." (p.5), loss averaged over the K = 4 values (Eq. 2). Rate selection: the user picks λ at encode time. Continuity: λ is a scalar input to an MLP, so any value can be fed, and Fig. 4 (p.8) shows "16 rate points" from one model on Tanks and Temples spanning about 6.0 to 7.6 MB and 24.4 to 24.6 dB PSNR (read from the plot axes, not tabulated). Which 16 λ values were used, and whether they lie inside or outside the trained set {1e-3, 2e-3, 3e-3, 4e-3}, is not stated. Quality loss versus single-rate: no number. The paper says "achieving performance comparable to HEMGS without using the variable-rate predictor" and "other compression methods (e.g., HAC [3], Context-GS [39]) not only require training multiple models for different storage but also exhibit slightly inferior compression performance." (p.8). The size reached for a given λ is not predictable in advance, it is read off after encoding.

### 8. Scene dependence
Per-scene tables are in the supplementary material, which is not part of this PDF. The only per-scene numbers in the PDF: Fig. 5 (p.8) "bicycle" from Mip-NeRF360: Ours 22.12 MB, PSNR 26.99, SSIM 0.8127 versus HAC 28.06 MB, PSNR 26.66, SSIM 0.8094. Table 3 (p.7) "playroom" from DeepBlending: Ours 30.75 dB at 3.31 MB. No statement about retuning per scene.

### 9. Main quantitative results (as reported, Table 1, p.6, size in MB)
Mip-NeRF360:
- 3DGS 27.49 / 0.813 / 0.222 / 744.7
- Scaffold-GS 27.50 / 0.806 / 0.252 / 253.9
- HAC 27.77 / 0.811 / 0.230 / 21.87
- Context-GS 27.72 / 0.811 / 0.231 / 21.58
- CompGS 27.26 / 0.803 / 0.239 / 16.50
- Ours (low-rate) 27.68 / 0.809 / 0.239 / 12.52
- Ours (high-rate) 27.89 / 0.815 / 0.226 / 17.56
Tanks and Temples:
- 3DGS 23.69 / 0.844 / 0.178 / 431.0, Scaffold-GS 23.96 / 0.853 / 0.177 / 86.50
- HAC 24.40 / 0.853 / 0.177 / 11.24
- Context-GS 24.29 / 0.855 / 0.176 / 11.80
- Ours (low-rate) 24.41 / 0.854 / 0.183 / 6.13
- Ours (high-rate) 24.62 / 0.857 / 0.180 / 7.46
DeepBlending:
- 3DGS 29.42 / 0.899 / 0.247 / 663.9, Scaffold-GS 30.21 / 0.906 / 0.254 / 66.00
- HAC 30.34 / 0.906 / 0.258 / 6.35
- Context-GS 30.39 / 0.909 / 0.258 / 6.60
- Ours (low-rate) 30.24 / 0.909 / 0.258 / 3.67
- Ours (high-rate) 30.40 / 0.911 / 0.255 / 4.25
Synthetic-NeRF: Ours (low-rate) 33.33 / 0.967 / 0.038 / 1.18, Ours (high-rate) 33.82 / 0.968 / 0.034 / 1.62.
BDBR (Table 2, p.7, positive = more storage than HEMGS at equal PSNR): Context-GS 66.47 % (Mip-NeRF360), 42.03 % (DeepBlending). HAC 52.52 %, 82.85 %. Text (p.6): "our HEMGS saves 46.43% storage at 24.4 PSNR on the Tank&Temples dataset [20] and 39.34% storage at 30.3 PSNR on the DeepBlending dataset [15] compared to HAC."
Full Mip-NeRF360 RD curve points: not tabulated, only Fig. 3 (p.7), where the HEMGS curve spans about 14 to 22 MB and 27.5 to 27.9 dB. For reference, PCGS Table VII (PCGS PDF p.15) lists HEMGS on Mip-NeRF360 as 27.74 / 0.807 / 0.249 / 11.96 and 27.94 / 0.813 / 0.230 / 20.03 (reported there by the PCGS authors).

### 10. Densification and count control
Not stated. The paper inherits the anchor structure of Scaffold-GS and says L_anchor includes "some auxiliary items used in the compression framework as in HAC [3]" (p.5) with details deferred to the supplementary material, which is not in this PDF. No mask or pruning mechanism is described, and no statement links the rate term to the anchor count.

### 11. Stated limitations and future work (verbatim)
- p.8: "Limitations and Future Works. Currently, our method HEMGS applies only anchor-based 3DGS method, which requires the anchor structure to distribute local 3D Gaussians. In our future work, we plan to extend it to anchor-free data structures to broaden its applicability."
- p.8: "Our work also paves the way for further research into comprehensive models that address a wide range of compression scenarios, from extreme to high-quality compression."

### 12. Sentences touching a target, rate control, budget or desired size (verbatim)
- p.1: "Second, these methods compress 3DGS into a single storage format, significantly limiting their flexibility and applicability in real-world scenarios that require varying storage options due to different conditions (e.g., network bandwidth)."
- p.1: "These methods achieve multi-rate compression by applying rate-distortion (RD) optimizations with different rate-control hyperparameters, resulting in multiple models corresponding to multiple specific rates."
- p.1: "First, for multi-rate lossy compression, using multiple separate models significantly increases training and storage overhead, especially when many compression rates are required."
- p.2: "First, for variable-rate lossy compression of each attribute, we introduce a Variable-rate Predictor that accepts a hyperparameter λ to adjust storage."
- p.2: "By enabling multi-rate compression with a single model, we reduce both the training overhead and model storage burden, making our lossy compression procedure more versatile for real-world applications with varying storage requirements."
- p.4: "The adaptive quantization process is defined as f̂ᵃ = Round(fᵃ/s_f) × s_f and the with the storage adjusted according to the user-specified λ."
- p.4: "This approach enables users to adjust storage requirements using a single model controlled by the variable hyperparameter λ. Specifically, a larger λ induces greater quantization, leading to reduced compressed attribute storage, whereas a smaller λ results in less quantization and increased storage."
- p.5: "To enable variable-rate encoding within a single model, K different values of λ are predefined as inputs to HEMGS, while each λ also serves to balance the various loss components."
- p.6: "To achieve variable-rate compression, we predefine four λ values, including 1e-3, 2e-3, 3e-3, 4e-3, to optimize the entire compression framework within a single model."
- p.8: "Across 16 rate points, our single HEMGS model demonstrates effective variable-rate adjustment while achieving performance comparable to HEMGS without using the variable-rate predictor."
- p.8: "Given that different scenarios require varying levels of compression, a versatile compression method should support multiple rates."
The words "target", "budget" and "desired" do not occur. Storage is "adjusted" through λ, never set to a number.

### 13. Code
Not stated, no URL in the PDF. (SALVQ, p.6, states "we exclude HEMGS [16] (no released code)".) Licence: not stated.

---

## Paper 4: SALVQ (arXiv 2509.13482)

### 1. Citation
"Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization". Hao Xu (McMaster University), Xiaolin Wu (Southwest Jiaotong University, corresponding), Xi Zhang (Tongji University). Venue not named in the PDF (IEEE journal template header "JOURNAL OF LATEX CLASS FILES", file name says TIP 2025, the paper states it "is based on Chapter 4 of the Ph.D. thesis of Hao Xu [1]"). arXiv:2509.13482v2 [cs.CV] 20 May 2026.

### 2. Base representation
A drop-in quantizer for anchor-based (Scaffold-GS) compression pipelines, evaluated inside HAC, HAC++, ContextGS and PCGS. SALVQ replaces uniform scalar quantization of the anchor latent feature f (dimension 50) by lattice vector quantization with a learned per-scene basis B = UΣVᵀ. Rendering is unchanged and uses the Scaffold-GS MLPs: "rendering is performed after decompression on the recovered 3DGS parameters, meaning that the quantizer is not involved during rendering and the rendering pipeline remains unchanged." (p.9). The decoded model is the host's Scaffold-GS model, not a vanilla 3DGS point set.

### 3. Budget handle
A knob. Single-rate: λ in L = L_distortion + λ L_rate + λ_reg L_reg (Eq. 5, p.5), "The Lagrange multiplier λ is varied over the set {0.002, 0.004, 0.008, 0.015, 0.025} to evaluate compression performance across a wider range." (p.6). Variable-rate (VBR): a learned gain vector g = [g₁, ..., g_M] with M "target rates", each paired with a Lagrange multiplier λᵢ, M = 4 with λ ∈ {0.002, 0.004, 0.006, 0.008} (p.12). Piecewise VBR: a high-rate model with λ ∈ {0.002, 0.004, 0.006, 0.008} and a low-rate model with λ ∈ {0.008, 0.012, 0.016, 0.02} (p.14). The paper uses the words "target rate" and "desired bitrate" throughout, but every "target" is a λ-indexed operating point whose byte size is read off after training. Nothing takes a size in MB as input.

### 4. When it binds
During training from the host's schedule: "we train each scene for 30k iterations for HAC [14], HAC++ [17], and ContextGS [18], and for 40k iterations for PCGS [57]." (p.6). VBR training: "At each training iteration, we randomly sample an index from {1, · · · , M}, apply gi to modulate the step, and use λi in the loss. After training, the anchor attributes and qs remain fixed; at inference one simply selects a learned gain gi to meet the desired bitrate." (p.6). Wall clock (Table VII, p.8, seconds, USQ then SALVQ): HAC 1949 / 2206 (Mip-NeRF360), 1385 / 1701 (Tanks and Temples), 1533 / 1766 (DeepBlending). HAC++ 2735 / 2937, 1880 / 2055, 1857 / 2060. ContextGS 3927 / 4167, 2505 / 2639, 2320 / 2375. Overhead "ranging from 55 to 316 seconds" (p.9). Hardware not named.

### 5. What "size" means
"We use the size of the encoded bitstream (in MB) as the rate metric." (p.6). The lattice basis adds 50² + 50² + 50 = 5050 float32 parameters, "a memory overhead of merely 0.02 MB" (p.8 to 9), plus 78 parameters when applied to scaling. Component breakdown otherwise inherited from the host (HAC, HAC++, ContextGS, PCGS).

### 6. Runtime cost reported
Hardware: not named (only "GPU vRAM").
- Rendering FPS (Table III, p.8, Mip-NeRF360 / Tanks and Temples / DeepBlending): HAC USQ 131 / 146 / 179, HAC SALVQ 136 / 139 / 173, HAC FS-SALVQ 136 / 149 / 199. HAC++ USQ 128 / 158 / 169, SALVQ 143 / 164 / 182. ContextGS USQ 106 / 130 / 161, SALVQ 104 / 130 / 169, Sp-SALVQ 102 / 129 / 194.
- Encoding time in seconds (Table IV, p.8): HAC USQ 4.20 / 2.51 / 1.33, SALVQ 4.56 / 2.80 / 1.49. HAC++ USQ 8.48 / 5.63 / 2.98, SALVQ 8.37 / 5.30 / 2.91. ContextGS USQ 32.73 / 29.37 / 16.78, SALVQ 31.93 / 31.64 / 12.59.
- Decoding time in seconds (Table V, p.8): HAC USQ 10.05 / 5.77 / 2.85, SALVQ 10.76 / 6.43 / 3.11. HAC++ USQ 13.86 / 8.81 / 4.41, SALVQ 13.81 / 8.61 / 4.42. ContextGS USQ 33.32 / 29.05 / 16.63, SALVQ 31.84 / 28.80 / 12.49.
- Training GPU vRAM in MB (Table VI, p.8): HAC USQ 8228 / 4130 / 5729, SALVQ 10197 / 5493 / 8175. HAC++ USQ 9732 / 4026 / 6361, SALVQ 11654 / 6534 / 7681. ContextGS USQ 11738 / 5841 / 6789, SALVQ 12012 / 5958 / 6884.
- GPU memory at render time: not stated.

### 7. Rate model, with the variable-rate mechanism in full
- Entropy model: inherited from the host. Inside HAC (hyperprior only): mean removal with the hyperprior-predicted μ, f_c = f − μ, transform f_t = B⁻¹ f_c, rounding (Babai rounding technique, Eq. 3, p.5), probability p(f̂_t) = Πᵢ [N(0, σᵢ²) ∗ U(−q_s/2, q_s/2)](f̂_t^(i)) (Eq. 4, p.5), reconstruction f̂ = B f̂_t + μ. Inside ContextGS and HAC++: a global mean μ_g for one-pass quantization, the original non-zero-mean Gaussian entropy models with hyperprior plus causal context (p.5).
- Quantization relaxation: "a rounding operation, which is approximated by adding uniform noise during training and replaced by hard quantization during inference" (p.5).
- Step sizes: the host's per-attribute step q_s (predicted from the hash grid in HAC), SALVQ learns the basis B per scene, the gain vector is "learned separately for anchor latent features, scaling factors, and offsets" (footnote 1, p.5). SALVQ applied to f by default, FS-SALVQ also to l, not to offsets because masking makes their dimension variable (p.5).
- Gap between entropy estimate and encoded bytes: not stated.
- VARIABLE-RATE mechanism ("Rate control scheme", Sec. IV-E, p.5 to 6): one model, M learned gains gᵢ scale the shared base step q_s to gᵢ q_s ("A larger gain leads to a coarser LVQ and a lower bitrate"). Each gain is tied to a λᵢ during training, one index sampled per iteration. Selection at inference is discrete: pick one of the M = 4 learned gains. No continuous interpolation is described. Quality loss versus single-rate models (Table XI, p.12, BD-rate relative to the host's USQ single-rate baseline, positive = worse): HAC USQ-VBR +4.54 % / −0.99 % / +1.74 % (Mip-NeRF360 / Tanks and Temples / DeepBlending), HAC SALVQ-VBR −6.44 % / −13.83 % / −16.89 %. ContextGS USQ-VBR +2.49 % / +9.87 % / +1.94 %, SALVQ-VBR −5.01 % / −4.65 % / −18.04 %. HAC++ USQ-VBR +14.99 % / +18.89 % / +4.94 %, SALVQ-VBR +4.16 % / +4.62 % / −8.30 %. Range limitation: "a single gain-controlled VBR model is not yet sufficient to cover a broad operating range" (p.12). Extensions: piecewise VBR with two models covering the five λ points (2.5× fewer trainings, p.14), and PCGS-style primitive count as a second rate dimension, where SALVQ-PCGS beats USQ-PCGS by −3.47 % / −20.57 % / −7.01 % BD-rate (p.14).

### 8. Scene dependence
Table XX (p.19), HAC++ + SALVQ on Mip-NeRF360, sizes in MB at λ = 0.002: bicycle 17.50, bonsai 6.91, counter 6.24, flowers 14.38, garden 17.10, kitchen 7.09, room 4.56, stump 12.42, treehill 14.52 (AVG 11.19). At λ = 0.025: bicycle 4.74, room 1.47 (AVG 3.24). Table XIV (p.17), HAC + SALVQ at λ = 0.002: bicycle 32.93, room 5.86, garden 26.16 (AVG 17.68). Per-scene basis learning is the whole point: "3DGS compression is scene-specific; learning the lattice basis per scene can better align the quantizer with the true source statistics" (p.7). No statement about retuning λ per scene.

### 9. Main quantitative results (as reported, Table X, p.12, all at λ = 0.004, size in MB)
Mip-NeRF360:
- 3DGS 27.46 / 0.812 / 0.222 / 750.9, Scaffold-GS 27.50 / 0.806 / 0.252 / 253.9
- HAC 27.53 / 0.807 / 0.238 / 15.26, HAC + SALVQ 27.60 / 0.807 / 0.239 / 14.30, HAC + FS-SALVQ 27.61 / 0.807 / 0.239 / 14.23
- Context-GS 27.62 / 0.808 / 0.237 / 12.68, + SALVQ 27.62 / 0.808 / 0.238 / 12.12, + Sp-SALVQ 27.65 / 0.808 / 0.237 / 12.15
- HAC++ 27.60 / 0.803 / 0.253 / 8.34, HAC++ + SALVQ 27.61 / 0.803 / 0.252 / 8.31
Tanks and Temples:
- HAC 24.04 / 0.846 / 0.187 / 8.10, + SALVQ 24.20 / 0.847 / 0.186 / 7.27
- Context-GS 24.20 / 0.852 / 0.184 / 7.05, + SALVQ 24.35 / 0.852 / 0.184 / 6.81
- HAC++ 24.22 / 0.849 / 0.190 / 5.18, + SALVQ 24.26 / 0.849 / 0.190 / 5.22
DeepBlending:
- HAC 29.98 / 0.902 / 0.269 / 4.35, + SALVQ 30.06 / 0.903 / 0.268 / 3.98
- Context-GS 30.11 / 0.907 / 0.258 / 3.45, + SALVQ 30.14 / 0.907 / 0.266 / 3.30
- HAC++ 30.16 / 0.907 / 0.266 / 2.91, + SALVQ 30.18 / 0.907 / 0.266 / 2.87
BD-rate of SALVQ versus USQ (Table I, p.7): HAC −13.48 % / −16.16 % / −13.44 %, HAC++ −4.55 % / −8.95 % / −7.75 %, ContextGS −5.71 % / −8.69 % / −9.75 % (Mip-NeRF360 / Tanks and Temples / DeepBlending). "average savings ranging from 4.55% to 16.16%" (p.6).
Full Mip-NeRF360 RD curves (AVG rows of the supplementary tables):
- HAC++ + SALVQ (Table XX, p.19): λ 0.002: 27.76 / 0.808 / 0.241 / 11.19. 0.004: 27.61 / 0.803 / 0.252 / 8.31. 0.008: 27.40 / 0.796 / 0.266 / 6.08. 0.015: 27.15 / 0.786 / 0.283 / 4.36. 0.025: 26.86 / 0.773 / 0.303 / 3.24.
- HAC + SALVQ (Table XIV, p.17): 0.002: 27.77 / 0.810 / 0.233 / 17.68. 0.004: 27.60 / 0.807 / 0.239 / 14.30. 0.008: 27.34 / 0.800 / 0.249 / 11.38. 0.015: 27.02 / 0.791 / 0.264 / 9.36. 0.025: 26.68 / 0.779 / 0.280 / 8.15.
- ContextGS + SALVQ (Table XVII, p.18): 0.002: 27.71 / 0.809 / 0.233 / 14.77. 0.004: 27.62 / 0.808 / 0.238 / 12.12. 0.008: 27.45 / 0.805 / 0.244 / 9.74. 0.015: 27.27 / 0.800 / 0.254 / 7.82. 0.025: 27.03 / 0.793 / 0.266 / 6.57.

### 10. Densification and count control
Inherited unchanged from the host pipelines (HAC, HAC++, ContextGS, PCGS). SALVQ is not applied to offsets precisely because the host's offset masking makes the number of active offsets vary per anchor (p.5). No new count mechanism.

### 11. Stated limitations and future work (verbatim)
- p.12: "c) Limitation of a single VBR model and two extensions: The above results also reveal a practical limitation of the current single-model VBR formulation: regardless of whether SALVQ or USQ is used, a single gain-controlled VBR model is not yet sufficient to cover a broad operating range."
- p.12: "In the current single-model setting, a fixed training budget must be shared across all target rates. As the number of supported operating points increases, the effective optimization budget allocated to each rate decreases, which can noticeably degrade the overall R-D performance."
- p.9: "This suggests that SALVQ can still provide clear compression gains even with a compact low-dimensional latent representation, whereas more aggressive dimensionality reduction limits high-bitrate, high-fidelity reconstruction due to reduced feature capacity."
- p.6: "We will include these additional baselines once stable, accessible implementations are available."
- Section V (p.6) promises to "conclude with limitations and directions for future work", but no dedicated future-work paragraph exists beyond the sentences above.

### 12. Sentences touching a target, rate control, budget or desired size (verbatim)
- p.1 (abstract): "Moreover, by scaling the lattice basis vectors, SALVQ can dynamically adjust lattice density, enabling a single model to accommodate multiple bit rate targets. This flexibility eliminates the need to train separate models for different compression levels, significantly reducing training time and memory consumption."
- p.2: "The ability to support variable-rate 3DGS compression, offering flexibility in bitrate control while delivering high-quality reconstruction results."
- p.4: "Finally, we develop a rate-control scheme that enables our LVQ-based 3DGS compression system to operate in variable-rate mode."
- p.5: "For a specific λ, the 3DGS model is optimized to achieve the corresponding rate target. For each choice of λ, the 3DGS model is trained to target a specific rate, and once training is complete, the model operates only at that rate. Supporting multiple R-D trade-offs requires training separate models with different λ values, leading to computational and memory costs that grow linearly with the number of desired rates."
- p.5: "Motivated by this, we fix the anchor attributes' dynamic range and learn rate-specific lattice densities to control the R–D trade-off."
- p.5 to 6: "Specifically, we introduce a gain vector g = [g1, · · · , gM]1, where M is the number of target rates, to control the lattice density for each rate. The anchor attributes and the base quantization step qs are shared across all targets; target i uses its gain gi to scale the quantization step from qs to giqs. A larger gain leads to a coarser LVQ and a lower bitrate, enabling flexible rate control without retraining the model. To learn the gain vector, we associate each target with a Lagrange multiplier λi. At each training iteration, we randomly sample an index from {1, · · · , M}, apply gi to modulate the step, and use λi in the loss. After training, the anchor attributes and qs remain fixed; at inference one simply selects a learned gain gi to meet the desired bitrate."
- p.6: "When using this rate control scheme with either SALVQ or USQ, the distortion at each target rate is primarily determined by the quantization error."
- p.10: "Since these methods typically operate at only one or two bitrate targets, we cannot directly compare with them using R-D curves and BD-rate calculations."
- p.12: "Overall, with four target rates, SALVQ-VBR achieves a better trade-off between rate flexibility and coding performance, while reducing the number of trained models, total training time, and stored model parameters by 4× compared with training four separate single-rate models."
- p.12: "The second is to treat the number of transmitted Gaussian/anchor primitives as an additional rate-control dimension, jointly with quantization granularity, as exemplified by progressive compression methods such as PCGS [57]."
- p.14: "The USQ-based single-rate baseline is trained at five target rates corresponding to λ ∈{0.002, 0.004, 0.008, 0.015, 0.025}."
- p.14: "Overall, when a variable-rate 3DGS compression model is required, SALVQ is a better quantizer than USQ, whether the desired bitrate variation lies within a narrow operating range or over an extended operating range. This conclusion holds whether rate control is achieved solely through quantization granularity, through a piecewise VBR strategy with multiple region-specific models, or through the joint use of primitive transmission and quantization control as in PCGS."
- p.14: "Furthermore, by adaptively scaling the lattice basis vectors, SALVQ provides effective variable-rate control, often matching or surpassing the R–D performance of USQ-based single-rate models, while eliminating the need to train separate models for different R–D targets."
Every "target" and "desired bitrate" in this paper is a λ-indexed operating point. No sentence takes a byte or MB number as the input.

### 13. Code
Not stated, no repository URL in the PDF. Licence: not stated.

---

## Paper 5: PCGS (arXiv 2503.08511)

### 1. Citation
"PCGS: Progressive Compression of 3D Gaussian Splatting". Yihang Chen (Shanghai Jiao Tong University and Monash University) and Mengyao Li (Shanghai University and Monash University), equal contribution, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, Jianfei Cai. Venue not stated in the PDF (CVPR-style template). arXiv:2503.08511v1 [cs.CV] 11 Mar 2025.

### 2. Base representation
HAC++ (Scaffold-GS anchors, binary hash grid, GPCC for locations): "PCGS is implemented using the PyTorch [32] framework, building upon the HAC++ [10] repository." (p.6). Rendering uses the Scaffold-GS MLPs. Decoding is level by level: header (MLPs, anchor locations xᵃ via GPCC, hash grid H, Gaussian-level masks mᵍ), then per level Gaussian-distribution decoding of new anchors and trinomial decoding of refinements (p.7). Whether the hash grid can be dropped after decoding is not stated here (it can in HAC++). The decoded model is a Scaffold-GS model, not a vanilla 3DGS point set.

### 3. Budget handle
A set of knobs, one λ_s per progressive level s, plus the choice of how many levels to decode. Loss (Eq. 10, p.6): Loss = L_Scaffold + λ_s · 1/(N(Dₐ + 6 + 3K)) · (L_entropy + L_hash), "the trade-off parameter λs varies with s to balance the R-D performance at different levels." λ_s is also an input to the rate-aware MLP: q_s, μ_s, σ_s = MLP(fʰ, λ_s) (Eq. 5, p.5). Values used (supplementary tables, p.11 to 14): Tanks and Temples, DeepBlending, BungeeNeRF λ_s ∈ {8e−4, 4e−4, 0.5e−4} (3 levels), Mip-NeRF360 {4e−4, 2.5e−4, 1e−4, 0.2e−4} (4 levels), Synthetic-NeRF {4e−4, 2e−4, 0.25e−4}. Rate selection after training is discrete: transmit the header plus the first s levels. No byte target, the sizes per level are read off after encoding.

### 4. When it binds
During training, single training run: "we employ a set of λs to train the model progressively within a single training process. At each iteration, we randomly sample a level s and its corresponding λs from this set for training. Since all rates are covered within one training session, we appropriately extend the training iterations to 40k to ensure sufficient optimization. Additionally, we also present the results trained by 30k iterations for a fair comparison with the standard protocol." (p.6). Other stages inherited from HAC++ ("The hyperparameters remain consistent with HAC++"). Wall clock, hardware not named (supplementary Tables I to VI, seconds, 40k then 30k): Mip-NeRF360 4044 / 2704, Tanks and Temples 2447 / 1675, DeepBlending 2214 / 1670, BungeeNeRF 3946 / 2807, Synthetic-NeRF 1215 / 843.

### 5. What "size" means
MB of the bitstream, cumulative over levels: "For subsequent levels (i.e., s ≥2), the size is calculated as the sum of the previous level's size and the incremental size (i.e., ∆Size in the tables) of the current level." (p.11). Level 1 includes the header: "the encoding process also includes storing and coding of header information, i.e., MLPs, anchor locations xa, the hash grid H, and Gaussian-level masks mg (from which the anchor-level masks ma can be derived). This results in longer encoding and decoding times and larger sizes at the first level." (p.11). Fig. 6 (p.8), MB: room header 1.27, levels 3.73, 1.79, 2.06, 2.25 (295157 anchors). amsterdam header 2.60, levels 12.89, 5.59, 6.66 (735395 anchors). truck header 1.31, levels 4.89, 2.03, 2.67 (365307 anchors). All valid anchor locations are coded in the header, not incrementally (p.7). MLP overhead on Synthetic-NeRF "approximately 0.5 MB" (p.11).

### 6. Runtime cost reported
Hardware: not named. FPS: not stated. GPU memory: not stated.
Encoding and decoding times in seconds, incremental per level (supplementary Table IV, p.13, Mip-NeRF360 40k AVG): level 1 enc 13.0, dec 20.5. Level 2 2.4, 2.6. Level 3 2.5, 2.8. Level 4 2.8, 3.4. Tanks and Temples (Table I, p.11): 6.4, 9.6 then 1.3, 1.4 then 1.5, 1.7. DeepBlending (Table II, p.12): 3.5, 4.9 then 0.7, 0.8 then 0.9, 1.1. BungeeNeRF (Table III, p.12): 13.3, 19.0 then 2.6, 2.9 then 3.2, 3.9.

### 7. Rate model, with the variable-rate mechanism in full
- Entropy model: HAC++ hash-grid context with a rate-aware MLP conditioned on the level, q_s, μ_s, σ_s = MLP(fʰ, λ_s) (Eq. 5, p.5). Whether HAC++'s intra-anchor GMM context is kept is not stated. First decoding of an anchor at level s: Round with step q_s, Gaussian probability over the bin (Eq. 6, p.5), "f̂s is obtained by adding noise to f during training to preserve gradient flow, and Round during testing". Refinement at later levels: trit-plane quantization, the previous bin [f̂_{s−1} − q_{s−1}/2, f̂_{s−1} + q_{s−1}/2) is split into three, q_s = q_{s−1}/3, and a trinomial distribution {p_sc}_{c=1,2,3} = MLP(f̂_{s−1}, fʰ, λ_s) (Eq. 7, 8, p.5). To keep training path-independent the first-decode step is defined as q_s = q₁ / 3^(s−1) (p.6). Offsets are never refined, "only the quantity increases" (p.6). Incremental entropy loss (Eq. 9, p.6) weights new anchors by mᵃ_s Δmᵃ_s (Gaussian), refined anchors by mᵃ_s (1 − Δmᵃ_s) (trinomial), and new offsets by Δmᵍ_{s,k}. Using accumulated rate over all levels, or adding earlier-level fidelity, hurts (ablation, p.8).
- Step sizes: learned per anchor from the hash feature and λ_s, per attribute type through HAC++'s Q₀.
- Gap between entropy estimate and encoded bytes: not stated.
- VARIABLE-RATE mechanism: progressive levels. One training produces one bitstream whose prefixes are valid models. Each level adds anchors and Gaussians (monotone non-decreasing masks, field 10) and refines existing anchors by one trit-plane. Selection is discrete (3 or 4 levels), not continuous. Quality versus single-rate: "our PCGS, despite being trained only once to obtain the progressive R-D curve, achieves compression performance comparable to SoTA single-rate methods." (p.6). Numbers, Mip-NeRF360, Table VII (p.15): PCGS 27.82 dB at 12.05 MB and 27.96 dB at 24.74 MB versus HAC++ 27.60 at 8.34 and 27.82 at 18.48. At the 30k protocol (Table VI, p.14) PCGS levels give 27.69 / 12.64 MB, 27.81 / 16.68, 27.83 / 21.22, 27.83 / 26.17, against HAC++ 30k (HAC++ paper) 27.74 / 11.18 and 27.80 / 14.73, so at equal iterations PCGS pays some size for progressivity at the low end. No BD-rate versus HAC++ is given. Versus GoDe (30k + 30k) PCGS is better, versus HEMGS (60k per rate) "PCGS surpasses HEMGS on most datasets, even with fewer training iterations" (p.7).

### 8. Scene dependence
Supplementary Table IV (p.13), Mip-NeRF360, 40k, level 1 (λ_s = 4e−4) size in MB: bicycle 19.09, garden 18.27, stump 13.49, room 5.00, counter 6.80, kitchen 7.67, bonsai 7.44, flowers 15.73, treehill 16.24 (AVG 12.19). Cumulative level 4: bicycle 39.73, room 11.10, AVG 25.34. Fig. 6 (p.8) per-scene mask ratios: room r(mᵃ) 0.60 to 0.65 and r(mᵍ) 0.113 to 0.131 over 4 levels, amsterdam 0.72 to 0.76 and 0.154 to 0.173, truck 0.68 to 0.71 and 0.162 to 0.174. No retuning statement, but the λ_s sets differ per dataset (field 3).

### 9. Main quantitative results (as reported, Table VII, p.15, size in MB, two rows per method = two rate points)
Mip-NeRF360:
- 3DGS 27.46 / 0.812 / 0.222 / 750.9, Scaffold-GS 27.50 / 0.806 / 0.252 / 253.9
- HAC++ 27.60 / 0.803 / 0.253 / 8.34 and 27.82 / 0.811 / 0.231 / 18.48
- HEMGS 27.74 / 0.807 / 0.249 / 11.96 and 27.94 / 0.813 / 0.230 / 20.03
- CAT-3DGS 25.82 / 0.730 / 0.362 / 1.72 and 27.77 / 0.809 / 0.241 / 12.35
- ContextGS 27.62 / 0.808 / 0.237 / 12.68 and 27.75 / 0.811 / 0.231 / 18.41
- GoDe 27.07 / 0.780 / 0.336 / 7.5 and 27.75 / 0.810 / 0.284 / 19.7
- Our PCGS 27.82 / 0.808 / 0.240 / 12.05 and 27.96 / 0.811 / 0.236 / 24.74
Tanks and Temples:
- HAC++ 24.22 / 0.849 / 0.190 / 5.18 and 24.32 / 0.854 / 0.178 / 8.63
- HEMGS 24.40 / 0.848 / 0.192 / 5.75 and 24.55 / 0.856 / 0.176 / 9.67
- Our PCGS 24.501 / 0.852 / 0.186 / 5.67 and 24.59 / 0.856 / 0.181 / 9.87
DeepBlending:
- HAC++ 30.16 / 0.907 / 0.266 / 2.91 and 30.34 / 0.911 / 0.254 / 5.28
- HEMGS 30.24 / 0.909 / 0.270 / 2.85 and 30.40 / 0.912 / 0.258 / 6.39
- Our PCGS 30.19 / 0.907 / 0.264 / 3.26 and 30.38 / 0.910 / 0.259 / 5.81
Full Mip-NeRF360 progressive curve (supplementary Table IV AVG, 40k, p.13, PSNR / SSIM / LPIPS / ΔSize / cumulative size): level 1 λ_s 4e−4: 27.81 / 0.8079 / 0.2404 / 12.19 / 12.19. Level 2 2.5e−4: 27.93 / 0.8103 / 0.2377 / 3.95 / 16.14. Level 3 1e−4: 27.95 / 0.8109 / 0.2366 / 4.42 / 20.56. Level 4 0.2e−4: 27.96 / 0.8111 / 0.2361 / 4.79 / 25.34. The Table VII PCGS rows (27.82 / 12.05 and 27.96 / 24.74) differ slightly from the Table IV averages (27.81 / 12.19 and 27.96 / 25.34), the paper does not comment. 30k protocol (Table VI, p.14): 27.69 / 0.8082 / 0.2370 / 12.64, 27.81 / 0.8105 / 0.2346 / 16.68, 27.83 / 0.8110 / 0.2340 / 21.22, 27.83 / 0.8111 / 0.2336 / 26.17.

### 10. Densification and count control
Anchor spawning inherited from HAC++ / Scaffold-GS (not discussed). Count control is the "rate-aware progressive masking": Gaussian-level mask mᵍ_s = 1[Sig(fᵐ_base + Σ_{l=1..s} Sfp(fᵐ_l)) > εₘ] (Eq. 3, p.4) with learnable base and per-level features, softplus making the mask monotonically non-decreasing over levels so a decoded Gaussian stays valid, STE for gradients. Anchor-level mask mᵃ_s = at least one valid Gaussian (following HAC++). Δmᵃ_s = mᵃ_s − mᵃ_{s−1} flags newly decoded anchors (Eq. 4). The masks enter the incremental entropy loss (Eq. 9) as in HAC++, so the rate weight λ_s at each level drives how many anchors become valid at that level. "Disabling progressive masking (W/O Prog Mask, i.e., keeping the mask identical across levels) prevents the model from improving fidelity in the high-rate segments" (p.7). SALVQ (p.12) describes this as treating "the number of transmitted Gaussian/anchor primitives as an additional rate-control dimension". Still no count or byte target.

### 11. Stated limitations and future work (verbatim)
No limitation or future-work section exists. Related statements:
- p.6 to 7: "Notably, additional training iterations do not improve performance on the BungeeNeRF dataset due to its scale discrepancy between training and testing views, which lead to overfitting."
- p.11: "While this dataset is less suitable for progressive compression due to its limited size, we include it to ensure the completeness of our evaluations. Notably, the overhead introduced by MLPs (approximately 0.5 MB for this dataset) becomes more pronounced in this dataset."
- Future work: not stated.

### 12. Sentences touching a target, rate control, budget or desired size (verbatim)
- p.1: "Despite these advancements, existing methods exhibit two key limitations: (1) Single-rate constraint: Each bitstream from these compression models is only for a fixed rate/data size; and (2) Lack of progressivity: Although these models can be retrained to produce bitstreams for different rate targets, the resulting bitstreams are independent and non-reusable, as illustrated in the upper part of Fig. 1."
- p.1: "For instance, as transmission bandwidth increases, bitstreams with higher rate budgets can be transmitted for better rendering fidelity. However, due to their single-rate and non-progressive nature of existing 3DGS compression methods, generating a new bitstream requires retraining the model, where existing bitstreams cannot be reused, leading to inefficient resource utilization."
- p.1 (Fig. 1 caption): "Existing approaches generate multiple independent bitstreams targeting different rates and fidelity through multiple trainings, while a progressive compression approach (with only one training) can continuously improve the fidelity by incrementally adding bitstreams, which is resource-saving in on-demand applications."
- p.2: "Despite the advancements, these methods remain single-rate, requiring separate trainings and generating independent bitstreams for different rate-distortion (R-D) trade-offs."
- p.3: "In this work, we improve HAC++ by introducing progressive compression, further expanding its applicability to varying bandwidth and storage constraints."
- p.3: "For quantity, we introduce a rate-aware progressive masking strategy, as depicted in the lower-left part of Fig. 2."
- p.6: "where LScaffold and Lhash are the same as those defined in HAC++ [10] and the trade-off parameter λs varies with s to balance the R-D performance at different levels."
- p.6: "More importantly, PCGS enables compression into different progressive bitstreams, efficiently adapting to dynamic network and storage constraints."
- p.8: "More importantly, the progressive nature of PCGS makes it well-suited for on-demand applications, where dynamic bandwidth and diversion storage conditions often arise, significantly broadening the applicability of 3DGS."
- p.11: "Benefiting from this progressive design, on-demand applications require only the incremental bits and encoding/decoding times for levels s ≥2, underscoring the advantages of the progressive compression pipeline."
"rate budgets" (p.1) describes a receiver's bandwidth, not a training input. No sentence sets a byte, MB, memory or FPS number during training, and "desired" occurs only for "the desired f̂s can be decoded" (p.5).

### 13. Code
"Code available at: github.com/YihangChen-ee/PCGS." (p.1). Licence: not stated.

---

## Cross-paper facts for the gap question

- Handle in all five: a Lagrange multiplier λ (HAC λₑ 5e−4 to 4e−3 with λₘ 5e−4, HAC++ λ 0.5e−3 to 4e−3, HEMGS λ ∈ {1e−3, 2e−3, 3e−3, 4e−3} as an MLP input, SALVQ λ ∈ {0.002, 0.004, 0.008, 0.015, 0.025} with learned gains for M = 4 of them, PCGS one λ_s per level, for example {4e−4, 2.5e−4, 1e−4, 0.2e−4}). The rate loss is normalised per parameter (division by N(Dₐ + 6 + 3K)) in HAC, HAC++ and PCGS, so the loss never sees a total byte count.
- The size produced by one λ is scene dependent by 3.8× to 5× inside Mip-NeRF360 alone (HAC λₑ = 0.004: bicycle 27.54 MB, room 5.53 MB. HAC++ λ = 4e−3: bicycle 12.84 MB, room 3.41 MB. HAC++ + SALVQ λ = 0.002: bicycle 17.50, room 4.56). The headline "lowrate" and "highrate" rows in HAC and HAC++ are picked from different λ per dataset, and PCGS uses different λ_s sets per dataset.
- Variable-rate papers (HEMGS, SALVQ, PCGS) remove the retraining cost per rate point, but the operating points are still λ-indexed (HEMGS continuous λ input with 4 trained values and an untabulated 16-point sweep, SALVQ 4 discrete learned gains, PCGS 3 to 4 discrete levels). None maps a requested size to a λ or gain. SALVQ states that a single VBR model cannot cover a broad range and that the per-rate optimisation budget shrinks with the number of operating points.
- Count control tied to rate exists in HAC++ (mask-aware rate) and PCGS (rate-aware progressive masking), always through λ.
- Render-time MLP: yes for all five (Scaffold-GS neural Gaussians). The hash grid is discarded after decoding in HAC and HAC++. Reported FPS: HAC 283 FPS BungeeNeRF (RTX 4090), HAC++ 130 to 151 FPS Mip-NeRF360 (L40s), SALVQ hosts 102 to 199 FPS (GPU unnamed), HEMGS and PCGS none.
- Decoding is a one-time cost of seconds to tens of seconds on CPU-bound arithmetic coding (HAC 26.7 s BungeeNeRF, HAC++ 15.77 to 30.86 s Mip-NeRF360, ContextGS about 30 s), which matters for any "memory at deploy" reading of size.
- No paper states the gap between the entropy estimate used in training and the encoded bytes.
- No paper states a licence. Code: HAC https://github.com/YihangChen-ee/HAC, HAC++ https://github.com/YihangChen-ee/HAC-plus, PCGS github.com/YihangChen-ee/PCGS, HEMGS and SALVQ none printed.
