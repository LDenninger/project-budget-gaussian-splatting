# F1 Foundations: 3DGS, Scaffold-GS, 3DGS-MCMC, Mip-Splatting, 2DGS

Reading notes for the BudgetGS project. Every page of each PDF was read from a pymupdf
text extraction. Every number below is copied from the paper and marked `(reported)`. A count that
the paper does not print but that follows from stated attributes is marked `(derived)` with the
assumption spelled out. `not stated` means the paper does not say it. Page numbers refer to the PDF
page of the arXiv file named in each citation.

Conventions: the per-primitive byte counts assume 4 bytes per float at fp32 and 2 bytes at fp16,
with no header, index or compression.

---

## 1. 3D Gaussian Splatting for Real-Time Radiance Field Rendering

### 1.1 Citation

- Title: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- First author: Bernhard Kerbl (with Georgios Kopanas as equal first author), Inria, Université Côte d'Azur
- Venue and year as stated: ACM Transactions on Graphics, Vol. 42, No. 4, Article 1, August 2023 (SIGGRAPH 2023)
- arXiv id from file name: 2308.04079 (file: `3dgs_kerbl_siggraph2023_arxiv2308.04079.pdf`, arXiv v1, 8 Aug 2023)

### 1.2 Per-primitive payload

Attributes the paper states are stored and optimised per 3D Gaussian (p. 4, Sec. 4, and p. 4, Sec. 5):

| attribute | dimensionality | source |
|---|---|---|
| position (mean) μ | 3 | p. 4, "centered at point (mean) μ" |
| scale vector s | 3 | p. 4, "a 3D vector s for scaling" |
| rotation quaternion q | 4 | p. 4, "a quaternion q to represent rotation" |
| opacity α | 1 | p. 4, "This Gaussian is multiplied by α in our blending process" |
| SH colour coefficients | 4 bands, i.e. degrees 0 to 3, 16 coefficients per colour channel, 3 channels, 48 total (derived) | p. 8, "introduce one band of the SH after every 1000 iterations until all 4 bands of SH are represented" |

- Total: 3 + 3 + 4 + 1 + 48 = 59 floats per Gaussian (derived, the paper never prints a per-Gaussian float count).
- Bytes: 236 B at fp32, 118 B at fp16 (derived).
- Covariance is rebuilt as Σ = R S Sᵀ Rᵀ from s and q (p. 4, Eq. 6), so no covariance entries are stored.
- Activations (p. 5): sigmoid on α to keep it in [0, 1), exponential on the scale.
- Initial covariance (p. 5): isotropic, axes equal to the mean distance to the three closest SfM points.
- Compactness experiment on the Zhang et al. 2022 data uses "only two degrees of our spherical harmonics" (p. 9), which the paper does not translate into a coefficient count.

### 1.3 Count control (adaptive density control, ADC)

All values are reported unless marked.

- Initialisation: SfM sparse point cloud (p. 4, Sec. 3). For the synthetic NeRF dataset, 100K uniformly random Gaussians inside the scene bounds (p. 9).
- Warm-up: optimisation starts at 4 times smaller image resolution and upsamples twice, after 250 and 500 iterations (p. 8). SH bands are added one band every 1000 iterations until all 4 bands are present (p. 8).
- Densification interval: "After optimization warm-up (see Sec. 7.1), we densify every 100 iterations and remove any Gaussians that are essentially transparent, i.e., with α less than a threshold ε_α" (p. 5).
- Densification start iteration: not stated as a number. Densification end iteration: not stated.
- Densification criterion: average magnitude of the view-space positional gradient above τ_pos = 0.0002 (p. 5).
- Clone rule (under-reconstruction, small Gaussians): create a copy of the same size and move it along the positional gradient (p. 5).
- Split rule (over-reconstruction, large Gaussians): replace by two new Gaussians, divide their scale by φ = 1.6 "which we determined experimentally", and sample their positions using the original Gaussian as a PDF (p. 5).
- Small versus large decision: Algorithm 1 (p. 13) splits when ‖S‖ > τ_S and clones otherwise. The value of τ_S is not stated.
- Opacity reset: "set the α value close to zero every N = 3000 iterations" (p. 5 to 6), after which the optimiser raises α where needed and the culling step removes those with α < ε_α.
- Opacity prune threshold ε_α: not stated as a number in this paper. (The 3DGS-MCMC paper, p. 5, calls 0.005 "the default threshold used in 3D Gaussian Splatting [19]".)
- Size pruning: "we periodically remove Gaussians that are very large in worldspace and those that have a big footprint in viewspace" (p. 6). Algorithm 1 (p. 13) writes this as `IsTooLarge(μ, Σ)`. Thresholds and interval: not stated.
- Loss: L = (1 − λ) L1 + λ L_D-SSIM with λ = 0.2 (p. 5, Eq. 7).
- Learning-rate schedule: standard exponential decay, positions only (p. 5). Values not stated.
- Total iterations: two configurations reported, 7K and 30K iterations (p. 8, Table 1).
- Resulting counts: "1-5 million Gaussians for all scenes tested" (p. 2). Synthetic NeRF: 100K random initial Gaussians are "quickly and automatically" pruned to about 6–10K, and the final model after 30K iterations reaches about 200–500K Gaussians per scene (p. 9).
- The paper's own summary of the control effect: "This strategy results in overall good control over the total number of Gaussians." (p. 6). No explicit target count or cap exists.

### 1.4 Render-time cost

- Rendering pipeline (p. 6, Appendix C p. 13 to 14): 16×16 pixel tiles, frustum culling keeping Gaussians whose 99 % confidence interval intersects the frustum plus a guard band, one instance per overlapped tile, a single GPU radix sort on 64-bit keys (lower 32 bits depth, upper bits tile id), per-tile front-to-back α-blending that stops when a pixel saturates.
- Numerical guards (p. 14): blending updates with α < 1/255 are skipped, α is clamped to 0.99 from above, blending stops before accumulated opacity would exceed 0.9999.
- Hardware: "All results are reported running on an A6000 GPU, except for the Mip-NeRF360 method" (p. 8). Mip-NeRF360 was trained on a 4-GPU A100 node for 12 hours (p. 9, footnote 2).
- Reported FPS and storage (p. 8, Table 1, "Mem" is "memory used to store optimized parameters", p. 8):

| dataset | config | train time | FPS | Mem |
|---|---|---|---|---|
| Mip-NeRF360 | Ours-7K | 6m25s | 160 | 523 MB |
| Mip-NeRF360 | Ours-30K | 41m33s | 134 | 734 MB |
| Tanks&Temples | Ours-7K | 6m55s | 197 | 270 MB |
| Tanks&Temples | Ours-30K | 26m54s | 154 | 411 MB |
| Deep Blending | Ours-7K | 4m35s | 172 | 386 MB |
| Deep Blending | Ours-30K | 36m2s | 137 | 676 MB |

- Fig. 1 (p. 1): 135 fps after 6 min training (PSNR 23.6) and 93 fps after 51 min (PSNR 25.2) on one scene (reported).
- Synthetic scenes render at 180–300 FPS (p. 9, reported).
- Abstract target: real-time (≥ 30 fps) at 1080p (p. 1, reported).
- Training GPU memory: "During training of large scenes, peak GPU memory consumption can exceed 20 GB in our unoptimized prototype." (p. 11, reported).
- Rendering GPU memory: "Rendering the trained scene requires sufficient GPU memory to store the full model (several hundred megabytes for large-scale scenes) and an additional 30–500 MB for the rasterizer, depending on scene size and image resolution." (p. 11, reported).
- Training time split: about 80 % of training time is spent in Python code (p. 11, reported).
- Compactness experiment (p. 9): average model size 3.8 MB versus 9 MB for Zhang et al. 2022, with about one-fourth of their point count, using two SH degrees (reported).

### 1.5 Quality numbers (reported, p. 8, Table 1)

| dataset | config | SSIM | PSNR | LPIPS | size |
|---|---|---|---|---|---|
| Mip-NeRF360 (9 scenes) | Ours-7K | 0.770 | 25.60 | 0.279 | 523 MB |
| Mip-NeRF360 (9 scenes) | Ours-30K | 0.815 | 27.21 | 0.214 | 734 MB |
| Tanks&Temples (Truck, Train) | Ours-7K | 0.767 | 21.20 | 0.280 | 270 MB |
| Tanks&Temples (Truck, Train) | Ours-30K | 0.841 | 23.14 | 0.183 | 411 MB |
| Deep Blending (DrJohnson, Playroom) | Ours-7K | 0.875 | 27.78 | 0.317 | 386 MB |
| Deep Blending (DrJohnson, Playroom) | Ours-30K | 0.903 | 29.41 | 0.243 | 676 MB |

Per-scene tables 4 to 9 are on p. 14. Gaussian counts per dataset: not stated beyond "1-5 million" (p. 2).

### 1.6 Memory, storage, compactness, edge and budget statements (verbatim)

- p. 1 (abstract): "allow high-quality real-time (≥30 fps) novel-view synthesis at 1080p resolution."
- p. 2: "The optimization procedure produces a reasonably compact, unstructured, and precise representation of the scene (1-5 million Gaussians for all scenes tested)."
- p. 4: "This results in a reasonably compact representation of the 3D scene, in part because highly anisotropic volumetric splats can be used to represent fine structures compactly."
- p. 5: "The quality of the parameters of the covariances of the 3D Gaussians is critical for the compactness of the representation since large homogeneous areas can be captured with a small number of large anisotropic Gaussians."
- p. 6: "Our fast rasterizer allows efficient backpropagation over an arbitrary number of blended Gaussians with low additional memory consumption, requiring only a constant overhead per pixel."
- p. 6: "An effective way to moderate the increase in the number of Gaussians is to set the α value close to zero every N = 3000 iterations."
- p. 6: "This strategy results in overall good control over the total number of Gaussians."
- p. 8: "We also show the average training time, rendering speed, and memory used to store optimized parameters."
- p. 9: "Compactness. In comparison to previous explicit scene representations, the anisotropic Gaussians used in our optimization are capable of modelling complex shapes with a lower number of parameters. We showcase this by evaluating our approach against the highly compact, point-based models obtained by [Zhang et al. 2022]. ... We surpass their reported metrics using approximately one-fourth of their point count, resulting in an average model size of 3.8 MB, as opposed to their 9 MB. We note that for this experiment, we only used two degrees of our spherical harmonics, similar to theirs."
- p. 11: "Even though we are very compact compared to previous point-based approaches, our memory consumption is significantly higher than NeRF-based solutions. During training of large scenes, peak GPU memory consumption can exceed 20 GB in our unoptimized prototype. However, this figure could be significantly reduced by a careful low-level implementation of the optimization logic (similar to InstantNGP). Rendering the trained scene requires sufficient GPU memory to store the full model (several hundred megabytes for large-scale scenes) and an additional 30–500 MB for the rasterizer, depending on scene size and image resolution. We note that there are many opportunities to further reduce memory consumption of our method. Compression techniques for point clouds is a well-studied field [De Queiroz and Chou 2016]; it would be interesting to see how such approaches could be adapted to our representation."
- Edge, mobile, low-end deployment or a deployment budget: none found (MobileNeRF appears only as a bibliography entry, p. 12).

### 1.7 Limitations and future work (verbatim)

- p. 10, Sec. 7.4: "Our method is not without limitations. In regions where the scene is not well observed we have artifacts; in such regions, other methods also struggle (e.g., Mip-NeRF360 in Fig. 11). Even though the anisotropic Gaussians have many advantages as described above, our method can create elongated artifacts or "splotchy" Gaussians (see Fig. 12); again previous methods also struggle in these cases. We also occasionally have popping artifacts when our optimization creates large Gaussians; this tends to happen in regions with view-dependent appearance. One reason for these popping artifacts is the trivial rejection of Gaussians via a guard band in the rasterizer. A more principled culling approach would alleviate these artifacts. Another factor is our simple visibility algorithm, which can lead to Gaussians suddenly switching depth/blending order. This could be addressed by antialiasing, which we leave as future work. Also, we currently do not apply any regularization to our optimization; doing so would help with both the unseen region and popping artifacts. While we used the same hyperparameters for our full evaluation, early experiments show that reducing the position learning rate can be necessary to converge in very large scenes (e.g., urban datasets)."
- p. 11: the memory paragraph quoted in 1.6 (peak GPU memory above 20 GB, 30–500 MB rasterizer overhead, point-cloud compression as future work).
- p. 11, Sec. 8: "The majority (∼80%) of our training time is spent in Python code, since we built our solution in PyTorch to allow our method to be easily used by others. Only the rasterization routine is implemented as optimized CUDA kernels. We expect that porting the remaining optimization entirely to CUDA, as e.g., done in InstantNGP [Müller et al. 2022], could enable significant further speedup for applications where performance is essential."
- p. 11: "It would be interesting to see if our Gaussians can be used to perform mesh reconstructions of the captured scene."

### 1.8 Code

- "The source code and all our data are available at: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/" (p. 8).
- Licence: not stated in the paper.

---

## 2. Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

### 2.1 Citation

- Title: Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
- First author: Tao Lu (equal contribution with Mulin Yu), Shanghai Artificial Intelligence Laboratory and Nanjing University
- Venue and year as stated: the PDF carries no venue line beyond the arXiv stamp (arXiv v1, 30 Nov 2023). The file name records CVPR 2024.
- arXiv id from file name: 2312.00109 (file: `scaffold_gs_lu_cvpr2024_arxiv2312.00109.pdf`)

### 2.2 Per-primitive payload

Scaffold-GS stores anchors plus shared MLPs. Neural Gaussians are decoded per view and are not stored.

Per anchor (p. 3, Sec. 3.2.1, and p. 12 supplementary):

| attribute | dimensionality | source |
|---|---|---|
| anchor position x_v (voxel centre) | 3 | p. 3, "V ∈ R^{N×3} denotes voxel centers", "The center of each voxel v ∈ V is treated as an anchor point" (whether the position is optimised is not stated) |
| local context feature f_v | 32 | p. 3, "a local context feature f_v ∈ R^32" |
| scaling factor l_v (offset scale) | 3 | p. 3, "a scaling factor l_v ∈ R^3" |
| k learnable offsets O_v | k × 3, with k = 10 in all experiments, so 30 | p. 3, "k learnable offsets O_v ∈ R^{k×3}", p. 5, "we set k = 10 for all experiments" |
| base scaling s_v | 3 (derived from Eq. 16 on p. 12, "{s_0, ..., s_{k−1}} = Sigmoid(F_s) · s_v"), whether s_v is the same tensor as l_v is not stated | p. 12 |

- The feature bank {f_v, f_v↓1, f_v↓2} is produced by "slicing and repeating" f_v (p. 12, Fig. 10), so it adds no stored floats.
- Total per anchor: 3 + 32 + 3 + 30 = 68 floats (derived, k = 10, position counted, s_v assumed identical to l_v). With a separate 3-float s_v the count is 71 (derived).
- Bytes per anchor: 272 B at fp32, 136 B at fp16 (derived, 68 floats). With 71 floats: 284 B and 142 B.
- The paper never prints a per-anchor float count.

Per neural Gaussian (decoded, never stored, p. 4): position μ ∈ ℝ³, opacity α ∈ ℝ, quaternion q ∈ ℝ⁴, scale s ∈ ℝ³, colour c ∈ ℝ³ (RGB, not SH). Positions: {μ_0, ..., μ_{k−1}} = x_v + {O_0, ..., O_{k−1}} · l_v (p. 4, Eq. 8).

Shared MLPs (p. 5 and p. 12, Fig. 12):

- "All the MLPs employed in our approach are 2-layer MLPs with ReLU activation; the dimensions of the hidden units are all 32." (p. 5)
- Input to F_α, F_c, F_s, F_q: N × (32 + 3 + 1), i.e. the blended anchor feature f̂_v (32), the view direction d_vc (3) and the distance δ_vc (1) (p. 12, Fig. 12).
- F_α: 36×32 then 32×1×k, output activated with Tanh, "where value 0 serves as a natural threshold for selecting valid samples" (p. 12).
- F_c: 36×32 then 32×3×k, output Sigmoid (p. 12).
- F_s and F_q: 36×32 then 32×7×k, split into scale (3 per Gaussian, Sigmoid times s_v) and quaternion (4 per Gaussian, normalised) (p. 12).
- F_w: "a tiny MLP" taking (δ_vc, d_vc) and returning Softmax weights (w, w_1, w_2) for the feature bank (p. 4, Eq. 6, and p. 12, Eq. 13). Its size is not stated.
- Derived weight count for k = 10 with biases: F_α 36·32 + 32 + 32·10 + 10 = 1 514, F_c 36·32 + 32 + 32·30 + 30 = 2 174, F_s and F_q 36·32 + 32 + 32·70 + 70 = 3 494, total about 7.2k parameters, about 29 KB at fp32, F_w excluded. The paper prints no MLP parameter count.

### 2.3 Count control (anchor growing and pruning)

- Anchor initialisation (p. 3, Eq. 4): voxelise the COLMAP point cloud with voxel size ε, V = {⌊P/ε⌋ · ε}, duplicates removed. Voxel size ε (p. 12 to 13): either the median of the nearest-neighbour distances among all initial points, or set manually to 0.005 or 0.01 (reported).
- Growing (p. 4 to 5): neural Gaussians are quantised into voxels of size ε_g, the gradients of the included neural Gaussians are averaged over N training iterations into ∇_g, and a voxel with ∇_g > τ_g receives a new anchor at its centre if none exists. Multi-resolution levels m ∈ {1, 2, 3} with ε_g^(m) = ε_g / 4^(m−1) and τ_g^(m) = τ_g · 2^(m−1) (p. 5, Eq. 10). "To further regulate the addition of new anchors, we apply a random elimination to these candidates." (p. 5). Elimination rate: not stated. ε_g value: not stated.
- Growing hyperparameters (p. 5): N = 100 iterations, τ_g = 64ε by default, τ_g = 16ε "On intricate scenes and the ones with dominant texture-less regions" (reported verbatim, the paper writes these thresholds in terms of ε).
- Pruning (p. 5): "To eliminate trivial anchors, we accumulate the opacity values of their associated neural Gaussians over N training iterations. If an anchor fails to produce neural Gaussians with a satisfactory level of opacity, we then remove it from the scene." Threshold (p. 5): "An anchor is pruned if the accumulated opacity of its neural Gaussians is less than 0.5 at each round of refinement."
- Start and end iterations of refinement: not stated. Refinement round length: N = 100 iterations (p. 5).
- Per-view opacity filter (p. 4): only neural Gaussians with α ≥ τ_α are rasterised. τ_α numeric value: not stated (Tanh output with 0 as the "natural threshold", p. 12).
- Losses (p. 5, Eq. 11 to 12): L = L1 + λ_SSIM L_SSIM + λ_vol L_vol, L_vol = Σ[i=1..N_ng] Prod(s_i), λ_SSIM = 0.2, λ_vol = 0.001 (reported). The volume term "encourages the neural Gaussians to be small with minimal overlapping" (p. 5).
- Total iterations: 30k for both 3D-GS and Scaffold-GS (p. 5). Synthetic Blender: start from 100k grid points, grow and prune for 30k iterations, then re-run from the remaining anchors (p. 6 to 7).
- Counts: no absolute anchor or Gaussian counts are printed. Fig. 9 (p. 8) states that for different k "the final number of activated neural Gaussians converges to a similar amount".
- Ablation (p. 8, Table 5, Mem in MB, reported): DB-Playroom NONE 28.45 PSNR / 24 MB, W/ PRUNING 29.12 / 23, W/ GROWING 30.54 / 71, FULL 30.62 / 63. DB-DrJohnson NONE 28.81 / 12, W/ PRUNING 28.51 / 12, W/ GROWING 29.75 / 76, FULL 29.80 / 68.

### 2.4 Render-time cost

- Per view, per anchor inside the frustum: compute δ_vc and d_vc (p. 4, Eq. 5), run F_w to blend the feature bank (Eq. 6 to 7), then run F_α, F_c, F_s, F_q once per anchor to decode all k = 10 neural Gaussians ("attributes are decoded in one-pass", p. 4), discard Gaussians with α < τ_α, then rasterise with the 3D-GS tile rasteriser (p. 4). "only anchors visible within the frustum are activated to spawn neural Gaussians" (p. 4).
- Effect on FPS (p. 8, Table 4, reported, PSNR / FPS): DB-PLAYROOM NO FILTERS 30.4 / 84, FILTER 1 (frustum) 30.3 / 118, FILTER 2 (opacity) 30.6 / 109, FULL 30.62 / 150. DB-DRJOHNSON NO FILTERS 29.7 / 79, FILTER 1 29.6 / 100, FILTER 2 29.7 / 104, FULL 29.8 / 129. Caption: "The filtering method has no notable impact on fidelity, but greatly affects inference speed."
- Reported FPS and storage (p. 6, Table 2, "Rendering speed of both methods are measured on our machine", hardware not named):

| dataset | 3D-GS FPS | 3D-GS Mem | Ours FPS | Ours Mem |
|---|---|---|---|---|
| Mip-NeRF360 | 97 | 693 MB | 102 | 156 MB (4.4× ↓) |
| Tanks&Temples | 123 | 411 MB | 110 | 87 MB (4.7× ↓) |
| Deep Blending | 109 | 676 MB | 139 | 66 MB (10.2× ↓) |

- Fig. 1 (p. 1, reported): 3D-GS 17.16 dB / 242 MB / 127 FPS versus Ours 20.41 dB / 66 MB / 110 FPS. 3D-GS 34.60 dB / 204 MB / 113 FPS versus Ours 35.41 dB / 48 MB / 88 FPS. 3D-GS 29.93 dB / 288 MB / 109 FPS versus Ours 31.13 dB / 133 MB / 128 FPS.
- p. 2: "our approach can render at a similar speed (around 100 FPS at 1K resolution) as the original 3D-GS with little computational overhead."
- Other datasets (p. 7, Table 3, PSNR / Mem MB, reported): BungeeNeRF 3D-GS 24.89 / 1606, Ours 27.01 / 203 (7.9× ↓). VR-NeRF 3D-GS 28.94 / 263, Ours 29.24 / 69 (3.8× ↓). Synthetic Blender 3D-GS 33.32 / 53, Ours 33.68 / 14 (3.8× ↓).
- Per-scene storage (p. 13, Table 9, MB, reported): Mip-NeRF360 3D-GS bicycle 1291, garden 1268, stump 1034, room 327, counter 261, kitchen 414, bonsai 281. Ours 248, 271, 493, 133, 194, 173, 258. (p. 14, Table 13) Truck 578 vs 107, Train 240 vs 66, Dr Johnson 715 vs 69, Playroom 515 vs 63.
- Training GPU memory: not stated. Training time: not stated as a number ("our approach converged faster than 3D-GS", p. 6).

### 2.5 Quality numbers (reported, p. 6, Table 1, 30k iterations)

| dataset | method | PSNR | SSIM | LPIPS | size |
|---|---|---|---|---|---|
| Mip-NeRF360 (7 scenes: bicycle, garden, stump, room, counter, kitchen, bonsai) | 3D-GS | 28.69 | 0.870 | 0.182 | 693 MB |
| Mip-NeRF360 (7 scenes) | Ours | 28.84 | 0.848 | 0.220 | 156 MB |
| Tanks&Temples | 3D-GS | 23.14 | 0.841 | 0.183 | 411 MB |
| Tanks&Temples | Ours | 23.96 | 0.853 | 0.177 | 87 MB |
| Deep Blending | 3D-GS | 29.41 | 0.903 | 0.243 | 676 MB |
| Deep Blending | Ours | 30.21 | 0.906 | 0.254 | 66 MB |

Per-scene (p. 13 to 14, Tables 6 to 12): Ours Mip-NeRF360 PSNR bicycle 24.50, garden 27.17, stump 26.27, room 31.93, counter 29.34, kitchen 31.30, bonsai 32.70. Truck 25.77, Train 22.15, Dr Johnson 29.80, Playroom 30.62. Anchor counts: not stated.

### 2.6 Memory, storage, compactness, edge and budget statements (verbatim)

- p. 1 (Fig. 1 caption): "Our method achieves rendering quality and speed comparable to 3D-GS with a more compact model (last row metrics: PSNR/storage size/FPS)."
- p. 2: "This results in significant redundancy and limits its scalability, particularly in the context of complex large-scale scenes."
- p. 2: "As a result, our approach can render at a similar speed (around 100 FPS at 1K resolution) as the original 3D-GS with little computational overhead. Moreover, our storage requirements are significantly reduced as we only need to store anchor points and MLP predictors for each scene."
- p. 4: "To make the rasterization more efficient, we only keep neural Gaussians whose opacity values are larger than a predefined threshold τα. This substantially cuts down the computational load and helps our method maintain a high rendering speed on-par with the original 3D-GS."
- p. 5: "Apart from the commonly used metrics (PSNR, SSIM [47], and LPIPS [56]), we additionally report the storage size (MB) and the rendering speed (FPS) for model compactness and performance efficiency."
- p. 6: "Our method achieved real-time rendering while using less storage, indicating that our model is more compact than 3D-GS without sacrificing rendering quality and speed."
- p. 7 (Table 3 caption): "Our method is able to handle large-scale scenes (e.g. BUNGEENERF) with light-weight representation. Our method shows consistent compactness and effectiveness in complex lighting conditions and synthetic scenes."
- p. 7: "On contrary, our method efficiently encoded local structures into compact neural features, enhancing both rendering quality and convergence speed."
- p. 8 (Table 5 caption): "The pruning operation controls the increasing of storage size and optimizes the quality of remained anchors."
- p. 13: "1) Use the median of the nearest-neighbor distances among all initial points: ε is adapted to point cloud density, yielding denser anchors with enhanced rendering quality but might introduce more computational overhead; 2) Set ε manually to either 0.005 or 0.01: this is effective in most scenarios but might lead to missing details in texture-less regions."
- Edge, mobile, low-end deployment or a deployment budget: none found (MobileNeRF appears only as bibliography entry [11], p. 10).

### 2.7 Limitations and future work (verbatim)

- p. 8, Sec. 4.3: "However, there was a risk of masking pertinent neural Gaussians, which we aim to address in future works."
- p. 8 to 9, Sec. 4.4: "Through our experiments, we found that the initial points play a crucial role for high-fidelity results. Initializing our framework from SfM point clouds is a swift and viable solution, considering these point clouds usually arise as a byproduct of image calibration processes. However, this approach may be suboptimal for scenarios dominated by large texture-less regions. Despite our anchor point refinement strategy can remedy this issue to some extent, it still suffers from extremely sparse points. We expect that our algorithm will progressively improve as the field advances, yielding more accurate results. Further details are discussed in the supplementary material."
- p. 9, Sec. 5: "We further show that our anchor points encode local features in a meaningful way that exhibits semantic patterns to some degree, suggesting its potential applicability in a range of versatile tasks such as large-scale modeling, manipulation and interpretation in the future."
- p. 13: "As briefly discussed in the main paper, the voxelization process suggests that our method may behave sensitive to initial SfM results."

### 2.8 Code

- Project page: https://city-super.github.io/scaffold-gs/ (p. 1). Repository URL: not stated in the paper.
- Licence: not stated.

---

## 3. 3D Gaussian Splatting as Markov Chain Monte Carlo

### 3.1 Citation

- Title: 3D Gaussian Splatting as Markov Chain Monte Carlo
- First author: Shakiba Kheradmand, University of British Columbia
- Venue and year as stated: 38th Conference on Neural Information Processing Systems (NeurIPS 2024) (p. 1)
- arXiv id from file name: 2404.09591 (file: `3dgs_mcmc_kheradmand_neurips2024_arxiv2404.09591.pdf`, arXiv v3, 12 Feb 2025)

### 3.2 Per-primitive payload

- The representation is unchanged from 3DGS: "As our method results in the same 3D Gaussian Splat representation as prior work, inference time is the same with 3DGS [19]" (p. 10). Per Gaussian: colour c_i "stored as Spherical Harmonics" (p. 4), opacity o, centre μ, covariance Σ (p. 4, Eq. 2). SH degree: not stated (the method is implemented "on top of the 3DGS [19] framework", p. 6).
- Derived float count under the 3DGS parameterisation with 4 SH bands: 59 floats, 236 B at fp32, 118 B at fp16 (derived, see 1.2). The paper prints no float count.

### 3.3 Count control (maximum cap, relocation, noise, regularisers)

- Maximum cap and growth (p. 6 to 7): "we allow the number of Gaussians to gradually grow, so that Gaussians are placed at useful locations. We do this simply by initially starting with a selected number of Gaussians, then allowing more 'dead' Gaussians to become 'alive' through our relocation strategy we previously detailed in Section 3.4. Specifically, we gradually increase the number of live Gaussians by 5% until the maximum desired number of Gaussians is met." Growth cadence beyond "gradually": not stated separately from the relocation interval of 100 iterations.
- Dead versus live (p. 5): live means o_i ≥ 0.005, dead means o_i < 0.005, "This is the default threshold used in 3D Gaussian Splatting [19]" (p. 5, footnote 2).
- Count semantics (p. 6): "one could think of the state with a smaller number of Gaussians simply as the equivalent state with more Gaussians, but with those that have zero opacity, that is, dead Gaussians." Reducing the count is therefore a state with dead Gaussians, not a deletion.
- Relocation rule (p. 6, Eq. 9), moving N − 1 dead Gaussians g_1..N−1 onto live Gaussian g_N:
  - μ_new_{1..N} = μ_old_N
  - o_new_{1..N} = 1 − (1 − o_old_N)^(1/N)
  - Σ_new_{1..N} = (o_old_N)² · ( Σ[i=1..N] Σ[k=0..i−1] C(i−1, k) (−1)^k (o_new_N)^(k+1) / √(k+1) )^(−2) · Σ_old_N
  - Derived in Appendix A (p. 14 to 15) by equating the integral of the rasterised 1D slice before and after cloning. Colour is copied, c_new = c_old (p. 14).
- Relocation implementation (p. 6): applied every 100 iterations "to avoid disruptions in the training process". For each dead Gaussian a target is chosen "via multinomial sampling of the live Gaussians with the probabilities proportional to their opacity values". Eq. 9 is applied only after all movement decisions are made. Adam moment statistics are reset for the target Gaussian and retained for the relocated (source) Gaussians.
- Warm-up (p. 7): "we start with 500 warmup iterations, during which we do not perform our relocalization in Sec. 3.4 nor increase the number of Gaussians."
- SGLD update (p. 5, Eq. 7): g ← g − λ_lr · ∇_g E_I[L_total(g; I)] + λ_noise · ε, Adam with default β_1 and β_2.
- Noise term (p. 5, Eq. 8): ε_μ = λ_lr · σ(−k(t − o)) · Σ η with η ∼ N(0, I), ε = [ε_μ, 0], k = 100, t = 0.005. Noise is applied only to positions: "We further notice that exploration is not critical for opacity, scale, and color, and we do not add noise to these parameters." (p. 5). λ_noise = 5 × 10⁵ (p. 7). The noise follows the 3DGS exponential learning-rate schedule (p. 9). Position learning rate 1.6e−4 decayed exponentially to 1.6e−6 (p. 7).
- Regularisers (p. 6, Eq. 10): L_total = (1 − λ_D-SSIM) L1 + λ_D-SSIM L_D-SSIM + λ_o Σ[i] |o_i|₁ + λ_Σ Σ[i,j] |√eig_j(Σ_i)|₁ with λ_D-SSIM = 0.2 (p. 4), λ_o = 0.01, λ_Σ = 0.01, and λ_o = 0.001 for Deep Blending (p. 7). Purpose (p. 6): "we encourage Gaussians to disappear in non-useful locations and 'respawn' elsewhere."
- Initialisation (p. 7): random (100k Gaussians uniformly inside 3× the camera bounding box, as in 3DGS) or SfM points.
- How the cap is set in experiments: "we simply set the number of Gaussians used in the original 3DGS [19] and set it as our maximum number of Gaussians to be used during training and inference" (p. 7 to 8). Timing study on Room (p. 10): "maximum number of 1.5M Gaussians per the original 3DGS implementation", and a 300k cap in Table 4. Section 4.2 measures iteration time "at 1M Gaussian" (p. 10). Absolute per-dataset caps for Table 1: not stated.
- Limited budget experiment (p. 8, Fig. 3): budgets swept on all datasets except NeRF Synthetic. For 3DGS the densification is stopped once the limit is reached, "Note that the pruning strategy can cause densification to resume, should some Gaussians get pruned after this threshold is met." Result: "With a limited budget, the gap between our method and 3DGS [19] increases." Numeric values of Fig. 3: not extractable from the PDF text.
- Whether the cap is reached exactly: the paper states growth continues "until the maximum desired number of Gaussians is met" and treats dead Gaussians as members of the set, so the stored count equals the cap once met. The number of live Gaussians at convergence, and whether every slot holds a live Gaussian, are not stated.
- Ablation (p. 9, Table 3, Mip-NeRF 360, random init, PSNR / SSIM / LPIPS, reported): 3DGS 27.89 / 0.84 / 0.26, 3DGS w/ L_total 23.84 / 0.77 / 0.33, Ours w/ L_orig 23.90 / 0.67 / 0.42, Ours λ_noise = 0 27.41 / 0.83 / 0.26, Ours noise on all parameters 29.11 / 0.86 / 0.24, Ours full 29.72 / 0.89 / 0.19. On Tanks & Temples (p. 9): without the covariance term in the noise 23.16 PSNR versus 24.21 with it, without the opacity term 22.47 (requiring λ_noise = 0.05 or training fails with PSNR < 7). Linear noise scheduler 17.64, the scheduler of [30] 22.46, exponential 24.21.

### 3.4 Render-time cost

- Rendering: identical to 3DGS, "inference time is the same with 3DGS [19], which is highly efficient" (p. 10). FPS: not stated as a number. Related work cites 3DGS rendering "1080p images ... at 130 frames per second on modern GPUs" (p. 3).
- Training iteration time at 1M Gaussians (p. 10, reported): 80 ms for the MCMC method versus 76 ms for 3DGS. Resampling uses PyTorch `torch.multinomial`.
- Timings on Room, SfM init, 1.5M maximum unless noted (p. 10, Table 4, reported): 3DGS PSNR 31.7 in 25 minutes. Ours λ_o = 0.01 PSNR 32.5 in 42 minutes. Ours λ_o = 0.001 PSNR 32.4 in 30 minutes. Ours λ_o = 0.01 with 300k maximum PSNR 31.8 in 21 minutes.
- p. 10: "An independent implementation of our method [45] also confirms a 20% reduction in training time and a 65% reduction in required memory when using our method." ([45] is gsplat.)
- GPU memory in training or rendering: not stated as a number by the authors. Storage in MB: not stated. Hardware: not stated.

### 3.5 Quality numbers (reported, p. 7, Table 1, same number of Gaussians as 3DGS, averaged over three runs)

| dataset | method | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| MipNeRF 360 (7 public scenes, no Flowers or Treehill) | 3DGS (Random) | 27.89 | 0.84 | 0.26 |
| MipNeRF 360 | Ours (Random) | 29.72 | 0.89 | 0.19 |
| MipNeRF 360 | 3DGS (SfM) | 29.30 | 0.88 | 0.21 |
| MipNeRF 360 | Ours (SfM) | 29.89 | 0.90 | 0.19 |
| Tank & Temples | 3DGS (Random) | 21.93 | 0.79 | 0.27 |
| Tank & Temples | Ours (Random) | 24.21 | 0.86 | 0.19 |
| Tank & Temples | 3DGS (SfM) | 23.67 | 0.84 | 0.22 |
| Tank & Temples | Ours (SfM) | 24.29 | 0.86 | 0.19 |
| Deep Blending | 3DGS (Random) | 29.55 | 0.90 | 0.33 |
| Deep Blending | Ours (Random) | 29.71 | 0.90 | 0.32 |
| Deep Blending | 3DGS (SfM) | 29.64 | 0.90 | 0.32 |
| Deep Blending | Ours (SfM) | 29.67 | 0.89 | 0.32 |

Counts and sizes: the cap equals the 3DGS count per scene, absolute numbers not stated. Per-scene values and standard deviations are in Table 5 (p. 15). Note that the 3DGS SfM MipNeRF 360 number (29.30) is the authors' own re-run on 7 scenes at their resolutions.

### 3.6 Memory, storage, compactness, edge and budget statements (verbatim)

- p. 1 (abstract): "On various standard evaluation scenes, we show that our method provides improved rendering quality, easy control over the number of Gaussians, and robustness to initialization."
- p. 1: "Depending on the state of each Gaussian, they are cloned, split, or pruned, which is the primary way to control the number of Gaussians within the 3DGS representation."
- p. 2: "It is also nontrivial to estimate how many 3D Gaussians will be used for a given scene from just the hyperparameters, making it difficult to control the computation and memory budget in advance without affecting reconstruction quality during inference time."
- p. 2: "In some cases, this leads to sub-optimal placement of Gaussians resulting in poor quality renderings and wasted compute."
- p. 3: "Methods that focus on more compact representation, thus suitable for rendering on mobile devices, have also been proposed. These methods prune/cluster Gaussians, adaptively selecting the number of spherical harmonics to encode color [32], and quantize the parameters of the representation [31]."
- p. 6: "To make effective use of the memory and compute while improving the performance, we encourage Gaussians to disappear in non-useful locations and 'respawn' elsewhere."
- p. 7: "Specifically, we gradually increase the number of live Gaussians by 5% until the maximum desired number of Gaussians is met."
- p. 7: "Performance with the same number of Gaussians. As the number of Gaussians is directly related to the quality of novel-view rendering, we first compare our method with existing baselines using the same number of Gaussians as 3DGS [19]."
- p. 8: "Limited budget. We further verify the effectiveness of our formulation by limiting the budget for the number of Gaussians on all the datasets" and "With a limited budget, the gap between our method and 3DGS [19] increases."
- p. 10: "As our method results in the same 3D Gaussian Splat representation as prior work, inference time is the same with 3DGS [19], which is highly efficient."
- p. 10: "An independent implementation of our method [45] also confirms a 20% reduction in training time and a 65% reduction in required memory when using our method."
- Edge or low-end deployment: none found beyond the mobile-devices sentence on p. 3 about other methods.

### 3.7 Limitations and future work (verbatim)

- p. 3: "Their error-based densification is orthogonal to our research direction and could easily be incorporated into our method as well, which we leave to future works."
- p. 10: "That is, the added time for sampling and noise addition is not substantial, even with our implementation that implements resampling naively with PyTorch [33]'s torch.multinomial—this could be further accelerated with a CUDA implementation."
- p. 16, Appendix C: "While our method allows more robustness to initialization and higher rendering quality thanks to the exploration introduced by MCMC, it is still subject to the same limitations as 3DGS [19] in terms of its modelling capacity. For example, the aliasing issue solved in [48] can also be a problem with our method, as well as modelling reflections [44]. Our method, however, should be compatible with these advancements in Gaussian Splatting, as our method can be viewed as a better training framework for Gaussian Splats. In other words, it should enhance all other Gaussian Splatting methods that are available."

### 3.8 Code

- Project page: https://ubc-vision.github.io/3dgs-mcmc (p. 1). Repository URL: not stated in the paper text. Implementation is "on top of the 3DGS [19] framework using PyTorch" (p. 6).
- Licence: not stated. Dataset licences are listed in Appendix E (p. 16).

---

## 4. Mip-Splatting: Alias-free 3D Gaussian Splatting

### 4.1 Citation

- Title: Mip-Splatting: Alias-free 3D Gaussian Splatting
- First author: Zehao Yu, University of Tübingen and Tübingen AI Center
- Venue and year as stated: the PDF carries no venue line beyond the arXiv stamp (arXiv v1, 27 Nov 2023). The file name records CVPR 2024.
- arXiv id from file name: 2311.16493 (file: `mip_splatting_yu_cvpr2024_arxiv2311.16493.pdf`)

### 4.2 Per-primitive payload

- Base representation as in 3DGS (p. 3, Sec. 3.2): opacity α_k ∈ [0, 1], centre p_k ∈ ℝ³, covariance Σ_k = O_k s_k s_kᵀ O_kᵀ with scaling vector s ∈ ℝ³ and rotation O parameterised by a quaternion, plus spherical harmonics for view-dependent colour c_k (p. 4). SH degree: not stated. The method is built on the 3DGS code base with the same hyperparameters (p. 6).
- Added per-primitive quantity: the maximal sampling rate ν̂_k, recomputed every m = 100 iterations during training (p. 5 to 6). The 3D filter scale s / ν̂_k differs per primitive (p. 5). Whether it is stored as an extra float or folded into Σ_k and opacity after training is not stated. The paper says "Glow becomes an intrinsic part of the 3D representation, remaining constant post-training" (p. 5) and "the 3D smoothing filter can be fused with the Gaussian primitives per Eq. 9" (p. 8).
- Derived float count: 59 floats as in 3DGS (236 B fp32, 118 B fp16), or 60 floats (240 B fp32, 120 B fp16) if ν̂_k is kept as a separate scalar (derived, not stated).

### 4.3 Count control

- Unchanged from 3DGS: "Following [18], we train our models for 30K iterations across all scenes and use the same loss function, Gaussian density control strategy, schedule and hyper-parameters." (p. 6). No thresholds are restated. Primitive counts: not stated.
- Indirect effect of the filters on the count (p. 4): "simply discarding screen space dilation results in optimization challenges for complex scenes, such as those present in the Mip-NeRF 360 dataset [2], where a large number of small Gaussian are created by the density control mechanism [18], exceeding GPU capacity." And p. 11: "The absence of both the 3D smoothing filter and the 2D Mip filter leads to an excessive generation of small Gaussian primitives, due to the density control mechanism, resulting in out of memory error even on an A100 GPU with 40GB memory."

### 4.4 The two filters

3D smoothing filter (p. 5, Sec. 5.1):

- Sampling interval of a pixel back-projected to depth d with focal length f in pixels: T̂ = 1/ν̂ = d/f (Eq. 6). Nyquist gives reconstructable frequencies up to ν̂/2 = f/(2d), so "a primitive smaller than 2 T̂ may result in aliasing artifacts during the splatting process".
- Per-primitive maximal sampling rate over the N training cameras (Eq. 7): ν̂_k = max[n=1..N] ( 1_n(p_k) · f_n / d_n ), with 1_n(p_k) the indicator that the centre p_k lies in the frustum of camera n, and depth d approximated by the centre of the primitive, occlusion ignored. Recomputed every m = 100 iterations (p. 5 to 6).
- Filtered primitive (Eq. 9): G_k(x)_reg = √( |Σ_k| / |Σ_k + (s/ν̂_k) I| ) · exp( −½ (x − p_k)ᵀ (Σ_k + (s/ν̂_k) I)⁻¹ (x − p_k) ), obtained as the convolution G_k ⊗ G_low of two Gaussians (Eq. 8). s is a scalar hyperparameter, the 3D filter variance is 0.2 (p. 6).
- "By employing 3D Gaussian smoothing, we ensure that the highest frequency component of any Gaussian does not exceed half of its maximal sampling rate for at least one camera." (p. 5)

2D Mip filter (p. 5, Sec. 5.2):

- Replaces the 3DGS screen-space dilation G_k^2D(x) = exp( −½ (x − p_k)ᵀ (Σ_k^2D + s I)⁻¹ (x − p_k) ) (Eq. 5), which "adjusts the scale of the 2D Gaussian while leaving its maximum unchanged" and "is not mentioned in original paper" (p. 4, footnote 1).
- Mip filter (Eq. 10): G_k^2D(x)_mip = √( |Σ_k^2D| / |Σ_k^2D + s I| ) · exp( −½ (x − p_k)ᵀ (Σ_k^2D + s I)⁻¹ (x − p_k) ), "where s is chosen to cover a single pixel in screen space", variance 0.1 (p. 6). It approximates the box filter of the physical imaging process with a 2D Gaussian.
- Variance bookkeeping (p. 6): 2D Mip filter 0.1 plus 3D smoothing filter 0.2 "totaling 0.3 for a fair comparison with 3DGS [18] and 3DGS + EWA [59]".

Effect on primitive count: the density control strategy and hyperparameters are unchanged (p. 6). The paper reports no count, so the direct effect is not stated. The paper does state that without both filters the density control produces so many small Gaussians that training runs out of memory on a 40 GB A100 (p. 11), so the filters do change what ADC produces in practice.

Effect on render cost: "the sampling rate computation is the only prerequisite during training and the 3D smoothing filter can be fused with the Gaussian primitives per Eq. 9, thereby eliminating any additional overhead during rendering." (p. 8). The 2D Mip filter has the same functional form as the dilation it replaces plus a per-primitive scalar normalisation factor. Its render cost is not discussed. FPS: not stated. Training overhead: "a slight increase in training overhead as the sampling rate for each 3D Gaussian must be calculated every m = 100 iterations" (p. 7).

Ablations (p. 12, Table 5, Mip-NeRF 360 zoom-in, average PSNR / SSIM / LPIPS over 1×, 2×, 4×, 8×, reported): 3DGS 23.25 / 0.715 / 0.305, 3DGS + EWA 25.43 / 0.741 / 0.292, Mip-Splatting 27.37 / 0.803 / 0.252, w/o 3D smoothing filter 26.93 / 0.778 / 0.272, w/o 2D Mip filter 27.23 / 0.795 / 0.262. (p. 13, Table 6, Blender zoom-out average): 3DGS 24.84 / 0.890 / 0.063, 3DGS + EWA 29.40 / 0.960 / 0.034, 3DGS − Dilation 30.58 / 0.963 / 0.042, Mip-Splatting 31.97 / 0.974 / 0.024, w/o 3D smoothing filter 31.90 / 0.974 / 0.024, w/o 2D Mip filter 30.76 / 0.964 / 0.041.

### 4.5 Render-time cost

- FPS: not stated. GPU memory in training: only the 40 GB A100 out-of-memory statement for the no-filter ablation (p. 11). GPU memory in rendering: not stated. Storage in MB: not stated. Hardware for the main experiments: not stated (only "an A100 GPU with 40GB memory" in the ablation, p. 11).
- Training iterations: 30K (p. 6).

### 4.6 Quality numbers (reported)

Mip-NeRF 360, single-scale training and same-scale testing, indoor scenes downsampled by 2 and outdoor by 4 (p. 8, Table 4):

| method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 3DGS [18] (numbers from the 3DGS paper) | 27.21 | 0.815 | 0.214 |
| 3DGS [18]* (retrained by the authors) | 27.70 | 0.826 | 0.202 |
| 3DGS [18] + EWA [59] | 27.77 | 0.826 | 0.206 |
| Mip-Splatting (ours) | 27.79 | 0.827 | 0.203 |
| Zip-NeRF [3] | 28.54 | 0.828 | 0.189 |

Per-scene same-scale values are in Table 10 (p. 16). Mip-NeRF 360 zoom-in (trained at 1/8 resolution, p. 8, Table 3, PSNR at 1× / 2× / 4× / 8× / avg): 3DGS 29.19 / 23.50 / 20.71 / 19.59 / 23.25, Mip-Splatting 29.39 / 27.39 / 26.47 / 26.22 / 27.37. Tanks and Temples: not evaluated. Deep Blending: not evaluated. Counts and sizes: not stated.

### 4.7 Memory, storage, compactness, edge and budget statements (verbatim)

- p. 1: "This effectiveness and efficiency, coupled with the potential integration into the standard rasterization pipeline of GPUs represents a significant step towards practical usage of NVS methods."
- p. 2: "Our modifications to 3DGS are principled and simple, requiring only few changes to the original 3DGS code."
- p. 4: "However, simply discarding screen space dilation results in optimization challenges for complex scenes, such as those present in the Mip-NeRF 360 dataset [2], where a large number of small Gaussian are created by the density control mechanism [18], exceeding GPU capacity."
- p. 8: "As mentioned before, the sampling rate computation is the only prerequisite during training and the 3D smoothing filter can be fused with the Gaussian primitives per Eq. 9, thereby eliminating any additional overhead during rendering."
- p. 11: "The absence of both the 3D smoothing filter and the 2D Mip filter leads to an excessive generation of small Gaussian primitives, due to the density control mechanism, resulting in out of memory error even on an A100 GPU with 40GB memory. Hence, we don't report the result."
- Storage size, compactness, edge, mobile, low-end deployment or a budget: none found.

### 4.8 Limitations and future work (verbatim)

- p. 7 to 8, Sec. 6.4: "Our method employs a Gaussian filter as an approximation to a box filter for efficiency. However, this approximation introduces errors, particularly when the Gaussian is small in screen space. This issue correlates with our experimental findings, where increased zooming out leads to larger errors, as evidenced in Table 2. Additionally, there is a slight increase in training overhead as the sampling rate for each 3D Gaussian must be calculated every m = 100 iterations. Currently, this computation is performed using PyTorch [35] and a more efficient CUDA implementation could potentially reduce this overhead. Designing a better data structure for precomputing and storing the sampling rate, as it depends solely on the camera poses and intrinsics, is an avenue for future work. As mentioned before, the sampling rate computation is the only prerequisite during training and the 3D smoothing filter can be fused with the Gaussian primitives per Eq. 9, thereby eliminating any additional overhead during rendering."
- p. 7: "It's important to remark that rendering at higher resolutions is a super-resolution task, and models should not hallucinate high-frequency details absent from the training data."

### 4.9 Code

- Project page: https://niujinshuchong.github.io/mip-splatting (p. 1). Built on https://github.com/graphdeco-inria/gaussian-splatting (p. 6, footnote 3). Own repository URL: not stated in the paper.
- Licence: not stated.

---

## 5. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

### 5.1 Citation

- Title: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- First author: Binbin Huang, ShanghaiTech University
- Venue and year as stated: SIGGRAPH Conference Papers '24, July 27 to August 1, 2024, Denver, CO, USA (p. 1)
- arXiv id from file name: 2403.17888 (file: `2dgs_huang_siggraph2024_arxiv2403.17888.pdf`, arXiv v3, 22 Feb 2025)

### 5.2 Per-primitive payload (per surfel)

Stated on p. 4, Sec. 4.1:

| attribute | dimensionality | source |
|---|---|---|
| centre p_k | 3 | "central point p_k" |
| tangential vectors t_u, t_v (orientation, rotation R = [t_u, t_v, t_w] with t_w = t_u × t_v) | rotation with 2 free tangent directions, parameterisation not stated | "two principal tangential vectors t_u and t_v", "the rotation (t_u, t_v) are learnable parameters" |
| scaling (s_u, s_v) | 2 | "a scaling vector S = (s_u, s_v)", the 3×3 scaling matrix has last entry zero |
| opacity α | 1 | "Following 3DGS [Kerbl et al. 2023], each 2D Gaussian primitive has opacity α" |
| view-dependent colour c | SH, degree not stated | "view-dependent appearance c parameterized with spherical harmonics" |

- Total: not stated. Derived under the assumptions that rotation is a 4-float quaternion and SH has 4 bands (48 floats) as in the 3DGS code it builds on: 3 + 4 + 2 + 1 + 48 = 58 floats, 232 B at fp32, 116 B at fp16 (derived, both assumptions unstated in the paper). With two explicit 3-vectors for t_u and t_v the count would be 60 floats.
- Geometry as a homogeneous transform H = [s_u t_u, s_v t_v, 0, p_k; 0, 0, 0, 1] = [RS, p_k; 0, 1] (p. 4, Eq. 5), Gaussian value G(u) = exp(−(u² + v²)/2) in the tangent plane (Eq. 6).

### 5.3 Count control

- p. 5, Sec. 6.1: "During training, we increase the number of 2D Gaussian primitives following the adaptive control strategy in 3DGS. Since our method does not directly rely on the gradient of the projected 2D center, we hence project the gradient of 3D center p_k onto the screen space as an approximation. Similarly, we employ a gradient threshold of 0.0002 and remove splats with opacity lower than 0.05 every 3000 step."
- Densification interval, start and end, split and clone rules, scale thresholds: not restated (inherited from 3DGS).
- Total iterations: 30k for the main results (p. 7, Table 4 caption), 15k for DTU ablations (p. 7) and the 2DGS-15k rows (p. 6).
- Losses (p. 5, Eq. 16): L = L_c + α L_d + β L_n with L_c the 3DGS L1 plus D-SSIM loss, depth distortion L_d = Σ[i,j] ω_i ω_j |z_i − z_j| (Eq. 13), normal consistency L_n = Σ[i] ω_i (1 − n_iᵀ N) (Eq. 14), α = 1000 for bounded scenes, α = 100 for unbounded scenes, β = 0.05 for all scenes. In the appendix the distortion loss uses an L2 form on NDC depth with near and far planes 0.2 and 1000 (p. 9).
- Object-space low-pass filter (p. 5, Eq. 11): Ĝ(x) = max{ G(u(x)), G((x − c)/σ) } with σ = √2/2.
- Initialisation: sparse COLMAP point cloud (p. 4 to 6).
- Primitive counts: not stated.

### 5.4 Render-time cost

- Rendering (p. 4 to 5): explicit ray-splat intersection via three non-parallel planes (Eq. 8 to 10) instead of the affine projection, then the 3DGS-style rasteriser: per-primitive screen-space bounding box, sort by centre depth, tile lists, front-to-back α-blending terminated at opacity saturation (Eq. 12). Extra outputs for regularisation: depth distortion maps, depth maps and normal maps (p. 5).
- FPS: not stated as a number. The abstract claims "fast training speed, and real-time rendering" (p. 1).
- Hardware: "We conduct all the experiments on a single GTX RTX3090 GPU." (p. 5).
- Training time (reported): DTU 2DGS-15k 5.5 m, 2DGS-30k 10.9 m, 3DGS 11.2 m, SuGaR about 1 h (p. 6, Tables 1 and 3). Tanks and Temples 2DGS 15.5 m, 3DGS 14.3 m, SuGaR > 1 h, NeuS, Geo-Neus and Neuralangelo > 24 h (p. 6, Table 2).
- Storage (p. 6, Table 3, DTU, MB, reported): 3DGS 113, SuGaR 1247, 2DGS-15k 52, 2DGS-30k 52.
- GPU memory in training or rendering: not stated.

### 5.5 Quality numbers (reported)

Mip-NeRF 360 (p. 7, Table 4, 30k iterations, outdoor / indoor split):

| method | outdoor PSNR | outdoor SSIM | outdoor LPIPS | indoor PSNR | indoor SSIM | indoor LPIPS |
|---|---|---|---|---|---|---|
| 3DGS | 24.64 | 0.731 | 0.234 | 30.41 | 0.920 | 0.189 |
| SuGaR | 22.93 | 0.629 | 0.356 | 29.43 | 0.906 | 0.225 |
| 2DGS (Ours) | 24.34 | 0.717 | 0.246 | 30.40 | 0.916 | 0.195 |

Mip-NeRF 360 all 9 scenes (p. 10, Table 9, mean): 3DGS 27.20 / 0.815 / 0.214, SuGaR 25.82 / 0.752 / 0.298, 2DGS 27.03 / 0.805 / 0.223 (PSNR / SSIM / LPIPS).

Tanks and Temples (6 scenes: Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck): PSNR only (p. 10, Table 8, mean): SuGaR 24.16, 3DGS 25.33, 2DGS 25.56. Geometry F1 (p. 6, Table 2, mean): NeuS 0.38, Geo-Neus 0.35, Neuralangelo 0.50, SuGaR 0.19, 3DGS 0.09, 2DGS 0.32. SSIM and LPIPS on Tanks and Temples: not stated.

Deep Blending: not evaluated.

DTU Chamfer distance (p. 6, Table 1, mean): 3DGS 1.96, SuGaR 1.33, 2DGS-15k 0.83, 2DGS-30k 0.80. Counts: not stated. Size: 52 MB on DTU (p. 6, Table 3).

### 5.6 Memory, storage, compactness, edge and budget statements (verbatim)

- p. 1 (abstract): "We demonstrate that our differentiable renderer allows for noise-free and detailed geometry reconstruction while maintaining competitive appearance quality, fast training speed, and real-time rendering."
- p. 3: "It achieves similar or slightly better results compared to previous implicit neural surface representations, while being an order of magnitude faster."
- p. 6 (Table 3 caption): "Performance comparison between 2DGS (ours), 3DGS and SuGaR on the DTU dataset [Jensen et al. 2014]. We report the averaged chamfer distance, PSNR (training-set view), reconstruction time, and model size."
- p. 7: "Notably, our model demonstrates exceptional efficiency, offering a reconstruction speed that is approximately 100 times faster compared to implicit reconstruction methods and more than 3 times faster than the concurrent work SuGaR."
- GPU memory, compactness, edge, mobile, low-end deployment or a budget: none found (Mobile-NeRF appears only as a baseline row in Table 4, p. 7).

### 5.7 Limitations and future work (verbatim)

- p. 8, Sec. 7: "Limitations. While our method successfully delivers accurate appearance and geometry reconstruction for a wide range of objects and scenes, we also discuss its limitations: First, we assume surfaces with full opacity and extract meshes from multi-view depth maps. This can pose challenges in accurately handling semi-transparent surfaces, such as glass, due to their complex light transmission properties, as shown in Figure 12. Secondly, our current densification strategy favors texture-rich over geometry-rich areas, occasionally leading to less accurate representations of fine geometric structures. A more effective densification strategy could mitigate this issue. Finally, our regularization often involves a trade-off between image quality and geometry, and can potentially lead to over-smoothing in certain regions."
- p. 13 (Fig. 12 caption): "Our 2DGS struggles with the accurate reconstruction of semi-transparent surfaces, for example, the glass shown in example (A). Moreover, our method tends to create holes in areas with high light intensity, as shown in (B)."

### 5.8 Code

- Project page: https://surfsplatting.github.io (p. 1). Repository URL: not stated in the paper text. Built "upon the framework of 3DGS" with custom CUDA kernels (p. 5).
- Licence: not stated.

---

## 6. Cross-paper summary tables

### 6.1 Per-primitive payload

| method | stored unit | floats per unit | bytes fp32 | bytes fp16 | status |
|---|---|---|---|---|---|
| 3DGS | Gaussian | 59 (3 pos + 3 scale + 4 quat + 1 opacity + 48 SH, 4 bands) | 236 | 118 | derived from stated attributes, float total never printed |
| 3DGS-MCMC | Gaussian | 59 (same representation as 3DGS) | 236 | 118 | derived, SH degree not stated |
| Mip-Splatting | Gaussian | 59, or 60 if the per-primitive sampling rate ν̂_k is kept as its own scalar | 236 or 240 | 118 or 120 | derived, storage of ν̂_k not stated |
| 2DGS | surfel | 58 (3 pos + 4 quat + 2 scale + 1 opacity + 48 SH), 60 with two explicit tangent 3-vectors | 232 (240) | 116 (120) | derived, rotation parameterisation and SH degree not stated |
| Scaffold-GS | anchor | 68 (3 pos + 32 feature + 3 l_v + 30 offsets for k = 10), 71 with a separate 3-float base scale s_v | 272 (284) | 136 (142) | derived, plus about 7.2k shared MLP weights (derived, biases included, F_w excluded) |

### 6.2 Count-control mechanisms

| method | mechanism | explicit target count |
|---|---|---|
| 3DGS | gradient-threshold clone and split every 100 iterations, opacity prune, opacity reset every 3000 iterations, size prune | none, count emerges (1–5 M reported) |
| Scaffold-GS | voxel-gradient anchor growing (τ_g = 64ε, N = 100, multi-res with 4× finer voxels and 2× higher thresholds, random elimination) and accumulated-opacity anchor pruning (< 0.5) | none |
| 3DGS-MCMC | fixed maximum, live count grown by 5 % per relocation step until the cap, dead Gaussians (o < 0.005) relocated every 100 iterations, opacity and scale L1 regularisers (λ_o = 0.01, λ_Σ = 0.01) | yes, user-set maximum |
| Mip-Splatting | 3DGS ADC unchanged | none |
| 2DGS | 3DGS ADC with projected 3D-centre gradient, threshold 0.0002, prune opacity < 0.05 every 3000 steps | none |

### 6.3 Reported speed and size on the three benchmark datasets

| method | hardware | Mip-NeRF360 FPS / MB | T&T FPS / MB | DB FPS / MB |
|---|---|---|---|---|
| 3DGS-30K | A6000 | 134 / 734 | 154 / 411 | 137 / 676 |
| Scaffold-GS | not named | 102 / 156 | 110 / 87 | 139 / 66 |
| 3DGS-MCMC | not stated | same render time as 3DGS, no numbers | no numbers | no numbers |
| Mip-Splatting | not stated (A100 40 GB in one ablation) | not stated | not evaluated | not evaluated |
| 2DGS | RTX 3090 | not stated (52 MB on DTU) | not stated | not evaluated |
