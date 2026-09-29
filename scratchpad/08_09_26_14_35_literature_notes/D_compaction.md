# Literature notes D: early compaction and compression baselines

Four papers read page by page from the PDFs under `references/04_compaction_pruning/`. Page numbers
below are PDF page indices, which coincide with the printed page numbers in all four PDFs. Every
number is copied as reported by the paper. Where the paper does not say something the field reads
"not stated". Verbatim quotes are in double quotes with the page in parentheses.

Reading frame for the project: a 3DGS training method that takes a byte (later ms per frame)
budget and drives the model onto it during training. Direct ancestor SizeGS is post-hoc. Claimed
gap: no published method binds a byte target during training.

---

## Paper A: LightGaussian

### 1. Citation

- Title: "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS"
- First author: Zhiwen Fan (co-first with Kevin Wang), with Kairun Wen, Zehao Zhu, Dejia Xu,
  Zhangyang Wang. The University of Texas at Austin and XMU.
- Venue as stated in the PDF: "38th Conference on Neural Information Processing Systems (NeurIPS
  2024)." (p. 1). PDF stamp: arXiv:2311.17245v6 [cs.CV] 12 Nov 2024.
- arXiv id: 2311.17245

### 2. Base representation and renderer

- Base: vanilla 3D-GS [Kerbl et al.] trained from SfM points. "Our framework is implemented in
  PyTorch and integrates the differentiable Gaussian rasterization technique from 3D-GS [16]."
  (p. 7).
- Compressed form (p. 6, Fig. 2 p. 3): pruned Gaussian set, SH degree reduced from 3 to 2 by
  distillation, SH coefficients of the least significant 60 % of Gaussians replaced by indices into
  an 8192 entry codebook, everything else kept in float16: "To preserve essential attributes,
  including Spherical Harmonics with higher global significance scores, along with Gaussian
  position, shape, rotation, and opacity, we skip VQ for these elements and store them directly in
  float16 format." (p. 6).
- Rendered directly: not stated. The FPS gain is attributed to fewer Gaussians, not to reading the
  compressed layout: "Our method, LightGaussian, exceeds existing techniques with rendering speeds
  over 200 FPS, enabled by efficient rasterization that prunes insignificant Gaussians." (p. 8).
  Implied pipeline: codebook lookup plus float16 to float32 cast into the stock 3D-GS attribute
  layout, then the stock rasterizer.
- MLP or hash grid at render time: none for the 3D-GS variant. The Scaffold-GS experiment (Table 5
  p. 8) inherits Scaffold-GS's MLPs, which is a property of that base, not of LightGaussian.

### 3. Budget handle

KNOB. The user sets, in order of the pipeline:

- Pruning ratio: fraction of Gaussians removed by global significance rank. Value used for the
  main table is not stated in the text. "With a Gaussian pruning ratio of 66% and SH reduced to
  2 degrees, LightGaussian nearly maintains rendering quality" (Table 6 caption p. 9). Fig. 6a
  (p. 9) sweeps the prune ratio. Fig. 1 (p. 1) shows "Ours (0.5M points, 0.916)" against "3D-GS
  (1.4M points,0.917)".
- Volume power β = 0.1 in the significance score (p. 7).
- SH degree: 3 to 2 (p. 7), 1 degree also ablated (Table 6 p. 9).
- Pseudo-view noise σ = 0.1 (p. 7).
- VQ ratio: "selecting SHs with the least 60% significance score for the vector quantization (VQ
  ratio)" (p. 7). Fig. 6b sweeps it.
- Codebook size 8192 (p. 7).
- Storage precision: float16 for all non-VQ attributes (p. 6).

No target of any kind is accepted. The pruning ratio is a count fraction, so it fixes the Gaussian
count exactly, but it is set by hand and post-hoc.

### 4. When it binds

Post-hoc on a fully trained 3D-GS. Algorithm 1 (p. 15):

1. "Pre-Training 3D-GS" (iteration count not stated, the 3D-GS default is implied).
2. Global significance calculation over all training views, prune, then "Gaussian Co-adaptation"
   fine-tune "for 5,000 iterations" "without additional densification" (p. 5, p. 7).
3. SH distillation from degree 3 teacher to degree 2 student with pseudo views, "while Few Steps
   do" (Alg. 1 p. 15), iteration count not stated.
4. VQ of the least significant SHs with K-means init, then "We fine-tune the codebook for 5,000
   iterations, while fixing the gaussian-to-codebook mapping. We disable additional clone/split
   operations and leverage photometric loss on the training views." (p. 6).

Wall-clock time of the compression pipeline: not stated.

### 5. What "size" means

"Size" column of Table 1 in MB, called "storage" in the text: "reducing storage from 782MB to
45MB" (p. 1). What is counted is not stated explicitly. Table 4 (p. 7) shows the accounting
implicitly: Baseline 80.99 MB, +FP16 36.51 MB, +VQ SH × GS 21.10 MB, so the figure counts the float16
attribute arrays plus the VQ'd SH. Whether the codebook itself is included: not stated.

### 6. Runtime cost reported

- Hardware: "All performance evaluations are conducted on an A6000 GPU." (p. 7). Table 1 baselines
  re-trained "on our platform (NVIDIA A6000 GPU)" (p. 6).
- Render FPS (Table 1 p. 6): Mip-NeRF 360, 3D-GS* 144 FPS, Ours 237 FPS. Tanks and Temples, 3D-GS*
  106 FPS, Ours 357 FPS. Room scene (Table 2 p. 7): 156.21 to 302.01 FPS. Scaffold-GS (Table 5
  p. 8): 152 to 178 FPS. Synthetic-NeRF (p. 16): 310 to 411 FPS.
- Encode time: not stated (only the 5 000 + 5 000 iteration counts).
- Decode time: not stated.
- GPU memory: not stated.
- Mobile or low-end device results: none.

### 7. Size model

No size model. Size is the outcome of prune ratio, SH degree, VQ ratio, codebook size and float16.
Nothing about size is differentiable or predicted during any stage. The only rate-distortion
material is Fig. 6 (p. 9): "(a). SSIM, FPS vs. Prune Ratio" and "(b). SSIM, Quantize Ratio vs. VQ
Ratio", where "the (overall) quantization ratio reflects combined compression from VQ and FP32-to-
FP16 bitwise quantization." (p. 9). Text summary (p. 10): "we note a marked decline in rendering
quality when the pruning ratio reaches 70%, with a more rapid deterioration as the VQ ratio
approaches 65%."

### 8. Scene dependence

Per-scene sizes on Mip-NeRF 360 and Tanks and Temples: not stated (Tables 8 to 10 on pp. 17 to 19
list PSNR, SSIM and LPIPS only). Per-scene sizes exist for Synthetic-NeRF (Table 7 p. 17), 3D-GS
to Ours, MB, as reported:

- chair 94.612 to 13.785
- drums 64.163 to 9.596
- ficus 35.839 to 5.473
- lego 64.910 to 9.600
- ship 56.400 to 8.464
- average 52.381 to 7.838

Room (Mip-NeRF 360): 365 MB to 21 MB (Table 2 p. 7). Room in Table 6 (p. 9): 353.27 MB baseline,
140.16 MB with 66 % pruning and 2-degree SH before VQ, 88.89 MB with 1-degree SH. Fig. 5 (p. 9)
point counts before and after pruning: 1.8M to 0.7M and 5.6M to 2.3M.

### 9. Main quantitative results (Table 1 p. 6, as reported)

Columns: FPS, Size, PSNR, SSIM, LPIPS.

Mip-NeRF 360:

| method | FPS | Size | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| Ours | 237 | 45MB | 27.13 | 0.806 | 0.237 |
| 3D-GS* (re-trained) | 144 | 782MB | 27.40 | 0.813 | 0.217 |
| 3D-GS [16] (paper numbers) | 134 | 734MB | 27.21 | 0.815 | 0.214 |
| Compressed 3D-GS* [34] | 152 | 28MB | 27.03 | 0.802 | 0.238 |
| Compact 3D-GS [37] | 128 | 48MB | 27.08 | 0.798 | 0.247 |

Tanks and Temples:

| method | FPS | Size | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| Ours | 357 | 25MB | 23.44 | 0.832 | 0.202 |
| 3D-GS* | 106 | 433MB | 23.66 | 0.845 | 0.178 |
| 3D-GS [16] | 154 | 411MB | 23.14 | 0.841 | 0.183 |
| Compressed 3D-GS* | 202 | 17MB | 23.54 | 0.838 | 0.189 |
| Compact 3D-GS | 185 | 39MB | 23.32 | 0.831 | 0.201 |

Deep Blending: not evaluated. Gaussian counts: no table, only Fig. 1 (0.5M vs 1.4M points) and
Fig. 5. Internal inconsistency worth knowing: the text says of Tanks and Temples "reducing storage
from 380MB to 22MB" (p. 8) while Table 1 says 433MB to 25MB.

Strongest baselines in their own table: Compressed 3D-GS* (28MB / 27.03 PSNR on Mip-NeRF 360) and
3D-GS* (27.40 PSNR).

### 10. Densification and pruning

- Densification: stock 3D-GS during pre-training. Off in every later stage ("without additional
  densification" p. 5, "We disable additional clone/split operations" p. 6).
- Pruning score: global significance, Eq. 3 (p. 4):
  GS_j = ∑[i=1..MHW] 1(G(X_j), r_i) · σ_j · T · γ(Σ_j), with T = ∏[i=1..j−1] (1 − σ_i) the
  transmittance and γ(Σ) = (V_norm)^β, V_norm = min(V(Σ)/V_max90, 1), V = (4/3)π·abc (pp. 4 to 5).
  So: hit count over all training pixels × opacity × transmittance × clipped normalized volume to the
  power β = 0.1. Ablation in Table 3 (p. 7): hit count only, zero-shot, drops Room PSNR 31.34 to 28.16.
  Opacity-only zero-shot pruning fails (Fig. 3 p. 4, PSNR 27.2 to 25.3).
- After pruning: "The remaining Gaussians are then jointly adapted by fine-tuning their attributes
  ... to offset the minor loss from pruning. This adaptation is performed using photometric loss on
  the original training views, aligned with 3D-GS training for 5,000 iterations." (p. 5). Table 2
  (p. 7): pruning alone 31.34 to 30.67 PSNR, co-adaptation back to 31.64.
- Scaffold-GS variant: "pruning 80% of neural Gaussians increases the rendering speed from 152 to
  178 FPS" (p. 8).

### 11. Stated limitations and future work (verbatim)

- "Exploring zero-shot compression across various 3D-GS-based frameworks remains a promising
  direction for future research." (p. 10)
- "While we do not foresee any significant risk from this specific technique, 3D reconstruction may
  infringe on personal privacy when applied in public spaces or with drone footage, contradicting
  our ethical intentions." (p. 10)
- "we note a marked decline in rendering quality when the pruning ratio reaches 70%, with a more
  rapid deterioration as the VQ ratio approaches 65%." (p. 10)

No dedicated limitations paragraph beyond these.

### 12. Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, budget, or edge, mobile, web, low-end deployment (verbatim)

None of the following names a target, a budget or rate control. They are the closest passages.

- "LightGaussian achieves an average 15× compression rate while boosting FPS from 144 to 237
  within the 3D-GS framework, enabling efficient complex scene representation on the Mip-NeRF 360
  and Tank & Temple datasets." (p. 1)
- "However, NeRF and its variants face limitations in rendering speed, limiting their deployment in
  real-world scenarios." (p. 2)
- "Finally, vector quantization, including codebook initialization and assignment, reduces model
  bandwidth." (p. 3, Fig. 2 caption)
- "We aim to strike a balance between rendering quality loss and compression rate by leveraging the
  pre-computed significance score from Eq. 3." (p. 5)
- "In the Gaussian VQ step, the codebook size is configured to 8192, selecting SHs with the least
  60% significance score for the vector quantization (VQ ratio), to balance the trade-off between
  compression efficiency and fidelity." (p. 7)
- "Codebook quantization further reduces the required model size and bandwidth (#7, #8)." (p. 7,
  Table 2 caption)
- "We investigate the interplay between rendering quality and speed across various model
  compression ratios. Specifically, we adjust parameters such as Gaussian Pruning & Recovery, SH
  Distilling, and VQ." (pp. 9 to 10)
- "Broadly, LightGaussian's compact representation has the potential to democratize high-quality 3D
  content for applications in VR, AR, and autonomous driving by reducing resource demands, enabling
  more accessible and scalable deployment across industries." (p. 10)

Byte or FPS target during training: none found. Rate control: none found. Budget: none found.
Mobile, edge, web, low-end: none found.

### 13. Code

- "Project Website: https://lightgaussian.github.io" (p. 1). No repository URL is printed in the
  paper.
- Licence: not stated.

---

## Paper B: Compact 3D Gaussian Representation for Radiance Field (Compact 3DGS)

### 1. Citation

- Title: "Compact 3D Gaussian Representation for Radiance Field"
- First author: Joo Chan Lee, with Daniel Rho, Xiangyu Sun, Jong Hwan Ko, Eunbyung Park.
  Sungkyunkwan University and KT.
- Venue as stated in the PDF: not stated. PDF stamp: arXiv:2311.13681v2 [cs.CV] 15 Feb 2024. The
  file name records CVPR 2024.
- arXiv id: 2311.13681

### 2. Base representation and renderer

- Base: 3DGS with three replacements (Fig. 2 p. 4, Sec. 3):
  1. Learnable binary mask on scale and opacity via a straight-through estimator (Sec. 3.1 p. 4).
  2. View-dependent colour from a hash grid plus tiny MLP instead of SH: "We exploit hash grids [30]
     followed by a tiny MLP to represent color. Here, we input positions into the hash grids, and
     then the resulting feature and the view direction are fed into the MLP." (p. 5). Positions are
     contracted (Eq. 8 p. 5). Configuration: "hash grids with 2-channel features across 16 different
     resolutions (16 to 4096) and a following 2-layer 64-channel MLP", max hash map size 2¹⁹ for
     real scenes, 2¹⁶ for synthetic (p. 11).
  3. Scale and rotation from residual vector quantization, "codebook size C and the number of
     stages L of R-VQ to 64 and 6" (p. 11).
- Rendered directly: no. The renderer needs an MLP per view. "Second, by precomputing grid features
  at the testing phase, we minimize the operational time of the neural field. Since these grid
  features, which precede the subsequent MLP input, are not dependent on the view direction itself,
  they can be prepared in advance of testing. This allows our method to simply process a small MLP
  for generating view-dependent colors during testing. Third, in the testing phase, the time spent
  searching for suitable geometry is eliminated. In a manner similar to precomputing grid features,
  we can index the closest codes from multi-stage codebooks before testing." (p. 11). So the
  R-VQ codebooks are decoded to full scale and rotation once, the hash grid is evaluated once per
  Gaussian, and the MLP runs per view before the stock 3DGS rasterizer.
- MLP at render time: yes. Hash grid at render time: evaluated once, precomputable.

### 3. Budget handle

KNOB. Hyper-parameters (pp. 5, 6, 11):

- λ_m, "where λm is a hyperparameter to regularize the number of Gaussians." (p. 5). Values: "the
  control factor for the number of Gaussians λm to 5e−4" for real scenes, "4e−3" for synthetic
  (p. 11).
- Masking threshold ϵ (Eq. 1 p. 4), value not stated.
- Max hash map size 2¹⁹ / 2¹⁶ (p. 11).
- R-VQ codebook size C = 64, stages L = 6 (p. 11).
- Post-processing (Ours+PP): "Applying 8-bit min-max quantization to opacity and hash grid
  parameters. Pruning hash grid parameters with values below 0.1. Applying Huffman encoding [17] on
  the quantized opacity and hash parameters, and R-VQ indices." (p. 6).

No target. The rate-distortion curve of Fig. 8 (p. 12) is generated by "doubling each
hyper-parameter" (p. 11), not by a target.

### 4. When it binds

During training from iteration 0 for the count, post-hoc for the byte-level steps.

- Masking: "At every densification, we eliminate the Gaussians according to the binary mask.
  Furthermore, unlike the original 3DGS that stops densifying in the middle of the training and
  retains the number of Gaussians to the end, we consistently mask out along with the entire
  training process, reducing unessential Gaussians effectively and ensuring efficient computation
  with low GPU memory throughout the training phase (Fig. 3)." (p. 4).
- R-VQ: "we apply R-VQ and learn codebooks with K-means initialization, only for the last 1K
  training iterations. Except for that period, we set Lr, Ls to zero." (p. 5).
- Total: "trained during 30K iterations" (p. 6, p. 11). Positions and opacities stored as
  half-tensors (p. 6).
- Post-processing (+PP) is post-hoc after the 30K iterations (p. 6).
- Training time (Tables 1 and 2 p. 6, NVIDIA A100): Mip-NeRF 360, Ours 33m 06s vs 3DGS* 24m 07s.
  Tanks and Temples 18m 20s vs 13m 51s. Deep Blending 27m 33s vs 21m 52s. Synthetic (Table 3)
  8m 04s vs 6m 14s.

### 5. What "size" means

"Storage" in MB. Table 5 (p. 8) gives the accounting, average over Mip-NeRF 360, as reported:

| variant | Pos. | Opa. | Sca. | Rot. | Col. | MLP | Tot. |
|---|---|---|---|---|---|---|---|
| 3DGS (32f) | 37.9 | 12.6 | 37.9 | 50.6 | 606.9 | | 746 |
| Ours (Pos 16f, Opa 16f, Sca/Rot R-VQ, Col Hash 16f, MLP 16f) | 8.3 | 2.8 | 6.3 | 6.3 | 25.2 | 0.016 | 48.8 |
| Ours+PP (Opa 8b+H, Sca +H, Rot +H, Col +8b+P+H) | 8.3 | 1.2 | 5.9 | 6.2 | 7.4 | 0.016 | 29.1 |

So size includes positions, opacities, R-VQ indices and codebooks, the hash grid and the MLP
weights. The mask parameter is not stored: "Once training is completed, the mask parameter m does
not need to be stored since we removed the masked Gaussians." (p. 4). Comment on the colour part:
"While our end-to-end trainable framework demonstrates significant effectiveness, it requires
relatively large storage for color representation." (p. 8).

### 6. Runtime cost reported

- Hardware: "we re-evaluate 3DGS with the same training configurations as our method using an
  NVIDIA A100 GPU (denoted as 3DGS*)" (p. 6).
- FPS (Tables 1 to 3 p. 6): Mip-NeRF 360, Ours 128 vs 3DGS* 120. Tanks and Temples 185 vs 160.
  Deep Blending 181 vs 132. Synthetic 545 vs 359.
- Training time: see field 4.
- GPU memory (Table 8 p. 12, described as "the GPU memory requirement for inference" on p. 12):
  bicycle 3DGS 9.4 GB, Ours 7.6 GB. bonsai 8.7 GB vs 8.3 GB. drjohnson 7.5 GB vs 6.5 GB. playroom
  6.4 GB vs 5.6 GB.
- Encode and decode time: not stated.
- Mobile or low-end device results: none.

### 7. Size model

No byte model, nothing predicted. The count is regularised by a differentiable proxy:
L_m = (1/N) ∑[n=1..N] σ(m_n) (Eq. 3 p. 4), weighted by λ_m in L = L_ren + λ_m L_m + L_r + L_s
(Eq. 9 p. 5). This is a count penalty, not a byte penalty, and λ_m is a knob. Table 6 (p. 11,
Bonsai) shows the λ_m to count relation, as reported:

| mask on | λ_m | #Gaussian | PSNR |
|---|---|---|---|
| opacity | 0.0005 | 311786 | 31.50 |
| opacity | 0.0001 | 629837 | 31.86 |
| scale | 0.0005 | 508762 | 31.89 |
| scale | 0.0003 | 596449 | 31.97 |
| opacity + scale | 0.0005 | 601048 | 32.08 |

Rate-distortion curves (Fig. 8 p. 12) sweep λ_m, hash map size and R-VQ stages on Playroom,
Bonsai and Bicycle, with the finding "the storage need of our method can be halved with minimal
performance loss" (p. 11).

### 8. Scene dependence

Table 9 (p. 13), Ours storage in MB at the fixed real-scene setting, as reported: bicycle 62.99,
flowers 51.15, garden 62.78, stump 54.66, tree hill 59.33, room 34.21, counter 34.34, kitchen
44.45, bonsai 35.44, average 48.82. Ours+PP: bicycle 42.42, room 15.01, bonsai 16.40, average
29.07. Table 10 (p. 13): train 37.29, truck 41.57, drjohnson 47.98, playroom 38.45 (Ours), and
19.07, 22.64, 28.43, 19.21 (+PP). Gaussian counts, Ours: bicycle 2 221 689, room 529 136, bonsai
601 048, average 1 388 162. Spread across Mip-NeRF 360 at one λ_m: 34.21 to 62.99 MB, a factor of
1.84.

### 9. Main quantitative results (Tables 1 and 2 p. 6, as reported)

Columns: PSNR, SSIM, LPIPS, Train, FPS, Storage.

Mip-NeRF 360:

| method | PSNR | SSIM | LPIPS | Train | FPS | Storage |
|---|---|---|---|---|---|---|
| Ours | 27.08 | 0.798 | 0.247 | 33m 06s | 128 | 48.8 MB |
| Ours+PP | 27.03 | 0.797 | 0.247 | - | - | 29.1 MB |
| 3DGS* | 27.46 | 0.812 | 0.222 | 24m 07s | 120 | 746 MB |
| 3DGS (paper) | 27.21 | 0.815 | 0.214 | 41m 33s | 134 | 734 MB |
| Mip-NeRF 360 | 27.69 | 0.792 | 0.237 | 48h | 0.06 | 8.6 MB |

Tanks and Temples:

| method | PSNR | SSIM | LPIPS | Train | FPS | Storage |
|---|---|---|---|---|---|---|
| Ours | 23.32 | 0.831 | 0.201 | 18m 20s | 185 | 39.4 MB |
| Ours+PP | 23.32 | 0.831 | 0.202 | - | - | 20.9 MB |
| 3DGS* | 23.71 | 0.845 | 0.178 | 13m 51s | 160 | 432 MB |
| 3DGS (paper) | 23.14 | 0.841 | 0.183 | 26m 54s | 154 | 411 MB |

Deep Blending:

| method | PSNR | SSIM | LPIPS | Train | FPS | Storage |
|---|---|---|---|---|---|---|
| Ours | 29.79 | 0.901 | 0.258 | 27m 33s | 181 | 43.2MB |
| Ours+PP | 29.73 | 0.900 | 0.258 | - | - | 23.8MB |
| 3DGS* | 29.46 | 0.900 | 0.247 | 21m 52s | 132 | 663 MB |
| 3DGS (paper) | 29.41 | 0.903 | 0.243 | 36m 02s | 137 | 676 MB |
| Mip-NeRF 360 | 29.40 | 0.901 | 0.245 | 48h | 0.09 | 8.6 MB |

Counts (Tables 9 and 10 p. 13, average #Gaussians, 3DGS* to Ours): Mip-NeRF 360 3 161 131 to
1 388 162. Tanks and Temples 1 831 627 to 836 296. Deep Blending 2 810 698 to 1 058 679. Strongest
baselines in their table are 3DGS* and Mip-NeRF 360 (the only other 3DGS compressor is absent).

### 10. Densification and pruning

- Densification: stock. "We retained all hyper-parameters of 3DGS and trained models during 30K
  iterations" (p. 11).
- Pruning: the learned mask, M_n = sg(1[σ(m_n) > ϵ] − σ(m_n)) + σ(m_n), ŝ_n = M_n s_n,
  ô_n = M_n o_n (Eqs. 1 and 2 p. 4). "We apply binary masks not only on the opacities o ∈ [0, 1]^N
  but also on the non-negative scale attributes" (p. 4). Masked Gaussians are deleted at every
  densification step and the masking runs through all 30K iterations (p. 4). The score is a learned
  parameter, not a hand-crafted statistic.
- Fine-tuning after pruning: none as a separate stage, because removal is continuous inside
  training. Table 4 (p. 8): mask alone on Playroom takes 2.34 M Gaussians to 967 K at 29.87 to
  29.91 PSNR and 154 to 254 FPS.
- Table 7 (p. 11): representing opacity, scale or rotation with the hash grid fails (PSNR 9.3), so
  the grid is used for colour only.

### 11. Stated limitations and future work (verbatim)

No dedicated section. Passages that state a limitation:

- "The proposed color representation based on the neural field offers more than a threefold
  improvement in storage efficiency with a slightly reduced number of Gaussians compared to the
  directly storing high-degree SH, despite necessitating slightly more time for training and
  rendering." (p. 7)
- "While our end-to-end trainable framework demonstrates significant effectiveness, it requires
  relatively large storage for color representation. Nonetheless, as indicated in the table, this
  can be effectively reduced through simple post-processings." (p. 8)
- "The main paper's analysis shows that our method extends the overall training time slightly more
  than 3DGS, owing to the time for iterating neural fields and for R-VQ search." (p. 11)
- "Although the proposed geometry codebook effectively reduces the storage as validated in the main
  paper, it shows poor performance when the diversity of codebooks is extremely limited." (p. 11)

Future work: not stated.

### 12. Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, budget, or edge, mobile, web, low-end deployment (verbatim)

- "Since it demands dense point sampling along the ray to render a pixel, which requires
  significant computational resources, NeRFs often fail to achieve real-time rendering on hand-held
  devices or low-end GPUs. This challenge limits their use in practical scenarios where fast
  rendering speed is essential, such as various interactive 3D applications." (p. 2)
- "where λm is a hyperparameter to regularize the number of Gaussians." (p. 5)
- "For the real scenes, we set the max size of hash maps to 2^19, the control factor for the number
  of Gaussians λm to 5e−4, and the learning rate of the mask parameter and the neural fields to
  1e−2." (p. 11)
- "8.1. Rate-distortion curve based on each proposal" (section heading, p. 11)
- "We control λm, max hashmap size, and the number of R-VQ stages, respectively, by doubling each
  hyper-parameter." (p. 11)
- "In contrast, our neural field-based color representation demonstrates remarkable robustness even
  in a low-rate condition across the various scenes, indicating that scaling down the neural field
  is the best option in environments with severe resource limitations." (p. 11)
- "These results show that the storage need of our method can be halved with minimal performance
  loss, and our default configuration is a well-rounded choice for a wide range of scenes." (p. 11)
- "Our approach effectively reduces not only storage needs but also memory requirements, both of
  which are key aspects of efficiency." (p. 12)

Byte or FPS target during training: none found. Rate control: none found. Budget: none found. The
λ_m regulariser acts during training but is a knob on the count, not a target on bytes.

### 13. Code

- "Our project page is available at https://maincold2.github.io/c3dgs/." (p. 1). No repository
  URL printed.
- Licence: not stated.

---

## Paper C: Self-Organizing Gaussian Grids (SOG)

### 1. Citation

- Title: "Compact 3D Scene Representation via Self-Organizing Gaussian Grids"
- First author: Wieland Morgenstern, with Florian Barthel, Anna Hilsmann, Peter Eisert. Fraunhofer
  Heinrich Hertz Institute and Humboldt University of Berlin.
- Venue as stated in the PDF: not stated. The PDF uses the LNCS template with the running head
  "Abbreviated paper title". PDF stamp: arXiv:2312.13299v2 [cs.CV] 2 May 2024.
- arXiv id: 2312.13299

### 2. Base representation and renderer

- Base: stock 3DGS attribute set (position, SH DC, SH rest, opacity, scale, rotation), trained
  with the official implementation plus two additions: periodic sorting of all Gaussians into a
  square 2D grid (PLAS, Sec. 3.1) and a smoothness loss on the sorted grids (Sec. 3.2). Positions
  use a logarithmic contraction, x = sign(x_log) × (exp(|x_log|) − 1) (Eq. 1 p. 9).
- Compressed form: one multi-channel image per attribute, quantized and encoded with JPEG XL
  (Sec. 3.4 p. 9).
- Rendered directly: no. "The uncompressed Gaussians use the same structure as 3DGS, ensuring a
  seamless integration with established renderers." (p. 1). "We provide a simple to use interface
  for compressing and decompressing the resulting 3D scenes. The decompressed reconstructions share
  the structure of 3DGS, allowing integration into established renderers." (p. 3). Rendering FPS is
  quoted "in the 3DGS viewer" (p. 11).
- MLP or hash grid at render time: none.

### 3. Budget handle

KNOB. From Table 2 of the appendix (p. 19) and Sec. 3.4 (p. 9):

- Smoothness loss "Overall multiplier λ" = 1.0, kernel size 5, sigma 3, per-attribute smoothness
  weights: opacity 0.09, rotation 0.91, position, colour, scaling and SH rest 0.0.
- Sorting weights: position 1.0, colour 1.0, scaling 1.0, opacity 0.0, rotation 0.0, SH rest 0.0.
- Quantization levels: "qcoords = 2^14, qscale = qopacity = qrotation = 2^6 and qSH_rest = 2^5"
  (p. 9).
- Codec setting: "We compress the RGB grid with lossy JPEG XL, as an 8-Bit image with a quality
  level of 100. All other attributes are stored as lossless JPEG XL." (pp. 9 to 10).
- Densification parameters (see field 10), which set the Gaussian count.
- With or without SH ("w/o SH" variants in Table 1).

The λ sweep (Table 2 p. 12) is the only knob-to-size curve: λ = 0.01 gives 18.02 MB at 24.39
PSNR, λ = 1.5 gives 10.73 MB at 24.02 PSNR (scene not stated).

### 4. When it binds

During training for the arrangement, post-hoc for the bytes.

- Sorting: "At the start of the training, we sort once, and continue sorting with each
  densification step." (p. 8).
- Smoothness loss: applied throughout training and back-propagated through the renderer: "As we
  propagate the gradient of the smoothness regularization through the Gaussian renderer during
  training, we specifically enforce the 3D Gaussian Splatting optimization process to prefer
  Gaussians that improve the quality of the rendering while also considering the respective local
  smoothness." (p. 9).
- Quantization and JPEG XL: once after training. "Once training has finished, we can use
  off-the-shelf image compression methods to store the data on disk." (p. 9). "it would be
  interesting to perform the quantization during training, which is currently happening only once
  before the compression." (p. 14).
- Iteration count: not stated (Fig. 7 caption mentions "Until iteration 15000" on p. 12).
- Time: "our method only uses 10 to 30 minutes" and "we achieve the same training time as 3DGS"
  (pp. 10 to 11). Sorting "will usually take less than 10 seconds during training" (pp. 21 to 22),
  the Garden scene with 4.37M Gaussians (grid side 2091) sorts "in well below a minute" (p. 22).

### 5. What "size" means

File size of the JPEG XL encoded attribute grids: "All sizes are in MB." (Table 1 caption p. 10).
All six attribute groups are included as images. Whether per-attribute min and max metadata is
counted: not stated. Table 4 (p. 13) compares containers for the same trained Truck scene (w/o SH):
PLY 104.89, NPZ 76.10, JPEG XL lossless 66.05, PNG 16 37.00, EXR 30.56, JPEG XL quantized 11.86 MB.

### 6. Runtime cost reported

- Render FPS: "The vanilla Truck with 2.58M Gaussians renders at 385 fps with the default settings
  on the same GPU, while our scene with 1.55M Gaussians renders at 515 fps with better visual
  quality." (p. 11). GPU for that sentence not named. Sorting benchmarks use "an Nvidia RTX 4090"
  (p. 21).
- Sorting: random 512×512×3 grid, RTX 4090, "5.7 seconds after 8015 reorders" vs FLAS on an AMD
  Ryzen Threadripper PRO 5955WX in 131 s (p. 21).
- Training time: see field 4.
- Encode time, decode time, GPU memory: not stated.
- Mobile or web results: none, only motivation.

### 7. Size model

No size model, nothing differentiable in bytes. The smoothness loss is a proxy that makes the
grids cheaper to code but does not measure bytes.

- Codec: JPEG XL. RGB (SH DC) grid lossy 8-bit at quality 100, every other attribute quantized to
  q levels then lossless JPEG XL (pp. 9 to 10). "Our method is not restricted to this particular
  codec, but could also be used with other, existing 2D coding techniques." (p. 10).
- Dependence on grid layout: a single square grid of side ⌊√N⌋ shared by all attributes. "we find
  the largest square grid size that can be completely filled with Gaussians, and discard the
  Gaussians with the lowest opacity values that will not fit." (p. 14). One permutation for all
  attributes: "Therefore, the sorting algorithm has to find one permutation that satisfies all
  attributes at once. Otherwise, it would require additional storage size to align the attributes of
  the sorted grids." (p. 6).
- Dependence on λ: Table 2 (p. 12), as reported: λ 0.01 / 18.02 MB / 24.39 PSNR, 0.05 / 18.35 /
  24.48, 0.1 / 17.50 / 24.18, 0.5 / 14.62 / 24.40, 1.0 / 12.43 / 24.24, 1.5 / 10.73 / 24.02.
- Dependence on quantization and container: Table 4 (field 5).
- Dependence on SH: Mip-NeRF 360 40.3 MB with SH vs 16.7 MB without (Table 1 p. 10).
- Targeting size through codec quality: not stated. The paper picks a fixed quality and fixed
  quantization levels and reports what comes out.

### 8. Scene dependence

No per-scene table in this PDF. Available per-scene numbers, as reported:

- Truck with SH: 34.3 MB at 25.37 PSNR. Truck w/o SH: 17.3 MB at 24.90 PSNR (Table 3 p. 12).
- Truck w/o SH as JPEG XL quantized: 11.86 MB at 25.14 PSNR (Table 4 p. 13). The two Truck w/o SH
  figures differ (17.3 vs 11.86 MB) and the paper does not reconcile them.
- Truck PLY trained with their parameters: 104.89 MB (Table 4), vs vanilla 3DGS PLY 624.7 MB and
  .ply.zip 548.2 MB (Table 3).
- Dataset averages (Table 1 p. 10): Mip-NeRF 360 40.3, Tanks and Temples 21.4, Deep Blending 16.8,
  Synthetic-NeRF 4.1 MB.
- Counts: Truck 1.55M vs 2.58M vanilla (p. 11), Garden 4.37M (p. 22).

### 9. Main quantitative results (Table 1 p. 10 and appendix Table 1 p. 18, as reported)

Columns: PSNR, SSIM, LPIPS, Size (MB).

Mip-NeRF 360:

| method | PSNR | SSIM | LPIPS | Size |
|---|---|---|---|---|
| Ours | 27.64 | 0.814 | 0.220 | 40.3 |
| Ours w/o SH | 27.02 | 0.803 | 0.232 | 16.7 |
| 3DGS | 27.55 | 0.814 | 0.222 | 785 |
| 3DGS w/o SH | 26.94 | 0.803 | 0.234 | 212 |
| M-NeRF360 † | 27.69 | 0.792 | 0.237 | 8.6 |
| INGP-Big † | 25.59 | 0.699 | 0.331 | 48 |

Tanks and Temples:

| method | PSNR | SSIM | LPIPS | Size |
|---|---|---|---|---|
| Ours | 25.63 | 0.864 | 0.208 | 21.4 |
| Ours w/o SH | 25.27 | 0.857 | 0.217 | 8.2 |
| 3DGS | 25.54 | 0.866 | 0.201 | 454 |
| VQ-TensoRF † | 28.20 | 0.913 | - | 3.3 |

Deep Blending:

| method | PSNR | SSIM | LPIPS | Size |
|---|---|---|---|---|
| Ours | 30.35 | 0.909 | 0.258 | 16.8 |
| Ours w/o SH | 30.50 | 0.908 | 0.261 | 5.5 |
| 3DGS | 30.07 | 0.907 | 0.248 | 699 |
| M-NeRF360 † | 29.40 | 0.901 | 0.245 | 8.6 |

Synthetic-NeRF: Ours 33.70 / 0.969 / 0.031 / 4.1, 3DGS 33.88 / 0.970 / 0.031 / 71.6. Claimed
ratios: "a factor of 17x to 42x (depending on the dataset)", "The highest reduction of 41.6x is
observed on the Deep Blending dataset", w/o SH "a reduction factor of 127x over vanilla 3DGS while
improving in PSNR." (p. 10). Note that the 3DGS rows are the authors' own runs (Tanks and Temples
3DGS PSNR 25.54 is not the 23.14 of the 3DGS paper), the reason is not stated. Counts: not
tabulated. Strongest baselines in their table: 3DGS and Mip-NeRF 360.

### 10. Densification and pruning

- Densification, modified (appendix Table 2 p. 19, 3DGS default to Ours): densification interval
  100 to 1000, densify grad threshold 2 × 10⁻⁴ to 7 × 10⁻⁵, densify min opacity 0.005 to 0.1,
  opacity reset interval 3000 to ∞ (deactivated), percent dense 0.01 to 0.1. Purpose: "To minimize
  the sorting time during training, we make minimal changes to the default parameters of 3DGS, i.e.
  we reduce the number of Gaussians that are created during optimization." (p. 18).
- Pruning: "Our method creates less Gaussian splats and does not apply regular pruning, which
  happens every 3k steps with the default parameters." (p. 19). The only pruning is the grid-fit
  cut by lowest opacity at every re-sort (p. 8, p. 14). Fig. 8 (p. 14): removing up to 30 % of the
  lowest-opacity Gaussians leaves PSNR unchanged, removing the smallest-scale ones hurts.
- Fine-tuning after pruning: none as a stage, training simply continues after each re-sort.

### 11. Stated limitations and future work (verbatim)

- "In future work, we aim to further enhance compression efficiency. Besides an adapted weighting
  of the individual parameters of the smoothness loss, better and less correlated representations
  for rotations, shape and spherical harmonics will be investigated. To achieve even better results,
  it would be interesting to perform the quantization during training, which is currently happening
  only once before the compression. The main focus, however, will be on the extension to 4D scenes
  with consideration of temporal dependencies, supporting the strength of 3D Gaussians in the
  representation of dynamic scenes." (p. 14)
- "The grids, however, do not appear to be more organized with more iterations. This has two
  reasons. Firstly, the number of Gaussians grows substantially during the training, posing an
  increasingly more difficult sorting task. And secondly, as our method has to sort all attributes
  at once using the same permutation, isolated features, such as the color cannot be organized
  perfectly." (pp. 11 to 12)
- "It may be possible to encode the Gaussian Splats with conventional point cloud compression
  methods. The popular DRACO algorithm [10] supports encoding additional attributes on top of
  position and color in theory, but its implementations do not [22]. Thus, a direct comparison is
  unfortunately not possible." (p. 13)
- "Superior configurations might be uncovered through an exhaustive parameter sweep. We think
  performing this computationally expensive work is best done in future, after choosing a more
  compact representation than the high-dimensional spherical harmonics attributes, for
  view-dependent effects." (p. 20)

### 12. Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, budget, or edge, mobile, web, low-end deployment (verbatim)

- "However, the storage size is significantly higher, which hinders practical deployment, e.g. on
  resource constrained devices." (p. 1)
- "While rendering quality is of general importance, rendering speed and storage size play a
  crucial role especially for real time applications, portability on devices with limited
  performance and memory or, for fast web applications." (p. 1)
- "Our method achieves a reduction factor of 17x to 42x in size for complex scenes with no increase
  in training time, marking a substantial leap forward in the domain of 3D scene distribution and
  consumption." (p. 1)
- "Such a data structure can take up to several hundred megabytes depending on the scene, which
  hinders practical deployment, e.g. on resource constrained devices." (p. 2)
- "With our method, we enable novel-view synthesis at high quality, high rendering speed, and with
  a small storage footprint, bringing high quality 3D scene reconstruction and rendering one step
  closer towards applications on small devices with limited storage capacity or fast web
  applications." (p. 3)
- "This differentiates our method from compression methods that focus on reducing existing data in
  a pure post-processing step. Instead, we manipulate the data during creation to achieve high
  compression and high visual quality afterwards." (p. 9)
- "To find a good balance between both optimization targets, we tested a series of parameters for
  λ, see table 2. As expected, lower magnitudes for λ result in higher rendering quality, but also
  larger storage size." (p. 12)
- "By introducing lossy compression, the file size can be further decreased, with different
  tradeoffs between rendering quality and storage space used." (p. 13)
- "With the proposed solution, we can reduce the data down to 2% of the original size at the same
  visual quality, enabling the efficient handling of large 3DGS scenes in practical applications."
  (p. 14)
- "We found our current sets of parameters through an iterative search, optimizing with the goal to
  stay close to vanilla 3DGS quality, while minimizing file size." (p. 20)

Byte or FPS target during training: none found. Rate control: none found. Budget: none found.
The "optimization targets" on p. 12 are the two loss terms, not a size target.

### 13. Code

- "Additional information can be found on our project page:
  fraunhoferhhi.github.io/Self-Organizing-Gaussians/" (p. 1). Base code: "Our experiments are based
  off the official 3DGS implementation on GitHub" with footnote
  "https://github.com/graphdeco-inria/gaussian-splatting" (p. 10). No repository URL for the
  authors' own code is printed.
- Licence: not stated.

---

## Paper D: Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis (Compressed 3DGS)

### 1. Citation

- Title: "Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis"
- First author: Simon Niedermayr, with Josef Stumpfegger, Rüdiger Westermann. Technical University
  of Munich.
- Venue as stated in the PDF: not stated. PDF stamp: arXiv:2401.02436v2 [cs.CV] 22 Jan 2024. The
  file name records CVPR 2024.
- arXiv id: 2401.02436

### 2. Base representation and renderer

- Base: the released 3DGS reconstructions of Kerbl et al. "For Mip-Nerf360, Tanks&Temples and Deep
  Blending the reconstructions from Kerbl et al. [13] were used. We generated the 3D Gaussian
  representation for NeRF-Synthetic ourselves." (p. 6).
- Compressed form (Secs. 4.1 to 4.3): per Gaussian a position (16-bit float), opacity (8-bit
  min-max), a scalar scale factor η (8-bit, quantized before the exponential activation), one index
  into a colour codebook of full SH vectors and one index into a shape codebook of normalized
  covariances stored as quaternion plus scale (4 + 3 values, 8-bit). "Indices into the codebooks are
  stored as 32-bit unsigned integers. The data is then compressed using DEFLATE [5]" after Morton
  ordering, and "entropy encoding reduces the two codebook indices to their required bit-length
  according to the codebook sizes." (p. 5). Codebook size 4096 for both by default (p. 6), with
  high-sensitivity vectors appended uncompressed (p. 4).
- Renderer: their own. "Further, we propose a renderer for the compressed scenes using GPU sorting
  and rasterization." (p. 2). Compute pre-pass for frustum culling and SH colour evaluation,
  Onesweep radix sort, one quad per splat through vertex and pixel shaders (Sec. 5 p. 5). "The
  renderer is implemented with the WebGPU graphics API in the Rust programming language. Thus, it
  can run in a modern web browser on a large variety of devices." (p. 6).
- Rendered directly: yes for the quantized-plus-codebook layout. "We address the storage and
  rendering issue of 3D Gaussian splatting by compressing the reconstructed Scene parameters and
  rendering the compressed representation via GPU rasterization." (p. 2). "Roughly a 2x increase can
  be attributed to the compressed data's reduced bandwidth requirements, hinting at the software
  rasterizer's memory-bound performance by Kerbl et al. [13]." (p. 6). Whether the DEFLATE layer is
  decoded at load time is not stated (it is a file codec, so it must be, but the paper does not say
  so).
- MLP or hash grid at render time: none.

### 3. Budget handle

KNOB. From Sec. 6.2 (p. 6) and the ablations:

- Codebook sizes: "We use 4096 as the default codebook size in all our experiments" (p. 6),
  swept 1024 to 8192 (Table 4 p. 8).
- Sensitivity thresholds "βc = 6 · 10−7 and βg = 3 · 10−6" (p. 6), swept in Table 5 (p. 8). "They
  offer a trade-off between quality and compression rate." (p. 8).
- Bit widths: "We quantize all Gaussian parameters despite position to an 8-bit representation with
  the Min-Max scheme. 16-bit float quantization is used for position" (p. 5).
- Clustering: decay λ_d = 0.8, 800 update steps for Gaussians and 100 for SH, batch sizes 2¹⁸ and
  2²⁰ (p. 6).
- QA fine-tuning: "We perform 5000 optimization steps of quantization-aware fine-tuning." (p. 6).
- Pruning threshold: zero colour sensitivity (p. 12).

No target.

### 4. When it binds

Post-hoc on a frozen reconstruction. Stages and timing (Table 6 p. 11, "Measurements were taken
with an NVIDIA RTX A5000 graphics card"), average and maximum seconds, as reported:

| stage | average | maximum |
|---|---|---|
| Sensitivity Calculation | 8.05 | 11.38 |
| Clustering | 75.11 | 78.41 |
| QA Fine-tuning (5000 steps) | 213.30 | 278.05 |
| Encoding | 2.69 | 5.13 |
| Total | 299.15 | 365.94 |

"The compression process takes about 5-6 minutes and increases the reconstruction time by roughly
10%." (p. 6). "the fine-tuning stage takes up 70% of the total time." (p. 11).

### 5. What "size" means

"Size is measured in Megabytes." (Table 1 caption p. 6), the DEFLATE-encoded file after Morton
ordering. Table 3 (p. 8, garden) shows the accounting: +QA Finetune 86.69 MB, +Encode 58.40 MB,
+Morton Order 46.57 MB. Components (Fig. 7 p. 11): position, colour codebook, opacity, shape
codebook plus η, indices. "The coordinates of the 3D Gaussian center points and the codebook
indices take up the most memory in general. The amount of memory required by the color codebook
varies significantly between different scenes." (p. 11). Min and max values for de-quantization
are stored at 32-bit (p. 5).

### 6. Runtime cost reported

Render FPS (Table 2 p. 8), "Rendering performance at 1080p resolution in frames per second,
averaged over all training images. Bicycle consists of 6.1 million 3D Gaussians, Bonsai of 1.2
million 3D Gaussians.", as reported:

| scene / renderer | RTX A5000 | RTX 3070M | Intel UHD Graphics 11 | AMD Radeon R9 380 |
|---|---|---|---|---|
| Bicycle, Kerbl et al. [13] | 93 | 54 | - | - |
| Bicycle, Ours (uncompressed) | 215 | 134 | 9 | 41 |
| Bicycle, Compressed | 321 | 211 | 16 | 83 |
| Bonsai, Kerbl et al. [13] | 184 | 122 | - | - |
| Bonsai, Ours (uncompressed) | 414 | 296 | 23 | 76 |
| Bonsai, Compressed | 502 | 380 | 28 | 128 |

Fig. 1 (p. 1): "93 FPS 54 FPS" against "321 FPS 211 FPS", "25.2 PSNR / 1.5 GB" against "25.0 PSNR
/ 47 MB", "31× Compression", "Framerates in grey and white, respectively, are taken on NVIDIA's RTX
3070M and RTX A5000 at 1080p resolution."

Per-stage render timings in ms (Table 7 p. 11, RTX A5000), as reported: Bicycle uncompressed
preprocess 1.46, sorting 0.55, rasterization 2.81, total 4.82. Bicycle compressed 0.28, 0.48, 2.45,
3.22. Bonsai uncompressed 0.44, 0.20, 1.81, 2.44. Bonsai compressed 0.09, 0.19, 1.67, 1.95. "the
preprocessing stage is accelerated by a factor of 5× when using the compressed scene
representation." (p. 11).

Speed claims (verbatim): "We demonstrate that the compressed splat representation can be
efficiently rendered with hardware rasterization on lightweight GPUs at up to 4× higher framerates
than reported via an optimized GPU compute pipeline." (p. 1). "We see a significant increase of up
to a factor of 4× in rendering speed (see Tab. 2). Roughly a 2x increase can be attributed to the
compressed data's reduced bandwidth requirements, hinting at the software rasterizer's memory-bound
performance by Kerbl et al. [13]. The additional speedup is achieved by the hardware
rasterization-based renderer, which pays off on low- and high-end GPUs." (p. 6).

Memory claims (verbatim): "The optimized scenes usually consist of millions of Gaussians and
require up to several gigabytes of storage and memory. This makes rendering difficult or even
impossible on low-end devices with limited video memory, such as handhelds or head-mounted
displays." (p. 1). "The compressed scenes can be used in applications requiring network streaming,
and they can be rendered on low-end devices with limited video memory and bandwidth capacities."
(p. 2). No numeric GPU memory figure is given.

Low-end device results: the Intel UHD Graphics 11 and AMD Radeon R9 380 columns above. No phone
or headset measurement. Encode time: Table 6. Decode time: not stated.

### 7. Size model

No size model, nothing differentiable or predicted. Size follows from the knobs and DEFLATE.
Knob-to-size evidence on the Mip-NeRF 360 average (as reported):

- Codebook size (Table 4 p. 8): colour 1024 / 28.47 MB / 26.95 PSNR, 2048 / 28.65 / 26.95, 4096 /
  28.80 / 26.98, 8192 / 28.92 / 27.00. Gaussian 1024 / 28.14 / 26.95, 2048 / 28.45 / 26.97, 4096 /
  28.80 / 26.98, 8192 / 29.06 / 26.97. "the codebook size has little effect on the average
  reconstruction error" (p. 8).
- Sensitivity threshold βc (Table 5 p. 8): 6.0·10⁻⁸ / 56.50 MB / 27.22 PSNR, 3.0·10⁻⁷ / 33.00 /
  27.09, 6.0·10⁻⁷ / 28.80 / 26.98, 1.2·10⁻⁶ / 27.02 / 26.87, 6.0·10⁻⁶ / 25.97 / 26.74, no threshold
  / 25.65 / 26.55.
- βg: 3.0·10⁻⁷ / 33.90 / 27.05, 1.5·10⁻⁶ / 29.95 / 27.00, 3.0·10⁻⁶ / 28.80 / 26.98, 6.0·10⁻⁶ /
  28.08 / 26.91, 3.0·10⁻⁵ / 27.30 / 26.86, none / 27.10 / 26.80.

So the codebook size moves the file by under 1 MB while βc moves it from 25.65 to 56.50 MB. The
paper draws no rate-distortion curve and fits no model.

### 8. Scene dependence

Per-scene compressed sizes at the default setting (Tables 8 to 10 p. 13), MB and ratio, as
reported: bicycle 47.147 (30.761), bonsai 12.794 (23.011), counter 13.789 (20.977), flowers 31.140
(27.619), garden 46.565 (29.636), kitchen 18.874 (23.211), room 15.033 (25.068), stump 40.569
(28.926), treehill 33.318 (26.859). train 13.249 (18.324), truck 21.316 (28.196). drjohnson 28.938
(27.830), playroom 21.660 (27.802). Spread across Mip-NeRF 360 at one setting: 12.794 to 47.147 MB,
a factor of 3.7. The ratio itself spans 18.3 to 30.8.

### 9. Main quantitative results (Table 1 p. 6, as reported)

Columns: 3DGS PSNR / SSIM / LPIPS / SIZE, then Ours PSNR / SSIM / LPIPS / SIZE, Compression Ratio.

| dataset | 3DGS | Ours | ratio |
|---|---|---|---|
| Synthetic-NeRF | 33.21 / 0.969 / 0.031 / 69.89 | 32.936 / 0.967 / 0.033 / 3.68 | 19.17 |
| Mip-NeRF360 | 27.21 / 0.815 / 0.214 / 795.26 | 26.981 / 0.801 / 0.238 / 28.80 | 26.23 |
| Tanks&Temples | 23.36 / 0.841 / 0.183 / 421.90 | 23.324 / 0.832 / 0.194 / 17.28 | 23.26 |
| Deep Blending | 29.41 / 0.903 / 0.243 / 703.77 | 29.381 / 0.898 / 0.253 / 25.30 | 27.81 |
| average* (no synthetic) | 26.58 / 0.853 / 0.213 / 640.31 | 26.560 / 0.844 / 0.238 / 23.73 | 25.77 |

"Our compression method achieves a compression ratio of up to 31× with an average of 26× at the
indiscernible loss of quality (0.23 PSNR on average) for real-world scenes." (p. 6). The only
baseline is 3DGS itself. Counts: Bicycle 6.1M, Bonsai 1.2M (Table 2 caption p. 8), pruning removes
"up to 15%" (p. 12). The per-scene Table 8 average for 3DGS is 27.287 PSNR / 795.263 MB, versus
27.21 in Table 1.

### 10. Densification and pruning

- Densification: none, the method starts from a frozen reconstruction. QA fine-tuning optimizes
  "the position, opacity, and scaling factor of each Gaussian as well as the color and Gaussian
  codebook entries" (p. 5), no clone or split.
- Pruning score: parameter sensitivity, S(p) = (1 / ∑[i=1..N] P_i) ∑[i=1..N] |∂E_i/∂p| (Eq. 3
  p. 4), E the total image energy (sum of RGB over pixels), computed with one backward pass per
  training image, colours left unclamped for this pass (p. 12). "We observe that a notable number of
  Gaussians (up to 15%) do not have any impact on the training images. These particular splats
  exhibit zero sensitivity in the color parameters. Consequently, we opt to eliminate these splats
  from the scene (called Pruning in Tab. 3)." (p. 12).
- After pruning: clustering, then 5000 steps of QA fine-tuning (p. 6). Table 3 (p. 8, garden):
  baseline 27.179 PSNR / 1379.99 MB, +Pruning 27.083 / 1217.25, +Color Clustering 25.941 / 278.41,
  +Gaussian Clustering 25.781 / 164.15, +QA Finetune 26.746 / 86.69, +Encode 26.746 / 58.40,
  +Morton Order 26.746 / 46.57.
- More aggressive pruning was tried and rejected (see field 11).

### 11. Stated limitations and future work (verbatim)

- Sec. 6.5 Limitations: "As the main limitation for making the proposed compression and rendering
  pipeline even more powerful, we see the current inability to aggressively compress the Gaussians'
  positions in 3D space. We performed experiments where positions were quantized to a lattice
  structure, and we even embedded these positional constraints into the Gaussian splatting training
  process. Unfortunately, we were not able to further compress the positions without introducing a
  significant error in the rendering process." (p. 8)
- Conclusion: "In the future, we aim to explore new approaches for reducing the memory footprint
  during the training phase, and additionally compressing positional information end-to-end. We also
  believe that 3D Gaussian splatting has the potential for reconstructing volumetric scenes, and we
  will investigate advanced options for compressing and rendering the optimized representations."
  (pp. 8 to 9)
- "Experiments with higher pruning thresholds have shown that more Gaussians can be removed with
  minimal loss in PSNR. However, this can lead to fine details in the scene being removed, which we
  consider undesirable." (p. 12)

### 12. Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, budget, or edge, mobile, web, low-end deployment (verbatim)

- "Making such representations suitable for applications like network streaming and rendering on
  low-power devices requires significantly reduced memory consumption as well as improved rendering
  efficiency." (p. 1)
- "The learned codebooks have low bitrates and achieve a compression rate of up to 31× on
  real-world scenes with only minimal degradation of visual quality. We demonstrate that the
  compressed splat representation can be efficiently rendered with hardware rasterization on
  lightweight GPUs at up to 4× higher framerates than reported via an optimized GPU compute
  pipeline." (p. 1)
- "This affects both the training and rendering times and often prohibits the use of such
  representations in applications like network streaming and mobile rendering." (p. 1)
- "The optimized scenes usually consist of millions of Gaussians and require up to several
  gigabytes of storage and memory. This makes rendering difficult or even impossible on low-end
  devices with limited video memory, such as handhelds or head-mounted displays." (p. 1)
- "Quantization-aware fine-tuning: To regain information that is lost during clustering we
  fine-tune the scene parameters at reduced bit-rates using quantization-aware training." (p. 2)
- "It enables novel view synthesis in real-time, even on low-end devices, and can be easily
  integrated into applications rendering polygonal scene representations." (p. 2)
- "The compressed scenes can be used in applications requiring network streaming, and they can be
  rendered on low-end devices with limited video memory and bandwidth capacities." (p. 2)
- "A number of works have especially addressed memory reduction during inference, to make
  grid-based scene representations more suitable for low-end devices with limited video memory [19,
  27]. To our knowledge, our approach is the first that aims at the compression of point-based
  radiance fields to enable high-quality novel view synthesis at interactive frame rates on such
  devices." (p. 2)
- "To render 3D Gaussian scenes fast especially on low-power GPUs, our novel view renderer utilizes
  hardware rasterization." (p. 5)
- "The renderer is implemented with the WebGPU graphics API in the Rust programming language. Thus,
  it can run in a modern web browser on a large variety of devices." (p. 6)
- "The sensitivity thresholds βc and βg are used to decide whether to consider SH coefficients and
  shape parameters for clustering. They offer a trade-off between quality and compression rate."
  (p. 8)
- "We performed experiments where positions were quantized to a lattice structure, and we even
  embedded these positional constraints into the Gaussian splatting training process." (p. 8)
- "The compressed data can be streamed over networks and rendered on low-power devices, making it
  suitable for mobile VR/AR applications and games. In the future, we aim to explore new approaches
  for reducing the memory footprint during the training phase, and additionally compressing
  positional information end-to-end." (p. 8)

Byte or FPS target during training: none found. Rate control: none found. Budget: none found. The
closest training-time statement is the failed lattice-position constraint embedded in training
(p. 8), which is a quantization constraint, not a size target.

### 13. Code

- "The source code is available at https://github.com/KeKsBoTer/c3dgs." (p. 6).
- Licence: not stated.

---

## Cross-paper summary for the gap question

| paper | handle | binds when | size meaning | rendered directly |
|---|---|---|---|---|
| LightGaussian | knobs: prune ratio, SH degree, VQ ratio 60 %, codebook 8192, fp16 | post-hoc, 5 000 + 5 000 fine-tune iterations | MB on disk, fp16 arrays plus VQ'd SH, codebook accounting not stated | no (implied decode to stock 3D-GS layout, no MLP) |
| Compact 3DGS | knobs: λ_m count regulariser 5e-4, hash map 2¹⁹, R-VQ 64 × 6, +PP 8-bit and Huffman | mask from iteration 0 of 30K, R-VQ last 1K, PP post-hoc | MB: positions, opacity, R-VQ, hash grid, MLP | no, tiny MLP per view plus precomputed hash features and codebook decode |
| SOG | knobs: smoothness λ = 1.0, sorting weights, quantization levels, JPEG XL quality 100, densification params | sorting and smoothness loss during training, quantization and codec post-hoc | MB of JPEG XL grids | no, decoded to stock 3DGS, no MLP |
| Compressed 3DGS | knobs: codebooks 4096, βc 6e-7, βg 3e-6, 8-bit (16-bit position), 5000 QA steps | post-hoc, about 5 to 6 minutes on RTX A5000 | MB of DEFLATE file after Morton ordering | yes, own WebGPU hardware rasterizer reads quantized attributes and codebook indices |

Facts that matter for the claimed gap:

- None of the four accepts a byte, bitrate, memory or FPS target. Every handle is a knob whose
  effect on bytes is observed afterwards.
- Two of them shape the representation during training (Compact 3DGS's mask and R-VQ, SOG's
  sorting and smoothness loss), but neither measures or constrains bytes during training. Compact
  3DGS regularises the count through L_m, and its Fig. 8 rate-distortion curves are produced by
  doubling hyper-parameters after the fact.
- The closest statements to a training-time size mechanism are future-work sentences: SOG "perform
  the quantization during training" (p. 14) and Compressed 3DGS "reducing the memory footprint
  during the training phase" (p. 8), plus Compressed 3DGS's failed attempt to embed lattice position
  constraints into training (p. 8).
- Scene dependence at fixed knobs is large: Compressed 3DGS 12.794 to 47.147 MB over Mip-NeRF 360
  (3.7×), Compact 3DGS 34.21 to 62.99 MB (1.84×). A fixed knob does not give a fixed size, which is
  the premise of a target-driven method.
- Only Compressed 3DGS renders the compressed form directly, and it is also the only one reporting
  low-end GPU numbers (Intel UHD Graphics 11: 16 and 28 FPS compressed, AMD Radeon R9 380: 83 and
  128 FPS compressed, 1080p).
- Nothing in these four papers contradicts the claim that no published method binds a byte target
  during training.
