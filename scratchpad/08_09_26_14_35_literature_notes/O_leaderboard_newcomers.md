# O — 3DGS.zip leaderboard newcomers

Five papers read cover to cover (main text plus every appendix page). Every number below is
**reported** by the paper, copied from the PDF text layer. Fields the paper does not state are
marked "not stated". Fields 11 and 12 are verbatim quotes with page numbers.

Source PDFs:
- `references/02_rate_distortion/smol_gs_wang_2025_arxiv2512.00850.pdf`
- `references/02_rate_distortion/gsico_martin_2026_arxiv2601.14510.pdf`
- `references/02_rate_distortion/codecgs_lee_2025_arxiv2501.03399.pdf`
- `references/04_compaction_pruning/eagles_girish_2023_arxiv2312.04564.pdf`
- `references/04_compaction_pruning/compact3d_navaneet_2023_arxiv2311.18159.pdf`

---

## 1. Smol-GS

### 1. Citation
Title: "Smol-GS: Compact Representations for Abstract 3D Gaussian Splatting". First author:
Haishan Wang (with Mohammad Hassan Vali, Arno Solin), ELLIS Institute Finland and Aalto
University, Espoo, Finland. Venue as printed on p1: "Preprint." Stamp on p1:
"arXiv:2512.00850v3  [cs.CV]  18 Jun 2026". arXiv id from the file name: 2512.00850.

### 2. Base representation
Splat-wise features with tiny MLP decoders. No anchors and no offsets. Each splat carries a
coordinate x ∈ ℝ³, a feature vector f ∈ ℝ^nf with nf = 8, and a scaling controller s ∈ ℝ³ (p4).
Rendering **requires** five MLPs plus a hash grid: "Smol-GS models the 3D scene as a set of
neural Gaussian splats with the support of a few MLPs (MLPo, MLPc, MLPs, MLPr, and MLPoct)"
(p4). Opacity, color, rotation and scaling are all predicted per frame per splat from f*, which
is built at render time from the camera center: f* = concat(f, (xc − x)/‖xc − x‖, ‖xc − x‖,
MLPoct(foct)) (p4, Eq. 2). Because f* depends on the viewing direction and camera distance, the
MLPs must run **every frame**, not once at load. MLPh + InstantNGP hash tables are used only for
the entropy model (coding), not for rasterization. Decoded model is **not** a standard 3DGS
point set: the on-disk model is octree occupancy bytes + arithmetic-coded f and s + MLP weights;
after entropy decoding you get quantized f, s and x, and standard 3DGS attributes only exist
transiently as MLP outputs. Rasterization itself is the unmodified Inria 3DGS α-blending (p5,
Eqs. 5–6). Entropy decoding is a one-time load-time cost (Tables A13/A14).

MLP shapes (Table A6, p15): MLPo/MLPc/MLPr/MLPs each 3 layers, hidden 128, input nf + noct + 4
= 8 + 8 + 4 = 20, outputs 1/3/4/3. MLPoct: input 3R = 48, 3 layers, hidden 64, output noct = 8.
MLPh: input 96, 3 layers, hidden 128, output 8 + 2nf = 24.

### 3. Budget handle
**KNOB, not a target.** The user supplies λo (opacity regularization strength) to pick a variant,
plus λq (quantization/NLL weight), R (octree recursion depth) and nf (feature dimension). No
byte target anywhere.

Exact values used for the rate points:
- Variants (p7): "small: λo = 4×10−7, base: λo = 3×10−7, large: λo = 1×10−7".
- Fixed defaults (p7): λs = 0.2, λq = 5×10−4, τg = 2×10−4, τo = 0.005, nf = 8, R = 16.
- λo sweep (Table A9, p17): {1e−7, 2e−7, 3e−7, 4e−7, 5e−7} → Mip-NeRF 360 sizes
  {10.31, 7.563, 5.896, 4.825, 4.033} MB.
- λq sweep (Table A10, p18): {2e−4, 5e−4, 1e−3, 2e−3, 3e−3} → Mip-NeRF 360 sizes
  {6.516, 5.896, 5.454, 5.065, 4.797} MB.
- R sweep (Table A7, p16): R ∈ {14,15,16,17,18} → Mip-NeRF 360 sizes
  {5.132, 5.513, 5.896, 6.084, 6.207} MB.
- nf sweep (Table A8, p16): nf ∈ {4,6,8,16,24} → Mip-NeRF 360 sizes
  {5.038, 5.036, 5.896, 8.398, 10.24} MB.

Note a **discrepancy inside the paper**: Table 2 (p8) lists Smol-GS-base at 27.76 dB / 5.860 MB,
while every ablation table's default row (λo = 3e−7, λq = 5e−4, R = 16, nf = 8) lists
27.69 dB / 5.896 MB. Same configuration, two different numbers. Treat ±0.07 dB / ±0.04 MB as
the paper's own run-to-run noise.

### 4. When it binds
Binds **during training**, but in two late stages, not from iteration 0. Five stages (p6):
1. Warm-up 0–0.5k. x and s from SfM, f random.
2. Densification 0.5k–15k. Gradient-based cloning plus opacity/visibility pruning.
3. Compaction 15k–20k. Opacity L1 regularization (λo > 0) plus pruning. This is the only stage
   where λo acts.
4. Feature compression 20k–30k. f and s quantized with learned steps, NLL rate term active
   (λq > 0), arithmetic coding.
5. Coordinate compression 30k–35k. Coordinates quantized and stored with the occupancy octree.

"The opacity regularization is only effective (i.e., λo > 0) in the compaction stage (3), and the
quantization regularization is only active (i.e., λq > 0) during the compression stages of (4)
and (5)." (p6)

Total 35k iterations. Wall clock: "The average training time of Smol-GS-base is around 26.58
minutes (35k iterations) per scene." (p14)

### 5. What "size" means
"The total Smol-GS size reported in this paper is the size of a single zip file containing all
these four components (following the 3DGS.zip survey [52])." (p7). The four components (p7):
(i) occupancy-octree bit strings for x̂, (ii) arithmetic codes for f̂ and ŝ, (iii) MLP
architecture and weights for all six MLPs, (iv) metadata (scene lower/upper bounds, recursion
depth R, and Hash(·)). **Decoder weights and the hash table are included.**

Component breakdown, Table 1 (p7), MB averaged over scenes:

| dataset | Total | x | f | s | MLPs | Others |
|---|---|---|---|---|---|---|
| MIP-NERF 360 | 5.860 | 1.186 | 2.753 | 1.751 | 0.157 | 0.013 |
| DEEP BLENDING | 2.860 | 0.739 | 1.097 | 0.846 | 0.169 | 0.009 |
| TANKS AND TEMPLES | 4.784 | 0.843 | 2.260 | 1.526 | 0.143 | 0.012 |

So on Mip-NeRF 360: features 47 %, scaling controller 30 %, coordinates 20 %, MLPs 2.7 %.

### 6. Runtime cost
Hardware: "Our experiments are conducted on NVIDIA H200 GPUs with 141 GB of memory. The code is
implemented in Python 3.11 using the PyTorch framework (version 2.4.1) [57] with CUDA 12.4."
(p14). GPU memory during training or rendering: not stated.

Render FPS, Table 3 (p8), averaged over all scenes:

| | Smol-GS | 3DGS | HAC++ |
|---|---|---|---|
| MIP-NERF 360 | 210.0 | 139.3 | 128.0 |
| DEEP BLENDING | 399.9 | 178.8 | 149.8 |
| TANKS AND TEMPLES | 223.3 | 218.9 | 102.7 |

Encode time, Smol-GS-base, seconds (Table A11, p20): Garden 3.654 total (x 0.262, f 2.158,
s 1.149, MLPs 0.026); Bicycle 1.845; Room 0.961; Drjohnson 1.315; Train 1.417; Truck 1.794.
Decode time, Smol-GS-base, seconds (Table A13, p21): Garden 5.507 (x 0.230, f 3.216, s 2.030,
MLPs 0.019); Bicycle 2.882; Room 1.439; Flowers 4.635; Truck 2.752. Decode is dominated by the
arithmetic decoding of f and s, not by coordinates.

### 7. Rate model (detail for a controller)
**Feature and scaling rate.** Hash-grid-conditioned Gaussian entropy model with learned
per-dimension step sizes, taken from HAC. "we condition feature coding on spatial hash
descriptors by predicting a distribution and a per-dimension step size with a tiny network MLPh:
µf, Σf, ∆f = MLPh(Hash(x))" (p5). Quantization relaxation: **learned step size with rounding**,
f̂ = ∆f ⊙ round(f/∆f) (p5). Rate proxy is the Gaussian bin NLL:
NLL(f) = −log ∫[f̂−∆f .. f̂+∆f] N(f′; µf, Σf) df′ (p5, Eq. 7), averaged over the N splats:
LNLL(f) = (1/N) Σ[i=1..N] NLL(fi). Same construction for s. "The NLL is the theoretical lower
bound for the number of bits needed to encode the features." (p5). Actual coding is arithmetic
coding. Hash parameters are quantized to binary: "For better compression, the parameters of
Hash(·) are quantized as binary." (p5). Hash grid (p14): 2D table size 2¹⁵ and 3D table size
2¹³, 4 features per level, 2D resolutions (130, 258, 514, 1026), 3D resolutions (18, 24, 33, 44,
59, 80, 108, 148, 201, 275, 376, 514).
**Stated gap between estimated and actual encoded bytes: not stated.** The paper never compares
the NLL estimate against the produced bitstream.

**Coordinate compression, recursive voxel hierarchy (p4, Sec. 3.1).** Start from the
axis-aligned bounding box of all splats. At recursion r = 1 split it into 8 equal sub-boxes; at
each later recursion only non-empty boxes (≥ 1 splat) are split into 8 children. Box size at
depth r is br = (xmax − xmin)/2^r. Every subdivision emits one 8-bit occupancy byte (1 =
non-empty, 0 = empty) with children in Morton order. After R recursions each finest box holds at
most one splat and its center is the quantized coordinate:
x̂ = bR · floor((x − xmin)/bR) + 0.5·bR + xmin (Eq. 1).
"Excessive splats that fall into the same finest-level cell are pruned." (p4) — the octree depth
therefore also acts as a splat-count cap. Stored payload = occupancy bytes of all non-empty
internal nodes (breadth-first) + xmin, xmax + R, then **Huffman coded** (p4, Fig. 4 p5). The
bytes are highly sparse: Fig. A16 (p20) shows most divisions produce few non-empty sub-boxes on
Bonsai, Room, Garden and DrJohnson. Ablation w/o Huffman (Table 4, p8): 7.038 MB versus 5.860 MB
at identical 27.76/0.807/0.248, so Huffman buys 1.18 MB on Mip-NeRF 360.

**Positional encoding derived from the same octree (p4, Sec. 3.2).** With R recursions there are
at most 8^R finest boxes, so each box is named by a 3R-bit sequence, which is the Morton (Z-order)
index of the box, and this bit string **is** the positional embedding foct ∈ {0,1}^{3R}. Bits
(3r−2)..(3r) of foct select the sub-box at recursion depth r. Example given: "when there is only
one recursion level, foct = (1, 0, 0) indicates that the splat is located at the 4th sub-box of
the entire space" (p4). foct is fed through MLPoct (input 3R = 48 with R = 16) to an 8-dim
embedding. So the positional encoding is **free** — it is a deterministic function of the stored
octree, costing no extra bytes. This also couples geometry precision, PE resolution and splat
count into the single knob R.

**Controller assessment.** Every rate is differentiable except the coordinate branch. λq
multiplies a genuine differentiable bit estimate on f and s, which together are 4.5 MB of the
5.86 MB Mip-NeRF 360 total, so a λq controller could steer roughly 77 % of the bytes. λo is a
count knob acting only in 15k–20k and only through the opacity L1. The coordinate branch (1.19
MB, 20 %) is controlled by R, a discrete non-differentiable choice made once at stage 5, and R
also changes the PE width, so it cannot be varied cheaply during training. Splat count itself is
the dominant multiplier on all three of f, s and x.

**Quantization sensitivity evidence (Fig. A9, p15).** A trained 3DGS-30k model quantized to
depth r gives Garden PSNR: r = 10 → 13.843, r = 12 → 20.950, r = 14 → 26.537, r = 16 → 27.332.
The paper reads this as "16-bit per-axis quantization yields an imperceptible quality drop" (p4).

### 8. Scene dependence at a fixed setting
Smol-GS-base, per-scene total size in MB (Table A15, p22), same hyperparameters for every scene:
Garden 11.283, Bicycle 5.644, Stump 7.015, Bonsai 4.010, Room 2.858, Treehill 3.782, Flowers
9.178, Drjohnson 3.331, Train 4.198. Spread on Mip-NeRF 360 alone is 2.858 MB (Room) to 11.283
MB (Garden), a factor of 3.9 at one fixed λo. **A single global λo does not produce a single
per-scene size.**

Smol-GS-small per scene (Table A16, p22): Garden 8.976, Bicycle 4.725, Stump 5.147, Bonsai
3.669, Room 2.530, Treehill 2.696, Flowers 7.900, Drjohnson 2.777, Playroom 2.067, Train 3.827,
Truck 4.590.

### 9. Main quantitative results (all reported)
Table 2 (p8), copied exactly. Size in MB.

| Method | M360 PSNR | SSIM | LPIPS | Size | T&T PSNR | SSIM | LPIPS | Size | DB PSNR | SSIM | LPIPS | Size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DGS-30K [1] | 27.21 | 0.815 | 0.214 | 734.000 | 23.14 | 0.841 | 0.183 | 411.000 | 29.41 | 0.903 | 0.243 | 676.000 |
| HEMGS [5] | 27.75 | 0.806 | 0.248 | 12.539 | 24.42 | 0.848 | 0.192 | 6.034 | 30.24 | 0.908 | 0.266 | 2.993 |
| HAC++ [4] | 27.60 | 0.803 | 0.253 | 8.742 | 24.22 | 0.849 | 0.190 | 5.427 | 30.16 | 0.907 | 0.266 | 3.051 |
| Smol-GS-base | 27.76 | 0.807 | 0.248 | 5.860 | 24.25 | 0.843 | 0.199 | 4.784 | 30.12 | 0.905 | 0.262 | 2.860 |
| Smol-GS-small | 27.61 | 0.801 | 0.258 | 4.867 | 24.21 | 0.841 | 0.204 | 4.209 | 29.97 | 0.903 | 0.266 | 2.422 |

The two strongest baselines are HEMGS (best PSNR of the baselines on all three sets) and HAC++
(smallest of the baselines). Other rows in the same table: ContextGS 27.62/0.808/0.237/13.297;
HAC 27.53/0.807/0.238/16.005; CompGS 27.03/0.804/0.243/18.000; Compressed3D
26.98/0.801/0.238/28.803; Reducing3DGS 27.10/0.809/0.226/29.000; SOG 27.08/0.799/0.230/40.285;
Compact3DGS 27.08/0.798/0.247/48.800.

**Splat / anchor count: not stated anywhere in the paper.** Fig. 5 (p6) plots #Gaussians for the
Garden scene against an unlabelled axis but no numeric count is printed.

**All rate points on Mip-NeRF 360.** The main table gives only base and small. The third named
variant (large, λo = 1e−7) does not appear in Table 2, but its row is in the λo ablation.
Collecting every printed Mip-NeRF 360 (PSNR, SSIM, LPIPS, size MB) pair for Smol-GS:

| source | setting | PSNR | SSIM | LPIPS | Size MB |
|---|---|---|---|---|---|
| Table 2 | Smol-GS-base | 27.76 | 0.807 | 0.248 | 5.860 |
| Table 2 | Smol-GS-small | 27.61 | 0.801 | 0.258 | 4.867 |
| Table A9 | λo = 1e−7 ("large") | 27.85 | 0.812 | 0.231 | 10.31 |
| Table A9 | λo = 2e−7 | 27.78 | 0.810 | 0.239 | 7.563 |
| Table A9 | λo = 3e−7 (base) | 27.69 | 0.806 | 0.248 | 5.896 |
| Table A9 | λo = 4e−7 (small) | 27.59 | 0.801 | 0.258 | 4.825 |
| Table A9 | λo = 5e−7 | 27.44 | 0.794 | 0.270 | 4.033 |
| Table A10 | λq = 2e−4 | 27.73 | 0.807 | 0.248 | 6.516 |
| Table A10 | λq = 5e−4 | 27.69 | 0.806 | 0.248 | 5.896 |
| Table A10 | λq = 1e−3 | 27.67 | 0.806 | 0.249 | 5.454 |
| Table A10 | λq = 2e−3 | 27.53 | 0.804 | 0.251 | 5.065 |
| Table A10 | λq = 3e−3 | 27.42 | 0.802 | 0.253 | 4.797 |
| Table A7 | R = 14 | 27.36 | 0.787 | 0.265 | 5.132 |
| Table A7 | R = 15 | 27.60 | 0.801 | 0.252 | 5.513 |
| Table A7 | R = 16 | 27.69 | 0.806 | 0.248 | 5.896 |
| Table A7 | R = 17 | 27.68 | 0.807 | 0.248 | 6.084 |
| Table A7 | R = 18 | 27.70 | 0.808 | 0.248 | 6.207 |
| Table A8 | nf = 4 | 27.37 | 0.799 | 0.255 | 5.038 |
| Table A8 | nf = 6 | 27.61 | 0.803 | 0.252 | 5.036 |
| Table A8 | nf = 8 | 27.69 | 0.806 | 0.248 | 5.896 |
| Table A8 | nf = 16 | 27.80 | 0.809 | 0.245 | 8.398 |
| Table A8 | nf = 24 | 27.84 | 0.809 | 0.245 | 10.24 |
| Table 4 | w/o Huffman | 27.76 | 0.807 | 0.248 | 7.038 |
| Table 4 | w/o Coordinate Quant. | 27.74 | 0.809 | 0.246 | 10.79 |
| Table 4 | w/o Feature Quant. | 27.72 | 0.805 | 0.250 | 23.71 |

The leaderboard's second Smol-GS point (27.86 dB at 10.4 MB) does not appear verbatim in the
PDF. The nearest printed rows are λo = 1e−7 at 27.85/10.31 and nf = 24 at 27.84/10.24.

### 10. Densification and count control
Standard 3DGS-style ADC with a Scaffold-GS-flavoured clone rule (p6). Initialization from SfM
for x and s, random for f. Densification: "Splats with a high magnitude of the coordinate
gradient are densified during training [21]. We choose threshold τg such that the splat whose
coordinate's gradient magnitude is higher than τg will be cloned as a new splat. The newly
introduced splat inherits the feature f from the parent splat, while its coordinate is randomly
sampled from a Gaussian N(x, Σ)". Only cloning, no splitting. Pruning: "an individual splat is
pruned when either its average opacity is less than the threshold τo, or if it remains
consistently invisible." Opacity regularization: Lo = Σ[i=1..N] |MLPo(fi*)|. Settings (p14):
τo = 0.005 and τg = 2×10−4 during 0.5k–15k; during 15k–20k "there is no densification module
applied, and the pruning threshold on opacity is set to τo = 0.005."

**Does the size objective influence densification?** Only indirectly and only in stage 3. λo is a
scalar opacity penalty, not a size penalty. λq (the true bit-rate term) acts on f and s **after**
densification and compaction are over, so the rate model never feeds back into the splat count.
The octree also prunes: "Excessive splats that fall into the same finest-level cell are pruned."
(p4), which is a geometric count cap set by R, applied once at stage 5.

### 11. Stated limitations and future work (verbatim)
p9: "Limitations Smol-GS requires training for each scene, which is time-consuming and limits its
scalability to large-scale applications. The positional encoding relies on the octree hierarchy,
which is sensitive to splat edits and suboptimal for large-scale scenes."

p9: "Future work The discrete representations of splat features in Smol-GS could serve as 3D
information tokens, enabling integration with other learning modalities and language models. The
discrete primitive distribution can potentially be improved by depth estimation techniques. The
octree hierarchy implies relations among splats, which can improve representation by leveraging
prior knowledge of the geometry."

### 12. Sentences touching a size / byte / bitrate / FPS TARGET, rate control, budget, desired
size, edge or mobile deployment (verbatim, with page)
No sentence in the paper sets or accepts a byte, bitrate or FPS target. The closest statements
are aspirational or descriptive:

- p1: "This design enables orders-of-magnitude reductions in storage while preserving
  representation flexibility."
- p1 (Fig. 1 caption): "On MIP-NERF 360, this gives state-of-the-art results in size/PSNR: 4.87
  MB/27.61 dB."
- p2: "• Experiments on standard benchmarks show that Smol-GS substantially reduces the storage
  requirements for 3DGS while maintaining high reconstruction fidelity and rendering speed,
  competitive with state-of-the-art approaches, making Smol-GS well-suited for real-time
  applications."
- p3 (Fig. 3 caption): "and employ adaptive density control with stage-wise training (loss-terms
  in red) to balance fidelity and size (Secs. 3.5 and 3.6)."
- p3: "Gradient-based densification and pruning, together with stage-wise optimization, balance
  reconstruction fidelity and model size."
- p5: "The NLL is the theoretical lower bound for the number of bits needed to encode the
  features."
- p5/p6: "The average NLL over all N splats serves as the training loss to encourage reducing the
  storage for encoding the features: LNLL(f) = 1/N Σ[i=1..N] NLL(fi)."
- p7: "The total Smol-GS size reported in this paper is the size of a single zip file containing
  all these four components (following the 3DGS.zip survey [52])."
- p9: "The high rendering speed further demonstrates the practicality of Smol-GS for real-time
  applications."
- p9: "The compact storage and fast inference also facilitate real-time rendering and
  transmission of 3D scenes in resource-constrained environments."
- p17 (Table A9 discussion): "The larger λo encourages the model to use fewer splats to represent
  the scene, resulting in worse reconstruction quality but smaller storage size."
- p17 (Table A10 discussion): "The value of λq controls the strength of quantization
  regularization, which encourages the model to use fewer bits to represent the abstract features
  f and scaling controller s."

No sentence mentions a byte budget, a desired size, rate control in the codec sense, edge devices
or mobile devices.

### 13. Code URL and licence
Project page printed on p1: `https://aaltoml.github.io/Smol-GS`. No repository URL and **no
licence stated**. The paper does note dataset licensing: "we benchmark our Smol-GS on standard
permissively-licensed real-world 3D reconstruction data sets" (p7).

---

## 2. GSICO

### 1. Citation
Title: "Structured Image-based Coding for Efficient Gaussian Splatting Compression". First
author: Pedro Martin (with António Rodrigues, João Ascenso, Maria Paula Queluz), Instituto de
Telecomunicações, Instituto Superior Técnico, University of Lisbon. Venue as printed on p1
footer: "This work has been submitted to the IEEE for possible publication." No year printed in
the body. arXiv id from file name: 2601.14510.

### 2. Base representation
Neither of the above exactly — GSICO is a **post-training codec** that consumes an already-trained
model and is agnostic to which one. Two input families are supported: raw 3DGS (P = 59
parameters per Gaussian) and Scaffold-GS (P = 66 parameters per voxel). Parameters are mapped
into P structured 2D images, quantized, and coded with JPEG XL.

Render-time requirements depend on the input model, not on GSICO: "the Gaussians (or voxels) are
reconstructed by reading the same pixel position across all reconstructed 2D maps, since each
position across the parameter images corresponds to a single Gaussian (or voxel). This
synchronized organization enables fast decoding." (p6). For 3DGS inputs the decoded model **is a
standard 3DGS point set** and needs no network at render time. For Scaffold-GS inputs the three
small MLPs (7086 weights and biases in total, p3/p4) are needed at render time, and they are
transmitted uncompressed in the bitstream. Decoding is a JPEG XL decode plus inverse
quantization, done once, not per frame. No hash grid, no feature planes, no learned entropy
model.

Parameter tables (Table I, p4): 3DGS — position 3, SH DC 3, SH AC 45, scale 3, rotation 4,
opacity 1. Scaffold-GS — anchor position 3, offset features 30, anchor features 32, scale
factor 1.

### 3. Budget handle
**KNOB, codec quality plus per-parameter bit depth.** The user picks one of five predefined
operating points per representation. No target of any kind.

Exact values (Table II, p6, with P1 position, P2 scale, P3 rotation, P4 SH DC, P5 SH AC Y 1º,
P6 SH AC Y 2º/3º, P7 SH AC UV, P8 opacity):

3DGS-based input. Bit depth is the same for all five points: P1 = 14, P2 = 8, P3 = 8, P4 = 8,
P5 = 6, P6 = 5, P7 = 0 (chrominance AC discarded), P8 = 6. JPEG XL quality level per RD point:

| RD point | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 100 | 100 | 96 | 96 | 96 | 96 | 100 |
| 2 | 100 | 100 | 100 | 90 | 90 | 90 | 90 | 100 |
| 3 | 100 | 100 | 100 | 70 | 70 | 70 | 70 | 100 |
| 4 | 100 | 100 | 100 | 20 | 20 | 20 | 20 | 100 |
| 5 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 100 |

Scaffold-GS-based input. JPEG XL quality is 100 (lossless) for all parameters at all five points;
the knob is bit depth (P1 position, P2 scale factor, P3 offset features, P4 anchor features):

| RD point | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| 1 | 16 | 8 | 8 | 8 |
| 2 | 16 | 8 | 8 | 6 |
| 3 | 16 | 8 | 6 | 6 |
| 4 | 16 | 8 | 6 | 4 |
| 5 | 16 | 8 | 4 | 4 |

"The GSICO operating points were selected empirically by assessing how quantization and JPEG XL
coding impact the rendering quality" (p6). "The bit depth assigned to each parameter is assumed
to be known to both the encoder and decoder." (p5). Quantization step size
Qstep_i = (xmax_i − xmin_i) / 2^{b_i}, with xmin and xmax carried in the bitstream (p5, Eqs. 1–2).

### 4. When it binds
**Post-hoc only.** "the main objective of this work is to provide an efficient and versatile
solution for compressing GS models without requiring any post-training optimization or
fine-tuning of the model." (p1). No training, no fine-tuning, no iteration counts. Pipeline
stages (p3–p6): color-space conversion (3DGS only, RGB→YUV BT.601 on the SH coefficients) →
pruning to make the Gaussian count a factorable multiple of 256 → fixed-size K-means clustering
into 256-element clusters → NNS cluster-to-block assignment → NNS block filling → uniform
mid-tread quantization → JPEG XL encoding. **Encoder wall-clock time is not stated anywhere.**

### 5. What "size" means
"The full bitstream size, expressed in megabytes (MB), was adopted as the rate measure." (p9).
The bitstream contains: "the JPEG XL-encoded 2D maps along with the associated quantization
metadata (minimum and maximum parameter values), and, in Scaffold-GS-based models, also includes
the MLP parameters." (p5–p6). So **MLP weights are included for Scaffold-GS inputs** (uncompressed,
7086 weights and biases) and per-map min/max headers are included. No codebooks are stored (the
K-means clusters only determine placement, not values). No learned entropy tables.

### 6. Runtime cost
**Not stated.** No encode time, no decode time, no FPS, no GPU memory, no hardware named anywhere
in the paper. The only complexity remark is about block size: "larger block sizes yield only
marginal compression gains while significantly increasing runtime." (p4). The paper argues for
hardware decode qualitatively: "Image/video codecs bring additional advantages: they benefit from
decades of optimization, have standardized decoders, and are hardware-accelerated on most
platforms." (p3).

### 7. Rate model
**No entropy model, no quantization relaxation, no learned step sizes.** Rate control is entirely
non-learned: fixed per-parameter bit depths (Table II) with uniform mid-tread quantization whose
step size is derived from the observed min and max of each map, followed by JPEG XL. Lossy JPEG XL
is used only on the SH maps for 3DGS inputs; every other map, and every Scaffold-GS map, is
lossless. "In its lossy configuration, JPEG XL employs 8×8 DCT-based predictive coding." (p5).
Since JPEG XL produces the actual bytes there is no estimate to compare against, so no gap is
stated (and none can exist).

The compression gain comes from the **mapping**, not from a rate model. Fixed-size K-means (256
elements per cluster, K-means++ init, Euclidean distance on a feature vector) groups similar
elements (p6). The feature vector is the luminance SH AC coefficients for 3DGS input, and the
anchor positions concatenated with offset features for Scaffold-GS input (p4). The Nearest-
Neighbor-based Sorting (NNS) algorithm then places clusters into 16×16 blocks and elements within
blocks by picking, at each snake-scan position, the remaining element closest to the mean of its
already-assigned neighbours (p7, Eqs. 4–5). Measured mapping gain on a 2304-Gaussian visualization
(p8): "the random map results in a PNG file with 1558 bytes (using only the lossless PNG
compression), whereas the NNS-based map leads to 995 bytes".

Ablation (Table IV, p12), T&T, FSIM / DISTS / Size MB:
3DGS-based — 3DGS (no compression) 0.925/0.071/306.2; 3DGS (w/ pruning) 0.925/0.071/306.1;
GSICO 0.921/0.074/15.0; GSICO w/o NNS-based mapping 0.920/0.075/21.6; GSICO w/o JPEG XL coding
0.922/0.072/21.3; GSICO w/o NNS and JPEG XL 0.922/0.072/29.9.
Scaffold-GS-based — Scaffold-GS (no compression) 0.929/0.068/80.0; w/ pruning 0.929/0.068/79.9;
GSICO 0.926/0.070/9.9; w/o NNS 0.926/0.070/10.7; w/o JPEG XL 0.926/0.070/11.3; w/o NNS and JPEG
XL 0.926/0.070/12.5.

### 8. Scene dependence at a fixed setting
Per-scene sizes at the highest-quality operating point, from Fig. 7 (p10), printed under each
rendering as (PSNR, size):
- 3DGS-based GSICO: truck 19.1 MB, train 10.9 MB, drjohnson 11.9 MB, playroom 10.7 MB, bicycle
  36.8 MB, kitchen 16.0 MB.
- Scaffold-GS-based GSICO: truck 10.9 MB, train 8.9 MB, drjohnson 7.0 MB, playroom 5.6 MB,
  bicycle 33.2 MB, kitchen 11.9 MB.
Bicycle is 3.4× playroom for the 3DGS path and 5.9× playroom for the Scaffold-GS path, at one
fixed operating point. Because GSICO is a fixed-bit-depth codec, the output size tracks the input
Gaussian count almost linearly, which the paper acknowledges: "The challenging nature of the
Mip-NeRF360 dataset, where GS methods tend to produce a higher number of Gaussians than in other
datasets, makes compression particularly difficult." (p12).

### 9. Main quantitative results (all reported)
Table III (p9), highest-quality operating point only. Raw extracted rows are column-shifted (the
method name trails the row); the mapping below is confirmed against the prose on p10–p11.

| Method | M360 Size MB | LPIPS | SSIM | PSNR | DB Size | LPIPS | SSIM | PSNR | T&T Size | LPIPS | SSIM | PSNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DGS | 556.2 | 0.214 | 0.815 | 27.52 | 223.2 | 0.260 | 0.899 | 29.27 | 306.9 | 0.176 | 0.848 | 23.88 |
| Scaffold-GS | 187.6 | 0.221 | 0.815 | 27.73 | 57.4 | 0.253 | 0.909 | 30.33 | 80.0 | 0.176 | 0.854 | 24.13 |
| GSICO (w/ 3DGS) | 27.6 | 0.232 | 0.798 | 26.70 | 11.3 | 0.265 | 0.890 | 29.15 | 15.0 | 0.189 | 0.837 | 23.50 |
| GSICO (w/ Scaffold-GS) | 20.8 | 0.232 | 0.800 | 27.27 | 6.3 | 0.256 | 0.907 | 30.18 | 11.2 | 0.179 | 0.849 | 24.03 |

Raw extracted row for reference (Table III, p9): `556.2 0.214 0.815 27.52 223.2 0.260 0.899 29.27
306.9 0.176 0.848 23.88 | 3DGS` — the label follows the numbers in the text layer, and the T&T
"306.9" here conflicts with "306.6 MB" and "306.2" quoted in the prose (p10) and Table IV.

Note the paper's own prose on p10 says "from 306.6 MB to 15.0 MB on T&T" while Table III prints
306.9 and Table IV prints 306.2 for the same quantity.

**Gaussian / voxel counts: not stated.**

The two strongest baselines are the uncompressed inputs themselves (3DGS and Scaffold-GS), copied
above. The comparison against compression methods is **only in RD curves (Fig. 8, p11), with no
numeric table**. The prose gives these reported operating-point comparisons:
- p12: "At a rate of 10 MB, GSICO outperforms CompGS by approximately 0.3 dB in PSNR." (T&T)
- p12: "requiring approximately 5.9 MB less bitrate at a PSNR of 23.5 dB" versus SOG (T&T)
- p12: "Relative to the strongest competing solutions, Compact3D and CodecGS, at a SSIM of 0.906,
  GSICO requires approximately 5.85 MB and 2.85 MB less rate, respectively." (DB)
- p12: "at 19.1 MB (corresponding to its low-rate configuration), it achieves a quality level
  similar to the high-rate configurations of both Compact3D and RDO-Gaussian" (Mip-NeRF360)

**All rate points on Mip-NeRF 360: only one numeric point exists (GSICO w/ 3DGS at 27.6 MB /
26.70 dB and GSICO w/ Scaffold-GS at 20.8 MB / 27.27 dB), plus the prose reference to 19.1 MB as
the Scaffold-GS low-rate configuration.** The remaining four operating points per representation
appear only as unlabelled RD curve markers in Fig. 6 and Fig. 8.

Benchmarks excluded, verbatim (p9): "Although recent GS methods such as HAC [17], HAC++ [18],
HEMGS [37], and ContextGS [38] have reported competitive GS compression performances, they were
excluded from the comparison: HEMGS due to the lack of a publicly available implementation, while
the remaining methods did not provide a functional decoder at the time of this work." So GSICO is
**not** compared against the current leaderboard front-runners.

### 10. Densification and count control
None. GSICO never trains, so it has no densification. It has one count-reducing step, driven
purely by geometry of the image grid: "if the total number of Gaussians (or voxels) is not
divisible by 16, the excess elements are pruned to satisfy this requirement. Furthermore, if the
resulting number of blocks that can be formed from all Gaussians (or voxels) is a prime number,
256 Gaussians (equivalent to one 16×16 block) are pruned to make the block count factorable."
(p4–p5). Criterion is opacity: "Whenever pruning is required, opacity is used as the selection
criterion" (p5). Impact is negligible (Table IV: 306.2 → 306.1 MB and 80.0 → 79.9 MB). **The size
objective does not influence the count at all.**

### 11. Stated limitations and future work (verbatim)
The paper has no Limitations section. Future work, p13: "Future work will focus on three main
directions: i) implementing a learning-based quantization strategy, where quantization step sizes
are adaptively determined based on both the statistical properties and the perceptual relevance
of each GS parameter; ii) while JPEG XL was selected for its outstanding performance in coding
structured parameter images, emerging learning-based image codecs, such as JPEG AI [39], offer
promising opportunities for even greater efficiency, provided they are properly fine-tuned to the
statistical characteristics of GS parameters; and iii) extending the proposed codec framework to
support dynamic GS models represents another compelling research direction."

A limitation is stated in-line about Mip-NeRF360, p12: "The challenging nature of the Mip-NeRF360
dataset, where GS methods tend to produce a higher number of Gaussians than in other datasets,
makes compression particularly difficult. In this case, joint GS training and compression methods
tend to perform better."

### 12. Sentences touching a size / byte / bitrate / FPS TARGET, rate control, budget, desired
size, edge or mobile deployment (verbatim, with page)
- p1: "A single GS model size may achieve hundreds of megabytes, limiting its use in
  bandwidth-constrained or storage-limited settings."
- p1: "However, the computational cost associated with NeRF training and rendering pose critical
  challenges for practical deployment."
- p5: "The goal is to reduce parameter precision, thereby lowering the bitrate after image
  compression while preserving the quality of the reconstructed scene."
- p5: "Furthermore, for 3DGS models, the SH AC chrominance components are discarded to further
  reduce the bitrate, with negligible perceptual impact."
- p6: "Five operating points were defined for each representation, namely 3DGS-based and
  Scaffold-GS-based, ensuring that the resulting RD curves reflect a meaningful trade-off between
  rate and quality."
- p8 (the word "target" here is about block geometry, not a rate target): "The NNS algorithm is
  applied to the Gaussians (or voxels) in 𝐂N with a target block size of 16×16 pixels, NNS(𝐂N,
  16, 16)."
- p10: "Notably, no abrupt quality drops are observed between adjacent operating points,
  demonstrating the robustness and consistency of the quantization strategy across bitrate
  levels."
- p12: "The GSICO 3DGS-based configuration, although less competitive than its Scaffold-GS
  counterpart, still outperforms the image-based compression method SOG, requiring approximately
  5.9 MB less bitrate at a PSNR of 23.5 dB."
- p12: "Overall, GSICO distinguishes itself from prior work through its training-free and
  model-agnostic design, ensuring immediate applicability across diverse GS models and thus
  supporting practical deployment in real-world multimedia systems."
- p12: "The main objective of this work was to address the challenge of efficiently compressing
  GS models for practical deployment in real-world multimedia scenarios."
- p13: "However, their high storage requirements limit their applicability in
  bandwidth-constrained or storage-limited environments."

No sentence sets a byte target, a desired size, or names edge or mobile devices as a deployment
target for the method.

### 13. Code URL and licence
p2: "The link for the GSICO implementation will be available after acceptance." **No URL, no
licence stated.**

---

## 3. CodecGS

### 1. Citation
Title: "Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video
Codecs". First author: Soonbin Lee (with Fangwen Shu, Yago Sánchez, Thomas Schierl, Cornelius
Hellge), Fraunhofer Heinrich-Hertz-Institute (HHI), Germany. Venue: **not stated** (the PDF
carries only "arXiv:2501.03399v1 [cs.CV] 6 Jan 2025"). arXiv id from file name: 2501.03399.

### 2. Base representation
**Feature planes.** A progressive tri-plane (static k-planes / Hex-plane without the time axis)
predicts every Gaussian attribute from the point position: g(P(x)) = {c̃, s̃, q̃, σ̃} (p3, Eq. 1).
Point positions are stored explicitly and separately, everything else is implicit in the planes.

Render time **requires both feature-plane sampling and MLP decoding**. Four decoder MLPs, one per
attribute group: "Each attribute is decoded using an MLP g, a three-layer, fully connected
network. The intermediate layers contain 128 units each and use ReLU activation, except for the
output layer. The four decoder MLPs have a total size of 0.28 MB, which is included in the final
size results." (p11). Each feature plane has 8 channels at 512×512, four attributes, 32 channels
total (p5). Positions are contracted with the MERF piecewise-projective contraction (p6, Eq. 9)
before plane lookup. No hash grid. Video decoding (HEVC) is a one-time load-time cost.

The paper claims decoding is amortized: "once all attributes are predicted from the decompressed
feature plane, no additional overhead is required for rendering. Since the proposed method
follows the same densification process as the original 3DGS, its rendering time is comparable to
that of the original model." (p8). Read literally that means the decoded model **is** a standard
3DGS point set at render time, materialized once after load. Confirmed by "the predicted
attributes are used as inputs to the original 3DGS rasterizer" (p11). This is the one paper here
whose network cost is genuinely load-time only.

### 3. Budget handle
**KNOB, λ_ent.** The user supplies the entropy-loss weight λ_ent in
L = L_render + λ_ent·L_ent + λ_1·L_1 (p5, Eq. 8). "By adjusting the parameter λent in various
experiments, the feature plane effectively balances the trade-off between compressed size and
reconstructed visual quality." (p5). No target.

Exact λ_ent values printed: **only in Fig. 9 (p12), λ_ent ∈ {1e−10, 1e−9, 1e−8}** for the Room
scene. **The λ_ent values behind Table 1, Table 2 and Table 3 are not stated.** Table 3 says only
"We use λent to adjust the trade-off between the compressed size of the feature plane and visual
quality." (p8).

Secondary knobs with printed values: quantization step size Qstep ∈ {2⁰, 2², 2⁴, 2⁶, 2⁸, 2¹⁰,
2¹²} (Table 5, p11; 2⁸ chosen); video QP ∈ {1, 8, 16} (default QP = 1, Fig. 9). The paper
explicitly rejects QP as the rate knob: "Controlling rate-distortion with video QP is worse than
using λent with QP=1." (Fig. 9 caption, p12).

### 4. When it binds
**Fine-tuning stage, and the rate term only after 30k.** Two-phase training (p3): "First, we run
15k iterations using the original 3DGS training until the point densification phase concludes.
Then, we begin feature plane training to predict all gaussian attributes". Feature planes are
then trained for 40k iterations (p5). Progressive channel masking uses stages Ti = {0, 5000,
10000, 15000} with Li = {2, 4, 6, 8}, so "all channels are activated in training after 15k
iterations" of the plane phase (p5).

The rate term binds late: "To mitigate computational intensity, we apply entropy loss only after
the 30k iteration. We observed that performance converges properly with this condition. We also
calculate channel importance score at 30k iteration and determine the weights wc." (p5). So the
rate loss is active for the last 10k of 40k plane iterations, i.e. iterations 30k–40k of the
plane phase, on top of the 15k 3DGS phase.

Wall clock: "For the Mip-NeRF360 dataset, including the point initialization stage, training
takes about 90 minutes per scene." (p8). Hardware: "We conducted all experiments with a single
NVIDIA A100 GPU with 40GB memory." (p5).

### 5. What "size" means
Total = compressed feature-plane video + compressed point-position video + decoder MLPs. The
0.28 MB of MLP weights is explicitly included (p11). Table 3 (p8) splits it per scene into
"Feature Plane", "Positions", "Total Size". Positions are coded losslessly, so they set a floor:
Kitchen 2.34 MB, Playroom 2.87 MB, Truck 3.35 MB regardless of λ_ent. "Because the proposed
method does not involve pruning, the number of points remains the same as in the original 3DGS."
(p8). Codebooks: none. Entropy tables: none (a standard video bitstream carries its own).

### 6. Runtime cost
Hardware: single NVIDIA A100, 40 GB (p5). Training: ~90 min per Mip-NeRF360 scene including point
init (p8). GPU memory at render: not stated.

Encode / decode times, T&T (Table 4, p8):

| Video codec | Size (MB) | PSNR (dB) | Enc/Dec time (s) |
|---|---|---|---|
| HM [16] (w/o ours) | 27.9 | 23.71 | 292.9/2.9 |
| HM [16] (ours) | 7.46 | 23.63 | 267.8/2.8 |
| libx265 [11] (w/o ours) | 29.2 | 23.72 | 27.5/0.7 |
| libx265 [11] (ours) | 7.79 | 23.59 | 25.3/0.6 |

Render FPS: **not stated numerically.** Only the claim "its rendering time is comparable to that
of the original model" (p8) and the Fig. 1 caption "achieving comparable rendering speeds with
minimal overhead" (p1).

### 7. Rate model
**Frequency-domain entropy parameterization with additive uniform noise (not STE despite the
wording), on DCT coefficients of the planes.**
I(P) = E[−log p(P̃)], P̃ = P + u, u ~ U(−1/Qstep, 1/Qstep) (p3–p4, Eq. 2). The text says "the
model minimizes its entropy I with uniform noise U to use straight through estimator (STE)"
(p3–p4), which conflates the two relaxations; the equation shown is additive-noise.

The key move is that entropy is measured on transformed coefficients, not on raw plane values:
"the entropy optimization target is not the plane parameters, but their transformed coefficients.
We found the minimization of the I(F(P)) is highly beneficial for video codec performance."
(p4). F is a block-wise 2D DCT with 4×4 blocks: "our method uses a 2D DCT for each channel of
planes with a 4×4 block size, which is commonly used as the minimum transform unit (TU) size."
(p4, Eq. 3).

**Quantization step size is empirical, not learned:** "Unlike [23], which introduced a learnable
quantization step size Qstep, video codec may not be correctly incorporated into this optimization
process. Therefore, we must empirically determine the Qstep for transformed coefficients. Assuming
with the 16-bit scalar quantization mentioned above, we found that a Qstep of 2⁸ yields the best
results." (p4–p5). Qstep sweep on bonsai (Table 5, p11): 2⁰→16.6 MB/31.55; 2²→14.2/31.66;
2⁴→11.5/31.54; 2⁶→9.07/31.75; 2⁸→8.45/31.72; 2¹⁰→12.1/31.77; 2¹²→17.8/31.32.

**Channel-wise bit allocation.** Channel importance CIc(P) = (1/Σ Pi)·|∂Ei/∂Pc| summed over
training views (p5, Eq. 4), then wc = CI1(P)/CIc(P) with w1 = 1 (p5, Eq. 5) and
L_ent = Σ[c ∈ C] wc·I(F(Pc)) (p5, Eq. 6). "A higher value of wc further reduces entropy, resulting
in a lower bitrate allocation for channel c." (p5).

**Gap between estimated and actual encoded bytes: not stated.** The entropy term is never
compared to the HEVC bitstream size. Structurally there must be a gap, because the codec is
non-differentiable and outside the loop: "conventional standard video codecs are
non-differentiable, which prevents the establishment of an explicit optimization loss" (p3). The
entropy loss is a proxy that shapes the signal for the codec, not a byte estimate.

Post-training, planes are normalized to [0,1], scaled to 16-bit integers, concatenated
channel-to-frame into 32 frames, and coded with HM 16.0 RExt in random-access with all inter-frame
QP offsets zeroed and QP = 1, YUV400 16-bit (p5–p6, full command lines p11). Positions: Morton
sorted, quantized to 16-bit, packed into 512×512, coded **losslessly** with x265 `lossless=1`
(p6, p11).

### 8. Scene dependence at a fixed setting
Per-scene Mip-NeRF 360 storage at the Table 1 setting (Table 6, p13): bicycle 9.82, flowers
10.25, garden 15.22, stump 14.30, tree hill 9.92, room 7.91, counter 6.25, kitchen 7.68, bonsai
6.71, average 9.78 MB. Garden is 2.4× counter at one fixed λ_ent. Per-scene T&T and DB (Table 7,
p14): train 6.21, truck 8.72, avg 7.46; drjohnson 9.39, playroom 7.86, avg 8.62.

### 9. Main quantitative results (all reported)
Table 1 (p5), size in MB.

| Methods | M360 PSNR | SSIM | LPIPS | Size | DB PSNR | SSIM | LPIPS | Size | T&T PSNR | SSIM | LPIPS | Size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DGS [17] | 27.49 | 0.813 | 0.222 | 745 | 29.42 | 0.899 | 0.247 | 664 | 23.69 | 0.844 | 0.178 | 431.0 |
| Scaffold-GS [24] | 27.50 | 0.806 | 0.252 | 254 | 30.21 | 0.906 | 0.254 | 66.0 | 23.96 | 0.853 | 0.177 | 86.50 |
| Self-Organizing [26] | 26.01 | 0.772 | 0.259 | 23.9 | 28.92 | 0.891 | 0.276 | 8.40 | 22.78 | 0.817 | 0.211 | 13.1 |
| EAGLES [13] | 27.15 | 0.808 | 0.238 | 68.9 | 29.91 | 0.910 | 0.250 | 62.0 | 23.41 | 0.840 | 0.200 | 34.0 |
| LightGaussian [10] | 27.00 | 0.799 | 0.249 | 44.5 | 27.01 | 0.872 | 0.308 | 33.9 | 22.83 | 0.822 | 0.242 | 22.4 |
| Compact3DGS [20] | 27.08 | 0.798 | 0.247 | 48.8 | 29.79 | 0.901 | 0.258 | 43.2 | 23.32 | 0.831 | 0.201 | 39.4 |
| C3DGS [29] | 26.98 | 0.801 | 0.238 | 28.8 | 29.38 | 0.898 | 0.253 | 25.3 | 23.32 | 0.832 | 0.194 | 17.3 |
| RDOGaussian [37] | 27.05 | 0.802 | 0.239 | 23.4 | 29.63 | 0.902 | 0.252 | 18.0 | 23.34 | 0.835 | 0.195 | 12.0 |
| CompGS [23] | 27.26 | 0.802 | 0.239 | 16.5 | 29.69 | 0.900 | 0.280 | 8.77 | 23.71 | 0.840 | 0.210 | 9.61 |
| HAC [9] | 27.53 | 0.807 | 0.238 | 15.3 | 30.19 | 0.905 | 0.262 | 7.46 | 23.70 | 0.846 | 0.185 | 8.44 |
| Ours | 27.30 | 0.810 | 0.236 | 9.78 | 29.82 | 0.907 | 0.251 | 8.62 | 23.63 | 0.842 | 0.192 | 7.46 |

Strongest two baselines: HAC (15.3 MB / 27.53 dB on Mip-NeRF360) and CompGS-Liu (16.5 MB / 27.26
dB), both copied above.

**Gaussian counts: not stated numerically.** The paper only says the count equals the original
3DGS count because there is no pruning (p8).

**All rate points on Mip-NeRF 360.** Table 1 gives one point (9.78 MB). The ablation Table 2 (p7)
gives five configurations that were "adjusted ... to achieve results with similar visual quality",
i.e. these are ablation points, not a rate sweep of the final method:

| PC | wc | L_ent | PR | L_1 | PSNR (dB) | Size (MB) |
|---|---|---|---|---|---|---|
| | | | | ✓ | 27.27 | 23.45 |
| | | | ✓ | ✓ | 27.40 | 22.81 |
| | | ✓ | ✓ | ✓ | 27.31 | 10.68 |
| | ✓ | ✓ | ✓ | ✓ | 27.29 | 9.96 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 27.30 | 9.78 |

(The extracted checkmark grid is column-shifted in the text layer; the caption states PR is
progressive training and PC is piecewise-projective contraction, and rows add one component at a
time down to the full method at 9.78 MB.)

Per-scene three-point rate sweeps exist only in Table 3 (p8): Kitchen (11.2 MB/30.88, 9.07/30.56,
5.98/29.96), Playroom (12.1/30.38, 9.32/30.17, 6.49/28.92), Truck (12.2/25.19, 9.60/25.05,
7.93/24.28). The Mip-NeRF360-average RD curve (Fig. 5, p6) has no printed numbers.

### 10. Densification and count control
Densification is the **unmodified 3DGS process**, run for 15k iterations before the plane phase
begins, and it is then frozen: "we run 15k iterations using the original 3DGS training until the
point densification phase concludes. Then, we begin feature plane training ... This method allows
us to maintain a constant number of points from the start of feature plane training, while
continuously refining their positions." (p3). No pruning at all.

**The size objective does not influence the count.** Stated plainly: "The proposed method does not
affect the densification process, so the size of the point positions remains consistent across
scenes." and "Because the proposed method does not involve pruning, the number of points remains
the same as in the original 3DGS. However, because our model is independent of the densification
process, there is potential to reduce position size by adjusting the number of points." (p8).
This is why positions become the rate floor at low bitrates: "As we compress to lower bitrates
with higher Lent values, the proportion of the point position size gradually increases." (p8).

### 11. Stated limitations and future work (verbatim)
p8: "Complexity Overhead. The main limitation of the proposed method is its increased training
time, which is based on the slow convergence speed of the grid-based approach. For the
Mip-NeRF360 dataset, including the point initialization stage, training takes about 90 minutes
per scene. However, once all attributes are predicted from the decompressed feature plane, no
additional overhead is required for rendering. Since the proposed method follows the same
densification process as the original 3DGS, its rendering time is comparable to that of the
original model."

p8: "However, because our model is independent of the densification process, there is potential to
reduce position size by adjusting the number of points."

p8: "Since our method effectively leverages any DCT-based standard codec technology, there is
potential for further improvements as video codecs continue to advance."

There is no separate Limitations or Future Work section beyond these.

### 12. Sentences touching a size / byte / bitrate / FPS TARGET, rate control, budget, desired
size, edge or mobile deployment (verbatim, with page)
- p1: "We also propose channel-wise bit allocation to achieve a better trade-off between bitrate
  consumption and feature plane representation."
- p1: "This presents a significant challenge for rendering on devices with limited computing
  resources, such as mobile devices or head-mounted displays."
- p2: "For experimental data sets, our approach achieves high quality rendering with only a few MB
  of storage (typically <10MB) without significant loss in rendering quality."
- p2: "Despite these benefits, the high storage requirements of 3DGS make it challenging to deploy
  in resource-constrained environments."
- p4: "That is, the entropy optimization target is not the plane parameters, but their transformed
  coefficients."
- p5: "A higher value of wc further reduces entropy, resulting in a lower bitrate allocation for
  channel c."
- p5: "This weighting factor wc allocates bitrate for each channel, based on its importance. In
  other words, this method automatically determines wc to achieve better rate-distortion
  optimization."
- p5: "By adjusting the parameter λent in various experiments, the feature plane effectively
  balances the trade-off between compressed size and reconstructed visual quality."
- p7: "On the Mip-NeRF 360 dataset, our method consistently achieves significant reductions in
  bitrate consumption."
- p7: "Our method outperforms CompGS at high bitrates."
- p7: "A comparable trend is observed in the Tank&Temples dataset, where our approach demonstrates
  visual quality on par with HAC at low bitrates."
- p7: "Contrary to HAC, our method easily integrates with video codecs, enabling more efficient
  real-world deployment and applications across various hardware."
- p8: "We use λent to adjust the trade-off between the compressed size of the feature plane and
  visual quality." (Table 3 caption)
- p8: "As we compress to lower bitrates with higher Lent values, the proportion of the point
  position size gradually increases."
- p8: "This feature plane-based compression format makes a wide range of applications possible on
  mobile devices and other platforms with codec support."
- p12: "Beyond adjusting the parameter λent, the rate-distortion trade-off can also be controlled
  by modifying the quantization parameter (QP) in video codecs such as HM or FFmpeg. However, our
  experimental results demonstrate that modifying the video codec QP yields inferior performance
  compared to adjusting λent, as shown in Fig. 9."
- p12 (Fig. 9 caption): "RD curves with video QP adjustment. Controlling rate-distortion with
  video QP is worse than using λent with QP=1."

No byte target, no desired size, no rate-control loop.

### 13. Code URL and licence
Project page printed on p1: `https://fraunhoferhhi.github.io/CodecGS`. **No licence stated.**
Tooling URLs printed: `https://vcgit.hhi.fraunhofer.de/jvet/HM` (HM 16.0) and
`http://ffmpeg.org/` (p9/p10 reference list).

---

## 4. EAGLES

### 1. Citation
Title: "EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS". First author:
Sharath Girish (with Kamal Gupta, Abhinav Shrivastava), University of Maryland, College Park.
Venue: **not stated in the PDF** (stamp reads "arXiv:2312.04564v3 [cs.CV] 26 Sep 2024"; the
document is typeset in Springer LNCS style). arXiv id from file name: 2312.04564.

### 2. Base representation
Raw 3DGS with **quantized per-point latents plus tiny per-attribute MLP decoders**. Position,
scale and base-band (DC) color stay uncompressed floats; color SH, rotation and opacity are stored
as integer latents q ∈ ℤ^l and decoded by D: ℤ^l → ℝ^k (p5, Eq. 6): a = D(STE(q̂)). Latent
dimensions (Table 1, p19): color 16, rotation 8, opacity 1.

**Decode is once at load, not per frame.** "Before measuring FPS, we decode all latent attributes
using our decoder which is a one-time amortized cost of loading the parameters." (p9). So the
decoded model **is a standard 3DGS point set** and rendering needs no network, no hash grid and no
feature-plane sampling. Entropy decoding is also load-time: "Post training, we round q̂ to the
nearest integer and use entropy coding for efficiently storing the latents along with the decoder
D." (p5). Runtime GPU memory is not reduced by the latents: "Since our quantization decodes the
latents to floating point values before a forward or backward pass, no gains are obtained in
terms of runtime memory consumption for each Gaussian." (p14).

### 3. Budget handle
**KNOB, and an unusually indirect one.** There is no λ on any rate term. The user gets three
preset configurations plus a pruning percentage and a progressive resize scale. No target.

Exact values used for the three points (p9): "a) training for 30K iterations which is until
convergence b) a smaller configuration corresponding to more pruning c) training for 21K
iterations which is the end of the progressive training schedule." Named EAGLES, EAGLES-Small,
EAGLES-Fast. **The exact pruning percentage that defines EAGLES-Small is not stated.**

Defaults (p8): opacity reset every 2500 iterations; densification frequency 175 iterations;
pruning every 5000 iterations until 25000; "At each stage we remove 15% of the Gaussians although
a higher value can lead to even larger reductions at the cost of reconstruction quality";
progressive scale from 0.3 to 1.0 on a cosine schedule over 70 % of iterations; SH degree 3;
30000 iterations total.

Progressive resize-scale sweep values (Fig. 8, p13): 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
Progressive filter variants (Table 4, p14): None, Mean, Gaussian 5×5, Gaussian 7×7, Gaussian
15×15, Downsample.

### 4. When it binds
**During training from iteration 0 for the quantization**, and at fixed intervals for pruning.
Latents are trained end-to-end from the start with STE ("The latents are thus trained end-to-end
similar to the standard procedure of 3D-GS", p5). Progressive resolution runs over the first 70 %
of iterations (0 → 21k of 30k). Pruning runs every 5000 iterations until 25000, removing 15 % each
time. Densification interval raised from 100 to 175.

Wall clock and iteration counts, Mip-NeRF 360 averages (Table 1, p9): EAGLES 30k iterations
21m34s; EAGLES-Small 17m3s; EAGLES-Fast (21k iterations) 16m24s. 3D-GS* reference 23m20s. T&T:
EAGLES 11m39s, Small 10m7s, Fast 8m43s. DB (Table 2, p10): EAGLES 21m50s, Small 17m40s, Fast
16m30s. **Hardware for the timings is not named** beyond "UMD's supercomputing resources" (p14–p15);
Table 5 reports GPU RAM in GB but no GPU model.

### 5. What "size" means
"We calculate memory of all quantized and non-quantized parameters of the Gaussians for the
storage size." (p8). That wording covers the latents and the uncompressed attributes. Whether the
**decoder weights** are inside the reported number is **not stated** — p5 says the decoder is
stored alongside the latents ("use entropy coding for efficiently storing the latents along with
the decoder D"), but the evaluation sentence on p8 mentions only Gaussian parameters. Entropy
coding is applied to the latents; no codebooks, no planes. Header cost not stated.

Quantized-attribute compression measured in the ablation (p11): "The quantized attributes are
compressed from 220 MB, 452 MB, 1046 MB to 6 MB, 12 MB, 28 MB for the 3 scenes respectively
achieving ∼20−30× memory reduction." and "Note that the bulk of the memory post quantization is
from the non quantized attributes of scale, position, base color."

### 6. Runtime cost
Render FPS and train time are in Tables 1 and 2 (see field 9). Peak GPU RAM (Table 5, p14):

| Method | Bicycle Train | Bicycle Render | Truck Train | Truck Render | Playroom Train | Playroom Render |
|---|---|---|---|---|---|---|
| 3D-GS | 17.4G | 9.5G | 8.5G | 4.8G | 9.6G | 6.0G |
| EAGLES | 10G | 7.4G | 5.3G | 3.6G | 7.1G | 5.3G |

"For the Bicycle scene especially, compared to the 17G required by 3D-G, we consume only 10G GPU
RAM during training making it practical for many consumer GPUs with 12G RAM." (p14).
Encode / decode times: **not stated** (decode is only described as "a one-time amortized cost",
p9). Hardware model: **not named**.

### 7. Rate model
**No entropy model and no rate loss.** Quantization is a straight integer latent with STE: "As
quantized vectors are not differentiable, we maintain continuous approximations q̂ during training
and use the Straight-Through Estimator (STE) which rounds off q̂ to the nearest integer and
directly passes the gradient during backpropagation." (p5). Entropy coding is applied only
**post-training** to the rounded latents. Step sizes are not learned; the latent dimension per
attribute is a fixed hyperparameter (16 / 8 / 1). No estimated-versus-actual byte comparison
exists because nothing is estimated.

The paper deliberately declines a learned rate model: "While it is possible to improve feature
compression with additional tools such as complex decoders, learnable probability models [1] or
Gumbel annealing [44] and so on, they introduce a large overhead in various metrics such as
runtime GPU memory and training speed. We aim to utilize an approach which quantizes per-point
attributes with little to no cost to these efficiency metrics while still maintaining the
reconstruction quality." (p5).

What is quantized and what is not (p5): "While each vector in the attribute set A = {p, s, r, c,
o} can be quantized, we do not encode the base band color SH coefficient, the scaling coefficients
and the position vector as they are sensitive to initialization and result in large performance
drops when quantized."

Latent initialization (p18): q̂ = D⁻¹(a), by least squares for a linear decoder; latent learning
rate is the attribute learning rate scaled and divided by the decoder norm.

Side benefit claimed for opacity quantization (p6): "Quantization acts as a soft regularizer as it
requires more gradient updates to increase the opacity value from one quantization bin to the next
higher bin, thus preventing opacity saturation."

### 8. Scene dependence at a fixed setting
Per-scene Mip-NeRF 360 storage for EAGLES at the default 30k configuration (Table 2, p19–p20):
Bicycle 104MB / 2.26M Gaussians, Bonsai 29MB / 0.64M, Counter 25MB / 0.56M, Flowers 60MB / 1.33M,
Garden 74MB / 1.65M, Kitchen 45MB / 1.00M, Room 30MB / 0.67M, Stump 100MB / 2.22M, Treehill 72MB
/ 1.60M, Average 54MB / 1.33M. Bicycle is 4.2× counter at one fixed setting.
T&T (Table 3, p20): Train 21MB / 0.46M, Truck 38MB / 0.83M, avg 29MB / 0.65M.
DB (Table 4, p20): Drjohnson 69MB / 1.57M, Playroom 36MB / 0.80M, avg 52MB / 1.19M.

### 9. Main quantitative results (all reported)
Table 1 (p9), Mip-NeRF360 and Tanks&Temples. Storage in MB or GB as printed.

| Method | M360 PSNR | SSIM | LPIPS | Storage | FPS | Train Time | T&T PSNR | SSIM | LPIPS | Storage | FPS | Train Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plenoxels | 23.08 | 0.63 | 0.46 | 2.1GB | 7 | 25m49s | 21.08 | 0.72 | 0.38 | 2.3GB | 13 | 25m5s |
| INGP | 25.59 | 0.70 | 0.33 | 48MB | 9 | 7m30s | 21.92 | 0.75 | 0.31 | 48MB | 14 | 6m59s |
| M-NeRF360 | 27.69 | 0.79 | 0.24 | 9MB | 0.06 | 48h | 22.22 | 0.76 | 0.26 | 9MB | 0.14 | 48h |
| 3D-GS | 27.21 | 0.82 | 0.21 | 734MB | 134 | 41m33s | 23.61 | 0.84 | 0.18 | 411MB | 154 | 26m54s |
| 3D-GS* | 27.45 | 0.81 | 0.22 | 745MB | 110 | 23m20s | 23.63 | 0.85 | 0.18 | 430MB | 157 | 12m5s |
| EAGLES (Ours) | 27.23 | 0.81 | 0.24 | 54MB | 131 | 21m34s | 23.37 | 0.84 | 0.20 | 29MB | 227 | 11m39s |
| EAGLES-Small | 26.94 | 0.80 | 0.25 | 47MB | 166 | 17m3s | 23.10 | 0.82 | 0.22 | 19MB | 272 | 10m7s |
| EAGLES-Fast | 26.99 | 0.81 | 0.23 | 71MB | 111 | 16m24s | 23.02 | 0.83 | 0.20 | 38MB | 190 | 8m43s |

Table 2 (p10), Deep Blending:

| Method | PSNR | SSIM | LPIPS | Storage | FPS | Train Time |
|---|---|---|---|---|---|---|
| Plenoxels | 23.06 | 0.80 | 0.51 | 2.7GB | 11 | 27m49s |
| INGP | 24.96 | 0.82 | 0.39 | 48MB | 3 | 8m |
| M-NeRF360 | 29.40 | 0.90 | 0.25 | 8.6MB | 0.09 | 48h |
| 3D-GS | 29.41 | 0.90 | 0.24 | 676MB | 137 | 36m2s |
| 3D-GS* | 29.55 | 0.90 | 0.25 | 656MB | 123 | 23m5s |
| EAGLES (Ours) | 29.86 | 0.91 | 0.25 | 52MB | 130 | 21m50s |
| EAGLES-Small | 29.92 | 0.90 | 0.25 | 33MB | 160 | 17m40s |
| EAGLES-Fast | 29.85 | 0.91 | 0.25 | 63MB | 108 | 16m30s |

Strongest two baselines: 3D-GS* (27.45 dB / 745 MB on M360) and M-NeRF360 (27.69 dB / 9 MB but
0.06 FPS and 48 h), both copied above.

**All rate points on Mip-NeRF 360:** exactly three, 54 MB / 27.23 dB, 47 MB / 26.94 dB and 71 MB /
26.99 dB. Counts: 1.33M average for the 30k variant (Table 2, p20); counts for Small and Fast are
not stated.

Component ablation, Table 3 (p11–p12), three individual scenes, cumulative:

| Method | Train PSNR / Storage / #Gauss / FPS | Playroom PSNR / Storage / #Gauss / FPS | Bicycle PSNR / Storage / #Gauss / FPS |
|---|---|---|---|
| Vanilla | 21.94 / 262MB / 1.11M / 177 | 30.07 / 542MB / 2.29M / 144 | 25.13 / 1254MB / 5.31M / 61 |
| + Quantization | 21.60 / 46MB / 1.03M / 179 | 30.48 / 82MB / 1.81M / 142 | 24.86 / 192MB / 4.19M / 65 |
| + Progressive | 21.63 / 38MB / 0.85M / 194 | 30.39 / 75MB / 1.67M / 140 | 25.07 / 190MB / 4.19M / 71 |
| + Densification | 21.62 / 29MB / 0.64M / 202 | 30.40 / 54MB / 1.20M / 146 | 25.02 / 142MB / 3.11M / 82 |
| + Pruning | 21.65 / 21MB / 0.46M / 234 | 30.38 / 36MB / 0.80M / 169 | 25.04 / 104MB / 2.26M / 87 |

### 10. Densification and count control
Three levers, all count-based, none rate-based.
(a) **Densification interval** raised from 100 to 175 (p8). "increasing the densification interval
to 175 leads to fewer Gaussians without loss in reconstruction quality (penultimate row). Beyond
this value, we observe a sharp drop off in reconstruction quality." (p12).
(b) **Influence pruning.** W_{i,p} = αi·Ti = αi·Π[j=1..i−1](1 − αj), W_i = Σ_p W_{i,p} (p7, Eq. 7).
"This metric has no computational overhead as the weight values are calculated directly during the
rasterization process in Eq. (4). We thus obtain weight vectors for each iteration and accumulate
the weight values over N iterations (set as a hyperparameter) to account for all training views of
the full scene. After computing the weight vector, we identify a percentage of the Gaussians with
lowest weights and prune them and continue the training process." (p8). Fixed at 15 % every 5000
iterations until 25000 (p8).
(c) **Progressive coarse-to-fine rendering resolution**, 0.3 → 1.0 over 70 % of iterations, which
reduces the count as a side effect: "Another added benefit of progressive training is that fewer
Gaussians are required to represent coarser scenes" (p7); "decreasing the scale upto 0.3 has no
effect on PSNR but reduces the number of Gaussians needed to represent that scene" (p12).

**The size objective does not influence any of them** — there is no size objective. The pruning
percentage is a fixed schedule, not a controller, and the paper notes the trade-off is manual: "At
each stage we remove 15% of the Gaussians although a higher value can lead to even larger
reductions at the cost of reconstruction quality." (p8).

### 11. Stated limitations and future work (verbatim)
The paper has **no Limitations and no Future Work section**. The nearest statements:

p5: "While it is possible to improve feature compression with additional tools such as complex
decoders, learnable probability models [1] or Gumbel annealing [44] and so on, they introduce a
large overhead in various metrics such as runtime GPU memory and training speed."

p5: "While each vector in the attribute set A = {p, s, r, c, o} can be quantized, we do not encode
the base band color SH coefficient, the scaling coefficients and the position vector as they are
sensitive to initialization and result in large performance drops when quantized."

p11: "Note that the bulk of the memory post quantization is from the non quantized attributes of
scale, position, base color."

p12: "This is depicted in the views at Figs. 4 and 7 where the reconstruction quality is similar
to without pruning although it reintroduces minor artifacts."

p14: "Since our quantization decodes the latents to floating point values before a forward or
backward pass, no gains are obtained in terms of runtime memory consumption for each Gaussian."

p3 (scope note): "While our method can benefit from meta-learning as well, we restrict our current
approach to compressing a single scene for brevity."

### 12. Sentences touching a size / byte / bitrate / FPS TARGET, rate control, budget, desired
size, edge or mobile deployment (verbatim, with page)
- p2: "This leads to representations of each scene requiring high amounts of memory for storage
  (>1GB). The GPU runtime memory requirements during training and rendering is also much higher
  compared to standard NeRF methods, requiring almost 20GB of GPU RAM for several high-resolution
  scenes. They are thus not very practical for graphic systems with strong memory-constraints of
  storage or runtime memory or in low-bandwidth applications."
- **p8: "We optimize for 30000 iterations but can be controlled based on the time and memory budget
  for training."** — the single sentence in this paper that names a budget, and it is a
  **training-time** budget on the iteration count, not a model-size target.
- p10: "We reduce storage size by ∼15× making the representation suitable for devices with limited
  memory budgets."
- p14: "For the Bicycle scene especially, compared to the 17G required by 3D-G, we consume only 10G
  GPU RAM during training making it practical for many consumer GPUs with 12G RAM."

No byte target, no rate control, no bitrate, no edge or mobile deployment sentence.

### 13. Code URL and licence
p1: "Project page and code is available here." with the hyperlink `https://efficientgaussian.github.io`.
**No licence stated.**

---

## 5. Compact3D / CompGS (Navaneet et al.)

Naming caution: the file is named `compact3d`, the paper title in this v3 is **CompGS**, and it is
a different paper from the ACM-MM "CompGS" of Liu et al. that CodecGS and GSICO cite. Smol-GS
cites this one as "CompGS [6]".

### 1. Citation
Title: "CompGS: Smaller and Faster Gaussian Splatting with Vector Quantization". First authors:
K L Navaneet and Kossar Pourahmadi Meibodi (equal contribution), with Soroush Abbasi
Koohpayegani and Hamed Pirsiavash, University of California, Davis. Venue: **not stated in the
PDF** (stamp reads "arXiv:2311.18159v3 [cs.CV] 26 Sep 2024"; Springer LNCS typesetting). arXiv id
from file name: 2311.18159.

### 2. Base representation
**Raw 3DGS with vector-quantized attributes.** No MLP anywhere. "we aim to compress 3DGS which
uses a collection of 3D Gaussians to represent the 3D scene and does not contain grid like
structures or neural networks." (p5). Four independent K-means codebooks over grouped attributes:
DC color, SH, scale, rotation. Position and opacity are never quantized (p6).

Render time needs **no MLP, no decoder network, no hash grid, no feature planes and no entropy
decoding** — only a codebook lookup. The decoded model is a standard 3DGS point set, and the
lookup can even stay in place at render time: "it can reduce the memory footprint at the rendering
time since the index can act as a pointer to the correct code freeing the memory needed to
replicate those parameters for all Gaussians." (p2). This is the cheapest render-time story of the
five papers, and it shows in the FPS numbers.

### 3. Budget handle
**KNOB, two of them: codebook size and opacity-regularization weight λ_reg.** No target.

Exact values used for the rate points:
- Codebook sizes (p7): "we use a codebook of size 4096 for the color and 16384 (CompGS 16K) for
  the covariance parameters." CompGS 32K uses 32768 for covariance. Codebook-size sweep in Table 8
  (p14): #Codes ∈ {4K, 8K, 16K, 32K}; Fig. 4 (p15) sweeps codebook length per attribute.
- λ_reg (p7): "For opacity regularization, we use λreg = 10−7 from iterations 15K to 20K along
  with opacity based pruning every 1000 iterations and remove regularization thereafter."
  Sweep (Table 7, p13): λ_reg × 10⁻⁷ ∈ {0.5, 1.0, 2.0, 3.0}.
- Post-training bit quantization (CompGS 32K BitQ, Table 1 caption p8): "position parameters are
  16-bits, opacity is 8 bits, and the rest are 32 bits."
- K-means schedule knobs (Table 8, p14): iters ∈ {1, 3, 5, 10}, freq ∈ {50, 100, 200, 500}.

### 4. When it binds
**Fine-tuning stage inside a 30k run.** "The Gaussian parameters are trained without any vector
quantization till 20K iterations and K-means quantization is used for the remaining 10K
iterations." (p7). Within that: "We use just 1 such K-means iteration in our experiments once
every 100 iterations till iteration 25K and keep the assignments constant thereafter till the last
iteration, 30K. The K-means cluster centers are updated using the non-quantized Gaussian
parameters after each iteration of training." (p7). Opacity regularization binds earlier and in a
narrow window: iterations 15K–20K, with opacity pruning every 1000 iterations, then removed (p7).
Gradients flow through STE (p6).

Wall clock (Table 1, p8), minutes: Mip-NeRF360 — 3DGS* 21.6, CompGS 16K 22.8, CompGS 32K 29.4.
T&T — 3DGS* 12.2, CompGS 16K 15.6, CompGS 32K 20.6. "CompGS 16K variant requires marginally more
time than 3DGS while CompGS 32K needs 1.4× to 1.7× more training time." (p9). Hardware: "All
experiments were run on a single RTX-6000 GPU." (p7); the 3DGS † rows used an A6000 (Table 1
caption, p8).

### 5. What "size" means
Codebooks + per-Gaussian code indices + the unquantized position and opacity parameters. Indices
for one attribute are further run-length compressed after sorting: "we sort the Gaussians based on
one of the indices, e.g., rotation, so that Gaussians using the same code appear together in the
list. Then, for that index, instead of storing n integers, we store how many times each code
appears in the list, reducong the storage from n integers to k integers." (p7).

Breakdown (Table 6, p13): of 59 values per Gaussian, 4 are unquantized (3 position + 1 opacity)
and 55 are quantized. Those 4 are 68 % of memory in the 16-bit variant and 81 % in the 32-bit
variant. Within the quantized part, "Index 99% / Codebook 1%" for one variant and "98% / 2%" for
the other. So codebooks are ~0.2–0.4 % of the total file.

**Warning on the Mem column.** Table 1 reports megabytes. Tables 2, 4 and 9 report a **ratio**:
"All memory values are reported as a ratio of the method with our smallest model." (Table 2
caption, p10) and "In comparing model sizes, we normalize all methods by dividing them by the size
of our method to obtain compression ratio." (p9). Do not read "20.0" in Table 2 as 20 MB.

### 6. Runtime cost
Hardware: single RTX-6000 (p7). Encode / decode times: **not stated** (K-means cost is folded into
train time). GPU memory: not stated as a number, only the claim "CompGS maintains the other
advantages of 3DGS such as low inference memory usage and training time." (p9).

Render FPS (Table 1, p8): Mip-NeRF360 — 3DGS † 134, 3DGS * 149, CompGS 16K 346, CompGS 32K 344,
CompGS 32K BitQ 344. T&T — 3DGS † 154, 3DGS * 206, CompGS 16K 479, CompGS 32K 475, BitQ 475.
DL3DV-140 (Table 5, p12): 3DGS* 282, CompGS 32K 566. Per-scene FPS is in Table D.3 (p23), for
example Bicycle 3DGS 68 → CompGS 32K 242, Room 190 → 444, Playroom 181 → 589.

### 7. Rate model
**No entropy model, no rate loss, no learned step sizes.** Rate is set purely by codebook size and
index bit width. Quantization relaxation is STE: "In the forward pass of learning 3DGS, we
quantize the parameters and replace them with the quantized version (centroids) to do the
rendering and calculate the loss. Then, we do the backward pass to get the gradients for the
quantized parameters and copy the gradients to the non-quantized parameters to update them. We use
straight-through estimator proposed in STE [7]." (p6).

Cost control on K-means (p6): "we update the centroids after each iteration and update the
assignments once every t iterations. We observe that the modified approach works well even for
values of t as high as 500."

Why four codebooks (p6): "Performing a single K-means for the whole d dimensional parameters
requires a huge codebook since the different parameters of the Gaussian are not necessarily
correlated. Hence, we group similar types of parameters, e.g., all rotation matrices, together and
cluster them independently to learn a separate codebook for each."

Index coding is sorting plus RLE, not entropy coding. The paper flags entropy coding as unexploited
headroom (p21, Fig. C.1 caption): "Such a non-uniform distribution of cluster sizes suggest that
further compression can be achieved by using Huffman coding to store the assignment indices."
There is no bit estimate, so **no estimated-versus-actual gap is stated**.

Codebook transfer result (Fig. 5 p15, Table B.2 p21): a codebook trained on Counter and frozen,
with only assignments learned on the other eight Mip-NeRF360 scenes, gives
CompGS Shared Codebook 4K 0.797 / 26.64 / 0.242 and Shared Codebook 32K 0.800 / 26.780 / 0.247,
versus per-scene CompGS 4K 0.804 / 26.97 / 0.234 and CompGS 32K 0.806 / 27.12 / 0.240.

### 8. Scene dependence at a fixed setting
Per-scene Mip-NeRF360 memory for CompGS 32K (Table D.3, p23), same hyperparameters everywhere:
Bicycle 29 MB / 1314018 Gaussians, Bonsai 10 MB / 377227, Counter 10 MB / 385515, Flowers 23 MB /
1037132, Garden 30 MB / 1370624, Kitchen 13 MB / 564382, Room 9 MB / 327191, Stump 27 MB /
1226044, Treehill 23 MB / 1002290. Garden is 3.3× room at one fixed λ_reg and codebook size.
T&T: Train 12 MB / 500811, Truck 13 MB / 540081. DB: DrJohnson 17 MB / 714902, Playroom 10 MB /
393414.

### 9. Main quantitative results (all reported)
Table 1 (p8), Mem in MB, Train Time in minutes. † reported from 3DGS; * reproduced with official
code; timings for CompGS and 3DGS on RTX6000, † rows on A6000.

| Method | M360 SSIM | PSNR | LPIPS | FPS | Mem (MB) | Train (m) | T&T SSIM | PSNR | LPIPS | FPS | Mem (MB) | Train (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plenoxels† | 0.626 | 23.08 | 0.463 | 6.79 | 2,100 | 25.5 | 0.719 | 21.08 | 0.379 | 13.0 | 2300 | 25.5 |
| INGP-Base† | 0.671 | 25.30 | 0.371 | 11.7 | 13 | 5.37 | 0.723 | 21.72 | 0.330 | 17.1 | 13 | 5.26 |
| INGP-Big† | 0.699 | 25.59 | 0.331 | 9.43 | 48 | 7.30 | 0.745 | 21.92 | 0.305 | 14.4 | 48 | 6.59 |
| M-NeRF360† | 0.792 | 27.69 | 0.237 | 0.06 | 8.6 | 48h | 0.759 | 22.22 | 0.257 | 0.14 | 8.6 | 48h |
| 3DGS† | 0.815 | 27.21 | 0.214 | 134 | 734 | 41.3 | 0.841 | 23.14 | 0.183 | 154 | 411 | 26.5 |
| 3DGS* | 0.813 | 27.42 | 0.217 | 149 | 778 | 21.6 | 0.844 | 23.68 | 0.178 | 206 | 433 | 12.2 |
| LigthGaussian [16] | 0.805 | 27.28 | 0.243 | 209 | 42 | - | 0.817 | 23.11 | 0.231 | 209 | 22 | - |
| CGR [37] | 0.797 | 27.03 | 0.247 | 128 | 29.1 | - | 0.831 | 23.32 | 0.202 | 185 | 20.9 | - |
| CGS [47] | 0.801 | 26.98 | 0.238 | - | 28.8 | - | 0.832 | 23.32 | 0.194 | - | 17.28 | - |
| CompGS 16K | 0.804 | 27.03 | 0.243 | 346 | 18 | 22.8 | 0.836 | 23.39 | 0.200 | 479 | 12 | 15.6 |
| CompGS 32K | 0.806 | 27.12 | 0.240 | 344 | 19 | 29.4 | 0.838 | 23.44 | 0.198 | 475 | 13 | 20.6 |
| CompGS 32K BitQ | 0.797 | 26.97 | 0.245 | 344 | 12 | 29.4 | 0.832 | 23.35 | 0.202 | 475 | 8 | 20.6 |

Strongest two baselines: LightGaussian (27.28 dB / 42 MB / 209 FPS on M360) and CGS (26.98 dB /
28.8 MB), both copied above.

**Deep Blending has no averaged row with MB anywhere in the paper.** p8 says "Please see the
Appendix for results on Deep Blending dataset", and the appendix gives only per-scene rows
(Table D.3, p23): DrJohnson CompGS 32K 0.906 / 29.445 / 0.249 / 379 FPS / 17 MB / 714902 Gauss;
Playroom CompGS 32K 0.908 / 30.347 / 0.253 / 589 FPS / 10 MB / 393414 Gauss. Quality-only DB rows
appear in Table 2 (p10): 3DGS 0.899 / 29.49 / 0.246; K-means 4K 0.904 / 29.76 / 0.248; K-means 32K
0.903 / 29.75 / 0.247.

**All rate points on Mip-NeRF 360** for the full method: three, at 18 MB / 27.03 dB (16K),
19 MB / 27.12 dB (32K) and 12 MB / 26.97 dB (32K BitQ). Counts for those are not printed in
aggregate; Table 3 (p12) gives 845K for the opacity-regularized model versus 3.30M for 3DGS.
Additional Mip-NeRF360 points where **size is a ratio, not MB** (Table 2, p10): 3DGS 0.813 /
27.42 / 0.217 / 20.0; 3DGS-No-SH 0.802 / 26.80 / 0.229 / 4.8; Post-train K-means 4K 0.768 / 25.46
/ 0.266 / 1.7; K-means 4K Only-SH 0.811 / 27.25 / 0.223 / 4.8; K-means 4K 0.804 / 26.97 / 0.234 /
1.7; K-means 32K 0.808 / 27.16 / 0.228 / 1.8; Int16 0.804 / 27.25 / 0.223 / 10.0; Int8 no-pos
0.812 / 27.38 / 0.219 / 5.8; Int8 0.357 / 14.41 / 0.629 / 5.0; Int4 no-pos 0.489 / 17.42 / 0.525 /
3.4; 3DGS-No-SH Int16 0.789 / 26.59 / 0.237 / 2.4; K-means 4K, Int16 0.796 / 26.83 / 0.239 / 1.0.

Other datasets: DL3DV-140 (Table 5, p12) 3DGS* 0.905 / 29.06 / 27.37 PSNR-AM / 0.134 / 282 FPS /
291 MB versus CompGS 32K 0.895 / 28.42 / 26.97 / 0.149 / 566 FPS / 10 MB. ARKit-200 (Table 4,
p12) 3DGS 0.909 / 25.76 / 20.73 / 0.226 / 20.0 (ratio) versus CompGS 0.909 / 25.70 / 20.73 / 0.229
/ 1.7 (ratio).

### 10. Densification and count control
Count is controlled by an **opacity L1 regularizer plus threshold pruning**, and the paper's whole
Table 3 is a comparison against count-control baselines. Loss (p7):
L = L_3DGS + λ_reg·Σ[i=1..N] σi, "where L3DGS is the original loss of 3DGS with or without
quantization and λreg controls the sparsity of opacity. Finally, similar to the original 3DGS, we
remove the Guassians with opacity smaller than a threshold". Active 15K–20K only (p7).

Table 3 (p12), count-reduction comparison, #Gauss:

| Method | M360 SSIM/PSNR/LPIPS/#Gauss | T&T | DB |
|---|---|---|---|
| 3DGS | 0.813 / 27.42 / 0.217 / 3.30M | 0.844 / 23.68 / 0.178 / 1.83M | 0.899 / 29.49 / 0.246 / 2.80M |
| Min Opacity | 0.802 / 27.12 / 0.244 / 1.46M | 0.833 / 23.44 / 0.204 / 780K | 0.902 / 29.50 / 0.255 / 1.01M |
| Densify Freq | 0.794 / 26.98 / 0.255 / 1.07M | 0.832 / 23.36 / 0.206 / 709K | 0.902 / 29.76 / 0.258 / 844K |
| Densify Iters | 0.780 / 27.02 / 0.267 / 1.12M | 0.835 / 23.55 / 0.194 / 810K | 0.896 / 29.42 / 0.264 / 795K |
| Grad Thresh | 0.769 / 26.57 / 0.292 / 809K | 0.825 / 23.31 / 0.217 / 578K | 0.900 / 29.49 / 0.260 / 1.01M |
| Opacity Reg | 0.813 / 27.42 / 0.227 / 845K | 0.844 / 23.71 / 0.188 / 520K | 0.905 / 29.73 / 0.249 / 554K |

"Our opacity regularization results in 3.5× to 5× reduction in Gaussian count with nearly
identical performance as the larger models." (p12). λ_reg sweep (Table 7, p13), Mip-NeRF360:
0.5e−7 → 0.808 / 27.17 / 0.234 / 1.21M; 1.0e−7 → 0.806 / 27.12 / 0.240 / 845K; 2.0e−7 → 0.801 /
26.98 / 0.253 / 536K; 3.0e−7 → 0.794 / 26.83 / 0.266 / 390K. Note the sweep reports counts, not MB.

Baseline sweeps for the count knobs are in the appendix: min opacity {0.05, 0.1} (Table E.4,
p26), densification interval {300, 500} (Table E.5), densification end iteration {5000, 3000}
(Table E.6), gradient threshold {0.00035, 0.00045, 0.00055} (Table E.7).

**Does the size objective influence densification?** No. The regularizer targets opacity, not
bytes, and 3DGS densification runs unmodified: "There are no changes in the hyperparameters used
for training compared to 3DGS." (p7). The count knob and the byte knob (codebook size) are
separate and never coupled.

### 11. Stated limitations and future work (verbatim)
The paper has no Limitations or Future Work section. Verbatim limitation statements in the body:

p9: "A limitation of CompGS compared to 3DGS is the overhead in compute and training time
introduced by the K-means clustering algorithm. This is compensated in part by the reduced compute
and time due to the decrease in Gaussian count. CompGS 16K variant requires marginally more time
than 3DGS while CompGS 32K needs 1.4× to 1.7× more training time. However, this is still orders of
magnitude smaller than the high-quality NeRF based approaches like MipNerf-360."

p7 (Table 6 discussion, p13): "Some parameters like position of the Gaussians cannot be quantized
easily, so as shown in Table 6 after quantization, they dominate the memory(more than 80% of
memory). This means quantization cannot improve the compression any further."

p9–p10: "Note that there are large differences in reproduced results for 3DGS across various works
in the literature. We observe a median standard deviation of 0.05dB for PSNR when the experiment
is repeated 20 times with several scenes having differences more than 0.4dB across runs (refer
Appendix). One must be careful when analyzing as these variations are often comparable to
differences in performance between methods."

p21 (Fig. C.1 caption): "Such a non-uniform distribution of cluster sizes suggest that further
compression can be achieved by using Huffman coding to store the assignment indices."

p14/p21: "Sharing learnt codebook can further reduce the memory requirement and can help speed up
the training of CompGS. The quality of the codebook can be improved by learning it over multiple
scenes."

p28: "Further comparison with various NeRF based approaches and more analysis can help improve the
results on this dataset."

### 12. Sentences touching a size / byte / bitrate / FPS TARGET, rate control, budget, desired
size, edge or mobile deployment (verbatim, with page)
- p2: "This increases the storage and communication requirements of the model and its memory at the
  inference time, which can be very limiting in many real-world applications involving smaller
  devices. For instance, the large memory consumption may be prohibitive in storing, communicating,
  and rendering several radiance field models on AR/VR headsets."
- p2: "We are interested in compacting 3DGS representations without sacrificing their rendering
  speed to enable their usage in various applications including low-storage or low-memory devices
  and AR/VR headsets."
- p4: "However, the main drawback of 3DGS is its increased storage compared to NeRF methods which
  may limit its usage in many applications such as edge devices."
- p6: "This makes it inefficient for some applications including edge devices."
- p13: "Compressing the Gaussian parameters comes with a trade-off, particularly between
  performance and training time. In our method, the size of codebook, frequency of code assignment
  and number of iterations in code computation control this trade-off. Similarly, regularization
  strength can be modified in Gaussian count reduction to obtain a trade-off between performance
  and compression."
- p14 (Table 8 caption): "Depending on user's needs, it is possible to obtain models with fast
  training or high performance. The hyperparameters of vector quantization - number of K-Means
  iterations (iters), K-Means index assignment frequency (freq) and codebook size (#codes) can be
  varied to obtain the desired point on the curve."

No byte target, no rate control, no bitrate, no size budget.

### 13. Code URL and licence
p1: "Our code is available here: https://github.com/UCDvision/compact3d". **No licence stated in
the paper.**

---

## Cross-cutting summary

| Paper | Budget handle | Binds when | Size includes | Render-time network? | Rate model |
|---|---|---|---|---|---|
| Smol-GS | KNOB λo (count) + λq (bits), also R, nf | Training, stages 3 (15k–20k) and 4–5 (20k–35k) of 35k | zip of octree bytes + AC codes + 6 MLPs + hash + metadata | **Yes**, 4 MLPs run per frame per splat on view-dependent input | Learned hash-conditioned Gaussian NLL, learned step sizes, arithmetic coding |
| GSICO | KNOB, 5 preset (bit depth, JPEG XL quality) | Post-hoc, no training | JPEG XL streams + min/max headers + MLPs (Scaffold input only) | No for 3DGS input; yes (3 small MLPs) for Scaffold-GS input | None. Fixed bit depths + JPEG XL |
| CodecGS | KNOB λ_ent | Fine-tuning; rate loss only after iteration 30k of the 40k plane phase | HEVC feature-plane video + lossless position video + 0.28 MB MLPs | Decode once at load, then plain 3DGS rasterizer | DCT-domain entropy on 4×4 blocks, additive-uniform relaxation, fixed Qstep = 2⁸, CI-weighted per channel |
| EAGLES | KNOB, 3 presets (iterations, pruning) | Training from iteration 0 (STE latents), pruning every 5k until 25k | quantized + unquantized Gaussian params; decoder inclusion not stated | Decode once at load, then plain 3DGS rasterizer | None. STE integer latents, entropy coding applied post-training only |
| CompGS (Navaneet) | KNOB codebook size + λ_reg | Fine-tuning; VQ 20k–30k, opacity reg 15k–20k | codebooks + RLE'd indices + unquantized position and opacity | No | None. K-means VQ with STE, sorting + RLE on indices |

**Nothing in these five papers binds a byte target during training.** Every one exposes a knob and
reports where the knob landed. The single sentence in the set that uses the word "budget" as an
input is EAGLES p8, and it is a training-time budget on iteration count, not a model size:
"We optimize for 30000 iterations but can be controlled based on the time and memory budget for
training."

**Per-scene spread at one fixed setting** is 3.3× to 4.2× across Mip-NeRF360 in every paper that
prints per-scene sizes (Smol-GS 2.858–11.283 MB, CodecGS 6.25–15.22 MB, EAGLES 25–104 MB, CompGS
9–30 MB). A global λ therefore cannot deliver a per-scene byte target, which is exactly the gap a
controller would close.

**Smol-GS as a substrate.** The rate term λq·[LNLL(f) + LNLL(s)] is a real differentiable bit
estimate on 77 % of the Mip-NeRF360 bytes and it is already a loss term, so a per-scene controller
could adapt λq online against a measured bitstream size with no architectural change. The
remaining 20 % (coordinates) sits behind the discrete octree depth R, which also sets the width of
the positional encoding (3R = 48 bits into MLPoct), so R cannot be varied freely mid-training
without changing MLPoct's input dimension. Splat count is the multiplier on all three streams and
is governed by λo, which is active only during 15k–20k and only through an opacity penalty. A
size-targeting controller on Smol-GS would therefore most naturally act on λq continuously and on
λo during the compaction window, holding R fixed. The paper never validates the NLL estimate
against actual encoded bytes, so that gap must be measured before it can be used as a feedback
signal.
