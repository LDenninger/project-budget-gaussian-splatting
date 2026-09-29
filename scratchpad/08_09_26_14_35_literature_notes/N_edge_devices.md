# N. Edge devices, browsers, accelerators: what limits 3DGS rendering on constrained hardware

Reading notes for the BudgetGS project. Every numeric value below is **reported** by the
paper unless explicitly marked otherwise. Nothing is inferred, recomputed or estimated. Fields 9
and 10 are verbatim quotes with the PDF page number. Page numbers refer to pages of the PDF as
extracted by pymupdf (they match the printed page numbers in all seven files).

Papers covered:

| # | file | short name |
|---|---|---|
| 1 | `references/07_runtime_cost/mobile_gs_du_2026_arxiv2603.11531.pdf` | Mobile-GS |
| 2 | `references/07_runtime_cost/websplatter_han_2026_arxiv2602.03207.pdf` | WebSplatter |
| 3 | `references/07_runtime_cost/gaussian_blending_unit_ye_hpca2025_arxiv2503.23625.pdf` | GBU |
| 4 | `references/07_runtime_cost/ls_gaussian_wei_iccad2025_arxiv2507.21572.pdf` | LS-Gaussian |
| 5 | `references/07_runtime_cost/pocket_slam_li_icra2026_arxiv2606.24796.pdf` | Pocket-SLAM |
| 6 | `references/07_runtime_cost/octree_gs_ren_2024_arxiv2403.17898.pdf` | Octree-GS |
| 7 | `references/07_runtime_cost/flashgs_feng_2024_arxiv2408.07967.pdf` | FlashGS |

---

## 1. Mobile-GS

### 1. Citation
- Title: **Mobile-GS: Real-Time Gaussian Splatting for Mobile Devices**
- First author: Xiaobiao Du (University of Technology Sydney; also Li Auto Inc.). Co-authors Yida
  Wang, Kun Zhan, Xin Yu (Adelaide University, corresponding).
- Venue as printed on every page header: "Published as a conference paper at ICLR 2026".
- arXiv id from file name: **2603.11531** (page 1 stamp: `arXiv:2603.11531v1 [cs.CV] 12 Mar 2026`).

### 2. What it does (3 lines)
Replaces sorted alpha blending with a depth-aware order-independent weighted sum, then adds a
per-Gaussian MLP that predicts view-dependent opacity and a weight ϕ to repair the resulting
transparency artefacts. Compresses the model with first-order SH distillation (3 × 4 coefficients),
neural vector quantisation (K-means sub-vector codebooks + Huffman) and contribution-based pruning
on opacity ∧ scale. **Changes both the model (training, distillation, quantisation, pruning) and the
renderer (no sort, no tile-based rasteriser).**

### 3. Device bottleneck
**Sorting is named as the primary bottleneck.** Fig. 2 (p2) gives a per-stage runtime breakdown of
original 3DGS on four Mip-NeRF 360 scenes. Raw extracted values, in the legend order
"Preprocessing / Sorting / Rasterization" for Counter, Kitchen, Bicycle, Garden (ms):

```
Preprocessing  0.6  0.3  0.9  1.1
Sorting        3.1  4.1  4.9  3.5
Rasterization  2.9  4.6  5    4.3
```

So sorting ≈ 3.1–4.9 ms, comparable to or larger than rasterisation, and 3–10× preprocessing.
Fig. 2 right, FPS with and without sorting (3DGS vs 3DGS w/o sorting): Counter 431 → 882,
Bicycle 134 → 857, Garden 145 → 871, Room 337 → 864. Reported. (Figure caption: "Removing the
sorting step substantially accelerates 3DGS, achieving several-fold speedup".)

Fig. 3 (p4) annotates alpha blending as "Cost up to 1/2 render time".

Mobile-GS own breakdown, Fig. 6 (p9), legend order "MLPs / Preprocessing / Rasterization", same
four scenes (ms, read off the figure text; raw extraction quoted because the columns are jumbled):

```
0.24 0.23 0.23 0.25   (MLPs)
0.27 0.14 0.16 0.16   (Preprocessing)
0.54 0.51 0.57 0.62   (Rasterization)
```

Driving model properties named: primitive **count**, **SH degree** (3rd order → 3 × 16 coefficients
is called out as a storage and speed burden), and the sort, whose cost scales with the number of
sorted entries. Tile intersection is removed entirely, since Mobile-GS "eliminates the tile-based
rendering" (p4, Fig. 3 caption). Resolution appears only as the 1600 × 1063 figure for the mobile
test.

Power breakdown per Vulkan operator on Snapdragon 8 Gen 3, Table 13 (p19), watts:

| Method | Preprocessing | Sorting | MLP | Rasterization | Total |
|---|---|---|---|---|---|
| 3DGS* | 1.64 | 2.09 | 0 | 2.16 | 5.89 |
| SortFreeGS* | 1.78 | 0 | 0 | 2.25 | 4.03 |
| Mobile-GS | 0.17 | 0 | 0.24 | 0.42 | 0.83 |

All reported. Sorting alone draws 2.09 W of 5.89 W in 3DGS.

### 4. Representation on device
**Decoded, but only once, not per frame.** Attributes are stored quantised (K-means sub-vector
codebooks, Huffman-coded bitstream); SH is stored as two 3-vectors h_d, h_v and reconstructed by
two 16-bit MLPs: "In the inference stage, we only use these MLPs to decode the SH features once as
described in Eq. 6" (p6). The view-dependent opacity/ϕ MLP (three layers, 256/128/64) does run per
Gaussian per view, and Fig. 6 charges it 0.23–0.25 ms per frame.

On-device model footprint: **4.6 MB** on Mip-NeRF 360, **2.5 MB** on Tanks and Temples, **4.6 MB**
on Deep Blending (Table 1, p8). Codebook size 2¹⁰ chosen (Table 7, p10: 2⁶ → 3.84 MB / 25.52 dB,
2⁸ → 4.2 MB / 26.83, 2¹⁰ → 4.6 MB / 27.12, 2¹² → 7.9 MB / 27.15). **Render-buffer footprint: not
stated.**

### 5. Budget input
**No.** No FPS, latency or byte target is ever passed in. The closest control knobs are
free-parameters swept post hoc, not requested targets:
- Pruning quantile threshold τ (Table 5, p10): baseline 0.56 M Gaussians / 27.22 dB / 109 FPS*;
  τ = 0.1 → 0.55 M / 27.15 / 111; τ = 0.2 → 0.47 M / 27.12 / 127; τ = 0.4 → 0.34 M / 26.47 / 141;
  τ = 0.6 → 0.18 M / 25.85 / 164. Reported. τ = 0.2 adopted.
- Codebook size (Table 7, above).

Achieved-vs-requested pairs: **none exist in this paper.**

### 6. Device numbers (all reported)
Mobile device is a phone with a **Qualcomm Snapdragon 8 Gen 3 GPU**, renderer implemented in
"Vulkan 2.0" (p7). Desktop numbers are on an RTX 3090 (training) / RTX 3090 Ti (Fig. 1b,c).

Table 2 (p8), Mip-NeRF 360, FPS* = mobile FPS on Snapdragon 8 Gen 3:

| Method | PSNR | FPS* | Storage | Training |
|---|---|---|---|---|
| 3DGS* | 27.01 | 8 | 61.8 MB | 0.5 h |
| Mini-Splatting* | 27.02 | 12 | 36.9 MB | 0.4 h |
| Speedy-Splat | 26.92 | 19 | 79.5 MB | 0.4 h |
| HAC | 26.98 | 12 | 11.8 MB | 0.7 h |
| LocoGS-S | 27.02 | 17 | 8.5 MB | 0.8 h |
| C3DGS | 27.03 | 14 | 30.6 MB | 0.6 h |
| GES | 26.98 | 18 | 29.4 MB | 0.7 h |
| SortFreeGS* | 26.74 | 24 | 64.3 MB | 1.3 h |
| Mobile-GS | 27.12 | **127** | 4.6 MB | 1.5 h |

(* = quantised through Huffman encoding.)

Table 12 (p19), thermal behaviour on the same phone: Cold-start FPS 3DGS 8, Speedy-Splat 19,
SortFreeGS 24, Mobile-GS 127. Steady-state FPS 3, 10, 18, **74**. Reported. Text: "On mobiles, FPS
drops over time because of thermal throttling, power limits, GPU clock downscaling, and NPU/CPU
frequency limits."

Table 8 (p16), also on Snapdragon 8 Gen 3, Mip-NeRF 360: SortFreeGS* 26.74 / 64.3 MB / 18 FPS*,
GES 27.02 / 29.4 MB / 24 FPS*, Ours 27.12 / 4.6 MB / 127 FPS*. **Note the inconsistency with
Table 2, where GES is 26.98 / 18 FPS* and SortFreeGS* is 26.74 / 24 FPS*. The two tables swap the
FPS values of GES and SortFreeGS*.**

Fig. 1 (p1) teaser: "116 FPS rendering speed in the 1600 × 1063 resolution for Bicycle on the mobile
equipped with the Snapdragon 8 Gen 3 GPU". Desktop teaser triples (PSNR / Size / FPS) for that
scene: 3DGS 24.89 / 1.2 GB / 134; Mobile-GS 24.82 / 4.8 MB / 1098; SortFreeGS 24.15 / 1.3 GB / 612.

Power (Table 13) is quoted in field 3 above: total 0.83 W vs 5.89 W for 3DGS* and 4.03 W for
SortFreeGS*.

### 7. Quality numbers (Table 1, p8; all reported)

| Method | MipNeRF360 PSNR / SSIM / LPIPS / Storage / FPS | T&T | Deep Blending |
|---|---|---|---|
| 3DGS | 27.21 / 0.815 / 0.214 / 839.9 MB / 174 | 23.14 / 0.841 / 0.183 / 371.5 MB / 236 | 29.41 / 0.903 / 0.243 / 697.3 MB / 214 |
| LightGaussian | 27.08 / 0.801 / 0.244 / 60.4 MB / 227 | 22.61 / 0.803 / 0.264 / 29.9 MB / 392 | 28.74 / 0.856 / 0.325 / 48.2 MB / 271 |
| AdR-Gaussian | 26.95 / 0.792 / 0.259 / 358.2 MB / 254 | 22.74 / 0.809 / 0.251 / 214.6 MB / 372 | 28.92 / 0.863 / 0.305 / 251.4 MB / 284 |
| SortFreeGS | 27.02 / 0.775 / 0.267 / 851.4 MB / 731 | 22.81 / 0.817 / 0.254 / 471.5 MB / 848 | 28.69 / 0.852 / 0.326 / 724.2 MB / 793 |
| Speedy-Splat | 26.92 / 0.782 / 0.296 / 79.4 MB / 401 | 23.08 / 0.821 / 0.241 / 62.4 MB / 527 | 29.11 / 0.864 / 0.309 / 71.2 MB / 463 |
| C3DGS | 27.03 / 0.797 / 0.247 / 30.6 MB / 184 | 23.32 / 0.831 / 0.202 / 21.8 MB / 174 | 29.73 / 0.900 / 0.258 / 24.7 MB / 189 |
| **LocoGS-S** (strong baseline 1) | 27.02 / 0.805 / 0.241 / **8.5 MB** / 292 | 23.23 / 0.837 / 0.204 / 6.8 MB / 325 | 29.76 / 0.903 / 0.251 / 7.8 MB / 322 |
| **3DGS** (strong baseline 2) | see row 1 | | |
| Mobile-GS | **27.12 / 0.807 / 0.235 / 4.6 MB / 1125** | 23.09 / 0.831 / 0.208 / 2.5 MB / 1179 | 29.93 / 0.906 / 0.243 / 4.6 MB / 1132 |

FPS in Table 1 is desktop (RTX 3090 / 3090 Ti). Counts in millions appear only in the ablations:
baseline 0.56 M, pruned 0.47 M Gaussians on Mip-NeRF 360 (Table 4, p10). Table 6 (p10): pruning
transplanted onto MaskGaussian 27.24 / 1.21 M → 27.16 / 0.84 M; onto Mini-Splatting 27.41 / 0.58 M
→ 27.38 / 0.47 M.

Ablation (Table 3, p8, Mip-NeRF 360, RTX 3090): full 27.12 / 1125 FPS / 4.6 MB; w/o
order-independent 27.26 / 684 / 4.5 MB; w/o view-dependent 26.68 / 1227 / 4.4 MB; w/o neural
quantisation 27.33 / 841 / **121 MB**; 0th-order SH 27.04 / 1219 / 3.6 MB; 2nd-order 27.13 / 917 /
7.3 MB; 3rd-order 27.15 / 841 / 9.6 MB.

### 8. Paper-specific (n/a)
Not Octree-GS or FlashGS.

### 9. Stated limitations and future work (verbatim, p17)
> "Despite its advantages, Mobile-GS contains several limitations: (1) Training Cost and Complexity:
> Although inference is fast, training Mobile-GS remains computationally intensive due to the
> proposed components (e.g., spherical harmonics distillation, neural vector quantization, neural
> view-dependent enhancement). Additionally, the model requires pretraining on desktop GPUs before
> mobile deployment, limiting its accessibility for real-time data acquisition and retraining on the
> device. (2) Scene Generalization: While Mobile-GS performs well on standard benchmarks, it is
> optimized per-scene and does not generalize across scenes without retraining. This limits its
> immediate usage in applications requiring dynamic scene capture or rendering in unseen
> environments, such as real-time AR reconstruction. (3) Quantization Degradation: Although the
> proposed neural vector quantization is highly effective in compressing Gaussian attributes, there
> still remains a trade-off between compression ratio and reconstruction quality, especially for
> fine-grained appearance details. Excessive quantization may introduce minor color shifts or
> blurring artifacts in highly textured regions." (p17)

### 10. Every sentence touching a memory / FPS / time TARGET, "budget", or edge/mobile constraint (verbatim)
- p1: "However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices."
- p1: "Furthermore, to facilitate deployment on memory-constrained mobile platforms, we also introduce first-order spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks."
- p1: "Our Mobile-GS integrates depth-aware order-independent rendering, compression, and distillation techniques to deliver comparable rendering quality compared with the original 3DGS, while substantially reducing the storage requirements to 4.8 MB and achieving 1098 FPS on the unbounded scene, thereby enabling efficient deployment on mobile devices."
- p2: "The computational overhead of rendering tens of thousands of Gaussians, especially with the view-dependent effects, exceeds the capabilities of most modern mobile GPUs."
- p2: "This limitation highlights the pressing need for efficient solutions to enable real-time Gaussian splatting on resource-constrained platforms, such as smartphones and AR headsets."
- p2: "Therefore, to achieve real-time rendering performance on such platforms, there are several critical factors: (1) Order-free rendering: eliminating the time-consuming sorting process; (2) Quantization: compressing 3D Gaussians to reduce memory and bandwidth consumption; (3) Fewer Gaussian points: reducing the number of primitives while preserving visual quality."
- p3: "However, these methods cannot be directly employed in edge devices due to the large storage and significant inference delay."
- p3: "Neural Vector Quantization : To deploy 3DGS on mobile devices, the quantization process is necessary to largely reduce storage usage and improve rendering speed."
- p4: "To address these limitations, we propose a more efficient compression framework that preserves essential Gaussian features while achieving compact attribute representation, making it particularly suitable for deployment on resource-constrained mobile devices."
- p4: "Although this sorting-based mechanism ensures that Gaussians closer to the camera have a more significant contribution to the final image, it incurs considerable computational overhead, particularly detrimental in latency-sensitive and resource-constrained equipment like mobile devices."
- p6: "This entropy-based compression technique significantly reduces the bitstream size without compromising runtime performance, enabling the deployment of our method on storage-constrained devices."
- p6: "This factorization further reduces memory costs for mobile devices."
- p7: "Deployment on Mobiles: To evaluate the efficiency of our method on resource-constrained devices, we implement our approach using Vulkan 2.0, a modern, cross-platform graphics and compute API."
- p8: "Evaluation on Mobile: To validate the real-time performance on the edge device, we deploy our proposed Mobile-GS on a mobile device equipped with the Snapdragon 8 Gen 3 GPU for the evaluation."
- p9: "When we do not utilize the proposed neural vector quantization, the storage cost increases dramatically, indicating its necessity for mobile deployment."
- p16: "Overall, Mobile-GS is carefully tailored to minimize resource consumption, reduce Gaussian parameter storage, and maintain real-time rendering performance on mobile hardware."
- p16: "To address memory limitations, a neural vector quantization strategy is employed, improving storage efficiency and enabling large-scale scene representations to be deployed on mobile devices with limited memory."
- p19: "On mobiles, FPS drops over time because of thermal throttling, power limits, GPU clock downscaling, and NPU/CPU frequency limits."

### 11. Code URL and licence
Project page as printed on p1: `https://xiaobiaodu.github.io/mobile-gs-project/`. Reproducibility
statement (p11): "we will release all codes, including training and evaluation code, upon acceptance
of the paper." No GitHub URL printed. **Licence: not stated.**

---

## 2. WebSplatter

### 1. Citation
- Title: **WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via
  WebGPU**
- First author: Yudong Han (Institute for Artificial Intelligence, Peking University). Co-authors
  Chao Xu, Xiaodan Ye, Zilong Dong (Tongyi Lab, Alibaba Group), Weichen Bi, Yun Ma (PKU).
- Venue: **not stated** on the PDF (no conference header; arXiv preprint style).
- arXiv id from file name: **2602.03207** (page 1 stamp: `arXiv:2602.03207v1 [cs.GR] 3 Feb 2026`).

### 2. What it does (3 lines)
A full WebGPU compute+render pipeline for 3DGS: a wait-free hierarchical-Blelloch radix sort that
never spin-waits across workgroups, screen-space AABB culling in a compute preprocess, and hardware
rasterisation of one quad per splat whose size is derived from the splat's opacity. **Renderer
only.** No training, no pruning, no change to the stored model: it reads standard PLY files.

### 3. Device bottleneck
**Device-dependent, and the paper says so explicitly.** Table 2 (p6), stage share of frame time:

| Device | Scene | Total (ms) | Pre-process | Sort | Render |
|---|---|---|---|---|---|
| RTX 3070 | garden | 9.5 | 13.1 % | 24.3 % | 62.6 % |
| RTX 3070 (Firefox) | garden | 11.4 | 5.6 % | 37.7 % | 56.8 % |
| MacBook Air M4 | garden | 63.7 | 30.1 % | 18.1 % | 51.8 % |
| MacBook Pro M1 | garden | 112.0 | 24.3 % | 17.0 % | 58.7 % |
| Nvidia MX350 | garden | 164.3 | 47.2 % | 27.6 % | 25.3 % |
| Intel NUC (iGPU) | garden | 151.2 | 40.0 % | 29.2 % | 30.8 % |
| Redmi K70 Pro | bicycle-c | 33.6 | 10.2 % | 28.8 % | 61.0 % |
| Oppo Find X2 | bicycle-c | 105.2 | 13.4 % | 37.7 % | 48.8 % |

All reported. Paper's own reading (p6): high-end GPUs, Apple M-series and the Redmi K70 Pro are
**render-bound** (51.8 %–62.6 % of frame time in rasterisation); weak GPUs (MX350, Intel iGPU) shift
to **pre-process-bound** (up to 47.2 %), "because a large number of raw splats need to be scanned in
the pre-processing stage. The limited memory throughput of these low-end devices causes performance
degradation." **"WebSplatter never becomes sort-bound across all tested hardware"** (p6), after
their sort fix. The baselines are sort-bound, see Table 4 below.

Scaling with splat count (Table 3, p7). The property that drives it is the **raw splat count N**,
and specifically the pre-process cost grows faster than render cost as N grows:

| Device | Scene | Splats N | Total | Pre-proc | Sort | Render |
|---|---|---|---|---|---|---|
| MacBook Air M4 | van gogh | 341,294 | 11.654 | 0.761 | 2.046 | 8.847 |
| MacBook Air M4 | train | 1,026,508 | 31.518 | 5.332 | 8.522 | 17.664 |
| MacBook Air M4 | bonsai | 1,244,819 | 13.088 | 1.349 | 1.746 | 9.993 |
| MacBook Air M4 | truck | 2,541,226 | 30.350 | 7.715 | 6.805 | 15.830 |
| MacBook Air M4 | bicycle | 6,131,954 | 56.013 | 19.640 | 7.683 | 28.690 |
| MacBook Air M4 | garden | 5,834,784 | 63.737 | 19.183 | 11.518 | 33.036 |
| Redmi K70 Pro | van gogh | 341,294 | 28.960 | 0.422 | 4.171 | 24.367 |
| Redmi K70 Pro | train | 1,026,508 | 76.369 | 2.185 | 5.066 | 69.118 |
| Redmi K70 Pro | bicycle-c | 1,063,091 | 33.633 | 3.431 | 9.673 | 20.529 |
| Redmi K70 Pro | bonsai | 1,244,819 | 52.874 | 2.290 | 5.688 | 44.896 |
| Redmi K70 Pro | truck | 2,541,226 | 52.719 | 2.101 | 4.435 | 46.183 |
| Lenovo MX350 | van gogh | 341,294 | 20.946 | 1.360 | 9.104 | 10.482 |
| Lenovo MX350 | train | 1,026,508 | 76.791 | 22.861 | 15.985 | 37.945 |
| Lenovo MX350 | bonsai | 1,244,819 | 47.665 | 15.439 | 11.297 | 20.929 |
| Lenovo MX350 | truck | 2,541,226 | 81.666 | 29.702 | 18.650 | 33.314 |
| Lenovo MX350 | bicycle | 6,131,954 | 142.771 | 67.336 | 38.659 | 36.776 |
| Lenovo MX350 | garden | 5,834,784 | 164.255 | 77.452 | 45.253 | 41.550 |

All ms, reported. Note the frame time is NOT monotone in N (bonsai at 1.24 M is faster than train at
1.03 M on every device), so **depth complexity / overdraw matters more than count** in the render
stage.

**Tile-Gaussian pairs are deliberately avoided**: "Tile-based methods, common in CUDA
implementations, build and process a large buffer of ⟨tile_i, splat_i⟩ pairs that map screen tiles
to intersecting splats. Generating and sorting this buffer incurs heavy memory traffic. On
heterogeneous devices such as laptops and mobile phones, limited bandwidth makes this step a major
bottleneck." (p4)

Overdraw / quad area is the other driver: disabling opacity-based quad sizing (-RADIUS) raises frame
time 15 % (63.7 → 73.2 ms) with the render stage up 18.8 % (33.0 → 39.2 ms) on the MacBook Air M4
(Table 5, p8).

### 4. Representation on device
**Full uncompressed Gaussians are uploaded; the compression is per-frame and internal.** Model comes
from "standard PLY" (p5). The preprocess packs the projected 2D ellipse major/minor axes into two
u32 (fp16 per component) and the RGB colour + opacity into a single u32 in RGBA8 (4 bytes/splat),
for the stated reason "This data compression significantly reduces the memory bandwidth required
during the final rendering stage" (p3–4). That is a per-frame intermediate, not a stored compressed
model.

**Peak GPU memory, garden scene (5,834,784 splats), RTX 3070 on Windows** (p8, all reported):
WebSplatter ≈ **1.20 GB**; W1 (KeKsBoTer/web-splat) 1.90 GB (36 % more); W2
(MarcusAndreasSvensson) **> 2.80 GB** (over 57 % more); S2 (mkkellogg) 1.19 GB; S1 (antimatter15)
0.83 GB. Model-only footprint and render-buffer footprint are **not broken out.**

Memory capacity is a *hard* failure mode here, not just a slowdown: baselines crash out of memory on
the Oppo Find X2 and the iPhone 15 Pro Max, and KeKsBoTer crashes on the Redmi K70 Pro for truck
(2.5 M splats) while WebSplatter renders it in 52.7 ms.

### 5. Budget input
**No.** Nothing takes a memory, FPS or time target. V-Sync is enabled for all experiments, which
caps but does not target. No achieved-vs-requested pairs exist.

### 6. Device numbers (all reported)
Devices (p5): desktop RTX 3070; Intel NUC mini-PC with integrated Core i9-9980HK GPU; MacBook Pro
(M1); MacBook Air (M4); Lenovo laptop with Nvidia MX350; phones iPhone 15 Pro Max, Redmi K70 Pro
(Snapdragon 8 Gen 3), Oppo Find X2 (Snapdragon 865). Browsers: Chrome (default), Safari (S),
Firefox (F). Implementation: TypeScript + WGSL, ~6,600 LOC, built with Vite.

Table 1 (p5), total frame time in **ms**, lower is better, "—" = crash:

| Device | Scene | Ours | S1 antimatter15 | S2 mkkellogg | W1 KeKsBoTer | W2 Svensson |
|---|---|---|---|---|---|---|
| RTX 3070 | garden | 9.5 | 62.4 | 88.8 | 14.4 | 15.1 |
| RTX 3070 (F) | garden | 11.4 | 54.9 | 67.8 | — | — |
| MacBook Air M4 | garden | 63.7 | 80.1 | 144.3 | 67.8 | 268.3 |
| MacBook Air M4 (S) | garden | 68.6 | 78.5 | 111.6 | 81.3 | — |
| MacBook Pro M1 | garden | 112.0 | 124.4 | 225.2 | 510.8 | 402.3 |
| Nvidia MX350 | garden | 164.3 | 180.0 | 869.5 | 341.0 | — |
| Intel NUC (iGPU) | garden | 151.2 | 341.7 | 368.4 | 404.0 | — |
| iPhone 15 Pro Max (S) | bicycle-c | 38.5 | 50.5 | — | 45.7 | — |
| Redmi K70 Pro | bicycle-c | 33.6 | 55.3 | 76.4 | 39.5 | — |
| Oppo Find X2 | bicycle-c | 105.2 | 112.7 | 182.6 | — | 4553.6 |

Speedups quoted in text (p5): 1.52× over best WebGPU baseline on RTX 3070; 1.06× on MacBook Air M4;
1.18× on Redmi K70 Pro; 2.26× on Intel NUC; "over 4.5× faster than the WebGPU baseline" on the
MacBook Pro M1. Abstract claims "1.2× to 4.5× speedups"; the conclusion says "1.06 to 2.26 times".
Stability on MacBook Air M4 garden (5.8 M splats): standard deviation 22.303 ms, P99 116.851 ms.

Table 4 (p8), sort-stage time in ms:

| Device | Ours | S1 | S2 | W1 | W2 |
|---|---|---|---|---|---|
| RTX 3070 | 2.306 | 55.8 | 72.0 | 5.374 | 5.308 |
| RTX 3070 (F) | 4.307 | 49.0 | 50.6 | — | — |
| Redmi K70 Pro | 9.673 | 39.7 | 48.6 | 17.629 | — |
| MacBook Air M4 (S) | 15.022 | 32.0 | 48.6 | 20.557 | — |
| MacBook Pro M1 | 19.027 | 58.7 | 154.2 | **458.424** | 261.816 |
| Lenovo MX350 | 45.253 | 98.5 | 556.2 | 150.733 | — |

Sort speedups quoted: >24× vs CPU sort on RTX 3070 (2.3 vs 55.8 ms), >31× vs the WASM sort, >24×
vs the spin-wait sort on MacBook Pro M1 (19.0 vs 458.4 ms).

Ablation, MacBook Air M4, garden (Table 5, p8, ms): FULL pre-process 19.18 / sort 11.52 / render
33.04 / total 63.74; -CULL 21.79 / 11.05 / 34.52 / 67.36; -RADIUS 20.42 / 13.59 / 39.20 / 73.20.

Firefox notes (p5–6): on Intel NUC "the built-in queries report a frame time of 2.8 ms, the
JavaScript API measures 258.36 ms", a confirmed Firefox bug; on MX350 "WebGPU is disabled by
blocklist".

### 7. Quality numbers on Mip-NeRF 360 / T&T / Deep Blending
**Not stated.** No PSNR / SSIM / LPIPS table anywhere. Quality is a single visual side-by-side
against the native CUDA renderer (Fig. 3, p7): "our WebGPU pipeline achieves a visual quality
comparable to the native baseline". Scenes used, with splat counts (p5, reported): bicycle
6,131,954; garden 5,834,784; truck 2,541,226; bonsai 1,244,819; train 1,026,508; bicycle-c
1,063,091; Van Gogh Room 341,294.

### 8. Paper-specific (n/a)

### 9. Stated limitations and future work
**No limitations or future-work section exists in this paper.** The closest statements are
observations about the ecosystem and about where optimisation should go next:
> "These results suggest that, for highly complex web scenes, optimizing data preparation and splats culling becomes as critical as optimizing the rasterization pipeline." (p7)

> "Furthermore, we find that Firefox remains unstable within the current WebGPU ecosystem." (p5)

> "Due to these issues, we omit Firefox results for both the Lenovo MX350 and Intel NUC systems." (p6)

### 10. Every sentence touching a memory / FPS / time TARGET, "budget", or edge/mobile constraint (verbatim)
- p1: "Furthermore, we propose an opacity-aware geometry culling stage that dynamically prunes splats before rasterization, significantly reducing overdraw and peak memory footprint."
- p1: "However, the web is heterogeneous, spanning a wide range of devices from high-end discrete GPUs to mobile SoCs."
- p2: "Furthermore, we demonstrate that WebSplatter reduces peak memory consumption compared to existing WebGPU ports, preventing crashes on memory-constrained devices."
- p4: "On heterogeneous devices such as laptops and mobile phones, limited bandwidth makes this step a major bottleneck."
- p6: "In contrast, on systems with less powerful GPUs, such as the Nvidia MX350 and the Intel NUC's integrated graphics, the bottleneck shifts to the pre-processing stage, consuming up to 47.2% of the frame time."
- p6: "In contrast, several baselines fail on certain devices due to excessive memory requirements that surpass hardware limitations."
- p8: "A critical factor for web-based deployment, particularly on mobile devices, is the consumption of video memory (VRAM). Excessive memory usage in existing WebGPU ports often results in browser context loss or crashes on lower-end hardware."
- p8: "This efficient memory management explains the stability results observed in Table 1, where WebSplatter successfully runs on memory-constrained mobile devices (e.g., iPhone 15 Pro Max, Oppo Find X2) while the unoptimized WebGPU baselines frequently crash due to out-of-memory errors."
- p8: "Many work aims to reduce the model size or the memory footprint of 3DGS. These methods reduce the number of Gaussians through pruning strategies based on importance metrics [6, 18] or by modifying the densification process during training [11]. While effective at reducing model size, these approaches can sometimes compromise rendering quality and do not address the computation bottlenecks of the rendering itself." (spans p8–p9)

### 11. Code URL and licence
Footnote 1, p1, as printed: `https://anonymous.4open.science/r/webgs` ("The source code is available
at ..."). Baseline repos, p11: `https://github.com/antimatter15/splat` (commit 367a943),
`https://github.com/mkkellogg/GaussianSplats3D` (2dfc83e),
`https://github.com/KeKsBoTer/web-splat` (959a3ec),
`https://github.com/MarcusAndreasSvensson/gaussian-splatting-webgpu` (f22ae04).
**Licence: not stated.**

---

## 3. Gaussian Blending Unit (GBU)

### 1. Citation
- Title: **Gaussian Blending Unit: An Edge GPU Plug-in for Real-Time Gaussian-Based Rendering in
  AR/VR**
- First author: Zhifan Ye (Georgia Institute of Technology). Senior author Yingyan (Celine) Lin.
- Venue as printed on p1: **2025 IEEE International Symposium on High-Performance Computer
  Architecture (HPCA)**.
- arXiv id from file name: **2503.23625** (page 1 stamp: `arXiv:2503.23625v1 [cs.GR] 30 Mar 2025`).

### 2. What it does (3 lines)
Profiles three Gaussian pipelines (3DGS, 4DGS, SplattingAvatar) on a Jetson Orin NX, finds Gaussian
Blending (per-fragment opacity + α-blending) is the bottleneck, and proposes an Intra-Row Sequential
Shading dataflow that shades a row left-to-right after two exact coordinate transforms, sharing
intermediates and skipping non-contributing fragments and rows. Then adds a hardware Gaussian
Blending Unit (row-centric tile engine with per-row PEs + a Gaussian Reuse Cache with precomputed
reuse distances) that plugs into an edge GPU, replacing one SM. **Renderer / hardware only. The
model is untouched, no retraining, no pruning.**

### 3. Device bottleneck
**Gaussian Blending (step 3: per-fragment opacity + α-blending) is the bottleneck on every workload.**
- Share of rendering time (p2, p5, reported): "accounting for 48% to 78% of the rendering time in
  these applications"; static scenes **70 %–78 %**, dynamic scenes **62 %–65 %**, human avatars
  **48 %–51 %**. Sorting is **14 %–24 %** across all three types. Preprocessing takes the rest and
  grows for dynamic/avatar workloads.
- Baseline speeds on Jetson Orin NX 16 GB: static scenes **7–17 FPS** (abstract) / 13 FPS average,
  dynamic 18 FPS, avatars 41 FPS, vs "the over 60 FPS standard required for truly immersive AR/VR"
  (p1).
- **What drives it: depth complexity, i.e. fragments, not Gaussian count.** "Unlike the other steps,
  where computational complexity is determined by the number of Gaussians, Rendering Step ❸ involves
  per-fragment computation. Our profiling shows that the average fragment-to-Gaussian ratio is
  **541:1, 161:1, and 688:1** across the three types of applications [5], [7], [32]" (p5). Reference
  order there is [5] human avatar, [7] static, [32] dynamic, i.e. the ratios are printed in
  reference order, not in the static/dynamic/avatar order used elsewhere.
- Per-fragment cost: Eq. 7 is **11 FLOPs per fragment**; "For real-world static scene rendering [7],
  Eq. 7 alone would require **1.1 TFLOPs to achieve 60 FPS, which is 58 % of Jetson Orin NX's peak
  floating-point throughput**" (p5).
- **Fragment redundancy**: "only **7.6 %, 13.7 %, and 9.9 %** of fragments make a non-negligible
  contribution (i.e., opacity greater than a predefined threshold) to the output colors on the three
  types of applications [5], [7], [32]" (p5).
- **SIMT utilisation**: after redundancy skipping, per-row workloads are imbalanced, giving "only
  **18.9 % utilization of GPU threads/lanes** on the real-world static scene dataset" (p7).
- **Memory bandwidth**: "Our profiling of real-world static scenes [7] shows that Rendering Step ❸
  alone requires **62.1 % of DRAM bandwidth** to achieve real-time (60 FPS) rendering performance.
  The experiments in Sec. VI-C indicate that this limitation could lead to a **13.5 % slowdown** in
  end-to-end rendering." (p7)
- **Resolution scales the bottleneck**: GBU speedup 3.7×–4.1× at 676 × 507, **9.5×–13.2× at
  2704 × 2028**, "because the number of fragments grows with the increase in rendering resolution"
  (p11).

### 4. Representation on device
**Uncompressed Gaussians rendered directly**; the only precision change is FP16 inside the Row PEs
("GBU hardware only minimally degrades rendering quality (< 0.1 PSNR and < 0.01 LPIPS), which is
mainly due to the use of FP-16 precision in the Row-Centric Tile PE", p10). No decode step.

On-chip storage (Table II, p9): **GBU SRAM 63 KB**, area 0.90 mm², 1 GHz, 28 nm, typical power
0.22 W; the baseline Orin NX is 4 MB SRAM, 450 mm², 918 MHz, 8 nm, 15 W. Area/power breakdown
(Table III, p9): Row PEs 0.36 mm² / 0.11 W, Row Gen. 0.14 / 0.04, D&B Engine 0.10 / 0.03, Cache &
Others 0.30 / 0.04.

**Gaussian Reuse Cache sized to 32 KB**, because hit rate saturates there: "doubling the cache size results
in a linear increase in hit rate when the cache size is below 8 KB. However, the hit rate saturates
around 32 KB on all three datasets, with minimal further gains ... (i.e., less than 0.1 %
improvement)". At 64 KB the average hit rates are 59.7 % / 47.4 % / 37.7 % across the three datasets
(Fig. 17, p11). Reduces off-chip accesses by **44.9 %**, giving **1.14×** speedup.

Model footprint in bytes and render-buffer footprint: **not stated.**

### 5. Budget input
**No target is passed to anything.** 60 FPS is used as a fixed pass/fail line for AR/VR ("real-time
rendering (i.e., ≥60 FPS [54])"), and once as a matched operating point for a hardware comparison:
> "under the same target rendering speed in both the Tanks&Temples dataset [23] and the Deep
> Blending dataset [16] used by GS-Core, GBU-Standalone demonstrates superior area and energy
> efficiency" (p11).

That is the only "target" in the paper, and it is a comparison protocol, not an input to a training
or pruning algorithm. **No achieved-vs-requested pairs.**

### 6. Device numbers (all reported)
Device: **NVIDIA Jetson Orin NX 16 GB** edge GPU, profiled with Nsight Systems; GBU simulated with a
cycle-accurate emulator on GPGPU-Sim, "The emulated runtime and power consumption are within 10 %
error of the real-device measurement" (p10). GBU RTL synthesised with Cadence Genus in commercial
28 nm CMOS at 1 GHz. GBU replaces one SM on the Orin NX and reuses the SM-to-DRAM network.

End-to-end FPS (p10): Orin NX alone **13 FPS static / 18 FPS dynamic / 41 FPS avatars**; Orin NX +
GBU **92 / 80 / 102 FPS**. Energy efficiency improvement **10.8× / 4.4× / 2.5×**. Energy for
rendering 60 images falls from **76 J, 52 J, 23 J to 7 J, 12 J, 9 J** respectively.

Ablation, static scenes (Table V, p10):

| Configuration | FPS | Energy efficiency | PSNR | LPIPS |
|---|---|---|---|---|
| Jetson Orin NX | 12.8 | 1× | 28.90 | 0.196 |
| + IRSS Dataflow | 22.0 | 1.71× | 28.90 | 0.196 |
| + GBU Tile Engine | 66.1 | 7.22× | 28.84 | 0.197 |
| + GBU D&B Engine | 80.6 | 9.40× | 28.84 | 0.197 |
| + GBU Reuse Cache | 91.5 | 10.8× | 28.84 | 0.197 |

IRSS deployed as a plain CUDA kernel gives "a significant 59 % reduction in latency during Rendering
Step ❸, increasing the rendering speed from 13 FPS to 22 FPS" (p7), and "a non-trivial 1.72× speedup
on real-world static scenes" (p1). Compute sharing takes Eq. 7 from **11 FLOPs → 3 FLOPs → 2 FLOPs**
per fragment (transform P→P′ then P′→P″), a "5.5×" reduction.

Standalone accelerator comparison (Table VI, p11): GS-Core 272 KB SRAM / 3.95 mm² / 0.87 W, step-3
PE 1.81 mm² / 0.25 W. GBU-Standalone 63 KB / 1.78 mm² / 0.78 W, step-3 PE 0.50 mm² / 0.15 W.

Versus NeRF accelerators on NeRF-Synthetic (Table VII, p11): ICARUS PSNR 30.21, 40 nm, 0.3 GHz,
0.3 W, 0.03 FPS; RT-NeRF 31.79, 28 nm, 1.0 GHz, 18.85 mm², 8 W, 45 FPS; Instant-3D 33.18, 28 nm,
0.8 GHz, 6.8 mm², 1.9 W, > 30 FPS; **GBU-Standalone 33.26, 28 nm, 1.0 GHz, 1.78 mm², 0.78 W,
172 FPS**.

Resolutions profiled (Table I, p4): static 779 × 519 to 1245 × 825; dynamic 1352 × 1014; avatar
1080 × 1080.

Extreme case (p11): "increasing the camera-to-scene distance by 4× in the static scene dataset [7]
reduces GBU's speedup over a vanilla GPU [2] from the original 10.8× to 4.7×."

### 7. Quality numbers on Mip-NeRF 360 / T&T / Deep Blending
Table IV (p10), reported. Note the "static scenes" set is the 6 Mip-NeRF 360 scenes (bicycle,
bonsai, counter, kitchen, room, stump) as listed in Table I:

| | Static scenes PSNR / LPIPS | Dynamic scenes PSNR / LPIPS | Human avatar PSNR / LPIPS |
|---|---|---|---|
| 3D-GS (baseline 1) | 28.90 / 0.196 | 33.80 / 0.976 | 32.19 / 0.022 |
| GBU | 28.84 / 0.197 | 33.71 / 0.977 | 32.17 / 0.022 |

(The dynamic-scene LPIPS values 0.976 / 0.977 look implausible for LPIPS but are copied exactly as
printed.) **Second strong baseline for quality: not stated**. The paper's quality table has only
3D-GS and GBU. Tanks and Temples and Deep Blending are named only as the datasets GS-Core used for
the matched-speed area/power comparison (p11); no PSNR is given for them. **No count or model-size
column anywhere.**

### 8. Paper-specific (n/a)

### 9. Stated limitations and future work (verbatim, pp11–12)
> "Limitations in extreme cases. While GBU demonstrates strong performance across three widely used
> datasets [5], [7], [32], it may face challenges under certain extreme conditions: (1) Distant
> camera poses. The efficiency of the IRSS dataflow relies on each Gaussian covering multiple pixels
> per row. However, when the camera is significantly farther from the scene, Gaussians may cover
> fewer pixels, reducing compute sharing. For instance, increasing the camera-to-scene distance by
> 4× in the static scene dataset [7] reduces GBU's speedup over a vanilla GPU [2] from the original
> 10.8× to 4.7×. Future work could address this by adaptively merging Gaussians based on camera
> distance [21]; (2) Highly dynamic scenes. GBU primarily accelerates the Rendering Step ❸, but in
> highly dynamic scenes, other rendering steps may dominate computation. For example, multi-avatar
> settings [34] may require substantial processing in the Rendering Step ❶ for modeling the human
> bodies, limiting GBU's overall speedup. A specialized accelerator for the Rendering Step ❶ could
> improve efficiency in such scenarios." (pp11–12)

### 10. Every sentence touching a memory / FPS / time TARGET, "budget", or edge/mobile constraint (verbatim)
- p1: "The rapidly advancing field of Augmented and Virtual Reality (AR/VR) demands real-time, photorealistic rendering on resource-constrained platforms."
- p1: "However, despite its effectiveness on high-end GPUs, it struggles on edge systems like the Jetson Orin NX Edge GPU, achieving only 7-17 FPS—well below the over 60 FPS standard required for truly immersive AR/VR experiences."
- p1: "Additionally, GBU integrates a Gaussian Reuse Cache, reducing off-chip memory accesses by 44.9% and resulting in a 1.14× speedup in rendering."
- p1: "Despite the potential of 3D Gaussians for real-time rendering on server and desktop devices, a significant performance gap remains for real-time rendering (i.e., ≥60 FPS [54]) on edge devices. For example, rendering real-world scenes from the MipNeRF-360 dataset [7] on the Jetson Orin NX [2], an edge GPU from NVIDIA, achieves only 7 to 17 FPS."
- p2: "Through this, we identified the Gaussian Blending stage as the common bottleneck that prohibits real-time rendering on resource-constrained AR/VR devices."
- p2: "To meet the real-time framerate requirements (i.e. over 60 FPS) of AR/VR applications [54], we identified the imbalanced workload between pixel rows as a primary performance bottleneck in GPUs, resulting in only 18.9% GPU utilization on real-world static scenes."
- p2: "The experiment results show that the GBU provides a comprehensive solution for real-time rendering, achieving speeds greater than 60 FPS on edge devices across a broad range of AR/VR applications, while consistently maintaining SOTA rendering quality."
- p4 (Fig. 4 caption): "The red line represents the maximum rendering time required to achieve real-time rendering (60 FPS)."
- p5: "We observe that (1) on the edge GPU, none of the three types of scenes achieves real-time rendering performance (≥ 60 FPS [54])."
- p5: "For real-world static scene rendering [7], Eq. 7 alone would require 1.1 TFLOPs to achieve 60 FPS, which is 58% of Jetson Orin NX's peak floating-point throughput [2]."
- p7: "Limitation 2: High Memory Footprint. The memory footprint for reading Gaussian features (Fig. 9(a)) in the Rendering Step ❸ can negatively impact the throughput of the first two rendering steps when the three steps are pipelined. Our profiling of real-world static scenes [7] shows that Rendering Step ❸ alone requires 62.1% of DRAM bandwidth to achieve real-time (60 FPS) rendering performance."
- p11: "As shown in Tab. VI, under the same target rendering speed in both the Tanks&Temples dataset [23] and the Deep Blending dataset [16] used by GS-Core, GBU-Standalone demonstrates superior area and energy efficiency, primarily due to the proposed Tile Engine."
- p11: "As a result, the dedicated acceleration for Rendering Step ❸ plays a more vital role in enabling real-time rendering with 3D Gaussians on a higher resolution, making the proposed GBU even more desirable for future-generation AR/VR devices at higher screen resolutions."
- p12: "Achieving real-time rendering speeds on edge devices remains a significant challenge due to the substantial computational demands associated with SOTA Gaussian-based rendering pipelines."

### 11. Code URL and licence
**No code URL printed anywhere in the paper. Licence: not stated.**

---

## 4. LS-Gaussian

### 1. Citation
- Title: **No Redundancy, No Stall: Lightweight Streaming 3D Gaussian Splatting for Real-time
  Rendering**
- First author: Linye Wei (Institute for Artificial Intelligence + School of Integrated Circuits,
  Peking University). Corresponding author Meng Li.
- Venue: **not stated on the PDF** (no conference header). The file name records ICCAD 2025; the PDF
  itself only carries the arXiv stamp.
- arXiv id from file name: **2507.21572** (page 1 stamp: `arXiv:2507.21572v2 [cs.AR] 30 Jul 2025`).

### 2. What it does (3 lines)
Exploits inter-frame similarity: fully renders one frame in every n (default n = 5, i.e. 1 in 6 in
the Synthetic-NeRF comparison), reprojects the rest, and re-renders only whole tiles whose missing
pixel count exceeds 1/6 of the tile (tile warping rather than pixel warping), with a mask that
blocks accumulated interpolation error. Adds a two-stage accurate Gaussian-tile intersection test
and a depth-based early-stopping predictor that estimates per-tile workload. Co-designs an
accelerator on top of GSCore with a Load Distribution Unit. **Training-free: "our training-free
approach enables seamless integration with existing methods" (p2). Renderer + hardware only.**

### 3. Device bottleneck
Two named causes, redundancy and stalls (Fig. 3, p3):
- **Inter-frame redundancy**: large fraction of pixels overlap between consecutive frames (Fig. 4a,
  proportions not extractable as text).
- **Intra-frame redundancy: false-positive Gaussian-tile pairs from the AABB test** (Fig. 4b for
  drjohnson). "this coarse bounding box approximation associates each tile with numerous Gaussians
  that do not actually contribute to rendering, significantly increasing the overhead in both the
  sorting and rasterization stages" (p3). Three named sources: 3√λ ignores opacity; the
  circumscribed square over-covers the minor axis; ellipse-vs-rectangle mismatch.
- **Inter-block stall**: "variations in scene complexity cause the per-tile Gaussian count to vary
  by **more than an order of magnitude**, leading to severe load imbalance" (p3, Fig. 5, "train"
  scene).
- **Intra-block bubbles**: rasterisation waiting on sorting when stages are decoupled into dedicated
  units, as in GSCore.

Stage percentages of frame time are **not stated numerically**; the paper gives the pipeline
diagram, not a percentage breakdown. Rasterisation is called "the dominant factor in overall
rendering time" (p4).

Hardware-utilisation numbers, rasterisation core utilisation in % (Table I, p8, reported):

| Method | Synthetic | T&T | DB | Mip | Average |
|---|---|---|---|---|---|
| Original | 45.6 | 43.1 | 49.5 | 67.9 | 51.5 |
| LS-Gaussian | 88.5 | 89.8 | 78.0 | 98.2 | 88.6 |

Model properties that drive cost: **Gaussian-tile pairs** (the two-stage intersection test targets
these directly, ~2× speedup on all scenes), **per-tile depth complexity up to the early-stopping
point** (the DPES predictor), and **inter-frame camera motion** (the sparse-rendering gain is
scene-type dependent: 1.56–2.35× outdoor, 2.41–3.55× indoor).

### 4. Representation on device
**Uncompressed Gaussians, rendered directly, no retraining and no quantisation.** LS-Gaussian
explicitly positions itself against quantisation/pruning/LOD lines: "these methods inevitably
degrade reconstruction quality and rely on retraining" (p2). Frame-to-frame it stores a reference
frame's colour, depth and truncated-depth maps for reprojection, and a 16 KB counter buffer on chip.

Model footprint in bytes and render-buffer footprint: **not stated.** Accelerator area is given
instead: total design **1.84 mm²** in 16 nm FinFET, "just a 0.39 mm² increase over the scaled GSCore
design in 16nm (1.45 mm²), and remaining significantly smaller than Jetson-series edge GPUs
(~350 mm²) and MetaSapiens (2.73 mm²)" (p7). Hardware reuse of the VTU counter buffer and
comparators saves 32 % of the extra area; adding GSU reuse takes it to 36 %.

### 5. Budget input
**No byte, FPS or ms target is an input.** The one tunable that behaves like a budget knob is the
**warping window size n** (frames between two fully rendered frames), swept in Fig. 12a: "increasing
the window size leads to greater speedups but also results in image quality degradation. To balance
rendering quality and efficiency, we set **n = 5** as the default configuration". The Load
Distribution Unit does compute an internal per-block workload budget: "We first compute the ideal
average workload W and assign approximately N tiles per block. Tiles are then assigned to blocks
sequentially. If the cumulative number of Gaussian-tile pairs in the current block exceeds
(1 + 1/N)W, the current tile is deferred to the next block." (p6). That is a load-balancing
constraint, not a user-supplied deployment budget.

The paper does state a **requirement** that the setup is built to hit: "we perform interpolation to
construct continuous view sequences suitable for real time rendering at **90 FPS**. This setup
simulates camera motion at 1.8 meters and a rotational speed of 90 degrees per second." (pp6–7) and
"achieves an average speedup of 5.41× over the GPU baseline, **satisfying the 90 FPS real-time
rendering requirement on the Deep Blending dataset**" (p7). **Achieved-vs-requested pair (the only
one in the paper): requested 90 FPS, achieved on Deep Blending only.**

### 6. Device numbers (all reported)
Device: **NVIDIA Jetson AGX Orin** GPU baseline. Accelerator: RTL synthesised with Synopsys in
**16 nm FinFET**, area 1.84 mm².

- Fig. 1 (p1), Jetson AGX Orin: "Original 3DGS PSNR: 33.76 Time: 49.06ms/frame" vs "LS-Gaussian
  PSNR: 33.24 Time: 1.04ms/frame", with the caption "LS-Gaussian is accelerated by sparse rendering
  between consecutive frames and only needs to fully render one in every 6 frames."
- GPU speedups (p7): **average 5.41×** over the Jetson AGX Orin baseline; **1.85×** over
  AdR-Gaussian and **1.75×** over SeeLe.
- Accelerator speedups (p8): **17.3×** average, vs GSCore **9.1×** and MetaSapiens **14.5×**, all
  normalised to the same 1.45 mm² area using MetaSapiens' Speedup-Area Curve.
- Per-technique ablation on GPU (p8): TWSR 1.56–2.35× on the three outdoor scenes (train, truck,
  garden), 2.41–3.55× on indoor scenes; TAIT ≈ 2× on all scenes; DPES "offers a modest speedup".
- Fig. 12b, Jetson AGX Orin, two representative real-world frames: "PSNR: 31.40 FPS: 51 (5.7×)" and
  "PSNR: 35.74 FPS: 60 (7.1×)".
- Fig. 11b: "Original 3DGS PSNR: 36.65 SSIM: 0.981" vs "LS-Gaussian PSNR: 35.01 SSIM: 0.975".

### 7. Quality numbers on Mip-NeRF 360 / T&T / Deep Blending
**Absolute PSNR/SSIM/LPIPS tables for these datasets are not stated.** LS-Gaussian is training-free,
so it reports *deltas* against whatever 3DGS model it accelerates. Reported deltas:
- Synthetic-NeRF, one full frame in six: "Our method results in an average SSIM loss of only
  **0.005** and a **PSNR loss of 1.4 dB** across six scenes, which is significantly lower than
  Potamoi's SSIM loss of **0.063** and PSNR loss of **6.8 dB**." (p7)
- Real-world scenes used (three indoor: playroom, drjohnson, room; three outdoor: train, truck,
  garden) from Tanks and Temples, Deep Blending and Mip-NeRF 360, with quality measured "by
  measuring the difference between rendered outputs w/ and w/o the proposed viewpoint
  transformation" (p7).
- **Counts and model sizes: not stated anywhere.**
Strongest baselines named: Potamoi (quality), AdR-Gaussian and SeeLe (GPU speed), GSCore and
MetaSapiens (accelerators).

### 8. Paper-specific (n/a)

### 9. Stated limitations and future work
**No limitations section and no future-work section exist in this paper.** The nearest thing to a
stated constraint is the scene-type dependence:
> "We observe that indoor scenes, which typically exhibit smaller depth variations, achieve both better rendering quality and higher speedups compared to outdoor scenes with more edges and corners." (p7)

> "Unfortunately, as the number of consecutive viewpoint transformations increases, we observe a noticeable degradation in image quality. This is primarily due to the accumulation of interpolation errors across frames. The more transformation rounds applied, the more severe the quality deterioration becomes." (p4)

### 10. Every sentence touching a memory / FPS / time TARGET, "budget", or edge/mobile constraint (verbatim)
- p1: "However, 3DGS still faces major efficiency challenges when faced with high frame rate requirements and resource-constrained edge deployment."
- p1: "Experimental results demonstrate that LS-Gaussian achieves 5.41× speedup over the edge GPU baseline on average and up to 17.3× speedup with the customized accelerator, while incurring only minimal visual quality degradation."
- p1: "On resource-constrained edge platforms, such as the Jetson AGX Orin, which are critical for many real-world tasks, 3DGS still struggles to achieve real-time rendering at 90 FPS [20]."
- p1: "However, as scene scales expand in practical applications [18], [19], 3DGS must process millions or more Gaussians, incurring substantial memory and computational overhead."
- p2: "Extensive experiments across multiple datasets demonstrate that LS-Gaussian achieves an average 5.41× speedup on Jetson AGX Orin GPU. With dedicated hardware support, LS-Gaussian further achieves a 17.3× speedup while maintaining high quality."
- p2: "Pixel and Gaussian reuse [32], [33] exploit the similarity between adjacent frames to alleviate computational and memory constraints, but fail to account for the tile-based rendering characteristics of 3DGS, resulting in limited speedup."
- p3: "On edge devices, additional processing waves due to limited compute resources help mitigate this imbalance by allowing underutilized blocks to render more tiles."
- pp6–7: "Since these real world datasets typically contain sparse camera trajectories, we perform interpolation to construct continuous view sequences suitable for real time rendering at 90 FPS. This setup simulates camera motion at 1.8 meters and a rotational speed of 90 degrees per second."
- p7: "Experimental results demonstrate that LS-Gaussian achieves an average speedup of 5.41× over the GPU baseline, satisfying the 90 FPS real-time rendering requirement on the Deep Blending dataset."
- p7: "Benefiting from the reuse of existing hardware modules by the Load Distribution Unit (LDU), our total design area is only 1.84 mm2, representing just a 0.39 mm2 increase over the scaled GSCore design in 16nm (1.45mm2), and remaining significantly smaller than Jetson-series edge GPUs (˜350 mm2) and MetaSapiens (2.73 mm2)."
- p8: "In this paper, we present LS-Gaussian, a lightweight and streaming framework designed to accelerate 3D Gaussian Splatting on resource-constrained platforms."

### 11. Code URL and licence
**No code URL printed. Licence: not stated.**

---

## 5. Pocket-SLAM

### 1. Citation
- Title: **Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM**
- First author: Leshu Li (University of Minnesota, Twin Cities). Co-authors Jie Peng (UNC Chapel
  Hill), Yang Zhao (UMN, corresponding).
- Venue: **not stated on the PDF**. The file name records ICRA 2026; the PDF carries only the arXiv
  stamp.
- arXiv id from file name: **2606.24796** (page 1 stamp: `arXiv:2606.24796v1 [cs.CV] 23 Jun 2026`).

### 2. What it does (3 lines)
Prunes Gaussians during 3DGS-SLAM mapping by their **effective pixel coverage** C_i = ∑[p ∈ Ω]
α_i(p), normalised to S_i, instead of by opacity or gradient. Adds a **tile-level budget** B_k
allocated in proportion to the tile's mean tracking gradient and clipped to [B_min, B_max], so
texture-dense tiles keep small Gaussians. **Changes the model (runtime pruning inside the SLAM
mapping loop). The renderer is unchanged.**

### 3. Device bottleneck
**Memory capacity, not frame time.** The paper targets "peak runtime memory consumption" of
3DGS-SLAM in large outdoor scenes, defined as "all intermediate parameters involved in 3DGS-SLAM
computation, reflecting the minimum memory required for a device to execute 3DGS-SLAM" (p4). No
stage-level (preprocess / sort / blend) breakdown is given, and no per-stage timing exists anywhere
in the paper. FPS is reported as a whole-system metric only.

The model property that drives it is the **Gaussian count accumulating over time**: "The memory
consumption increases continuously over time as Gaussian points accumulate" (p1), and "the change in
the number of Gaussians at each keyframe directly determines the peak memory consumption" (p7).
Memory-usage-vs-frame curve, KITTI seq. 3 (Fig. 5, p7): peaks 32.1 GB LSG-SLAM, 26.3 GB
MaskGaussian, **10.08 GB Pocket-SLAM**, over ~2500 frames.

### 4. Representation on device
**Uncompressed Gaussians. No quantisation, no codebook, no decode step.** The only reduction is
count. The relevant footprint number is peak runtime memory (Tables II and IV, below), which
includes optimiser and intermediate state, not just the model. Render-buffer footprint is **not
broken out**.

### 5. Budget input
**Yes, a count budget, in two forms, but no bytes, no FPS, no ms.**
- **Global target count**: "Given a global target Ntar, we allocate per-tile budgets as: B_k^trk =
  clip(⌈Ntar · G_k / ∑_j G_j⌉, B_min, B_max)" (p4). Implementation: "the global target Gaussian
  count Ntar is set to **0.4·Ninit**, where Ninit denotes the number of Gaussians before pruning.
  The minimum per-tile budget B_min is fixed at **5** ... while the maximum budget B_max is set to
  **200**" (p5).
- All baselines are compared "under the same pruning ratio" (p5). The ratio is the shared input.

**Achieved-versus-requested pairs.** The requested quantity is a *count ratio* (0.4·Ninit, i.e. a
60 % prune). The achieved quantities are memory reductions, which are what the paper reports:
- Requested: same pruning ratio for all methods. Achieved on EuRoC: Pocket-SLAM peak memory
  **10.1 GB** average vs LSG-SLAM 25.4 GB, i.e. "**reduces peak memory consumption by 61.3 %** on
  average across the five sequences and achieves a **2.7× improvement in FPS**"; MaskGaussian at the
  same ratio achieves "only about **30 % savings**" (17.6 GB) because of deferred deletion (p5).
- Requested: same pruning ratio. Achieved on KITTI: "Pocket-SLAM reduces peak memory consumption by
  **65.7 %** while achieving a **2.9× speedup in FPS**" (p6); 11.9 GB vs 34.7 GB average.
- Fig. 1 (p1), KITTI seq. 10: "PSNR: 26.82 dB Peak memory Usage: 34.2 GB" (LGS-SLAM) vs
  "PSNR: 26.55 dB Peak memory Usage: 13.3 GB" (ours), "reducing peak memory usage by **61 %**".
- Prune-ratio sweep without the tile budget (Fig. 3a, p4): the x-axis runs over pruning ratios and
  both ATE and PSNR degrade as it rises; numeric axis values are not extractable as text. Fig. 3b,
  ratio 0.6: "PSNR 25.8" with tile budget vs "PSNR 18.2" without.

### 6. Device numbers
**The GPU used is not an edge device.** "all methods are compared under the same pruning ratio on
the **NVIDIA A6000** [36] platform" (p5). Edge hardware appears only as motivation and future work.
All numbers below are reported on the A6000.

EuRoC performance (Table II, p5), Peak Memory in GB / FPS:

| Method | MH01 | MH02 | MH03 | MH04 | MH05 | Avg |
|---|---|---|---|---|---|---|
| LSG-SLAM memory | 24.2 | 21.4 | 25.6 | 30.2 | 25.6 | 25.4 |
| LSG-SLAM FPS | 1.2 | 1.5 | 1.2 | 1.1 | 1.3 | 1.3 |
| MaskGaussian memory | 16.3 | 13.8 | 17.9 | 22.2 | 17.8 | 17.6 |
| MaskGaussian FPS | 1.5 | 1.9 | 1.6 | 1.5 | 1.6 | 1.6 |
| **Ours memory** | 9.6 | 8.4 | 10.2 | 12.1 | 10.2 | **10.1** |
| **Ours FPS** | 3.3 | 4.2 | 3.5 | 3.3 | 3.6 | **3.6** |

KITTI performance (Table IV, p6), Peak Memory in GB / FPS, sequences 00–10 (01 omitted):

| Method | 00 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LSG-SLAM memory | 40.2 | 36.6 | 32.1 | 37.1 | 30.6 | 31.2 | 38.5 | 36.2 | 30.4 | 34.2 | 34.7 |
| LSG-SLAM FPS | 0.7 | 0.8 | 0.9 | 0.7 | 1.1 | 0.9 | 0.7 | 0.8 | 1.0 | 0.8 | 0.8 |
| MaskGaussian memory | 35.2 | 31.6 | 26.3 | 32.3 | 24.8 | 25.2 | 32.4 | 30.5 | 24.3 | 29.6 | 29.2 |
| MaskGaussian FPS | 0.9 | 1.0 | 1.1 | 0.9 | 1.4 | 1.1 | 0.9 | 1.0 | 1.3 | 1.1 | 1.1 |
| **Ours memory** | 13.9 | 12.7 | 10.8 | 12.9 | 10.2 | 10.5 | 14.0 | 12.4 | 10.1 | 13.3 | **11.9** |
| **Ours FPS** | 2.0 | 2.2 | 2.5 | 2.0 | 3.1 | 2.4 | 1.8 | 2.3 | 2.9 | 2.3 | **2.4** |

All reported. **These are SLAM systems: 0.8–3.6 FPS, an order of magnitude below interactive
rendering, because tracking and mapping optimisation dominate.**

### 7. Quality numbers on Mip-NeRF 360 / T&T / Deep Blending
**Not applicable and not stated.** The datasets are EuRoC MAV (MH01–MH05) and KITTI odometry
(00–10 except 01). Reported averages, tracking ATE RMSE [m] / PSNR / SSIM / LPIPS:

EuRoC (Table I, p5), averages: LSG-SLAM 0.06 / 30.81 / 0.98 / 0.05; LightGaussian 1.31 / 12.34 /
0.53 / 0.41; LP-3DGS 0.67 / 14.64 / 0.63 / 0.35; MaskGaussian 1.08 / 19.4 / 0.72 / 0.31; Ours w/o
Tile Budget 0.38 / 23.23 / 0.82 / 0.23; **Ours w/ Tile Budget 0.07 / 30.53 / 0.98 / 0.05**.

KITTI (Table III, p6), averages: LSG-SLAM 3.85 / 26.58 / 0.97 / 0.07; LightGaussian 18.88 / 15.83 /
0.79 / 0.32; LP-3DGS 9.89 / 18.36 / 0.83 / 0.25; MaskGaussian 7.51 / 20.56 / 0.91 / 0.19; Ours w/o
Tile Budget 5.37 / 23.55 / 0.94 / 0.13; **Ours w/ Tile Budget 3.81 / 26.46 / 0.97 / 0.07**.

Strongest two baselines: **LSG-SLAM** (the unpruned upper bound) and **MaskGaussian** (the only
pruning baseline that completes all sequences). LightGaussian and LP-3DGS lose tracking on several
sequences ("-" = tracking lost).

**Counts / model sizes: not stated**, only the prune ratio 0.4·Ninit and peak memory in GB.

### 8. Paper-specific (n/a)

### 9. Stated limitations and future work (verbatim, p7)
> "Future directions include designing adaptive pruning schedules that dynamically adjust pruning
> ratios based on scene complexity and temporal variations; exploring long-term, dynamic outdoor
> scenarios such as autonomous driving at night, across seasonal changes, or under severe weather
> conditions; and deploying Pocket-SLAM on resource-constrained hardware platforms such as embedded
> GPUs [37] and edge accelerators [38]–[40] to enable practical and reliable robotic applications." (p7)

Also, stated as a limitation of the method's own first half:
> "While rendering-area–aware pruning reduces redundant Gaussians by evaluating their pixel coverage, it inevitably removes small Gaussians in texture-dense regions, leading to local information loss and degraded mapping quality." (p7)

> "However, applying rendering-area–aware pruning alone, while effective in reducing memory, inevitably results in severe information loss in texture-dense regions." (p2)

### 10. Every sentence touching a memory / FPS / time TARGET, "budget", or edge/mobile constraint (verbatim)
- p1: "The memory consumption increases continuously over time as Gaussian points accumulate, leading to poor memory efficiency and limiting its applicability."
- p1: "This perspective directly targets the sources of memory redundancy, effectively reducing the peak memory footprint of 3DGS-SLAM during runtime."
- p1: "Evaluations on the EuRoC and KITTI datasets demonstrate that our method consistently outperforms existing pruning approaches in large-scale outdoor scenes, achieving over 60% memory reduction and more than 2× FPS improvement while preserving localization and mapping accuracy."
- p1: "Nevertheless, a critical limitation persists: large-scale deployments introduce substantial memory redundancy."
- p1: "While the number of Gaussians in compact indoor scenes is relatively modest, vast and unstructured outdoor environments necessitate storing and updating millions of Gaussians in real time, resulting in prohibitively high peak memory consumption that becomes a major bottleneck for deployment."
- p1: "This challenge is further exacerbated by the increasing demand for 3DGS-SLAM to operate efficiently on resource-constrained edge devices (e.g., GPUs embedded in autonomous vehicles or drones). In this context, reducing peak runtime memory consumption emerges as a central requirement for practical deployment."
- p1: "This limitation is particularly important, as peak memory directly determines the feasibility of deploying 3DGS-SLAM on edge devices with limited memory capacity, and therefore has far greater practical implications for real-world large-scale applications."
- p2: "To mitigate this issue, we introduce a tile-level budget mechanism, which adaptively constrains the pruning ratio within each tile to prevent over-pruning in both texture-dense and texture-scarce areas, thereby maintaining balanced Gaussian distributions and preserving texture information."
- p2: "We introduce a tile-level budget mechanism that adaptively constrains pruning according to per-tile Gaussian allocation, preventing excessive pruning in both texture-dense and texture-sparse regions, and ensuring balanced Gaussian distributions with robust texture preservation."
- p2: "However, outdoor environments require millions of Gaussians to represent unstructured regions, resulting in prohibitively high peak memory consumption and limiting deployment on resource-constrained edge devices such as those in autonomous vehicles and drones."
- p2: "GEVO [13] is among the few works addressing memory in 3DGS-SLAM, but it focuses only on storage via reduced keyframe retention rather than runtime peak memory consumption."
- p4: "Given a global target Ntar, we allocate per-tile budgets as: Btrk_k = clip(⌈Ntar · Gk / ∑_j Gj⌉, Bmin, Bmax), where Bmin and Bmax ensure that no tile is completely depleted or overly concentrated."
- p4: "Therefore, introducing a budget mechanism to prevent regions from being entirely depleted is essential for ensuring reliable SLAM performance."
- p4: "Notably, peak memory consumption accounts for all intermediate parameters involved in 3DGS-SLAM computation, reflecting the minimum memory required for a device to execute 3DGS-SLAM."
- p5: "For pruning, the global target Gaussian count Ntar is set to 0.4Ninit, where Ninit denotes the number of Gaussians before pruning. The minimum per-tile budget Bmin is fixed at 5 to prevent regions from being completely emptied, while the maximum budget Bmax is set to 200 to avoid excessive concentration of Gaussians within a single tile."
- p5: "As shown in Tab. II, our method reduces peak memory consumption by 61.3% on average across the five sequences and achieves a 2.7× improvement in FPS."
- p5: "Moreover, by substantially lowering peak memory consumption, our method significantly improves runtime efficiency, achieving a 2.2× speedup over MaskGaussian."
- p6: "As shown in the table, Pocket-SLAM reduces peak memory consumption by 65.7% while achieving a 2.9× speedup in FPS."
- p7: "We observe that memory consumption fluctuates at each keyframe, since mapping operations periodically introduce new Gaussians. Consequently, the change in the number of Gaussians at each keyframe directly determines the peak memory consumption."
- p7: "By combining these two complementary strategies, Pocket-SLAM achieves over 60% memory reduction and more than 2× FPS improvement on the EuRoC [29] and KITTI [14] datasets, while maintaining robust tracking and high-fidelity reconstruction."
- p7 (future work quote in field 9 also belongs here): "and deploying Pocket-SLAM on resource-constrained hardware platforms such as embedded GPUs [37] and edge accelerators [38]–[40] to enable practical and reliable robotic applications."

### 11. Code URL and licence
As printed on p1: `https://github.com/UMN-ZhaoLab/Pocket-SLAM.git` (the PDF line-breaks it as
"https://github.com/UMN-ZhaoLab/Pocket-SLAM." / "git."). **Licence: not stated.**

---

## 6. Octree-GS

### 1. Citation
- Title: **Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians**
- First author: Kerui Ren (Shanghai Jiao Tong University and Shanghai AI Laboratory), equal
  contribution with Lihan Jiang. Corresponding author Bo Dai.
- Venue as printed in the page header: **IEEE Transactions on Pattern Analysis and Machine
  Intelligence** (TPAMI). No year is printed in the header.
- arXiv id from file name: **2403.17898** (page 1 stamp: `arXiv:2403.17898v2 [cs.CV] 17 Oct 2024`).

### 2. What it does (3 lines)
Organises Scaffold-GS-style anchors into a sparse octree with K levels and selects, per view, the
cumulative set of anchors whose level is at or below a distance-derived LOD, so the rendered
primitive count stays roughly constant as the camera zooms out. Trains with a next-level grow
operator, view-frequency pruning and coarse-to-fine progressive activation of levels.
**Changes the model (training, densification, pruning, storage layout) and the per-view selection
rule. It is not a renderer change: the rasteriser is the stock tile-based one.**

### 3. Device bottleneck
**No device-level bottleneck analysis exists.** There is no preprocess/sort/blend breakdown, no
bandwidth number, no edge device. All experiments run on "a single **NVIDIA A100 80G GPU**" (p9),
and "To avoid the impact of image storage on GPU memory, all images were stored on the CPU."

The cost model the paper does assert is **rendered primitive count per view**: "they still rely on
visibility-based filtering for primitive selection, considering all primitives within the view
frustum without accounting for their projected sizes. As a result, every object detail is rendered,
regardless of distance, leading to redundant computations and inconsistent rendering speeds" (p1).
FPS is reported as a function of camera **distance** along a trajectory (Fig. 9a, p10).

### 4. Representation on device
**Neural anchors decoded to Gaussians every frame.** Following Scaffold-GS, each selected anchor
emits k neural Gaussians whose opacity, scale, rotation and colour are decoded by MLPs from the
anchor feature, the relative viewing distance and the view direction (Eq. 3, Eq. 4, p4). So the
compact representation is **not** rendered directly: it is expanded to explicit Gaussians per view,
per frame, before rasterisation. The Our-2D-GS / Our-3D-GS variants store explicit Gaussians per
anchor instead and skip the decode.

Model footprint: reported per dataset as "#GS(k)/Mem" in Tables I, II, IV, e.g. Our-Scaffold-GS
139.6 MB on Mip-NeRF 360, 88.5 MB on Tanks and Temples, 71.7 MB on Deep Blending. **Render-buffer
footprint: not stated.** No device-memory measurement of any kind.

### 5. Budget input
**No.** The LOD level is a function of geometry (distance, focal scale, learned bias), never of a
requested frame time or byte budget. There is no knob that accepts "render at ≥ X FPS" or "fit in Y
MB". Compared methods do have one: CityGaussian "selects LOD levels based on distance intervals ...
lacks robustness due to the need for manual distance threshold adjustments" and Hierarchical-GS is
evaluated at granularities τ₁ = 3, τ₂ = 6, τ₃ = 15 pixels (p7), which is a **pixel-footprint
threshold**, the closest thing in the paper to a tunable budget, and it belongs to a baseline.

**Achieved-versus-requested pairs: none.** The claim is a *floor* that is checked post hoc:
"Octree-GS consistently achieves real-time rendering performance (≥30 FPS)" (p2) and "Our method
shows consistent rendering speed above 30 FPS at 2k image resolution while all baseline methods fail
to meet the real-time performance" (Table V caption, p10).

### 6. Device numbers
**No edge device, no phone, no browser, no accelerator.** Single **NVIDIA A100 80 GB** (p9). All
numbers reported.

Rendering speed on MatrixCity Block_All, average FPS over three trajectories at 2k resolution
(Table V, p10):

| Method | T1 | T2 | T3 |
|---|---|---|---|
| 3D-GS | 13.81 | 11.70 | 13.50 |
| Scaffold-GS | 6.69 | 7.37 | 8.04 |
| Hierarchical-GS | 9.13 | 8.54 | 8.91 |
| Hierarchical-GS(τ₁) | 16.14 | 13.26 | 14.79 |
| Hierarchical-GS(τ₂) | 19.70 | 19.59 | 18.94 |
| Hierarchical-GS(τ₃) | 24.33 | 25.29 | 24.75 |
| **Our-3D-GS** | **57.08** | **56.85** | **56.07** |
| **Our-Scaffold-GS** | 40.91 | 35.17 | 40.31 |

SmallCity street-view scene (Table III, p9): 3D-GS 25.34 PSNR / 99 FPS; Hierarchical-GS 26.62 / 58;
τ₁ 26.53 / 86; τ₂ 26.29 / 110; τ₃ 25.68 / 159; Our-3D-GS 25.77 / 130; Our-Scaffold-GS 26.10 / 89.

Fig. 1 teaser (p2), MatrixCity zoom-out, "FPS/#GS(M)": Scaffold-GS 20.3 FPS / 3.20 M, Octree-GS
48.5 FPS / 1.25 M, Hierarchical-GS 11.9 FPS / 2.21 M; then 8.68 / 13.0 M, 31.1 / 3.21 M, 13.5 /
4.91 M; then 6.91 / 20.8 M, 32.0 / 3.59 M, 16.5 / 4.51 M. (The caption reads "First row metrics:
FPS/storage size" but the axis label is #GS(M); quoted as printed.)

Training times, Mip-NeRF 360, 40k iterations (p10): 2D-GS 28 min, 3D-GS 34 min, Mip-Splatting
46 min, Scaffold-GS 29 min, Our-2D-GS 20 min, Our-3D-GS 21 min, Our-Scaffold-GS 23 min,
Hierarchical-GS 69 min total (38 min first stage).

### 7. Quality numbers on Mip-NeRF 360 / T&T / Deep Blending (Table I, p7; all reported)
Format: PSNR / SSIM / LPIPS / **#GS(k) rendered per view / storage**.

| Method | Mip-NeRF360 | Tanks&Temples | Deep Blending |
|---|---|---|---|
| Mip-NeRF360 | 27.69 / 0.792 / 0.237 / – | 23.14 / 0.841 / 0.183 / – | 29.40 / 0.901 / 0.245 / – |
| 2D-GS | 26.93 / 0.800 / 0.251 / 397 / 440.8M | 23.25 / 0.830 / 0.212 / 352 / 204.4M | 29.32 / 0.899 / 0.257 / 196 / 335.3M |
| **3D-GS** (strong baseline 1) | 27.54 / 0.815 / 0.216 / 937 / 786.7M | 23.91 / 0.852 / 0.172 / 765 / 430.1M | 29.46 / 0.903 / 0.242 / 398 / 705.6M |
| Mip-Splatting | 27.61 / 0.816 / 0.215 / 1013 / 838.4M | 23.96 / 0.856 / 0.171 / 832 / 500.4M | 29.56 / 0.901 / 0.243 / 410 / 736.8M |
| **Scaffold-GS** (strong baseline 2) | 27.90 / 0.815 / 0.220 / 666 / 197.5M | 24.48 / 0.864 / 0.156 / 626 / 167.5M | 30.28 / 0.909 / 0.239 / 207 / 125.5M |
| Anchor-2D-GS | 26.98 / 0.801 / 0.241 / 547 / 392.7M | 23.52 / 0.835 / 0.199 / 465 / 279.0M | 29.35 / 0.896 / 0.264 / 162 / 289.0M |
| Anchor-3D-GS | 27.59 / 0.815 / 0.220 / 707 / 492.0M | 24.02 / 0.847 / 0.184 / 572 / 349.2M | 29.66 / 0.899 / 0.260 / 150 / 272.9M |
| Our-2D-GS | 27.02 / 0.801 / 0.241 / 397 / 371.6M | 23.62 / 0.842 / 0.187 / 330 / 191.2M | 29.44 / 0.897 / 0.264 / 84 / 202.3M |
| Our-3D-GS | 27.65 / 0.815 / 0.220 / 504 / 418.6M | 24.17 / 0.858 / 0.161 / 424 / 383.9M | 29.65 / 0.901 / 0.257 / 79 / 180.0M |
| **Our-Scaffold-GS** | **28.05 / 0.819 / 0.214 / 657 / 139.6M** | **24.68 / 0.866 / 0.153 / 443 / 88.5M** | **30.49 / 0.912 / 0.241 / 112 / 71.7M** |

Large-scale results (Table II, p8), PSNR / SSIM / LPIPS / #GS(k) / Mem, selected rows:
Block_Small: 3D-GS 26.82 / 0.823 / 0.246 / 1432 / 3387.4M; Scaffold-GS 29.00 / 0.868 / 0.210 / 357
/ 371.2M; Hierarchical-GS 27.69 / 0.823 / 0.276 / 271 / 1866.7M; Our-Scaffold-GS **29.83 / 0.887 /
0.192 / 360 / 380.3M**. Block_All: 3D-GS 24.45 / 979 / 3584.3M; Scaffold-GS 26.30 / 690 / 2272.2M;
Our-Scaffold-GS **27.31 / 0.849 / 0.229 / 344 / 1648.6M**.

BungeeNeRF (Table IV, p10) is the count-consistency evidence, PSNR / #GS(k) per scale 1→4
(scale-1 = closest, scale-4 = whole landscape):
3D-GS 30.00/522, 28.97/1272, 26.19/4407, 24.20/5821 (11.1× count growth);
Mip-Splatting 29.79/503, 29.37/1231, 26.74/4075, 24.44/5298;
Scaffold-GS 30.48/303, 29.18/768, 26.56/2708, 24.95/3876;
Our-2D-GS 30.09/249, 28.72/511, 25.42/1003, 23.41/775;
Our-3D-GS 31.11/411, 29.42/819, 25.88/1275, 23.77/938 (2.3× count growth);
Our-Scaffold-GS 31.11/486, 29.59/1010, 26.51/2206, 25.07/2167 (4.5× count growth).

Ablation (Table VII, p12, Mip-NeRF 360): Scaffold-GS 27.90 / 666 k / 197.5M; w/o next grow 27.64 /
594 k / 99.7M; w/o progressive 27.86 / 698 k / 142.3M; w/o LOD bias 27.85 / 667 k / 146.8M; w/o view
freq. 27.74 / **765 k / 244.4M**; full 28.05 / 657 k / 139.6M.

### 8. Paper-specific: the LOD selection rule and how it bounds rendered count per view
**Number of levels.** K is fixed at initialisation from the observed camera-to-point distance range,
Eq. 5 (p5):

  K = ⌊log₂(d̂_max / d̂_min)⌉ + 1

where d̂_max and d̂_min are the r_d-th largest and smallest camera-to-SfM-point distances, r_d =
0.999 to discard outliers. Anchors at level L sit on a voxel grid of size V_L = ⌊P / (δ/2ᴸ)⌋ · δ/2ᴸ
(Eq. 6), with base voxel δ for LOD 0 and δ = 0.02 for the intermediate level in their setting.

**Per-view level, Eq. 7 (p5).** For viewpoint i and anchor j:

  L̂_ij = ⌊L*_ij⌋ = ⌊Φ(log₂(d_max / (d_ij · s))) + ΔL_j⌋

with d_ij the viewpoint-to-anchor distance, s a focal scale factor applied "For varying intrinsics
... to adjust the distance equivalently", Φ a clamp to [0, K−1], and ΔL_j a **learnable per-anchor
LOD bias** raised by ε = 0.01 whenever the anchor's average Gaussian gradient ∇v > τ_g^L · 0.25.

**Selection rule.** "In summary, the anchor will be selected if its LOD level L_j ≤ L̂_ij." The LOD
is **cumulative**: "the rendered images at LOD K rasterize all Gaussian primitives from LOD 0 to K"
(p4). Anti-popping: anchors that satisfy L_j = L̂_ij + 1 are also submitted, with "their opacities
scaled by L*_ij − L̂_ij", giving piecewise-linear blending between adjacent levels.

**How this bounds the rendered count.** The bound is *implicit and geometric*, not a set number.
Because level index L̂ falls logarithmically with distance (log₂ of the distance ratio) while the
number of anchors per level grows as 8ᴸ in an octree, the two effects cancel: as the camera pulls
back, deeper levels drop out of the cumulative set at exactly the rate at which their anchors would
have become sub-pixel. The evidence that it works is empirical, not a proof or a guarantee:
BungeeNeRF count from scale-1 to scale-4 grows 522 → 5821 (11.1×) for 3D-GS but 411 → 938 (2.3×) for
Our-3D-GS (Table IV), and FPS stays above 30 across trajectory T1 as distance varies (Fig. 9a).
**There is no mechanism to request a specific per-view count or frame time**; the count that comes
out is whatever the octree depth, the base voxel size δ, the learned biases and the pruning
threshold τ_v produce. τ_v is set to 0.7 for small scenes, 0.01 for large scenes and 0.2 for
BungeeNeRF (p9), i.e. hand-tuned per dataset.

### 9. Stated limitations and future work (verbatim, p12)
> "However, certain model components, like octree construction and progressive training, still
> require hyperparameter tuning. Balancing anchors in each LOD level and adjusting training
> iteration activation are also crucial. Moreover, our model still faces challenges associated with
> 3D-GS, including dependency on the precise camera poses and lack of geometry support. These are
> left as our future works." (p12)

> "This suggests potential for future real-world streaming experiences, demonstrating the capability of advanced rendering methods to deliver seamless, high-quality interactive 3D scene and content." (p12)

### 10. Every sentence touching a memory / FPS / time TARGET, "budget", or edge/mobile constraint (verbatim)
The word "budget" never appears, and no edge or mobile device is used. The sentences that touch a
speed target or a resource constraint are:
- p1: "Experiments on diverse datasets demonstrate that our method achieves real-time speeds, with even 10 × faster than state-of-the-art methods in large-scale scenes, without compromising visual quality."
- p1: "However, it struggles in large-scale scenes due to the high number of Gaussian primitives, particularly in zoomed-out views, where all primitives are rendered regardless of their projected size."
- p1: "2) results in redundant and overlap primitives that fail to efficiently represent scene details for real-time rendering, especially in large-scale urban scenes with millions of primitives."
- p1: "As a result, every object detail is rendered, regardless of distance, leading to redundant computations and inconsistent rendering speeds, particularly in zoom-out scenarios involving large, complex scenes."
- p1: "As scene complexity increases, the growing number of Gaussians amplifies bottlenecks in real-time rendering."
- p2 (Fig. 1 caption): "Both SOTA methods fail to render the excessive number of Gaussian primitives included in distant views in real-time, whereas Octree-GS consistently achieves real-time rendering performance (≥30 FPS)."
- p2: "By adaptively querying LOD levels from the octree-based Gaussian structure based on viewing distance and scene complexity, our method minimizes the number of primitives needed for rendering, ensuring consistent efficiency, as shown in Fig. 1."
- p2: "Despite the impressive rendering quality and speed of 3D-GS, its ability to sustain stable real-time rendering with rich content is hampered by the accompanying rise in resource costs." (spans p2–p3)
- p3: "This limitation hampers its practicality in speed-demanding applications, such as gaming in open-world environments and other immersive experiences, particularly for large indoor and outdoor scenes with computation-restricted devices."
- p5: "In this section, we explain how to select the appropriate visible anchors to maintain both stable real-time rendering speed and high rendering quality."
- p10 (Table V caption): "Our method shows consistent rendering speed above 30 FPS at 2k image resolution while all baseline methods fail to meet the real-time performance."
- p10 (Fig. 9 caption): "The figure shows the rendering speed with respect to distance for different methods along trajectory T1, both Our-3D-GS and Our-Scaffold-GS achieve real-time rendering speeds (≥30FPS)."
- p10: "A common limitation across these methods is that each LOD level independently represents the entire scene, increasing storage demands." (this sentence is on p3)

### 11. Code URL and licence
Project page as printed on p1 (line-broken in the PDF): `https://city-super.github.io/octree-gs/`.
**No GitHub URL printed. Licence: not stated.**

---

## 7. FlashGS

### 1. Citation
- Title: **FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering**
- First author: Guofeng Feng (Shanghai AI Laboratory), equal contribution with Siyan Chen.
  Corresponding author Rong Fu.
- Venue: **not stated**. The PDF footer carries a placeholder, "Conference'17, July 2017,
  Washington, DC, USA / 2024". (The published version later appeared at CVPR 2025 per citations in
  the other papers, but this PDF does not say so.)
- arXiv id from file name: **2408.07967** (page 1 stamp: `arXiv:2408.07967v2 [cs.CV] 19 Aug 2024`).

### 2. What it does (3 lines)
A drop-in CUDA rasteriser: an opacity-aware ellipse radius plus an exact ellipse-vs-tile test
replaces the 3σ AABB test, killing false-positive Gaussian-tile pairs before sort and render;
preprocessing and duplicateWithKeys are fused; rendering gets two-level prefetch software
pipelining, warp-divergence control, coarse-grained partitioning and assembly-level tricks.
**Renderer only. "we do not apply pruning or quantization strategies in our implementation so there
is no accuracy loss" (p12). Model untouched, identical PSNR.**

### 3. Device bottleneck
**Rendering (per-pixel blending), then sorting, then preprocess.** Baseline 3DGS breakdown on A100
and V100 with MatrixCity (Fig. 3, p4, and the breakdown text on p11):

| Stage | 3DGS share | FlashGS share |
|---|---|---|
| rendering | 59.6 % (text says "about 60% of the total time", p4) | 47.6 % |
| sorting | 25.4 % | 29.1 % |
| preprocess | 13.2 % | 19.6 % |

Both reported (averaged over 6 representative frames, Fig. 13). Text p4: "Gaussian sorting
(SortPairs) and the preceding generation of the unsorted key-value list (DuplicateWithKeys) take up
nearly 20% of the time. Preprocessing time is within 10%".

**The property that drives everything is the number of Gaussian-tile (key-value) pairs.** Three
sources of false positives are named (Fig. 5, p4): opacity is ignored when defining the ellipse
(65.1 % of Gaussians in MatrixCity smallcity have centre opacity below 0.35); the AABB is the
bounding box of the *circle* of the semi-major axis, so it over-covers the minor axis, worst for
elongated Gaussians; and every tile inside that AABB is binned even when the ellipse misses it.
Secondary: an imbalance between compute-bound (render) and bandwidth-bound (preprocess,
duplicateWithKeys, sort) stages, "which can lead to under-utilization of GPU hardware resources"
(p5). Resolution matters strongly: speedup is always larger at 4K than 1080p on the same scene.

### 4. Representation on device
**Uncompressed Gaussians rendered directly.** No compression, no quantisation, no decode. PSNR is
bit-identical in the reported case ("keeping 31.52 PSNR with 8.57× speedup", p12).

Memory, MatrixCity frame 800, on an **NVIDIA A100** (Table 2, p13, all reported):

| Method | Before rendering | After rendering | #key-value pairs |
|---|---|---|---|
| 3DGS | 7.86 GB | 13.45 GB | 56,148,670 |
| gsplat (packed=False) | 4.52 GB | 10.75 GB | 56,996,302 |
| gsplat (packed=True) | 4.52 GB | 9.83 GB | 56,998,101 |
| **FlashGS** | 6.83 GB | **6.83 GB** | **3,436,142** |

FlashGS is flat before/after because it uses static allocation: "FlashGS also ensures consistency and
predictability in memory usage through a static allocation method". The 6.83 GB is the whole render
working set including the key-value buffers; **the model-only footprint is not broken out**, and the
render-buffer size is not itemised.

### 5. Budget input
**No.** Nothing takes a time, FPS or byte target. All configuration is per-kernel scheduling
(adaptive warp remapping when an ellipse covers more than one tile). The one derived threshold is
the opacity cut-off τ = 1/255 used to define the effective ellipse radius r = √(2 ln(α₀/τ) λ), with
the crossover rule "when α₀ ≤ 0.35, our method outperforms the three-sigma rule, and still use 3σ
conversely" (p6). That is an accuracy-preserving geometric threshold, not a resource budget.
**No achieved-versus-requested pairs.**

### 6. Device numbers
Devices: **NVIDIA RTX 3090** (consumer-grade, Ampere, 24 GB GDDR6X, CC 8.6) for most tests, and
**NVIDIA A100** (80 GB HBM2e, CC 8.0) for sensitivity. GCC 10.5.0, CUDA 12.0. **No phone, no Jetson,
no browser, no accelerator.** The word "mobile" refers to "mobile consumer GPUs" in the abstract
and to GScore's target.

Table 1 (p11), rendering time in **ms** on RTX 3090 unless stated:

| Metric | Truck 1080p | Truck 4k | Train 1080p | Train 4k | Playroom 1080p | Playroom 4k | DrJohnson 1080p | DrJohnson 4k | MatrixCity 1080p | MatrixCity 4k | Rubble 4608×3456 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AvgTime FlashGS / 3DGS | 2.22 / 8.21 | 3.46 / 24.19 | 1.93 / 7.82 | 3.32 / 12.82 | 1.44 / 6.83 | 2.72 / 10.74 | 1.63 / 9.11 | 2.99 / 28.74 | 3.22 / 20.90 | 4.90 / 66.55 | 6.19 / 44.11 |
| MaxTime FlashGS / 3DGS | 4.44 / 10.18 | 4.65 / 29.87 | 3.48 / 12.82 | 4.22 / 42.73 | 2.92 / 10.74 | 4.69 / 32.11 | 4.95 / 16.92 | 6.91 / 57.67 | 4.77 / 40.74 | 7.25 / 140.52 | 9.32 / 67.57 |
| AvgSpeedup | 3.76 | 7.01 | 4.20 | 7.48 | 4.92 | 7.86 | 6.18 | 9.99 | 6.56 | 13.64 | 7.41 |
| MaxSpeedup | 4.74 | 8.60 | 6.89 | 11.51 | 7.91 | 11.16 | 14.99 | 21.66 | 13.49 | 30.53 | 14.10 |
| MinSpeedup | 2.29 | 5.53 | 2.19 | 4.46 | 3.16 | 5.69 | 3.07 | 6.39 | 3.92 | 7.30 | 5.23 |
| AvgTime FlashGS on A100 | 3.37 | 4.87 | 3.05 | 4.58 | 2.36 | 3.98 | 2.59 | 4.35 | 4.18 | 6.55 | 8.08 |

All reported. Aggregate claims (p11): "We can always achieve > 100FPS rendering on RTX 3090";
slowest frame in Rubble reaches **107.3 FPS**; "**7.2× average speedup on all 11 scenes**", "**8.6×
speedup on the 7 large-scale or high-resolution tests, which is 3.87× of the GScores results**";
"**up to 30.53× speedup with an average of 12.18× on the Matrixcity dataset at 4k**". On A100
(p12): "an average of **123.8-423.7 FPS** across 11 scenes", and the A100 is **1.43× slower** than
the 3090 on average "primarily due to the rendering step is dominant ... which heavily relies on
FP32 computation, where the FP32 peak performance of the A100 is only 19.5TFLOPS while the 3090 is
35.6TFLOPS." **Note the abstract and intro state a weaker claim, "an average 4x acceleration over
mobile consumer GPUs" and "up to 4× speedup and 49% memory reduction", which conflicts with the
7.2× / 30.53× numbers in the evaluation.**

Baseline comparison: GScore is a hardware unit and could not be run; "They claim that GScore
achieves 1.86× speedup with their intersection and scheduling techniques", footnote: "Their
shape-aware intersection test provides 1.71× speedup and subtle skipping offers an additional 15%
improvement" (p11).

Energy: **not stated.**

### 7. Quality numbers on Mip-NeRF 360 / T&T / Deep Blending
**No PSNR table.** Mip-NeRF 360 is not used at all. Scenes are Truck and Train (Tanks and Temples),
Playroom and DrJohnson (Deep Blending), MatrixCity, Rubble (Mill19). Quality is asserted invariant:
"The result shows that FlashGS does not change the quality, keeping **31.52 PSNR with 8.57×
speedup**" on MatrixCity 1080p frame 800, where 3DGS took 40.744 ms and FlashGS 4.749 ms (Fig. 16,
p12). Counts appear only as the key-value pair counts in Table 2. Models are "train 30K iterations
on each dataset with 3DGS to obtain the Gaussian model for rendering" (p11). Baselines: **3DGS** and
**gsplat** (memory), plus **GScore** (cited speedup only).

### 8. Paper-specific: which optimisation gave what speed-up, and the memory reduction

**Speed-up attribution.** FlashGS reports an end-to-end number and a set of *component measurements*
rather than a clean per-optimisation speed-up ladder. What is actually attributed:

| Optimisation | Reported effect |
|---|---|
| Opacity-aware ellipse radius, r = √(2 ln(α₀/τ) λ) with τ = 1/255, used when α₀ ≤ 0.35 | motivated by "65.1% Gaussians have the center opacity less than 0.35" (p5); shrinks the ellipse and hence the AABB |
| Tight ellipse bounding box instead of the circumscribed square of the semi-major-axis circle | part of the two-stage filter; Fig. 8 example: 3DGS AABB 16 tiles → GScore OBB 8 tiles → precise intersection **4 tiles** (p6) |
| Exact per-tile ellipse intersection (segment-overlap test, Algorithm 3), with algebraic simplification to avoid division and square root (Eq. 7) | together with the two above: **rendered key-value pairs reduced by 68 %–96 %** (Fig. 15, p12) |
| Kernel fusion of precise intersection with duplicateWithKeys, plus constant-memory parameter passing and static allocation | **global memory accesses in preprocessing reduced by 43 %–87 %** (Fig. 14, p12) |
| Two-level prefetch software pipelining, warp-divergence control (opacity check hoisted to preprocess), coarse-grained workload partition + CSE, base-2 `ex2.approx.ftz.f32` and FMA | **instructions issued per key-value pair reduced by 67 %–71 %** (Fig. 15, p12); total issued instructions in rendering "significantly reduced by one to two orders of magnitude" (p12) |
| Adaptive size-aware scheduling (one thread for a single-tile ellipse, a whole warp for a large one) | load balance; no isolated number given |
| **All combined** | **7.2× average over 11 scenes; 8.6× on the 7 large-scale/high-res tests; 2.29×–30.53× across all frames** (p11) |

Stage shares move from 13.2 / 25.4 / 59.6 % (preprocess / sort / render) in 3DGS to 19.6 / 29.1 /
47.6 % in FlashGS, i.e. every stage is faster but render least dominant afterwards (p11).

**Memory reduction.** "FlashGS allocates less memory than 3DGS and gsplat, **up to 49.2 %
reduction**" (p13). Concretely, MatrixCity frame 800 on A100: **6.83 GB** (FlashGS, identical before
and after rendering because allocation is static) versus 13.45 GB after rendering for 3DGS, 10.75 GB
for gsplat packed=False and 9.83 GB for gsplat packed=True. The cause is the key-value pair count:
**3,436,142 for FlashGS versus 56,148,670 for 3DGS** (a 16.3× reduction, computed from the table by
me, since the paper states the two counts but not the ratio). The intro also states "up to 4× speedup and
49% memory reduction" (p2).

### 9. Stated limitations and future work
**No limitations section and no future-work section exist in this paper.** The nearest statement of
a boundary condition is:
> "And if in a very small and simple scene, each Gaussian might be so small that it intersects only one tile, leading to a lower improvement of our precise intersection." (p12, repeated verbatim on p13)

> "The total rasterizerization time is slower on the A100 compared to the 3090, with on average 1.43× slower. This is primarily due to the rendering step is dominant, as shown in Figure 13, which heavily relies on FP32 computation, where the FP32 peak performance of the A100 is only 19.5TFLOPS while the 3090 is 35.6TFLOPS." (p12)

### 10. Every sentence touching a memory / FPS / time TARGET, "budget", or edge/mobile constraint (verbatim)
The word "budget" never appears. There is no target of any kind. The resource-constraint sentences:
- p1: "The empirical findings demonstrate that FlashGS consistently achieves an average 4x acceleration over mobile consumer GPUs, coupled with reduced memory consumption."
- p1: "Despite 3DGS's advantages, real-time rendering of large-scale or high-resolution areas on city-scale scenes [16] or high quality scenes recorded by consumer GPS receivers [4] is still hindered by limited computational and memory resources. This is due to the increase in the number of Gaussians and the size of each Gaussian as the image scale and resolution increase."
- p1: "A recent work, GScore [15], attempts to analyze the original 3DGS algorithm but primarily addresses limitations on mobile GPUs by designing a novel domain-specific hardware."
- p2: "Compared to 3DGS [13], GScore [15] and gsplat [28], FlashGS achieves up to 4× speedup and 49% memory reduction, while keeping the same high quality metric peak signal-to-noise ratio (PSNR)."
- p2: "We make it possible to achieve efficient rendering for large-scale and high-resolution scenes with low overhead."
- p3: "In some high-resolution, large-scale scenes, 3DGS can generate millions of or even more (huge) Gaussians, putting immense pressure on memory units."
- p3: "2) Introducing multi-GPU parallelism to avoid the memory constraints of a single GPU."
- p3: "GScore [15] is a specific hardware acceleration unit to efficiently support the rendering pipeline over the mobile GPU along with some algorithmic co-design, based on an analysis of Gaussian-based rendering."
- p11: "3DGS claims that they achieve real-time rendering with 1080p resolution on these datasets. We use another 2 large-scale and high-resolution datasets which beyond 3DGS's capabilities to render in real-time."
- p11: "We can always achieve > 100FPS rendering on RTX 3090, even for high-resolution and large-scale datasets. In the slowest frame in Rubble of all datasets, we achieve 107.3 FPS."
- p12: "However, we still achieve significant speedups to 3DGS in all datasets on the A100 with an average of 123.8-423.7 FPS across 11 scenes in Table 1, always reaching real-time rendering."
- p13: "FlashGS also ensures consistency and predictability in memory usage through a static allocation method, with a maximum memory allocation of 6.83 GB, which is lower than the other models."
- p13: "The number of kv pairs generated by each method is a crucial factor in determining memory usage when rendering."

### 11. Code URL and licence
Footnote 1 on p1, as printed: `https://github.com/InternLandMark/FlashGS` ("an open-source CUDA
Python library"). **Licence: not stated in the PDF.**

---

## Cross-paper synthesis

### Device table (every number reported)

| Paper | Device (as named) | Model size or count | Frame rate / time | Memory | Bottleneck stage |
|---|---|---|---|---|---|
| Mobile-GS | Snapdragon 8 Gen 3 phone GPU, Vulkan | 4.6 MB, 0.47 M Gaussians (Mip-NeRF 360) | 127 FPS cold, **74 FPS steady-state**; 116 FPS @1600×1063 Bicycle | not stated (0.83 W total power) | **sort** (3.1–4.9 ms per stage-bar; column totals 6.6–10.8 ms, summed by me from Fig. 2; 2.09 W of 5.89 W) |
| Mobile-GS | Snapdragon 8 Gen 3, 3DGS* baseline | 61.8 MB | 8 FPS cold, 3 FPS steady | not stated | sort |
| WebSplatter | Redmi K70 Pro (Snapdragon 8 Gen 3), Chrome | 1,063,091 splats (bicycle-c) | 33.6 ms | not stated | **render 61.0 %** |
| WebSplatter | Oppo Find X2 (Snapdragon 865), Chrome | 1,063,091 splats | 105.2 ms | not stated | render 48.8 %, sort 37.7 % |
| WebSplatter | iPhone 15 Pro Max, Safari | 1,063,091 splats | 38.5 ms | not stated (baselines OOM-crash) | not broken out |
| WebSplatter | Lenovo MX350 laptop dGPU, Chrome | 5,834,784 splats (garden) | 164.3 ms | not stated | **pre-process 47.2 %** |
| WebSplatter | Intel NUC iGPU (i9-9980HK), Chrome | 5,834,784 splats | 151.2 ms | not stated | **pre-process 40.0 %** |
| WebSplatter | RTX 3070, Chrome | 5,834,784 splats | 9.5 ms | **1.20 GB VRAM** (W1 1.90, W2 >2.80) | render 62.6 % |
| GBU | Jetson Orin NX 16 GB (baseline) | not stated | 13 FPS static / 18 dynamic / 41 avatar | 4 MB SRAM, 15 W | **Gaussian blending 48–78 %**, 62.1 % of DRAM BW at 60 FPS |
| GBU | Jetson Orin NX + GBU plug-in (28 nm, 0.90 mm², 0.22 W) | not stated | **92 / 80 / 102 FPS** | 63 KB SRAM (32 KB reuse cache) | balanced; 10.8× energy efficiency |
| GBU | GBU-Standalone, NeRF-Synthetic | not stated | **172 FPS**, 0.78 W, 1.78 mm² | 63 KB SRAM | — |
| LS-Gaussian | Jetson AGX Orin GPU (baseline 3DGS) | not stated | 49.06 ms/frame | not stated | Gaussian-tile pairs + load imbalance (util. 51.5 %) |
| LS-Gaussian | Jetson AGX Orin + algorithm only | not stated | **5.41× avg**; 51 and 60 FPS on two frames; 1.04 ms on a warped frame | not stated | as above |
| LS-Gaussian | 16 nm accelerator, 1.84 mm² | not stated | **17.3×** (GSCore 9.1×, MetaSapiens 14.5×) | 16 KB counter buffer | util. 88.6 % |
| Pocket-SLAM | NVIDIA A6000 (**not** an edge device) | 0.4·N_init Gaussians | 3.6 FPS EuRoC, 2.4 FPS KITTI | **10.1 GB / 11.9 GB peak** (vs 25.4 / 34.7 GB) | **memory capacity** |
| Octree-GS | NVIDIA A100 80 GB (**no edge device**) | 344–657 k rendered/view; 71.7–1648.6 MB | 35–57 FPS MatrixCity 2k; 89–130 FPS SmallCity | not stated | rendered primitive count per view |
| FlashGS | RTX 3090 | 3,436,142 kv pairs (vs 56.1 M) | 1.44–6.19 ms avg; >100 FPS always; 107.3 FPS worst frame | **6.83 GB** (vs 13.45 GB) | render 59.6 % → 47.6 % |
| FlashGS | A100 80 GB | as above | 123.8–423.7 FPS; 1.43× slower than 3090 | 6.83 GB | render, FP32-bound |

### Which papers accept a budget input

| Paper | Budget input? | Form |
|---|---|---|
| Pocket-SLAM | **Yes, partially** | Global target Gaussian **count** N_tar = 0.4·N_init, plus per-tile budgets B_k clipped to [5, 200]. Not bytes, not FPS, not ms. Reports the achieved memory (61.3 % / 65.7 % reduction) that follows from that count ratio. |
| LS-Gaussian | **Almost** | A 90 FPS *requirement* is stated and checked post hoc ("satisfying the 90 FPS real-time rendering requirement on the Deep Blending dataset"), and the warping window n is the tuning knob, but n is chosen by a sweep, not solved for the target. Internal per-block Gaussian-tile-pair budget (1 + 1/N)·W is a load balancer. |
| Mobile-GS | No | Pruning quantile τ and codebook size are swept, then fixed. |
| WebSplatter | No | — |
| GBU | No | 60 FPS is a pass/fail line and once a matched operating point for an area comparison. |
| Octree-GS | No | LOD level is a function of distance and a learned bias. Hierarchical-GS's τ ∈ {3, 6, 15} pixels is the only threshold-style knob in the paper, and it is a baseline's. |
| FlashGS | No | — |

**Nothing in this set closes a loop between a requested deployment budget (bytes or ms on a named
device) and the training or pruning procedure.** Pocket-SLAM is the closest and it targets a count
ratio, not a resource. This is a real gap for the project.

### Which representation each renders directly

| Paper | Rendered directly, or decoded first? |
|---|---|
| Mobile-GS | **Quantised on disk, decoded once at load.** K-means codebooks + Huffman; SH reconstructed by two fp16 MLPs "once" at inference. But a 3-layer MLP (256/128/64) runs per Gaussian per view for opacity and ϕ, costing 0.23–0.25 ms/frame. |
| WebSplatter | **Full uncompressed splats from PLY.** Per-frame internal packing only: colour+opacity → one u32 RGBA8, axes → two u32 fp16, explicitly to cut render-stage bandwidth. |
| GBU | **Uncompressed, rendered directly**, FP16 in the blending PEs (< 0.1 PSNR loss). No decode. |
| LS-Gaussian | **Uncompressed, rendered directly.** Training-free by design; reuses previous frames' colour/depth instead of compressing. |
| Pocket-SLAM | **Uncompressed, rendered directly.** Only the count is reduced. |
| Octree-GS | **Decoded per view.** Scaffold-style anchors → k neural Gaussians via MLPs conditioned on view distance and direction. The Our-2D-GS / Our-3D-GS variants store explicit Gaussians and skip the decode. |
| FlashGS | **Uncompressed, rendered directly.** Explicitly no pruning or quantisation, PSNR unchanged. |

### Three most useful facts for choosing a target edge device and a cost model

1. **The bottleneck stage moves with the device, so a single global cost model will be wrong.**
   WebSplatter measures the same 5.8 M-splat scene on eight machines and the dominant stage flips:
   render 51.8–62.6 % on RTX 3070, Apple M-series and the Redmi K70 Pro, but pre-process 40.0–47.2 %
   on the Intel iGPU and the MX350, "because a large number of raw splats need to be scanned in the
   pre-processing stage. The limited memory throughput of these low-end devices causes performance
   degradation" (p6). Mobile-GS finds **sort** dominant on Snapdragon 8 Gen 3 (3.1–4.9 ms of ~7–10 ms
   and 2.09 W of 5.89 W); GBU finds **blending** dominant on Jetson Orin NX (48–78 %). A cost model
   must therefore be at least three-term, c₁·N (preprocess and sort, scaling with primitive count)
   + c₂·(Gaussian-tile pairs) (sort key volume) + c₃·(fragments) (blending, scaling with depth
   complexity × resolution), with the coefficients **fitted per device**. That three-term form is my
   proposal, not a model any of these papers states. Note also that frame time
   is not monotone in N: WebSplatter's bonsai (1.24 M) renders faster than train (1.03 M) on all
   three devices in Table 3, which means a pure count budget is a poor proxy for time.

2. **Fragment count, not Gaussian count, is what a low-end device actually pays for, and only ~10 %
   of it is useful.** GBU measures average fragment-to-Gaussian ratios of **541:1, 161:1 and 688:1**
   across three application classes, of which only **7.6 %, 13.7 % and 9.9 %** of fragments exceed
   the opacity threshold (p5). The exponent alone would need **1.1 TFLOPs to hit 60 FPS, 58 % of the
   Jetson Orin NX peak**, and blending alone would need **62.1 % of DRAM bandwidth** at 60 FPS (p7).
   FlashGS confirms the same from the other side: cutting Gaussian-tile pairs from 56.1 M to 3.4 M
   gave a 7.2× average speedup *and* the memory drop from 13.45 GB to 6.83 GB, on an unchanged model
   with unchanged PSNR. **For a render-time budget, target expected Gaussian-tile pairs and expected
   fragments per view, not the byte size or the primitive count.** Bytes and time decouple sharply:
   Mobile-GS's 8.5 MB LocoGS-S baseline runs at 17 FPS on a phone while its own 4.6 MB model runs at
   127, a 7.5× time gap for a 1.8× size gap.

3. **The device menu, with the numbers to choose from.** For a phone target, Snapdragon 8 Gen 3 is
   the reference SoC used by two independent papers, and it comes with a hard caveat: **cold-start
   127 FPS falls to steady-state 74 FPS (a 42 % drop) after thermal equilibrium** (Mobile-GS
   Table 12), and 3DGS itself falls 8 → 3 FPS. Any ms/frame budget must be specified against the
   *steady-state* number. For an embedded-GPU target, the **Jetson Orin NX 16 GB** is the shared
   reference for accelerator work (stock 3DGS at **7–17 FPS** on Mip-NeRF 360 scenes at
   779 × 519–1245 × 825, versus a ≥ 60 FPS AR/VR bar), and the **Jetson AGX Orin** for the 90 FPS
   automotive/AR bar. For a zero-install target, WebGPU in Chrome now covers phones, laptops and
   desktops, but **memory capacity is a cliff, not a slope**: a 5.8 M-splat scene needs 1.20–2.80 GB
   VRAM and several baselines simply crash on the Oppo Find X2, the iPhone 15 Pro Max and the Redmi
   K70 Pro. A size budget therefore buys reliability, not just speed, on the web. Finally, note that
   two of the seven papers (Octree-GS on an A100, Pocket-SLAM on an A6000) never run on an edge
   device at all, so their FPS and memory figures should not be used to calibrate an edge cost model.
