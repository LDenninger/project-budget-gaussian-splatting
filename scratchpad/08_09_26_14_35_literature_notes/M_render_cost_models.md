# M. Render-time cost models for 3DGS

Reading notes for the render-time-budget phase of the BudgetGS project. Every number below
is **reported** by the paper named in its section, copied from the PDF text. Nothing is measured or
inferred here. Where a paper does not state something, the field says "not stated".

Papers, in the order read:

| # | short name | file |
|---|---|---|
| 1 | MetaSapiens | `references/07_runtime_cost/metasapiens_lin_asplos2025_arxiv2407.00435.pdf` |
| 2 | Speedy-Splat | `references/07_runtime_cost/speedy_splat_hanson_cvpr2025_arxiv2412.00578.pdf` |
| 3 | MEGS² | `references/07_runtime_cost/megs2_chen_iclr2026_arxiv2509.07021.pdf` |
| 4 | Splats under Pressure | `references/07_runtime_cost/splats_under_pressure_tajwar_2026_arxiv2604.07177.pdf` |
| 5 | HiGS | `references/07_runtime_cost/higs_pajak_2026_arxiv2606.00352.pdf` |
| 6 | AdaGScale | `references/07_runtime_cost/adagscale_jo_dac2026_arxiv2604.18980.pdf` |
| 7 | RoofGS | `references/07_runtime_cost/roofgs_luo_2026_arxiv2608.15785.pdf` |

---

## 1. MetaSapiens

### 1.1 Citation
- Title: **MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning and Accelerated Foveated Rendering**
- First author: Weikai Lin (University of Rochester). Co-authors Yu Feng (Shanghai Jiao Tong University), Yuhao Zhu (University of Rochester). Both first authors marked "Both authors contributed equally".
- Venue as printed: "ASPLOS '25, March 30-April 3, 2025, Rotterdam, Netherlands", Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1. DOI 10.1145/3669940.3707227.
- arXiv id: 2407.00435 (stamp on p. 1: `arXiv:2407.00435v3  [cs.GR]  6 Dec 2024`). Speedy-Splat cites the same id under the earlier title "RTGS".

### 1.2 What it does (3 lines)
Prunes a dense 3DGS model by a *compute-efficiency* score rather than by point count, then adds a
scale-decay loss term that shrinks the ellipses that touch the most tiles.
Adds the first foveated-rendering scheme for point-based neural rendering, using a strict-subset
point hierarchy with four quality levels and selective multi-versioning of opacity and SH_DC.
Co-designs an ASIC on top of GSCore with tile merging and incremental pipelining to fix the
tile-level load imbalance.

### 1.3 The cost model

**What frame time depends on.** The central claim of the paper (p. 4):

> "Therefore, what does impact the inference speed is the number of tile-ellipse intersections. Fig. 4 shows the latency vs. the average number of intersections per tile (right 𝑦-axis) for each pruned LightGS model; the latency reduction rate and intersection reduction rate match." (p. 4)

and the negative result it replaces (p. 4):

> "Fig. 4 shows the inference latency (𝑥-axis) vs. point count (left 𝑦-axis) of LightGS [17] (which prunes 3DGS [34]) trained on the bicycle trace in the Mip-NeRF 360 dataset at different pruning levels (between 75% and 97%). The latency reduction rate is slower than that of the point reduction rate." (p. 4)

**The efficiency score CE_i, exactly as printed** (Eq. 3, p. 4):

    CE_i = Val_i / Comp_i

with the two terms defined verbatim (p. 4):

> "Val𝑖, contribution of a point 𝑖to pixel values, is defined as the number of pixels that are "dominated" by that point. A pixel is dominated by a point if and only if that point, among all the points, has the highest numerical contribution to the pixel value during rasterization (Sec. 2.1). The numerical contribution of a point 𝑖is quantified by 𝑇𝑖𝛼𝑖 in Eqn. 1a."
> "Comp𝑖, the compute cost of a point 𝑖, which is ignored in all existing pruning methods, is quantified by the number of tiles that intersect and use (the ellipse of) that point, which directly affects the rendering speed as established above."

Per-frame aggregation (p. 4): "We empirically find that the final CE of a point is adequately
measured by the maximum CE across all poses (as opposed to the average, which is susceptible to
dataset bias) in the training set."

**Scale-decay metric** (Eqs. 4-6, p. 5):

    WS  =  (1/N) ∑[i=0..N−1] S_i G_i
    G_i =  (U_i > T) · (U_i − T)
    L   =  L_quality + γ · WS

with S_i the maximum span of point i's ellipse in any direction, U_i the number of tiles point i is
used in when rendering, T a threshold, and γ a hyper-parameter.

**How CE was validated against measured latency.** Only qualitatively, through Fig. 4 on the
`bicycle` trace: latency per frame on the x-axis (axis range 0-40 ms), point count on the left
y-axis (0-1.6e6) and number of tile-ellipse intersections on the right y-axis (0-4.0e6), for LightGS
pruned between 75 % and 97 %. The paper states the *rates* match. **No fitted latency model, no
regression, no prediction-error figure is given anywhere in the paper.**

**Stage breakdown with numbers.** The paper gives no ms-per-stage table. What it does give:
- "First, all 𝑁models must go through the Projection and Filtering stages. In our profiling, these two stages can take up to 18% of the rendering time. Second, blending also adds overhead. Empirically we find that about 25% of the pixels are to be blended and, thus, rendered twice." (p. 6) — reported.
- Accelerator area split: "the volume Rendering Core takes 63% of the total area, other stages occupy the rest 30 %; the SRAMs comprise 7% of the total area." Total area 2.73 mm², vs GSCore 1.45 mm² scaled to 16 nm. (p. 9) — reported.

**Workload-imbalance model.** "The amount of work a tile involves can be quantified by the number of
tile-ellipse intersections." (p. 8). Heatmap of a `bicycle` frame with 16×16 tiles, colour scale
0-2500 intersections per tile: "The amount of intersections can vary by over three orders of
magnitude." (p. 8) — reported.

### 1.4 Is the cost constrained or optimised during training?
**Yes, but the input target is a quality target, not a time target.** The pruning score CE is
compute-aware (it divides by the tile-intersection count), and the scale-decay term γ·WS enters the
training loss directly (Eq. 6). The iteration loop is stopped on quality, not on latency (p. 5):

> "Given a dense model, we first compute the CE for all the points, and repetitively prune a small percentage (𝑅= 10% in our implementation) of the points with the lowest CEs until the quality loss (Lquality in Eqn. 6) is above a prescribed threshold."

> "Second, it does not require quality-specific hyper-parameter tuning to achieve a specific visual quality: monitoring and controlling for L𝑞𝑢𝑎𝑙𝑖𝑡𝑦 automatically yield a model at a given quality." (p. 5)

For the foveated levels the quality target is the HVSQ metric (Eq. 2), used as L_quality (p. 7-8):
"We control for L𝑞𝑢𝑎𝑙𝑖𝑡𝑦 so that the HVSQ at all quality levels is the same as that of 𝐿1".

**A time or FPS target cannot be given as input.** No FPS budget, latency budget or byte budget is an
input anywhere in the method.

**Every achieved-versus-requested pair in the paper** (p. 9, the only one):

| requested (L1 PSNR as fraction of dense) | variant | achieved total model size (fraction of dense) |
|---|---|---|
| 99 % | MetaSapiens-H | 16 % |
| 98 % | MetaSapiens-M | 12 % |
| 97 % | MetaSapiens-L | 10 % |

Verbatim: "The 𝐿1 model in the three variants is pruned to have a PSNR of 99%, 98%, and 97% of that
of the dense model. The total model size of the three variants is 16%, 12%, and 10%, respectively,
of that of the dense model." (p. 9) — reported.

Related achieved numbers (p. 11): "CE reduces the model size by 85%, and FR diminishes the pruning
rate only marginally to 84% owing to selective multi-versioning" — reported.

### 1.5 Device numbers (all reported)
Device named throughout: **mobile Volta GPU on NVIDIA Jetson AGX Xavier**.

- "PBNR is still far from real-time on mobile devices, rendering generally below 10 Frames-Per-Second (FPS) on the mobile Volta GPU [2]." (p. 1)
- Fig. 3 is an FPS boxplot on Mip-NeRF 360 + Tanks&Temples + Deep Blending across 13 traces, measured on the mobile Volta GPU on Jetson Xavier, axis 0-100 FPS with a marked line at 90. Per-model FPS values are not printed as text.
- Ablation (Fig. 12, Jetson Xavier, averaged over all traces, axes FPS 0-120 and PSNR 26.0-28.5 dB): "With a similar quality, our SD implementation achieves 1.6× speedup compared to original dense model; CE-based pruning and FR bring the speedup to 5.8× and 7.4×, respectively." (p. 11)
- "Our slowest variant MetaSapiens-H is 1.9× faster than the fastest baseline while having better or similar objective quality. The fastest variant MetaSapiens-L is 7.9× faster than 3DGS, and can be up to 19.8× on the largest bicycle trace." (p. 10)
- Table 1 (FPS on the mobile Volta GPU, storage in MB, HVS Quality ×10⁻⁵ per level L1/L2/L3/L4), averaged across all datasets:

| method | FPS ↑ | Storage (MB) ↓ | HVSQ L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| SMFR | 125.9 (1×) | 161.6 (1×) | 2.12 | 10.1 | 21.7 | 28.3 |
| MMFR | 52.6 (0.42×) | 311.0 (1.92×) | 2.12 | 1.87 | 1.79 | 1.76 |
| MetaSapiens-H | 102.2 (0.81×) | 171.8 (1.06×) | 2.12 | 2.10 | 2.09 | 2.08 |

- Accelerator (TSMC 16 nm FinFET, area 2.73 mm², four channels Micron 16 Gb LPDDR3-1600): base accelerator "achieves a 18.5× speedup (geomean), up to 24.8×, compared to the GPU baseline"; with tile merging and incremental pipelining "an average 20.9× (up to 27.7×) speedup" (p. 11). Energy: "Our base accelerator achieves a 54.4× energy reduction compared to the GPU baseline. MetaSapiens-TM-IP improves the energy saving to 56.8×" (p. 11). At ~6 mm² area "MetaSapiens out-performs GSCore by 1.6×" (p. 12).
- User study: workstation with an RTX 4090 GPU streaming to a Meta Quest Pro headset, "both of which render smoothly at 90 FPS" (p. 10).
- Model size context: "the bicycle scene in the Mip-NeRF 360 dataset [7] takes about 1.4 GB of space when trained with 3DGS [34]; recent pruning methods [17] reduce the model size of that scene to about 490 MB, which is still large for mobile devices." (p. 6)

### 1.6 Quality on Mip-NeRF 360 / Tanks&Temples / Deep Blending
The paper evaluates all three datasets (13 traces) but reports quality **only as scatter plots**
(Fig. 13: FPS vs PSNR, FPS vs SSIM, FPS vs LPIPS on the mobile Volta GPU). Axis ranges printed:
FPS 0-150, PSNR 26.6-27.4 dB, SSIM 0.81-0.85, LPIPS 0.17-0.25. **Per-dataset numeric PSNR/SSIM/LPIPS
values and per-dataset Gaussian counts or sizes are not stated in the text.**

Text-level claims (all reported):
- "Compared to five state-of-the-art PBNR methods, we out-perform all of them in both objective rendering quality (by up to 0.4 dB in PSNR) and rendering speed (by up to 7.4× on a mobile Volta GPU and 20.9× with our hardware support)." (p. 2)
- Subjective study vs Mini-Splatting-D: "users either have no preference or prefer our method over Mini-Splatting-D. The results are statistically significant through a binomial test with 𝑝< 0.01" (p. 10), 12 participants, four scenes (bicycle, room, drjohnson, truck), 2IFC procedure.

Strongest two baselines named: **Mini-Splatting-D** ("the current-best in quality", used as the dense
teacher) and **Mini-Splatting** / **LightGS** among the pruned models. Their numeric values appear
only in Fig. 13.

### 1.7 Stated limitations and future work (verbatim)
The paper has **no Limitations and no Future Work section**. The only self-identified cost:

> "Our training time is roughly three times as much as that of the original, dense model. Much of the slow down is because the HVS loss is calculated using an open-source Python implementation [5], which could be accelerated with a more efficient implementation in, e.g., CUDA." (p. 9)

### 1.8 Sentences touching a target, budget, latency predictor, hardware-awareness, edge or mobile (verbatim, with page)

> "Point-Based Neural Rendering (PBNR) is emerging as a promising class of rendering techniques, which are permeating all aspects of society, driven by a growing demand for real-time, photorealistic rendering in AR/VR and digital twins. Achieving real-time PBNR on mobile devices is challenging." (p. 1)

> "This paper proposes MetaSapiens, a PBNR system that for the first time delivers real-time neural rendering on mobile devices while maintaining human visual quality." (p. 1)

> "Nevertheless, PBNR is still far from real-time on mobile devices, rendering generally below 10 Frames-Per-Second (FPS) on the mobile Volta GPU [2]." (p. 1)

> "Existing pruning methods focus on reducing the point count in a model, which is ineffective for improving speed in PBNR." (p. 4)

> "While effective for reducing the model size, these methods do not significantly speed up rendering. For instance, CompactGS, LightGS, and Mini-Splatting in Fig. 3 are all pruned models; while generally faster than dense models, they are still below real-time, especially for immersive applications such as AR/VR, which would normally require an FPS of 75–90 [3, 4, 6]." (p. 4)

> "The reason that reducing the point count is ineffective for acceleration is because the computational costs associated with different points vary." (p. 4)

> "Therefore, what does impact the inference speed is the number of tile-ellipse intersections." (p. 4)

> "Challenge 2: Storage Overhead. FR could increase the model size due to the need to store multiple models, exacerbating the storage pressure of PBNR models. For instance, the bicycle scene in the Mip-NeRF 360 dataset [7] takes about 1.4 GB of space when trained with 3DGS [34]; recent pruning methods [17] reduce the model size of that scene to about 490 MB, which is still large for mobile devices." (p. 6)

> "The 𝐿1 model is obtained from the dense model through pruning and scale decay as described in Sec. 3.4, with an iteration budget of 50,000, followed by another 5,000 iterations of fine-tuning with HVSQ loss. The three lower-quality models are obtained from their immediately higher-quality model as described in Sec. 4.3, with a 7,500 iteration budget." (p. 9)

> "We now show the performance results on the mobile Volta GPU on Nvidia Jetson AGX Xavier [2], a representative mobile device for use-cases such as VR." (p. 10)

> "Since Mini-Splatting-D does not render in real-time on a mobile device, we use a workstation with an RTX 4090 GPU to execute both models, both of which render smoothly at 90 FPS." (p. 10)

> "It is thus the slowest of the three — its FPS is way below a 90 FPS real-time requirement, and has the largest storage requirement." (p. 11-12)

> "First, we show that point count is not indicative of performance; tile intersections are (Sec. 3.1). We propose an intersection-aware metric to guide pruning (Sec. 3.2)." (p. 12)

### 1.9 Code and licence
- Code URL as printed: `https://horizon-lab.org/metasapiens/` ("The code and demo are available at: https://horizon-lab.org/metasapiens/.", p. 1).
- Licence as printed: "This work is licensed under a Creative Commons Attribution 4.0 International License." (p. 1)

---

## 2. Speedy-Splat

### 2.1 Citation
- Title: **Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives**
- First author: Alex Hanson (University of Maryland, College Park). Co-authors Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, Tom Goldstein.
- Venue: **not stated inside the PDF** (no venue line on p. 1). Stamp: `arXiv:2412.00578v3  [cs.CV]  14 Aug 2025`. The file name says CVPR 2025, and RoofGS ref. [10] cites it as "2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 21 537–21 546".
- arXiv id: 2412.00578.

### 2.2 What it does (3 lines)
SnugBox computes an opacity-aware tight axis-aligned bounding box for each projected Gaussian, and
AccuTile refines it to the exact set of intersected tiles, both without changing the rendered image.
Soft Pruning removes 80 % of Gaussians at each of the three opacity resets during densification, and
Hard Pruning removes a further 30 % every 3000 iterations after densification.
A reparameterised Hessian score, 36× cheaper in memory than PUP 3D-GS, makes the pruning score
usable inside training.

### 2.3 The cost model

**What frame time depends on** (p. 1-2):

> "We begin by observing that the cost of 3D-GS rendering is proportional to both the number of Gaussians in the scene and the number of pixels processed per Gaussian. Our approach optimizes both factors."

That is the whole model. It is stated in words, never fitted.

**Extent equations.** 3DGS radius (Eq. 8):  r = ⌈3 √λ_max⌉ , with λ_max the largest eigenvalue of
Σ_i2D. Speedy-Splat replaces it with the true alpha-threshold extent (Eqs. 9, 13, 14):

    2 log(255 σ_i) = (p − μ_i2D) Σ_i2D⁻¹ (p − μ_i2D)ᵀ
    t = 2 log(255 σ_i),  x_d = p_x − μ_x,  y_d = p_y − μ_y
    t = a x_d² + 2 b x_d y_d + c y_d²

with Σ_i2D⁻¹ = [[a, b], [b, c]]. SnugBox solves ∂y_d/∂x_d = 0 for the box edges (Eqs. 15, 16):

    y_d = ( −b x_d ± √((b² − a c) x_d² + t c) ) / c
    x_d_args = ± √( −b² t / ((b² − a c) a) )

AccuTile then iterates the shorter side of the tile extent, computing two ellipse points per row:
"our AccuTile algorithm counts tiles in time proportional to the shorter side of the tile extent and
processes tiles in time proportional to the tile count." (p. 5)

**Pruning score** (Eqs. 17-21). Hessian of the L2 loss H = ∇²_G L2 = ∑[φ ∈ P_gt] ∇_G I_G(φ) ∇_G I_G(φ)ᵀ,
per-Gaussian block H_i, PUP score U_i = log |∇_{μ_i, s_i} I_G ∇_{μ_i, s_i} I_Gᵀ|. Speedy-Splat
reparameterises to the scalar 2D value g_i(p):

    Ũ_i = log |∇_{g_i} I_G ∇_{g_i} I_Gᵀ|  =  (∇_{g_i} I_G)²

"the maximum space requirement for this score is proportional to the number of Gaussians N, reducing
the storage requirement by 36× and allowing this score to be used during training." (p. 6)

**Stage breakdown of frame time, Table 1, average execution time in milliseconds across all scenes,
RTX A5000, cumulative rows** (reported):

| method | Preprocess | Inclusive Sum | Duplicate with Keys | Radix Sort | Identify Tile Ranges | Render | Overall |
|---|---|---|---|---|---|---|---|
| Baseline | 0.665 | 0.045 | 0.570 | 1.551 | 0.082 | 4.483 | 7.478 |
| +SnugBox | 0.656 | 0.046 | 0.208 (2.738×) | 0.729 (2.126×) | 0.041 (1.980×) | 2.344 (1.913×) | 4.102 (1.823×) |
| +AccuTile | 0.668 | 0.046 | 0.221 (2.575×) | 0.612 (2.533×) | 0.035 (2.326×) | 2.062 (2.175×) | 3.748 (1.995×) |
| +Soft Pruning | 0.370 (1.798×) | 0.030 (1.494×) | 0.146 (3.906×) | 0.404 (3.843×) | 0.024 (3.422×) | 1.337 (3.354×) | 2.381 (3.141×) |
| +Hard Pruning | 0.091 (7.293×) | 0.016 (2.769×) | 0.090 (6.325×) | 0.215 (7.217×) | 0.013 (6.537×) | 0.619 (7.247×) | 1.114 (6.712×) |

Read as a cost split for the baseline: Render 4.483 / 7.478 ≈ 60 %, Radix Sort 1.551 / 7.478 ≈ 21 %,
Preprocess 0.665 / 7.478 ≈ 9 %, Duplicate 0.570 / 7.478 ≈ 8 % (the percentages are my arithmetic on
the paper's reported ms, the paper prints only the ms and the speedup ratios).

Table 7 repeats the breakdown against StopThePop tile-based culling: Baseline 7.457 ms overall,
Tile-Based Culling 4.051 ms (1.841×), AccuTile 3.660 ms (2.038×). Per-scene overall ms in Table 8,
e.g. bicycle Baseline 14.034 / TBC 6.609 / AccuTile 5.880.

**No fitted or analytic latency model, and no prediction error, is stated.**

### 2.4 Is the cost constrained or optimised during training?
Pruning is scheduled by fixed percentages, with no target of any kind as input:

> "we augment the densification pipeline to include our Soft Pruning method, where the model is pruned immediately before the three opacity resets at 6000, 9000, and 12000 iterations. Surprisingly, we find that we can set extremely high Soft Pruning ratios – in our experiments, visual fidelity is preserved at 80% pruning." (p. 6)

> "In practice, our Hard Pruning method prunes the model by a constant ratio every 3000 iterations starting at iteration 15000. We Hard Prune 30% of Gaussians in each interval, which, when paired with Soft Pruning, empirically reduces the total number of Gaussians across scenes by 10.6×." (p. 7)

**No time, FPS, count or byte target can be given as input.** No achieved-versus-requested pairs
exist. The (80 %, 30 %) operating point was chosen by a sweep: "Our (80%, 30%) pruning percentages
are empirically selected to produce a favorable balance between speed and quality." (p. 12,
Appendix A.2, sweep over Hard 0-40 % in 5 % steps and Soft 0 %, 50-95 %).

### 2.5 Device numbers (all reported)
**Only one device: NVIDIA RTX A5000.** No Jetson, phone, laptop, iGPU or low-end GPU result.

> "All experiments are conducted on an Nvidia RTXA5000 GPU, and the reported Speedy-Splat results represent the average metrics across three independent runs for each scene." (p. 7)

Timing method: "all times in Table 1 and FPS values in Tables 2 and 3 are measured using CUDA events
at the start and end of the forward rendering procedure." (p. 7)

The paper's only edge statement is a gap claim: "Real-time rendering on resource-constrained edge
devices, such as mobile phones, has yet to be achieved [19]." (p. 1)

### 2.6 Quality on the three datasets (all reported, RTX A5000)

Aggregate over all 13 scenes (Table 2), with count, FPS and training time in minutes:

| method | # Gaussians ↓ | FPS ↑ | Training Time ↓ |
|---|---|---|---|
| Baseline (3D-GS) | 2.93M | 134 | 23.2 |
| +SnugBox | 2.97M | 244 (1.82×) | 21.2 (1.09×) |
| +AccuTile | 2.97M | 267 (1.99×) | 21.0 (1.10×) |
| +Soft Pruning | 1.64M (1.79×) | 420 (3.14×) | 17.5 (1.32×) |
| +Hard Pruning | 0.28M (10.6×) | 898 (6.71×) | 15.7 (1.47×) |

**Mip-NeRF 360** (Table 3, ratios relative to 3D-GS; Comp = compression in Gaussian count):

| method | Comp ↑ | FPS ↑ | Train ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|
| 3D-GS | 1.00× | 1.00× | 1.00× | 27.55 | 0.814 | 0.222 |
| Mini-Splat (strongest baseline) | 6.84× | 3.20× | 1.26× | 27.34 | 0.822 | 0.217 |
| PUP (2nd strongest by compression) | 8.65× | 2.55× | – | 26.83 | 0.792 | 0.268 |
| ELMGS | 5.00× | 2.69× | – | 27.00 | 0.779 | 0.286 |
| +SnugBox (ours) | 0.99× | 1.81× | 1.08× | 27.55 | 0.814 | 0.221 |
| +AccuTile (ours) | 0.99× | 1.99× | 1.10× | 27.57 | 0.814 | 0.221 |
| +Soft Pruning (ours) | 1.79× | 3.14× | 1.30× | 27.32 | 0.807 | 0.246 |
| +Hard Pruning (ours) | 10.6× | 6.51× | 1.45× | 26.94 | 0.782 | 0.296 |

**Tanks & Temples** (Table 5):

| method | Comp ↑ | FPS ↑ | Train ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|
| 3D-GS | 1.00× | 1.00× | 1.00× | 23.70 | 0.849 | 0.178 |
| Mini-Splat | 9.20× | – | – | 23.18 | 0.835 | 0.202 |
| PUP | 10.0× | 4.00× | – | 22.72 | 0.801 | 0.244 |
| ELMGS | 5.00× | 4.05× | – | 23.90 | 0.825 | 0.233 |
| +AccuTile (ours) | 0.99× | 1.67× | 1.12× | 23.73 | 0.849 | 0.177 |
| +Hard Pruning (ours) | 10.1× | 6.30× | 1.58× | 23.45 | 0.821 | 0.241 |

**Deep Blending** (Table 6):

| method | Comp ↑ | FPS ↑ | Train ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|
| 3D-GS | 1.00× | 1.00× | 1.00× | 29.09 | 0.886 | 0.288 |
| Mini-Splat | 8.06× | – | – | 29.98 | 0.908 | 0.253 |
| EAGLES | – | 1.30× | 1.31× | 29.92 | 0.900 | 0.250 |
| PUP | 10.0× | 4.51× | – | 28.85 | 0.881 | 0.301 |
| +AccuTile (ours) | 0.97× | 2.32× | 1.13× | 29.12 | 0.885 | 0.288 |
| +Hard Pruning (ours) | 11.1× | 7.46× | 1.57× | 29.32 | 0.887 | 0.311 |

Per-scene FPS (Table 12), Baseline → +Hard Pruning: bicycle 71 → 662 (9.25×), garden 91 → 640
(7.03×), truck 185 → 1149 (6.21×), train 200 → 1392 (6.95×), drjohnson 126 → 1122 (8.91×),
playroom 172 → 1277 (7.42×), kitchen 117 → 809 (6.90×), room 140 → 942 (6.73×), bonsai 201 → 978
(4.87×), counter 142 → 842 (5.94×), flowers 164 → 825 (5.02×), stump 141 → 724 (5.12×),
treehill 138 → 957 (6.93×).

Teaser numbers (p. 1, Tanks & Temples truck): 3DGS "2.6M Gaussians  25.39 PSNR  184.87 FPS" versus
Ours "0.26M Gaussians  25.34 PSNR 1148.77 FPS". Fig. 3 (p. 7): playroom 3D-GS "2.5M Gaussians
172.15 FPS", PUP "0.45M Gaussians 261.67 FPS", Ours "0.46M Gaussians 723.79 FPS"; bicycle 3D-GS
"3.4M Gaussians 125.97 FPS", PUP "0.34M Gaussians 413.30 FPS", Ours "0.25M Gaussians 463.83 FPS";
drjohnson 3D-GS "4.5M Gaussians 141.47 FPS", PUP "0.32M Gaussians 1121.97 FPS", Ours "0.22M
Gaussians 1277.20 FPS".

### 2.7 Stated limitations and future work (verbatim)

> "6. Limitations
> A limitation of Speedy-Splat is that it produces slightly lower image quality than 3D-GS. However, this degradation is expected at high compression ratios and is also observed in comparable techniques. Additionally, a direct comparison of our efficient pruning score to the PUP 3D-GS pruning score illuminates a slight, yet noticeable, gap in performance. Future work could explore the possibility of another efficient pruning score that delivers higher performance." (p. 8)

### 2.8 Sentences touching a target, budget, latency predictor, hardware-awareness, edge or mobile (verbatim, with page)

> "However, its rendering speed and model size still present bottlenecks, especially in resource-constrained settings." (p. 1)

> "Efficient rendering is essential in applications such as virtual reality, networked systems, and multi-view streaming. Real-time rendering on resource-constrained edge devices, such as mobile phones, has yet to be achieved [19]." (p. 1)

> "Although recent works on compressing 3D-GS models [6, 7, 11, 25] achieve some speed-ups by reducing the number of parameters, few approaches directly target rendering speed [8, 19]." (p. 1)

> "The real-time rendering speed of 3D-GS [14] on desktop GPUs has inspired research focused on further accelerating both its training and inference in resource-constrained environments. In this section, we review related works that specifically target these performance improvements." (p. 2)

> "Taming 3DGS [22] proposes a constructive optimization process that limits the number of Gaussians to a pre-defined threshold set by the user." (p. 2)

> "Several works reduce training time and memory requirements by enforcing geometric constraints [21, 28, 31, 32]. Most of these approaches are orthogonal to ours and can be applied alongside it." (p. 2)

No sentence in the paper contains "budget", "latency predictor" or "hardware-aware".

### 2.9 Code and licence
- Code URL as printed: `https://speedysplat.github.io` (p. 1, under the author list).
- Licence: not stated.

---

## 3. MEGS²

### 3.1 Citation
- Title: **MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning**
- First authors: Jiarui Chen (HKUST) and Yikeng Chen (HKUST, SZU), marked "Equal contribution". Corresponding authors Yuan Liu (HKUST) and Yujing Sun (NTU).
- Venue as printed: running header "Published as a conference paper at ICLR 2026".
- arXiv id: 2509.07021 (stamp: `arXiv:2509.07021v3  [cs.CV]  27 Feb 2026`).

### 3.2 What it does (3 lines)
Replaces spherical harmonics entirely with up to three arbitrarily oriented, prunable spherical
Gaussian lobes, halving the per-primitive colour parameter count.
Formulates primitive-count pruning and lobe-count pruning as one L0-constrained optimisation with a
single total-parameter budget κ, solved by an ADMM-style proximal scheme during training.
Targets **rendering VRAM** rather than file size, and demonstrates the result in a WebGL viewer on
phones and a laptop GPU.

### 3.3 The cost model (memory, not time)

**What rendering memory depends on** (p. 2):

> "In this paper, we observe that the overall memory consumption for 3DGS rendering is intrinsically tied to two key factors: the total primitive count and the parameters per primitive."

> "Specifically, the rendering VRAM can be divided into a static component and a dynamic component. The static part relates directly to the total number of Gaussian primitives loaded into the renderer, which is a product of the primitive count and the parameters per primitive. On the other hand, the dynamic part, which consists of intermediate data like projected 2D Gaussian parameters and the tile-depth-gaussian key-value table, is also related to the primitive count in the specific camera viewpoint." (p. 2)

So, informally:  VRAM_static ≈ N · P_per-primitive  and  VRAM_dynamic ≈ f(N_visible, tile-depth-Gaussian
key-value table size). The dynamic part is never written as an equation and never fitted.

Measurement protocol (Appendix A.2, p. 16): static VRAM is "the VRAM required to load all 3DGS
primitives into the renderer and dequantize or decode them into a ready-to-render state"; rendering
VRAM is "the peak VRAM usage during the rendering process, which is typically larger than the static
VRAM due to the introduction of intermediate variables (e.g., projected 2D Gaussian attributes,
key-value tables for tile-based rendering) during rendering. We measure this value across all test
viewpoints and report the average." All VRAM values are taken from PyTorch's own allocator.

**The training-time budget constraint** (Eq. 3, p. 5):

    min[o, s, Θ]  L(o, s, Θ)     s.t.   ρ_o ‖o‖₀ + ρ_s ‖s‖₀ ≤ κ      (total memory constraint)

with o ∈ ℝ^(N×1) the opacity vector for N primitives, s ∈ ℝ^(N×3) the flattened sharpness vector,
Θ ∈ ℝ^(N×13) the remaining Gaussian variables, **ρ_o = 11** and **ρ_s = 7** the base parameter counts
of one primitive and one SG lobe respectively, and κ the total parameter budget.

Solved by ADMM with an indicator function h (Eqs. 4-7), factorised proximal projections
‖õ‖₀ < κ_o and ‖s̃‖₀ < κ_s (Eqs. 8-9), and lobe importance by dynamic range (Eq. 10):

    D_i = | max_v(c_i) − min_v(c_i) | = |a_i| (1 − e^(−2 s_i))

Colour compensation for a removed lobe (Eq. 12):

    Δc₀ = a_i · (1 − e^(−2 s_i)) / (2 s_i),      c′₀ = c₀ + Δc₀

Spherical Gaussian lobe (Eq. 1): G(v; μ, s, a) = a e^(s(μ·v − 1)), colour (Eq. 2):
c(v) = c₀ + ∑[i=1..n] G(v; μ_i, s_i, a_i).

**Per-primitive colour parameter cost, Table 3 (float32 counts, values for our method computed on
DeepBlending)** — reported:

| method | Storage | Rendering | Decode Overhead |
|---|---|---|---|
| 3DGS | 48 | 48 | No |
| EAGLES | < 1 | > 48 | Yes |
| CompactGaussian | < 1 | > 32 | Yes |
| ours w/o lobe-pruning | 24 | 24 | No |
| ours | 9.7 | 9.7 | No |

**Frame time.** MEGS² states no frame-time model and no stage breakdown. FPS is measured only, in
Table 4 (RTX 3090 desktop) and Table 9 (WebGL across devices). **No fitted latency model, no
prediction error.**

### 3.4 Is the cost constrained or optimised during training?
**Yes, and this is the one paper in this set where a budget is an explicit input to training.** κ (a
total parameter budget serving as a memory proxy) is the constraint of the optimisation problem
solved over 10,000 iterations at an interval of 50 iterations, with penalty δ = 0.0005 (p. 16).

> "More importantly, we propose a unified soft pruning framework that models primitive-number and lobe-number pruning as a single constrained optimization problem." (p. 1, Abstract)

> "To achieve a globally optimal memory footprint, our framework models the two traditionally separate pruning problems—primitive-count pruning and per-primitive lobe pruning—as a single constrained optimization problem, with the total parameter budget serving as a unified constraint." (p. 2)

> "Building upon GaussianSpa (Zhang et al., 2025b)'s sparsification framework and our analysis, we extend the pruning objective from primitive count optimization to total memory budget control" (p. 5)

**A time or FPS target cannot be given as input.** The budget is a parameter count, and the mapping
from κ to megabytes of rendering VRAM is never given.

**Achieved-versus-requested pairs.** The paper reports **no table of requested κ (or bytes) against
achieved VRAM**. The only requested-versus-achieved evidence is Figure 6 (p. 20), a sensitivity study
on the `garden` scene over the lobe budget κ_s ∈ {60 %, 65 %, 70 %} (x-axis) and the primitive budget
κ_o ∈ {65 %, 68 %, 70 %} (three lines), PSNR axis 24.50-27.00 dB. Its caption says "The values
annotated in the boxes indicate the specific rendering VRAM usage for each configuration", but those
annotations are rendered as image and do not extract as text, so the achieved VRAM per configuration
is **not recoverable from the PDF text**. The two shipped operating points are named HQ and LM:
"HQ denotes the version prioritizing high rendering quality, and LM denotes the version prioritizing
lower VRAM consumption." (Table 1 caption, p. 8)

### 3.5 Device numbers (all reported)
Table 9, `Bicycle` scene, WebGL-based viewer modified from `antimatter15/splat`, FPS. "cannot render"
means the browser crashed or showed a black screen, "render error" means incorrect rendering:

| method | RTX3060 | Dimensity 9400+ | Snapdragon 8+ Gen 1 | Snapdragon 888 |
|---|---|---|---|---|
| 3DGS | 26.3 | 6.6 | cannot render | connot render |
| GaussianSpa | 165.0 | 31.4 | render error | render error |
| Ours | 165.0 | 91.0 | 120.9 | 60.1 |

Figure 1 on p. 1 gives a different set of live numbers on named hardware:
- NVIDIA GeForce RTX 3060 Laptop GPU: "Ours: 117.4 FPS", "3DGS(w SH): 27 FPS"
- MediaTek Dimensity 9400+: "Ours: 91.0 FPS", "3DGS(w SH): 6.6 FPS"
- Qualcomm Snapdragon 865: "Ours: 34.9 FPS", "3DGS(w SH): N/A"

Note the RTX 3060 figures differ between Fig. 1 (117.4 / 27) and Table 9 (165.0 / 26.3), and Fig. 1
names a Snapdragon 865 that does not appear in Table 9. Both are copied here as reported.

Named consumer devices from the appendix figure captions (pp. 22-23):
- OnePlus Ace 5 Ultra with MediaTek Dimensity 9400+ (Fig. 10)
- Lenovo laptop with NVIDIA GeForce RTX3060 Laptop GPU (Fig. 11)
- Huawei MatePad Air with Qualcomm Snapdragon 888 (Fig. 12)
- RedMi K60 with Qualcomm Snapdragon 8+ Gen 1 (Fig. 13)

Desktop FPS, Mip-NeRF360, Table 4 (single NVIDIA RTX 3090, 24 GB): Scaffold-GS 123 FPS / VRAM 612 MB,
HAC++ (highrate) 115 / 637, HAC++ (lowrate) 132 / 514, Ours 200 / 265.

Training time (Table 12): MEGS² 37min51s on RTX 3090 and 23min20s on RTX 4090 for `Bicycle`, versus
GaussianSpa 27min57s and 3DGS 25min28s on RTX 4090.

### 3.6 Quality on the three datasets (Table 1, VRAM in MB as Stat. / Rend.) — all reported

| method | M360 PSNR | SSIM | LPIPS | Stat. | Rend. | T&T PSNR | SSIM | LPIPS | Stat. | Rend. | DB PSNR | SSIM | LPIPS | Stat. | Rend. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DGS | 27.48 | 0.813 | 0.217 | 648 | 1717 | 23.68 | 0.849 | 0.171 | 370 | 1021 | 29.71 | 0.902 | 0.242 | 582 | 1569 |
| GaussianSpa (strongest baseline) | 27.56 | 0.824 | 0.215 | 115 | 448 | 23.73 | 0.857 | 0.162 | 106 | 336 | 30.00 | 0.912 | 0.239 | 104 | 372 |
| Mini-Splatting (2nd strongest) | 27.40 | 0.821 | 0.219 | 125 | 477 | 23.45 | 0.841 | 0.186 | 72 | 253 | 30.05 | 0.909 | 0.254 | 89 | 324 |
| MaskGaussian | 27.43 | 0.811 | 0.227 | 271 | 799 | 23.72 | 0.847 | 0.181 | 132 | 517 | 29.69 | 0.907 | 0.244 | 156 | 501 |
| Ours(HQ) | 27.54 | 0.824 | 0.209 | 55 | 265 | 23.45 | 0.853 | 0.159 | 51 | 211 | 30.17 | 0.912 | 0.233 | 54 | 243 |
| Ours(LM) | 27.21 | 0.814 | 0.227 | 40 | 224 | 23.27 | 0.851 | 0.167 | 37 | 163 | 30.01 | 0.908 | 0.246 | 33 | 193 |

Gaussian counts in millions (Table 13): 3DGS 2.718 / 1.568 / 2.461; Mini-splatting 0.559 / 0.320 /
0.397; GaussianSpa 0.528 / 0.447 / 0.409; Ours(HQ) 0.611 / 0.618 / 0.598; Ours(LM) 0.462 / 0.437 /
0.411 (M360 / T&T / DB).

Storage in MB (Table 10): MaskGaussian 271 / 156, GaussianSpa 115 / 104, EAGLES 54 / 52,
Ours(HQ) 55 / 54, Ours(LM) 40 / 33 (MipNeRF 360 / DeepBlending).

FPS on Mip-NeRF360 appears only in Table 4 (Ours 200 FPS). Headline claims: "more than an 8×
compression rate for static VRAM and nearly a 6× compression rate for rendering VRAM across all
datasets"; versus GaussianSpa "a nearly 2× compression rate for static VRAM on the Mip-NeRF360
dataset and reduce rendering VRAM by approximately 40%"; versus decode-based methods "our method
achieves superior perceptual quality (SSIM/LPIPS) with 50-60% less VRAM and a 1.5-1.7x rendering
speedup" (p. 8).

### 3.7 Stated limitations and future work (verbatim)

> "Limitation Our method focuses on compressing static memory for broad, renderer-agnostic applicability. We leave the optimization of implementation-specific dynamic VRAM for future work, and note that the model's performance on highly complex highlights warrants further investigation." (p. 10)

### 3.8 Sentences touching a target, budget, latency predictor, hardware-awareness, edge or mobile (verbatim, with page)

> "3D Gaussian Splatting (3DGS) has emerged as a dominant novel-view synthesis technique, but its high memory consumption severely limits its applicability on edge devices." (p. 1)

> "We present MEGS2, a memory-efficient framework designed to solve the rendering memory bottleneck of 3D Gaussian Splatting and enable high-quality, real-time rendering on edge devices. As demonstrated in our WebGL-based viewer, 3D Gaussian Splatting(3DGS) with Spherical Harmonics (SH) exhibits low frame rates on desktop GPU and fails to run on some mobile platforms. In contrast, MEGS2 achieves interactive frame rates across all tested devices, significantly expanding the applicability of 3DGS." (p. 1, Fig. 1 caption)

> "compression of 3DGS (Bagdasarian et al., 2025) is gaining significantly more attention, with the goal of enabling real-world use cases on edge devices, such as mobile 3D scanning and previewing, virtual try-on, and real-time rendering in video games." (p. 2)

> "Nevertheless, existing compression methods only focus on storage compression rather than memory compression. While the former storage compression speeds up the one-time data transfer of 3DGS files, the rendering memory dictates whether the 3DGS rendering process can run smoothly on edge devices. This limits the applicability of 3DGS applications to low-end devices like cell phones." (p. 2)

> "To achieve a globally optimal memory footprint, our framework models the two traditionally separate pruning problems—primitive-count pruning and per-primitive lobe pruning—as a single constrained optimization problem, with the total parameter budget serving as a unified constraint." (p. 2)

> "We propose a unified soft pruning framework that models both primitive-count and lobe-count as a memory-constrained optimization problem, which yields superior performance compared to existing staged or hard-pruning methods." (p. 2)

> "Building upon GaussianSpa (Zhang et al., 2025b)'s sparsification framework and our analysis, we extend the pruning objective from primitive count optimization to total memory budget control, formalized as:" (p. 5)

> "where o ∈ RN×1 means opacity vector for N Gaussian primitives, s ∈ RN×3 means flattened sharpness vector for N Gaussians, Θ ∈ RN×13 (other Gaussian variables), ρo = 11 and ρs = 7 count base parameters for a single Gaussian primitive and a single SG lobe respectively, and κ is the total parameter budget." (p. 5)

> "Learning rate η, penalty δo, δs, budget κ" (p. 6, Algorithm 1 inputs)

> "where L is the loss function, ∥· ∥0 denotes the L0 norm (count of non-zero elements), and κ represents the memory budget." (p. 14)

> "MEGS² shifts the focus of 3DGS compression towards rendering VRAM efficiency, paving the way for high-quality rendering on edge devices." (p. 10)

> "We investigate the impact of varying the lobe budget κs (x-axis) and the primitive budget κo (represented by different colored lines) on rendering quality (PSNR) and memory consumption on the garden scene. The values annotated in the boxes indicate the specific rendering VRAM usage for each configuration." (p. 20, Fig. 6 caption)

No sentence contains "latency", "hardware-aware" or "latency predictor".

### 3.9 Code and licence
- Code URL: **not stated**. "While not included with this submission, our full project (including the WebGL renderer, training, and evaluation scripts) will be released on GitHub upon publication." (p. 11)
- The WebGL viewer is stated to be modified from `https://github.com/antimatter15/splat` (Kwok & Ye, 2023).
- Licence: not stated.

---

## 4. Splats under Pressure

### 4.1 Citation
- Title: **Splats under Pressure: Exploring Performance–Energy Trade-offs in Real-Time 3D Gaussian Splatting under Constrained GPU Budgets**
- First author: Muhammad Fahim Tajwar (National University of Singapore). Co-authors Arthur Wuhrlin (NUS / Telecom Paris), Anand Bhojan (NUS).
- Venue: **not stated** (ACM CCS Concepts and Keywords blocks are present but no conference line). Stamp: `arXiv:2604.07177v1  [cs.GR]  8 Apr 2026`.
- arXiv id: 2604.07177.

### 4.2 What it does (3 lines)
Emulates four consumer GPU tiers on one RTX 4090 by matching sustained FP32 TFLOPS through power
caps and core/memory clock limits.
Measures gsplat rasterisation FPS on the Mip-NeRF 360 `garden` scene at 1920×1080 across four
LapisGS level-of-detail splat counts, with and without a 4D animation deformation MLP.
Reports the tier-by-count frame-rate envelope and identifies the splat counts at which each tier
falls below real time.

### 4.3 The cost model

**There is no analytic or fitted cost model.** The paper is a measurement study. The only equations
are the two derived energy metrics (p. 4):

    E_frame = P_avg / FPS          (Eq. 1, J/frame)
    η_perf  = FPS / P_avg          (Eq. 2, FPS/W)

with P_avg the average GPU power in W, logged by `nvidia-smi dmon`.

**Emulation calibration.** Sustained TFLOPS is estimated as 66.6 % of the vendor theoretical FP32
peak, then matched empirically with large GEMM workloads under `nvidia-smi -pl`, `-lgc`, `-lmc`.
Table 1 (all reported):

| Target GPU | Theoretical FP32 TFLOPs | Estimated Sustained TFLOPs | Nominal Power (W) | Emulated Power (W) | Nominal Core Clock (MHz) | Emulated Core Clock (MHz) | Nominal Memory Bandwidth (GB/s) | Required Memory Clock (MHz) | Emulated Memory Clock (MHz) | Measured TFLOPs after Emulation |
|---|---|---|---|---|---|---|---|---|---|---|
| RTX 4090 | 82.58 | 55.05 | 450 | 450 | 2520 | 2520 | 1008 | 10 501 | 10 501 (exact) | 53.58 |
| RTX 4070 Ti | 40.09 | 26.73 | 285 | 285 | 2610 | 1125 | 504 | 5 250 | 5 001 (–4.7%) | 26.49 |
| RTX 3070 | 20.31 | 13.54 | 220 | 150 | 1725 | 570 | 448 | 4 667 | 5 001 (+7.2%) | 13.49 |
| RTX 3050 | 9.10 | 6.07 | 130 | 150 | 1777 | 255 | 224 | 2 333 | 5 001 (+114%) | 6.12 |

**FPS versus splat count per GPU tier** (Table 2, `garden`, 1920×1080, identical camera paths,
mean ± SD, all reported):

| GPU | Animations | # Splats | FPS (mean ± SD) |
|---|---|---|---|
| RTX 4090 | Yes | 3.49 M | 38.9 ± 2.7 |
| RTX 4090 | Yes | 2.83 M | 41.2 ± 3.2 |
| RTX 4090 | Yes | 1.87 M | 45.3 ± 3.9 |
| RTX 4090 | Yes | 0.62 M | 49.6 ± 4.8 |
| RTX 4090 | No | 3.45 M | 44.8 ± 2.6 |
| RTX 4090 | No | 2.79 M | 47.9 ± 3.8 |
| RTX 4090 | No | 1.83 M | 51.3 ± 5.4 |
| RTX 4090 | No | 0.58 M | 58.8 ± 6.0 |
| RTX 4070 Ti | Yes | 3.49 M | 31.1 ± 2.6 |
| RTX 4070 Ti | Yes | 2.83 M | 35.4 ± 2.7 |
| RTX 4070 Ti | Yes | 1.87 M | 39.3 ± 3.5 |
| RTX 4070 Ti | Yes | 0.62 M | 45.0 ± 4.8 |
| RTX 4070 Ti | No | 3.45 M | 36.2 ± 2.9 |
| RTX 4070 Ti | No | 2.79 M | 40.2 ± 3.3 |
| RTX 4070 Ti | No | 1.83 M | 48.1 ± 4.9 |
| RTX 4070 Ti | No | 0.58 M | 58.6 ± 5.3 |
| RTX 3070 | Yes | 3.49 M | 28.1 ± 1.9 |
| RTX 3070 | Yes | 2.83 M | 29.8 ± 1.6 |
| RTX 3070 | Yes | 1.87 M | 32.9 ± 2.2 |
| RTX 3070 | Yes | 0.62 M | 40.4 ± 5.2 |
| RTX 3070 | No | 3.45 M | 30.2 ± 1.7 |
| RTX 3070 | No | 2.79 M | 34.1 ± 3.2 |
| RTX 3070 | No | 1.83 M | 41.2 ± 4.9 |
| RTX 3070 | No | 0.58 M | 57.0 ± 4.1 |
| RTX 3050 | Yes | 3.49 M | 17.3 ± 1.2 |
| RTX 3050 | Yes | 2.83 M | 19.6 ± 1.2 |
| RTX 3050 | Yes | 1.87 M | 22.7 ± 1.4 |
| RTX 3050 | Yes | 0.62 M | 29.9 ± 2.0 |
| RTX 3050 | No | 3.45 M | 19.7 ± 1.0 |
| RTX 3050 | No | 2.79 M | 22.4 ± 1.4 |
| RTX 3050 | No | 1.83 M | 29.1 ± 2.2 |
| RTX 3050 | No | 0.58 M | 45.8 ± 7.3 |

LoD tiers: LOD 0 = 580,604 Gaussians, LOD 1 = 1,834,311, LOD 2 = 2,795,038, LOD 3 = 3,448,340, plus
38,844 animated splats when animation is on.

Animation overhead (p. 4, reported): "RTX 4090. Average loss of 9 fps (≈15%) at 0.58 M splats. RTX
4070 Ti. ≈13 FPS drop (22%). RTX 3070. ≈17 FPS drop (30%). RTX 3050. ≈16 FPS drop, (35%)."

**The energy numbers.** The paper defines E_frame and η_perf and says it measures them, but
**reports no J/frame value and no FPS/W value anywhere.** Section 5.3 in full: "In addition to frame
rate measurements, we analyze the relationship between rendering performance and power consumption.
Specifically, we examine FPS–power curves, energy per frame (J/frame), and performance per watt
(FPS/W) across the emulated GPU capability tiers. These metrics provide insight into how efficiently
real-time 3DGS rasterization utilizes available GPU resources under constrained power budgets."
(p. 5). The only power numbers in the paper are the nominal and emulated watts in Table 1
(450 / 450, 285 / 285, 220 / 150, 130 / 150 W).

**No prediction error is stated,** because no model is fitted.

### 4.4 Is the cost constrained or optimised during training?
**No.** The paper trains a LapisGS layered LoD hierarchy offline and then renders. No cost term
enters any loss. No time or FPS target is an input. The only control knob is the number of enabled
LoD layers: "With this method, one can control the number of gaussians by lower or increasing the
number of used layers." (p. 2). No achieved-versus-requested pairs exist.

### 4.5 Device numbers (all reported)
Every result is on the single physical **NVIDIA RTX 4090**, throttled to emulate the RTX 4090,
RTX 4070 Ti, RTX 3070 and RTX 3050 tiers (Tables 1 and 2 above). Rendering at 1920×1080, gsplat
CUDA rasteriser, tests run multiple times for 2 minutes each and averaged.

Feasibility conclusions (p. 5, reported):
> "Real-time (≥60 FPS) 3DGS is readily achievable on 4090/4070 Ti/3070 for scenes that fit below ∼600 000 splats; even the entry-level 3050 approaches 46 fps."

> "At higher LoDs or scene sizes (∼3–4 M splats) the 4090 remains interactive (∼45 FPS), whereas a 3070 drops below 30 FPS and a 3050 gets unusable fps."

Jetson and mobile SoCs are explicitly excluded from measurement (p. 3):
> "We deliberately exclude embedded systems such as the NVIDIA Jetson AGX Orin 64GB and standalone VR SoCs (e.g., Qualcomm XR2, Apple M2) from our study. While these platforms are important in mobile and embedded compute, they differ significantly in architecture and software stack."

### 4.6 Quality on Mip-NeRF 360 / Tanks&Temples / Deep Blending
**None.** Only the `garden` scene from Mip-NeRF 360 is used, and no PSNR, SSIM or LPIPS is reported
anywhere. No quality baselines are compared.

### 4.7 Stated limitations and future work (verbatim)

> "6.1 Limitations of the GPU–Performance Emulation
> Table 1 summarises the clock and power caps used to throttle our RTX 4090 so that its sustained FP32 throughput matches that of four reference consumer GPUs.
> Bandwidth mismatch. The 4090 can only be down-clocked to 405 MHz, 810 MHz, and 5.001 MHz. As a result the emulated 3070 and 3050-tiers retain×2–×3 more memory bandwidth than the physical cards. For splat counts above ∼1 M, where production systems may be bandwidth-bound, our frame-rate numbers for the lower tiers are therefore optimistic; the extra bandwidth partly compensates for the compute throttle and inflates FPS relative to real hardware." (p. 5)

> "SM–count disparity. Matching sustained TFLOPS via core–clock reduction does not perfectly reproduce differences in SM count, L1/L2 cache sizes, or scheduler granularity. Fewer SMs often translate into poorer warp occupancy and higher register spills, which our emulation ignores." (p. 6)

> "Single efficiency point. We estimate sustained performance with a fixed 66%. Real GPUs vary in terms of FP32 TFLOPs measures compared to their theoretical maxima. Although we chose a conservative estimate of 66%, a per-device efficiency sweep would improve the fidelity of sustained GEMM TFLOPs estimation." (p. 6)

> "6.2 Study-Design Limitations
> • Scene diversity. We benchmark a single "Garden"–style outdoor capture; highly occluded indoor scenes or city-scale captures may stress culling and cache differently.
> • CUDA exclusivity. Results hold only for NVIDIA/CUDA. Porting the gsplat rasterizer to Vulkan or Metal may uncover new bottlenecks (e.g., subgroup ballot vs. CUDA atomics).
> • No network evaluation. Our prototype operates in a client server architecture, but all timings are collected with the LoD layers pre-loaded in GPU memory; we do not measure latency, packet loss, or bandwidth adaptation. Consequently, the results reflect pure client-side rasterization cost." (p. 6)

> "6.3 Future Work
> • Bandwidth-aware emulation. Implement software throttles (e.g. CUDA clamp) to cap DRAM throughput, matching each target GPU's memory bandwidth more closely.
> • Animation compression. Replace per-frame MLP inference with TC-3DGS style blend-splats and time-spline interpolation, reducing both memory traffic and MLP inference overhead on mid-tier hardware.
> • Mobile-SoC feasibility at lower LoD. Our data show that an RTX 3050 sustains only 46 fps for the 0.5M-splat LoD at 1080p; a standalone VR SoC would therefore miss the real-time target at that density. What remains unclear is how far the splat budget or resolution must be reduced (e.g. 50-100k splats, foveated crops, or quarter-HD resolution) before mobile chips become viable, and whether the CUDA-centric rasterizer can be expressed efficiently in Vulkan compute, or Apple Metal. A follow-up study could prototype the core kernels in Vulkan/Metal, then benchmark modern high-bandwidth SoCs (Apple M-series, Snapdragon X Elite) across a grid of LoD budgets and resolutions to chart the "mobile viability envelope" for 3DGS.
> • Resolution & foveation. Extend benchmarks to 1440 p, 4 K, and foveated rendering to map splat density against perceived quality on HMDs." (p. 6)

### 4.8 Sentences touching a target, budget, latency predictor, hardware-awareness, edge or mobile (verbatim, with page)

> "We investigate the feasibility of real-time 3D Gaussian Splatting (3DGS) rasterization on edge clients with varying Gaussian splat counts and GPU computational budgets." (p. 1)

> "Our objective is to explore the practical lower bounds of client-side 3DGS rasterization and assess its potential for deployment in energy-constrained environments, including standalone headsets and thin clients. Through this analysis, we provide early insights into the performance–energy trade-offs that govern the viability of edge-deployed 3DGS systems." (p. 1)

> "We provide the first empirical characterization of real-time 3D Gaussian Splatting performance under constrained GPU compute budgets across multiple levels of scene complexity." (p. 1)

> "A key open question is whether the real-time rasterization component of 3DGS, which is typically offloaded to a high-end GPU, can be feasibly run on lower end consumer edge devices. This is particularly relevant for distributed or hybrid rendering pipelines, where moving some computational load from the cloud to the client can reduce motion-to-photon delay and server-side cost." (p. 1)

> "Kerbl et al. 2024 extend their original system with a Gaussian octree that merges distant splats into coarser parents and refines them on-line when the camera approaches [4]. This yields kilometre-scale scenes that fit into a single-GPU budget while retaining fine detail nearby." (p. 2)

> "Most prior 3DGS studies assume high-end desktop-class GPUs at the client[3][4][7]. No work quantifies how frame-rate scale when hardware is throttled to mid or low-range desktop GPUs. Our study fills this gap." (p. 2)

> "The results provide the first empirical curve relating 3DGS viability to available TFLOPS and bandwidth, informing future ports to Vulkan/Metal on mobile XR SoCs and guiding LOD/streaming system design." (p. 2)

> "Although not an exact replication, this approach implicitly approximates the lower power budgets, memory bandwidths, core frequencies, and compute unit count of the reference GPUs." (p. 3)

> "From these measurements we derive additional energy-aware metrics including energy per frame and performance-per-watt, which help characterize the efficiency of real-time 3DGS rendering under constrained GPU budgets." (p. 3)

> "At present, the 3DGS rendering pipeline, particularly implementations based on CUDA such as gsplat, appears to be constrained to laptop and desktop class GPUs." (p. 3)

> "Our inclusion of the RTX 3050 is thus not meant as a proxy for mobile hardware, but rather to identify the lowest-tier desktop-class GPU on which real-time 3DGS rendering is still achievable using current methods. As such, our study sets a practical lower bound on viable deployment targets for CUDA-based 3DGS pipelines." (p. 3)

> "Tuning the animation budget (e.g. key-frame blend-splats in place of per-frame inference) therefore could be a strategy for regaining performance on lower-tier hardware, albeit with some more memory cost." (p. 4-5)

> "In our experiments LoD reduction is implemented by disabling fine-detail layers." (p. 5, section heading "5.2 LoD Budget vs. Scene Scale")

> "These metrics provide insight into how efficiently real-time 3DGS rasterization utilizes available GPU resources under constrained power budgets." (p. 5)

> "This analysis highlights how energy-aware metrics can complement traditional graphics performance benchmarks when evaluating real-time rendering systems for edge deployment." (p. 5)

> "Overall, real-time 3DGS is already feasible on desktop GPUs down to RTX 3070, provided LoD stays under a million visible splats. Lower-tier GPUs like the RTX 3050 reach usable frame rates only for carefully pruned scenes or when assisted by a server that transmits coarser LoD layers. These findings motivate future work on LoD prediction and bandwidth-aware hierarchies, as well as hybrid client–server rasterization pipelines that shift fine-detail layers to the edge cloud." (p. 5)

> "Despite these limitations, our results delineate a clear performance frontier: desktop-class GPUs down to RTX 3070 sustain interactive frame rates for scenes ≤1 M splats, whereas lower-end GPUs like RTX 3050 require aggressive LoD reduction or cloud assistance. Addressing the bandwidth and animation costs identified above is key to real-world deployment of 3DGS in power-constrained XR clients." (p. 6)

> "These results also suggest that incorporating energy-efficiency metrics alongside frame rate provides a more complete understanding of the trade-offs involved in deploying real-time 3DGS on energy-constrained devices such as edge or mobile platforms." (p. 6)

> "These data give developers and system architects a first, device-agnostic chart of which client GPUs can shoulder 3DGS rasterization unaided and where hybrid or server-side solutions must step in." (p. 6)

> "We hope the simple process: match sustained TFLOPS, log frame rate per LoD, and report the break-even scene size, serves as a common benchmarking framework for upcoming work on making 3DGS truly ubiquitous." (p. 6)

### 4.9 Code and licence
- Code URL: **not stated**. The paper says only "The full set of throttling configurations, measurement scripts, and benchmark settings are described in sufficient detail to enable replication of the GPU capability emulation methodology on other hardware platforms." (p. 3)
- Licence: not stated.

---

## 5. HiGS

### 5.1 Citation
- Title: **HiGS: A Hierarchical Rendering Architecture for Real-Time 3D Gaussian Splatting**
- First author: Dawid Pająk (NVIDIA). Co-authors Martin Bisson, Rodolfo Lima (NVIDIA).
- Venue: **not stated**. Date line "2026-6-2", copyright line "© 2026 NVIDIA. All rights reserved.", stamp `arXiv:2606.00352v1  [cs.GR]  29 May 2026`.
- arXiv id: 2606.00352.

### 5.2 What it does (3 lines)
Decouples the tile size used for binning and depth sorting (a macro-tile of 8×4 render tiles, 64×32
pixels) from the tile size used for rasterisation (8×8 pixels), resolving the tile-size trade-off.
Splits each macro-tile's sorted list into fixed batches of 1,024 Gaussians so that dense macro-tiles
spawn more parallel work units, removing the rasteriser tail effect.
Adds a fully asynchronous four-tier segmented depth-sort cascade over 32-bit depth keys and a
half-precision data path stabilised by Cholesky factors of the conic.

### 5.3 The cost model

**What frame time depends on.** HiGS names the drivers qualitatively and measures each one:
tile-Gaussian pair count, sort key width and entry count, per-tile Gaussian density (the tail
effect), screen resolution, and per-Gaussian projection plus SH evaluation. Statement of the
trade-off (p. 2):

> "Partitioning favors large tiles: a larger spatial cell is less likely to be straddled by any single gaussian's screen-space extent, so the (tile, gaussian) pair count—and every dependent stage of binning, sort, and intersection testing—shrinks with tile area. Rasterization, on the other hand, favors small tiles. As tile area grows, more gaussians overlap each tile and a larger fraction of pixel–gaussian evaluations fall outside the gaussian's effective support, inflating per-pixel work with computations that contribute nothing to the final image. The one-block-per-tile dispatch compounds the imbalance: a few dense tiles dominate frame time."

Work decomposition rule: "a macro-tile holding 𝑛 gaussians spawns ⌈𝑛/𝐵𝑟⌉ units" with B_r = 1,024
(p. 6). Compositing across batches (Eq. 1, p. 9):

    C = C₀ + T₀ · C₁ + T₀T₁ · C₂ + · · ·,   with early termination when ∏[i] T_i < ε

Half-precision Mahalanobis form: q = (l₀ · dx + l₁ · dy)² + (l₂ · dy)², from Σ⁻¹ = L Lᵀ (p. 10).

Memory-footprint rules (explicit, and directly reusable as a rendering-memory model):
- per-batch partial RGBT buffer "occupying 16 KB per batch slot at the default 8 × 8 render-tile size (32 tiles per macro-tile × 64 pixels per tile × 4 RGBT channels × 2 bytes for half precision); the total footprint is 16𝑁𝑅 KB for 𝑁𝑅 macro-tile gaussian batches" (p. 13)
- binning histogram "occupies 2𝑁𝐵𝑁𝑀 bytes" with N_B = ⌈N_g / B_p⌉ partition batches (B_p = 8,192) and N_M macro-tiles, "𝑁𝑀 ≈ 10³ at 1080p, 4 × 10³ at 4K" (p. 12)
- pair list "occupies 8𝑁macro bytes for 𝑁macro macro-tile pairs (4-byte depth key + 4-byte gaussian-id)" (p. 12)
- rasteriser shared memory "∼27 KB per CTA" (p. 11)
- "The resulting footprint scales with the number of macro-tile gaussian batches in the frame—roughly linearly in scene density at fixed resolution" (p. 6)

**Tile-Gaussian pair counts** (Table 1, Mip-NeRF 360, 8×8 render tiles, averaged across test cameras)
— all reported:

| Scene | 1080p gsplat | 1080p AccuTile | 1080p Ours | Reduction | 4K gsplat | 4K AccuTile | 4K Ours | Reduction |
|---|---|---|---|---|---|---|---|---|
| bicycle | 12.0M | 7.56M | 2.15M | 82 % | 31.9M | 15.98M | 2.95M | 91 % |
| bonsai | 6.6M | 4.16M | 0.74M | 89 % | 21.2M | 11.98M | 1.21M | 94 % |
| counter | 10.6M | 6.39M | 1.05M | 90 % | 34.3M | 18.22M | 1.80M | 95 % |
| garden | 11.9M | 8.22M | 2.60M | 78 % | 29.5M | 16.99M | 3.41M | 88 % |
| kitchen | 12.0M | 8.06M | 1.81M | 85 % | 34.8M | 20.65M | 2.69M | 92 % |
| room | 9.1M | 5.31M | 0.77M | 91 % | 30.5M | 16.00M | 1.39M | 95 % |
| stump | 8.4M | 5.39M | 1.45M | 83 % | 22.9M | 12.16M | 2.03M | 91 % |
| Mean | 10.1M | 6.44M | 1.51M | 85 % | 29.3M | 16.00M | 2.21M | 92 % |

**Stage breakdown of frame time** (Section 5.2, RTX PRO 6000 Blackwell, all reported):
- gsplat baseline at 8×8 tiles, 1080p: "splits roughly evenly between partitioning (46–60% of frame time at 1080p) and rasterization (26–33%)"; at 4K "partitioning rises to 61–73%"; "the global radix sort alone consumes 29–46% of total baseline frame time at 1080p and 44–57% at 4K"
- HiGS: "preprocessing-dominated on large scenes (37–42% on bicycle, garden, stump at 5.0–6.1 M gaussians) and rasterization-dominated on small ones (48–51% on bonsai, counter, kitchen, room at 1.2–1.9 M)"
- within HiGS's intersection+sort group: "binning accounts for roughly 60–70 % of that group's time (0.10–0.13 ms at 1080p)"; "sort ... accounts for 30–41 % of intersection+sort time and under 0.08 ms even at 4K"
- SH compression: "the 32-byte SH representation (Section 4.2) cuts the projection kernel's runtime by 23%, which reduces total frame time by 7% at 1080p and 4% at 4K on average across the seven scenes"

**Total frame time in ms** (Table 2, median, RTX PRO 6000 Blackwell):

| Scene | Gaussian# | 1080p 8×8 gsplat | 1080p 8×8 Ours | 1080p 16×16 gsplat | 1080p 16×16 Ours | 4K 8×8 gsplat | 4K 8×8 Ours | 4K 16×16 gsplat | 4K 16×16 Ours |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | 6.1M | 3.51 | 0.76 | 2.81 | 0.89 | 8.13 | 1.05 | 4.88 | 1.16 |
| bonsai | 1.2M | 1.41 | 0.29 | 1.02 | 0.35 | 4.94 | 0.57 | 2.22 | 0.56 |
| counter | 1.2M | 2.22 | 0.36 | 1.38 | 0.43 | 7.36 | 0.68 | 3.35 | 0.72 |
| garden | 5.8M | 3.46 | 0.83 | 2.77 | 0.99 | 7.54 | 1.10 | 4.72 | 1.25 |
| kitchen | 1.9M | 2.66 | 0.49 | 1.80 | 0.59 | 7.90 | 0.82 | 4.07 | 0.90 |
| room | 1.6M | 1.94 | 0.34 | 1.28 | 0.41 | 6.75 | 0.66 | 3.05 | 0.66 |
| stump | 5.0M | 2.24 | 0.55 | 1.87 | 0.62 | 5.84 | 0.87 | 3.20 | 0.84 |
| Mean | — | 2.49 | 0.52 | 1.85 | 0.61 | 6.92 | 0.82 | 3.64 | 0.87 |

**Sort stage in ms** (Table 3, median): global CUB 64-bit sort 0.47-1.19 ms at 1080p and 2.53-4.24 ms
at 4K; CUB segmented 32-bit sort 0.10-0.15 ms at 1080p and 0.21-0.30 ms at 4K; HiGS cascade
0.04-0.07 ms at 1080p and 0.05-0.07 ms at 4K. "the three-tier custom cascade described in Section 4.4
stays below 0.08 ms regardless of scene—a 42–73 × advantage over the global sort."

**The near-linear relation of frame time to Gaussian count, with the numbers** (Section 5.2,
"Scaling with gaussian count", nvcampus park capture, six Gaussian-budget caps from 5 M to 75 M):

> "HiGS leads every scheme at every budget at both 1080p and 4K, and its frame time grows roughly linearly in gaussian count—from 1.25 to 9.97 ms at 1080p and 1.97 to 10.29 ms at 4K—preserving the resolution-stability property of Table 2. The gap against partitioning-bound schemes compresses with scene size: at 1080p the ratio over gsplat drops from 4.2 × at 5 M to 2.9 × at 75 M as per-frame fixed overheads amortize, while 3DGS's 4K frame time grows steeply across the same range (9.85 to 70.67 ms). FlashGS and Faster-GS keep a roughly 2 × gap across the full budget range" (p. 16)

Read as a slope: 5 M → 75 M is a 15× count increase, and HiGS goes 1.25 → 9.97 ms at 1080p (≈ 8.0×)
and 1.97 → 10.29 ms at 4K (≈ 5.2×), so the growth is sub-linear in count with a fixed overhead of
roughly 1 ms (that ratio arithmetic is mine, the ms values are the paper's).

Resolution stability: "mean 0.52 → 0.82 ms (1.6 ×) from 1080p to 4K against the baseline's
2.49 → 6.92 ms (2.8 ×)"; and "HiGS's macro-tile pair count grows only 1.3–1.8 × from 1080p to 4K,
against 2.5–3.4 × for every render-tile scheme".

**No fitted latency model and no prediction error are stated.**

### 5.4 Is the cost constrained or optimised during training?
**No.** HiGS is a forward-only renderer.

> "Forward-only The current implementation covers forward rendering; differentiable training is not addressed in this paper." (p. 18)

No target of any kind is an input. The only budget in the paper is the Gaussian-count cap used to
reconstruct the nvcampus scene at six sizes (5 M, 10 M, 20 M, 30 M, 60 M, 75 M) for the scaling
study, which is a data-generation knob, not a constraint HiGS enforces.

### 5.5 Device numbers
**None on an edge device.** All timings are on one workstation GPU:

> "Hardware is an NVIDIA RTX PRO 6000 Blackwell (SM 120, 188 SMs). The default tile size is 8 × 8; we also evaluate 16 × 16 for tile-agnostic comparison. 32-byte SH compression is enabled for our pipeline." (p. 14)

Timing protocol: "All timings are 100-iteration medians across the full COLMAP test camera split per
scene (16–39 cameras depending on scene)."

Hardware-portability note (p. 18): "The rasterizer's per-block shared-memory footprint (∼27 KB;
Section 4.1) sits well within the ∼100 KB per-SM shared memory of Ampere-class and newer desktop
SKUs, so we expect the core pipeline to run well across commodity GPUs in that range, though we have
not yet characterized performance on Ampere or Ada."

### 5.6 Quality and throughput on the datasets (all reported)
**Only Mip-NeRF 360** (seven scenes). No Tanks & Temples, no Deep Blending.

Throughput in FPS (Table 4, 1000 / median ms; each prior scheme at its fastest configuration,
16×16 tiles, single tile-level depth ordering, no per-pixel sort queues, no antialiasing):

| Scene | #Gauss | Ours 1080p | Ours 4K | FlashGS 1080p | FlashGS 4K | Faster-GS 1080p | Faster-GS 4K | TC-GS 1080p | TC-GS 4K | Speedy-Splat 1080p | Speedy-Splat 4K | StopThePop 1080p | StopThePop 4K | gsplat 1080p | gsplat 4K | 3DGS 1080p | 3DGS 4K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 6.1M | 1316 | 949 | 582 | 508 | 599 | 454 | 468 | 356 | 415 | 296 | 374 | 218 | 356 | 205 | 163 | 60 |
| bonsai | 1.2M | 3460 | 1745 | 1543 | 1121 | 1471 | 725 | 1443 | 839 | 1156 | 608 | 1036 | 481 | 981 | 448 | 557 | 167 |
| counter | 1.2M | 2809 | 1471 | 1212 | 746 | 1302 | 754 | 1242 | 643 | 925 | 456 | 789 | 317 | 725 | 298 | 343 | 110 |
| garden | 5.8M | 1199 | 907 | 591 | 480 | 559 | 426 | 466 | 336 | 418 | 278 | 375 | 225 | 360 | 212 | 222 | 94 |
| kitchen | 1.9M | 2049 | 1214 | 852 | 541 | 931 | 568 | 916 | 488 | 679 | 343 | 615 | 265 | 554 | 246 | 295 | 106 |
| room | 1.6M | 2967 | 1515 | 1466 | 907 | 1353 | 773 | 1232 | 708 | 934 | 497 | 809 | 348 | 782 | 328 | 331 | 104 |
| stump | 5.0M | 1818 | 1147 | 912 | 796 | 921 | 626 | 682 | 492 | 625 | 407 | 560 | 341 | 528 | 304 | 371 | 140 |
| Mean | — | 1937 | 1214 | 893 | 670 | 897 | 588 | 765 | 499 | 643 | 385 | 573 | 293 | 541 | 275 | 286 | 102 |
| Speedup of ours | — | — | — | 2.17 | 1.81 | 2.16 | 2.06 | 2.53 | 2.43 | 3.01 | 3.16 | 3.38 | 4.14 | 3.58 | 4.42 | 6.77 | 11.86 |

Strongest two baselines: **FlashGS** (mean 893 FPS at 1080p, 670 at 4K) and **Faster-GS** (897 / 588).

Image quality (Table 5, means; left vs the fp32 gsplat baseline, right vs COLMAP test images at
factor-4 native resolution):

| Method | vs gsplat PSNR | SSIM | LPIPS | vs GT PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| Ours w/o SH comp. | 67.03 | .9999 | .0000 | 27.68 | .8649 | .1034 |
| Ours w/ SH comp. | 55.59 | .9995 | .0007 | 27.67 | .8645 | .1034 |
| FlashGS | 49.74 | .9983 | .0010 | 27.65 | .8633 | .1046 |
| Faster-GS | 73.86 | 1.0000 | .0000 | 27.68 | .8649 | .1035 |
| Speedy-Splat | 94.43 | 1.0000 | .0000 | 27.68 | .8649 | .1034 |
| StopThePop | 94.37 | 1.0000 | .0000 | 27.68 | .8649 | .1034 |
| 3DGS | 75.04 | 1.0000 | .0000 | 27.68 | .8649 | .1034 |
| gsplat | — | — | — | 27.68 | .8649 | .1034 |

Teaser (p. 1, Fig. 1): "HiGS renders a 5.8M-gaussian scene at 1,124 FPS (1080p)", with gsplat's
per-tile peak Gaussian count 2,485 and HiGS's per-work-unit peak dropping to ∼340. Headline:
"HiGS renders up to ∼15.8× faster than the original 3DGS".

### 5.7 Stated limitations and future work (verbatim)

> "Dataset scope Results span seven Mip-NeRF 360 [1] scenes (1.2–6.1 M gaussians) and the larger nvcampus capture (up to ∼75 M gaussians), covering both the standard small-scene benchmark and the multi-tens-of-millions regime relevant to production capture. Within this range the macro-tile schedule's load-balancing property holds without scene-specific tuning, and the pipeline's per-stage costs (Table 2) extrapolate roughly linearly in gaussian count; we have not validated on scenes outside this scale. A separate dimension we do not exercise is feature-parity comparison with per-pixel sort queues or antialiasing enabled across schemes." (p. 18)

> "Forward-only The current implementation covers forward rendering; differentiable training is not addressed in this paper. Most of the architecture—macro-tile decomposition, per-macro-tile sort segments, batched rasterization with sparse active masks—carries over to a backward pass essentially unchanged, but three pieces of bookkeeping change shape. First, the post-blend reduction that composites partial RGBT slots into a single per-pixel value (Section 4.5) becomes a prefix scan: each batch's backward step needs the cumulative RGB and T before that batch as starting state, which the forward pass currently produces and discards. Second, gradient accumulation needs to be re-introduced in fp32 even though forward keeps the per-pixel state in fp16; the algebraic stability of the Cholesky form (Section 3.5) carries over, but the division by (1 −𝛼) that reverses the alpha-blend amplifies fp16 rounding by up to 𝛼−2 near saturation and is the natural place for selective upcasting. Third, the per-pixel last-contributing-gaussian index that backward uses for early termination [14] becomes a per-(pixel, batch) index in our scheme, since transmittance can saturate inside any batch rather than at a single ordered position. None of these requires changing the macro-tile decomposition itself, and we expect the extension to be additive." (p. 18)

> "Hardware specificity The architecture itself is hardware-agnostic (Section 3), and its numerical design rests on packed half-precision (half2) storage and arithmetic, available on NVIDIA desktop GPUs since Volta and Turing. The rasterizer's per-block shared-memory footprint (∼27 KB; Section 4.1) sits well within the ∼100 KB per-SM shared memory of Ampere-class and newer desktop SKUs, so we expect the core pipeline to run well across commodity GPUs in that range, though we have not yet characterized performance on Ampere or Ada. The one launch-time dependency on recent hardware is the PDL-coordinated kernel chain (Section 4.4), which uses Hopper-class programmatic dependent launch; on architectures without it the same schedule can be expressed with conventional launches at some coordination cost." (p. 18-19)

> "Complementary techniques Several published optimizations operate at orthogonal pipeline stages: training-time model compression [3], Faster-GS's backward-pass acceleration [9], and SplatShop's interactive editing [33] apply before, after, or alongside our rendering kernel, and large-scale and distributed schemes [15, 19] compose with renderer-side acceleration on a different scaling axis. FlashGS's [5] software-pipelined fetch loop could be adopted in our per-batch rasterizer to overlap memory and compute; tensor-core blending [21] would primarily help in a math-bound regime, which is not where our profile sits. Per-pixel depth-ordering [32] is not directly composable: it would require materializing the per-pixel sort queues our macro-tile scheme deliberately avoids." (p. 19)

> "Extending HiGS to differentiable training is a natural next step; the macro-tile decomposition carries over to the backward pass, with the additional per-batch bookkeeping discussed in Section 6." (p. 19)

### 5.8 Sentences touching a target, budget, latency predictor, hardware-awareness, edge or mobile (verbatim, with page)

HiGS contains no "edge", "mobile", "latency predictor" or "hardware-aware" sentence. The
budget-touching and hardware-constraint sentences are:

> "Taming 3DGS [22] replaces 3DGS's gradient-driven densification with a score-guided procedure under a user-specified budget, producing 4–5× smaller models at comparable quality." (p. 3)

> "we evaluate on seven scenes from the Mip-NeRF 360 dataset [1] ... and on the nvcampus park capture for scaling experiments at six gaussian-budget caps from 5M to 75M (Fig. 5)." (p. 14)

> "Figure 5 extends the comparison to the nvcampus park capture across six gaussian-budget caps from 5 M to 75 M. HiGS leads every scheme at every budget at both 1080p and 4K, and its frame time grows roughly linearly in gaussian count" (p. 16)

> "FlashGS and Faster-GS keep a roughly 2 × gap across the full budget range, mirroring the Mip-NeRF 360 finding above." (p. 16)

> "Figure 5: Cross-scheme scaling on the nvcampus park scene reconstructed at six gaussian-budget caps (5M–75M)." (p. 17)

> "The macro-tile decomposition trades memory for parallelism. Per-macro-tile gaussian lists, per-batch partition histograms, and per-batch partial blending buffers are all preallocated up front so that binning batches, sort segments, and rasterization processing units run concurrently without dynamic allocation or producer–consumer stalls. The resulting footprint scales with the number of macro-tile gaussian batches in the frame—roughly linearly in scene density at fixed resolution—and is an explicit consequence of replacing 3DGS's serial-by-tile model with a density-proportional schedule." (p. 6)

> "A separate hardware line resolves the trade-off architecturally—by sorting at coarser-than-tile granularity while preserving exact compositing—but realizes the design on custom hardware datapaths [13, 17]." (p. 2)

> "The authors report, however, that on a GPU its per-gaussian bitmask cannot be generated concurrently with the sort, leaving preprocessing slower than the baseline and motivating their dedicated accelerator." (p. 4, on GS-TG)

> "HiGS demonstrates that design directions recently explored for custom hardware pipelines translate efficiently to commodity GPUs." (p. 5)

### 5.9 Code and licence
- Code URL: **not stated**. The acknowledgement thanks "David Lesage, Vincent Caux-Brisebois and Eric Shangguan for their help in integrating HiGS into gsplat" but prints no repository.
- Licence as printed: "© 2026 NVIDIA. All rights reserved." (p. 1)

---

## 6. AdaGScale

### 6.1 Citation
- Title: **AdaGScale: Viewpoint-Adaptive Gaussian Scaling in 3D Gaussian Splatting to Reduce Gaussian-Tile Pairs**
- First author: Joongho Jo (Korea University, Seoul). Co-authors Hyerin Lim, Hanjun Choi, Jongsun Park.
- Venue: **not stated in the PDF** (no conference line, no arXiv stamp visible in the extracted text). File name indicates DAC 2026.
- arXiv id: 2604.18980 (from the file name).

### 6.2 What it does (3 lines)
Observes that tiles intersecting the peripheral region of a Gaussian contribute almost nothing to
pixel colour, and defines a peripheral score as the summed contribution over the annulus between the
default alpha threshold and an adjustable threshold.
Derives a closed-form per-Gaussian alpha threshold Th_i from depth and 2D covariance so the score can
be computed during preprocessing, then shrinks each Gaussian's effective extent for the tile
intersection test only.
Keeps the original extent during colour accumulation (dual-size strategy), so no retraining is
needed, and the whole method plugs into 3D-GS, GaussianSpa, CityGaussian and Scaffold-GS.

### 6.3 The cost model

**What frame time depends on** (p. 2, NVIDIA A6000, six scenes Train, Truck, Drjohnson, Playroom,
Bicycle, Counter):

> "Among these stages, the Gaussian-tile pair generation, sorting, and rasterization processes account for approximately 88.1% of total runtime on average. Since their computational workloads are largely determined by the number of Gaussian-tile pairs, reducing the pair count is a key factor in accelerating 3D-GS rendering."

That is the paper's cost model: frame time ≈ f(number of Gaussian-tile pairs), with preprocessing the
remaining ≈ 11.9 %. Fig. 2 shows the per-scene normalised split into preprocessing, Gaussian-tile
pair generation, sorting and rasterization, but the per-stage percentages are not printed as text.

**The contribution and peripheral-score equations** (Eqs. 3-5, 9, 10):

    α_i(P_k) = σ_i · exp( −½ (P_xy_k − G_xy_i)ᵀ G_cov_i⁻¹ (P_xy_k − G_xy_i) )      (1)
    PixelColor(P_k) = ∑[i=1..N_Pk] α_i(P_k) · T_i(P_k) · c_i,  with α_i(P_k) ≥ τ    (2)
    Contribution(P_k, G_i) = α_i(P_k) · T_i(P_k)                                    (3)
    PS(G_i, x) = ∑[τ ≤ α_i(P_k) < x] Contribution(P_k, G_i)                         (4)
    Th_i = max { x | PS(G_i, x) ≤ K }                                               (5)

Three approximation steps reduce PS to a closed form:

    PS(G_i, x) ≈ max(T_i(P_k)) · ∑[τ ≤ α_i(P_k) < x] α_i(P_k)                       (6)
    PS(G_i, x) ≈ T_Upper_i(depth) · ∑[τ ≤ α_i(P_k) < x] α_i(P_k)                    (7)
    PS(G_i, x) ≈ T_Upper_i(depth) · ∫[τ ≤ α_i(P_k) < x] α_i(P_k) dP_k               (8)
    PS(G_i, x) ≈ T_Upper_i(depth) · 2π √(det(G_cov_i)) · (x − τ)                     (9)
    Th_i = K / ( T_Upper_i(depth) · 2π √(det(G_cov_i)) ) + τ                        (10)

with τ = 1/255 and K the user-defined acceptable peripheral loss. T_Upper_i(depth) is a piecewise
constant upper bound on max(T_i) over 20 uniform intervals of the depth range [0, 100], calibrated
offline from 16 sampled training viewpoints and stored in a GPU look-up table.

**Approximation quality, stated as PSNR loss at a given skip ratio** (the closest thing to a
prediction error in this paper, all reported):
- exact contribution, skip 80 % of accumulations: "the PSNR decreases only by an average of 0.60 dB" (p. 2); per-scene deltas printed in Fig. 4(a): −0.59, −0.92, −0.47, −0.42
- max(T_i) approximation, skip 60 %: "the average PSNR decreases by merely 0.41 dB" (p. 4); Fig. 5(a) deltas −0.36, −0.72, −0.32, −0.24
- T_Upper_i(depth) approximation, skip 50 %: "the average PSNR decreases by about 0.51 dB" (p. 4); Fig. 5(b) deltas −0.69, −0.81, −0.39, −0.14

**Latency breakdown for the MLP-based pipeline** (Fig. 8(e), Scaffold-GS on Drjohnson, all reported):
"The MLP operations for Gaussian generation and preprocessing operations for voxel-level culling,
which are absent in the original 3D-GS rendering, account for 38% of the total latency. The actual
rendering stage accounts for 62%, and when AdaGScale is applied, considering only the rendering
portion, it achieves a 1.65× speedup at 0.3 dB PSNR drop." (p. 6). Normalised latency bar values
printed in the figure: 1.0 baseline (0.62 render + 0.38 MLP+Pre.), then 0.42, 0.39, 0.37 render
against a constant 0.38 MLP+Pre.

**Gaussian-tile pair reduction versus FlashGS** (Table IV, %, all reported):

| model | target 0.1 dB | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| Original 3D-GS | 31.03 | 35.77 | 38.88 | 41.19 | 43.07 |
| GaussianSpa | 20.20 | 24.15 | 26.91 | 29.09 | 30.93 |
| CityGaussian | 19.88 | 25.33 | 28.88 | 31.54 | 33.64 |

**No fitted latency model, and no latency prediction error, is stated.**

### 6.4 Is the cost constrained or optimised during training?
**No training at all.** "AdaGScale can be seamlessly integrated into existing 3D-GS pipelines without
retraining or fine-tuning" (p. 1). K and T_Upper_i are calibrated offline once per scene.

**But the paper does expose a target as an input, and it reports the achieved-versus-requested
table.** The target is a *quality* target (PSNR drop), never a time or FPS target:

> "For 𝐾, binary search is used with a target PSNR degradation threshold, completing in approximately 10-30 seconds per scene. This efficiency arises from the proportional relationship: larger 𝐾 eliminates more Gaussian-tile pairs, proportionally increasing PSNR loss." (p. 5)

**Every achieved-versus-requested pair** (Table III, "AVERAGE ACTUAL PSNR DROP FOR EACH GAUSSIAN
MODEL", requested target PSNR drop in dB across the top, achieved actual PSNR drop in the body) —
all reported:

| model | target 0.1 | target 0.2 | target 0.3 | target 0.4 | target 0.5 |
|---|---|---|---|---|---|
| Original 3D-GS | 0.053 | 0.104 | 0.157 | 0.210 | 0.264 |
| GaussianSpa | 0.083 | 0.165 | 0.244 | 0.321 | 0.398 |
| Scaffold-GS | 0.094 | 0.182 | 0.268 | 0.351 | 0.434 |
| CityGaussian | 0.082 | 0.165 | 0.248 | 0.331 | 0.415 |

> "As can be seen from the results, the actual PSNR drop does not exceed the target value in all cases." (p. 5)

Note the calibration is conservative rather than tight: the achieved drop is 47-53 % of the requested
drop for original 3D-GS and 82-87 % for the other three models (that ratio arithmetic is mine).

For the speedup evaluation the paper re-fits K on the test set to remove this calibration error:
"to ensure fair comparison of pure speedup gains at exactly identical PSNR drop levels (e.g., -0.5
dB) on the test dataset, we fine-tune 𝐾 using the test dataset for this evaluation." (p. 5)

### 6.5 Device numbers
**None on an edge device.** All experiments are on a single **NVIDIA A6000** ("All experiments are
conducted on an NVIDIA A6000 GPU. The code is compiled using GCC 11.4.0 and NVCC from CUDA 11.6.",
p. 5). The paper's motivating device numbers, both reported:

> "For instance, even on Nvidia's server-grade A6000 GPU, the original 3D-GS implementation [1] achieves only around 10-15 frames per second (FPS) when rendering 4608×3456-resolution images. This is far below the real-time target of 90-120 FPS demanded by Apple Vision Pro, which features binocular displays with a combined resolution of 2×(3660×3200)." (p. 1)

### 6.6 Quality on the datasets (all reported)
Datasets: Tanks&Temples, DeepBlending, Mip-NeRF360, Mill-19, UrbanScene3D. Resolutions (Table I):
T&T 980×545, DB 1332×876 and 1264×832, Mip360 1245×825 to 1559×1039, M19 4608×3456, US3D 5472×3648
and 4864×3648.

Baseline PSNR (Table II), Original 3D-GS: Train 21.79, Truck 24.97, Drjohn. 28.92, Playroom 29.95,
Bicycle 25.14, Bonsai 31.92, Counters 28.84, Flowers 21.46, Garden 27.19, Kitchen 30.67, Room 31.28,
Stump 26.58, Treehill 22.34.
GaussianSpa: 21.41, 25.36, 29.43, 30.50, 25.31, 31.23, 28.67, 21.63, 27.07, 30.93, 31.29, 27.10, 23.00.
Scaffold-GS: Train 22.21, Truck 25.80, Drjohn. 29.72, Playroom 30.81.
CityGaussian: Building 19.48, Rubble 22.01, Residence 20.17, Sci-Art 20.92.

Speedups at a 0.5 dB PSNR drop (Fig. 8, geometric means, text-stated, all reported):
- original 3D-GS model: **5.46×** over the original 3D-GS pipeline, **1.84×** over FlashGS
- GaussianSpa model: **4.92×** over original 3D-GS, **1.50×** over FlashGS
- CityGaussian (city-scale, four scenes): **13.8×** over original 3D-GS, **1.98×** over FlashGS
- Scaffold-GS, at a 0.3 dB drop: **1.26×** geometric mean, and 1.65× counting only the rendering portion

Figure 8 also carries the bar labels 3.18×, 2.60× and 2.76× on the Geo. Mean groups of panels (a),
(b) and (c). Their assignment is **not stated in the text**, so they are recorded here without an
interpretation.

**No FPS or ms value is reported anywhere in the paper**, only speedup ratios. No Gaussian counts or
model sizes are reported either. The strongest two baselines are **FlashGS** (the direct competitor,
speedups above) and **GaussianSpa** ("a pruned model that represents 3D scenes with approximately 8×
fewer Gaussians on average compared to original 3D-GS", p. 5), used as a base model rather than as a
speed baseline.

### 6.7 Stated limitations and future work
**Not stated.** The paper has no limitations section and no future-work section. The conclusion
(p. 6) restates the contributions only.

### 6.8 Sentences touching a target, budget, latency predictor, hardware-awareness, edge or mobile (verbatim, with page)

> "Despite these advantages, 3D-GS still falls short of achieving the real-time rendering performance required for 3D computer vision applications such as augmented and virtual reality (AR/VR). For instance, even on Nvidia's server-grade A6000 GPU, the original 3D-GS implementation [1] achieves only around 10-15 frames per second (FPS) when rendering 4608×3456-resolution images. This is far below the real-time target of 90-120 FPS demanded by Apple Vision Pro, which features binocular displays with a combined resolution of 2×(3660×3200)." (p. 1)

> "Recently, the methods for reducing the number of Gaussian-tile pairs [11]–[13] have attracted attention as GPU-friendly approaches that require no retraining, can be easily integrated with other optimization techniques." (p. 1)

> "To control the trade-off between rendering speed and quality, we introduce 𝐾, a user-defined parameter representing the acceptable peripheral loss." (p. 3)

> "Both 𝐾 and 𝑇𝑈𝑝𝑝𝑒𝑟 𝑖 are determined offline using 16 viewpoints sampled from the training dataset. For 𝐾, binary search is used with a target PSNR degradation threshold, completing in approximately 10-30 seconds per scene. This efficiency arises from the proportional relationship: larger 𝐾 eliminates more Gaussian-tile pairs, proportionally increasing PSNR loss." (p. 5)

> "Although AdaGScale determines the PSNR drop through a predefined 𝐾, as explained in subsection 3.4, 𝐾 is determined using the training dataset, so the actual PSNR drop on the test dataset may differ from the target PSNR drop. Therefore, we analyze the actual PSNR drop according to the target PSNR drop." (p. 5)

> "As can be seen from the results, the actual PSNR drop does not exceed the target value in all cases." (p. 5)

> "As shown in the previous subsection, when 𝐾 is determined using the target PSNR drop on the calibration dataset, the actual PSNR drop on the test dataset exhibits some deviation from the target value. This deviation can vary across different scenes, potentially introducing confounding factors when comparing speedups." (p. 5)

> "Therefore, to ensure fair comparison of pure speedup gains at exactly identical PSNR drop levels (e.g., -0.5 dB) on the test dataset, we fine-tune 𝐾 using the test dataset for this evaluation. This guarantees that all compared methods operate at the same quality level, eliminating the influence of 𝐾 calibration errors." (p. 5)

> "Fig. 8 (e) shows the latency breakdown with and without AdaGScale applied to Scaffold-GS on the Drjohnson scene." (p. 6)

No sentence contains "budget", "latency predictor", "hardware-aware", "edge" or "mobile".

### 6.9 Code and licence
- Code URL: **not stated**. "AdaGScale is implemented using CUDA programming by modifying the author-released codes from FlashGS [13] and Scaffold-GS [19]." (p. 5)
- Licence: not stated.

---

## 7. RoofGS

### 7.1 Citation
- Title: **RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting**
- First author: Yang Luo (State Key Laboratory of Robotics and Systems, Harbin Institute of Technology). Co-authors Yan Gong, Yongsheng Gao (corresponding), Jie Zhao (Senior Member, IEEE).
- Venue: **not stated** (IEEE two-column preprint format, no journal or conference line). Stamp: `arXiv:2608.15785v1  [cs.CV]  16 Aug 2026`.
- arXiv id: 2608.15785.
- **Caveat on this version:** the PDF jumps from Section IV (Methods, p. 6) straight to Fig. 3, Table II and Section V (Conclusion) on p. 7. There is **no experiments section, no experimental-setup section and no per-dataset results table**, and three cross-references print as "Appendix ??". Treat this as an incomplete preprint.

### 7.2 What it does (3 lines)
Runs a stage-wise Roofline characterisation of the 3DGS pipeline to decide, per stage, whether to cut
memory traffic or to cut instruction count.
For the memory-bound front end: INT8 per-degree per-channel SH quantisation, FP16 rotation/scale/
opacity, a resolution-adaptive 32-bit sort key packing tile index and quantised depth, and fusion of
preprocessing, duplication and key generation into one kernel.
For the compute-bound rasteriser: coefficient folding with register-resident coordinate terms,
dual-pixel incremental evaluation, and a Schraudolph bit-level exponential valid on the culled
exponent range, with a derived per-pixel colour error bound.

### 7.3 The cost model

**The Roofline bound** (Eq. 1, p. 3):

    P = min(P_peak, BW · I),      I = F / B  (arithmetic intensity, FLOPs/Byte)

**The knee point on the target GPU** (Eq. 2, p. 3):

    I_knee = 82.6 TFLOPS / 1008 GB/s ≈ 81.9 FLOPs/Byte      (NVIDIA RTX 4090)

> "Stages far below this point are generally more sensitive to memory traffic, whereas stages closer to it benefit more from compute optimizations." (p. 3)

**Per-stage arithmetic intensity, which stage is memory-bound and which is instruction-bound:**

*Preprocessing and attribute evaluation — memory-bound* (Eqs. 3-4, p. 3):

    D_in_pre = 44 + 16 × 3 × 4 = 236 Bytes/Gaussian
    I_in_pre = 400–800 / 236 ≈ 1.69–3.39 FLOPs/Byte

> "Point projection, covariance construction and transformation, SH evaluation, and screen-space extent computation require an estimated 400–800 FLOPs per Gaussian." … "The estimated arithmetic intensity is far below the knee point of 81.9 FLOPs/Byte defined in Eq. (2), suggesting that preprocessing is memory-bound, consistent with the observations reported in GCC and GScore [12], [13]." (p. 3)

*Duplication and key-value generation — memory-bound* (Eq. 5, p. 4):

    B_dup ≥ 12 M Bytes,      M = average number of intersected tiles

> "Key generation requires minimal computation, but reading Gaussian attributes and writing key–value pairs incur substantial global memory traffic. Since the number of generated entries grows with the number of tiles overlapped by each Gaussian, larger footprints can further increase memory traffic." (p. 4)

*Radix-based depth sorting — memory-bound* (Eq. 6, p. 4):

    B_sort ∝ S_kv · N_dup · N_pass

> "Because each pass performs limited computation but accesses the full array, GPU radix sort is often memory-bound [12], [47]. Smaller entries and fewer passes can therefore reduce sorting traffic." (p. 4)

*Rendering / rasterisation — instruction (compute) bound* (Eq. 7, p. 4):

    α(x) = o exp( −½ (x − μ′)ᵀ Σ′⁻¹ (x − μ′) )

> "Evaluating the exponential function introduces substantial instruction overhead, as a single IEEE-754 compliant single-precision exp operation typically expands into 15 to 20 GPU assembly (SASS) instructions [48], [49]. Additionally, each Gaussian–pixel contribution is evaluated and composited sequentially within a thread. Prior roofline analyses [13], [43], [44] have also shown that the rasterization stage of 3DGS is compute-bound." (p. 4)

**Quantitative traffic reductions claimed** (all reported):
- attribute footprint per Gaussian: "this reduces the per-Gaussian attribute footprint from 236 to approximately 96 bytes, a 59.3% reduction" (p. 4)
- sort key: 64 bits → 32 bits, "cuts N_pass from 8 to 4" (p. 5); at 1920×1080 with 16×16 tiles, "the tile grid contains 120 × 68 = 8,160 tiles. Since 2¹² < 8,160 ≤ 2¹³, the tile index requires 13 bits, leaving 19 bits for depth and providing 2¹⁹ = 524,288 quantization levels" (p. 5); quantisation step Δd = (z_far − z_near) / (2^b_d − 1)
- kernel fusion: "reducing front-end global memory transactions from five to two per primitive" (p. 5)
- dual-pixel ILP: "This dual-pixel formulation also halves per-pixel shared-memory traffic, since one attribute load is reused for both vertically adjacent pixels." (p. 6)

**Derived per-pixel error bound of the fast exponential** (Eqs. 23-25, p. 6):

    P(x, y) ∈ [−ln 255, 0] ≈ [−5.541, 0]
    I(P) = ⌊ P · 2²³ · log₂(e) + 2²³ · (127 − σ) ⌋,   exp̂(P) = reinterpret_float(I(P))
    σ = 0.045677,   ε_max ≈ 0.03,   α̂ = α(1 + ε) with |ε| ≤ ε_max
    |ΔC|_∞ ≤ ε_max ≈ 0.03    (in continuous colour space, independent of the number of blended Gaussians)

**Stated prediction error of the Roofline model itself: not stated.** The paper never compares a
predicted stage time against a measured stage time. There is no per-stage ms table and no measured
arithmetic-intensity plot in this version of the PDF.

### 7.4 Is the cost constrained or optimised during training?
**No.** RoofGS is an inference-side implementation framework. No cost term enters any loss, and no
time, FPS or byte target is an input. There are no achieved-versus-requested pairs.

The only forward-looking statement (p. 7): "This perspective may also be applicable to future
optimization of training pipelines and resource-constrained deployment settings."

### 7.5 Device numbers
**None on an edge device.** All reported numbers are on an **NVIDIA RTX 4090**.

Table II, "Orthogonality verification on RTX 4090: FPS and speedup over 3DGS (1080p and 4K)" — all
reported:

| Scene | 3DGS 1080p | 3DGS 4K | RoofGS 1080p | RoofGS 4K | PUP 3D-GS 1080p | PUP 3D-GS 4K | PUP + RoofGS 1080p | PUP + RoofGS 4K |
|---|---|---|---|---|---|---|---|---|
| train | 157 | 46 | 1478 | 530 | 642 | 160 | 3000 | 1217 |
| truck | 167 | 52 | 1348 | 585 | 352 | 98 | 2673 | 997 |
| drjohnson | 131 | 38 | 1294 | 586 | 576 | 158 | 3248 | 1371 |
| playroom | 175 | 53 | 1570 | 699 | 633 | 164 | 3414 | 1408 |
| Mean | 158 | 47 | 1423 | 600 | 551 | 145 | 3084 | 1248 |
| Speedup | 1.0× | 1.0× | 9.0× | 12.7× | 3.5× | 3.1× | 19.6× | 26.4× |

### 7.6 Quality on Mip-NeRF 360 / Tanks&Temples / Deep Blending
Only aggregate claims, because the results section is absent from this version:

> "At 4K resolution on an NVIDIA RTX 4090 GPU, RoofGS increases the average rendering throughput from 61 FPS to 616 FPS across Mip-NeRF 360 [16], Tanks & Temples [17], and Deep Blending [18], delivering about 10× end-to-end speedup with an average PSNR reduction of only 0.028 dB." (pp. 1-2)

Fig. 1 (p. 1) prints the aggregate operating point: 3DGS 61 FPS at 30.875 dB versus RoofGS 616 FPS at
30.847 dB, "(10.1x)" and "(-0.028)".

Fig. 3 (p. 7) gives four single-view PSNR pairs: 3DGS 30.45 dB / RoofGS 30.45 dB, 3DGS 33.38 /
RoofGS 33.33, 3DGS 22.33 / RoofGS 22.32, 3DGS 28.65 / RoofGS 28.65. Scene names are not stated.

Table II gives per-scene FPS on train, truck, drjohnson, playroom (above). **No per-dataset PSNR,
SSIM, LPIPS, Gaussian count or model size table exists in this PDF.** The strongest baseline reported
is **PUP 3D-GS** (mean 551 FPS at 1080p, 145 at 4K), and the second is **3DGS** itself (158 / 47).
The named-but-unmeasured competitors are TC-GS, GEMM-GS, Speedy-Splat, AdR-GS, FlashGS, StopThePop,
SEELE and Mobile-GS.

### 7.7 Stated limitations and future work
**Not stated.** There is no limitations section and no future-work section. The one forward-looking
sentence, verbatim:

> "The results show that coordinating established optimization mechanisms according to stage-specific hardware constraints can provide substantial end-to-end gains for high-resolution 3DGS rendering. This perspective may also be applicable to future optimization of training pipelines and resource-constrained deployment settings." (p. 7)

Three appendix cross-references are broken in this version ("As derived in Appendix ??", "whose
bit-level construction is derived in Appendix ??", "By the error propagation analysis in Appendix
??", p. 6), and the referenced appendices are not present.

### 7.8 Sentences touching a target, budget, latency predictor, hardware-awareness, edge or mobile (verbatim, with page)

> "Through a stage-wise Roofline characterization, we identify two distinct hardware bottlenecks: global memory traffic dominates the front end, whereas instruction throughput limits rasterization." (p. 1)

> "However, its rendering cost depends on factors such as the number of Gaussian primitives, image resolution, and scene complexity. For large-scale scenes or high-resolution rendering, the associated computation, memory usage, and data movement remain substantial, hindering the practical deployment of 3DGS in real-time applications." (p. 1)

> "Some studies, particularly hardware-oriented analyses, have also observed that the stages of the 3DGS rendering pipeline exhibit substantially different execution characteristics [12]–[14]." (p. 1)

> "However, these observations have been explored mainly in hardware-oriented contexts, with limited attention to coordinated software optimization across the whole rendering pipeline." (p. 1)

> "RoofGS uses stage-wise performance characterization to match optimization strategies to the dominant hardware constraints of the baseline pipeline, by reducing intermediate data movement in the front end and reducing arithmetic overhead in rasterization." (p. 1)

> "We present a Roofline-guided stage-wise optimization methodology that allocates optimization effort according to the bottleneck of each stage, rather than applying uniform kernel acceleration." (p. 2)

> "Existing systems such as FlashGS [23], StopThePop [24], SEELE [25], and Mobile-GS [26] optimize dataflow, memory access, and execution scheduling to reduce synchronization overhead and improve data locality. However, these designs are largely empirical and do not explicitly link their optimizations to the distinct hardware limits of each stage, making it difficult to decide how to balance memory-traffic reduction and arithmetic-cost reduction across scenes." (p. 2)

> "Although these paradigms address complementary aspects of the rendering pipeline, limited attention has been given to using a unified stage-wise hardware characterization to guide the allocation of optimization effort between memory-bound and compute-bound stages. This motivates a coordinated end-to-end design." (p. 2)

> "The Roofline model [41], [42] is a principled analytical framework that bounds the attainable performance of an application on a given hardware architecture. By characterizing workloads through arithmetic intensity—the ratio of floating-point operations to memory traffic (FLOPs/Byte)—it establishes an upper performance bound and clearly distinguishes bandwidth-bound from compute-bound regimes." (p. 2)

> "Although recent 3DGS accelerators [43], [44] employ the Roofline model to analyze and optimize individual kernels, few works leverage a quantitative hardware performance model to systematically determine which stages should prioritize memory-traffic reduction versus arithmetic-cost reduction. RoofGS uses the Roofline model as a practical design guide for selecting and integrating stage-aware execution optimizations across the rendering pipeline." (p. 2)

> "The 32-bit budget is dynamically partitioned between the tile index and depth according to the current tile layout." (p. 5)

> "This perspective may also be applicable to future optimization of training pipelines and resource-constrained deployment settings." (p. 7)

No sentence contains "latency predictor", "edge device" or "mobile device" outside the citation of
Mobile-GS and SEELE.

### 7.9 Code and licence
- Code URL: **not stated**.
- Licence: not stated.

---

## 8. Cross-paper synthesis

### 8.1 Consolidated cost-model table

| paper | what frame time (or memory) is said to depend on | fitted / analytic model? | stated prediction error | measurement device |
|---|---|---|---|---|
| MetaSapiens | number of tile-ellipse intersections, not point count. Per-point cost = number of tiles that use the ellipse | No fit. CE_i = Val_i / Comp_i is a per-point cost proxy validated only by matching reduction *rates* in Fig. 4 | not stated | Jetson AGX Xavier mobile Volta GPU, plus a 16 nm RTL accelerator |
| Speedy-Splat | "proportional to both the number of Gaussians in the scene and the number of pixels processed per Gaussian" | No. Words only, plus a measured six-function ms breakdown | not stated | NVIDIA RTX A5000 |
| MEGS² | *rendering VRAM* = static (primitive count × parameters per primitive) + dynamic (projected 2D attributes and the tile-depth-Gaussian key-value table, scaling with visible primitive count) | Constraint ρ_o‖o‖₀ + ρ_s‖s‖₀ ≤ κ with ρ_o = 11, ρ_s = 7 is analytic in *parameters*, not bytes. VRAM itself is measured, not predicted | not stated | RTX 3090 / RTX 4090 for training, WebGL viewer on RTX 3060 laptop, Dimensity 9400+, Snapdragon 8+ Gen 1, Snapdragon 888, Snapdragon 865 |
| Splats under Pressure | sustained FP32 TFLOPS, memory bandwidth, splat count, per-frame animation MLP cost | No. Empirical FPS grid only. Energy metrics defined (E_frame = P_avg / FPS, η_perf = FPS / P_avg) but **no energy value reported** | not stated | one RTX 4090 throttled to emulate RTX 4090 / 4070 Ti / 3070 / 3050 |
| HiGS | tile-Gaussian pair count, sort key width and entry count, per-tile Gaussian density (tail effect), resolution, per-Gaussian projection + SH. Memory: 16·N_R KB partial buffers, 2·N_B·N_M byte histogram, 8·N_macro byte pair list | No fit, but explicit closed-form memory rules and a measured near-linear frame-time-versus-count curve (1.25 → 9.97 ms at 1080p over 5 M → 75 M) | not stated | NVIDIA RTX PRO 6000 Blackwell (188 SMs) |
| AdaGScale | number of Gaussian-tile pairs. "pair generation, sorting, and rasterization ... account for approximately 88.1% of total runtime on average" | Analytic *contribution* model, not a latency model: PS(G_i, x) ≈ T_Upper_i(depth) · 2π √(det Σ2D) · (x − τ), giving Th_i in closed form | For the contribution approximation: 0.41 dB average PSNR loss at 60 % skip with max(T_i), 0.51 dB at 50 % skip with T_Upper(depth). No latency error stated | NVIDIA A6000 |
| RoofGS | Roofline P = min(P_peak, BW·I). Front end memory-bound (I ≈ 1.69-3.39 FLOPs/Byte versus a knee of 81.9), rasterisation instruction-bound (exp = 15-20 SASS instructions) | Analytic per-stage arithmetic intensity and traffic models (D_in_pre = 236 B/Gaussian, B_dup ≥ 12M B, B_sort ∝ S_kv N_dup N_pass) | Roofline error not stated. Fast-exp error is derived: ε_max ≈ 0.03, |ΔC|_∞ ≤ 0.03 | NVIDIA RTX 4090 |

**Nobody in this set fits a latency predictor and reports its accuracy.** Not one of the seven papers
regresses measured frame time against a feature vector and reports a MAPE, an R², or a
predicted-versus-measured plot. The closest thing to a validated cost proxy is MetaSapiens's Fig. 4
(reduction rates of latency and intersections "match", stated qualitatively) and RoofGS's analytic
arithmetic intensities (compared against a knee point, never against a measurement).

### 8.2 Which papers constrain the cost during training, and how

| paper | constrains cost during training? | mechanism | is a time or FPS target an input? |
|---|---|---|---|
| **MEGS²** | **Yes** | L0 constraint ρ_o‖o‖₀ + ρ_s‖s‖₀ ≤ κ on the joint (primitive, lobe) parameter count, solved by ADMM for 10,000 iterations at interval 50, penalty δ = 0.0005, followed by threshold pruning, colour compensation and fine-tuning | No. κ is a parameter-count budget. No mapping from κ to MB or ms is given |
| **MetaSapiens** | **Yes, partly** | Pruning ranks by a compute-aware score CE_i = Val_i / Comp_i (Comp_i is the tile-intersection count), and a scale-decay term γ·WS is added to the training loss (L = L_quality + γ·WS). The iterative prune-retrain loop is stopped by a **quality** threshold on L_quality (PSNR, SSIM or HVSQ) | No. Only a quality target (99 %, 98 %, 97 % of dense PSNR) |
| **AdaGScale** | No training, but a target is an input | K is binary-searched offline (10-30 s per scene, 16 sampled training views) against a **target PSNR degradation**, then converted per-Gaussian and per-viewpoint into Th_i by Eq. 10 at preprocessing time | No. Only a quality target. Achieved drop is 47-87 % of requested (Table III) |
| Speedy-Splat | No | Fixed pruning percentages (80 % soft at iterations 6000/9000/12000, 30 % hard every 3000 from 15000), chosen by sweep | No |
| HiGS | No | Forward-only renderer, "differentiable training is not addressed in this paper" | No |
| RoofGS | No | Inference-side implementation framework | No |
| Splats under Pressure | No | Measurement study over a pre-trained LapisGS LoD hierarchy | No |

**The hardware-aware-NAS analogue does not exist in this set.** No paper builds a differentiable
latency predictor, and no paper puts a measured or predicted milliseconds-per-frame term into a
training loss. MEGS² is the only paper whose *training* is driven by a resource constraint at all,
and that constraint counts parameters. MetaSapiens is the only one whose *pruning score* is a
compute-cost proxy rather than a quality proxy.

### 8.3 Device table (all values reported by the paper named)

| paper | device (as named) | class | result |
|---|---|---|---|
| MetaSapiens | NVIDIA Jetson AGX Xavier, mobile Volta GPU | embedded | prior PBNR models "generally below 10 FPS"; MetaSapiens-H 102.2 FPS, SMFR 125.9 FPS, MMFR 52.6 FPS averaged over all datasets; ablation speedups 1.6× (SD), 5.8× (+CE), 7.4× (+FR) over dense Mini-Splatting-D |
| MetaSapiens | 16 nm FinFET RTL accelerator, 2.73 mm², LPDDR3-1600 ×4 | ASIC | 18.5× geomean (up to 24.8×) over the Jetson GPU baseline; 20.9× (up to 27.7×) with tile merging + incremental pipelining; energy 54.4× and 56.8× reduction |
| MetaSapiens | RTX 4090 workstation + Meta Quest Pro headset | desktop + HMD | both models "render smoothly at 90 FPS" |
| MEGS² | Lenovo laptop, NVIDIA GeForce RTX 3060 Laptop GPU | laptop | Ours 165.0 FPS (Table 9) / 117.4 FPS (Fig. 1); 3DGS 26.3 (Table 9) / 27 (Fig. 1) |
| MEGS² | OnePlus Ace 5 Ultra, MediaTek Dimensity 9400+ | phone | Ours 91.0 FPS, GaussianSpa 31.4, 3DGS 6.6 |
| MEGS² | RedMi K60, Qualcomm Snapdragon 8+ Gen 1 | phone | Ours 120.9 FPS, GaussianSpa render error, 3DGS cannot render |
| MEGS² | Huawei MatePad Air, Qualcomm Snapdragon 888 | tablet | Ours 60.1 FPS, GaussianSpa render error, 3DGS "connot render" |
| MEGS² | Qualcomm Snapdragon 865 (Fig. 1 only) | phone | Ours 34.9 FPS, 3DGS(w SH) N/A |
| MEGS² | NVIDIA RTX 3090 (24 GB) | desktop | 200 FPS on Mip-NeRF360, rendering VRAM 265 MB (Table 4) |
| Splats under Pressure | emulated RTX 3050 (6.12 measured TFLOPs, 150 W emulated) | low-end desktop | 45.8 ± 7.3 FPS at 0.58 M splats, 19.7 ± 1.0 FPS at 3.45 M splats, 1080p |
| Splats under Pressure | emulated RTX 3070 (13.49 TFLOPs, 150 W) | mid desktop | 57.0 ± 4.1 FPS at 0.58 M, 30.2 ± 1.7 FPS at 3.45 M |
| Splats under Pressure | emulated RTX 4070 Ti (26.49 TFLOPs, 285 W) | upper-mid desktop | 58.6 ± 5.3 FPS at 0.58 M, 36.2 ± 2.9 FPS at 3.45 M |
| Splats under Pressure | RTX 4090 (53.58 TFLOPs, 450 W) | flagship | 58.8 ± 6.0 FPS at 0.58 M, 44.8 ± 2.6 FPS at 3.45 M |
| Speedy-Splat | NVIDIA RTX A5000 | workstation | mean 898 FPS at 0.28M Gaussians vs 134 FPS at 2.93M for 3D-GS |
| HiGS | NVIDIA RTX PRO 6000 Blackwell | workstation | mean 0.52 ms at 1080p, 0.82 ms at 4K; 1937 / 1214 FPS |
| AdaGScale | NVIDIA A6000 | server | 3D-GS 10-15 FPS at 4608×3456; speedups only, no absolute FPS reported |
| RoofGS | NVIDIA RTX 4090 | flagship | 61 → 616 FPS at 4K (10.1×); Table II mean 158 → 1423 FPS at 1080p, 47 → 600 at 4K |

Explicitly excluded devices: Splats under Pressure excludes NVIDIA Jetson AGX Orin 64 GB, Qualcomm
XR2 and Apple M2 by design.

### 8.4 Which cost models to reuse for a latency-bounded training phase

**1. MetaSapiens's CE_i, as the per-Gaussian cost attribution.** It is the only per-primitive score in
this set whose denominator is a measured hardware cost (the tile count a Gaussian is actually used
in), and the only paper that shows point count is the wrong knob (Fig. 4). CE_i = Val_i / Comp_i is
already differentiable-adjacent: Comp_i is exactly the quantity our budget must control, and the
companion scale-decay term γ·WS shows how to push it down through the loss rather than through a
pruning step. Its weakness is that it never produces milliseconds, so it needs a calibration layer on
top.

**2. RoofGS's stage-wise Roofline decomposition, as the analytic skeleton of the ms model.** It is
the only paper that writes down per-stage bytes and FLOPs in closed form:
D_in_pre = 44 + 16·3·4 = 236 B/Gaussian, B_dup ≥ 12M B with M the average intersected-tile count,
B_sort ∝ S_kv · N_dup · N_pass, and a device knee I_knee = 82.6 TFLOPS / 1008 GB/s ≈ 81.9 FLOPs/Byte.
That gives a two-term predictor, t ≈ max(bytes/BW, FLOPs/P_peak) per stage, parameterised by
(N, M, N_dup, resolution, SH degree) and by two device constants that a target device can be probed
for. It also settles which stage a budget should attack on which device: front end memory-bound,
rasteriser instruction-bound. Its weakness is that the paper never validates the model against
measurement, so the calibration has to be ours.

**3. HiGS's measured scaling laws, as the validation set and the memory model.** It is the only paper
with a resolution × count × scheme grid dense enough to fit and check a predictor against: pair
counts at 1080p and 4K for seven scenes (Table 1), median frame time for two tile sizes at two
resolutions (Table 2), sort time isolated (Table 3), FPS for seven schemes (Table 4), and the
near-linear count sweep 5 M → 75 M (1.25 → 9.97 ms at 1080p, 1.97 → 10.29 ms at 4K). It also gives
the rendering-memory side in closed form (16·N_R KB, 2·N_B·N_M B, 8·N_macro B), which is what the
storage-budget phase of this project will need to extend into a *runtime* memory budget. Its weakness
is that it is forward-only and single-GPU, so it cannot supply the training hook.

**Honourable mention: MEGS² supplies the optimisation machinery, not the cost model.** The ADMM
formulation min L s.t. ρ_o‖o‖₀ + ρ_s‖s‖₀ ≤ κ, with proxy variables, a proximal projection onto the
budget, and dual updates, is exactly the shape a latency-budgeted training loop needs. Swap the
linear parameter-count constraint for a calibrated latency surrogate t̂(N, M, ...) ≤ t_budget and the
same algorithm carries over. Its cost model itself (parameters, not bytes, not ms) is too coarse to
reuse directly.

**Not recommended as cost models:** Speedy-Splat (cost stated in words, no model), AdaGScale (the
analytic object is a colour-contribution integral, and the only target it accepts is a PSNR drop),
Splats under Pressure (an FPS lookup grid with no model and no energy numbers despite the title).
Speedy-Splat and AdaGScale remain valuable as *mechanisms* that reduce M, the average intersected-tile
count, which is the load-bearing variable in both the MetaSapiens and RoofGS models.

### 8.5 Gaps this set leaves open for our project
- No latency predictor with a stated accuracy exists for 3DGS in this set. Building one and reporting its error would be novel relative to all seven.
- No paper accepts a milliseconds-per-frame or FPS target as a training input. MEGS² accepts a parameter budget, MetaSapiens and AdaGScale accept a quality target.
- Per-pixel depth complexity is discussed qualitatively (HiGS's tail effect, MetaSapiens's per-tile intersection heatmap with a 2,485 peak and a ~340 post-balancing peak) but never enters a cost equation.
- Energy is named in one title and never quantified (Splats under Pressure reports no J/frame and no FPS/W).
- Rendering memory has exactly one paper behind it (MEGS², static plus dynamic, measured through PyTorch) and one closed-form treatment (HiGS's buffer arithmetic). Nobody predicts peak rendering VRAM from model parameters.
