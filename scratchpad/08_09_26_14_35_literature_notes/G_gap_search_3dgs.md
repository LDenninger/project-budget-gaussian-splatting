# G. Gap-collapse search and render-time / runtime-memory line for BudgetGS

Date: 2026-09-08. Scope: (1) try to refute "no published method binds a byte-denominated target during training", (2) map the render-time and runtime-memory line. Every paper below marked verified was checked against the arXiv export API (id, title, first author, date) or, for papers without an arXiv id, against Semantic Scholar / OpenAlex / Crossref by DOI. Abstract paragraphs are condensed from the arXiv summary field. Mechanism details marked "fetched" come from the arXiv HTML full text.

Classification line format: supplies = what the user gives (λ, count, MB, ms, none) / binds = when the budget is enforced (training, post-hoc, inference) / constrains = quantity constrained / gap = yes, partly, no + one sentence.

## 0. Verdict

**Partly refuted.** No paper takes a byte target before training starts and drives densification onto it. Two things the previous session missed:

1. HybridGS (2505.01938, ICML 2025) accepts a target size B in MB and enforces it inside a continued-training phase with an explicit size model R(GS) = n · P_bit / L (n primitives, P_bit bits per primitive from per-attribute bit depths, L the G-PCC lossless ratio). Rate control is interleaved with gradient optimisation (progressive pruning of N_p/F_p primitives per step and bit-depth reduction, training continues until T_u). B is supplied after the top-quality point T_top, and the size is met to within 3 to 14 % (6 MB target gives 5.82 or 5.15 MB, 10 MB target gives 8.59 MB). This is a byte target bound during training, with the caveat that it is a second stage that only shrinks a converged model.
2. Byte-denominated post-hoc size targeting has moved on from SizeGS: MesonGS++ (2604.26799) jointly optimises reserve ratio and per-group bit widths under a target storage budget by 0-1 ILP with a linear size estimator, and RAVE (2512.07052) interpolates any rate between two bounds without retraining.

Count-denominated training-time budgets are now numerous and mature (CDGS, YOGO, Matryoshka, Gaussians on a Diet, ConeGS, SPARE-GS, Smart target point control, ControlGS via a single λ), and continuous-LoD / anytime training (Matryoshka, CLoD-GS, FlexGS) gives one model for many count budgets. Nobody maps a device (bytes of VRAM, ms per frame) to that count inside the training loop.

Render-time line: the only training-time or pruning-time cost model that is not "count" is MetaSapiens' per-point computational efficiency CE_i = Val_i / Comp_i, where Comp_i is the number of tiles that intersect the point's ellipse (fetched). AdaGScale, Speedy-Splat, AdR-Gaussian and QuadBox all reduce Gaussian-tile pairs as the cost driver. HiGS measures frame time growing roughly linearly with Gaussian count (1.25 to 9.97 ms at 1080p for 5 M to 75 M Gaussians on an RTX PRO 6000). Splats under Pressure gives FPS versus splat count per GPU tier (RTX 3050: 45.8 fps at 0.58 M to 19.7 fps at 3.45 M at 1080p). No 3DGS paper cites hardware-aware NAS or builds a latency predictor / lookup table over a target device.

## 1. Search log

Web searches (WebSearch, result links returned):

| # | query | links |
|---|---|---|
| 1 | "Gaussian splatting" "target size" OR "size budget" OR "storage budget" training | 10 |
| 2 | "Gaussian splatting" "rate control" OR "bit allocation" OR "target bitrate" compression | 9 |
| 3 | "Gaussian splatting" "user-specified" size OR "desired size" OR "controllable compression" | 7 |
| 4 | ControlGS Gaussian splatting quantity quality control single hyperparameter across scenes arXiv | 8 |
| 5 | "Gaussian splatting" "budget-aware" OR "resource-constrained" OR "memory-constrained" training | 9 |
| 6 | "Gaussian splatting" "hardware-aware" OR "latency-aware" OR "FPS constraint" training pruning | 10 |
| 7 | "Gaussian splatting" Jetson OR "edge device" OR "on-device" rendering real-time 2025 2026 | 10 |
| 8 | "Gaussian splatting" render time prediction OR "rendering cost model" OR "tile intersections" GPU memory | 9 |
| 9 | "Gaussian splatting" hardware accelerator GSCore GauSPU MetaSapiens rendering cost | 10 |
| 10 | arXiv 2026 "Gaussian splatting" compression rate-distortion | 10 |
| 11 | arXiv 2026 "Gaussian splatting" budget primitives training | 10 |
| 12 | "Gaussian splatting" "bitrate" "constrained" OR "byte budget" OR "memory budget" 2025 | 8 |
| 13 | SteepGS steepest descent density control Gaussian splatting arXiv | 9 |
| 14 | FlashGS EfficientGS "Reduced 3D Gaussian Splatting" Papantonakis EAGLES Girish arXiv | 10 |
| 15 | "Optimized Minimal Gaussians" OMG LocoGS TC-GS NeuralGS ELMGS Gaussian splatting compression arXiv | 9 |
| 16 | Turbo-GS DashGaussian "Mini-Splatting2" fast 3D Gaussian splatting training arXiv | 9 |
| 17 | "Revising Densification" OR "Improving Adaptive Density Control" Rota Bulò ; "efficient density control" EDC | 9 |
| 18 | MetaSapiens foveated ... ; GSCore Lee 2024 accelerator ; GauSPU SLAM co-processor arXiv | 10 + 10 + 9 |
| 19 | Octree-GS "Hierarchical 3D Gaussian" Kerbl CityGaussian LetsGo FLoD "flexible level of detail" arXiv | 9 |
| 20 | AdR-Gaussian "Balanced 3DGS" StopThePop "sort-free" Gaussian splatting rendering acceleration arXiv | 7 |
| 21 | "Textured Gaussians" LiteGS MobileGS EdgeGaussians GaussianLens "CompGS++" Gaussian splatting arXiv | 6 |
| 22 | "Gaussian splatting" "anytime" OR "progressive" OR "scalable coding" OR "level of detail" streaming bandwidth 2025 2026 | 8 |
| 23 | "Gaussian splatting" "quality control" OR "quantity control" OR "single hyperparameter" OR "controllable" primitive count training | 8 |
| 24 | "Gaussian splatting" mobile OR "web browser" OR WebGL OR WebGPU OR smartphone real-time rendering memory footprint 2025 2026 arXiv | 9 |
| 25 | "Gaussian splatting" "neural architecture search" OR "hardware-aware NAS" OR "latency predictor" OR "latency lookup table" | 9 (all NAS papers, none 3DGS) |
| 26 | "Gaussian splatting" "frame rate" OR "FPS" target OR guarantee OR constraint pruning "level of detail" 2026 arXiv | 10 |
| 27 | LocoGS ... ; TC-GS ... ; NeuralGS ... arXiv | 8 + 7 + 9 |
| 28 | ELMGS ... ; DashGaussian "200 seconds" ; "Mini-Splatting2" arXiv | 10 + 10 + 8 |
| 29 | "Balanced 3DGS" OR "LiteGS" OR "FLoD" OR "GaussianLens" Gaussian splatting arXiv id | 7 |
| 30 | "LS-Gaussian" Jetson AGX Orin speedup ; "GSArch" breaking memory barriers arXiv | 8 |
| 31 | RLGS reinforcement learning ... ; "Reducing the Memory Footprint of 3D Gaussian Splatting" Papantonakis | 9 + 10 |
| 32 | "Gaussian splatting" "size-constrained" OR "storage-constrained" OR "compute-constrained" OR "compute budget" training 2025 2026 | 8 |
| 33 | "Gaussian splatting" "target device" OR "target hardware" OR "device-specific" OR "device-aware" training pruning rendering 2025 2026 | 10 |
| 34 | "Gaussian splatting" "efficiency-aware" OR "speed-aware" OR "runtime-aware" OR "cost-aware" pruning Pareto rendering speed | 9 |
| 35 | "Gaussian splatting" "quantization-aware" training "target" bits OR "bit-width" OR "bpp" rate "during training" 2025 2026 | 8 + 10 |
| 36 | HAC++ successor anchor-based Gaussian splatting compression 2026 "HAC" context model new arXiv | 8 |
| 37 | REACT3D MICRO 2025 ... ; LTS DASH streaming multi-layer Gaussian splatting MMSys 2025 | 9 + 8 |
| 38 | "Gaussian splatting" "energy" OR "power" mobile pruning "frames per second" constraint optimization training 2026 | 9 |
| 39 | FLoD "Flexible Level of Detail" 3D Gaussian Splatting customizable rendering arXiv 2408 | 10 |
| 40 | FlexGS "train once, deploy everywhere" flexible 3D Gaussian splatting elastic inference arXiv CVPR 2025 | 9 |
| 41 | GSCore "Efficient Radiance Field Rendering via Architectural Support for 3D Gaussian Splatting" ASPLOS 2024 doi | 9 |
| 42 | "Gaussian splatting" "render budget" OR "time budget" OR "latency budget" OR "frame budget" OR "runtime budget" rendering training | 7 |
| 43 | "Gaussian splatting" "performance model" OR "analytical model" OR "cost model" rendering time "number of Gaussians" tiles sort rasterization | 10 |
| 44 | "Gaussian splatting" "rate-constrained" OR "Lagrangian" OR "constrained optimization" model size bits "during training" entropy | 10 |
| 45 | "Gaussian splatting" "deployment-aware" OR "deployment" "resource budget" OR "hardware budget" OR "hardware constraints" training controllable 2026 | 8 |
| 46 | "Gaussian splatting" AutoOpti3DGS OR Opti3DGS coarse-to-fine frequency modulation Gaussian count arXiv | 9 |
| 47 | "Gaussian splatting" "low-end GPU" OR "integrated GPU" OR "laptop" OR "Raspberry Pi" real-time rendering pruning memory 2025 2026 | 7 |
| 48 | "Gaussian splatting" "once-for-all" OR "slimmable" OR "elastic inference" OR "any budget" single model deploy multiple devices | 9 |
| 49 | "Gaussian splatting" "peak memory" OR "VRAM" rendering "predict" OR "estimate" OR "model" number of Gaussians resolution tile overlap | 10 |

arXiv export API listing sweeps (search_query, sorted by submittedDate desc, first 80 read in full):

| key | query | totalResults |
|---|---|---|
| budget | all:"gaussian splatting" AND all:budget | 67 |
| ratecontrol | all:"gaussian splatting" AND ("rate control" OR "bit allocation" OR bitrate OR "bit rate") | 31 |
| targetsize | all:"gaussian splatting" AND ("target size" OR "size budget" OR "storage budget" OR "memory budget" OR "size-aware" OR "size constraint") | 4 |
| fps_constraint | all:"gaussian splatting" AND ("frame rate" OR latency) AND (constraint OR budget OR constrained) | 25 |
| rendertime_model | all:"gaussian splatting" AND ("render time" OR "rendering time" OR "rendering cost" OR "rendering speed") AND (predict OR model OR estimate) | 165 |
| controllable | all:"gaussian splatting" AND (controllable OR "user-specified" OR "user-defined" OR desired) AND (size OR count OR number OR compression) | 108 |
| edge | all:"gaussian splatting" AND (jetson OR "edge device(s)" OR smartphone OR "mobile device(s)" OR browser) | 64 |
| accelerator | all:"gaussian splatting" AND (accelerator OR "architectural support" OR "hardware acceleration") | 220 |
| hwaware | all:"gaussian splatting" AND ("hardware-aware" OR "latency-aware" OR "resource-constrained" OR "resource constrained" OR "memory-constrained" OR deployment) | 163 |
| lod | all:"gaussian splatting" AND ("level of detail" OR "level-of-detail" OR LOD) | 53 |
| progressive | all:"gaussian splatting" AND (progressive OR scalable OR anytime OR "variable rate" OR "variable-rate") | 564 |
| vram | all:"gaussian splatting" AND (VRAM OR "GPU memory" OR "memory footprint") AND (rendering OR inference) | 55 |

arXiv title searches: ti:GSArch 0, ti:GauSPU 0, ti:GSCore 0, ti:"LS-Gaussian" 0, ti:REACT3D 1 (unrelated articulation paper 2510.11340), ti:GaussianLens 1 (2509.25603, off-topic), ti:"Optimized Feature Planes" 1 (2501.03399).

DOI verifications (Semantic Scholar unless noted): GSCore 10.1145/3620666.3651385 (Junseo Lee, ASPLOS 2024); REACT3D 10.1145/3725843.3756109 (Hongyi Wang, MICRO 2025); LTS 10.1145/3712676.3714445 (Yuan-Chun Sun, MMSys 2025); Multi-frame Bitrate Allocation 10.1145/3672196.3673394 (Yuan-Chun Sun, EMS@SIGCOMM 2024); MetaSapiens 10.1145/3669940.3707227 (ASPLOS 2025); Reduced 3DGS 10.1145/3651282; FLoD 10.1145/3731430 (TOG); RTGS 10.1145/3725843.3756099 (MICRO 2025); GauSPU 10.1109/micro61859.2024.00114 (Lizhou Wu, MICRO 2024, OpenAlex + Crossref); GSArch 10.1109/hpca61900.2025.00037 (Houshu He, HPCA 2025, OpenAlex + Crossref).

Verification script: `verify.py` and `abstracts.py` in this directory, raw outputs `verify_batch*.tsv`, `apisearch_*.tsv`, `abstracts_*.txt`.

## 2. Part 1: gap-collapse candidates (verified)

### 2.1 Byte-denominated targets

**2505.01938 HybridGS: High-Efficiency Gaussian Splatting Data Compression using Dual-Channel Sparse Representation and Point Cloud Encoder.** Qi Yang. ICML 2025. 2025-05-03.
Generates compact explicit 3DGS with a dual-channel sparse representation that supervises primitive positions and per-attribute bit depth during training, then encodes the result with a standard point-cloud codec (G-PCC). A rate control scheme pivots the pipeline. Fetched: user gives a bandwidth B in MB after the top-quality point T_top; R(GS) = n · P_bit / L with P_bit = 3(BD_p + BD_s) + k_c BD_c + BD_o + k_r BD_r; two levers, progressive pruning (N_p / F_p primitives per step) and bit-depth reduction spread evenly over non-position attributes; training continues between steps until T_u. Table 4 target vs actual: 6 MB gives 5.82 / 5.15 MB, 10 MB gives 8.59 MB on train; sizes 0.51 to 30.21 MB across playroom, train, bicycle, room, dance.
supplies = MB / binds = training (second stage after convergence) / constrains = bytes on disk / gap = partly. A byte target bound inside a training loop with an explicit size model, but only as a shrink stage on a converged model and without any device or render-time term.

**2604.26799 MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter Searching.** Shuzhao Xie. 2026-04-29.
Size-aware post-training codec: importance pruning, octree geometry coding, attribute transform, selective VQ of higher SH, group-wise mixed-precision quantisation and entropy coding. Treats reserve ratio and bit-width allocation as the dominant RD knobs and optimises them under a target storage budget via discrete sampling and 0-1 ILP with a linear size estimator and a CUDA quantisation operator. Over 34× compression, "accurately meeting target size budgets".
supplies = MB / binds = post-hoc / constrains = bytes on disk / gap = no (post-hoc), but the linear size estimator and ILP allocation are directly reusable inside a training loop. Extends SizeGS and MesonGS (same first author).

**2512.07052 RAVE: Rate-Adaptive Visual Encoding for 3D Gaussian Splatting.** Hoang-Nhat Tran. 2025-12-07.
Flexible compression that supports interpolation at any rate between predefined bounds, no retraining per rate, dynamic rate control for bandwidth and device constraints.
supplies = rate between bounds / binds = post-hoc / constrains = bytes / gap = no.

**2607.14513 Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding.** Amir Said. 2026-07-16.
Compresses SH coefficients with BC1/BC7 texture formats decoded by GPU hardware, with a bit-rate control strategy that preserves random access.
supplies = bit rate / binds = post-hoc / constrains = bytes and GPU decode / gap = no.

**2510.22812 Region-Adaptive Learned Hierarchical Encoding for 3D Gaussian Splatting Data.** Shashank N. Sridhara. 2025-10-26.
Overfits hierarchical latents per attribute on the octree of the voxelised geometry "under a global rate constraint", autoregressive entropy model estimates bitrate during training of the codec. Up to 2 dB gain below 1 MB.
supplies = λ / binds = post-hoc codec training / constrains = bytes / gap = no.

**2605.11427 PD-4DGS: Progressive Decomposition of 4D Gaussian Splatting for Bandwidth-Adaptive Dynamic Scene Streaming.** Jiachen Li. 2026-05-12.
Three independently transmittable layers, Gaussian-entropy attribute RD loss with bit rate as an explicit loss variable, capacity-weighted rollout gated by a learnt activation rate. One training run gives a DASH/HLS-compatible scalable bitstream, first-frame latency 1.7 s at 2 Mbps.
supplies = λ / binds = training / constrains = bytes (streamed) / gap = no (λ, dynamic scenes).

**2512.17528 Voxel-GS: Quantized Scaffold Gaussian Splatting Compression with Run-Length Coding.** Chunyang Fu. DCC 2026. 2025-12-19.
Differentiable quantisation of Scaffold-GS attributes with a Laplacian rate proxy as entropy constraint during training, then octree plus run-length coding. The proxy "accurately estimates the bitrate of run-length coding".
supplies = λ / binds = training / constrains = bytes / gap = no.

**2607.12656 SpeedyGS: Content-Aware 3D Gaussian Splatting Compression via Two-Stage Optimization.** Junteng Zhang. 2026-07-14.
Joint adaptive quantisation and pruning under a unified RD objective where the rate term is a lightweight proxy of the next-stage entropy-coding cost, then octree tokens and a complexity-controllable autoregressive coder. Up to 160× reduction, 9× faster optimisation.
supplies = λ / binds = training / constrains = bytes / gap = no.

**2601.12814 CSGaussian: Progressive Rate-Distortion Compression and Segmentation for 3D Gaussian Splatting.** Yu-Jen Tseng. WACV 2026. 2026-01-19.
Joint RD compression and segmentation with an INR hyperprior, quantisation-aware training.
supplies = λ / binds = training / constrains = bytes / gap = no.

**2501.03399 Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs.** Soonbin Lee. 2025-01-06.
Progressive tri-plane features coded with standard video codecs, frequency-domain entropy model and channel-wise bit allocation.
supplies = λ / binds = training / constrains = bytes / gap = no.

**2608.28272 Non-Uniform Quantisation for 3DGS Compression.** Bert Van hauwermeiren. 2026-08-28.
Importance-weighted non-uniform quantisation and merging for point-cloud style 3DGS, intended as an MPEG 3DGS standardisation contribution.
supplies = none / binds = post-hoc / constrains = bytes / gap = no.

**2603.09718 GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System.** Zhiye Tang. 2026-03-10.
Collaborative viewport prediction and DRL bitrate adaptation for 3DGS streaming.
supplies = network bitrate / binds = inference (ABR) / constrains = bytes streamed / gap = no.

**2507.14454 Adaptive 3D Gaussian Splatting Video Streaming: Visual Saliency-Aware Tiling and Meta-Learning-Based Bitrate Adaptation.** Han Gong. 2025-07-19.
Saliency tiling, per-tile quality levels, meta-learning ABR.
supplies = network bitrate / binds = inference / constrains = bytes streamed / gap = no.

**DOI 10.1145/3672196.3673394 Multi-frame Bitrate Allocation of Dynamic 3D Gaussian Splatting Streaming Over Dynamic Networks.** Yuan-Chun Sun. EMS@SIGCOMM 2024. No arXiv id.
Per search text: Model-driven Gradient Ascent searches encoding parameters across attribute categories and frames to keep bitrate below a threshold.
supplies = bitrate threshold / binds = post-hoc / constrains = bytes / gap = no.

**DOI 10.1145/3712676.3714445 LTS: A DASH Streaming System for Dynamic Multi-Layer 3D Gaussian Splatting Scenes.** Yuan-Chun Sun. MMSys 2025 (best paper per search text). No arXiv id.
Trains each dynamic scene into dependent layers, tiles and segments for DASH.
supplies = network bitrate / binds = training (layers) + inference (ABR) / constrains = bytes streamed / gap = no.

### 2.2 Count-denominated budgets bound during training

**2505.10473 ControlGS: Consistent Structural Compression Control for Deployment-Aware Gaussian Splatting.** Fengdi Zhang. 2025-05-15 (v3; v1 title "Consistent Quantity-Quality Control across Scenes for Deployment-Aware Gaussian Splatting").
Maps the count-quality trade-off to one continuous, scene-agnostic control hyperparameter. One fixed training setup plus one global hyperparameter yields models from compact to high-fidelity across object to outdoor scales, with higher quality at equal or fewer Gaussians than competitors.
supplies = λ / binds = training / constrains = count (indirectly) / gap = no. Scene-consistent λ, but the user still cannot name a size or a device.

**2602.03538 Constrained Dynamic Gaussian Splatting.** Zihan Zheng. 2026-02-03.
Budget-constrained optimisation for dynamic scenes with a differentiable budget controller driven by a fused geometric, motion and perceptual importance score, static/dynamic capacity allocation, three-phase schedule, target count met to < 2 % error, plus a dual-mode compression scheme, over 3× compression versus SOTA.
supplies = count / binds = training / constrains = count / gap = partly. The controller design (differentiable, error < 2 %) is the closest template for a byte controller, but the budget is a count.

**2604.21400 You Only Gaussian Once: Controllable 3D Gaussian Splatting for Ultra-Densely Sampled Scenes.** Jinrang Jia. 2026-04-23.
Reformulates heuristic growth as a deterministic budget-aware equilibrium with a feedback budget controller and per-region targets N^target_m (fetched), allocating a densification quota max(0, ΔN_m(k) / (K − k + 1)) per step so the final count converges to the sum of targets. Introduces the Immersion v1.0 ultra-dense indoor dataset. Fetched: no memory, VRAM or FPS table.
supplies = count per region / binds = training / constrains = count / gap = partly (deterministic count profile "for hardware-defined primitive budgets", no bytes or ms).

**2603.19234 Matryoshka Gaussian Splatting.** Zhilin Guo. 2026-03-19.
Stochastic budget training: each iteration samples a random splat budget and optimises both that prefix and the full set, giving one ordered Gaussian set where any prefix renders coherently. Two forward passes, no architecture change, matches full-capacity quality, continuous speed-quality trade-off from one model.
supplies = none at training, count at inference / binds = training (all budgets) / constrains = count / gap = partly. Anytime property removes the need to know the budget at training time, which is the complementary route to ours.

**2604.20046 Gaussians on a Diet: High-Quality Memory-Bounded 3D Gaussian Splatting Training.** Yangming Zhang. 2026-04-21.
Memory-bounded training that alternates incremental pruning of low-impact Gaussians with strategic growth plus adaptive compensation, keeping near-constant memory. Fetched: the bound is a target peak number of Gaussians F, with no explicit count-to-bytes model; peak training memory on Jetson AGX Xavier 2.98 GB with a parallel loader versus 18.59 GB for 3DGS, up to 80 % lower; no FPS reported.
supplies = count (as memory proxy) / binds = training / constrains = training peak memory via count / gap = partly. The intent is bytes of device memory, the mechanism is a count cap.

**2511.06810 ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction With Fewer Primitives.** Bartłomiej Baranowski. 2025-11-10.
iNGP depth proxy, insertion of Gaussians along viewing cones of high-error pixels, pre-activation opacity penalty, and a primitive budgeting strategy that controls the total count "either by a fixed budget or by adapting to scene complexity".
supplies = count or none / binds = training / constrains = count / gap = no.

**2511.16980 Gradient-Driven Natural Selection for Compact 3D Gaussian Splatting.** Xiaobin Deng. 2025-11-21.
Survival pressure as a regularisation gradient field on opacity, opacity decay with finite prior; over 0.6 dB PSNR gain under 15 % budgets.
supplies = count fraction / binds = training / constrains = count / gap = no.

**2505.05587 Steepest Descent Density Control for Compact 3D Gaussian Splatting.** Peihao Wang. CVPR 2025. 2025-05-08.
Optimisation-theoretic densification: splitting escapes saddle points, minimal offspring count, optimal update direction, analytic opacity normalisation, about 50 % fewer points.
supplies = none / binds = training / constrains = count (indirectly) / gap = no.

**2607.10912 DP-Splat: Bayesian Nonparametric Complexity Control for Gaussian Splatting.** Aqi Dong. 2026-07-12.
Truncated stick-breaking Dirichlet-process prior over mixture weights in variational Bayes GS so the number of occupied components adapts to data, closed-form updates, 5.9 to 7.6× fewer components than VBGS.
supplies = concentration α / binds = training / constrains = count / gap = no.

**2607.05522 Rendering-Aware Bayesian 3D Gaussian Splatting with Native Uncertainty and Adaptive Complexity Control.** Gaoxiang Jia. 2026-07-06.
NIW posterior over Gaussian geometry with renderer-derived summaries, optional DP extension for component usage, active view selection under fixed acquisition budgets.
supplies = none / binds = training / constrains = count (adaptive) / gap = no.

**2605.00408 Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting (LeGS).** Zhenhua Ning. 2026-05-01.
Density control as an RL policy with a sensitivity-based reward, O(N) reward computation.
supplies = none / binds = training / constrains = count (indirectly) / gap = no, but an RL controller could take a device cost as reward.

**2508.04078 RLGS: Reinforcement Learning-Based Adaptive Hyperparameter Tuning for Gaussian Splatting.** Zhan Li. 2025-08-06.
Plug-and-play RL policies adjusting learning rates and densification thresholds, +0.7 dB on Taming-3DGS at a fixed Gaussian budget.
supplies = count (via backbone) / binds = training / constrains = count / gap = no.

**2508.09239 Gradient-Direction-Aware Density Control for 3D Gaussian Splatting.** Zheng Zhou. 2025-08-12.
Per search text, a hyperparameter p sets the proportion of Gaussians eligible for densification, trading memory for quality.
supplies = λ-like p / binds = training / constrains = count / gap = no.

**2503.14475 Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation (Opti3DGS).** Umar Farooq. 2025-03-18. **2506.23042 From Coarse to Fine: Learnable Discrete Wavelet Transforms for Efficient 3D Gaussian Splatting (AutoOpti3DGS).** Hung Nguyen. ICCV workshop. 2025-06-29. **2602.14199 Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation.** Hung Nguyen. 2026-02-15.
Blur or wavelet-filter training images coarse-to-fine so fine Gaussians form late; 62 % fewer Gaussians and 40 % lower training GPU memory for Opti3DGS.
supplies = none / binds = training / constrains = count (indirectly) / gap = no.

**2404.06109 Revising Densification in Gaussian Splatting.** Samuel Rota Bulò. 2024-04-09. **2503.14274 Improving Adaptive Density Control for 3D Gaussian Splatting.** Glenn Grubert. 2025-03-18. **2411.10133 Efficient Density Control for 3D Gaussian Splatting.** Xiaobin Deng. 2024-11-15. **2508.12313 Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering.** Xiaobin Deng. 2025-08-17. **2510.26921 DC4GS: Directional Consistency-Driven Adaptive Density Control.** Moonsoo Jeong. NeurIPS 2025. 2025-10-30. **2604.28016 Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification.** Linjie Lyu. 2026-04-30. **2605.06876 AdpSplit: Error-Driven Adaptive Splitting.** Yongjae Lee. 2026-05-07.
Densification criteria (pixel error, gradient direction coherence, split placement). Revising Densification also has a primitive-count cap per search text.
supplies = none or count cap / binds = training / constrains = count / gap = no.

**2504.14548 VGNC: Reducing the Overfitting of Sparse-view 3DGS via Validation-guided Gaussian Number Control.** Lifeng Lin. 2025-04-20.
Generated validation views pick the Gaussian number that minimises overfitting.
supplies = none / binds = training / constrains = count / gap = no.

**2602.06830 GaussianPOP: Principled Simplification Framework for Compact 3D Gaussian Splatting via Error Quantification.** Soonbin Lee. 2026-02-06.
Analytic per-Gaussian error criterion from the rendering equation, single forward pass, supports on-training pruning and post-training simplification.
supplies = count fraction / binds = training or post-hoc / constrains = count / gap = no.

**2602.24136 Prune Wisely, Reconstruct Sharply.** Haoran Wang. CVPR 2026. 2026-02-27. **2506.09534 Gaussian Herding across Pens.** Tao Wang. 2025-06-11. **2606.09074 REFINE: Super-efficient 3D Gaussian Splatting Pruning via Rendering-Free Primitive Importance.** Zhang Chen. 2026-06-08. **2607.02721 Provable Pruning for Efficient 3D Gaussian Splatting via Coresets.** Waseem Mousa. 2026-07-02. **2603.16103 NanoGS: Training-Free Gaussian Splat Simplification.** Butian Xiong. ECCV 2026. 2026-03-17. **2601.14821 POTR: Post-Training 3DGS Compression.** Bert Ramlot. TCSVT 2026. 2026-01-21. **2605.30396 Smaller and Faster 3DGS via Post-Training Dictionary Learning.** Jiarong Gong. 2026-05-28. **2406.18214 Trimming the Fat.** Muhammad Salman Ali. BMVC 2024. **2410.23213 ELMGS.** Muhammad Salman Ali. 2024-10-30. **2406.17074 Reducing the Memory Footprint of 3D Gaussian Splatting.** Panagiotis Papantonakis. PACMCGIT 2024.
Pruning and simplification: reconstruction-aware timing (Prune Wisely, 90 % count reduction), OT mixture reduction (10 % Gaussians), Hessian-proxy importance with 20× lower pruning latency (REFINE), sensitivity-sampled coresets with resolution-dependent guarantees (Provable), CPU-only pairwise merging (NanoGS), rasteriser-computed removal effect and SH entropy reduction (POTR, 1.5 to 2× faster inference), dictionary learning (3.95× and 23 % faster rendering), gradient-informed iterative pruning to 600 FPS (Trimming), differentiable quantisation plus entropy estimator (ELMGS), resolution-aware pruning plus adaptive SH plus codebook (Reduced 3DGS, 27× smaller, 1.7× faster).
supplies = count fraction or none / binds = post-hoc (GaussianPOP and Trimming also in training) / constrains = count, bytes / gap = no.

**2404.12777 EfficientGS.** Wenkai Liu. 2024-04-19. **2312.04564 EAGLES.** Sharath Girish. 2023-12-07. **2503.16924 Optimized Minimal 3D Gaussian Splatting (OMG).** Joo Chan Lee. 2025-03-21. **2510.03857 Optimized Minimal 4D Gaussian Splatting.** Minseo Lee. 2025-10-04. **2501.05757 Locality-aware Gaussian Compression (LocoGS).** Seungjoo Shin. 2025-01-10. **2503.20221 TC-GS.** Taorui Wang. 2025-03-26. **2503.23162 NeuralGS.** Zhenyu Tang. 2025-03-29. **2504.13022 CompGS++.** Xiangrui Liu. 2025-04-17. **2603.28431 LG-HCC.** Xuan Deng. 2026-03-30. **2601.04348 SCAR-GS.** Diego Revilla. 2026-01-07. **2608.14136 HiCo-GS.** Wei Zhang. 2026-08-14. **2508.15372 Image-Conditioned 3D Gaussian Splat Quantization.** Xinshuang Liu. 2025-08-21. **2604.05366 3DTurboQuant.** Jae Joong Lee. 2026-04-07. **2512.11186 Lightweight 3DGS Compression via Video Codec.** Qi Yang. DCC 2026. **2505.18197 GausPcgc benchmark.** Kangli Wang. 2025-05-21. **2506.01822 GSCodec Studio.** Sicheng Li. 2025-06-02. **2512.24742 Splatwizard.** Xiang Liu. 2025-12-31.
Compact representations and codecs (named-method checks and HAC++ successors). None takes a size target; all are λ or fixed-configuration.
supplies = λ or none / binds = training or post-hoc / constrains = bytes / gap = no.

**Training-speed line (count as a side effect):** 2503.18402 DashGaussian (Youyu Chen, CVPR 2025), 2411.12788 Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D Gaussians (Mini-Splatting2, Guangchi Fang), 2412.13547 Turbo-GS (Ankit Dhiman), 2606.15924 TurboGS (Zheng Dong, 2026), 2511.04283 FastGS (Shiwei Ren), 2602.09999 Faster-GS (Florian Hahlbohm, 2026), 2501.14534 Trick-GS (Anil Armagan), 2503.01199 LiteGS (Kaimin Liao), 2412.07608 Group Training (Chengbo Wang), 2504.06716 GSta (Anil Armagan), 2603.09277 Shorter Gaussian Lists (Jiaqi Liu, CVPR 2026), 2603.08997 SkipGS (Jingxing Li), 2609.03334 Laplacian Frequency Hierarchies (Yixiong Yang, PG 2026), 2606.21244 ACE-GS (Jijian Zhao), 2507.20239 Decomposing Densification (Binxiao Huang), 2510.14564 BalanceGS (Junyi Wu, ASP-DAC 2026). All: supplies = none / binds = training / constrains = training time / gap = no.

### 2.3 Elastic, anytime, progressive and LoD representations (one model, many budgets)

**2506.04174 FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting.** Hengyu Liu. CVPR 2025. 2025-06-04.
Given a desired model size as an input percentage, a tiny learnable module selects a Gaussian subset and a transformation module adjusts it, no fine-tuning per ratio, up to 18× faster than ratio-specific LightGaussian.
supplies = count fraction at inference / binds = training (elastic) + inference / constrains = count / gap = partly (size input, but as a fraction of primitives).

**2510.09997 CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting.** Zhigang Cheng. ICLR 2026. 2025-10-11.
Learnable distance-dependent opacity decay per Gaussian, virtual distance scaling, coarse-to-fine training with a rendered point count regularisation, continuous quality scaling "across a wide range of performance targets" from one model.
supplies = none / binds = training / constrains = rendered count / gap = partly (rendered-count regulariser during training).

**2512.18692 EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images.** Jongmin Park. 2025-12-21.
Feed-forward model that predicts a representation for any target primitive count at inference, via importance-aware fine-tuning that ranks primitives.
supplies = count / binds = inference / constrains = count / gap = no (feed-forward).

**2606.05102 ZipSplat: Fewer Gaussians, Better Splats.** Alexander Veicht. 2026-06-03. Token-based feed-forward model, k-means at inference spans the quality-efficiency curve, about 6× fewer Gaussians. supplies = cluster count / binds = inference / gap = no.

**2501.13558 GoDe: Gaussians on Demand for Progressive Level of Detail and Scalable Compression.** Francesco Di Sario. 2025-01-23.
Gradient-sensitivity ordering reorganises a trained model into a fixed progressive hierarchy with multiple RD operating points, one quantisation-aware fine-tune.
supplies = none / binds = post-hoc / constrains = count, bytes / gap = no.

**2408.14823 LapisGS.** Yuang Shi. 3DV 2025. **2606.07179 EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree for Scalable 3D Streaming.** Yuang Shi. 2026-06-05. **2603.09703 ProGS: Towards Progressive Coding for 3D Gaussian Splatting.** Zhiye Tang. 2026-03-10. **2607.25971 SplatStream.** Muhammad Talha. Asilomar 2026. **2504.05517 L3GS: Layered 3D Gaussian Splats for Efficient 3D Scene Delivery.** Yi-Zhen Tsai. 2025-04-07. **2409.01761 PRoGS: Progressive Rendering of Gaussian Splats.** Brent Zoomers. 2024-09-03. **2604.11685 Unfolding 3D Gaussian Splatting via Iterative Gaussian Synopsis.** Yuqin Lu. 2026-04-13. **2602.00671 HPC.** Yangzhi Ma. 2026-01-31.
Layered or prefix-decodable bitstreams. ProGS: one fixed-λ run yields five deployable rate points from one bitstream, 30.7 to 60.7 % fewer bytes than HAC-Rand at equal quality. EvoGS: continuous layering cuts VRAM footprint up to 5.5×. L3GS: scheduling which splats to download when, VR traces.
supplies = λ or none / binds = training or post-hoc / constrains = bytes streamed / gap = no.

**2408.12894 FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering.** Yunji Seo. TOG (DOI 10.1145/3731430). 2024-08-23.
Multi-level 3DGS via level-specific 3D scale constraints, each level reconstructs the whole scene at a different GPU memory use, level-by-level training, selective per-region levels; "adjustable options for a broad range of GPU settings".
supplies = level / binds = training / constrains = runtime VRAM (by level) / gap = partly (memory-tiered levels, no numeric target).

**2403.17898 Octree-GS.** Kerui Ren. 2024-03-26. **2406.12080 A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets.** Bernhard Kerbl. 2024-06-17. **2505.23158 LODGE.** Jonas Kulhanek. NeurIPS 2025. **2507.01110 A LoD of Gaussians.** Felix Windisch. SIGGRAPH 2026. **2404.01133 CityGaussian.** Yang Liu. **2411.00771 CityGaussianV2.** Yang Liu. **2404.09748 LetsGo.** Jiadi Cui. **2505.06523 Virtualized 3D Gaussians.** Xijie Yang. **2603.23891 FilterGS.** Yixian Wang. 2026-03-25. **2601.18475 LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction.** Xinhui Liu. 2026-01-26. **2506.19415 Virtual Memory for 3D Gaussian Splatting.** Jonathan Haberl. 2025-06-24. **2511.19202 NVGS.** Brent Zoomers. 2025-11-24.
LoD, chunking, out-of-core and culling for large scenes. LODGE explicitly targets memory-constrained devices with distance-based subsets and chunk loading. supplies = none / binds = post-hoc or training / constrains = runtime VRAM and time (by view) / gap = no.

**2608.14112 Fixed-Budget Gaussian Volume Encoding with Structure-Aware Allocation.** Michael R. Martin. 2026-08-14.
Scientific scalar volumes encoded as anisotropic Gaussians under a fixed primitive budget allocated analytically from field structure before refinement, no densification or pruning; "the selected budget determines encoded storage before refinement" and, with the iteration schedule, the refinement-time budget. 1.4 M Gaussians encode a billion-voxel volume in four minutes.
supplies = count / binds = training (fixed from the start) / constrains = count, hence bytes and encode time / gap = partly (a count budget chosen a priori that fixes storage exactly, but for volumes, not view synthesis).

**2604.19127 OT-UVGS.** Byunghyun Kim. EG 2026 short. UV-mapped Gaussians as capacity allocation under a fixed UV budget. supplies = UV resolution / binds = post-hoc / gap = no.

### 2.4 Other 2026 compression papers surfaced (verified, no size target)
2601.14510 GSICO (Pedro Martin), 2602.09816 CompSplat (Hojun Song), 2605.25563 CodecSplat (Pengpeng Yu), 2607.24403 GenSplatCodec (Qiang Hu), 2607.26525 AtlasLC (ByungHyun Kim), 2608.22344 Polarized Opacity Prior (Zi-Ming Wang), 2608.04581 ACA-GS (Seunghyeon Song, ACM MM 2026), 2607.10237 CoSAG (Yuang Jia), 2603.23297 Drop-In Perceptual Optimization (Ezgi Ozyilkan, ECCV 2026, WD-R loss gives about 50 % bitrate savings at equal perceptual quality), 2605.07287 SplatWeaver (Yecong Wan), 2607.22956 ParticleGS (Bo Jiang), 2510.26694 The Impact and Outlook of 3DGS (Bernhard Kerbl), 2512.07197 SUCCESS-GS survey (Seokhyun Youn), 2407.17418 survey (Yanqi Bao), 2412.06257 XR survey (Shi Qiu). All: gap = no.

## 3. Part 2: render-time and runtime-memory line (verified)

### 3.1 (a) Papers that model or measure render cost as a function of model state

**2407.00435 MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning and Accelerated Foveated Rendering.** Weikai Lin. ASPLOS 2025. 2024-06-29 (v1 title "RTGS: Enabling Real-Time Gaussian Splatting on Mobile Devices Using Efficiency-Guided Pruning and Foveated Rendering").
Fetched: per-point computational efficiency CE_i = Val_i / Comp_i, Val_i = number of pixels dominated by the point, Comp_i = number of tiles intersecting the point's ellipse "which directly affects the rendering speed". Shows that reducing point count alone reduces latency slower than count, and that latency tracks the number of tile-ellipse intersections. Prunes by CE, adds foveated rendering with strict point subsets per eccentricity, and an accelerator for load balance. Jetson AGX Xavier: MetaSapiens-L 7.9× faster than 3DGS, 20.9× with hardware, target 75 to 90 FPS for AR/VR.
supplies = none (pruning ratio) / binds = post-hoc pruning / constrains = render time via tile intersections / gap = partly for the render-time half: the only per-primitive cost proxy validated against measured latency.

**2604.18980 AdaGScale: Viewpoint-Adaptive Gaussian Scaling in 3D Gaussian Splatting to Reduce Gaussian-Tile Pairs.** Joongho Jo. DAC 2026. 2026-04-21.
States that Gaussian-tile pair generation, sorting and rasterisation dominate runtime and that workload is set by the number of Gaussian-tile pairs; scales low-importance Gaussians down for the intersection test only, 13.8× geometric-mean speedup on city scenes at about 0.5 dB.
supplies = none / binds = inference (preprocess) / constrains = render time / gap = n/a (cost model: pairs).

**2606.00352 HiGS: A Hierarchical Rendering Architecture for Real-Time 3D Gaussian Splatting.** Dawid Pająk. 2026-05-29.
Partitioning cost falls with larger tiles, rasterisation cost with smaller; macro-tiles for binning and sort, fine tiles for rasterisation, work issued proportional to Gaussians per macro-tile. Fetched: frame time grows roughly linearly in Gaussian count, 1.25 to 9.97 ms at 1080p and 1.97 to 10.29 ms at 4K over budget caps 5 M to 75 M on an RTX PRO 6000 Blackwell; 0.52 to 0.82 ms from 1080p to 4K on standard scenes.
constrains = render time / cost model: near-linear in count, weakly dependent on resolution.

**2604.07177 Splats under Pressure: Exploring Performance-Energy Trade-offs in Real-Time 3D Gaussian Splatting under Constrained GPU Budgets.** Muhammad Fahim Tajwar. 2026-04-08.
Emulates GPU tiers by underclocking and power caps, measures FPS, power, energy per frame E = P_avg / FPS and performance per watt over splat counts and scenes. Fetched: at 1080p, RTX 4090 58.8 to 44.8 fps, 4070 Ti 58.6 to 36.2, 3070 57.0 to 30.2, 3050 45.8 to 19.7 fps from 0.58 M to 3.45 M splats; 60 FPS needs under about 600 k splats; above about 1 M visible splats low tiers degrade; animation MLP costs 15 % (4090) to 35 % (3050).
constrains = none (measurement) / cost model: FPS versus count per device tier, the empirical table our render-time budget needs.

**2608.15785 RoofGS: Roofline-Guided End-to-End Acceleration of 3D Gaussian Splatting.** Yang Luo. 2026-08-16.
Stage-wise roofline characterisation: front end is global-memory bound, rasteriser is instruction-throughput bound; 32-bit quantised sort keys, bit-level exponential approximation with per-pixel error bound; 10.1× at 4K on an RTX 4090 (61 to 616 FPS).
cost model: memory traffic for preprocess and sort, instruction count for rasterisation.

**2412.00578 Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives.** Alex Hanson. CVPR 2025. 2024-11-30.
Precise localisation of Gaussians in screen space (fewer tile pairs) plus a pruning technique integrated in training; 6.71× average rendering speedup with smaller models and faster training.
supplies = pruning fraction / binds = training / constrains = render time via count and tile pairs / gap = partly for render time.

**2409.08669 AdR-Gaussian.** Xinzhe Wang. SIGGRAPH Asia 2024. Early culling of low-opacity Gaussian-tile pairs with adaptive radius and AABB, pixel-thread load balancing, 310 % speed. **2605.04844 QuadBox.** Xinze Li. ICIP 2026. Four axis-aligned boxes tightly cover projected ellipses, 1.85×. **2412.17378 Balanced 3DGS.** Hao Gui. Gaussian-wise parallelism and fine-grained tiling against load imbalance in training, render kernel up to 7.52×. **2606.16566 Local-GS.** Yang Luo. 2026. Warp-coherent hoisting, culling and blending, 7.76× on Deep Blending. **2607.17842 CaT-GS.** Tingjia Zhang. CVPR 2026. Inter-frame caching and tile scheduling for large scenes, up to 10×. **2607.03390 TemporalGS.** Yuhongze Zhou. 2026. Training-free temporal culling and selective tile rendering, 1.48×. **2605.17855 TensorGS.** Sheng Li. 2026. Rasterisation is compute-bound; FP16 on Tensor Cores with cross-tile grouping, 1.65×. **2604.02120 GEMM-GS.** Haomin Li. DAC 2026. Blending as GEMM, 1.42×. **2503.24366 StochasticSplats.** Shakiba Kheradmand. 2025. Sorting-free Monte Carlo rasterisation where the sample count trades compute for quality; notes that with sorted rendering lower resolution is not necessarily faster. **2408.07967 FlashGS.** Guofeng Feng. 4× on mobile consumer GPUs. **2402.00525 StopThePop.** Lukas Radl. **2410.18931 Sort-free Gaussian Splatting via Weighted Sum Rendering.** Qiqi Hou. **2504.12811 AAA-Gaussians.** Michael Steiner. **2503.05168 SeeLe.** Xiaotong Huang. 2025. Hybrid preprocessing and contribution-aware rasterisation for mobile, 2.6× and 32 % model reduction. **2603.09277 Shorter Gaussian Lists.** Jiaqi Liu. CVPR 2026. Scale resets and an alpha-blending entropy constraint shorten per-pixel Gaussian lists during training. **2603.11531 Mobile-GS.** Xiaobiao Du. 2026. Identifies depth sorting in alpha blending as the mobile bottleneck, order-independent rendering plus neural view-dependent enhancement, first-order SH distillation, neural VQ and contribution pruning; per search text 116 FPS at 1600 × 1063 on Snapdragon 8 Gen 3 with 4.8 MB.
All: cost drivers named are Gaussian-tile pairs, per-pixel list length, sorting bandwidth and tile load imbalance.

### 3.2 (b) Papers that constrain render time, FPS or runtime memory during training or pruning

**2509.07021 MEGS²: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning.** Jiarui Chen. ICLR 2026. 2025-09-07.
Targets rendering memory rather than storage: replaces SH with spherical Gaussian lobes and poses primitive-number and lobe-number pruning as a single constrained optimisation; 50 % static VRAM and 40 % rendering VRAM reduction.
supplies = sparsity targets / binds = training / constrains = runtime VRAM / gap = partly for the memory half (constrained optimisation on the two factors of rendering memory, still not a byte target).

**2606.24796 Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM.** Leshu Li. ICRA 2026. 2026-06-23.
Prunes by contribution to effective rendering area rather than opacity or gradient, reducing peak runtime memory over 60 % and more than 2× FPS on EuRoC and KITTI.
supplies = none / binds = online (SLAM) / constrains = runtime memory and FPS / gap = no.

**2510.09997 CLoD-GS** (rendered point count regularisation, above), **2603.19234 Matryoshka** (speed-quality prefixes, above), **2408.12894 FLoD** (VRAM-tiered levels, above), **2505.23158 LODGE** (memory-constrained devices, above), **2412.00578 Speedy-Splat** (above), **2603.09277 Shorter Gaussian Lists** (above).

**2410.17932 VR-Splatting: Foveated Radiance Field Rendering via 3D Gaussian Splatting and Neural Points.** Linus Franke. 2024-10-23. Few-primitive smooth Gaussians for the periphery and neural points for the fovea "within the rendering budget" of VR. **2505.10144 VRSplat.** Xuechang Tu. I3D 2025. Mini-Splatting, StopThePop and Optimal Projection combined with a single-launch foveated rasteriser, 72+ FPS in VR.
supplies = none / binds = training and rendering / constrains = frame time (VR threshold) / gap = no.

**2605.15324 P-WRFGS.** Chenghong Bian. IEEE WCL 2026. Learnable per-Gaussian masks with a weighted rendering plus regularisation loss for a "flexible trade-off between rendering quality and efficiency", 100× storage and 7× speed on wireless radiance fields. supplies = λ / binds = training / constrains = count, speed / gap = no.

**2509.03775 ContraGS.** Sankeerth Durvasula. 2025. Training directly on codebook-compressed parameters via MCMC, 3.49× lower peak training memory, 1.88× faster rendering. **2509.15645 GS-Scale.** Donghyun Lee. 2025. Host offloading, 3.3 to 5.6× lower GPU memory, 18 M Gaussians on an RTX 4070 Mobile. **2511.04951 CLM.** Hexu Zhao. ASPLOS 2026. CPU offloading renders 100 M Gaussians on one RTX 4090. **2605.20150 TideGS.** Chonghao Zhong. ICML 2026. Out-of-core SSD-CPU-GPU training beyond one billion Gaussians on 24 GB. **2608.27735 ABCD.** Ka Heng Shiu. SIGGRAPH Posters 2026. Block coordinate descent with O(1) peak VRAM in scene extent. **2502.14938 GS-Cache.** Miao Tao. 2025. Cache-centric multi-GPU VR rendering, 42 % lower GPU memory. **2509.13536 MemGS.** Yinlong Bai. 2025. Voxel-space merging for SLAM on MAV-class hardware.
supplies = none / binds = training or rendering system / constrains = memory by system design, not by a target / gap = no.

### 3.3 (c) Edge-device results

| paper | device | result |
|---|---|---|
| 2407.00435 MetaSapiens | Jetson AGX Xavier | 7.9× over 3DGS, 20.9× with accelerator, 75 to 90 FPS target |
| 2604.20046 Gaussians on a Diet | Jetson AGX Xavier (training) | peak training memory 2.98 GB versus 18.59 GB |
| 2503.23625 Gaussian Blending Unit (Zhifan Ye, HPCA 2025) | Jetson Orin NX | baseline 7 to 17 FPS, blending stage is the bottleneck, GBU plug-in for real time |
| 2507.21572 LS-Gaussian (Linye Wei, ICCAD 2025) | Jetson AGX Orin | 5.41× over edge GPU, 17.3× with accelerator, tile workload prediction from viewpoint transformation |
| 2510.06644 RTGS (Leshu Li, MICRO 2025) | edge GPU with accelerator | >= 30 FPS 3DGS-SLAM, 82.5× energy |
| 2603.08499 Continual learning on COTS edge devices (Ivan Zaino, 2026) | Jetson Orin Nano | VBGS training peak memory 9.44 to 1.11 GB, first NVS training on Orin Nano, 19× lower per-frame latency |
| 2603.11531 Mobile-GS | Snapdragon 8 Gen 3 (search text) | 116 FPS at 1600 × 1063, 4.8 MB |
| 2511.16298 Optimizing 3D Gaussian Splattering for Mobile GPUs (Md Musfiqur Rahman Sanim, PACT 2025 per search) | mobile GPU | texture-cache cost model for sorting, 1.7× end-to-end, 1.6× less memory |
| 2506.02774 Voyager (Zheng Liu, 2025) | mobile devices | temporal LoD search and preemptive alpha filtering, 6.6× and 85 % energy savings city-scale |
| 2511.12930 Neo (Changhun Oh, 2025) | edge GPU / ASIC | sorting is bandwidth bound, reuse-and-update sort, 10× over edge GPU, 94.5 % less DRAM traffic |
| 2602.03207 WebSplatter (Yudong Han, 2026) | web browsers, WebGPU | wait-free radix sort, opacity-aware culling cuts peak memory, 1.2 to 4.5× over web viewers |
| 2609.05255 Compact Neural Appearance Models (Florian Hahlbohm, 2026) | laptop and mobile GPUs via WebGL | per-primitive appearance 192 to 28 bytes |
| 2601.17354 PocketGS (Wenzhi Guo, 2026) | mobile phone (training) | on-device training under minute-scale time and peak-memory budgets |
| 2604.07177 Splats under Pressure | emulated tiers down to RTX 3050 class | FPS versus count table above |
| 2406.17074 Reduced 3DGS | mobile device | reduced download times |
| 2406.19434 Lightweight Predictive 3D Gaussian Splats (Junli Cao, 2024) | mobile device | parent points plus tiny MLPs, real-time on mobile |
| 2408.07967 FlashGS | mobile consumer GPUs | 4× |
| 2506.09070 STREAMINGGS (Chenqi Zhang, 2025) | mobile Ampere GPU | 2 to 9 FPS baseline, 45.7× with co-design, memory-centric rendering |
| 2511.18755 Splatonic (Xiaotong Huang, 2025) | mobile GPUs | sparse pixel sampling, 14.6× end-to-end on off-the-shelf GPUs |
| 2609.02352 Atlas (He Zhu, 2026) | mobile VR | hierarchical memory offloading, 18.5× over GPU baseline |
| 2404.11285 Cinematic anatomy (Simon Niedermayr, 2024) | lightweight mobile devices | < 70 MB compressed 3DGS |
| 2506.19415 Virtual Memory for 3DGS | desktop and mobile | just-in-time streaming of visible Gaussians |
| 2605.08699 TIGAS (Emanuele Artioli, 2026) | thin web client | remote rendering under 10 ms per frame, ABR over QUIC |

### 3.4 (d) Hardware accelerators and what they say the cost depends on

| paper | venue | cost dependence stated |
|---|---|---|
| GSCore (DOI 10.1145/3620666.3651385, Junseo Lee) | ASPLOS 2024 | per search text: AABB tests yield false-positive Gaussian-tile pairs for anisotropic splats; shape-aware support, hierarchical sort, subtile skipping |
| GauSPU (DOI 10.1109/micro61859.2024.00114, Lizhou Wu) | MICRO 2024 | per search text: 3DGS-SLAM tracking at 33.6 FPS, 63.9× energy versus RTX 3090 |
| GSArch (DOI 10.1109/hpca61900.2025.00037, Houshu He) | HPCA 2025 | per search text: backward pass memory access dominates training |
| 2503.23625 GBU (Zhifan Ye) | HPCA 2025 | blending stage, per-Gaussian per-pixel contribution, row-sequential shading |
| 2507.15300 GCC (Minnan Pei) | MICRO 2025 | preprocessed-but-unrendered Gaussians and repeated per-tile loads; alpha-based boundaries |
| 2608.02099 DeGS (Minnan Pei) | MICRO 2026 | PE underutilisation from irregular coverage and asynchronous pixel termination; 720p to 8K |
| 2604.10223 129 FPS accelerator (Fang-Chi Chang) | TVCG 2026 | zero-Jacobian skipping, comparison-free tile sort with deterministic latency, 0.219 W at 1080p |
| 2511.12930 Neo (Changhun Oh) | 2025 | sorting memory bandwidth |
| 2603.01158 FLICKER (Wenhui Ou) | DATE 2026 | non-contributing Gaussians, contribution testing cost |
| 2511.16831 Vorion (Yipeng Wang) | 2025 | RISC-V GPU with Gaussian rasteriser, 19 FPS render, training 38.6 it/s |
| 2503.16681 GauRast (Sixu Li) | DAC 2025 | reuse of triangle rasteriser, 23× stage speed, 24 or 46 FPS end to end |
| 2507.19133 3DGauCIM (Wei-Hsing Huang) | 2025 | DRAM loads for culling, sorting latency grows with parameter count, buffer reuse; >200 FPS at 0.28 W |
| 2506.07069 Axis-shared rasterisation (Zhican Wang) | ISCA 2026 | MAC count in rasterisation, sorting scalability, pipeline imbalance |
| 2511.18755 Splatonic | 2025 | rendered pixel count for tracking |
| REACT3D (DOI 10.1145/3725843.3756109, Hongyi Wang) | MICRO 2025 | per search text: 30 FPS mapping threshold, redundant training compute and irregular memory access |
| 2510.06644 RTGS | MICRO 2025 | tile load imbalance, backprop reuse, atomic gradient traffic |
| 2507.21572 LS-Gaussian | ICCAD 2025 | per-tile workload predicted from viewpoint change, stalls from imbalance |
| 2609.02352 Atlas, 2512.20495 Nebula (He Zhu) | 2026, 2025 | per-frame visible subset, LoD search, stereo redundancy |
| 2506.09070 STREAMINGGS | 2025 | DRAM traffic of tile-centric rendering |
| 2608.13143 ProbSplat (Siddarth Gottumukkula) | ISVLSI 2026 | analog CIM log-likelihood, not a rasteriser |

Hardware-aware NAS analogue: web search 25 and the arXiv sweeps found no 3DGS paper citing hardware-aware NAS, latency predictors or lookup tables. MetaSapiens' CE metric and LS-Gaussian's per-tile workload prediction are the nearest profiling-derived cost proxies. Texture3dgs (2511.16298) analyses its sorting algorithm "in view of a cost model for the texture cache".

## 4. Unverified or off-topic name resolutions
- GSNorm (ASP-DAC 2025, DOI 10.1145/3658617.3697781 seen only in a search result title): not fetched, unverified.
- GaussianLens resolves to 2509.25603 (Yijia Weng, "Localized High-Resolution Reconstruction via On-Demand Gaussian Densification"), verified, off-topic.
- EdgeGaussians resolves to 2409.12886 (3D edge mapping), verified, off-topic. Textured Gaussians resolves to 2407.09733 Textured-GS (Zhentao Huang), verified, off-topic for budgets.
- MobileGS resolves to Mobile-GS 2603.11531. LS-Gaussian is the method name of 2507.21572. "RTGS" is both MetaSapiens v1 (2407.00435) and the MICRO 2025 SLAM paper (2510.06644).
- Visionary (2512.08478, Yuning Gong, WebGPU platform) appeared in the listing sweep only, title verified, abstract not read.
- All web-search claims about Snapdragon FPS (Mobile-GS), PACT venue (Texture3dgs), and GSCore/GauSPU/GSArch mechanisms are from search snippets, not from fetched full texts, and are marked "per search text".
