# Project: Rate Control for 3D Gaussian Splatting: Training onto a Byte Budget

## Problem Statement

A 3D Gaussian Splatting scene is optimized for one thing, image quality, and the resulting representation is whatever size the optimization happens to produce. Every place such a scene is actually used imposes a second number on it: the bytes it may occupy on disk or in transmission, the memory it may hold on the GPU while rendering, the milliseconds it may take per frame on the device in hand. Today those numbers are honoured after the fact. A scene is trained to convergence and then compressed, pruned or quantized until it fits, or the budget is approached indirectly through a knob that trades quality against size with no way of telling in advance where a given scene will land. Both routes leave the practitioner retraining or searching until the result happens to fit, and both decide what to throw away from a finished representation rather than deciding where to spend a fixed allowance in the first place.

This project asks what changes when the resource constraint becomes part of the optimization itself, present from the first iteration rather than applied to its output. The budget the user names is treated as a constraint the training has to respect while it runs, measured directly rather than inferred from a proxy such as the number of primitives, so that the optimization continuously decides how much of the allowance goes into more primitives, how much into higher precision, and where in the scene either belongs. The storage budget comes first, because it is the one the field already measures and compares on, and the same machinery then carries over to the constraints that matter on an edge device: memory at render time, and frame time on a named board.

## Requirements

### Required

- Strong Python, and PyTorch beyond the tutorial level: custom autograd paths, optimizer internals, reading and modifying a research codebase rather than calling a trainer script
- 3D computer vision fundamentals: pinhole camera model, intrinsics and extrinsics, the COLMAP SfM output format, and the novel-view-synthesis protocol (held-out views, PSNR, SSIM, LPIPS)
- Working understanding of 3D Gaussian Splatting: the primitive, the tile-based rasterizer, spherical harmonics, and how densification (clone, split, prune, or MCMC relocation) drives the count
- Comfort with a CUDA-backed codebase. The substrate is `gsplat` (Apache-2.0). No kernel writing is required, but the code has to be read and extended
- Experimental discipline: one variable per run, fixed and recorded seeds, a run registry, results reported with mean and standard deviation over at least 3 seeds, negative results written down
- Linux, git, and remote multi-GPU work. Every GPU job on the node goes through the shared scheduler, never a hand-picked device

### Nice-to-have

- Learned image or video compression: entropy models, the noise-relaxed quantization proxy, hyperpriors, arithmetic coding. The entropy layer of this project is a Ballé-style model over raw Gaussian attributes conditioned on a hash grid of position
- Classical rate control: the HEVC R-λ loop, PI control in the log domain, anti-windup and integral clamping. The controller is a proxy-Lagrangian with a dual step driven by a real encode
- Constrained optimization: Lagrangian duality, dual ascent, the proxy-Lagrangian of Cotter et al.
- Point-cloud compression, in particular G-PCC through `tmc3`, for the position codec
- Embedded deployment: Jetson, CUDA profiling, peak-VRAM measurement at steady state. This unlocks the stretch goal below
- Prior reading in the 3DGS compression line: Scaffold-GS, HAC++, Taming 3DGS, SizeGS, MesonGS++

## Goals

1. **Let the user state the budget.** A resource budget becomes an input to the optimization, given once at the start, in the unit the deployment actually constrains rather than in a proxy quantity that happens to be easy to count.
2. **Make the budget bind.** The scene that comes out of a single training run respects the budget it was given, on every scene of a standard benchmark, without a per-scene search and without a correction step afterwards.
3. **Spend the budget well.** Within the allowance, the optimization decides where the resources go, and the resulting quality should be at least as good as reaching the same budget by reducing a finished scene.
4. **Generalize beyond storage.** The storage budget is the first and best-measured case. The same construction should carry over to the constraints that actually decide whether a scene runs on a given device, memory during rendering and time per frame, and should not be tied to one particular scene representation.
5. **Evaluate it honestly.** Results on the benchmarks and protocol the field already compares on, against the established ways of reaching a budget, reproducible from recorded configurations and seeds.

Goals 1 to 3 are the core and are enough on their own for a thesis. Goal 4 is where the project becomes interesting beyond compression, and how far it is taken depends on the student and the time available.

## References

The starting reading list. Every entry is in the project's reference library under `references/`, which holds 67 papers with a per-paper note on why it is there.

**The representation and its densification**

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023, arXiv:2308.04079. The base representation, the tile rasterizer and the evaluation protocol every number in this field is reported under.
- Lu et al., *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering*, CVPR 2024, arXiv:2312.00109. The anchor-based variant the whole compression frontier is built on.
- Kheradmand et al., *3D Gaussian Splatting as Markov Chain Monte Carlo*, NeurIPS 2024, arXiv:2404.09591. Densification as relocation at a fixed count, the densifier this project drives.
- Yu et al., *Mip-Splatting: Alias-free 3D Gaussian Splatting*, CVPR 2024, arXiv:2311.16493, and Huang et al., *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*, SIGGRAPH 2024, arXiv:2403.17888. The two standard variants of the primitive itself.
- Baranowski et al., *ConeGS: Error-Guided Densification Using Pixel Cones*, arXiv:2511.06810. Where a densification step should go, which is the question the byte allowance has to answer.

**Compression through a rate-distortion trade-off**

- Chen et al., *HAC: Hash-grid Assisted Context for 3DGS Compression*, ECCV 2024, arXiv:2403.14530, and *HAC++: Towards 100X Compression of 3D Gaussian Splatting*, TPAMI 2025, arXiv:2501.12255. The reference entropy model, and the family whose rate multiplier this project replaces with a budget.
- Wang et al., *ContextGS: Compact 3DGS with Anchor Level Context Model*, NeurIPS 2024, arXiv:2405.20721.
- Niedermayr et al., *Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis*, CVPR 2024, arXiv:2401.02436. Quantized attributes rendered directly, which is what an edge device needs.
- Wang et al., *End-to-End Rate-Distortion Optimized 3D Gaussian Representation* (RDO-Gaussian), ECCV 2024, arXiv:2406.01597. The rate-distortion loss applied to raw 3DGS rather than to anchors, which is the substrate of this project.
- Zhan et al., *CAT-3DGS: Context-Adaptive Triplane Approach to RD-Optimized 3DGS Compression*, ICLR 2025, arXiv:2503.00357, and Liu et al., *HEMGS: A Hybrid Entropy Model for 3DGS Data Compression*, arXiv:2411.18473. The current top of the rate-distortion frontier.
- Xu et al., *Improving 3DGS Compression by Scene-Adaptive Lattice Vector Quantization* (SALVQ), IEEE TIP 2025, arXiv:2509.13482, and Chen et al., *PCGS: Progressive Compression of 3D Gaussian Splatting*, arXiv:2503.08511. One model covering several rate points, the closest thing to a size handle the field currently offers.
- Lee et al., *Compression of 3DGS with Optimized Feature Planes and Standard Video Codecs* (CodecGS), arXiv:2501.03399. A standard video codec in the loop, where classical rate control already lives.

**Methods that already take a size as input**

- Xie et al., *SizeGS: Size-aware Compression of 3D Gaussian Splatting via Mixed Integer Programming*, ACM MM 2025, arXiv:2412.05808, and *MesonGS++: Post-training Compression of 3DGS with Hyperparameter Searching*, arXiv:2604.26799. The post-hoc ancestors and the main baselines.
- Yang et al., *HybridGS: High-Efficiency Gaussian Splatting Data Compression*, ICML 2025, arXiv:2505.01938. A size target inside a training run, late and shrink-only.
- Zhang & Sui, *GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3DGS*, arXiv:2605.02086. A storage budget as a training input, and an honest account of why it does not bind.
- Tian et al., *FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3DGS*, ACM MM 2025, arXiv:2507.06671.
- Morgenstern et al., *KISS-GS: 3D Gaussian Splatting Compression Kept Simple*, ECCV 2026, arXiv:2608.26948. The argument that compression and training should stay decoupled, worth reading as the counter-position to this project.
- Xie et al., *MesonGS: Post-training Compression of 3D Gaussians*, ECCV 2024, arXiv:2409.09756. The predecessor of both size-targeted baselines above.
- Zhang et al., *ControlGS: Consistent Structural Compression Control for Deployment-Aware Gaussian Splatting*, arXiv:2505.10473. A single knob hand-mapped to device tiers, the practice this project is trying to replace with a measured constraint.

**Budgets denominated in primitive count**

- Mallick et al., *Taming 3DGS: High-Quality Radiance Fields with Limited Resources*, SIGGRAPH Asia 2024, arXiv:2406.15643. A count budget scheduled from the first iteration.
- Fang & Wang, *Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians*, ECCV 2024, arXiv:2403.14166, and Zhang et al., *GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework*, CVPR 2025, arXiv:2411.06019. Over-densify, then sparsify, the strategy a budget schedule has to beat.
- Hanson et al., *PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting*, CVPR 2025, arXiv:2406.10219. The post-hoc pruning baseline.
- Liu et al., *MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks*, CVPR 2025, arXiv:2412.20522. Pruning as a learned mask inside training rather than a threshold after it.
- Chen et al., *SPARE-GS: Structural Parsimony and Resource Efficiency for 3DGS*, arXiv:2607.16624. The allocation rule this project restates in bytes: spend where marginal utility per unit cost is highest, equalised across regions.
- Bisht & Kolb, *Smart target point control for Gaussian Splatting methods*, arXiv:2605.16158, and Zhang et al., *Gaussians on a Diet: High-Quality Memory-Bounded 3DGS Training*, arXiv:2604.20046. How a budget should be approached over time, smoothly rather than as an abrupt cap.
- Fan et al., *LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction*, NeurIPS 2024, arXiv:2311.17245, and Lee et al., *Compact 3D Gaussian Representation for Radiance Field*, CVPR 2024, arXiv:2311.13681. The standard compaction pipelines a new method is measured against.

**Constraints that are not bytes on disk**

- Chen et al., *MEGS²: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning*, ICLR 2026, arXiv:2509.07021. Rendering memory as a constrained optimization.
- Lin et al., *MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning and Accelerated Foveated Rendering*, ASPLOS 2025, arXiv:2407.00435. A per-primitive cost proxy validated against measured latency on a Jetson.
- Hanson et al., *Speedy-Splat: Fast 3DGS with Sparse Pixels and Sparse Primitives*, CVPR 2025, arXiv:2412.00578, and Seo et al., *FLoD: Integrating Flexible Level of Detail into 3DGS for Customizable Rendering*, ACM TOG 2025, arXiv:2408.12894.
- Tajwar et al., *Splats under Pressure: Performance-Energy Trade-offs in Real-Time 3DGS under Constrained GPU Budgets*, arXiv:2604.07177, and Pająk et al., *HiGS: A Hierarchical Rendering Architecture for Real-Time 3DGS*, arXiv:2606.00352. What actually costs time and memory at render time, measured rather than assumed.
- Luo et al., *RoofGS: Roofline-Guided End-to-End Acceleration of 3DGS*, arXiv:2608.15785, and Jo et al., *AdaGScale: Viewpoint-Adaptive Gaussian Scaling to Reduce Gaussian-Tile Pairs*, DAC 2026, arXiv:2604.18980. A per-stage cost model of the rasterizer, the form a frame-time predictor would take.
- Liu et al., *FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3DGS*, CVPR 2025, arXiv:2506.04174, and Guo et al., *Matryoshka Gaussian Splatting*, arXiv:2603.19234. One model serving many budgets at inference, the alternative answer to the same deployment problem.

**Background from learned compression**

- Ballé et al., *Variational image compression with a scale hyperprior*, ICLR 2018, arXiv:1802.01436. The entropy model, the quantization proxy and the rate term this field inherited.
- Choi et al., *Variable Rate Deep Image Compression With a Conditional Autoencoder*, ICCV 2019, arXiv:1909.04802, and Cui et al., *Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation*, CVPR 2021, arXiv:2003.02012. Per-channel gain vectors as a fast, exact rate actuator, which is where this project's quantization-step lever comes from.

**Surveys, for the overview and the leaderboard**

- Bagdasarian et al., *3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods*, CGF 2025, arXiv:2407.09510. The maintained leaderboard and the protocol.
- Ali et al., *Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions*, arXiv:2502.19457.
