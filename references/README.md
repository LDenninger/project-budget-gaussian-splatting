# references

Related-work library for the **BudgetGS** project: a training-time controller that takes a
resource target (bytes on disk first, then bytes in device memory, then milliseconds per frame) and
drives a 3D Gaussian Splatting model onto it.

Every PDF here was fetched from arXiv and its id, title, first author and submission month were
verified against the arXiv API before download (36 on 2026-09-08 in a first survey, 31 more
later the same day). Entries whose venue line reads `arXiv <year-month>` have a peer-reviewed
venue that was **not** verified here. Every paper was read in full during the 2026-09-08 sessions.
The per-paper reading notes live in `scratchpad/08_09_26_14_35_literature_notes/`, the transcribed
result tables in `results/`, and the synthesis in
`docs/02-research-program.md`.

## Table of Contents

- [Project context](#project-context)
- [Folder structure](#folder-structure)
- [00 — Foundations](#00--foundations)
- [01 — Size-targeted compression (the direct ancestors)](#01--size-targeted-compression-the-direct-ancestors)
- [02 — Rate-distortion training and entropy models](#02--rate-distortion-training-and-entropy-models)
- [03 — Count-budget training](#03--count-budget-training)
- [04 — Compaction and pruning](#04--compaction-and-pruning)
- [05 — Surveys](#05--surveys)
- [06 — Rate-control theory](#06--rate-control-theory)
- [07 — Runtime cost: render time, runtime memory, edge devices](#07--runtime-cost-render-time-runtime-memory-edge-devices)
- [Transcribed results](#transcribed-results)
- [The gap this library maps](#the-gap-this-library-maps)
- [Evaluation protocol the field uses](#evaluation-protocol-the-field-uses)
- [Not included](#not-included)

## Project context

The sibling project `~/Research/textured_triangle_splatting` already runs a budget controller whose
target is a **learnable parameter count**: `compute_spend` returns
`max(min(target_params, 1.3 · params_now) − params_now, 0) + freed`
(`workspaces/textured-triangle-splatting/textured_triangle_splatting/model/budget.py:51-52`), with
`target_params` set per scene in the experiment configs.

This project changes the currency from parameters to **stored bytes**, and later to bytes of device
memory and to frame time. That is not a rescaling: the byte count of a 3DGS model is produced by a
quantizer and an entropy coder, so the map from model state to bytes is neither linear in the
primitive count nor known in closed form. Each section below says *why* a paper is here. Read the
"→" line, not just the title.

## Folder structure

```
references
├── 00_foundations              # 3DGS/2DGS lineage the representation comes from
├── 01_size_targeted            # Methods that take a size in MB as input, the closest prior art
├── 02_rate_distortion          # D + λ·R training and the entropy models that make R differentiable
├── 03_count_budget             # Budgets denominated in primitive count, and elastic or anytime models
├── 04_compaction_pruning       # Post-hoc size reduction without an explicit rate term
├── 05_surveys                  # Aggregated size-vs-quality tables, the field's scoreboard
├── 06_rate_control_theory      # Variable-rate / target-rate machinery from learned image compression
├── 07_runtime_cost             # Render time, runtime memory, edge devices, LOD, accelerators
├── results                     # Evaluation tables transcribed out of the PDFs, as CSV
├── manifest.txt                # dir|slug|arxiv-id, the regeneration recipe for every PDF
└── README.md
```

## 00 — Foundations

| Paper | Venue | File |
|---|---|---|
| Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* | SIGGRAPH 2023 | `3dgs_kerbl_siggraph2023_arxiv2308.04079.pdf` |
| Huang et al., *2D Gaussian Splatting for Geometrically Accurate Radiance Fields* | SIGGRAPH 2024 | `2dgs_huang_siggraph2024_arxiv2403.17888.pdf` |
| Yu et al., *Mip-Splatting: Alias-free 3D Gaussian Splatting* | CVPR 2024 | `mip_splatting_yu_cvpr2024_arxiv2311.16493.pdf` |
| Lu et al., *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering* | CVPR 2024 | `scaffold_gs_lu_cvpr2024_arxiv2312.00109.pdf` |
| Kheradmand et al., *3D Gaussian Splatting as Markov Chain Monte Carlo* | NeurIPS 2024 | `3dgs_mcmc_kheradmand_neurips2024_arxiv2404.09591.pdf` |

→ 3DGS defines the primitive whose 59 floats per Gaussian are the thing being counted, and the
adaptive-density-control loop any size controller has to interpose on. Its p. 11 puts the
rasterizer's own working memory at 30 to 500 MB beside the model, which is why bytes on disk and
bytes in VRAM are different budgets. 2DGS is the surfel variant with a different payload.
**Scaffold-GS matters more than its citation count suggests**: almost every entropy-coded method
in `02_rate_distortion` is built on its anchors, and its per-anchor MLPs run at render time, so
the choice of base model decides which prior art is comparable and whether a runtime bound is
even expressible. 3DGS-MCMC replaces heuristic densification with a relocation sampler under a
fixed maximum count, and states on p. 2 that heuristic ADC makes it "difficult to control the
computation and memory budget in advance", which is the sentence this project starts from.

## 01 — Size-targeted compression (the direct ancestors)

| Paper | Venue | File |
|---|---|---|
| Xie et al., *SizeGS: Size-aware Compression of 3D Gaussian Splatting via Mixed Integer Programming* | ACM MM 2025 | `sizegs_xie_acmmm2025_arxiv2412.05808.pdf` |
| Xie et al., *MesonGS++: Post-training Compression of 3DGS with Hyperparameter Searching* | arXiv 2026-04 | `mesongs_plusplus_xie_2026_arxiv2604.26799.pdf` |
| Yang et al., *HybridGS: High-Efficiency Gaussian Splatting Data Compression using Dual-Channel Sparse Representation and Point Cloud Encoder* | ICML 2025 | `hybridgs_yang_icml2025_arxiv2505.01938.pdf` |
| Zhang & Sui, *GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3DGS* | arXiv 2026-05 | `geta_3dgs_zhang_2026_arxiv2605.02086.pdf` |
| Tian et al., *FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3DGS* | ACM MM 2025 | `flexgaussian_tian_acmmm2025_arxiv2507.06671.pdf` |
| Morgenstern et al., *KISS-GS: 3D Gaussian Splatting Compression Kept Simple* | ECCV 2026 | `kiss_gs_morgenstern_eccv2026_arxiv2608.26948.pdf` |

→ **The works that accept a size as an input.** SizeGS and MesonGS++ are post-hoc: a frozen 30 k
checkpoint, a linear size estimator S(Q) = Σ P_ij Q_ij + C + S_Δ calibrated by real encodes, and a
0-1 ILP over per-group bit widths and the reserve ratio. SizeGS lands within 1 % in about a
minute, MesonGS++ within 0.3 to 0.9 %. **SizeGS is the direct ancestor and the baseline to
beat**, and its p. 2 names the failure mode of the λ family: "it is difficult to predict the final
size given a certain λ".

**GETA-3DGS and HybridGS are the two papers that put a byte budget inside training, and neither
binds it.** GETA-3DGS's abstract frames the budget as a user handle, but **the paper itself
discloses that the budget does not bind**: its Fig. 2 and Fig. 6 show all eight target sizes
B in {3, 5, 8, 10, 12, 30, 60, 120} MB landing in 4.0 to 4.2 MB, because B enters as "a sparsity
penalty rather than a hard byte cap" (p. 11), and it names "sub-gradient ascent on the Lagrangian
dual of the byte-counter constraint" as future work, which is this project. Those rows are
transcribed in `results/01_size_targeted/geta_3dgs_zhang_2026.csv`. HybridGS takes B at
iteration 36 000 of 70 000, after densification has ended, and applies one of two mutually
exclusive shrink levers (prune, or lower bit depths) with a hand-set G-PCC ratio L = 1.3, missing
its Table 4 targets by −45 % to +8 %. FlexGaussian is the training-free, seconds-scale post-hoc
end (quality-drop or ratio target). KISS-GS deliberately decouples compression from training and
encodes into web-native image formats, and its "training-format coupling" argument (p. 2 to 3) is
the position a training-time method has to answer.

## 02 — Rate-distortion training and entropy models

| Paper | Venue | File |
|---|---|---|
| Wang et al., *End-to-End Rate-Distortion Optimized 3D Gaussian Representation* (RDO-Gaussian) | ECCV 2024 | `rdo_gaussian_wang_eccv2024_arxiv2406.01597.pdf` |
| Chen et al., *HAC: Hash-grid Assisted Context for 3DGS Compression* | ECCV 2024 | `hac_chen_eccv2024_arxiv2403.14530.pdf` |
| Chen et al., *HAC++: Towards 100X Compression of 3D Gaussian Splatting* | TPAMI 2025 | `hac_plusplus_chen_tpami2025_arxiv2501.12255.pdf` |
| Wang et al., *ContextGS: Compact 3DGS with Anchor Level Context Model* | NeurIPS 2024 | `contextgs_wang_2024_arxiv2405.20721.pdf` |
| Liu et al., *HEMGS: A Hybrid Entropy Model for 3DGS Data Compression* | arXiv 2024-11 | `hemgs_liu_2024_arxiv2411.18473.pdf` |
| Zhan et al., *CAT-3DGS: Context-Adaptive Triplane Approach to RD-Optimized 3DGS Compression* | ICLR 2025 | `cat_3dgs_zhan_iclr2025_arxiv2503.00357.pdf` |
| Chen et al., *PCGS: Progressive Compression of 3D Gaussian Splatting* | arXiv 2025-03 | `pcgs_chen_2025_arxiv2503.08511.pdf` |
| Xu et al., *Improving 3DGS Compression by Scene-Adaptive Lattice Vector Quantization* (SALVQ) | IEEE TIP 2025 | `salvq_xu_tip2025_arxiv2509.13482.pdf` |
| Tran et al., *RAVE: Rate-Adaptive Visual Encoding for 3DGS* | ICASSP 2026 | `rave_tran_2025_arxiv2512.07052.pdf` |
| Chen et al., *Fast Feedforward 3D Gaussian Splatting Compression* (FCGS) | ICLR 2025 | `fcgs_chen_2024_arxiv2410.08017.pdf` |
| Liu et al., *CompGS: Efficient 3D Scene Representation via Compressed Gaussian Splatting* | ACM MM 2024 | `compgs_liu_2024_arxiv2404.09458.pdf` |
| Xie et al., *MesonGS: Post-training Compression of 3D Gaussians* | ECCV 2024 | `mesongs_xie_eccv2024_arxiv2409.09756.pdf` |
| Wang et al., *Smol-GS: Compact Representations for Abstract 3DGS* | arXiv 2025-11 | `smol_gs_wang_2025_arxiv2512.00850.pdf` |
| Martin et al., *Structured Image-based Coding for Efficient Gaussian Splatting Compression* (GSICO) | arXiv 2026-01 | `gsico_martin_2026_arxiv2601.14510.pdf` |
| Lee et al., *Compression of 3DGS with Optimized Feature Planes and Standard Video Codecs* (CodecGS) | arXiv 2025-01 | `codecgs_lee_2025_arxiv2501.03399.pdf` |

→ **This is where a differentiable estimate of the byte count lives**, and where the size-versus-
quality frontier currently sits (Smol-GS 27.76 dB at 5.9 MB, HEMGS 27.93 dB at 21 MB on
Mip-NeRF 360 per the leaderboard). Every method trains D + λ·R with λ a knob, and the per-scene
size at one λ spreads 3.2× to 5.5× inside Mip-NeRF 360 (HAC: bicycle 27.54 MB, room 5.53 MB).
In HAC, HAC++ and PCGS the rate loss is a per-parameter bit average, so the objective never sees a
total byte count, and no paper reports the gap between the entropy estimate and the encoded
bytes. HEMGS, SALVQ, PCGS and RAVE give one model at several rates, but the operating points are
λ-indexed and SALVQ says one variable-rate model "is not yet sufficient to cover a broad operating
range" (p. 12). Smol-GS carries the two levers a byte controller needs, a count weight λ_o and a
bits weight λ_q, at the cost of four MLPs per frame at render time. GSICO and CodecGS are the
image- and video-codec routes, MesonGS is SizeGS's base, FCGS the feed-forward variant.

## 03 — Count-budget training

| Paper | Venue | File |
|---|---|---|
| Mallick et al., *Taming 3DGS: High-Quality Radiance Fields with Limited Resources* | SIGGRAPH Asia 2024 | `taming_3dgs_mallick_sigasia2024_arxiv2406.15643.pdf` |
| Fang & Wang, *Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians* | ECCV 2024 | `mini_splatting_fang_eccv2024_arxiv2403.14166.pdf` |
| Zhang et al., *GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework* | CVPR 2025 | `gaussianspa_zhang_cvpr2025_arxiv2411.06019.pdf` |
| Liu et al., *MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks* | CVPR 2025 | `maskgaussian_liu_cvpr2025_arxiv2412.20522.pdf` |
| Hanson et al., *PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting* | CVPR 2025 | `pup_3dgs_hanson_cvpr2025_arxiv2406.10219.pdf` |
| Chen et al., *SPARE-GS: Structural Parsimony and Resource Efficiency for 3DGS* | arXiv 2026-07 | `spare_gs_chen_2026_arxiv2607.16624.pdf` |
| Bisht & Kolb, *Smart target point control for Gaussian Splatting methods* | arXiv 2026-05 | `smart_target_point_control_bisht_2026_arxiv2605.16158.pdf` |
| Zheng et al., *Constrained Dynamic Gaussian Splatting* | arXiv 2026-02 | `constrained_dynamic_gs_zheng_2026_arxiv2602.03538.pdf` |
| Zhang et al., *ControlGS: Consistent Structural Compression Control for Deployment-Aware Gaussian Splatting* | arXiv 2025-05 | `controlgs_zhang_2025_arxiv2505.10473.pdf` |
| Guo et al., *Matryoshka Gaussian Splatting* | arXiv 2026-03 | `matryoshka_gs_guo_2026_arxiv2603.19234.pdf` |
| Zhang et al., *Gaussians on a Diet: High-Quality Memory-Bounded 3DGS Training* | arXiv 2026-04 | `gaussians_on_a_diet_zhang_2026_arxiv2604.20046.pdf` |
| Jia et al., *You Only Gaussian Once: Controllable 3DGS for Ultra-Densely Sampled Scenes* (YOGO) | arXiv 2026-04 | `yogo_jia_2026_arxiv2604.21400.pdf` |
| Liu et al., *FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3DGS* | CVPR 2025 | `flexgs_liu_cvpr2025_arxiv2506.04174.pdf` |
| Baranowski et al., *ConeGS: Error-Guided Densification Using Pixel Cones* | arXiv 2025-11 | `conegs_baranowski_2025_arxiv2511.06810.pdf` |

→ **The currency the existing controller already speaks**, and the closest structural analogues to
port. *Taming 3DGS* spends a per-event allowance on score-ranked candidates along a parabolic
count schedule and hits its count exactly. *Smart Target Point Control* is a quota governor on the
native thresholds with a deadband and a prune lockout, and shows smooth tracking beats an abrupt
cap. *Constrained Dynamic GS* is the best controller template: a differentiable soft gate with a
quadratic count penalty on the Taming schedule, hitting the target within 0.1 to 1.4 %, and losing
1.24 dB when its unconstrained warm-up is removed. *GaussianSpa* is the penalty-projection
alternation, *SPARE-GS* the marginal-utility optimality condition ∂U_r/∂n_r = λ(t) across regions,
*ControlGS* an explicit explore-then-sparsify with 8× overshoot rounds. **The evidence they give
together:** at matched count, Taming (budget from iteration 0) sits below GaussianSpa and
Mini-Splatting (over-densify, then sparsify), and no paper runs the clean test of train-to-N
against prune-to-N with fine-tuning. *Matryoshka* and *FlexGS* are the elastic route (any prefix
or fraction at inference), *Gaussians on a Diet* caps count as a training-memory proxy (2.98 GB
peak on a Jetson AGX Xavier), and *YOGO* and *ConeGS* are per-region and error-guided budgets.
None of the fourteen binds bytes.

## 04 — Compaction and pruning

| Paper | Venue | File |
|---|---|---|
| Fan et al., *LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction* | NeurIPS 2024 | `lightgaussian_fan_neurips2024_arxiv2311.17245.pdf` |
| Lee et al., *Compact 3D Gaussian Representation for Radiance Field* | CVPR 2024 | `compact_3dgs_lee_cvpr2024_arxiv2311.13681.pdf` |
| Morgenstern et al., *Compact 3D Scene Representation via Self-Organizing Gaussian Grids* | ECCV 2024 | `self_organizing_gaussians_morgenstern_2023_arxiv2312.13299.pdf` |
| Niedermayr et al., *Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis* | CVPR 2024 | `compressed_3dgs_niedermayr_cvpr2024_arxiv2401.02436.pdf` |
| Girish et al., *EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS* | arXiv 2023-12 | `eagles_girish_2023_arxiv2312.04564.pdf` |
| Navaneet et al., *CompGS: Smaller and Faster Gaussian Splatting with Vector Quantization* (Compact3D) | arXiv 2023-11 | `compact3d_navaneet_2023_arxiv2311.18159.pdf` |

→ The size-reduction baselines that predate the rate-distortion framing, and the ones every table
in `05_surveys` reports against. Self-Organizing Gaussians makes size a function of a layout and a
codec quality rather than a per-primitive bit allocation, and says "it would be interesting to
perform the quantization during training" (p. 14). Compressed 3DGS renders its quantized layout
directly and is the one paper here with low-end numbers (16 FPS on an Intel UHD iGPU, 83 FPS on a
Radeon R9 380 at 1080p), and it reports that folding positional constraints into training did not
work (p. 8). The other four are knobs with no target.

## 05 — Surveys

| Paper | Venue | File |
|---|---|---|
| Bagdasarian et al., *3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods* | CGF 2025 | `3dgs_zip_bagdasarian_cgf2025_arxiv2407.09510.pdf` |
| Ali et al., *Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions* | arXiv 2025-02 | `compression_3dgs_survey_ali_2025_arxiv2502.19457.pdf` |

→ **Read 3DGS.zip first.** It supplies the taxonomy this library's folder split follows,
*compression* (reduce file size) versus *compaction* (reduce Gaussian count), and the maintained
leaderboard (<https://w-m.github.io/3dgs-compression-survey/>, raw CSVs in the
`w-m/3dgs-compression-survey` repository, MIT). Its p. 18 names "changing hardware constraints"
and evaluation "on real-time performance and energy efficiency, particularly for deployment on
mobile and embedded systems" as open directions, and its p. 8 notes that for structured methods
"reducing the number of primitives does not further decrease VRAM consumption". Neither survey
tabulates training time, decode time, peak VRAM or FPS, and their sizes differ by a factor of
1.045 (MiB against MB).

## 06 — Rate-control theory

| Paper | Venue | File |
|---|---|---|
| Ballé et al., *Variational image compression with a scale hyperprior* | ICLR 2018 | `scale_hyperprior_balle_iclr2018_arxiv1802.01436.pdf` |
| Choi et al., *Variable Rate Deep Image Compression With a Conditional Autoencoder* | ICCV 2019 | `variable_rate_conditional_ae_choi_iccv2019_arxiv1909.04802.pdf` |
| Cui et al., *Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation* | CVPR 2021 | `gained_vae_rate_adaptation_cui_cvpr2021_arxiv2003.02012.pdf` |

→ Where the D + λ·R objective every method in `02_rate_distortion` uses comes from, and where the
"one model, arbitrary rate" problem was solved first. None of the three gives a procedure to reach
a target rate. What transfers: the noise relaxation and the hyperprior unchanged, Cui's per-channel
gain vectors as the cheapest exact rate knob, and the observation that a per-scene model is one
sample, so the rate is measurable exactly and λ can become a dual variable. The ancestors that do
hit a target live outside this library and are cited in the research program: the HEVC R-λ model
(Li et al., IEEE TIP 2014), feedback rate control for learned video coding (Xu et al., arXiv
2604.20104), the proxy-Lagrangian (Cotter et al., ALT 2019, arXiv 1804.06500), controlled sparsity
via dual ascent (Gallego-Posada et al., NeurIPS 2022, arXiv 2208.04425), hardware-aware NAS
(ProxylessNAS, arXiv 1812.00332), budget-aware pruning (Lemaire et al., CVPR 2019, arXiv
1811.09332) and byte-denominated memory penalties (Uhlich et al., ICLR 2020, arXiv 1905.11452).

## 07 — Runtime cost: render time, runtime memory, edge devices

| Paper | Venue | File |
|---|---|---|
| Lin et al., *MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning and Accelerated Foveated Rendering* | ASPLOS 2025 | `metasapiens_lin_asplos2025_arxiv2407.00435.pdf` |
| Hanson et al., *Speedy-Splat: Fast 3DGS with Sparse Pixels and Sparse Primitives* | CVPR 2025 | `speedy_splat_hanson_cvpr2025_arxiv2412.00578.pdf` |
| Chen et al., *MEGS²: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning* | ICLR 2026 | `megs2_chen_iclr2026_arxiv2509.07021.pdf` |
| Seo et al., *FLoD: Integrating Flexible Level of Detail into 3DGS for Customizable Rendering* | ACM TOG 2025 | `flod_seo_2024_arxiv2408.12894.pdf` |
| Cheng et al., *CLoD-GS: Continuous Level-of-Detail via 3DGS* | ICLR 2026 | `clod_gs_cheng_iclr2026_arxiv2510.09997.pdf` |
| Tajwar et al., *Splats under Pressure: Performance-Energy Trade-offs in Real-Time 3DGS under Constrained GPU Budgets* | arXiv 2026-04 | `splats_under_pressure_tajwar_2026_arxiv2604.07177.pdf` |
| Pająk et al., *HiGS: A Hierarchical Rendering Architecture for Real-Time 3DGS* | arXiv 2026-05 | `higs_pajak_2026_arxiv2606.00352.pdf` |
| Jo et al., *AdaGScale: Viewpoint-Adaptive Gaussian Scaling to Reduce Gaussian-Tile Pairs* | DAC 2026 | `adagscale_jo_dac2026_arxiv2604.18980.pdf` |
| Luo et al., *RoofGS: Roofline-Guided End-to-End Acceleration of 3DGS* | arXiv 2026-08 | `roofgs_luo_2026_arxiv2608.15785.pdf` |
| Ye et al., *Gaussian Blending Unit: An Edge GPU Plug-in for Real-Time Gaussian-Based Rendering* | HPCA 2025 | `gaussian_blending_unit_ye_hpca2025_arxiv2503.23625.pdf` |
| Wei et al., *No Redundancy, No Stall: Lightweight Streaming 3DGS for Real-time Rendering* (LS-Gaussian) | ICCAD 2025 | `ls_gaussian_wei_iccad2025_arxiv2507.21572.pdf` |
| Du et al., *Mobile-GS: Real-time Gaussian Splatting for Mobile Devices* | arXiv 2026-03 | `mobile_gs_du_2026_arxiv2603.11531.pdf` |
| Han et al., *WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers* | arXiv 2026-02 | `websplatter_han_2026_arxiv2602.03207.pdf` |
| Li et al., *Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM* | ICRA 2026 | `pocket_slam_li_icra2026_arxiv2606.24796.pdf` |
| Ren et al., *Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians* | arXiv 2024-03 | `octree_gs_ren_2024_arxiv2403.17898.pdf` |
| Feng et al., *FlashGS: Efficient 3DGS for Large-scale and High-resolution Rendering* | arXiv 2024-08 | `flashgs_feng_2024_arxiv2408.07967.pdf` |

→ **The second phase of the project, and the family the first survey did not cover.** What frame time
depends on, according to these papers: Gaussian-tile pairs (AdaGScale puts pair generation, sort
and rasterisation at 88.1 % of the frame), per-pixel fragments (GBU measures 541 fragments per
Gaussian with 7.6 % contributing on a Jetson Orin NX), sorting bandwidth (RoofGS, Mobile-GS on a
Snapdragon 8 Gen 3), and tile load imbalance (LS-Gaussian on an AGX Orin). The bottleneck stage
moves with the device (WebSplatter: render-bound on an RTX 3070, preprocess-bound on an Intel
iGPU) and frame time is not monotone in count. **No paper fits a latency predictor and reports its
error, and none takes milliseconds or VRAM bytes as an input.** MetaSapiens' per-point efficiency
CE_i = Val_i / Comp_i is the only cost proxy validated against measured latency (102.2 FPS on a
Jetson AGX Xavier where prior models ran below 10). MEGS² is the only training method that
targets rendering VRAM, through an ADMM constraint in parameters, with 91.0 FPS on a Dimensity
9400+ phone against 6.6 for 3DGS. FLoD's levels are the memory tiers (an MX250 with 2 GB runs out
of memory at the finest level), CLoD-GS regularises the rendered count, Splats under Pressure and
HiGS give FPS against count per GPU tier, Octree-GS bounds the rendered count per view, FlashGS
is the fast rasterizer, Pocket-SLAM prunes by rendering area under a count budget.

## Transcribed results

`results/` holds the evaluation tables read out of these PDFs, one CSV per paper plus a merged
`results/all_results.csv`: 3 787 rows over 31 papers, with the source table and page on every row.
Read `results/README.md` before using them. It records the schema, the MB-versus-MiB convention,
and the cross-paper conflicts that make a naive PSNR comparison wrong, including the fact that
GETA-3DGS's storage budget is non-binding by its own disclosure. The 31 papers added on
2026-09-08 afternoon are not transcribed there yet.

## The gap this library maps

Splitting the families by what the user supplies and when it binds:

| Family | User supplies | Binds | Hits an exact byte target |
|---|---|---|---|
| `01` post-hoc size search (SizeGS, MesonGS++, FlexGaussian) | MB, ratio or count | after training | yes, within 0.3 to 1 %, on a frozen model |
| `01` byte budget as training input (GETA-3DGS) | MB | during training | no, non-binding |
| `01` byte target in a shrink stage (HybridGS) | MB, at iteration 36 000 of 70 000 | late, shrink only | no, −45 % to +8 % |
| `02` rate-distortion with λ | λ | during training | no, 3.2× to 5.5× per-scene spread |
| `02` variable rate | λ index, gain, level | encode time | no, discrete points |
| `03` count budget | count or trajectory | during training | no, count is not bytes under compression |
| `03` elastic, anytime | fraction or level at inference | during training | no |
| `07` runtime resources | sparsity targets, pruning fraction | training or post hoc | no, none takes VRAM bytes or ms |

The empty cells: **an entropy-coded byte target bound on both growth and precision from the
first iteration, hit without a sweep**, **a runtime-memory target in bytes**, and **a frame-time
target on a named device**. The full argument, the holes, the directions, the kill experiments and
the roadmap are in `docs/02-research-program.md`.

## Evaluation protocol the field uses

Every paper in `01`, `02` and `04` reports on the same four datasets, and the comparison is a curve
of size in MB against quality, never a single number:

- **Mip-NeRF 360** — 9 scenes including `flowers` and `treehill`, test images downscaled to at most
  1600 px on the long side, every 8th image held out
- **Tanks and Temples** — `truck`, `train`
- **Deep Blending** — `drjohnson`, `playroom`
- **NeRF Synthetic** — 8 scenes, the predefined split

Metrics: PSNR, SSIM, LPIPS, size in MB with 1 MB = 10⁶ bytes, and increasingly encode and decode
time. The leaderboard's Pareto frontier on Mip-NeRF 360 (reported): Smol-GS base 27.76 dB at
5.86 MB, Smol-GS large 27.86 dB at 10.40 MB, HEMGS high rate 27.93 dB at 21.00 MB. Below 5.86 MB
no method is listed, and the next points down (RDO-Gaussian 26.03 dB at 6.16 MB, gsplat
compression 26.64 dB at 6.92 MB) are 1.1 to 1.7 dB lower, so byte targets matter below 6 MB. The
size-aware table to match is SizeGS Table 1 (p. 6), which fixes the budget per dataset:

| Method | Mip-NeRF 360 @ 18.33 MB | Tanks&Temples @ 11 MB | Deep Blending @ 8 MB |
|---|---|---|---|
| HAC | 27.17 dB / 18.29 MB / 16 627 s | 24.45 dB / 10.92 MB / 10 978 s | 30.27 dB / 8.29 MB / 9 993 s |
| SizeGS | 27.48 dB / 18.17 MB / 1 328 s | 24.04 dB / 10.93 MB / 1 381 s | 30.24 dB / 7.92 MB / 1 263 s |

Numbers as reported in the SizeGS paper, not measured here. SizeGS's 27.48 dB at 18.17 MB sits
0.38 dB below the leaderboard frontier at 10.40 MB, and its HAC column is the 2024 HAC with a λ
binary search, not HAC++ or HEMGS.

## Not included

- 4D / dynamic Gaussian compression (P-4DGS, PD-4DGS and relatives), except Constrained Dynamic GS,
  which is here for its controller and not for its dynamics.
- Feed-forward / generalisable 3DGS reconstruction (FCGS is the exception, as a compression
  baseline).
- Hardware accelerator papers beyond GBU and LS-Gaussian (GSCore, GauSPU, GSArch, GCC, DeGS, Neo),
  streaming systems (GSStream, LTS, L3GS), and the training-speed line (DashGaussian, Turbo-GS,
  FastGS). They are listed with verified ids in the gap-search notes.
- Mesh and point-cloud codecs (G-PCC, Draco) beyond their use inside MesonGS, SizeGS and HybridGS.
