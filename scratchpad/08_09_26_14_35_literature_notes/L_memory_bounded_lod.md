# L_memory_bounded_lod: count budgets, training-memory bounds and level-of-detail

Reading notes for five papers that bound training memory, primitive count, rendered count or build
level-of-detail hierarchies. Every number below is copied from the PDF and marked **reported**
unless it is explicitly labelled as a note-taker derivation. Quotes carry a page number from the
PDF page index (not the printed page number, which differs for FLoD and ConeGS appendices).

Source PDFs:

| short name | path |
|---|---|
| Diet | `references/03_count_budget/gaussians_on_a_diet_zhang_2026_arxiv2604.20046.pdf` |
| YOGO | `references/03_count_budget/yogo_jia_2026_arxiv2604.21400.pdf` |
| CLoD-GS | `references/07_runtime_cost/clod_gs_cheng_iclr2026_arxiv2510.09997.pdf` |
| FLoD | `references/07_runtime_cost/flod_seo_2024_arxiv2408.12894.pdf` |
| ConeGS | `references/03_count_budget/conegs_baranowski_2025_arxiv2511.06810.pdf` |

---

## 1. Gaussians on a Diet

### 1.1 Citation

- Title: "Gaussians on a Diet: High-Quality Memory-Bounded 3D Gaussian Splatting Training" (p1).
- First author: Yangming Zhang, Dept. of Computer Science, University of Texas at Arlington.
  Corresponding author Miao Yin (UTA). Co-authors from Georgia Tech, University of Georgia,
  University of Minnesota Twin Cities, Snap Inc.
- Venue: **not stated**. The PDF carries only the arXiv stamp "arXiv:2604.20046v2 [cs.CV] 23 Apr
  2026" (p1). The layout is the CVPR/ICCV two-column template.
- arXiv id: 2604.20046 (from file name and p1 stamp).

### 1.2 Base representation and renderer

Vanilla 3DGS (Kerbl et al. 2023), anisotropic 3D Gaussians with position, scale, opacity, rotation
and SH colour, standard tile-based alpha-blending rasteriser. Equations 1-3 on p3 are the plain
3DGS forward model. No renderer modification is described. The training loop is compared against
the 3DGS and Taming 3DGS codebases and "All render quality experiments are conducted under the
same environment specified in the original 3DGS [17] and Taming 3DGS [25] using an NVIDIA GTX 4090
GPU" (p7).

### 1.3 Budget handle

**TARGET, denominated in primitive count.** Algorithm 1 (p6) takes:

> "Input: Gaussian primitives G, target peak number of Gaussians F, maximum number of iterations T"

So the handle is a hard cap F on the number of Gaussians. The paper's *claim* is about peak
training memory in GB, but the user never supplies a GB number. GB is a measured outcome, not an
input.

**Achieved versus requested.** The paper never prints the requested F for any scene. It prints only
the achieved peak count, column "#G/M" ("the peak number of Gaussians in training (in millions)",
Table 1 caption p8). No achieved-versus-requested pair exists anywhere in the paper. Achieved peak
counts (reported, Table 1 p8):

| dataset | Ours #G/M | Taming 3DGS #G/M | 3DGS #G/M |
|---|---|---|---|
| Mip-NeRF 360 | 0.628 | 0.632 | 3.310 |
| Tanks&Temples | 0.318 | 0.319 | 1.840 |
| Deep Blending | 0.292 | 0.294 | 2.810 |

The achieved count sits 0.001-0.004 M below the Taming 3DGS peak in every row, which reads as F
being set to Taming 3DGS's peak count for a matched comparison, but the paper does not say so.

Per-scene achieved peaks (reported, Tables 3-5, p11), Ours: bicycle 0.800, bonsai 0.410, counter
0.310, flowers 0.570, garden 1.900, kitchen 0.480, room 0.220, stump 0.480, treehill 0.480,
train 0.365, truck 0.270, drjohnson 0.400, playroom 0.185 (all in millions).

### 1.4 The resource model

**There is no resource model.** The paper never writes an equation mapping count to peak training
memory in bytes. It asserts the direction ("keeping the total peak training memory under the
constraint", p6) and then measures the memory empirically. Two measurement sets exist:

Figure 1 axis labels (reported, p1, on "GTX 4090", peak training memory usage in GB):
Ours 8.86, Taming 3DGS 9.00, Mini-Splatting 12.50, EAGLES 13.08, 3DGS 18.29, Reducing-3DGS 19.74,
Compact-3DGS 20.09.

Table 2(b) "Peak training memory usage on NVIDIA Jetson Xavier", Mem usage (GB) (reported, p8):
3DGS 18.59, Taming 3DGS 10.01, Ours 8.55, Ours w/loader 2.98.

Note the footnote qualifying the Figure 1 measurement (p2):

> "1Here, we report our peak memory usage on GTX 4090 without engineering optimization for a fair
> comparison, while Mini-Splatting [9] leverages extra compression by downgrading the orders of SH
> coefficients to one before pruning to reduce memory."

Stated accuracy of a count → memory model: **not stated**, because no such model is given.

A large part of the measured memory is not the model at all (p7):

> "Notably, we observed that up to three-quarters memory is used for dataset storage, we develop a
> parallel dataloader that dynamically prefetches and moves data between the storage and the memory
> according to the training pipeline. This effort further reduces peak training memory by more than
> 5 GB."

### 1.5 The control law

Alternating grow / compensate / prune with a hard count cap, run every 50 iterations (growing and
pruning) and every 100 iterations (importance recomputation).

Growing criterion, mixed gradient (p6):

  ∇mix = ∇p + ∇c

where ∇p is the view-space positional gradient and ∇c the colour gradient. Derivations on p5:
dℓ/dp_k = (dℓ/dC)(dC/dα_k)(dα_k/dp_k) with dC/dα_k = ∑[j=k..N] −c_k α_k Π(1 − α_k), and
dℓ/dc_k = (dℓ/dC)(dC/dc_k) with dC/dc_k = α_k Π[j=1..k−1](1 − α_j).

Position adjustment of the cloned Gaussian, Eq. 6 (p6):

  μ_new = μ_old + ∑[i ∈ N] ∇μᵢ

Pruning importance, Eq. 7 (p6), the RadSplat max-over-rays blending weight:

  Rᵢ = max[r ∈ R_f] α_i^r τ_i^r,   τᵢ = αᵢ Π[j=1..i−1](1 − α_j)

Compensation depth, Eq. 8 (p7):

  D = ∑[i ∈ N] dᵢ αᵢ Π[j=1..i−1](1 − α_j)

then top-K highest-colour-gradient pixels are back-projected with d_mid and seeded as new Gaussians
with the ground-truth pixel colour.

Algorithm 1 verbatim (p6), which is the actual control law:

```
Input: Gaussian primitives G, target peak number of Gaussians F, maximum number of iterations T;
1: if densifyBegin < t < densifyEnd then
2:    if |G| < F then
3:       cloneAndSplit(G);
4:       shiftNewGaussians(G);
5:    end if
6:    prune(G, oi < ot);
7: else if compensateBegin < t < compensateEnd then
8:    compensateGaussians(G);
9: end if
10: if |G| > F then
11:    K = F − |G|;
12:    G′ ← findLeastK(G, K);
13:    prune(G, G′);
14: end if
15: t = t + 1;
```

Reading of the control law:

- **Can the count grow while the controller is active?** Yes, but only while |G| < F. Line 2 gates
  cloning and splitting on |G| < F.
- **Is overshoot allowed?** Yes, transiently. Line 10 fires only when |G| > F, so the count can
  exceed F inside one step (compensation on line 8 adds Gaussians with no cap check) and is pulled
  back at the end of the same iteration. The bound is therefore per-iteration, not per-instant.
- **Line 11 as printed is `K = F − |G|`, which is negative in the branch where |G| > F.** This is
  almost certainly a sign typo for |G| − F. Quoted verbatim above, flagged, not corrected.
- Schedule: "those steps are alternately performed in every predefined iteration (e.g., every 50
  iterations)" (p5). Importance pruning "is performed every 100 iterations, resulting in negligible
  overhead" (p6). Compensation runs on its own window: "Our Gaussian compensation step starts at the
  10K-th iteration and ends at the 15K-th iteration. After that, we fine-tune the result to a
  certain iteration depending on each scene." (p7). Opacity and position of all Gaussians are reset at iteration 5K for Mip-NeRF 360
  outdoor scenes (p7).

### 1.6 Device numbers

All reported.

| device | quantity | value |
|---|---|---|
| NVIDIA Jetson Xavier (called "NVIDIA Jetson AGX Xavier" on p7) | peak training memory, 3DGS | 18.59 GB |
| NVIDIA Jetson Xavier | peak training memory, Taming 3DGS | 10.01 GB |
| NVIDIA Jetson Xavier | peak training memory, Ours | 8.55 GB |
| NVIDIA Jetson Xavier | peak training memory, Ours w/loader | 2.98 GB |
| "GTX 4090" (Fig. 1, p1) | peak training memory, Ours | 8.86 GB |
| "GTX 4090" | peak training memory, 3DGS | 18.29 GB |
| "NVIDIA GTX 4090 GPU" (p7, implementation) | training time on bicycle, Ours | 7 minutes (from 10 minutes for the per-splat parallelised backprop of Taming-GS), "achieving a 30% speedup" (p11) |

No FPS number is reported anywhere in this paper. The Jetson is the only edge device.

### 1.7 Main quantitative results

Table 1, p8, all reported. "3DGS, Mini-Splatting and Taming 3DGS results are reported from [25]",
that is from Taming 3DGS. Reducing-3DGS, Compact-3DGS and EAGLES were replicated by the authors.

| Method | Mip-NeRF 360 PSNR/SSIM/LPIPS/#G-M | T&T PSNR/SSIM/LPIPS/#G-M | Deep Blending PSNR/SSIM/LPIPS/#G-M |
|---|---|---|---|
| 3DGS | 27.46 / 0.815 / 0.215 / 3.310 | 23.65 / 0.847 / 0.176 / 1.840 | 29.64 / 0.904 / 0.243 / 2.810 |
| Mini-Splatting | 27.26 / 0.822 / 0.217 / 4.320 | 23.42 / 0.847 / 0.181 / 4.320 | 30.04 / 0.910 / 0.244 / 4.510 |
| Reducing-3DGS | 27.21 / 0.811 / 0.225 / 2.749 | 23.59 / 0.841 / 0.187 / 1.507 | 29.61 / 0.903 / 0.248 / 2.218 |
| Compact-3DGS | 26.96 / 0.797 / 0.244 / 2.590 | 23.34 / 0.831 / 0.202 / 1.465 | 29.80 / 0.900 / 0.257 / 2.268 |
| EAGLES | 27.15 / 0.811 / 0.231 / 1.928 | 23.27 / 0.837 / 0.201 / 0.954 | 29.83 / 0.909 / 0.246 / 1.981 |
| **Taming 3DGS** | 27.22 / 0.795 / 0.260 / 0.632 | 23.68 / 0.836 / 0.211 / 0.319 | 29.49 / 0.900 / 0.270 / 0.294 |
| **Ours** | 27.30 / 0.809 / 0.234 / 0.628 | 23.62 / 0.842 / 0.192 / 0.318 | 29.64 / 0.906 / 0.256 / 0.292 |

Strongest two baselines: Taming 3DGS (the only baseline in the same count class, within 0.004 M)
and Mini-Splatting (best PSNR on Deep Blending at 30.04, but at 4.510 M peak Gaussians).

Headline deltas as the paper states them (p7): "we outperform Taming 3DGS [25] by an average of
0.15 dB PSNR and 0.03 LPIPS with fewer peak Gaussians across all scenes" and "our method improves
PSNR by 0.5 dB on the Tank&Temple dataset and reduces peak numbers of Gaussians by more than 6×"
versus Mini-Splatting and vanilla 3DGS.

**Internal inconsistency flagged.** The appendix per-scene averages (p11) do not match Table 1 in
three places. Deep Blending Ours average is 29.69 / 0.906 / 0.256 / 0.292 in Table 3 but 29.64 in
Table 1. Tanks&Temples 3DGS average is 23.63 in Table 4 but 23.65 in Table 1. Mip-NeRF 360 3DGS
average is 27.45 / 0.810 / 0.220 / 3.110 in Table 5 but 27.46 / 0.815 / 0.215 / 3.310 in Table 1.
The appendix also states its 3DGS numbers come from EAGLES [13], while Table 1 says they come from
Taming 3DGS [25].

Ablation, Deep Blending LPIPS (Table 2(a), p8, reported): Baseline 0.279 (Playroom) / 0.270
(Drjohnson); +Iterative Pruning 0.264 / 0.257; +Gaussian Compensation 0.259 / 0.253. "Note that all
configurations yield the same final number of Gaussians." Mixed-gradient ablation (p9): "+0.04 dB
PSNR" on test, "+0.87 dB PSNR" on train, on kitchen.

### 1.8 Level selection (not applicable)

No LoD in this paper.

### 1.9 Stated limitations and future work

The paper has **no limitations section and no future-work section**. The Conclusion (p9) ends:

> "As a result, our framework offers a scalable solution for deploying 3DGS under hardware
> constraints." (p9)

Nothing else in the paper is framed as a limitation.

### 1.10 Verbatim quotes touching a size, memory, budget or edge/mobile target

> "However, it suffers from a substantial memory footprint, particularly during training due to
> uncontrolled densification, posing a critical bottleneck for deployment on memory-constrained
> edge devices." (p1)

> "While existing methods prune redundant Gaussians post-training, they fail to address the peak
> memory spikes caused by the abrupt growth of Gaussians early in the training process." (p1)

> "In other words, the proposed framework alternates between incremental pruning of low-impact
> Gaussians and strategic growing of new primitives with an adaptive Gaussian compensation,
> maintaining a near-constant low memory usage while progressively refining rendering fidelity."
> (p1)

> "We comprehensively evaluate the proposed training framework on various real-world datasets under
> strict memory constraints, showing significant improvements over existing state-of-the-art
> methods." (p1)

> "Particularly, our proposed method practically enables memory-efficient 3DGS training on NVIDIA
> Jetson AGX Xavier, achieving similar visual quality with up to 80% lower peak training memory
> consumption than the original 3DGS." (p1)

> "This reliance on a large number of primitives not only inflates the model size but also
> restricts deployment on edge devices or other memory-constrained platforms [28, 37]." (p2)

> "In practice, the heavy memory footprint of 3DGS-based models has become a key bottleneck,
> limiting their scalability and adoption in resource-limited settings." (p2)

> "Even though those methods successfully reduce the memory footprints in the rendering phase, the
> peak memory consumption is significantly higher than the memory size of edge systems, thereby
> hindering real-time 3D applications in real-world settings [26]." (p2)

> "Despite the practical significance of peak training memory usage in 3DGS, this issue remains
> understudied." (p2)

> "Prior work [25] mitigates memory spikes by regulating Gaussian growth via a predictable curve
> and selectively cloning/splitting primitives using a computationally intensive importance score."
> (p2)

> "1Here, we report our peak memory usage on GTX 4090 without engineering optimization for a fair
> comparison, while Mini-Splatting [9] leverages extra compression by downgrading the orders of SH
> coefficients to one before pruning to reduce memory." (p2)

> "Our method iteratively grows and prunes Gaussians under the memory constraint, while 3DGS [17]
> and Mini-Splatting [9] densify Gaussians to millions and remove them afterwards." (p2, Fig. 2
> caption)

> "To address those limitations and practically enable real-time 3DGS training on
> memory-constrained devices, in this paper, we conduct an in-depth study on the unsatisfactory
> performance of existing training approaches." (p2)

> "We extend this hypothesis to 3DGS, whether an optimal sparse Gaussian model can be trained from
> scratch under strict memory limitations." (p2)

> "Building on this insight, we propose a systematic memory-bounded 3DGS training framework based
> on dynamic growing and removal of Gaussian primitives, which can strictly satisfy the practical
> memory constraints." (p2)

> "This iterative process ensures consistently low memory usage while discovering effective
> primitives that match the rendering quality of the original 3DGS at significantly reduced
> training memory consumption." (p2)

> "Although these post-training pruning strategies reduce memory usage during inference, they do
> not alleviate the high peak memory consumption incurred during training [11, 29, 39]." (p3, and
> repeated verbatim on p4)

> "To address this, Taming 3DGS [25] introduces a steerable densification mechanism that
> selectively densifies impactful Gaussians, enabling a more predictable and memory-aware growth
> trajectory." (p3, and repeated verbatim on p4)

> "Based on the in-depth study on the limitations of [25], our work progressively refines the model
> via iteratively growing and pruning, dynamically preserving most "healthy" Gaussians under memory
> bounds." (p3, and repeated verbatim on p4)

> "Even though the above methods can effectively regulate the number of Gaussians, they grow the
> Gaussians slowly and achieve the user-specified budget after a long-term period, i.e., 15,000
> iterations." (p4)

> "This growing strategy limits the representation power due to the limited number of Gaussians
> before reaching the budget, leading to a performance drop." (p4)

> "In summary, our framework dynamically grows, compensates, and prunes Gaussians in an iterative
> way, where those steps are alternately performed in every predefined iteration (e.g., every 50
> iterations), progressively refining the representative capability under a consistent memory
> bound." (p5)

> "On the other hand, in the pruning phase, to ensure the model remains within a memory budget, we
> concurrently remove an equal number of less important Gaussians when the total count exceeds a
> predefined threshold." (p5)

> "As new Gaussians are added in the previous growing steps, an equal number of the least important
> Gaussians are subsequently removed in this pruning step, keeping the total peak training memory
> under the constraint." (p6)

> "In summary, our iterative growing and pruning have two advantages: firstly, it enables
> consistent training on devices with strict memory constraints where one-shot pruning approaches
> [8, 9] fail." (p6)

> "Furthermore, we assess memory efficiency by measuring peak training memory usage on real-world
> edge settings, i.e., NVIDIA Jetson AGX Xavier." (p7)

> "More importantly, our method practically achieves on-device training on memory-constrained
> platforms." (p7)

> "Experiments conducted on Jetson Xavier reveal our method reduces peak memory usage by nearly 2×
> compared to the original 3DGS, as shown in Table 2(b)." (p7)

> "Notably, we observed that up to three-quarters memory is used for dataset storage, we develop a
> parallel dataloader that dynamically prefetches and moves data between the storage and the memory
> according to the training pipeline. This effort further reduces peak training memory by more than
> 5 GB." (p7)

> "(b) Peak training memory usage on NVIDIA Jetson Xavier." (p8, Table 2 caption)

> "We stop densification after Gaussians exceed a target number in the original 3DGS [17] and
> report it as the baseline." (p8)

> "We have presented a memory-efficient training framework for 3DGS that dynamically balances
> primitive growth and pruning under strict memory constraints." (p9)

> "As a result, our framework offers a scalable solution for deploying 3DGS under hardware
> constraints." (p9)

> "Although our proposed framework primarily focuses on reducing training memory consumption, we
> also evaluate its impact on training speed." (p11)

### 1.11 Code URL and licence

**Not stated.** No repository, project page or URL appears anywhere in the PDF. No licence is
stated.

---

## 2. YOGO (You Only Gaussian Once)

### 2.1 Citation

- Title: "You Only Gaussian Once: Controllable 3D Gaussian Splatting for Ultra-Densely Sampled
  Scenes" (p1).
- First author: Jinrang Jia, KE Holdings Inc., Beijing, China. Co-authors Zhenjia Li and Yifeng Shi
  (corresponding). The running head reads "J. Jin et al.", inconsistent with the author list.
- Venue: **not stated**. The PDF carries the arXiv stamp "arXiv:2604.21400v4 [cs.CV] 15 Jul 2026"
  (p1) and uses the Springer LNCS single-column template.
- arXiv id: 2604.21400.

### 2.2 Base representation and renderer

Vanilla 3DGS (Kerbl et al. 2023) with three modifications inside the optimisation loop, not the
rasteriser: area-normalised absolute gradient accumulation, maximum-effective-opacity pruning, and
principal-axis densification. Nothing in the rasteriser itself is changed. Baselines used are 3DGS,
AbsGS, Mip-Splatting, Scaffold-GS and Perceptual-GS.

### 2.3 Budget handle

**TARGET, denominated in primitive count, optionally per spatial polygon.**

> "Each polygon is assigned a target Gaussian budget N target m based on user or hardware
> constraints." (p5)

The scene is partitioned into M disjoint spatial polygons P = {P₁, ..., P_M}, each with its own
target N_target_m. The global target is ∑ N_target_m. No GB, byte or millisecond handle exists.

**Achieved versus requested.** The paper never states the requested N_target for any run. Table 2
(p9) prints only the achieved "Point" count for three budget variants named YOGO1, YOGO2, YOGO3.
The only requested figure printed anywhere is the illustrative one in the teaser and abstract-level
text: "under a deterministic budget (e.g., 1.5M points)" (p1) and "e.g., "exactly 2M Gaussians""
(p3). Achieved counts (reported, Table 2 p9):

| track | YOGO1 | YOGO2 | YOGO3 | 3DGS | AbsGS |
|---|---|---|---|---|---|
| SSS (single-sensor sparse) | 1.76 M | 3.98 M | 5.83 M | 1.46 M | 3.13 M |
| SSD (single-sensor dense) | 1.49 M | 3.50 M | 5.64 M | 2.68 M | 4.28 M |
| MSD (multi-sensor dense) | 1.45 M | 3.45 M | 5.23 M | 3.14 M | 5.33 M |

Text on p11 says "our most constrained variant (YOGO1, 1.5M points)", which is consistent with the
achieved 1.49 M and 1.45 M on SSD and MSD but not with the 1.76 M on SSS. The SSS 1.76 M value is
the only case where the "1.5M" label and the printed count diverge. No requested-versus-achieved
table is given, so hit accuracy cannot be computed from this paper.

Ablation Table 3 (p11) prints achieved counts of 1.49 M, 1.39 M, 1.45 M, 1.47 M, 1.45 M, 1.43 M
across six fusion strategies at what is presumably the same YOGO1 budget. That 0.10 M spread is the
closest thing to a hit-accuracy signal in the paper, and it is confounded by the fusion strategy
change.

### 2.4 The resource model

**There is no resource model.** The paper maps model state to the resource by identity: the
resource *is* the primitive count. It never converts a count to bytes, GB or milliseconds. Memory
enters only rhetorically ("performance-memory trade-off", p11) and as a failure mode ("Out-Of-Memory
(OOM) failures", p11). Calibration and stated accuracy: **not stated**, because no model exists.

### 2.5 The control law

A per-polygon proportional controller over K densification events. Equations (p5):

Number of densification events, S = start iteration, E = end iteration, D = interval:

  K = [(E − S)/D] + 1

(The bracket in the PDF is a floor or ceiling glyph that did not survive text extraction.)

Per-event budget gap, after pruning, with N_cur_m(k) the current count inside polygon P_m:

  ΔN_m(k) = N_target_m − N_cur_m(k)

Per-event densification quota:

  N_densify_m(k) = max(0, ΔN_m(k) / (K − k + 1))

(A rounding bracket around the fraction did not survive extraction.)

Selection rule: "YOGO selects the top Q = N densify m (k) Gaussians with the highest accumulated
gradient magnitudes ‖∇μ‖ within each Pm" (p5).

Order of operations per event: "we apply a "prune-then-densify" sequence" (p5).

- **Can the count grow while the controller is active?** Yes. The quota is positive whenever the
  current count is below target, and it shrinks as k → K, so growth decays to zero at the end of the
  schedule.
- **Is overshoot allowed?** No, structurally. The quota is clamped at zero by max(0, ·), so a
  polygon already at or above target adds nothing. Pruning runs first each event, so the count is
  pulled down before the quota is computed. The paper claims exactness: "The final primitive count
  strictly converges to ∑ N target m , eliminating the need for iterative parameter tuning" (p5).
  No measurement backs the word "strictly".
- Schedule: an S / E / D densification window with interval D, values **not stated**.

Supporting terms in the loss and pruning (p6):

Area-normalised absolute gradient, V = visible views, Ωᵢ = pixel footprint in view i:

  Ḡ_accum = ( ∑[i ∈ V] ∑[p ∈ Ωᵢ] |∇p| ) / ( ∑[i ∈ V] |Ωᵢ| )

Effective opacity α̂ = α · T with T the accumulated transmittance, and the per-view maximum:

  α̂_max = max[i ∈ V] ( ∑[p ∈ Ωᵢ] α̂_p / |Ωᵢ| )

Primitives with α̂_max < τ_opacity are pruned.

Principal axis densification, q = argmax(s), Δl ∼ N(0, 0.3 s_q) · e_q (p7):

  x_new = { x + RΔl, x, x − RΔl },  s ← s/1.6,  o ← 0.3 o

Availability score for auxiliary sensor frames, A_j = [G_j | b_j] (p6):

  S_j = ‖diag(G_j) − 1‖_∞ + avg(|offdiag(G_j)|) + avg(|b_j|)

with rejection at S_j > τ. Best τ reported as 0.15 (Table 3, p11).

### 2.6 Device numbers

**None.** No Jetson, phone, laptop or low-end GPU result appears. No peak memory number, no FPS
number, no training time. The only hardware in the paper is the capture rig (Insta360 X5, four Osmo
units, SHARE SLAM S20). The training GPU is not named. The word "fps" appears only as camera frame
rate: "Captures 7680 × 3840 panoramic video (24 fps)" and "Four synchronized units capturing 3840 ×
2880 (25 fps)" (p8).

The only hardware-adjacent result is a negative one (p11):

> "Consequently, due to extreme computational overhead and Out-Of-Memory (OOM) failures, these
> specific baselines could not be evaluated on the SSD and MSD tracks."

### 2.7 Main quantitative results

**No Mip-NeRF 360, Tanks and Temples or Deep Blending experiment exists in this paper.** Those
datasets appear only in the dataset comparison Table 1 (p8) as scale references: Mip-NeRF 360 "9
scenes, ∼200 imgs, ∼158K init points"; Tanks & Temples "21 scenes, ∼300 imgs, ∼159K init points".

All results are on the authors' own Immersion v1.0 (7 scenes, ∼30K imgs, ∼2.55M init points, ∼100
avg area, ∼500 IDSM, ∼72% coverage). Table 2, p9, reported, three tracks. Metrics are PSNR / SSIM /
LPIPS / Qalign(validation) / Qalign(roaming test).

SSS track:

| Method | Point | PSNR | SSIM | LPIPS | Qalign val | Qalign test |
|---|---|---|---|---|---|---|
| 3DGS | 1.46 M | 22.49 | 0.8407 | 0.3375 | 2.7290 | 2.6571 |
| AbsGS | 3.13 M | 21.85 | 0.8214 | 0.3572 | 2.7215 | 2.6076 |
| Mip-Splatting | 1.04 M | 22.87 | 0.8429 | 0.3298 | 2.6522 | 2.5662 |
| Scaffold-GS | 1.04 M | 24.38 | 0.8527 | 0.3158 | 2.5800 | 2.5197 |
| Perceptual-GS | 3.46 M | 22.59 | 0.8412 | 0.3234 | 2.8443 | 2.8226 |
| YOGO1 | 1.76 M | 25.83 | 0.8674 | 0.3001 | 3.1789 | 3.1764 |
| YOGO2 | 3.98 M | 25.90 | 0.8695 | 0.2930 | 3.2801 | 3.3053 |
| YOGO3 | 5.83 M | 25.92 | 0.8701 | 0.2904 | 3.3220 | 3.3485 |

SSD track:

| Method | Point | PSNR | SSIM | LPIPS | Qalign val | Qalign test |
|---|---|---|---|---|---|---|
| 3DGS | 2.68 M | 27.50 | 0.8812 | 0.2737 | 3.6292 | 3.6203 |
| AbsGS | 4.28 M | 27.62 | 0.8823 | 0.2700 | 3.6376 | 3.6404 |
| YOGO1 | 1.49 M | 27.73 | 0.8870 | 0.2681 | 3.6839 | 3.7142 |
| YOGO2 | 3.50 M | 27.81 | 0.8885 | 0.2645 | 3.7380 | 3.7774 |
| YOGO3 | 5.64 M | 27.84 | 0.8891 | 0.2632 | 3.7543 | 3.7959 |

MSD track:

| Method | Point | PSNR | SSIM | LPIPS | Qalign val | Qalign test |
|---|---|---|---|---|---|---|
| 3DGS | 3.14 M | 27.24 | 0.8806 | 0.2719 | 3.6573 | 3.6763 |
| AbsGS | 5.33 M | 27.34 | 0.8820 | 0.2669 | 3.6565 | 3.6805 |
| YOGO1 | 1.45 M | 27.57 | 0.8862 | 0.2681 | 3.6816 | 3.7771 |
| YOGO2 | 3.45 M | 27.63 | 0.8878 | 0.2643 | 3.7294 | 3.8211 |
| YOGO3 | 5.23 M | 27.69 | 0.8883 | 0.2629 | 3.7571 | 3.8426 |

Strongest two baselines: Scaffold-GS (24.38 PSNR on SSS at 1.04 M, the best non-YOGO PSNR there)
and AbsGS (27.62 PSNR on SSD at 4.28 M). Mip-Splatting, Scaffold-GS and Perceptual-GS could not be
run on SSD or MSD.

Saturation behaviour of the budget knob (p11, reported):

> "We observe distinct performance saturation: on the SSS track, upgrading from YOGO1 to YOGO2
> (+∼2M points) yields a significant 0.13 Qalign improvement, whereas the transition to YOGO3
> (+∼2M points) yields a marginal 0.04 gain."

Solid Optimization Suite ablation, SSD track (Table 4, p12, reported): baseline 27.50 / 0.8812 /
0.2737 / 3.6292 / 3.6203; +Ḡ 27.62 / 0.8825 / 0.2702 / 3.6581 / 3.6414; +α̂max 27.69 / 0.8854 /
0.2687 / 3.6670 / 3.7006; +PAD 27.73 / 0.8870 / 0.2681 / 3.6839 / 3.7142.

### 2.8 Level selection (not applicable)

No LoD. The nearest analogue is the per-polygon budget, which is a spatial partition of the count
budget, fixed at training time and not selected at render time.

### 2.9 Stated limitations and future work

> "Limitations and Future Work: A current limitation of Immersion v1.0 is its constrained scene
> count, a direct consequence of the severe computational overhead required to process massive,
> saturated multi-sensor captures. We are actively expanding this frontier; Immersion v2.0 is
> currently underway, featuring over 40 complex environments. Beyond data expansion, our future
> work will focus on integrating semantic-aware budget allocation into the DBC and extending YOGO
> to unbounded dynamic environments for autonomous navigation." (p13)

The limitation is about the dataset, not the controller. The paper states no limitation of the
budget controller itself.

### 2.10 Verbatim quotes touching a size, memory, budget or edge/mobile target

> "3.5 Million Points (OOM Risk) Unpredictable Growth, Low Fidelity 1.5 Million Points
> Deterministic Budget, High Fidelity" (p1, Fig. 1 in-figure labels)

> "(A) Vanilla 3DGS suffers from uncontrollable growth and OOM risks on our challenging Immersion
> dataset. (B) YOGO ensures high-fidelity reconstruction under a deterministic budget (e.g., 1.5M
> points) via robust multi-sensor fusion." (p1)

> "We identify a critical "Industry-Academia Gap" hindering real-world application: unpredictable
> resource consumption from heuristic Gaussian growth, the "sparsity shield" of current benchmarks
> that rewards hallucination over physical fidelity, and severe multi-sensor data pollution." (p2)

> "To bridge this gap, we propose YOGO (You Only Gaussian Once), a system-level framework that
> reformulates the stochastic growth process into a deterministic, budget-aware equilibrium." (p2)

> "YOGO integrates a novel budget controller for hardware-constrained resource allocation and an
> availability-registration protocol for robust multi-sensor fusion." (p2)

> "While excelling on sparse-view benchmarks [1, 14, 19], 3DGS struggles with the extreme textures,
> complex occlusions, and strict resource constraints inherent to production environments like
> digital twins [8, 9, 11, 12, 26, 33, 31] and autonomous driving [10, 15, 17, 30, 2]." (p2)

> "First, standard 3DGS relies on heuristic densification, rendering the final primitive count an
> unpredictable byproduct. For edge deployment, this non-determinism necessitates costly
> trial-and-error parameter tuning." (p2)

> "Deterministic Budget Control: We replace heuristic growth with a feedback-loop controller,
> enforcing strict adherence to hardware-defined primitive budgets. This control can be localized
> via spatial polygons for extreme fidelity in specific regions of interest." (p3)

> "Ultimately, YOGO transcends the laboratory prototype, establishing a new standard for
> resource-controllable, high-fidelity rendering." (p3)

> "Despite these improvements, existing methods rely on heuristic densification thresholds,
> treating reconstruction as an uncontrollable process. This unpredictability in memory footprint
> hinders strict hardware deployment—a critical gap YOGO bridges via deterministic resource
> control." (p3)

> "To mitigate the massive memory footprint of 3DGS, existing literature primarily employs pruning
> [5, 21, 32] or post-training compression techniques [4, 25, 27]. However, these "train-then-fix"
> approaches lack determinism: users cannot predefine exact hardware budgets (e.g., "exactly 2M
> Gaussians") prior to training." (p3)

> "YOGO diverges by reformulating densification as a controllable equilibrium, enabling one-stage
> training that strictly adheres to pre-set resource constraints from the outset." (p3)

> "Under the deterministic budget controller (Sec. 3.1), the number of Gaussian points at each
> stage is strictly controlled, which regulates growth based on preset constraints and Polygon
> regions." (p4, Fig. 2 caption)

> "To facilitate production-level deployment, we introduce the Deterministic Budget Controller
> (DBC), reformulating stochastic Gaussian growth into a budget-driven process for strict hardware
> compliance and Region of Interest (ROI) refinement." (p4)

> "Each polygon is assigned a target Gaussian budget N target m based on user or hardware
> constraints." (p5)

> "Deterministic Predictability: The final primitive count strictly converges to ∑ N target m ,
> eliminating the need for iterative parameter tuning." (p5)

> "Multi-Granular Control: Adjusting N target m enables highly localized resource allocation,
> preserving extreme fidelity in complex ROIs while maintaining high compression ratios in
> backgrounds." (p5)

> "Primitives with ˆαmax < τopacity are pruned. This eliminates visually redundant Gaussians that
> possess high intrinsic opacity but contribute zero to the rendering, significantly compressing the
> model without quality degradation." (p6)

> "Unlike heuristic methods, YOGO facilitates explicit control over resource consumption. As shown
> in Table 2, we benchmark three budget variants (YOGO1, YOGO2, YOGO3) to analyze the
> performance-memory trade-off." (p11)

> "This determinism empowers users to pinpoint the optimal balance between fidelity and
> computational cost, strictly avoiding redundant memory allocation." (p11)

> "Consequently, due to extreme computational overhead and Out-Of-Memory (OOM) failures, these
> specific baselines could not be evaluated on the SSD and MSD tracks." (p11)

> "Remarkably, on both SSD and MSD tracks, our most constrained variant (YOGO1, 1.5M points)
> surpasses the rendering quality of AbsGS (4.28M points), demonstrating superior structural
> efficiency." (p11)

> "Concurrently, YOGO fundamentally transforms heuristic densification into a budget-aware
> regulation task, enabling arbitrary polygon-level resource allocation and robust heterogeneous
> sensor fusion." (p13)

> "Beyond data expansion, our future work will focus on integrating semantic-aware budget
> allocation into the DBC and extending YOGO to unbounded dynamic environments for autonomous
> navigation." (p13)

### 2.11 Code URL and licence

As printed on p2:

> "To facilitate reproducibility, part scenes of Immersion v1.0 dataset and source code of YOGO has
> been publicly released. The project link is https://jjrcn.github.io/yogo-project-home/."

Licence: **not stated**.

---

## 3. CLoD-GS

### 3.1 Citation

- Title: "CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting" (p1).
- First author: Zhigang Cheng, Tsinghua University (work done during internship at AMAP).
  Co-corresponding authors Mu Xu, Yangyan Li (AMAP) and Peng Pan (Tsinghua).
- Venue as printed in the running head of every page: "Published as a conference paper at ICLR
  2026".
- arXiv id: 2510.09997, stamp "arXiv:2510.09997v2 [cs.GR] 1 Apr 2026" (p1).

### 3.2 Base representation and renderer

Vanilla 3DGS with **one extra learnable scalar per Gaussian**, the distance decay factor σ_d,i.
Rendering is the standard 3DGS rasteriser fed a pre-filtered subset: the attenuated opacity is
computed on the CPU/GPU side and a boolean mask decides "which primitives are significant enough to
be sent to the rasterizer" (p4). The paper explicitly claims renderer compatibility, contrasting
with Octree-GS which "is not natively supported by existing renderers" (p6).

Storage overhead is stated exactly (p10):

> "Our method adds only one additional float parameter per Gaussian. In a standard 3DGS
> implementation, each Gaussian requires approximately 248 bytes of storage, an increase of only
> 1.6%, which is an entirely acceptable overhead."

### 3.3 Budget handle

**KNOB, not a target. Denominated in a dimensionless virtual distance scale s_v ≥ 1.**

Two distinct knobs, one at train time and one at render time:

- Train time: the maximum s_v of the sampling range, "tested at values of 1, 3, 5 and 7, defines
  the maximum allowable value for the sv during training" (p6). This selects which model is
  trained, and it is the thing labelled "scale=1", "scale=3", "scale=7" in every table.
- Render time: the user picks s_v continuously. "The virtual distance scale sv ≥ 1 allows the user
  to simulate the effect of viewing the scene from farther away, thereby increasing the
  attenuation" (p4), and "This simple, per-primitive computation allows for a continuous and smooth
  performance-quality trade-off by adjusting the single scalar sv" (p4).

The user cannot supply bytes, GB or milliseconds. There is no inverse map from a desired MB or FPS
to an s_v. The only quantity the loss actually targets is a **ratio** of rendered primitives, and
that target is derived from s_v, not supplied.

**Achieved versus requested.** No requested-versus-achieved table exists. The closest is the
internal training target η_target = 1/s_v^1.5 versus the achieved η_actual, which is never
tabulated. What is tabulated is the achieved count and memory at each train-time scale (Table 1, p6,
and Table 4, p13, both reported):

| dataset | scale=1 #GS(k) / Mem(MB) | scale=3 | scale=7 | 3DGS reference |
|---|---|---|---|---|
| BungeeNeRF | 4185 / 1005.87 | 2738 / 658.01 | 1855 / 445.72 | 6733 / 1592.48 |
| Tanks&Temples | 1159 / 278.53 | 984 / 236.58 | 884 / 212.54 | 1574 / 372.19 |
| Deep Blending | 1697 / 407.72 | 1258 / 302.27 | 662 / 159.04 | 2486 / 587.98 |
| Mip-NeRF 360 | 2414.38 / 580.24 | 1778.10 / 427.32 | 1361.49 / 327.20 | 2638.53 / 624.04 |

### 3.4 The resource model

The paper states no equation for memory. It reports a Mem(MB) column beside a #GS(k) column.

**Note-taker derivation, not a paper claim.** The Mem(MB) column is an exact linear function of the
count, with MB meaning MiB:

  Mem = #GS × 252 bytes / 2²⁰   for CLoD-GS rows
  Mem = #GS × 248 bytes / 2²⁰   for 3DGS and MaskGaussian rows

Checks against the reported table: 6733000 × 248 / 2²⁰ = 1592.3 versus reported 1592.48 for 3DGS on
BungeeNeRF; 4185000 × 252 / 2²⁰ = 1005.7 versus reported 1005.87 for Ours scale=1; 2738000 × 252 /
2²⁰ = 658.0 versus reported 658.01 for Ours scale=3; 5298000 × 248 / 2²⁰ = 1253.0 versus reported
1253.13 for MaskGaussian. The 252 versus 248 difference is exactly the one extra float per Gaussian
the paper describes. So the "memory" column is a byte accounting of the model, not a measured GPU
footprint, and the model is Mem = count × 248 bytes (+ 4 bytes for this method). No calibration and
no accuracy is stated, and the paper never presents this as a model.

**Frame time as a function of rendered count: asserted, not modelled.** (p8)

> "As shown, rendering speed exhibits a strong negative correlation with the number of Gaussians.
> Reducing the number of Gaussians boosts FPS for all methods, with our method showing a more
> significant increase."

No equation, no fit, no reported correlation coefficient.

### 3.5 The control law

The mechanism is a learned per-primitive opacity decay plus a training-time regulariser on the
*fraction* of primitives that survive the render-time mask.

Attenuated opacity, Eq. 2 (p4), with d′ᵢ = dᵢ / max[j ∈ N_view] d_j the normalised distance from the
camera centre and ε a stability constant:

  α″ᵢ = αᵢ · exp( − (d′ᵢ · s_v)² / (2 · (ReLU(σ_d,i))² + ε) )

Render-time mask, with τ a small base opacity threshold (p4):

  Mᵢ = ( α″ᵢ > τ · s_v )

> "By scaling the threshold with sv, we apply a stricter culling criterion when simulating more
> distant views. Only Gaussians with Mi = 1 are rendered." (p4)

Training: s_v is sampled fresh each iteration, "we randomly sample a virtual distance scale factor
sv from a predefined range (e.g., U(1, 10))" (p5). Note the example range U(1, 10) in the method
section conflicts with the implementation section, which says s_v max was tested at 1, 3, 5 and 7
(p6).

Target primitive ratio, Eq. 3 (p5):

  η_target = 1 / s_v^1.5

> "The exponent 1.5 is an empirically determined value that controls the rate of geometric
> simplification." (p5)

Regularisation loss, Eq. 4 (p5), one-sided, penalising only overshoot of the ratio:

  L_reg = (s_v − 1.0)² · ( ReLU(η_actual − η_target) )²,   η_actual = (∑ᵢ Mᵢ) / N_total

Total loss, Eq. 5 (p5), with w_s = (1 − 0.5 · s_v / max(s_v))²:

  L_total = w_s ( L_render + λ_reg L_reg )

Hyperparameters (p6): 30,000 iterations, mechanism enabled from iteration 5,000, learning rate for
σ_d,i is 1e-2, λ_reg = 1.0, same hyperparameters on all datasets.

- **Can the count grow while the controller is active?** Yes. Standard 3DGS densification is
  untouched. The regulariser acts on the rendered *fraction*, not on the total, so N_total evolves
  freely and only the surviving ratio at large s_v is penalised.
- **Is overshoot allowed?** Yes at s_v = 1, where the ReLU weight (s_v − 1)² = 0 kills the penalty
  entirely. At s_v > 1 the penalty is one-sided, so undershooting the target ratio is free and
  overshooting is penalised quadratically. This is a soft constraint with no hard cap anywhere.

### 3.6 Device numbers

**No edge, phone, Jetson or low-end GPU result.** The only hardware statement (p6):

> "All experiments are run on an Ubuntu server with four NVIDIA RTX 4090 GPUs."

The device used for the FPS measurements in Table 2 is **not stated** separately, so the FPS figures
are presumably on one RTX 4090. Table 2 (p6), all reported:

| Method | BungeeNeRF FPS | Tanks&Temples FPS | Deep Blending FPS | Mip-NeRF 360 FPS |
|---|---|---|---|---|
| Octree-GS | 76.05 | 134.87 | 141.59 | 129.45 |
| H-3DGS (τ=0) | / | 42.30 | 36.30 | / |
| H-3DGS (τ=3) | / | 54.05 | 46.50 | / |
| H-3DGS (τ=6) | / | 61.54 | 47.37 | / |
| H-3DGS (τ=15) | / | 75.82 | 51.03 | / |
| Ours (scale=1) | 57.34 | 169.89 | 128.68 | 109.42 |
| Ours (scale=3) | 69.74 | 176.74 | 145.83 | 125.33 |
| Ours (scale=7) | 87.88 | 199.12 | 187.48 | 140.58 |

Training-cost claim (p8): "our CLoD approach is also more efficient, requiring the training of only
one model, which typically takes half the time needed to train the two models required for the
DLoD setup."

### 3.7 Main quantitative results

Table 1, p6, and Table 4, p13, all reported.

Mip-NeRF 360 (appendix Table 4, p13):

| Method | PSNR | SSIM | LPIPS | #GS(k) | Mem(MB) |
|---|---|---|---|---|---|
| 3DGS | 27.59 | 0.8136 | 0.2206 | 2638.53 | 624.04 |
| Fast Rendering | 27.58 | 0.8240 | 0.2420 | 2638.53 | 624.04 |
| Octree-GS | 27.65 | 0.8150 | 0.2200 | / | 418.60 |
| MaskGaussian | 27.23 | 0.8035 | 0.2253 | 2145.10 | 507.34 |
| Ours (scale=1) | 27.01 | 0.8069 | 0.2299 | 2414.38 | 580.24 |
| Ours (scale=3) | 26.99 | 0.8042 | 0.2361 | 1778.10 | 427.32 |
| Ours (scale=7) | 26.71 | 0.7968 | 0.2494 | 1361.49 | 327.20 |

CLoD-GS is beaten on Mip-NeRF 360 by 3DGS, Octree-GS and Fast Rendering, and the paper concedes it
(p13): "We acknowledge that the results are not state-of-the-art on this specific dataset, which we
attribute to our use of universal parameters that were not optimized for these special scenes."

Tanks&Temples (Table 1, p6):

| Method | PSNR | SSIM | LPIPS | #GS(k) | Mem(MB) |
|---|---|---|---|---|---|
| 3DGS | 23.70 | 0.853 | 0.169 | 1574 | 372.19 |
| Fast Rendering | 23.62 | 0.853 | 0.194 | 1574 | 372.19 |
| Octree-GS | 24.17 | 0.858 | 0.161 | / | 383.90 |
| MaskGaussian | 23.56 | 0.846 | 0.180 | 1237 | 292.68 |
| H-3DGS (τ=0 / 3 / 6 / 15) | 21.71 / 21.77 / 21.76 / 21.55 | 0.820 / 0.821 / 0.818 / 0.800 | 0.200 / 0.200 / 0.206 / 0.239 | / | / |
| Ours (scale=1) | 23.75 | 0.843 | 0.185 | 1159 | 278.53 |
| Ours (scale=3) | 23.79 | 0.843 | 0.185 | 984 | 236.58 |
| Ours (scale=7) | 23.67 | 0.839 | 0.193 | 884 | 212.54 |

Deep Blending (Table 1, p6):

| Method | PSNR | SSIM | LPIPS | #GS(k) | Mem(MB) |
|---|---|---|---|---|---|
| 3DGS | 29.84 | 0.907 | 0.238 | 2486 | 587.98 |
| Fast Rendering | 29.00 | 0.902 | 0.303 | 2486 | 587.98 |
| Octree-GS | 29.65 | 0.901 | 0.257 | / | 180.00 |
| MaskGaussian | 29.66 | 0.907 | 0.244 | 1778 | 420.41 |
| H-3DGS (τ=0 / 3 / 6 / 15) | 27.41 / 27.40 / 27.38 / 27.26 | 0.887 / 0.887 / 0.887 / 0.884 | 0.254 / 0.254 / 0.255 / 0.265 | / | / |
| Ours (scale=1) | 29.93 | 0.908 | 0.239 | 1697 | 407.72 |
| Ours (scale=3) | 29.86 | 0.907 | 0.244 | 1258 | 302.27 |
| Ours (scale=7) | 29.64 | 0.907 | 0.251 | 662 | 159.04 |

BungeeNeRF (Table 1, p6), the dataset where the method wins hardest:

| Method | PSNR | SSIM | LPIPS | #GS(k) | Mem(MB) |
|---|---|---|---|---|---|
| 3DGS | 27.91 | 0.917 | 0.096 | 6733 | 1592.48 |
| Fast Rendering | / | / | / | 6733 | 1592.48 |
| Octree-GS | 27.94 | 0.909 | 0.110 | / | 1045.70 |
| MaskGaussian | 27.76 | 0.916 | 0.098 | 5298 | 1253.13 |
| Ours (scale=1) | 28.05 | 0.919 | 0.100 | 4185 | 1005.87 |
| Ours (scale=3) | 27.70 | 0.908 | 0.117 | 2738 | 658.01 |
| Ours (scale=7) | 27.09 | 0.885 | 0.150 | 1855 | 445.72 |

Strongest two baselines overall: Octree-GS (best compression: 180.00 MB on Deep Blending and 418.60
MB on Mip-NeRF 360, at competitive PSNR) and MaskGaussian (best pruning-based competitor,
consistently ~20% fewer Gaussians than 3DGS at near-parity quality).

Ablation, Table 3 (p8), all at max virtual scale s_v = 5, reported: BungeeNeRF Full 27.59 / 0.902 /
0.123, w/o weight 27.39 / 0.894 / 0.127, w/o loss 27.56 / 0.902 / 0.123, w/o both 26.71 / 0.871 /
0.169. Deep Blending Full 29.76 / 0.908 / 0.245, w/o weight 29.58 / 0.906 / 0.246, w/o loss 29.72 /
0.906 / 0.246, w/o both 29.57 / 0.905 / 0.252.

Per-scene appendix tables 5-11 (p14-17) give #GS in millions for scale=1, MaskGaussian, 3DGS,
scale=5 full and its three ablations across 12 scenes.

### 3.8 Level selection at render time, and whether a memory or time budget can be an input

**How a level is selected.** There are no levels. Selection is by the single continuous scalar s_v,
chosen by the user at render time. Given s_v, each Gaussian gets an attenuated opacity α″ᵢ (Eq. 2)
using its own learned decay σ_d,i and its normalised distance to the camera, and survives if
α″ᵢ > τ · s_v. The paper does **not** define a distance-to-s_v policy, so a real application must
supply its own map from viewing distance or from a performance target to s_v. There is no
screen-space error metric, no projected-size test and no hierarchy traversal.

**Can a memory or time budget be given as input?** **No.** The input is s_v, a dimensionless
attenuation multiplier. The paper positions this as covering "a wide range of performance targets"
(p1) but supplies no inversion from MB or ms to s_v. The trade-off must be found by sweeping s_v
and measuring.

**Memory and quality per level.** The train-time scale parameter creates three published operating
points per dataset. Those are exactly the Table 1 and Table 4 rows above (scale = 1, 3, 7). Read as
a per-level table for Deep Blending, reported: 407.72 MB → 29.93 dB, 302.27 MB → 29.86 dB, 159.04 MB
→ 29.64 dB. That is a 61% memory cut for 0.29 dB. For BungeeNeRF: 1005.87 MB → 28.05 dB, 658.01 MB
→ 27.70 dB, 445.72 MB → 27.09 dB, a 56% memory cut for 0.96 dB.

Important caveat: these three rows are three separately *trained* models, not three render-time
settings of one model. The render-time continuum within one model is shown only as curves (Figures
3, 4, 7) with no printed numbers.

### 3.9 Stated limitations and future work

The paper has no separate limitations section. The concession about Mip-NeRF 360 and the future
work sentence are the only two:

> "We acknowledge that the results are not state-of-the-art on this specific dataset, which we
> attribute to our use of universal parameters that were not optimized for these special scenes."
> (p13)

> "While Octree-GS achieves the highest compression ratios, it introduces considerable rendering
> overhead for smaller scenes and is not natively supported by existing renderers." (p6)

> "While our method demonstrates significant advantages for static scenes, future work could
> explore the integration of more sophisticated perceptual metrics beyond distance to guide the
> learned decay, or hybrid systems that combine our continuous per-primitive scaling with
> chunk-based loading for rendering massive-scale environments." (p10)

### 3.10 Verbatim quotes touching a size, byte, memory or FPS target, budget, or edge/mobile deployment

> "However, this long-standing paradigm suffers from two major drawbacks: it requires significant
> storage for multiple model copies and causes jarring visual "popping" artifacts during
> transitions, degrading the user experience." (p1)

> "Our approach not only eliminates the storage overhead and visual artifacts of discrete methods
> but also reduces the primitive count and memory footprint of the final model." (p1)

> "Extensive experiments demonstrate that CLoD-GS achieves smooth, quality-scalable rendering from
> a single model, delivering high-fidelity results across a wide range of performance targets."
> (p1)

> "The pursuit of photorealism in real-time computer graphics is characterized by a fundamental
> tension between ever-increasing scene complexity and the finite computational budget of rendering
> hardware." (p1)

> "To maintain interactive frame rates, systems must intelligently manage the number of primitives
> rendered per frame, a challenge first articulated decades ago (Funkhouser & Séquin, 1993)." (p1)

> "This constrained optimization problem—generating the best possible image within a fixed time
> budget—has driven the development of Level of Detail (LoD) techniques, which adaptively reduce an
> object's complexity based on its perceptual importance to the viewer (Luebke et al., 2002)." (p1)

> "First, storing multiple copies of every asset leads to a significant memory overhead, limiting
> scene scale and variety." (p1)

> "However, this paradigm does not escape the fundamental constraint of rendering cost; performance
> still scales with the number of primitives, making LoD a necessity for complex scenes." (p2)

> "While feasible, this strategy inevitably reintroduces the classic DLoD drawbacks: a significant
> storage overhead for maintaining multiple Gaussian clouds and the visually disruptive "popping"
> artifacts during transitions." (p2)

> "Our framework eliminates the storage overhead and popping artifacts inherent to discrete methods
> while simultaneously reducing the final model's primitive count." (p2)

> "DLoD's reliance on multiple asset copies leads to high storage costs and visually jarring
> "popping" artifacts during transitions (Luebke et al., 2002)." (p2)

> "While CLoD successfully eliminates storage overhead and popping, it shifts complexity from the
> asset pipeline to the runtime algorithm, often incurring significant CPU overhead to traverse the
> hierarchical data structures and generate the appropriate mesh each frame (Lindstrom et al.,
> 1996)." (p2)

> "They rely on rigid, explicit data structures that add algorithmic and memory overhead, and their
> management of discrete levels can cause popping artifacts." (p3)

> "While LoD is a runtime optimization for performance scaling, compression is a pre-process for
> reducing storage and transmission costs." (p3)

> "This allows for a seamless trade-off between visual quality and performance, catering to diverse
> hardware capabilities and user requirements." (p4)

> "The virtual distance scale sv ≥ 1 allows the user to simulate the effect of viewing the scene
> from farther away, thereby increasing the attenuation." (p4)

> "This simple, per-primitive computation allows for a continuous and smooth performance-quality
> trade-off by adjusting the single scalar sv." (p4)

> "The fifth and sixth columns indicate the number of Gaussian primitives (#GS) and memory
> consumption (Mem)." (p6, Table 1 caption)

> "Ultimately, when pruning Gaussians to enhance FPS, our method consistently achieves higher
> rendering quality." (p8)

> "As shown, rendering speed exhibits a strong negative correlation with the number of Gaussians.
> Reducing the number of Gaussians boosts FPS for all methods, with our method showing a more
> significant increase." (p8)

> "For our CLoD approach, we use a single trained model and render the four regions with
> progressively increasing detail by setting the scale factor sv, respectively, while keeping the
> number of rendered Gaussians comparable to the DLoD setup." (p8)

> "Our method adds only one additional float parameter per Gaussian. In a standard 3DGS
> implementation, each Gaussian requires approximately 248 bytes of storage, an increase of only
> 1.6%, which is an entirely acceptable overhead." (p10)

> "We identified the core limitations of applying traditional DLoD paradigms to 3DGS—namely,
> prohibitive storage overhead and jarring visual "popping" artifacts." (p10)

> "While our method demonstrates significant advantages for static scenes, future work could
> explore the integration of more sophisticated perceptual metrics beyond distance to guide the
> learned decay, or hybrid systems that combine our continuous per-primitive scaling with
> chunk-based loading for rendering massive-scale environments." (p10)

> "As shown in the Table 4, even without any dataset-specific tuning, CLoD-GS demonstrates a
> competitive trade-off between quality and memory." (p13)

### 3.11 Code URL and licence

**Not stated.** The Reproducibility Statement (p11) promises only that the datasets are public and
the hyperparameters are in Section 4:

> "All experiments were conducted on publicly available datasets, ensuring that the data is
> accessible to the broader research community. The detailed experimental setup, including all
> hyperparameters and training configurations, is described in Section 4 of the main paper."

No repository URL and no licence anywhere in the PDF.

---

## 4. FLoD

### 4.1 Citation

- Title: "FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable
  Rendering" (p1).
- First authors: Yunji Seo and Young Sun Choi (joint first authors, equal contribution), Yonsei
  University, South Korea. HyunSeung Son, Youngjung Uh (corresponding).
- Venue: ACM Transactions on Graphics. As printed (p2): "Yunji Seo, Young Sun Choi, HyunSeung Son,
  and Youngjung Uh. 2025. FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for
  Customizable Rendering. ACM Trans. Graph. 44, 4 (August 2025), 16 pages.
  https://doi.org/10.1145/3731430". Page footer on every page: "ACM Trans. Graph., Vol. 44, No. 4,
  Article . Publication date: August 2025."
- arXiv id: 2408.12894, stamp "arXiv:2408.12894v2 [cs.CV] 11 Jun 2025" (p1).

### 4.2 Base representation and renderer

Vanilla 3DGS, plus a Scaffold-GS variant (FLoD-Scaffold). Renderer unchanged. The representation is
**L_max separate complete Gaussian sets**, one per level, each of which independently reconstructs
the whole scene. Default L_max = 5.

The one structural change is the parameterisation of scale, Eq. 4 (p4):

  s^(l) = e^(s_opt) + s_min^(l)

so the level-specific lower bound s_min^(l) is baked into the scale activation rather than enforced
by clipping. There is no upper bound on Gaussian size at any level, and the vanilla 3DGS large-
Gaussian pruning and large-projected-size pruning are removed (p6).

### 4.3 Budget handle

**KNOB, denominated in a level index (and in a screen-size threshold γ in pixels for selective
rendering).** The training-side knob is a 3D scale in world units.

Train time, Eq. 3 (p4):

  s_min^(l) = λ · ρ^(1−l)   for 1 ≤ l < L_max,   s_min^(L_max) = 0

with λ = 0.2 (initial 3D scale constraint) and ρ = 4 (scale factor), reported p6. Overlap pruning
threshold d_OP^(l) is "set as half of the 3D scale constraint s(l) min for training level l" (p4),
using the mean distance to the three nearest neighbours, Eq. 6.

Render time: choose a single level l, or a level range [L_start, L_end] plus a screen-size threshold
γ (default γ = 1.0 pixel).

> "The threshold γ and the level range [Lstart, Lend] can be adjusted to accommodate specific memory
> limitations or desired rendering rates." (p5)

That is a knob to be swept, not a target to be hit.

**Achieved versus requested.** No requested-versus-achieved pair exists, because nothing numeric is
requested. What is reported is achieved GPU memory per level. Gaussian counts per level after
overlap pruning, Mip-NeRF 360 (Table 6, p12, reported):

| Level | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| w/o overlap pruning | 38K | 49K | 439K | 1001K | 2058K |
| w/ overlap pruning | 10K | 31K | 390K | 970K | 2048K |

The paper states the reductions as "90%, 34%, and 10% at levels 1, 2, and 3, respectively" (p11).

Per-level proportions on a DL3DV-10K scene (Fig. 6, p7, reported): FLoD-3DGS level 1 "#G's: 7K
(0.7%) SSIM: 0.56", level 2 "18K (2%) SSIM: 0.70", level 3 "223K (22%) SSIM: 0.88", level 4 "475K
(47%) SSIM: 0.93", level 5 "1015K (100%) SSIM: 0.96". Octree-3DGS on the same scene: "25K (9%) SSIM:
0.40", "119K (17%) SSIM: 0.56", "276K (39%) SSIM: 0.68", "560K (78%) SSIM: 0.83", "713K (100%) SSIM:
0.92". (Assignment of the two rows to the two methods follows the figure caption ordering; the
extracted text does not preserve the row binding, so treat the method labels as inferred.)

### 4.4 The resource model

**There is no resource model.** GPU memory is measured, never predicted. The mapping from level to
memory is stated qualitatively:

> "Adjusting the 3D scale constraint provides multiple rendering options with different memory
> requirements, as larger 3D scale constraints result in fewer Gaussians needed for scene
> reconstruction." (p13)

The only quantitative structure is the level-selection distance, Eq. 8 (p5), which is a geometric
map from level to a switching distance, not a resource model:

  d_proj^(l) = ( s_min^(l) / γ ) × f

derived by solving the proportion s_min^(l) : γ = d_proj^(l) : f, where f is the camera focal
length and γ is the screen size threshold in pixel lengths. Boundary conditions d_proj^(L_end) = 0
and d_proj^(L_start − 1) = ∞.

Stated accuracy of any count-to-memory or level-to-memory model: **not stated**.

### 4.5 The control law

There is no runtime controller in the training sense. Training is a fixed coarse-to-fine schedule,
one level at a time.

Level-by-level training (p4): level 1 initialised from SfM points. At the end of level l the
Gaussians are cloned and saved as the final level-l set, then used to initialise level l+1 with
scale re-parameterised, Eq. 5 (p4):

  s_opt ← log( s^(l) − s_min^(l+1) )

so that s^(l+1) = s^(l) at the transition. "It prevents abrupt initial loss by eliminating the gap"
(p4).

Schedule (p6, all reported): training iterations per level 1-5 are 10,000 / 15,000 / 20,000 /
25,000 / 30,000. Density control (densification, pruning, overlap pruning, opacity reset) runs for
the first 5,000 / 6,000 / 8,000 / 10,000 / 15,000 iterations respectively. Densification intervals
are 2,000 / 1,000 / 500 / 500 / 200 iterations, versus 100 in the 3DGS backbone. Overlap pruning
runs every 1000 iterations at all levels except the max level, where it is not applied.

- **Can the count grow while the controller is active?** Yes, unrestrictedly. Nothing caps the count
  at any level. The count is bounded only indirectly, by the scale floor s_min^(l) making small
  Gaussians impossible and by overlap pruning removing coincident ones.
- **Is overshoot allowed?** The question does not apply. There is no count or memory target to
  overshoot. The appendix explicitly rejects a count constraint as the mechanism (p13):

  > "An alternative method is to create multi-level 3DGS representations by directly limiting the
  > Gaussian count. However, limiting the Gaussian count without enforcing scale constraints cannot
  > reconstruct each level's representation with the level of detail controlled."

  > "Therefore, limiting the Gaussian count without scale constraints would degrade reconstruction
  > quality." (p13)

Selective rendering set, Eq. 7 (p5), where d_G^(l) is the Gaussian's distance from the camera:

  G_sel = ⋃[l = L_start .. L_end] { G^(l) ∈ G^(l) | d_proj^(l−1) > d_G^(l) ≥ d_proj^(l) }

Two variants: predetermined G_sel (distances measured once from the average training camera
position, fixed thereafter) and per-view G_sel (recomputed each frame from the current camera).

### 4.6 Device numbers

All reported. The low-cost device is a laptop GeForce MX250 with 2 GB VRAM. The server is an RTX
A5000 with 24 GB VRAM.

Figure 8 (p8) reports, for the six rendering options on one scene, the tuple PSNR / memory / FPS on
A5000 / FPS on MX250. The extracted values, in the order they appear in the text layer, are:

```
PSNR: 22.9  memory: 0.61GB  FPS: 304(A5000) 28.7(MX250)
PSNR: 23.0  memory: 0.76GB  FPS: 274(A5000) 17.9(MX250)
PSNR: 25.5  memory: 0.81GB  FPS: 218(A5000) 13.2(MX250)
PSNR: 25.8  memory: 1.27GB  FPS: 178(A5000) 10.6(MX250)
PSNR: 26.4  memory: 1.21GB  FPS: 150(A5000) 8.4(MX250)
PSNR: 26.9  memory: 2.06GB  FPS: 113(A5000) OOM(MX250)
```

The six configuration labels printed in the figure are, in their own layout order, "level {4,3,2}
level {3,2,1} level 3 level 4 level 5 level {5,4,3}". The text layer does not preserve which label
goes with which tuple, so the binding is **not recoverable** from the extraction and is not asserted
here. What is certain from the caption and body: level 5 alone hits OOM on the MX250, and the
level-5 configuration is the 2.06 GB / 113 FPS / 26.9 dB row.

> "Various rendering options of FLoD-3DGS are evaluated on a server with an A5000 GPU and a laptop
> equipped with a 2GB VRAM MX250 GPU. The flexibility of FLoD-3DGS provides rendering options that
> prevent out-of-memory (OOM) errors and allow near real-time rendering on the laptop setting."
> (p8, Fig. 8 caption)

Table 7 (p16), FPS on the MX250 laptop across Mip-NeRF 360 scenes, "✗" meaning OOM, all reported.
The caption says "for 7 scenes" but nine columns are printed, so the caption count is wrong:

| level config | bicycle | flowers | garden | stump | treehill | room | counter | kitchen | bonsai |
|---|---|---|---|---|---|---|---|---|---|
| level 5 | ✗ | 6.52 | ✗ | ✗ | 5.77 | 5.54 | 6.00 | 3.99 | 7.48 |
| levels 5,4,3 | 5.10 | 8.81 | 6.92 | 8.48 | 8.33 | 6.27 | 6.58 | 4.20 | 8.69 |
| level 4 | 7.71 | 10.25 | 7.27 | 10.41 | 9.87 | 8.35 | 8.71 | 5.67 | 9.16 |
| levels 4,3,2 | 8.53 | 11.38 | 7.98 | 13.20 | 11.39 | 8.42 | 8.79 | 5.73 | 9.31 |
| level 3 | 9.21 | 15.00 | 13.54 | 18.19 | 12.97 | 9.67 | 11.65 | 10.44 | 11.68 |
| levels 3,2,1 | 9.34 | 15.60 | 13.98 | 20.92 | 13.77 | 9.72 | 11.73 | 10.49 | 11.85 |

(Row-to-config binding derived from the checkmark pattern in the extracted table: rows alternate
single level and triple level from 5 down to 3.)

The claim about real-time on the laptop is weak against these numbers. The fastest laptop figure in
the table is 20.92 FPS on stump at levels 3,2,1, and most cells are under 12 FPS. The paper says
"in some cases (e.g., bonsai), FLoD enables real-time rendering" (p16), where bonsai tops out at
11.85 FPS.

Small City scene, Table 4 (p10), device not restated, presumably the A5000, all reported:

| config | PSNR | FPS | mem. | #G's |
|---|---|---|---|---|
| FLoD-3DGS (per-view) | 25.49 | 221 | 1.03 GB | 601K |
| FLoD-3DGS (predetermined) | 24.69 | 286 | 0.41 GB | 589K |
| Hierarchical-3DGS (τ=30) | 24.69 | 55 | 5.36 GB | 610K |
| FLoD-3DGS (max level) | 26.37 | 181 | 0.86 GB | 1308K |
| Hierarchical-3DGS (τ=0) | 26.69 | 17 | 7.81 GB | 4892K |

Storage after compression, Table 9 (p16), Mip-NeRF 360 level 5 single-level rendering, reported:
FLoD-3DGS 103 FPS / 518 MB / 27.8 / 0.815 / 0.224; FLoD-3DGS+LightGS 144 FPS / 31.7 MB / 27.1 /
0.799 / 0.250. The paper states "compressing FLoD-3DGS reduces storage disk usage by 93% and
enhances rendering speed" (p16).

### 4.7 Main quantitative results

Table 1 (p9), max-level rendering, all reported. **There is no Deep Blending experiment** in FLoD.
The third dataset is DL3DV-10K (six outdoor scenes).

| Method | Mip-NeRF 360 PSNR/SSIM/LPIPS | DL3DV-10K PSNR/SSIM/LPIPS | Tanks&Temples PSNR/SSIM/LPIPS |
|---|---|---|---|
| 3DGS | 27.36 / 0.812 / 0.217 | 28.00 / 0.908 / 0.142 | 23.58 / 0.848 / 0.177 |
| Mip-Splatting | 27.59 / 0.831 / 0.181 | 28.64 / 0.917 / 0.125 | 23.62 / 0.855 / 0.157 |
| Octree-3DGS | 27.29 / 0.815 / 0.214 | 29.14 / 0.915 / 0.128 | 24.19 / 0.865 / 0.154 |
| Hierarchical-3DGS | 27.10 / 0.797 / 0.219 | 30.45 / 0.922 / 0.115 | 24.03 / 0.861 / 0.152 |
| FLoD-3DGS | 27.75 / 0.815 / 0.224 | 31.99 / 0.937 / 0.107 | 24.41 / 0.850 / 0.186 |

Strongest two baselines: Hierarchical-3DGS (best non-FLoD on DL3DV-10K, 30.45 dB) and Octree-3DGS
(24.19 dB on T&T, 0.865 SSIM, best SSIM on T&T among all). Mip-Splatting has the best LPIPS on
Mip-NeRF 360 (0.181) and T&T (0.157), better than FLoD in both.

Table 2 (p9), Mip-NeRF 360 quality/speed/count trade-off across rendering options, all reported:

| levels | PSNR | SSIM | LPIPS | FPS | #G's |
|---|---|---|---|---|---|
| 5 | 27.75 | 0.815 | 0.224 | 103 | 2189K |
| 5,4,3 | 27.33 | 0.801 | 0.245 | 124 | 1210K |
| 4 | 26.67 | 0.764 | 0.292 | 150 | 1049K |
| 4,3,2 | 26.48 | 0.759 | 0.298 | 160 | 856K |
| 3 | 24.11 | 0.634 | 0.440 | 202 | 443K |
| 3,2,1 | 24.07 | 0.632 | 0.442 | 208 | 414K |

Comparison against compression baselines, Table 8 (p16), PSNR and memory in GB, all reported:

| Method | Mip PSNR / mem | DL3DV PSNR / mem | T&T PSNR / mem |
|---|---|---|---|
| FLoD-3DGS(lv5) | 27.8 / 1.8 | 31.9 / 1.0 | 24.4 / 1.1 |
| FLoD-3DGS(lv4) | 26.6 / 1.2 | 30.7 / 0.6 | 23.8 / 0.6 |
| FLoD-3DGS(lv3) | 24.1 / 0.8 | 28.3 / 0.5 | 21.7 / 0.5 |
| LightGS | 26.6 / 1.2 | 27.2 / 0.7 | 23.3 / 0.6 |
| CompactGS | 26.8 / 1.1 | 27.8 / 0.5 | 22.8 / 0.8 |

Backbone compatibility, Table 3 (p10), PSNR and memory in GB per level, all reported:

| Method | Mip PSNR / mem | DL3DV PSNR / mem | T&T PSNR / mem |
|---|---|---|---|
| FLoD-Scaffold(lv1) | 20.1 / 0.5 | 22.2 / 0.3 | 17.1 / 0.2 |
| FLoD-Scaffold(lv2) | 22.1 / 0.5 | 25.2 / 0.3 | 19.3 / 0.3 |
| FLoD-Scaffold(lv3) | 24.7 / 0.6 | 28.5 / 0.4 | 21.8 / 0.4 |
| FLoD-Scaffold(lv4) | 26.6 / 0.8 | 30.1 / 0.6 | 23.6 / 0.7 |
| FLoD-Scaffold(lv5) | 27.4 / 1.0 | 31.1 / 0.7 | 24.1 / 1.0 |
| Scaffold-GS | 27.4 / 1.3 | 30.5 / 0.8 | 24.1 / 0.7 |
| Octree-Scaffold | 27.2 / 1.0 | 30.9 / 0.6 | 24.6 / 0.8 |

Ablation, level-by-level training on DL3DV-10K (Table 5, p11), reported, PSNR w/o vs w/ LT per
level: lv5 31.20 → 31.97, lv4 29.05 → 30.73, lv3 27.05 → 28.29, lv2 23.41 → 24.01, lv1 20.41 →
20.81. Scale-constraint ablation (p11): "the case with the 3D scale constraint uses approximately
98.6% fewer Gaussians compared to the case without the 3D scale constraint" after level 2 training
(12K versus 246K, Fig. 12).

Dataset inconsistency to flag: Section 6.1.1 (p6) says "seven scenes are from Mip-NeRF360" while
Appendix A.2 (p13) says "we use the nine publicly available scenes: bicycle, bonsai, counter,
garden, kitchen, room, stump, treehill and flowers". Table 7 lists nine columns.

### 4.8 Level selection at render time, and whether a memory or time budget can be an input

**How a level is selected.** Two modes, both user-driven.

1. **Single-level rendering.** The user picks one l ∈ [1, L_max] by hand. Every level independently
   reconstructs the full scene, so this is legal for any l. "users can choose any single level for
   rendering to match their GPU memory capabilities. This approach is similar to how games or
   streaming services let users adjust quality settings to optimize performance for their devices."
   (p4)

2. **Selective rendering.** The user picks a range [L_start, L_end] and a screen-size threshold γ.
   The per-Gaussian test is on world-space distance from the camera, Eq. 7, using precomputed
   switching distances d_proj^(l) = (s_min^(l)/γ) × f. Because s_min^(l) is fixed per level, d_proj
   is fixed too:

   > "Since s(l) min is fixed for each level, d(l) proj is also fixed. Thus, constructing the
   > Gaussian set Gsel only requires calculating the distance of each Gaussian from the camera,
   > d_G(l) . This method is computationally more efficient than the alternative, which requires
   > calculating each Gaussian's 2D projection and comparing it with the screen size threshold γ at
   > every level." (p5)

   Semantics of γ: "The threshold γ and the level range [Lstart, Lend] can be adjusted... A smaller
   threshold and a high-level range prioritize fine details over memory and speed, while a larger
   threshold and a low-level range reduce memory use and speed up rendering at the cost of fine
   details." (p5). Default γ = 1.0, meaning a level is used from the distance where its scale floor
   projects to one pixel or less (p7). Increasing γ beyond 1 introduces visible seams (Fig. 19,
   p15).

   Predetermined variant computes d_G once from the average training-camera position and freezes the
   set, so non-selected Gaussians can be dropped from memory entirely. Per-view variant recomputes
   per frame and must keep all levels in [L_start, L_end] resident, which costs memory.

No interpolation between levels is implemented: "inconsistency due to level transitions in selective
rendering is unlikely, which is why we did not implement interpolation between successive levels"
(p16).

**Can a memory or time budget be given as input?** **No.** The inputs are a level index, a level
range and a pixel threshold γ. The paper repeatedly frames the level as something the user chooses
*to match* an available memory, never something computed from a supplied number. There is no
inverse map from GB or FPS to (l, γ). The workflow is: measure the per-level memory, then pick the
level that fits.

**Memory and quality per level.** Three consistent sources, all reported.

Mip-NeRF 360, FLoD-3DGS, Table 8 (p16): lv5 27.8 dB / 1.8 GB, lv4 26.6 dB / 1.2 GB, lv3 24.1 dB /
0.8 GB.

Mip-NeRF 360, FLoD-Scaffold, Table 3 (p10): lv1 20.1 / 0.5 GB, lv2 22.1 / 0.5, lv3 24.7 / 0.6,
lv4 26.6 / 0.8, lv5 27.4 / 1.0 GB.

Figure 5 (p6) prints ten per-level memory values for the two backbones on one scene. Raw extracted
sequence, binding to level and backbone **not recoverable** from the text layer:
"memory: 0.75GB memory: 1.27GB memory: 2.06GB memory: 0.31GB memory: 0.25GB memory: 0.43GB
memory: 0.68GB memory: 0.98GB memory: 0.24GB memory: 0.24GB".

Selective-rendering memory versus quality, garden scene, Fig. 7 (p7), reported, FLoD-3DGS:
level{3,2,1} 0.73 GB (29%) PSNR 24.02; level{4,3,2} 1.29 GB (52%) PSNR 26.23; level{5,4,3} 1.40 GB
(57%) PSNR 26.71; level5 2.45 GB (100%) PSNR 27.64. Hierarchical-3DGS on the same scene: τ=120 3.53
GB (79%) PSNR 20.98; τ=30 3.72 GB (83%) PSNR 23.47; τ=15 4.19 GB (93%) PSNR 24.71; τ=0 4.46 GB
(100%) PSNR 26.03. (Percentages disambiguate the binding here.)

Additional selective-rendering points, Fig. 16 (p15), Tanks&Temples and DL3DV-10K, reported:
FLoD-3DGS 0.52 GB (38%) PSNR 23.30; 0.59 GB (43%) 24.76; 0.75 GB (54%) 25.32; 1.37 GB (100%) 25.98.
Second scene: 0.54 GB (49%) 27.60; 0.60 GB (55%) 28.76; 0.68 GB (63%) 29.84; 1.09 GB (100%) 31.17.

### 4.9 Stated limitations and future work

The paper has a dedicated Section 8 "Limitation" (p12), quoted in full:

> "In scenes with long camera trajectories, using per-view Gaussian set is necessary to maintain
> consistent rendering quality during selective rendering. However, this method has the limitation
> that all Gaussians within the level range for selective rendering need to be kept on GPU memory
> to maintain fast rendering rates, as discussed in Section 6.5. Therefore, this method requires
> more memory capacity compared to single level rendering with only the highest level, 𝐿end, picked
> from the level range [𝐿start, 𝐿end] used for selective rendering. Future research could explore
> the strategic planning and execution of transferring Gaussians from the CPU to the GPU, to reduce
> the memory burden while also keeping the advantage of selective rendering." (p12)

Two further self-stated weaknesses:

> "In urban scenes, where cameras cover extensive areas, selective rendering with a predetermined
> Gaussian set Gsel can result in noticeable decline in rendering detail." (p10)

> "On the other hand, increasing the screen size threshold 𝛾beyond 1 can introduce visible
> inconsistencies in the rendering, as shown in Figure 19." (p16)

### 4.10 Verbatim quotes touching a size, memory or FPS target, budget, or edge/mobile deployment

> "GeForce MX250 (2GB VRAM) CUDA out of memory." (p1, Fig. 1 in-figure label)

> "The green box illustrates max-level rendering on a high-end server, while the pink box shows
> subset-level rendering for a low-cost laptop, where traditional 3DGS fails to render. Thus,
> FLoD-3DGS can flexibly adapt to diverse hardware settings." (p1, Fig. 1 caption)

> "However, 3DGS and its subsequent works are restricted to specific hardware setups, either on only
> low-cost or on only high-end configurations." (p1)

> "Approaches aimed at reducing 3DGS memory usage enable rendering on low-cost GPU but compromise
> rendering quality, which fails to leverage the hardware capabilities in the case of higher-end
> GPU." (p1)

> "Conversely, methods that enhance rendering quality require high-end GPU with large VRAM, making
> such methods impractical for lower-end devices with limited memory capacity." (p1)

> "FLoD constructs a multi-level 3DGS representation through level-specific 3D scale constraints,
> where each level independently reconstructs the entire scene with varying detail and GPU memory
> usage." (p1)

> "Furthermore, the multi-level structure of FLoD allows selective rendering of image regions at
> different detail levels, providing additional memory-efficient rendering options." (p1)

> "Experiments demonstrate that FLoD provides various rendering options with trade-offs between
> quality and memory usage, enabling real-time rendering under diverse memory constraints." (p1)

> "However, its reliance on numerous Gaussian primitives makes it impractical for rendering on
> devices with limited GPU memory." (p2)

> "Conversely, LightGaussian [Fan et al. 2023] and CompactGS [Lee et al. 2024] address memory
> limitations by removing redundant Gaussians, which helps reduce rendering memory demands as well
> as reducing storage size. However, the reduction in memory usage comes at the expense of rendering
> quality." (p2)

> "As a result, they lack the flexibility to adapt and produce optimal renderings across various GPU
> memory capacities." (p2)

> "At lower levels, models possess reduced geometric and textural detail, which decreases memory and
> computational demands. Conversely, at higher levels, models have increased detail, leading to
> higher memory and computational demands." (p2)

> "While these methods excel at creating detailed high-level representations, rendering with only
> lower-level representations to accommodate middle or low-cost GPU settings causes significant
> scene content loss and distortions." (p2)

> "Our method applies a level-specific 3D scale constraint, which increases each successive level,
> to limit the amount of detail reconstructed and the rendering memory demand." (p2)

> "Our trained FLoD representation provides the flexibility to choose any single level based on the
> available GPU memory or desired rendering rates." (p2)

> "In contrast, we propose a multi-level 3DGS that increases rendering flexibility by enabling
> rendering across various GPU settings, ranging from server GPUs with 24GB VRAM to laptop GPUs
> with 2GB VRAM." (p2)

> "However, these methods primarily target efficient rendering on high-end GPUs, such as A6000 or
> A100 GPUs with 48GB or 80GB VRAM." (p3)

> "Therefore, theses methods cannot provide rendering options with lower memory demands." (p3)

> "This method also reduces the overall memory footprint." (p4)

> "Levels and rendering methods can be adjusted to achieve the desired rendering rates or to fit
> within available GPU memory limits." (p4)

> "From our multi-level set of 3D Gaussians {G(𝑙) | 𝑙= 1, . . . , 𝐿max}, users can choose any single
> level for rendering to match their GPU memory capabilities." (p4)

> "However, rendering a large number of Gaussians may exceed the memory limits of commodity devices.
> In such cases, lower levels can be chosen to match the memory constraints." (p4)

> "Although a single level can be simply selected to match GPU memory capabilities, utilizing
> multiple levels can further enhance visual quality while keeping memory demands manageable." (p4)

> "This arrangement of multiple level Gaussians can achieve perceptual quality comparable to using
> only high-level Gaussians but at a reduced memory cost." (p5)

> "The threshold 𝛾and the level range [𝐿start, 𝐿end] can be adjusted to accommodate specific memory
> limitations or desired rendering rates. A smaller threshold and a high-level range prioritize fine
> details over memory and speed, while a larger threshold and a low-level range reduce memory use
> and speed up rendering at the cost of fine details." (p5)

> "This strategy enables high-quality rendering while reducing rendering memory and storage
> overhead." (p5)

> "This predetermined approach allows for non-sampled Gaussians to be excluded, significantly
> reducing memory consumption during rendering." (p5)

> "As a result, this method is especially beneficial for low-cost devices with limited GPU memory
> and storage capacity." (p5)

> "To maintain fast rendering rates, all Gaussians within the level range [𝐿start, 𝐿end] are kept in
> GPU memory. Therefore, with the cost of increased rendering memory, selective rendering with
> per-view Gsel effectively maintains consistent rendering quality over long camera trajectories."
> (p5)

> "FLoD can be integrated with both 3DGS and Scaffold-GS, with each level offering varying levels of
> detail and memory usage." (p6, Fig. 5 caption)

> "Additionally, we assess the number of Gaussians used for rendering the scenes, the GPU memory
> usage, and the rendering rates (FPS) to evaluate resource efficiency." (p6)

> "This configuration effectively distinguishes the level of detail across 𝐿max levels in most of
> the scenes we handle, enabling LoD representations that adapt to various memory capacities." (p6)

> "This enables users to select an appropriate level for rendering based on the desired visual
> quality and available memory." (p7)

> "Various rendering options of FLoD-3DGS are evaluated on a server with an A5000 GPU and a laptop
> equipped with a 2GB VRAM MX250 GPU. The flexibility of FLoD-3DGS provides rendering options that
> prevent out-of-memory (OOM) errors and allow near real-time rendering on the laptop setting."
> (p8)

> "For example, selectively using levels 5, 4, and 3 reduces memory usage by about half compared to
> using only level 5, while the PSNR decreases by less than 1. Similarly, selective rendering with
> levels 3, 2, and 1 reduce memory usage to approximately 30%, with PSNR drop of about 3.6." (p8)

> "Even when the target granularity 𝜏is set to 120, occupied GPU memory remains high, consuming
> approximately 79% of the memory used for the maximum rendering quality setting (𝜏= 0)." (p8)

> "The results show that FLoD-3DGS consistently uses less memory and achieves higher fps than
> Hierarchical-3DGS when compared at the same PSNR levels." (p8)

> "Reducing the number of Gaussians increases rendering speed while also reducing memory usage,
> allowing FLoD to adapt efficiently to hardware environments with varying memory constraints."
> (p8)

> "As shown in Figure 8, rendering with only level 4 or selective rendering using levels 5, 4, and 3
> achieves visual quality comparable to rendering with only level 5, while reducing memory usage by
> approximately 40%. This reduction prevents out-of-memory (OOM) errors that occur on low-cost GPUs,
> such as the MX250, when rendering with only level 5." (p8)

> "Furthermore, using lower levels for single-level rendering or selective rendering increases fps,
> enabling near real-time rendering even on low-cost devices." (p8)

> "Compared to using the predetermined Gsel, per-view Gsel increases PSNR by 0.8, but with a slower
> rendering speed and more rendering memory demands (Table 4)." (p10)

> "Therefore, the 3D scale constraint is crucial for ensuring varied detail across levels and
> enabling each level to maintain a different memory footprint." (p11)

> "This reduction is particularly important for minimizing memory usage for rendering on low-cost
> and low-memory devices that utilize low level representations." (p11)

> "Therefore, our method enables customizable rendering with a single or subset of levels, allowing
> the model to operate on devices ranging from high-end servers to low-cost laptops." (p12)

> "FLoD controls the level of detail and corresponding memory usage by training Gaussians with
> explicit 3D scale constraints. Adjusting the 3D scale constraint provides multiple rendering
> options with different memory requirements, as larger 3D scale constraints result in fewer
> Gaussians needed for scene reconstruction." (p13)

> "In Hierarchical-3DGS, increasing the target granularity 𝜏does not significantly reduce memory
> usage, even though fewer Gaussians are used for rendering at larger 𝜏values. This occurs because
> all Gaussians, across every hierarchy level, are loaded onto the GPU according to the release code
> for evaluation." (p15)

> "To demonstrate its effectiveness on low-cost devices, we measure FPS for Mip-NeRF360 scenes on
> the laptop equipped with an MX250 GPU (2GB VRAM)." (p16)

> "As shown in Table 7, single-level rendering at level 5 causes out-of-memory (OOM) errors in some
> scenes (e.g., stump). However, using selective rendering with levels 5, 4, and 3, or switching to
> a lower single level, resolves these errors. Additionally, in some cases (e.g., bonsai), FLoD
> enables real-time rendering. Thus, FLoD can provide adaptable rendering options even for low-cost
> devices." (p16)

> "LightGaussian [Fan et al. 2023] and CompactGS [Lee et al. 2024] also address memory-related
> issues, but their primary focus is on creating a single compressed 3DGS with small storage size.
> In contrast, FLoD constructs multi-level LoD representations to accommodate varying GPU memory
> capacities during rendering." (p16)

> "To demonstrate the efficiency of FLoD-3DGS in GPU memory usage during rendering, we compare PSNR
> and GPU memory consumption across levels 5, 4, and 3 of FLoD-3DGS and the two baselines." (p16)

> "FLoD-3DGS can store and render specific levels as needed. However, keeping the option of
> rendering with all levels requires significant storage disk space to accommodate them." (p16)

> "As shown in Table 9, compressing FLoD-3DGS reduces storage disk usage by 93% and enhances
> rendering speed." (p16)

> "Despite this, we demonstrate that FLoD-3DGS can be further optimized to suit devices with
> constrained storage by incorporating compression techniques." (p16)

### 4.11 Code URL and licence

Code URL: **not stated** in the PDF. No repository or project page appears in the text layer.

Licence, as printed on p1:

> "This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
> International License. © 2025 Copyright held by the owner/author(s). ACM 1557-7368/2025/8-ART
> https://doi.org/10.1145/3731430"

---

## 5. ConeGS

### 5.1 Citation

- Title: "ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction With
  Fewer Primitives" (p1).
- First author: Bartłomiej Baranowski, University of Tübingen, Tübingen AI Center. Co-authors
  Stefano Esposito, Patricia Gschoßmann, Anpei Chen (corresponding), Andreas Geiger.
- Venue: **not stated**. Only the arXiv stamp "arXiv:2511.06810v2 [cs.CV] 21 Jan 2026" (p1). Layout
  is the CVPR two-column template.
- arXiv id: 2511.06810.

### 5.2 Base representation and renderer

Vanilla 3DGS primitives and rasteriser, with an **iNGP model as a geometric proxy**. The iNGP is
the proposal-based NerfAcc implementation, trained 20k iterations, and it supplies per-pixel median
depth used both for initialisation and for every densification event. Gaussian optimisation runs
30k iterations with densification active for the first 25k. All SH components are optimised from
iteration 0 (p6). No rasteriser change.

### 5.3 Budget handle

**Two handles.**

1. **TARGET, denominated in primitive count (hard cap).** "The first enforces a hard upper bound,
   as in [25], ensuring that densification never exceeds the prescribed budget." (p5) Budgets tested
   in the paper: 10k, 50k, 100k, 200k, 500k, 1M, 2M (Fig. 6 x-axis, p8; tables for 100k, 500k, 1M,
   2M).

   The cap also gates initialisation: "We initialize the scene with Pinit Gaussians, set to one
   million as in [42], or fewer if a smaller budget is specified" (p4). Baselines are held to the
   same cap: "with the densification stopped for all of them if the primitive budget is reached. If
   the number of primitives at initialization would be higher than the specified budget, the number
   of primitives is sampled uniformly to fit below it." (p6)

2. **KNOB, dimensionless growth rate β (no-budget mode).** "The second strategy adapts the number of
   primitives to the scene's complexity, enabling controlled growth without imposing a fixed upper
   bound" (p6). β ∈ {0, 0.01, 0.02, 0.04} tested.

No byte, GB or millisecond handle exists.

**Achieved versus requested.** In fixed-budget mode the paper asserts the cap is never exceeded and
prints no achieved count, so the achieved-versus-requested pair is only asserted, never tabulated.
The one place where achieved counts appear is the β mode, where nothing was requested (Tables A1 and
A2, p12, reported):

| β | achieved on Mip-NeRF 360 | achieved on OMMO |
|---|---|---|
| 0 | 542k | 352k |
| 0.01 | 674k | 438k |
| 0.02 | 942k | 581k |
| 0.04 | 1.66 M | 927k |
| (matched to 3DGS) | 2.57 M | 1.75 M |

Averaged across both datasets, the β curve x-axis reads 441k, 548k, 750k, 1.27M, 2.19M (Fig. 6,
p8). So β = 0.04 produces 1.66 M on one dataset and 927k on the other, a 1.8× spread. This is the
paper's own evidence that β is a knob and not a target.

The count trace over training under a set budget versus an unset one is plotted in Figure A2 (p13)
for the garden scene, at budgets 100k and 1M and at β = 0.04 and 0.01, showing pruned, added,
accumulated and total counts per densification event. No numbers are printed.

### 5.4 The resource model

**There is no resource model.** The resource is the count itself. Memory is measured once, not
modelled. Table A4 (p13), peak GPU memory during optimisation on the 15 scene of OMMO at a 500k cap,
all reported:

| method | peak memory (MiB) |
|---|---|
| Ours | 9545 |
| 3DGS (SfM) | 9049 |
| MCMC (SfM) | 8671 |
| EDGS | 14517 |
| Perceptual-GS | 12403 |

Frame time is likewise measured, not modelled, though the paper offers a mechanistic explanation and
one supporting statistic. Mean Gaussians blended per pixel on Mip-NeRF 360 at a 1M cap (Table A7,
p14, reported): Ours 30.72, MCMC (SfM) 49.55, MCMC 48.45, 3DGS (SfM) 34.74. The paper reads that as
the cause of its FPS advantage:

> "This primitive distribution not only improves geometric alignment but also increases rendering
> speed by reducing blending and sorting overhead during rasterization. To support this hypothesis,
> Table A7 reports the mean number of Gaussians blended per pixel. Our method consistently requires
> less blending compared to the selected benchmarks. The difference is especially large compared to
> MCMC, which blends over 60% more Gaussians per pixel on average." (p15)

Stated accuracy of any model: **not stated**, because no model is fitted.

One near-model does exist, but for primitive *size* rather than resource. The initial scale of a new
Gaussian is set from the pixel cone, Eq. 15 (p5), with λ_scale = 2 and r_cone from Mip-NeRF Eq. 7:

  sᵢ = λ_scale · r_cone(t_med,i) · (1, 1, 1),   r_cone(t) = t (‖d_x − d‖ + ‖d_y − d‖)/2

and its accuracy from other viewpoints *is* measured (Fig. A3, p14): "we measure the size of newly
added pixel-sized primitives from multiple viewpoints... The distribution shows that the apparent
size remains near one pixel, with very few cases above four pixels or below 0.2 pixels." (p14)

### 5.5 The control law

Prune-then-replenish, applied every 100 iterations, over a pixel-error-driven sampler.

Pixel sampling, Eq. 12 (p4), with E(p) = |I_j(p) − I*(p)| the per-pixel L1 error and M a multinomial
without replacement:

  {p_s}[s=1..N_sample] ∼ M( N_sample, E(p) / ∑[p′ ∈ I_j] E(p′) )

Accumulation and merge, Eqs. 13-14 (p4):

  G_accum ← G_accum ∪ {G_s}[s=1..N_sample]
  every 100 iterations:  G_scene ← G_scene ∪ G_accum,  G_accum ← ∅

**Budgeted mode, Eq. 16 (p5-6):**

  N_sample = max(0.2 N_GS, 1.2 N_last) / 100

with N_GS the current total count, N_last the number inserted at the previous densification event,
and the division by 100 reflecting the densification interval.

> "This formulation keeps NGS close to the budget limit even under aggressive pruning, maintaining
> consistent scene coverage throughout optimization." (p6)

Note this formula contains **no budget term**. The cap is enforced separately as a hard clamp, and
Eq. 16 only sets how many candidates accumulate between merges. The paper explains the resulting
slack (p13-14):

> "When no budget is specified, all accumulated primitives are added to the scene (see Eq. 17).
> Under a fixed budget, however, not all of them are used. This is because accumulation happens
> every iteration and must remain available for pruning and densification, while the exact number
> that can be added is unknown beforehand. As a result, the system accumulates more primitives than
> are usually required to keep the total count close to the budget after pruning (see Eq. 16)."

**Unbudgeted mode, Eq. 17 (p6):**

  N_sample = β N_GS / 100

Pruning: opacity below 0.005 pruned every 100 iterations (p6), plus a **pre-activation** L1 opacity
penalty:

  L_o^pre = ‖o_pre‖₁,   λ_o = 0.0002

contrasted with the MCMC post-activation penalty L_o^post = ‖σ(o_pre)‖₁:

> "This applies the strongest constraint around 0.5 and only a weak penalty near the pruning
> threshold. In contrast, we employ a pre-activation opacity penalty Lpre o = ‖opre‖1. It provides a
> steady constraint across the full opacity range, including very low values, gradually reducing
> under-contributing primitives." (p6)

Initialisation: P_init Gaussians at iNGP median depth, Eqs. 8-11 (p4). Median depth from
transmittance, Eq. 9: t_med = t_k where τ_{k−1} > 0.5 ≥ τ_k. Position p_j = p_I + t_med d_I(u, v),
identity rotation, opacity 0.1, SH DC from the iNGP colour with zeroed view direction, scale from
3-NN mean distance (Eq. 3).

- **Can the count grow while the controller is active?** Yes, densification runs continuously for
  the first 25k of 30k iterations. Under a cap, growth is replenishment only: "we set the number of
  sampled pixels Nsample so that newly added Gaussians replace those pruned, avoiding excess
  primitives that would otherwise be discarded under the budget" (p5).
- **Is overshoot allowed?** No, at the scene level. The cap is a hard upper bound. Overshoot is
  allowed and expected in the *accumulation buffer*, which is deliberately over-filled and then
  truncated at merge time (p13-14 quote above). The budget is therefore respected exactly at every
  merge boundary, and slack exists off-scene.

Loss: standard 3DGS L_GS = (1 − λ) MAE(I, I*) + λ L_D-SSIM with λ = 0.2 (Eq. 4, p3), plus λ_o L_o^pre.

### 5.6 Device numbers

Named devices (p6):

> "All FPS measurements were recorded on an NVIDIA RTX 2080 Ti, whereas training speeds are reported
> on an NVIDIA A100, since EDGS requires more memory."

No Jetson, phone or laptop result. The RTX 2080 Ti is the closest thing to a modest GPU in the
paper. Selected FPS on the RTX 2080 Ti, all reported.

Mip-NeRF 360 averages by budget (Tables A8-A11, pp17-18):

| budget | Ours FPS | 3DGS (SfM) FPS | MCMC (SfM) FPS | EDGS FPS |
|---|---|---|---|---|
| 100k | 325 | 363 | 323 | 379 |
| 500k | 205 | 167 | 143 | 204 |
| 1M | 140 | 109 | 92 | 131 |
| 2M | 88 | 76 | 60 | 83 |

OMMO averages: 100k 361 / 177 / 320 / 359; 500k 217 / 136 / 142 / 115; 1M 148 / 96 / 97 / 101; 2M
87 / 78 / 64 / 86 (Ours / 3DGS / MCMC / EDGS).

Tanks&Temples averages: 100k 194 / 123 / 158 / 182; 500k 131 / 82 / 77 / 109; 1M 103 / 60 / 46 / 80;
2M 73 / 54 / 39 / 56.

Deep Blending averages: 100k 468 / 505 / 551 / 465; 500k 278 / 156 / 194 / 273; 1M 197 / 109 / 168 /
181; 2M 127 / 80 / 104 / 112.

Training time on the A100, Mip-NeRF 360, 1M cap (Table 2, p8, in minutes, reported):

| | Ours 10k iters | Ours 20k iters | Ours 40k iters | EDGS | 3DGS (SfM) | MCMC (rand) | MCMC (SfM) |
|---|---|---|---|---|---|---|---|
| 3DGS time | 20.7 | 20.5 | 20.5 | 19.4 | 22.6 | 26.3 | 25.1 |
| Init. time | 1.6 | 3.1 | 6.1 | 2.3 | - | - | - |
| Overall time | 22.3 | 23.6 | 26.6 | 21.7 | 22.6 | 26.3 | 25.1 |
| FPS | 134 | 137 | 137 | 131 | 112 | 92 | 95 |
| PSNR | 29.33 | 29.37 | 29.37 | 29.18 | 28.71 | 28.98 | 29.23 |

Also reported: OMMO 500k, "ConeGS 29.14 / 0.892 / 0.170 / 217 FPS / 25 min train" and "ConeGS (iNGP
retrain) 29.27 / 0.893 / 0.168 / 212 FPS / 29 min" (Table A5, p13).

### 5.7 Main quantitative results

Table 1 (p7), 100k and 500k caps, all reported. This is the paper's headline table and it does
include Deep Blending, plus OMMO.

At 100k Gaussians:

| Method | Mip-NeRF 360 | OMMO | Tanks&Temples | Deep Blending |
|---|---|---|---|---|
| 3DGS (SfM) | 23.61 / 0.693 / 0.413 | 26.45 / 0.820 / 0.296 | 22.38 / 0.774 / 0.333 | 24.65 / 0.827 / 0.412 |
| Foroutan et al. | 26.64 / 0.781 / 0.318 | 26.89 / 0.829 / 0.276 | 22.48 / 0.766 / 0.341 | 25.31 / 0.822 / 0.418 |
| MCMC (rand) | 25.72 / 0.730 / 0.369 | 25.92 / 0.808 / 0.313 | 21.45 / 0.750 / 0.365 | 27.94 / 0.859 / 0.369 |
| **MCMC (SfM)** | 27.06 / 0.800 / 0.303 | 27.01 / 0.841 / 0.266 | 22.50 / 0.780 / 0.332 | 28.94 / 0.876 / 0.333 |
| MCMC (iNGP) | 27.35 / 0.797 / 0.299 | 26.95 / 0.837 / 0.265 | 22.69 / 0.775 / 0.326 | 29.02 / 0.872 / 0.337 |
| GaussianPro | 25.57 / 0.766 / 0.338 | 26.14 / 0.822 / 0.289 | 20.59 / 0.757 / 0.348 | 28.15 / 0.870 / 0.342 |
| Perceptual-GS | 25.66 / 0.774 / 0.320 | 26.13 / 0.819 / 0.292 | 20.76 / 0.759 / 0.347 | 28.32 / 0.874 / 0.338 |
| **EDGS** | 27.09 / 0.798 / 0.296 | 26.99 / 0.838 / 0.261 | 22.32 / 0.777 / 0.324 | 28.43 / 0.872 / 0.337 |
| **Ours** | **27.74 / 0.809 / 0.285** | **27.59 / 0.852 / 0.243** | **23.12 / 0.791 / 0.310** | **29.44 / 0.880 / 0.328** |

At 500k Gaussians:

| Method | Mip-NeRF 360 | OMMO | Tanks&Temples | Deep Blending |
|---|---|---|---|---|
| 3DGS (SfM) | 28.22 / 0.821 / 0.260 | 28.96 / 0.883 / 0.196 | 23.54 / 0.816 / 0.265 | 29.25 / 0.882 / 0.302 |
| Foroutan et al. | 28.88 / 0.862 / 0.204 | 28.80 / 0.884 / 0.185 | 23.52 / 0.816 / 0.257 | 29.61 / 0.886 / 0.297 |
| MCMC (rand) | 28.38 / 0.844 / 0.237 | 28.23 / 0.874 / 0.212 | 22.96 / 0.808 / 0.279 | 28.84 / 0.875 / 0.315 |
| MCMC (SfM) | 28.82 / 0.861 / 0.214 | 28.72 / 0.885 / 0.194 | 23.68 / 0.825 / 0.259 | 29.44 / 0.887 / 0.308 |
| MCMC (iNGP) | 29.02 / 0.867 / 0.198 | 28.75 / 0.885 / 0.186 | 23.57 / 0.823 / 0.243 | 29.61 / 0.885 / 0.288 |
| GaussianPro | 28.02 / 0.827 / 0.253 | 28.32 / 0.876 / 0.206 | 22.31 / 0.802 / 0.286 | 29.33 / 0.886 / 0.301 |
| Perceptual-GS | 28.66 / 0.856 / 0.211 | 28.42 / 0.876 / 0.203 | 22.77 / 0.813 / 0.267 | 29.44 / 0.888 / 0.296 |
| **EDGS** | 28.82 / 0.865 / 0.193 | 29.01 / 0.893 / 0.168 | 23.47 / 0.831 / 0.229 | 29.33 / 0.889 / 0.286 |
| **Ours** | 29.08 / 0.870 / 0.190 | 29.14 / 0.892 / 0.170 | 23.69 / 0.829 / 0.229 | 29.86 / 0.891 / 0.285 |

Bounded by Mini-Splatting2 count: Mini-Splatting2 28.89 / 0.875 / 0.183 (Mip), 28.06 / 0.875 / 0.198
(OMMO), 22.79 / 0.823 / 0.239 (T&T), 29.99 / 0.898 / 0.279 (DB); Ours 29.26 / 0.875 / 0.179, 28.90 /
0.887 / 0.179, 23.66 / 0.829 / 0.231, 29.82 / 0.891 / 0.280.

At 1M (Table A3, p13): Ours 29.37 / 0.880 / 0.168 (Mip), 29.44 / 0.899 / 0.154 (OMMO), 23.70 / 0.835
/ 0.207 (T&T), 29.69 / 0.889 / 0.273 (DB). EDGS beats it on OMMO and T&T LPIPS at this budget.

Strongest two baselines: **MCMC (SfM/iNGP init.)**, which is the closest at 100k and the only method
to beat ConeGS on T&T PSNR at 1M (23.93), and **EDGS**, which beats ConeGS at 1M and 2M on several
LPIPS entries but costs 14517 MiB peak memory versus 9545 MiB.

Headline delta as stated (p8): "It achieves up to 0.6 PSNR increase and 20% speedup over
cloning-based baselines."

Ablation, 100k Gaussians on Mip-NeRF 360 (Table 3, p8), PSNR / SSIM / LPIPS / FPS, all reported:
Ours 27.74 / 0.810 / 0.285 / 328; (f) cone-sized initialization 27.38 / 0.806 / 0.287 / 437; (g)
without initialization 27.46 / 0.811 / 0.285 / 415; (h) uniform image-space sampling 27.51 / 0.806 /
0.286 / 307; (i) densify with 3DGS depth 27.43 / 0.797 / 0.296 / 332; (j) SfM init + 3DGS depth dens.
27.15 / 0.790 / 0.302 / 329; (k) k-NN scaling 27.54 / 0.802 / 0.295 / 299; (l) no opacity penalty
27.31 / 0.794 / 0.301 / 294; (m) post-densification opacity decrease 27.46 / 0.798 / 0.297 / 256;
(n) MCMC-style opacity penalty 27.49 / 0.803 / 0.293 / 239; (o) λ_scale = 1 27.70 / 0.810 / 0.285 /
329; (p) λ_scale = 4 27.63 / 0.808 / 0.285 / 329.

Note (f) and (g): dropping initialisation costs 0.28-0.36 dB but buys 87-109 FPS at the same 100k
count, which is a genuine count-versus-speed dissociation worth remembering.

### 5.8 Level selection (not applicable)

No LoD. The budget is fixed at training time and there is no render-time selection mechanism.

### 5.9 Stated limitations and future work

Dedicated Limitations paragraph, quoted in full (p8):

> "Limitations: ConeGS performs well on standard scenes but may struggle with large-scale
> environments, inaccurate poses, or sparse views, occasionally creating floaters (see appendix).
> Benefits are also reduced at high Gaussian budgets, where dense coverage limits the impact of
> error-guided placement, primarily offering faster rendering." (p8)

Expanded failure analysis (p15):

> "While our method generally produces strong reconstructions on almost all tested scenes, its
> reliance on iNGP can make it susceptible to floaters in challenging scenarios such as noisy poses,
> very large or sparse-view scenes, or other cases where reliable iNGP reconstruction is difficult.
> One such example is shown in Figure A6, where the scene contains distortions and lacks sufficient
> viewpoint coverage near the cameras. This leads to spurious high-density regions in iNGP close to
> the cameras, which in turn degrades densification quality by placing Gaussians in incorrect
> regions. Although this can reduce performance, the method still produces high-quality
> reconstructions overall (see per-scene results)."

On the β mode (p12):

> "However, this configuration does not allow explicit control over the number of primitives and may
> limit the achievable reconstruction quality."

On the L1 error signal (p4):

> "While L1 loss does not always indicate possible improvements and can also arise from noise or
> difficult-to-optimize reflections, we found it to be a reliable indication of lacking
> expressiveness, especially at low primitive budgets."

No future-work section.

### 5.10 Verbatim quotes touching a size, memory or FPS target, budget, or edge/mobile deployment

> "A pre-activation opacity penalty rapidly removes redundant Gaussians, while a primitive budgeting
> strategy controls the total number of primitives, either by a fixed budget or by adapting to scene
> complexity, ensuring high reconstruction quality." (p1)

> "Experiments show that ConeGS consistently enhances reconstruction quality and rendering
> performance across Gaussian budgets, with especially strong gains under tight primitive
> constraints where efficient placement is crucial." (p1)

> "We further incorporate two primitive budgeting strategies to regulate the total number of
> primitives, either through a fixed budget or by adapting to scene complexity." (p2)

> "ConeGS outperforms baseline methods across diverse datasets and a wide range of primitive
> budgets. The advantage is most pronounced under tight primitive budgets. At higher budgets, it
> matches the quality of cloning-based methods, where efficient primitive placement is less
> critical, while still rendering faster than the baselines." (p2)

> "An improved opacity penalty that promptly removes low-opacity Gaussians, combined with a
> budgeting strategy that balances scene complexity and primitive count." (p2)

> "Recent efforts have also targeted reducing computational and memory costs, often through feature
> quantization or code-book encoding [10, 15, 34, 41], or scene simplification [63]." (p3)

> "Like our method, they start by estimating scene geometry, but rely only on correspondences from a
> pretrained dense matching network, without enhancing densification, and at higher GPU memory cost
> than our approach." (p3)

> "Additionally, the depths can be evaluated on the fly during optimization, reducing both memory
> usage and computation compared to precomputing all depth maps." (p4)

> "We initialize the scene with Pinit Gaussians, set to one million as in [42], or fewer if a
> smaller budget is specified (see Section 4.2)." (p4)

> "While L1 loss does not always indicate possible improvements and can also arise from noise or
> difficult-to-optimize reflections, we found it to be a reliable indication of lacking
> expressiveness, especially at low primitive budgets." (p4)

> "Primitive Budgeting: We consider two budgeting strategies for controlling the number of Gaussians
> in the scene. The first enforces a hard upper bound, as in [25], ensuring that densification never
> exceeds the prescribed budget. This constraint regulates memory and computation while preventing
> uncontrolled growth of the primitive set." (p5)

> "At each densification step, we set the number of sampled pixels Nsample so that newly added
> Gaussians replace those pruned, avoiding excess primitives that would otherwise be discarded under
> the budget." (p5)

> "This formulation keeps NGS close to the budget limit even under aggressive pruning, maintaining
> consistent scene coverage throughout optimization." (p6)

> "The second strategy adapts the number of primitives to the scene's complexity, enabling
> controlled growth without imposing a fixed upper bound" (p5-6)

> "Here, β controls the growth rate of the primitive set. Smaller values balance the number of
> Gaussians added with those pruned, maintaining a relatively stable primitive count, whereas larger
> values yield higher primitive counts, increasing geometric detail at the cost of memory and
> computing power." (p6)

> "All FPS measurements were recorded on an NVIDIA RTX 2080 Ti, whereas training speeds are reported
> on an NVIDIA A100, since EDGS requires more memory." (p6)

> "...with the densification stopped for all of them if the primitive budget is reached. If the
> number of primitives at initialization would be higher than the specified budget, the number of
> primitives is sampled uniformly to fit below it." (p6)

> "We additionally test on Mini-Splatting2 [11] by matching their final number of primitives instead
> of a specific budget, due to their method relying on generating a high number of initial
> Gaussians." (p6)

> "For a lower limit on Gaussians, we outperform the benchmarks across all datasets and metrics,
> while providing a competitive reconstruction quality compared to the best performing baselines on
> the high budget scenarios." (p6)

> "The qualitative results in Figure 5 show significant improvement using a wide range of primitive
> budgets, demonstrating that for the same limit of primitives our method is able to produce a much
> better reconstruction, especially in areas that are challenging to properly capture on a low
> budget, such as isolated or high frequency structures." (p7)

> "ConeGS consistently improves reconstruction quality and rendering performance across Gaussian
> budgets, with strong gains under tight primitive constraints. It achieves up to 0.6 PSNR increase
> and 20% speedup over cloning-based baselines." (p8)

> "Benefits are also reduced at high Gaussian budgets, where dense coverage limits the impact of
> error-guided placement, primarily offering faster rendering." (p8)

> "The comparisons show that our method is able to produce high-quality results even without
> specifying a budget, instead adjusting the number of primitives based on the scene complexity,
> while still remaining sparse in the number of primitives." (p12)

> "However, this configuration does not allow explicit control over the number of primitives and may
> limit the achievable reconstruction quality." (p12)

> "If an even lower number of Gaussians is desired while still using a no-budget scenario, the
> number of initialized Gaussians can be reduced, effectively lowering the number of primitives
> generated at the end." (p12)

> "We evaluate the peak GPU memory usage during optimization for several methods in Table A4. The
> results show that our method is very close to 3DGS [24] and MCMC [25] in terms of memory
> consumption, while remaining considerably lower than EDGS [27] and Perceptual-GS [64]. This
> efficiency allows our method to run on a wider range of GPUs, making it more accessible to devices
> with limited memory." (p13)

> "Peak GPU memory usage on the 15 scene from the OMMO dataset [33] on the maximum budget of 500k
> primitives." (p13, Table A4 caption)

> "When no budget is specified, all accumulated primitives are added to the scene (see Eq. 17).
> Under a fixed budget, however, not all of them are used. This is because accumulation happens
> every iteration and must remain available for pruning and densification, while the exact number
> that can be added is unknown beforehand. As a result, the system accumulates more primitives than
> are usually required to keep the total count close to the budget after pruning (see Eq. 16)."
> (p13-14)

> "This primitive distribution not only improves geometric alignment but also increases rendering
> speed by reducing blending and sorting overhead during rasterization." (p15)

### 5.11 Code URL and licence

Project page as printed under the author list on p1:

> "baranowskibrt.github.io/conegs"

No explicit repository link and no licence stated in the PDF.

---

## 6. Cross-paper synthesis

### 6.1 Classification

| paper | handle | denomination | hard cap | growth allowed under control | overshoot | binds bytes or ms |
|---|---|---|---|---|---|---|
| Diet | TARGET F | primitive count (peak) | yes, per-iteration | yes, while \|G\| < F | transient, within one iteration | no (GB measured only) |
| YOGO | TARGET N_target_m | primitive count, per polygon | yes, by quota clamp | yes, quota decays to 0 | no, max(0, ·) clamp | no |
| CLoD-GS | KNOB s_v | dimensionless distance scale | no, soft ReLU penalty | yes, densification untouched | yes, free below target ratio | bytes only as an accounting identity, 248 (+4) B per Gaussian |
| FLoD | KNOB level l, γ in pixels | level index, screen-size pixels | no | yes, unrestricted per level | not applicable | no (GB measured only) |
| ConeGS | TARGET count cap, plus KNOB β | primitive count | yes, hard | yes, replenishment only | no on scene, yes in accumulation buffer | no |

### 6.2 What binds bytes or milliseconds rather than count

Only two threads in this set touch bytes or time as a first-class quantity, and neither is a
controller input.

1. **CLoD-GS bytes.** The paper states 248 bytes per Gaussian for a standard 3DGS implementation and
   +4 bytes for its own decay parameter (p10). Its Mem(MB) column is exactly count × 252 bytes / 2²⁰
   for its own rows and count × 248 bytes / 2²⁰ for baselines, verified to the reported decimals in
   four independent rows (note-taker derivation, not a paper claim). This is a byte model of the
   *model file*, not of runtime memory, and it is never inverted. A user still supplies s_v, not MB.

2. **FLoD milliseconds by proxy.** FPS is the paper's second axis alongside GB and it is measured on
   two named devices including a 2 GB MX250 laptop. The switching rule is the only thing in this set
   that is a genuine screen-space criterion in physical units: d_proj^(l) = (s_min^(l)/γ) × f puts a
   level boundary at the distance where a level's minimum world-space scale projects to γ pixels.
   That is a pixel budget, not a millisecond budget, but it is the one control law here whose input
   carries a unit the renderer can measure.

3. Storage in bytes appears once more, in FLoD Table 9 (p16): 518 MB → 31.7 MB with LightGaussian
   compression, 93% reduction, 27.8 → 27.1 dB.

Nothing in these five papers takes a byte count or a millisecond target as input and drives training
onto it. Every controller here is denominated in primitive count, level index, or a dimensionless
scale.

### 6.3 Device table

| paper | device | peak memory | FPS | notes |
|---|---|---|---|---|
| Diet | NVIDIA Jetson AGX Xavier | 8.55 GB (Ours), 2.98 GB (Ours w/loader), 10.01 GB (Taming 3DGS), 18.59 GB (3DGS), all training peak | not reported | the only true edge device in the set |
| Diet | "GTX 4090" | 8.86 GB (Ours), 18.29 GB (3DGS), training peak | not reported | 7 min bicycle training, from 10 min |
| YOGO | none | not reported | not reported | OOM failures for three baselines, device unnamed |
| CLoD-GS | four NVIDIA RTX 4090 (training) | model bytes only (159-1006 MB) | 57-88 (BungeeNeRF), 170-199 (T&T), 129-187 (DB), 109-141 (Mip) | FPS measurement device not stated |
| FLoD | GeForce MX250 laptop, 2 GB VRAM | OOM at level 5 on bicycle, garden, stump | 3.99-20.92 across 9 Mip-NeRF 360 scenes and 6 level configs | fastest laptop figure 20.92 FPS on stump at levels 3,2,1 |
| FLoD | RTX A5000, 24 GB VRAM | 0.2-2.06 GB rendering, 0.41-7.81 GB on Small City | 103-208 (Mip level sweep), 17-286 (Small City) | H-3DGS uses 5.36-7.81 GB where FLoD uses 0.41-1.03 GB |
| ConeGS | NVIDIA RTX 2080 Ti | not measured on this device | 88-468 depending on dataset and cap | all FPS measurements |
| ConeGS | NVIDIA A100 | 9545 MiB (Ours), 8671 (MCMC), 9049 (3DGS), 14517 (EDGS), 12403 (Perceptual-GS), training peak at 500k | not applicable | training speed device |

### 6.4 Notes for the project

- The only paper that measures **training** memory on an edge device is Diet, and it does so without
  ever building the count → GB model that would make its target meaningful in GB. Its F is a count.
  Roughly three quarters of its Jetson footprint was the dataset, not the model, which is a warning
  for any peak-training-memory budget: the model term may not dominate.
- Diet, YOGO and ConeGS all implement the same shape of controller, a per-event quota that closes
  the gap to a count target. They differ in the clamp. YOGO's max(0, Δ/(K−k+1)) is the cleanest
  formulation and is the only one that anneals the quota to zero as the schedule ends.
- ConeGS is the only paper in this set that reports quality at seven budgets (10k through 2M) on
  four datasets, so it is the best available quality-versus-count curve to calibrate against.
- FLoD and CLoD-GS are the two LoD papers and they sit at opposite ends: FLoD stores L_max complete
  independent models and picks one, so per-level memory is genuinely separable and a level can be
  dropped from disk; CLoD-GS stores one model plus one float per Gaussian and modulates it, so the
  disk footprint never shrinks below the full model. If the deployment budget is a *storage* budget,
  FLoD's structure fits and CLoD-GS's does not. If it is a *rendered count per frame* budget,
  CLoD-GS's mask is the finer instrument.
- No paper here reports a requested-versus-achieved hit-accuracy table. The closest evidence is
  ConeGS's β sweep, which shows a 1.8× spread in achieved count for the same β across two datasets,
  and YOGO's fusion ablation, which shows a 0.10 M spread at a fixed budget.
