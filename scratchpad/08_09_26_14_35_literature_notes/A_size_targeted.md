# Literature notes A: size-targeted family (plus MesonGS)

Written 2026-09-08 from the PDFs under `references/01_size_targeted/` and `references/02_rate_distortion/`, text
extracted with pymupdf (conda env `main`). Every page of every paper was read. Every number below is as reported by
the paper in question. Page numbers are PDF page indices (page 1 = first page of the file). For KISS-GS the appendix
carries its own printed numbering, so PDF p19 is printed appendix p1. "not stated" means the main PDF does not say it.
Supplementary material referenced by SizeGS and GETA-3DGS is not contained in the PDFs.

Papers:

1. SizeGS (Xie et al., ACM MM 2025, arXiv 2412.05808)
2. GETA-3DGS (Zhang and Sui, arXiv 2605.02086, formatted for IEEE TCSVT)
3. FlexGaussian (Tian et al., ACM MM 2025, arXiv 2507.06671)
4. KISS-GS (Morgenstern et al., ECCV 2026 per file name, arXiv 2608.26948)
5. MesonGS (Xie et al., ECCV 2024, arXiv 2409.09756)

---

## 1. SizeGS

### 1.1 Citation

"SizeGS: Size-aware Compression of 3D Gaussian Splatting via Mixed Integer Programming". First author Shuzhao Xie
(SIGS, Tsinghua University), equal contribution with Jiahang Liu. Venue as printed: "MM '25, October 27–31, 2025,
Dublin, Ireland", "Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)", DOI
10.1145/3746027.3755370 (p1). arXiv 2412.05808 (v2, 29 Nov 2025 per the arXiv stamp on p1). 10 pages, no
supplementary in the PDF.

### 1.2 Base representation

Three bases are compressed: Scaffold-GS anchors (Table 1, Table 3, all ablations), raw 3DGS via the MesonGS pipeline
(Table 2, Table 3 row "3DGS"), and 4DGS (Table 2). For Scaffold-GS the attributes searched are "features f, offsets O,
opacity o, scaling l, and rotation r" (p7, Fig. 5 caption) with 73 channels, versus 10 channels for the 3DGS/MesonGS
configuration (p7: "This is because 3DGS only has 10 channels that are involved in the search, while ScaffoldGS has
73."). Render-time MLP: yes for the Scaffold-GS base (the anchor MLPs of Scaffold-GS are inherited), no for the
3DGS/MesonGS base. Entropy decoding is required before rendering in all cases: coordinates are G-PCC coded and
attributes are LZ77 or torchac coded (p4). No hash grid.

### 1.3 Budget handle

The user supplies a size budget S_T in bytes. It is a TARGET. Problem (3) on p4:

    minimize_{τ,Q} M(τ, Q)  subject to  S(τ, Q) ≤ Size Budget,  τ ∈ [0, 1],  Q ∈ [1, 32]^{C×B} ∩ ℤ^{C×B}

The search loop terminates when |S_a − S_T| / |S_T| < 0.05 (Algorithm 1, line 11, p5), so the constraint is a 5 %
band by construction. Achieved versus requested, all as reported:

- Table 1 (p6): Mip-NeRF 360 budget 18.33 MB (average of per-scene budgets) → 18.17 MB, Tank&Temples 11 MB →
  10.93 MB, Deep Blending 8 MB → 7.92 MB. HAC with a binary search on λ: 18.29, 10.92 and 8.29 MB.
- Table 4 (p7), bicycle, budget 3 × 10^7 B: 0-1 ILP searched 29,831,203 B (Δ 168,797 B), Vanilla ILP 28,934,805 B
  (Δ 1,065,195 B), genetic algorithm 21,833,128 B (Δ 8,166,872 B).
- Table 6 (p8), bicycle: budget 30 MB → searched 29.85 / 29.91 / 29.86 MB for K = 40 / 30 / 50 blocks, budget 20 MB
  → 19.83 / 19.85 / 19.92 MB.
- Table 2 (p6): Our+MesonGS 35.36 MB against a per-scene target equal to the official MesonGS output (35.41 MB
  average). Our+4DGS 198.64 MB from a 5.10 GB 4DGS model, target not stated.

So the constraint is binding to within 1 % in the tables, tighter than the 5 % stopping band.

### 1.4 When it binds

Post-hoc on a frozen model, with optional fine-tuning afterwards. Algorithm 1 input: "Size budget S_T and a pre-trained
3DGS model" (p5). Pipeline (Fig. 2, p4): importance-score pruning with reserve ratio τ → voxelization (16-bit
coordinates, G-PCC) → group-wise quantization with bit-width setting Q → entropy coding, then "Piecewise Finetuning":
finetune after pruning, "fix xyz & finetune" after voxelization, finetune after quantization (p4 Fig. 2, p6 Sec. 3.3).

Iteration counts and times (all as reported): search per round on ScaffoldGS, Table 3 (p7): Ω 0.11 s, save 32.78 s,
0-1 ILP 145.22 s, finetune 857.71 s, end-to-end 1328.82 s. On 3DGS: 0.49 s, 15.10 s, 58.52 s, no finetune, 104.51 s.
"We set the time limitation for 0-1 ILP to 50s." (p7). "if we fine-tune for 6000 iterations, the average time required
is approximately 1000s" (p7). Table 1 comparison uses 4000 finetune iterations: "For comparison with the baseline
method, we set the number of finetune iterations as 4000. In practical applications, however, fine-tuning for only 500
steps is expected to achieve satisfactory visual quality." (p6). Table 5 (p8) fixes 6000 total finetune iterations
split over the three stages (best: 1000 after prune, 1000 after voxel, 4000 after quant). 4DGS: "SizeGS can achieves
the target size in 51s and improved visual quality after 28s of finetuning." (p6). "our method can search a set of
hyperparameters to compress 3D Gaussians to desired size in a minute" (p2). Hardware: not stated.

Size is fixed before fine-tuning and fine-tuning is done with hyperparameters frozen: "Since the hyperparameters are
fixed, our method has almost no impact on the file size during retraining." (p6). Coordinates are frozen during
fine-tuning because G-PCC decoding reorders points and the Morton-ordered quantization groups would change (p6).

### 1.5 What "size" means

On-disk bytes after entropy coding. "The final file includes geometry, attributes, and metadata (voxel size, block
count, and per-group scale/zero points)." (p4). Coordinates "are set to 16-bit and compressed using G-PCC", attributes
"compressed with LZ77 or torchac" (p4). Whether the Scaffold-GS MLP weights are counted: not stated. No hash grid.

### 1.6 Runtime cost reported

Encode/search time: Table 3 above and Table 1 "Time (s)" column: Ours 1328 / 1381 / 1263 s versus HAC 16627 / 10978 /
9993 s on Mip-NeRF 360 / Tank&Temples / Deep Blending (p6), the HAC time being the λ binary search with retraining.
Decode time: not stated. Render FPS: not stated. GPU memory at render: not stated. Hardware: not stated.

### 1.7 Size estimator and quality-loss estimator (full detail)

Quality-loss estimator Ω. Computing PSNR "requires traversing the entire training set and evaluating the metric, it
takes at least 10 seconds" (p5), so the metric M(·) is replaced by a quantization loss. Assumption: "the bit-width of
each group are independent of one another" (p5), so the loss is precomputed per group and per candidate bit-width with
only Q quantizations. Eq. (5), p5:

    Ω(i, j, b) = | Â^b_{i,j} − A_{i,j} |

where A ∈ ℝ^{C×(τN)} is the attribute matrix split into C×B groups, Â^b is the group restored after b-bit
quantization, and |·| "can be 1-norm, 2-norm, or ∞-norm" (p5). Q is set to 16 candidate bit-widths (1 to 16). The
relationship between PSNR and Ω is only shown in the supplementary: "which reveals that minimizing Ω is equal to
maximizing PSNR" (p5). Ablation Fig. 6b (p8): "the performance of 2-norm and ∞-norm are nearly the same, both of which
outperform 1-norm by a significant margin". Ω is computed by a CUDA kernel that quantizes all groups in parallel,
0.11 s for ScaffoldGS and 0.49 s for 3DGS (Table 3, p7). Quantizer (Eq. 2, p3): x̂ = ⌊clamp(x/s_x + z_x, 0, 2^{Q_b} − 1)⌉
with s_x = [max(x) − min(x)] / 2^{Q_b} and z_x = ⌊2^{Q_b} − max(x)/s_x⌉.

Size estimator S. The file is split into (1) voxelized coordinates, (2) quantized attributes, (3) metadata. Items (1)
and (3) are measured exactly by writing them to disk: "we can store the voxelized coordinates and metadata to obtain
the accurate compressed size in a few seconds" (p5). The information-theoretic estimate τN × (−Σ_i p_i log₂ p_i) is
rejected because it requires re-quantizing and histogramming at every ILP iteration and is nonlinear in Q (p5). The
estimator used is Eq. (6), p5:

    S(Q) = Σ_{i,j} P_{ij} Q_{ij} + C + S_Δ

"Here, P ∈ ℝ^{C×B} refers to size of quantization groups. C refers to the accurate storage consumption of the metadata
and the coordinates, which can be obtained by storing them to the disk directly." (p5). This is the raw bit count
(elements per group × bits) and ignores the entropy coder, and the paper says so: "Of course, such a estimation for the
compressed file size is not accurate. To calibrate it, we update the S_Δ multiple times, as shown in Algo. 1." (p5).

Calibration procedure (Algorithm 1, p5). Q is initialised to 8 bits everywhere. For each sampled τ ∈ {τ₁, τ₂, ...}:
compress the model, obtain the actual size S_a. If 2 × S_a < S_T, skip this τ (doubling all bit-widths up to the
16-bit cap could not fill the budget, so τ is too small, p5). Otherwise loop: S_Δ ← S_a − S_T, solve Q ← 01-ILP(Q, S_Δ)
with the previous Q as warm start, actually compress with the new Q and obtain S_a, stop when
|S_a − S_T| / |S_T| < 0.05. Keep the {τ, Q} with the best Ω. The "Save" column of Table 3 (32.78 s for ScaffoldGS) is
this actual encode used for calibration, and "size calibration and 0-1 ILP take similar time" (p7).

ILP formulation (Eq. 4, p5): Q ∈ {0, 1}^{C×B×Q} one-hot per group, minimize Ω(Q) subject to S(Q) ≤ Size Budget and
Σ_q Q_{i,j,q} = 1. Hierarchical solver (p6): step 1 searches channel-level bit-widths Q_c ∈ [1, 16]^C, step 2 derives a
per-channel budget S_c = S_T · Q_{c,i} / Σ Q_{c,i} and solves group-level Q_g ∈ [0, 16]^B per channel. Motivation: with
(C, B, Q) = (73, 60, 16) there are 1.8 × 10^58 options (p5). Solver: PuLP [46] (p2, p4). Table 4 (p7) shows the 0-1 ILP
reaches an "Information loss" of 11,826 versus 1,258,394 for a general integer ILP and 42,821,038 for a genetic
algorithm at the same 3 × 10^7 B budget.

Reported estimation error: the paper reports no estimate-versus-actual number for Eq. (6) itself. What it reports is the
outcome after calibration: the 5 % stopping band, Table 4's Δ size of 168,797 B on a 30,000,000 B budget, and Table 6's
19.83 to 19.92 MB on a 20 MB budget and 29.85 to 29.91 MB on a 30 MB budget.

### 1.8 Scene dependence

No per-scene size table in the main PDF: "The detailed size target of scenes are listed in supplementary material."
(p6, Table 1 caption). Per-scene numbers that do appear: bicycle at one operating point in Fig. 3 (p6): Ours 28.55 MB,
HAC 28.61 MB, MesonGS 46.70 MB, ReduGS 48.00 MB, SOGS (w/ SH) 51.00 MB, C3DGS 47.14 MB, Lee et al. 62.99 MB, EAGLES
102 MB, ScaffoldGS 248 MB. Bicycle with budgets 28 MB and 12 MB in Fig. 5 (p7). Fig. 1a (p3) plots size versus reserve
ratio for bicycle and train (curves only). Retuning: the method is built so that nothing is tuned by hand per scene,
τ and Q are searched per scene and per budget. For the MesonGS comparison "For each scene, the size target is set to
the compressed file size generated by the official MesonGS configuration." (p7). Bit-width pattern across budgets:
"under different size budgets, the choices of bit-widths are generally the same" (p7).

### 1.9 Main quantitative results (as reported)

Table 1 (p6), size-aware compression, Scaffold-GS base, HAC with λ binary search as the only baseline:

| Method | Mip-NeRF 360 (budget 18.33 MB) PSNR / SSIM / LPIPS / Size MB / Time s | Tank&Temples (11 MB) | Deep Blending (8 MB) |
|---|---|---|---|
| HAC | 27.17 / 0.789 / 0.261 / 18.29 / 16627 | 24.45 / 0.854 / 0.179 / 10.92 / 10978 | 30.27 / 0.910 / 0.254 / 8.29 / 9993 |
| Our | 27.48 / 0.806 / 0.240 / 18.17 / 1328 | 24.04 / 0.840 / 0.200 / 10.93 / 1381 | 30.24 / 0.903 / 0.271 / 7.92 / 1263 |

Table 2 (p6), 3DGS variants: MesonGS 25.95 / 0.7706 / 0.2679 / 35.41 MB / finetune 0 s, Our+MesonGS 26.06 / 0.7743 /
0.2642 / 35.36 MB / 0 s (Mip-NeRF 360). 4DGS 32.06 / 5.10 GB / 0 s, Our+4DGS 32.07 / 198.64 MB / 28 s (flame_steak,
N3DV). Fig. 4 (p7) gives rate-distortion curves against ScaffoldGS, Lee et al., C3DGS, EAGLES, ReduGS, SOGS, MesonGS
and HAC without numbers. The paper states "our goal is not to improve the marginal performance and defeat the existing
compression works" (p7, Fig. 4 caption).

### 1.10 Densification and count control

No densification (post-hoc). The count is τN, set by the reserve ratio τ chosen from a discrete list by the outer loop
of Algorithm 1 under the size constraint. Importance score Eq. (1), p3: I_d = Σ_{p∈P} α_i Π_{j=1}^{i−1}(1 − α_j), with
anchor importance for Scaffold-GS averaged over generated Gaussians (p4). Volume weighting is dropped: "Some works
[16, 59] use volume to weight I_d to obtain a more precise importance estimation, but this improvement is negligible."
(p3). For ScaffoldGS timing "we fix the value of τ to 0.6" (p7). The size objective influences the count only through
the outer τ sampling, never through densification.

### 1.11 Stated limitations and future work (verbatim)

- p5: "Of course, such a estimation for the compressed file size is not accurate. To calibrate it, we update the S_Δ
  multiple times, as shown in Algo. 1."
- p6: "During fine-tuning, we fix coordinates because G-PCC decompression yields unordered points, misaligned with
  attributes."
- p7: "Note that our goal is not to improve the marginal performance and defeat the existing compression works.
  Instead, we aim to design a hyperparameter parameter searching algorithm to compress the 3DGS model into the desired
  size while maximizing visual quality."
- p8: "Some works [8, 47, 48] address bitrate fluctuation via layered coding, which is orthogonal to our approach."
- No explicit "future work" paragraph exists. The conclusion (p8) only restates the method.

### 1.12 Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, or budget (verbatim)

- p1 (abstract): "We address this issue from the perspective of size-aware compression, where we aim to compress 3DGS to
  a desired size by quickly searching for suitable hyperparameters."
- p1: "with the goal of maximizing visual quality while respecting the size budget constraint."
- p2: "The key idea is to compress the 3DGS to desired size by automatically searching the suitable hyperparameter set."
- p2: "One is the offline method [16, 39, 41, 43, 58–60], which requires manually setting hyperparameters to adjust the
  compressed file size."
- p2: "The other is the online method [10, 27, 35, 54, 57], which selects hyperparameters based on a context model and
  imports an extra hyperparameter λ to balance rate and bit consumption. While λ can adjust the final size, each
  adjustment requires retraining the context model from scratch, which is time-consuming. Additionally, it is difficult
  to predict the final size given a certain λ. Therefore, existing methods are unable to search for feasible
  hyperparameters under the constraint of the size budget in a short amount of time."
- p3: "The online method balances rate and quality via λ, but adjusting hyperparameters requires retraining the context
  model. Hence, searching by size is time-consuming."
- p6: "We perform a binary search on the hyperparameter λ of HAC to find a configuration that meets the size budget.
  The search stops when the difference between the obtained size and the size budget is within 5%. The reason why HAC
  takes so much longer than ours is that, after each hyperparameter adjustment, it requires 10-20 minutes to retrain
  the mask and context models in order to converge to a specific size. In contrast, our method first searches for the
  appropriate hyperparameters based on the size, and then only requires a single retraining. Since the hyperparameters
  are fixed, our method has almost no impact on the file size during retraining."
- p6: "SizeGS can achieves the target size in 51s and improved visual quality after 28s of finetuning."
- p8: "FCGS[9] compresses 3DGS via one-shot inference but supports only fixed-size output and lacks adaptability to
  other 3DGS forms. Some works [8, 47, 48] address bitrate fluctuation via layered coding, which is orthogonal to our
  approach."
- p8: "Though some works [10, 35] employed MPQ, their required retraining for configuring the quantization settings."
- p8 (conclusion): "Experiments show SizeGS effectively controls file size while preserving quality."

None of these binds a size target during training. The closest is the HAC λ binary search, which the paper reports as
the baseline it beats on time.

### 1.13 Code

"Project page & Code: shuzhaoxie.github.io/sizegs." (p1). Licence of the code: not stated. The paper itself carries
"This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License." (p1),
which is the paper licence, not a code licence.

---

## 2. GETA-3DGS

### 2.1 Citation

"GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3D Gaussian Splatting". Authors Baobing Zhang
(University of Hertfordshire) and Wanxin Sui (Brunel University London). Venue as printed: running head "IEEE
TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO TECHNOLOGY, VOL. XX, NO. X, MONTH YEAR" and "Manuscript received Month
DD, YYYY" (p1), so a submitted manuscript, not a stated acceptance. arXiv 2605.02086 (v1, 3 May 2026, cs.LG). 14 pages,
supplementary (Tables SI to SX, Sec. S2) not in the PDF.

### 2.2 Base representation

Raw 3DGS Gaussians, D = 59 scalars per primitive at SH degree 3 (Eq. 4, p4), implemented on gsplat and PyTorch 2.5
(p9). Rendering needs no MLP, no hash grid and no entropy decoding, only dequantization of per-attribute symmetric
quantizers (Eq. 12, p6). "our prototype uses a generic per-tensor symmetric quantiser with no anchor or hash-grid
context" (p12). Table III reports a "Decode (ms)" of 2348 / 6209 / 761 ms described as "THE LOAD-AND-FIRST-RENDER
LATENCY AT DEPLOYMENT" (p11).

### 2.3 Budget handle

Three handles: "the storage budget B (or equivalently the Gaussian count K), (ii) the PSNR floor τ, and (iii) the
per-attribute bit ranges" (p8, Remark 2). B is presented as a TARGET in MB but is implemented as a KNOB (a soft sparsity
penalty) and is not binding. Verbatim, p8: "Known limitation: B is currently non-binding: the soft sparsity penalty
causes all swept B to saturate within 4.0–4.9 MB at the natural Gaussian count (Fig. 2). A hard-constraint
Lagrangian-dual controller (Supp. Sec. S2.7) is explicit future work; until then B is a coarse upper bound."

Achieved versus requested (as reported): Fig. 2 caption (p9): "GETA-3DGS sweep over 8 nominal target sizes B ∈ {3, 5,
8, 10, 12, 30, 60, 120} MB at the "compressive" bit preset; realised on-disk size is determined by the post-warmup
natural Gaussian count, so all eight target settings land in 4.0–4.2 MB — a ∼0.9 dB monotone trend emerges from
21.65 dB (B=5) to 22.55 dB (B=60)." Fig. 6 caption (p12): "Actual on-disk scene size against four nominal target
budgets {10, 30, 50, 100} MB; the dashed line marks y = budget. ... The size budget B is non-binding in the current
prototype: all four sweeps converge to a mean of ≈4 MB irrespective of target". The count target K is a hard constraint
in the formulation (Eq. 6, p4: |{i : g_i ≠ 0}| = K) but is relaxed whenever the PSNR floor would be violated (p8), and
Table IV (p12) at "SPARSITY TARGET K=50k" reports a mean post-pruning count N̄g of 77.1 k for the full method, so K was
not met either in that run. The production Table II runs use K = 500k with "no aggressive pruning" (p8).

### 2.4 When it binds

During training from scratch, with the compression stages appended to a vanilla warm-up. Algorithm 2 (p8): "Require:
Init scene Θ0 (or COLMAP cloud), target Gaussian count K, target size B, PSNR floor τ, per-attribute bit ranges ...".
Stages (p7 to p8): (W) warm-up [0, T1) "standard 3DGS training of Kerbl et al. [1] with adaptive densification and
opacity reset, no quantizer, no pruning", QADG built once at T1. (P) projection [T1, T2): quantizer on, bit upper bounds
contracted linearly by Eq. (16), Gaussian count fixed. (J) joint [T2, T3): saliency every T_sal = 500 iterations, prune
to K_t at 5 prune events, PSNR-floor feasibility check with relaxation factor λ ∈ (1, 2]. (C) cool-down [T3, T4):
structure and bits frozen, attributes trained. Production schedule (T1, T2, T3, T4) = (25k, 30k, 33k, 35k) (p7, p8,
p10). Ablation schedule (30k, 40k, 60k, 70k) with K = 50k (p7). The Implementation paragraph (p9) states the schedule
"uses cumulative boundaries (T1, T2, T3, T4) = (30k, 40k, 60k, 70k) iterations, matching the values in our reference
implementation", which conflicts with the 35k production schedule of Table II. Wall clock, Table III (p11): "Train (h)"
0.52 (Mip-NeRF 360), 0.33 (Tanks & Temples), 0.37 (Deep Blending), average 0.41 h on a single A100 for the 35k
schedule. Remark 1 (p8): "The end-to-end training cost of GETA-3DGS is 1.05–1.15× that of vanilla 3DGS at matched
iteration count". Note the tension with Sec. IV-A which starts from "A trained 3DGS scene" (p4) and Fig. 7's "All methods
are decoded from the same vanilla backbone" (p12): the warm-up stage is itself the vanilla training.

### 2.5 What "size" means

Internally inconsistent. Metrics paragraph (p9): "storage in megabytes (MB) after entropy coding". Table II caption
speaks of "OUR ENTROPY-CODED VANILLA BASELINE" (p10). But Table III (p11) defines "BYTES/GAUSS IS THE AVERAGE
POST-QUANTIZATION STORAGE COST PER PRIMITIVE IN BYTES (b̄×59/8)" = 63.7, a raw bit count, and Sec. V-E (p11) says "On
the converged GETA-3DGS quantised symbols, the zero-order Shannon entropy is 4.86 bits/scalar (vs. b̄=7.41), implying a
+33.6% rate-reduction headroom attainable by any entropy coder; off-the-shelf gzip already realises +19.0% at unchanged
PSNR. At our 4.86 MB Mip-NeRF 360 operating point, this projects to 3.23 MB (Shannon bound)". So the 4.86 MB figure
appears to be raw quantized bytes, not entropy coded. Included: quantized attributes only, no MLP, no hash grid,
quantizer state ϕ⋆ returned with the scene (Algorithm 2 line 31), headers not stated. The "Vanilla 3DGS" baseline is
20.83 MB on Mip-NeRF 360 with PSNR 25.15 (Table II), far below the usual vanilla numbers, and is the authors' own gsplat
run, presumably capped at K = 500k (not stated).

### 2.6 Runtime cost reported

Table III (p11), single A100, 35k-iteration schedule, averaged per benchmark: Train 0.52 / 0.33 / 0.37 h, Decode 2348 /
6209 / 761 ms, Render 5.04 / 4.11 / 4.35 ms per view, Peak GPU 419 / 248 / 116 MB (torch.cuda high-water mark during
rendering), Bytes/Gauss 63.7 for Mip-NeRF 360 / Tanks & Temples / Deep Blending, averages 0.41 h, 3106 ms, 4.50 ms,
261 MB. Table II FPS column: Vanilla 3DGS 692.8 / 664.8 / 944.9, Naive PTQ 656.2 / 473.7 / 846.5, GETA-3DGS "–". FPS
"on a single NVIDIA A100 GPU at 1080p" (p9).

### 2.7 Size estimator or rate model

None in the byte sense. The rate enters as (a) the count K enforced by projection at prune events, (b) per-attribute bit
ranges [b_l^(a), b_u^(a)] with a learned bit-width b^(a) = log₂((q_m^(a))^{t^(a)} / d^(a) + 1) + 1 (Eq. 11, p6),
projected by clamping t, and (c) a "soft sparsity penalty" that stands in for B (p8). How B is converted to K is not
stated. Bytes per Gaussian are reported as b̄ × 59 / 8. Gap between estimate and actual: the whole B sweep landing at
4.0 to 4.2 MB (p9) is the reported gap. The paper proposes, as future work, "sub-gradient ascent on the Lagrangian dual
of the byte-counter constraint" (p11).

### 2.8 Scene dependence

Per-scene storage only as reduction factors in Fig. 3b (p11): 5.1× on bicycle and 5.2× on the other eight scenes, "GETA
delivers a uniform ∼5.2× storage saving across scenes". Per-scene MB values are in "Supp. Tab. SII–SX" (p10), not in the
PDF. Fig. 6a (p12) shows "Faint blue traces are the nine individual scenes" all near 4 MB. Retuning statements: "a fully
automatic pipeline that requires no per-scene opacity, scale, or SH-degree thresholds" (p1), "the bit ranges are
framework defaults (Table I) and are never tuned per scene" (p8), "The saliency weights are set to (w1, w2, w3) = (0.5,
0.3, 0.2) and are not tuned per scene" (p9). Scene-dependent cost of uniform bits: "−6.74 dB on counter and −4.94 dB on
room ... versus only −0.18 to −0.34 dB on texture-uniform outdoor scenes (flowers, treehill)" (p9).

### 2.9 Main quantitative results (as reported, Table II p10)

| Dataset | Method | PSNR | SSIM | LPIPS | Size MB | Comp. ratio | FPS |
|---|---|---|---|---|---|---|---|
| Mip-NeRF 360 | Vanilla 3DGS | 25.15 | 0.712 | – | 20.83 | 1.0× | 692.8 |
| Mip-NeRF 360 | CompGS† | 27.16 | 0.808 | 0.228 | 54.60 | 0.4× | – |
| Mip-NeRF 360 | HAC++† | 27.60 | 0.803 | 0.253 | 8.34 | 2.5× | – |
| Mip-NeRF 360 | GETA-3DGS | 23.36 | 0.631 | 0.430 | 4.86 | 4.29× | – |
| Tanks & Temples | Vanilla 3DGS | 22.30 | 0.794 | – | 24.80 | 1.0× | 664.8 |
| Tanks & Temples | CompGS† | 23.47 | 0.840 | 0.188 | 54.60 | 0.5× | – |
| Tanks & Temples | HAC++† | 24.22 | 0.849 | 0.190 | 5.18 | 4.8× | – |
| Tanks & Temples | GETA-3DGS | 20.22 | 0.666 | 0.305 | 5.36 | 4.63× | – |
| Deep Blending | Vanilla 3DGS | 27.83 | 0.872 | – | 11.16 | 1.0× | 944.9 |
| Deep Blending | CompGS† | 29.75 | 0.903 | 0.247 | 54.60 | 0.2× | – |
| Deep Blending | HAC++† | 30.16 | 0.907 | 0.266 | 2.91 | 3.8× | – |
| Deep Blending | GETA-3DGS | 27.13 | 0.862 | 0.165 | 2.51 | 4.45× | – |

† = dataset averages copied from the original papers. Other rows: LP-3DGS† 27.47 / 0.812 / 0.227 / 271.6‡ MB,
FlexGaussian† 26.38 / 0.780 / 0.251 / 40.80 MB, Naive PTQ (8-bit) 16.73 / 0.398 / – / 20.13 MB on Mip-NeRF 360.
GETA-3DGS is 1.79 dB below its own vanilla on Mip-NeRF 360 and 4.24 dB below HAC++ at a smaller size. The paper's own
extrapolation at 4.90 MB: "the residual deficits are −1.84 dB versus HAC++, −1.74 dB versus CompGS, and −1.34 dB versus
FlexGaussian" (p10). Table IV ablation (p12, K = 50k): full 22.53 dB at N̄g = 77.1k and b̄ = 7.41, uniform 6-bit
20.35 dB (−2.18 dB). Time column: none in Table II, see Table III.

### 2.10 Densification and count control

Warm-up uses vanilla adaptive densification and opacity reset (p7, Algorithm 2 line 5). After T1 the count is only
reduced: prune mask of the bottom N − K_t Gaussians by render-aware saliency (Eq. 10, p6: convex fusion of
transmittance-weighted contribution, screen-space gradient and pixel coverage with weights 0.5 / 0.3 / 0.2), K_t =
N − ⌊ρ_t (N − K)⌋ over 5 prune events (p8). The PSNR floor overrides K: "if compressing to K_t primitives would drop
PSNR below τ, we relax K_t rather than τ, following the priority τ ≻ K ≻ ϕ. Concretely, we increase K_t by a
multiplicative factor λ ∈ (1, 2] and re-attempt the prune event; if the floor is still violated after a small budget of
retries, we abort pruning and continue cool-down at the current K." (p8). τ = PSNR_vanilla − 0.5 dB on training views
(p4). The size objective does not influence densification, and B does not influence the count in the current code ("all
sweeps converge to ~4 MB", Fig. 6a p12).

### 2.11 Stated limitations and future work (verbatim)

- p8: "Known limitation: B is currently non-binding: the soft sparsity penalty causes all swept B to saturate within
  4.0–4.9 MB at the natural Gaussian count (Fig. 2). A hard-constraint Lagrangian-dual controller (Supp. Sec. S2.7) is
  explicit future work; until then B is a coarse upper bound."
- p7: "The argument therefore justifies the per-attribute bit ordering of Table I but not the cross-scene magnitude of
  the uniform-6-bit cost; the latter is an open empirical observation requiring a higher-rate refinement of the analysis
  (Supp. Sec. S2.4)."
- p10: "Asymptotic saturation of bit-budget gains. A striking pattern in our sweep data (Fig. 2, Supp. Table SII) is
  that target sizes B ≥ 30 MB all converge to PSNR ≈ 22.5 dB at b̄ ≈ 7.4 bits, with no further gain from raising B."
- p11: "From soft to hard storage constraint. The non-binding behaviour of B (Fig. 6) is a direct consequence of
  treating B as a sparsity penalty rather than a hard byte cap. A standard sub-gradient ascent on the Lagrangian dual of
  the byte-counter constraint (see Supp. Sec. S2.7 for the full update rule and convergence discussion) would, in
  principle, restore B as a hard constraint without changing the QADG, saliency, or quantisation primitives, and is
  concrete future work rather than an architectural change to GETA-3DGS."
- p12: "Beyond the saturation regime, the absolute-quality gap also reflects that our prototype uses a generic per-tensor
  symmetric quantiser with no anchor or hash-grid context".
- p13 roadmap: "Replace the generic per-tensor quantizer with an anchor-based context model in the spirit of ContextGS
  [25]. Expected gain on Mip-NeRF 360: roughly +5 dB on PSNR at the same size", "Plug in a hash-grid entropy coder à la
  HAC/HAC++ [13], [14] on top of the quantized output of GETA-3DGS. Expected gain: a further +2 dB at matched size",
  "Co-design with rate-distortion-aware structured pruning. Our PPSG update currently optimizes a Lagrangian whose
  distortion term is the photometric+D-SSIM loss; replacing it with an explicit RD objective L + λR and folding the
  entropy coder gradient into PPSG should align the solver with the actual deployment metric.", "Extend to dynamic and
  4D Gaussian splatting", "Co-design with neural codecs at the bitstream layer."
- p13: "The remaining quality gap to dedicated codecs is a feature of the current prototype, not of the framework".

### 2.12 Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, or budget (verbatim)

- p1: "which limits their generalization across scenes and prevents users from directly specifying a target compression
  rate or rendering-quality budget."
- p1: "Mobile head-mounted displays, augmented reality glasses, and volumetric video streaming infrastructures, by
  contrast, demand model footprints in the tens of megabytes and per-frame bitrates of only a few megabits [27]."
- p1: "(iii) users cannot specify a hard target on storage size or a guaranteed lower bound on rendering quality."
- p2: "to deliver an automatic, white-box pipeline that produces models satisfying user-specified storage and bit-width
  budgets."
- p2: "a single user handle (storage budget B or PSNR floor τ) traces an entire rate–distortion operating curve."
- p3: "so the pruning ratio and SH-degree bit-widths are enforced as hard constraints rather than tuned through a
  sparsity coefficient."
- p3: "None of these methods, however, jointly optimizes structured pruning and mixed-precision quantization through a
  unified white-box optimizer with explicit size and quality constraints, which is the gap our work fills."
- p4 (Eq. 6 context): "K is the user-specified target number of surviving Gaussians, [b_l^(a), b_u^(a)] are the
  per-attribute bit-width bounds, and τ is a rendering-quality floor measured by PSNR over the training view set D_tr."
- p8: "This makes GETA-3DGS quality-safe by construction: the user can dial either K (storage-first), B (size-first), or
  τ (quality-first) and obtain a model that satisfies the stronger of the three constraints."
- p8: "Remark 2: The framework exposes three user handles: (i) the storage budget B (or equivalently the Gaussian count
  K), (ii) the PSNR floor τ, and (iii) the per-attribute bit ranges {[b_l^(a), b_u^(a)]}. With τ as a hard constraint,
  the front-end user typically sets only B and reads back the (possibly relaxed) achieved K; the bit ranges are
  framework defaults (Table I) and are never tuned per scene."
- p8: "Scope and operating point. GETA-3DGS exposes two user-facing knobs: a target storage budget B and a PSNR-floor
  tolerance τ."
- p8: "Known limitation: B is currently non-binding: the soft sparsity penalty causes all swept B to saturate within
  4.0–4.9 MB at the natural Gaussian count (Fig. 2). A hard-constraint Lagrangian-dual controller (Supp. Sec. S2.7) is
  explicit future work; until then B is a coarse upper bound."
- p9 (Fig. 2 caption): "Blue curve: GETA-3DGS sweep over 8 nominal target sizes B ∈ {3, 5, 8, 10, 12, 30, 60, 120} MB
  at the "compressive" bit preset; realised on-disk size is determined by the post-warmup natural Gaussian count, so
  all eight target settings land in 4.0–4.2 MB"
- p11: "From soft to hard storage constraint. The non-binding behaviour of B (Fig. 6) is a direct consequence of
  treating B as a sparsity penalty rather than a hard byte cap."
- p12 (Fig. 6 caption): "Size-budget non-binding behaviour on Mip-NeRF 360 (failure mode disclosure). (a) Actual on-disk
  scene size against four nominal target budgets {10, 30, 50, 100} MB; the dashed line marks y = budget. Faint blue
  traces are the nine individual scenes. The size budget B is non-binding in the current prototype: all four sweeps
  converge to a mean of ≈4 MB irrespective of target, and the solver returns the same operating point regardless of
  nominal budget (the failure mode is discussed in Section V-E). (b) PSNR is correspondingly nearly flat across nominal
  budgets, since the four sweeps share essentially the same achieved size."
- p12: "Automatic frontier traversal as a separate benefit. Independent of absolute quality, GETA-3DGS traces its own
  rate–distortion operating curve automatically by varying a single user hyperparameter (B or τ)."
- p12: "The joint optimization further enables dependable behaviour under hard constraints: when the size budget is
  tight, our scheme falls back on relaxing K rather than violating the PSNR floor τ, which is critical for production
  volumetric-video pipelines that expose τ as a deployment requirement rather than a soft objective."
- p13: "our method removes the per-scene threshold tuning common to prior work and lets users directly specify storage
  and quality budgets through a single handle."
- p13: "The implication for 3DGS compression is that whether enough bit budget is allocated matters far more than how it
  is allocated across attributes."

### 2.13 Code

No repository URL is printed anywhere in the PDF. The text refers to "our reference implementation (model.py,
quantizer.py, saliency.py, qasso.py, train_geta.py)" (p4) built on gsplat and PyTorch 2.5 (p9). Licence: not stated.

### 2.14 Additional questions for GETA-3DGS

Hard constraint or penalty weight? A penalty weight in the shipped code, presented as a target in the framing.
Verbatim: "The non-binding behaviour of B (Fig. 6) is a direct consequence of treating B as a sparsity penalty rather
than a hard byte cap." (p11). "Known limitation: B is currently non-binding: the soft sparsity penalty causes all swept
B to saturate within 4.0–4.9 MB at the natural Gaussian count (Fig. 2)." (p8). The formulation Eq. (6) (p4) writes the
count K and the bit ranges as hard constraints and the PSNR floor as a hard constraint, and B appears only in prose.

Does the achieved size equal what was requested? No. "all eight target settings land in 4.0–4.2 MB" for B ∈ {3, 5, 8,
10, 12, 30, 60, 120} MB (p9, Fig. 2 caption). "all four sweeps converge to a mean of ≈4 MB irrespective of target" for
B ∈ {10, 30, 50, 100} MB (p12, Fig. 6 caption). The B = 3 MB point is also missed in the upward direction (4.0 MB
achieved). At K = 50k the reported mean count is 77.1k (Table IV, p12).

From scratch or on a pretrained model? From scratch with a vanilla warm-up as the first stage. "Require: Init scene Θ0
(or COLMAP cloud)" (Algorithm 2, p8). "(W) Warm-up [0, T1): standard 3DGS training of Kerbl et al. [1] with adaptive
densification and opacity reset, no quantizer, no pruning. The purpose is to obtain a high-fidelity overcomplete scene
with which the dependency graph and the saliency are meaningful. The QADG of Section IV-B is built once at t=T1." (p7).
"Camera poses and sparse points come from COLMAP [12]." (p8). "The end-to-end training cost of GETA-3DGS is 1.05–1.15×
that of vanilla 3DGS at matched iteration count: warm-up is identical" (p8). The compression itself begins at T1 = 25k
iterations on the warm-up output, so functionally it is a pretrained checkpoint plus 10k compression iterations.

---

## 3. FlexGaussian

### 3.1 Citation

"FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3D Gaussian Splatting". First author Boyuan
Tian (UIUC), with Qizhe Gao, Siran Xianyu, Xiaotong Cui, Minjia Zhang. Venue as printed: "ACM MM, October 27–31, 2025,
Dublin, Ireland" (p2 running head). arXiv 2507.06671 (v1, 9 Jul 2025). 10 pages.

### 3.2 Base representation

Raw 3DGS Gaussians as an N × M matrix, M = 59 channels: xyz (3), scale (3), rotation quaternion (4), opacity, SH_base
(3), SH_adv (45) (p3). No MLP, no hash grid, no entropy decoding at render time. Rendering needs dequantization of
INT4/INT8 channels with per-sub-group ranges ("all attributes of each set of channels are arranged to 1000 sub-groups",
p4) and, after ADP, some Gaussians lack SH_adv.

### 3.3 Budget handle

"Given a user-specified target quality or compression ratio, FOA searches in the candidate sets until reaches the best
compression performance in observance to the constraints." (p4). Both are TARGETs on a discrete candidate set. The
knobs searched are the row-pruning ratio (Row-P, α%), the SH-pruning ratio (SH-P, 1 − α% − β%) and the per-channel
bit-widths (INT4 or INT8). The main experiments use a quality target, not a size target: "In our main experiments, we
target a <1 dB PSNR drop and evaluate adaptability separately." (p5). Achieved: "FlexGaussian reduces 94.9%, 96.1%, and
96.4% of the data across three datasets, with quality losses of 0.8 dB, 0.7 dB, and 0.8 dB, respectively — all well
below the 1 dB PSNR drop constraint" (p5). So the quality constraint binds. No experiment reports a requested
compression ratio against an achieved one. Fig. 11 (p7) shows reachable ratios "from 8 to 256×" with "PSNR losses of
0.05 to 9 dB" on Truck. No byte target appears.

### 3.4 When it binds

Post-hoc without training. "The solution should not require the training pipeline or access to the training images."
(p3). Training camera poses are still needed for the importance score: "Only complete evaluation sets are needed for
quality loss calculation, while training camera poses suffice for importance computation." (p6). FOA loop: one-time
model I/O and importance score, then per step "duplicates the input Gaussians, then prunes, quantizes, dequantizes, and
evaluates quality using standard metrics like PSNR" (p5). Times (Truck, RTX 3090, p6): total 20.3 s, of which 38.8 %
data loading, 13.9 % importance score, 47.1 % online adaptation, 0.2 % storage. "Each adaptation step takes about 1.56
seconds on average, with pruning, quantization, dequantization, and rendering taking 0.22, 0.84, 0.32, and 0.18 seconds,
respectively." (p6). Table 1 average "under 30 seconds" (p5). Per scene, desktop (Table 3, p8): 13.52 s (Train) to
37.75 s (Bicycle). Mobile Jetson Xavier: 59.42 s (Train) to 135.49 s (Bicycle). No iterations, no fine-tuning.

### 3.5 What "size" means

"File size is in MiB" (Table 1, p5). "The compression ratio is calculated as the file size of the 3D-GS models divided
by the uncompressed baseline." (p5). The pipeline stores pruned Gaussians with INT4/INT8 channels and per-sub-group
quantization ranges. No entropy coding, zip or codebook is described anywhere in the paper. Whether headers or
sub-group ranges are counted: not stated. No MLP or hash grid exists.

### 3.6 Runtime cost reported

Encode time: as in 3.4, Table 1 25.69 / 18.24 / 25.65 s average per dataset on an RTX 3090 (p5). Render: "The rendering
time reflects rendering 32 views at approximately 178 FPS." (p6, Truck, RTX 3090, during FOA evaluation). Decode time:
not stated. GPU memory: compression-time peak memory on an A100 for Grendel-GS models, Table 2 (p7): 3.89 / 6.75 / 12.39
/ 21.04 GiB for 3.04 / 6.16 / 9.36 / 11.84 million Gaussians, with FlexGaussian time 20.08 / 38.64 / 64.71 / 104.31 s.
GPU memory at render time: not stated. Hardware: Intel Core i9-10900K, 64 GB DRAM, RTX 3090 24 GB (p5), Nvidia Jetson
Xavier with 16 GB shared memory and a 512-core Volta GPU (p5), 4 × A100 40 GB for training large models (p5).

### 3.7 Size estimator or rate model

None. Every candidate is actually pruned and quantized and its quality rendered (p5). The search relies on an empirical
regularity: "although the quality drop varies, the linear correlation between quality loss and pruning ratio remains
consistent across scenes. Therefore, the Pareto-optimal frontier shares the same set of parameter pairs, narrowing down
the optimal parameter configurations to a limited set of candidates." (p4) and "we exploit the fact that the
Pareto-optimal frontier is convex, so that the priority trend at each direction is monotonious." (p4). No estimated
versus actual size gap is reported.

### 3.8 Scene dependence

Table 3 (p8), size in MiB at the <1 dB constraint: Bicycle 70.97, Bonsai 24.56, Counter 19.82, Flowers 31.20, Garden
89.80, Kitchen 36.70, Room 22.03, Stump 57.99, Treehill 23.98, Train 10.73, Truck 21.91, Drjohnson 29.19, Playroom
22.02 (3D-GS originals 1450.28, 294.42, 289.24, 860.06, 1379.99, 438.10, 376.85, 1173.52, 894.90, 242.78, 601.03,
805.36, 602.19 MiB). These are per-scene FOA outcomes, not a fixed hyperparameter set. Retuning statements: "we observe
minimal variation in channel sensitivity to bit-widths across different scenes, as certain Gaussian attributes (e.g.,
positions) are more prone to inaccuracies than others. Therefore, we use the same set of channel-specific bit-widths but
can support the search procedure with minimal cost (several seconds) when necessary." (p3). "While identifying a single
configuration for ADP and MPQ that generalizes across scenes is challenging, we find that the quality impact of MPQ is
scene-insensitive, mainly due to its scene-agnostic operation, while the primary sensitivity arises from the importance
score calculation for ADP." (p4).

### 3.9 Main quantitative results (as reported, Table 1 p5, RTX 3090, size in MiB, time in s)

| Method | Mip-NeRF360 PSNR / SSIM / LPIPS / Size / Time | Tanks&Temples | Deep Blending |
|---|---|---|---|
| 3D-GS | 27.21 / 0.815 / 0.214 / 795.26 / 1576.25 | 23.14 / 0.841 / 0.183 / 421.91 / 963.87 | 29.41 / 0.903 / 0.243 / 703.77 / 1654.11 |
| Compressed3D | 27.02 / 0.803 / 0.236 / 28.81 / 272.84 | 23.34 / 0.836 / 0.192 / 17.25 / 191.45 | 29.44 / 0.902 / 0.250 / 25.28 / 247.98 |
| CompGS | 26.99 / 0.801 / 0.250 / 21.08 / 2451.07 | 23.21 / 0.837 / 0.201 / 14.20 / 1397.70 | 29.98 / 0.911 / 0.250 / 15.15 / 1941.88 |
| FCGS-Opt | 27.04 / 0.799 / 0.231 / 62.34 / 53.57 | 23.36 / 0.839 / 0.186 / 31.12 / 30.09 | 29.61 / 0.902 / 0.243 / 55.41 / 42.61 |
| FlexGaussian | 26.38 / 0.780 / 0.251 / 40.80 / 25.69 | 22.44 / 0.804 / 0.219 / 16.30 / 18.24 | 28.61 / 0.884 / 0.269 / 25.48 / 25.65 |

Other rows: LightGaussian 26.96 / 0.800 / 0.244 / 52.05 / 917.61, PUP 3DGS 26.66 / 0.789 / 0.267 / 79.53 / 427.99 on
Mip-NeRF360. Time for retraining methods is total training time, for refinement methods the post-processing time (p5).
FlexGaussian is larger and lower in quality than Compressed3D and CompGS at every dataset, its claim is speed.

### 3.10 Densification and count control

No densification (post-hoc). Count set by ADP: Gaussians sorted by the LightGaussian global significance score (Eq. 1,
p4), "ADP retains the top α% of Gaussians and discards the bottom β% that contribute minimally to quality ... ADP
partially retains the remaining Gaussians (1 - α% - β%), preserving only critical attributes without re-training." (p4).
The (Row-P, SH-P) pair is chosen by FOA under the quality or ratio constraint. Importance scores from MesonGS also work
(Table 7, p9).

### 3.11 Stated limitations and future work (verbatim)

- p3: "While this approach does not explicitly consider the correlation between channels, we find that it is robust and
  efficient to select less sensitive columns."
- p6: "This comparison might seem unfair to retraining-based methods, as all others start with pre-trained models.
  However, we note that FlexGaussian remains faster than all others, even when including the time required to train the
  3D-GS models (3D-GS + FlexGaussian)."
- p7: "Note that the times reported include the entire compression pipeline, with data I/O being a non-trivial part of
  our method but less significant for others."
- p9: "The performance gap indicates that more accurate importance estimation can improve both compression ratio and
  quality."
- p9: "For future work, we plan to explore further pushing the compression ratio by investigating ternary or even binary
  attributes as well as achieving even faster adaption speed via advanced online search algorithms."

### 3.12 Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, or budget (verbatim)

- p1: "Existing compression methods effectively reduce 3D Gaussian parameters but often require extensive retraining or
  fine-tuning, lacking flexibility under varying compression constraints."
- p2: "FlexGaussian exposes a broad optimization space to explore and identify the optimal compression rate within
  hardware-defined constraints. To navigate this space, we introduce Fast Online Adaption (FOA), a lightweight yet
  effective adaptation module that jointly considers ADP and MPQ to identify the optimal parameter combinations for
  meeting hardware-specified constraints, such as target quality or compression ratio."
- p2: "It also adapts rapidly to varying memory or bandwidth constraints, with each adjustment taking just 1 to 2
  seconds, making it a flexible and cost-effective solution for Gaussian compression."
- p2: "Slimmable scene representations. Training a single model that allows for trade-offs between task quality and
  model size at runtime is crucial, especially given the varying resources available."
- p3: "Our goal is to obtain a matching compressed 3D-GS model, where its rendering quality is no less than ε
  (compression tolerance regime) in comparison with the uncompressed model."
- p3: "FlexGaussian is flexible in compressing pre-trained 3D Gaussian models at negligible cost, to either achieve a
  target compression ratio or to meet a quality target."
- p4: "Given a user-specified target quality or compression ratio, FOA searches in the candidate sets until reaches the
  best compression performance in observance to the constraints."
- p5: "In our main experiments, we target a <1 dB PSNR drop and evaluate adaptability separately."
- p6 to p7: "Fast adaptation to varying compression trade-offs is crucial for streaming Gaussians over fluctuating
  networks and on devices with diverse capabilities."
- p7: "In contrast, most existing methods [7, 10, 26, 27] are limited to fixed design points and struggle to adapt to
  varying compression needs, while methods like RDO-Gaussian [32] and MesonGS [37] require retraining or refinement to
  switch between design points."

None of these binds a target during training. The targets are post-hoc, on a frozen model, over a discrete candidate set.

### 3.13 Code

"The code is being prepared and will be released soon at: https://github.com/Supercomputing-System-AI-Lab/FlexGaussian"
(p1). Licence: not stated.

---

## 4. KISS-GS

### 4.1 Citation

"KISS-GS: 3D Gaussian Splatting Compression Kept Simple". First author Wieland Morgenstern (Fraunhofer HHI and
Humboldt-Universität zu Berlin), with Branschke, Fleischmann, Szatmari, Schlack, Barthel, Eisert, Hilsmann. Venue: the
PDF is in ECCV/LNCS format but prints no venue name. The file name says ECCV 2026. arXiv 2608.26948 (v1, 27 Aug 2026).
35 PDF pages (18 main including references, 17 appendix pages printed 1 to 17).

### 4.2 Base representation

Raw 3DGS Gaussians (means, quaternion + scales, SH degree 3, scalar opacity). "Importantly, this parameterization relies
on no MLPs, hash grids, or anchor structures, keeping the decoder simple." (p6). Rendering requires decoding the SOG-XT
container: WebP image decoding, per-channel min-max dequantization, signed-log inverse for means (16-bit split into two
8-bit planes), exp for scales, sigmoid for opacities, a 256 × 256 SH codebook lookup via UV indices (p9 to p10, decoder
listing p25 to p28). No MLP, no hash grid, no arithmetic decoding.

### 4.3 Budget handle

The handle is the primitive count (a TARGET on count, not bytes): "We control the rate-distortion tradeoff primarily
through primitive count, which has the additional benefit of reducing rendering cost and the work required for encoding
and decoding." (p5). POPSpa targets κ in the GaussianSpa ℓ0 constraint ‖a‖₀ ≤ κ (Eq. 2, p7) and prunes "a fixed fraction
with the lowest scores" (p7). Budgets used: 64k, 128k, 256k, 512k, 1024k, 2048k, 3000k (Fig. 4, p19), "For the smallest
model, we set the target to 64 k Gaussians for all datasets, except for Synthetic NeRF, where the lower scene complexity
motivates a target of 16 k." (p12). The count target is binding by construction (a fixed fraction is pruned, then
sparsified to κ), for example Table 1 (p12) reports 156 / 272 / 246 / 25 k Gaussians after removing 90 % from INRIA
checkpoints of 1,563 / 2,723 / 2,463 / 259 k. A secondary knob is the SH codebook size K = N/8: "the SH codebook size is
the clearest rate-control knob" (p14). No byte target is ever requested. Sizes in bytes are outcomes, and RD points are
compared by interpolation: "each entry is the interpolated file-size reduction at which a method matches the INRIA 3DGS
40k quality reference" (p14).

### 4.4 When it binds

Post-hoc, in three stages after reconstruction, with an optional fourth (Fig. 2, p5): (1) reconstruction, gsplat MCMC 30k
iterations (p6); (2) compaction POPSpa, post-training, 10k iterations: GaussianPOP scoring and pruning, 5k
optimize-sparsify iterations with effective-rank regularization, second pruning, 5k fine-tuning iterations (p7);
(3) SOG-XT encoding, no training, no renderer or cameras needed (p10); (4) optional encoding-aware fine-tuning, 4000
steps at batch size 8 with STE, "global step offset of 40 k" (p20). Wall clock on Bicycle (16 AMD EPYC 7200 CPU cores and
one A100, p19): at 512k primitives reconstruction 1215 s, POPSpa 277 s, SOG-XT encoding 34 s, fine-tuning 441 s (Fig. 4
caption). Across 64k / 128k / 256k / 512k / 1024k / 2048k / 3000k: training 992 / 1023 / 1095 / 1215 / 1268 / 1268 /
1268 s, POPSpa 187 / 203 / 229 / 277 / 368 / 476 / 502 s, encoding 8 / 13 / 15 / 34 / 81 / 279 / 402 s, fine-tuning
444 / 446 / 448 / 441 / 503 / 652 / 863 s (Fig. 4, p19). The size is never a training input, in any stage.

### 4.5 What "size" means

On-disk bytes of the SOG-XT container: 8-bit WebP images per attribute plus a YAML metadata file with float32 min-max
ranges ("all metadata including these ranges accounts for only 0.2% of the total size on disk", p9). Fig. 3 example
(p9): 4.12 MB total for 256,000 active Gaussians padded to 262,144, "129 bits/Gaussian", with means 20.6 %, rotations
and scales 33.3 %, colors 41.1 %, opacities and mask 4.9 %, metadata 0.2 %. No MLP, no hash grid, no codebook beyond the
SH centroid image (721.4 kB in the example). Baselines are raw .ply bytes (INRIA 40k: 738 MB on Mip-NeRF 360, p2) and
HAC++ file sizes recomputed by the authors.

### 4.6 Runtime cost reported

Encode: see 4.4. Decode: "On a Threadripper PRO 5955WX CPU, our end-to-end decoder takes 108 ms for 256k Gaussians, 192
ms for 512k, and 347 ms for 1024k. These timings include file reads but exclude PLY writing. Peak memory usage is 0.37
GB, 0.67 GB, and 1.24 GB, respectively." (p20). Render FPS: not stated. GPU memory at render time: not stated (only the
statement that the decoded footprint "is small enough to fit comfortably in GPU memory for real-time rendering", p20).

### 4.7 Size estimator or rate model

None. "Since storage scales approximately linearly with the primitive count, compaction is essential to achieve high
compression ratios." (p6). The SH codebook side length is a deterministic function of N: S from √(N/8), clamped to
[8, 256], rounded down to a multiple of 8 (p10). Rate control during fine-tuning is only through smoothness losses (TV
weights 1e−2 for quaternions, 1e−4 otherwise, p20). No estimated-versus-actual gap is reported.

### 4.8 Scene dependence

Per-scene sizes on Bicycle only: Table 7 (p35), POPSpa + SOG-XT-FT at 128k / 256k / 512k / 1024k / 2048k primitives:
1.9 / 3.7 / 7.4 / 13.5 / 25.0 MB, against INRIA 40k 1334.9 MB, HAC++ low-rate 13.8 MB, HAC++ high-rate 33.5 MB. Table 2
(p13): Bicycle 256k, SOG-XT 3.878 MB (3,877,944 bytes). Fig. 3 example 4.12 MB at 256,000 Gaussians. Dataset averages at
INRIA-Q (Figs. 1, 8, 9, 10, 11): Mip-NeRF 360 SOG-XT 7.13 MB, with FT 3.23 MB, POPSpa .ply 47.1 MB, INRIA 40k 738 MB
(p2); Tanks and Temples 1.83 / 1.3 / 18.5 / 414 MB (p31); Deep Blending FT 8.01 MB, POPSpa 47.6 MB, INRIA 684 MB (p33);
Synthetic NeRF 7.38 / 3.54 / 19.5 / 74.7 MB (p34). Retuning: "With a single parameter choice, as presented in Sec. 3.3,
our method works well across a large range of primitive counts, from several thousand to millions." (p13). "The SH
codebook size is the clearest optional rate-control knob, but its best setting is scene dependent. We therefore use
K = N/8 as a conservative default and use POPSpa primitive count as the primary rate-distortion control." (p21).

### 4.9 Main quantitative results (as reported)

There is no table with PSNR/SSIM/LPIPS/MB for the compressed method. Results are RD curves plus two tables.

Table 3 (p14), file-size reduction factor over INRIA 3DGS 40k at the point where the method matches INRIA-Q, per metric
(PSNR / SSIM / LPIPS):

| Method | Tanks and Temples | Mip-NeRF 360 | Deep Blending | Synthetic NeRF |
|---|---|---|---|---|
| HAC++ (recomputed, 44k) | 77* / 68 / - | 70 / 36 / - | 156 / 156 / - | - / - / - |
| HEMGS† | 69* / 69* / - | 59* / 59* / - | 229* / 229* / - | 57* / - / - |
| ContextGS† | 42* / 42* / - | 55* / 55* / - | 187* / 187* / - | n/a |
| Ours (SOG-XT) | 227 / 104 / 60 | 104 / 73 / 40 | - / - / - | 10 / - / - |
| Ours+FT | 319 / 131 / 72 | 228 / 109 / 64 | 85 / 98 / - | 21 / 14 / 14 |

"*" = smallest available point already reaches INRIA-Q (lower bound), "-" = INRIA-Q not reached, "†" = self-reported
from 3DGS.zip. INRIA-Q values: Mip-NeRF 360 27.20 dB, Tanks and Temples 23.52 dB, Deep Blending 29.76 dB, Synthetic NeRF
33.50 dB (Figs. 1, 8, 10, 11).

Table 1 (p12), compaction only (k Gauss, no MB), post-training on an INRIA 30k checkpoint with 90 % removed: INRIA 30k
Mip-NeRF 360 27.28 / .807 / .222 / 2,723 k; + Ours (+10k) 26.78 / .781 / .289 / 272 k; + Speedy-Splat 26.87 / .772 /
.305 / 272 k; + PUP 3D-GS 25.58 / .767 / .300 / 272 k. Compaction-aware block: gsplat-MCMC (30k) 27.95 / .829 / .196 /
2,560 k; + Ours (+10k) 27.79 / .811 / .239 / 512 k; GaussianSpa (40k) 27.40 / .817 / .227 / 561 k; Mini-Splatting2
27.33 / .815 / .223 / 619 k; Taming 3DGS 27.33 / .795 / .258 / 684 k. Tanks and Temples: gsplat-MCMC 24.54 / .869 /
.150 / 1,280 k, + Ours 24.47 / .856 / .179 / 256 k. Deep Blending: gsplat-MCMC 30.08 / .907 / .244 / 1,280 k, + Ours
29.94 / .904 / .262 / 256 k.

Table 5 (p23), HAC++ on Mip-NeRF 360, low-rate / high-rate: reported 30k 27.60 / 0.803 / 0.253 / 8.34 MB and 27.82 /
0.811 / 0.231 / 18.48 MB; recomputed 44k under the full protocol 27.09 / 0.795 / 0.261 / 8.39 MB and 27.61 / 0.806 /
0.236 / 19.62 MB.

### 4.10 Densification and count control

Reconstruction uses gsplat-MCMC with a cap: initial models of 320k primitives (80k Synthetic NeRF) for the 64k target,
doubling per RD point, "capped by a dataset-specific maximum of 3000 k Gaussians for Tanks and Temples and Mip-NeRF 360,
2048 k for Deep Blending, and 640 k for Synthetic NeRF" (p12), "an intended pruning ratio of approximately 80%" (p12).
POPSpa then prunes by the GaussianPOP error ΔSE_k = ‖C_render − C'_render‖² accumulated over training views (p7), runs
GaussianSpa optimize-sparsify with budget κ and effective-rank regularization Eq. (1) (p6 to p7), prunes again and
fine-tunes. The size objective never touches densification. Rationale: "Using more primitives with heavier compression to
reach the same operating point would instead increase all three, with no gain in rate or quality." (p5).

### 4.11 Stated limitations and future work (verbatim)

- p15: "With respect to simplicity, we fell short in one place: the encoder is a sophisticated modular pipeline with
  modules such as POPSpa and novel components such as PRAS."
- p15: "Deep Blending remains the main boundary, where anchor-based learned representations better absorb capture
  inconsistencies than the standard-splat reconstructions used as input to KISS-GS."
- p29: "We therefore view Deep Blending as a limitation of the standard-splat reconstruction used as input to KISS-GS
  under these captures, rather than as a limitation of the image-based encoder alone."
- p20: "For larger models, SH codebook clustering starts to dominate and is the clearest target for future encoder
  optimization."
- p20: "These measurements are intended as a conservative baseline because image decoding could be parallelized and
  handed off to hardware decoders, while per-field processing could be done more efficiently on the GPU."
- p8: "This 2D structure additionally enables fine-tuning to refine the codebook and provides a natural foundation for
  future vector quantization methods that exploit spatial coherence more aggressively."
- p15: "Its modular design makes future advances in reconstruction, compaction, and vector quantization directly useful
  for compression."
- p21: "The SH codebook size is the clearest optional rate-control knob, but its best setting is scene dependent."

### 4.12 Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, or budget (verbatim)

- p4: "Various criteria have been proposed, including strict primitive budgets via score-based sampling [25]"
- p4: "For example, MCMC-GS [17] removes low-opacity primitives and relocates them to high-opacity regions to improve
  reconstruction under a fixed primitive budget."
- p4: "RDO-GS [41] learns binary pruning masks jointly with the reconstruction objective using sparsity-inducing
  regularization, whereas GaussianSpa [47] introduces an auxiliary sparsity variable and uses an augmented-Lagrangian
  formulation that alternates between standard reconstruction optimization and a projection step that enforces a target
  sparsity level."
- p4: "Adaptation modifies scene parameters to make them more compatible with the target encoding. This can be achieved
  either during training or through post-training fine-tuning. RDO-GS [41] adds VQ-loss and rate-loss to the objective
  function, FCGS [4] incorporates entropy loss accounting for bit count, while HAC [3] and HAC++ [5] combine rendering
  fidelity, entropy, masking, and hash-grid size losses."
- p5: "We control the rate-distortion tradeoff primarily through primitive count, which has the additional benefit of
  reducing rendering cost and the work required for encoding and decoding. Using more primitives with heavier
  compression to reach the same operating point would instead increase all three, with no gain in rate or quality."
- p5 (Fig. 2 caption): "A standard 3DGS reconstruction yields a .ply, with primitive count setting the rate-distortion
  point."
- p6: "Since storage scales approximately linearly with the primitive count, compaction is essential to achieve high
  compression ratios."
- p7: "where Θ denotes all primitive parameters except opacity a, and κ is the primitive budget."
- p12: "The compaction-aware comparison is more nuanced: since most methods cannot easily predetermine the final scene
  size or quality, the compared samples sit at slightly different positions on each method's quality-size trade-off
  curve."
- p12: "For the smallest model, we set the target to 64 k Gaussians for all datasets, except for Synthetic NeRF, where
  the lower scene complexity motivates a target of 16 k."
- p14: "the SH codebook size is the clearest rate-control knob, while additional fine-tuning steps and stronger TV
  regularization do not improve the trade-off."
- p19: "In deployment, choosing between post-hoc and fine-tuned encoding is a trade-off between encoding time and
  bitrate, with the latter being more expensive but yielding better compression."
- p21: "The SH codebook size is the clearest optional rate-control knob, but its best setting is scene dependent. We
  therefore use K = N/8 as a conservative default and use POPSpa primitive count as the primary rate-distortion control."
- p23 (HAC++ loss, Eq. 3): "where λ controls the rate-distortion trade-off. At high λ (low-rate), extended training
  forces further compression without recovering quality."
- p24: "The evaluation thus helps ensure that bitrate reductions are not achieved by compressing away the generalization
  capacity."

Count budgets during training exist in the cited work (Taming 3DGS, MCMC, GaussianSpa). No byte target during training
appears anywhere.

### 4.13 Code

"Code and project page: https://fraunhoferhhi.github.io/KISS-GS/" (p1). Licence: not stated (the paper mentions licence
only as a property of reconstruction implementations, p6).

### 4.14 Additional questions for KISS-GS

Argument for decoupling compression from training. Two arguments, attribution and portability, plus a decoder-first
stance. Verbatim: "Many methods combine multiple strategies to achieve their final rate-distortion scores. While
effective, this obscures where gains actually originate, creating an attribution gap. Without clear attribution, it is
difficult to compare methods fairly and reuse components across pipelines." (p2). "Some approaches, such as anchor-based
families of methods, restructure the representation so fundamentally that compression becomes inseparable from the
reconstruction regime and data format. This training-format coupling effectively locks the codec to a particular 3DGS
training pipeline: when new reconstruction methods or parameterizations emerge, their improvements cannot be adopted
without reworking the compression scheme itself." (p2 to p3). "This is a deliberate design choice: by decoupling
compression from reconstruction, we avoid training-format coupling and ensure that improvements in reconstruction
quality can be adopted without changes to the compression pipeline." (p5). "restrict ourselves to a deliberately simple,
browser-compatible decoding pipeline backed by widely supported image codecs" (p3). Note that KISS-GS is not
training-free: its best numbers use 10k compaction iterations and 4k encoding-aware fine-tuning iterations, and the
paper frames this as "adaptation" that is kept as a separate, optional, decoder-preserving stage (p2, p10).

What it reports that a training-time method could not match (quotes). Compression ratio at INRIA quality on the two
real-world datasets where HAC++ is the training-time reference: "With 319× size reduction on Tanks and Temples in PSNR
and 228× on Mip-NeRF 360, it sets a new benchmark for 3DGS compression on these two real-world datasets." (p15). "On
Tanks and Temples and Mip-NeRF 360, KISS-GS reaches INRIA-Q at substantially higher reduction factors than HAC++ across
all reported metrics." (p14). "A pipeline built around a simple image-based encoding format can exceed the strongest
published baselines on the main real-world benchmarks while keeping deployment simple." (p15). Even the training-free
SOG-XT stage beats HAC++ on those datasets: Table 3 SOG-XT 227 / 104 versus HAC++ 77* / 70 (PSNR, Tanks and Temples /
Mip-NeRF 360). Deployment properties: "Decoding relies solely on web-native image formats" (p1), "The encoder does not
require camera parameters or a renderer." (p10), "A reference Python decoder reconstructs models with up to 1024k
Gaussians in well under a second on CPU, supporting our claim that deployment remains simple." (p15). Counterpoint the
paper itself gives: on Deep Blending the training-time method wins, "compressed HAC++ exceeds uncompressed gsplat-MCMC
and INRIA 3DGS on Playroom (30.63 vs. 30.42 and 30.09) and Dr. Johnson (29.60 vs. 29.48 and 29.43). This points to
HAC++'s anchor-based learned representation and regularization rather than to SOG-XT quantization or POPSpa pruning."
(p14). Also "Notably, many competing methods from the literature do not reach INRIA quality in the perceptual LPIPS
metric at any of their available operating points." (p15).

---

## 5. MesonGS

### 5.1 Citation

"MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation". First author Shuzhao Xie
(SIGS & TBSI, Tsinghua University), with Weixiang Zhang, Chen Tang, Yunpeng Bai, Rongwei Lu, Shijia Ge, Zhi Wang.
Venue: LNCS/ECCV format, no venue name printed in the PDF. The file name says ECCV 2024 and SizeGS cites it as
"In European Conference on Computer Vision. Springer." (SizeGS p10). arXiv 2409.09756 (v1, 15 Sep 2024). 18 pages,
supplementary not included.

### 5.2 Base representation

Raw 3DGS Gaussians trained for 30,000 iterations (p10). Rendering requires decoding: octree decode of positions, Euler
angle to rotation matrix (Eq. 5, p7), inverse RAHT of important attributes (opacity, scales, Euler angles, 0-degree SH),
block dequantization (Eq. 7 and 8, p8 to p9), codebook lookup for SH degrees 1 to 3, all after LZ77 decompression (p9).
No MLP, no hash grid. Entropy decoding: LZ77 only.

### 5.3 Budget handle

None. The user sets KNOBs: importance threshold τ (fraction of sorted Gaussians cut, 66 % or 50 % in Table 5, p14), bit
width b (8 by default, 16 in some experiments, p10), octree depth d ("We use the depth d to control the size of the
octree", p7), β in the view-independent score (p6), block size, codebook size. No target is requested and no
achieved-versus-requested size exists. Sizes are outcomes.

### 5.4 When it binds

Post-hoc on a 30k-iteration checkpoint: "To obtain the pre-trained 3D Gaussians for compression, we train 30, 000
iterations and then save the checkpoints for both datasets." (p10). Pipeline (Fig. 1, p6): prune → octree → quaternion
to Euler → RAHT → block quantization → VQ of 1+D SH → LZ77. Encoding time (Table 4, p12): MesonGS 4 s on NeRF-Synthetic
and 1 min on MipNeRF-360, versus C3DGS 30 s / 5 min and Lee et al. 480 s / 33 min. "When the number of Gaussian points
is less than 20,000, MesonGS can complete compression quickly using only the CPU" (p12). Optional finetune (Our-FT):
"we fix the coordinates of pruned 3D Gaussians and only finetune the attributes. We simulate the encoding and decoding
processes during the forward process. To pass the gradients during the backward process, we employ the straight-through
estimator [3] for quantization." (p9). Finetune iteration count and time: not stated in the main PDF. Hardware: not
stated.

### 5.5 What "size" means

On-disk bytes after LZ77 and a further zip: "All of the model sizes are calculated after a standard zip compression."
(p10). Included: "(1) Octree; (2) DC coefficients and Quantized coefficients; (3) Codebook and the corresponding mapping
table; (4) Metadata: Min-Max values of each block of quantized coefficients, octree depth, block size." (p9). Composition
(p12): Synthetic-NeRF octree 43 %, metadata 0.04 %, important attributes 34 %, unimportant attributes 23 %; Mip-NeRF 360
39 % / 0.02 % / 47 % / 14 %. No MLP, no hash grid.

### 5.6 Runtime cost reported

Encode time: Table 4 above (hardware not named). Decode time: listed as a metric on p10 but no number appears in the main
PDF. Render FPS: not stated. GPU memory: not stated.

### 5.7 Size estimator or rate model

None. No rate term, no estimator. The paper notes only the empirical bottleneck: "the quantization bit-width is the
bottleneck of overall performance here" (p14).

### 5.8 Scene dependence

No per-scene size table in the main PDF. Single-scene numbers in figures: Fig. 8 (p13, one Synthetic-NeRF scene, 16-bit):
rotations 1.48 MB, Euler angles 1.40 MB, R+ Scales 1.20 MB, R+ Euler angles 1.14 MB, R+ Opacity and 0-SHs 1.04 MB. Fig. 7
(p13): 3.5M / 3.3M / 3.4M / 3.6M / 3.8M / 5.5M / 5.3M for λ_c variants of the covariance replacement. Dataset averages at
fixed knobs in Table 5 (p14): channel-wise 66 % → 1.14 / 11.64 MB, 50 % → 1.59 / 16.47 MB, block 66 % → 1.21 / 12.46 MB,
50 % → 1.73 / 18.42 MB (Synthetic-NeRF / Mip-NeRF 360). Retuning statement: "pruning 33% of points resulted in lower
performance compared to pruning 66% of points. The reason for this counterintuitive performance drop is that the bit
depth of the quantization step is too low, and pruning 33% of points in the 3D-GS model of unbounded scenes still leaves
too many points, resulting in excessive information loss after quantization." (p14). Also "Note that RAHT is not applied
to the scale vectors at the 8-bit quantization." (p5), a per-bit-width rule.

### 5.9 Main quantitative results (as reported, Table 1 p10, size in MB after zip)

| Method | Mip-NeRF 360 PSNR / SSIM / LPIPS / Size | Tank&Temples | Deep Blending |
|---|---|---|---|
| 3DGS | 28.98 / 0.865 / 0.193 / 641.70 | 23.36 / 0.838 / 0.187 / 421.90 | 29.56 / 0.898 / 0.250 / 703.77 |
| C3DGS | 28.49 / 0.858 / 0.205 / 27.82 | 23.32 / 0.832 / 0.194 / 17.28 | 29.38 / 0.898 / 0.238 / 25.30 |
| Lee et al. | 28.60 / 0.856 / 0.209 / 46.98 | 23.32 / 0.831 / 0.201 / 39.40 | 29.79 / 0.901 / 0.258 / 43.20 |
| Our | 27.70 / 0.838 / 0.224 / 27.62 | 22.85 / 0.822 / 0.208 / 16.99 | 29.08 / 0.895 / 0.260 / 24.76 |
| Our-FT | 28.61 / 0.856 / 0.206 / 27.62 | 23.32 / 0.837 / 0.193 / 16.99 | 29.51 / 0.901 / 0.251 / 24.76 |

Mip-NeRF 360 here excludes flowers and treehill (p9). Time: only Table 4 encoding times. Table 2 (p10), Synthetic-NeRF:
3DGS 33.37 / 68.55 MB, C3DGS 32.94 / 3.68, Lee et al. 33.33 / 8.61, Our 32.25 / 3.65, Our-FT 32.92 / 3.66, and at 1 MB
Our 29.37 / 1.03, Our-FT 31.75 / 1.03 versus VQRF 31.77 / 1.43 and ACRF 31.79 / 1.15.

### 5.10 Densification and count control

No densification (post-hoc). Count set by the threshold τ on the importance score I_g = I_d I_i with I_d from Eq. (3)
(p6) and I_i = (V_norm)^β, "we use an importance threshold τ to prune Gaussians, meaning we cut the percent of τ of the
sorted Gaussians" (p6), justified by "40% of the Gaussians contain over 80% of the importance" (p6). Voxelization merges
Gaussians sharing a voxel: "When multiple Gaussians exist within a voxel, we average the corresponding attributes for
deduplication." (p7). No size objective anywhere.

### 5.11 Stated limitations and future work (verbatim)

- p11: "The reason is two-fold. On the one hand, our finetune scheme cannot support updating the coordinates. If
  appropriate updates could be made to the coordinates, the overall quality of our method will be improved. On the other
  hand, the file sizes of baselines are larger."
- p14: "Therefore, the quantization bit-width is the bottleneck of overall performance here."
- p13: "Due to the activation function of the scale being an exponential function, it is more sensitive than other
  attributes."
- No future-work paragraph: the conclusion (p14) reads "In this paper, we propose an elaborated designed 3D Gaussians
  codec. We propose several key components, including the universal Gaussian pruning strategy, elaborated attribute
  transformation, and flexible block quantization. Extensive experiments demonstrate the superior performance of
  MesonGS."

### 5.12 Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, or budget (verbatim)

None found. The nearest sentences concern a storage regime, not a target: "Additionally, we have compared our work with
compressed NeRF to reveal the strengths and limitations between Grid-based NeRF and 3D Gaussians in the 1MB storage."
(p3), and "This approach prevents quantization from becoming the quality bottleneck and provides increased flexibility."
(p3, block quantization).

### 5.13 Code

Project page "https://shuzhaoxie.github.io/mesongs/" (p1). No repository URL printed. Licence: not stated.

---

## 6. Cross-paper reading for the gap question

Classification (handle / binds when / size meaning / render-time MLP):

- SizeGS: size in MB, TARGET, binding within 1 % in tables (18.17 vs 18.33 MB, 10.93 vs 11, 7.92 vs 8) / post-hoc search
  on a frozen model, about 1 min for 3DGS, then optional fine-tuning with frozen hyperparameters / on-disk bytes after
  G-PCC plus LZ77 or torchac including metadata / MLP yes for the Scaffold-GS base, no for the 3DGS base.
- GETA-3DGS: storage budget B in MB or count K or PSNR floor τ, B nominally a TARGET but implemented as a sparsity
  penalty, non-binding (B from 3 to 120 MB all land at 4.0 to 4.2 MB) / during training from scratch, 25k vanilla
  warm-up then 10k compression iterations to 35k / "MB after entropy coding" per the metrics paragraph but bytes per
  Gaussian equal b̄ × 59 / 8 and a "+33.6% rate-reduction headroom attainable by any entropy coder", so raw quantized
  bytes in practice / MLP no.
- FlexGaussian: target quality (<1 dB) or compression ratio over a discrete candidate set, TARGET, quality target
  binding (0.8 / 0.7 / 0.8 dB) / post-hoc, training-free, 18 to 38 s per scene on an RTX 3090 / MiB of the pruned
  INT4/INT8 file, no entropy coding described / MLP no.
- KISS-GS: primitive count, TARGET on count, binding by construction, no byte target, SH codebook size as secondary knob /
  post-hoc after 30k gsplat-MCMC: 10k compaction iterations, encoding without training, optional 4k encoding-aware
  fine-tuning / on-disk bytes of WebP images plus YAML metadata / MLP no.
- MesonGS: no handle, KNOBs τ, b, d, β, block and codebook size / post-hoc on a 30k checkpoint, optional attribute-only
  fine-tuning / bytes after LZ77 and zip including octree, DC coefficients, codebook, metadata / MLP no.

Facts that bear on "no published method binds a byte target during training":

1. GETA-3DGS is the only one of the five that takes a byte budget as a training input and it says, in its own words,
   that B is "a sparsity penalty rather than a hard byte cap", "non-binding", with every nominal target from 3 MB to
   120 MB landing at 4.0 to 4.2 MB, and it names the "Lagrangian dual of the byte-counter constraint" as future work. Its
   quality is 1.79 dB below its own vanilla and 4.24 dB below HAC++ on Mip-NeRF 360. It is a submitted manuscript
   (arXiv May 2026) with no printed code URL and internally inconsistent schedule and size definitions.
2. SizeGS states the failure mode of λ methods plainly: "it is difficult to predict the final size given a certain λ",
   and its HAC baseline needed a λ binary search with 10 to 20 min retraining per step, 16627 s total on Mip-NeRF 360,
   stopping at a 5 % band. That is the de facto training-time way to hit a byte target and its cost.
3. SizeGS fine-tunes after fixing τ and Q and reports "almost no impact on the file size during retraining", which is
   evidence that a byte target can be held during fine-tuning once the quantization structure is frozen, at the price of
   frozen coordinates and no densification.
4. KISS-GS observes that "most methods cannot easily predetermine the final scene size or quality" and controls rate only
   by primitive count, noting storage is "approximately linearly" proportional to count. Count budgets during training
   exist in the works it cites (Taming 3DGS strict primitive budgets, MCMC fixed primitive budget, GaussianSpa target
   sparsity via augmented Lagrangian). A byte budget is the missing object, and count-to-bytes is only approximately
   linear (KISS-GS 129 bits per Gaussian at 256k, GETA 63.7 bytes per Gaussian raw).
5. GETA-3DGS does bind a quality floor during training (PSNR on training views checked at every prune event, K relaxed
   by λ ∈ (1, 2]). A precedent for a hard constraint enforced by relaxing the count during training exists.
6. Render-time budgets (ms or FPS) are bound by none of the five. GETA reports 4.50 ms per view and 261 MB peak GPU on an
   A100 post hoc. FlexGaussian names "hardware-defined constraints" but only implements quality and ratio targets.

What contradicts or weakens the claimed gap:

- GETA-3DGS's abstract and conclusion claim that users can "directly specify storage and quality budgets" during an
  end-to-end training pipeline, so a literal "no published method takes a byte target during training" is false. The
  defensible statement is "no published method binds a byte target during training", and GETA-3DGS's own Figs. 2 and 6
  are the evidence.
- FlexGaussian reaches any ratio target in 1 to 2 s post hoc and SizeGS reaches a byte target within 1 % in about a minute
  post hoc, so a training-time method must justify itself by quality at the target, not by the ability to hit the target.
- KISS-GS argues that decoupling beats coupling and shows a training-free encoder beating HAC++ (a training-time RD
  method) on Tanks and Temples and Mip-NeRF 360 at INRIA quality, while conceding Deep Blending to the training-time
  anchor method. A training-time byte-budget method has to answer why its coupling is worth the "training-format
  coupling" cost that KISS-GS names.
