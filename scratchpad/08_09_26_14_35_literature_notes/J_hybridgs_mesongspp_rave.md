# Literature notes J: HybridGS, MesonGS++, RAVE

Read in full, including appendices and supplementary material. Every number below is **as
reported in the PDF** unless explicitly labelled "computed here". Fields 10 and 11 are verbatim
quotes with page numbers. "not stated" means the paper does not say it.

---

# 1. HybridGS

## 1. Citation

- Title: "HybridGS: High-Efficiency Gaussian Splatting Data Compression using Dual-Channel Sparse
  Representation and Point Cloud Encoder"
- First author: Qi Yang (University of Missouri-Kansas City). Co-authors Le Yang, Geert Van Der
  Auwera, Zhu Li.
- Venue as printed on p1: "Proceedings of the 41 st International Conference on Machine Learning,
  Vancouver, Canada. PMLR 267, 2025." (the volume number 267 and the ordinal 41st are inconsistent
  in the PDF itself, quoted as printed)
- arXiv id: 2505.01938, "arXiv:2505.01938v1 [cs.CV] 3 May 2025" (p1)

## 2. Base representation

Raw 3DGS Gaussians, not Scaffold-GS anchors. p2: "To limit the scope of this work and, more
importantly, gain interpretability when evaluating compression-related losses, HybridGS is based
on the original 3DGS implementation."

Modifications to the raw representation:
- Colour DC plus SH (48 channels) replaced by a learnable latent of k_c ∈ {3, 6} channels plus a
  one-hidden-layer MLP decoder (50 hidden units, ReLU). Rotation (4 channels) replaced by a latent
  of k_r = 2 channels plus its own MLP decoder. Opacity and scaling stay explicit (p4, p12).
- Positions are integers produced by a Learnable Quantizer-based Method (LQM), decomposed as the
  inner product of a fixed basis vector [2^(N−2), 2^(N−3), ..., 4, 2, 1] and a learnable coding
  vector with entries in {−1, 0, 1} (p5, p15).

Render-time requirements:
- **MLP: yes.** Two small MLP decoders must run before rendering to reconstruct colour and
  rotation. p7 gives the cost: "For 'bicycle', color t_deq and t_mlp are around 1s and 0.9s, while
  0.6s and 0.001s for rotation."
- **Hash grid: no.**
- **Entropy decoding at render time: no per-frame decoding.** GPCC decoding is a one-off step
  before rendering. Positions need no de-quantization at all: p2, "owing to the alignment between
  scene and 3DGS scaling, position de-quantization is not necessarily needed before rendering."
- De-quantization of the other attributes is needed once, using the stored meta data a and b of the
  Robust Quantizer (p4, Figure 3).

## 3. Budget handle

The user supplies **B, a bandwidth limitation in MB**, and it is a **TARGET**, not a knob. p6:
"Given the bandwidth limitation as B, the rate control can be formulated as the following
optimization problem: max_n Q(GS), s.t., R(GS) ≤ B and R(GS) = n · P_bit / L".

There are two separate controllers that both consume B. Method 1 controls the primitive count.
Method 2 controls the feature bit depth. They are not combined.

The paper also exposes pure knobs elsewhere: the latent widths k_c and k_r, the bit depth BD, and
the HR/LR checkpoint choice. Those are knobs, not targets.

### Every achieved-versus-requested pair in the paper

All from **Table 4, "Rate control of HybridGS", p8**. Column layout verified by rendering the page:
`Target Rate | Method 1 PSNR | Method 1 Real Rate | Method 2 PSNR | Method 2 Real Rate`.
Sizes in MB. Percentages are computed here from the reported pairs.

**train:**

| Target | M1 PSNR | M1 Real | M1 gap (computed) | M2 PSNR | M2 Real | M2 gap (computed) |
|---|---|---|---|---|---|---|
| 4 MB | 21.13 | 3.96 | −1.0 % | 18.20 | 2.20 | −45.0 % |
| 6 MB | 21.60 | 5.82 | −3.0 % | 21.54 | 5.15 | −14.2 % |
| 8 MB | 21.44 | 7.56 | −5.5 % | 21.75 | 7.33 | −8.4 % |
| 10 MB | 21.54 | 8.59 | −14.1 % | 21.75 | 9.86 | −1.4 % |

**dance:**

| Target | M1 PSNR | M1 Real | M1 gap (computed) | M2 PSNR | M2 Real | M2 gap (computed) |
|---|---|---|---|---|---|---|
| 0.5 MB | 37.40 | 0.54 | **+8.0 % (over target)** | * | * | not achievable |
| 1 MB | 38.99 | 1.06 | **+6.0 % (over target)** | 38.80 | 0.77 | −23.0 % |
| 1.5 MB | 39.76 | 1.59 | **+6.0 % (over target)** | 39.86 | 1.40 | −6.7 % |
| 2 MB | 39.90 | 2.07 | **+3.5 % (over target)** | 39.96 | 1.95 | −2.5 % |

Raw extracted rows, quoted because the columns interleave:

```
Target Rate PSNR Real Rate PSNR Real Rate
4 MB 21.13 3.96 18.20 2.20
6 MB 21.60 5.82 21.54 5.15
8 MB 21.44 7.56 21.75 7.33
10 MB 21.54 8.59 21.75 9.86
dance
0.5 MB 37.40 0.54 * *
1 MB 38.99 1.06 38.80 0.77
1.5 MB 39.76 1.59 39.86 1.40
2 MB 39.90 2.07 39.96 1.95
```

**Key correction to the working assumption.** HybridGS does not land uniformly under target. On
"dance" Method 1 **exceeds** the target at all four rate points. The observed spread across the
15 usable pairs is roughly −45 % to +8 % (computed here). The paper's own summary of Table 4 says
only that Method 1 approximates the target better than Method 2, never that it stays under. p8:
"1) in most cases, method 1 demonstrates a more precise approximation to the target bitrate
compared to method 2".

## 4. When it binds, in full detail

**Total training length:** 70 000 iterations, called "epochs" in the paper. p6: "Training epochs
are fixed at 70, 000."

**The full schedule (p5 Figure 4, p6 Model Parameters, p12 Appendix A.1):**

| iteration | event |
|---|---|
| 0 to 7 500 | vanilla 3DGS warm-up (t₁ = 7 500), then outlier removal, translation and rescaling to the bit-depth box, position rounding and decomposition into basis and coding vectors |
| 7 500 | uniqueness activated (interval 500) |
| up to T_d = 15 000 | densification active (interval 100). **Densification ends here.** |
| T_top | "top quality point", T_d < T_top < T_p. **No numeric value is ever given.** |
| T_p = 36 000 | pruning starts (interval 2 500). **This is where the size target binds.** |
| T_u = 66 000 | uniqueness ends, positions frozen, only other attributes updated afterwards |
| T = 70 000 | training ends |

Constraint stated on p5: "Therefore, we set T_d < T_top < T_p < T_u ≈ T."

**When the target is given.** The target is consumed only after T_top. p6: "For a given scene, we
do not know the vanilla data volume before T_top, which indicates that we need to change these
parameters to satisfy the bandwidth requirement after T_top." The bit depth BD is fixed before
training (BD = 16 for both position and attributes, p6), and quantization is inside the training
loop from the start, but the value of B enters nothing before T_top.

**Operations the controller performs.**
- Method 1: prune only. p6: "The number of primitives in T_top is n_top, the number of primitive
  need to be pruned is n_p = n_top − n. Based on the training epoch and the pruning interval I_p,
  the pruning is performed F_p = (T − T_p)/I_p times. Each time N′ = N_p/F_p primitives are pruned
  until the training ends." Pruning importance follows LightGaussian (Fan et al., 2024), p2.
- Method 2: lower the bit depth only. p6: "With disabling pruning, we gradually reduce BD_ini with
  step size 1 until Δ = 0." Savings are spread evenly over all attributes except position:
  "Δpbit = Δ · (k_c + 1 + 3 + k_r)".
- The controller never changes λ, never grows, never splits, never clones, and never widens a
  latent. Changing k_c or k_r is explicitly rejected. p6: "Changing feature channels during
  training results in unstable results and extremely expensive training costs".

**Can the primitive count INCREASE while the controller is active? No.** Densification ends at
T_d = 15 000, the controller starts at T_p = 36 000, and uniqueness (7 500 to 66 000) only removes
duplicates. Every operation after T_p is monotone shrinking. The only stated caveat is p8: "We
also find after T_p, although we allow optimize primitive position, LQM will not generate
duplicated primitive position at most cases because 3DGS is very sparse, so uniqueness will not
influence rate control methods proposed in Section 3.2.2."

**What stops the controller.** The iteration budget, not the target. Pruning proceeds a fixed
F_p = (T − T_p)/I_p times "until the training ends" (p6). Method 2 stops when Δ = 0, that is when
the bit depth has walked down to BD_tar. There is no feedback loop that measures the encoded size
and stops when B is met. Note the appendix separately states a fixed pruning amount that does not
reconcile with the 47 % / 75 % headline figures. p12: "For pruning, each time we remove 0.1%
primitives." vs p6: "We select the samples in the 50, 000 and 70, 000 epochs as High and Low (i.e.,
pruning 47% and 75% primitives) Rate (HR, LR) points of HybridGS."

## 5. The size model

Two-level, closed form, no learned component.

**Bits per primitive (Equation 7, p4):**

  P_bit = 3·(BD_p + BD_s) + k_c·BD_c + BD_o + k_r·BD_r

- BD_p: position bit depth, 3 channels
- BD_s: scaling bit depth, 3 channels
- BD_c: bit depth of one latent colour channel, k_c channels
- BD_o: opacity bit depth, 1 channel
- BD_r: bit depth of one latent rotation channel, k_r channels
- k_c ∈ {3, 6}, k_r = 2 in the experiments. All BD set to 16 by default (p6).

**Predicted bitstream (Equations 8 and 9, p6):**

  R(GS) = n · P_bit / L          (Method 1, n = target primitive count)
  R(GS) = N · (pbit − Δpbit) / L (Method 2, Δpbit = Δ·(k_c + 1 + 3 + k_r))

- n, N: primitive count
- L: lossless compression ratio of the downstream GPCC encoder

**Calibration of L.** A single constant measured offline. p8: "lossless compression ratio L of
GPCC in 3DGS takes values within the range of 1.3 to 1.5 based on preliminary experiments, we set
L = 1.3 here." A per-scene L is left as future work: p8, "An improvement of the proposed rate
control strategy is to first calculate the point density to obtain a more accurate estimation of L."
The model ignores metadata and MLP weights: p7, "The quantized metadata, as well as the feature
decoders, are very small and can be ignored during rate control."

**Differentiable? No.** The size model is arithmetic over the discrete primitive count and integer
bit depths. It never appears in the loss. The only gradient path in the whole method is the
rendering loss through the STE-backed quantizers.

**Stated error between predicted and actual encoded bytes.** No aggregate error figure is given.
The only evidence is Table 4 itself and the two failure explanations on p8: the 10 MB train case
("method 1 reports larger rate error for 10 MB of 'train' than other rates. The reason is that
different primitive densities will influence the lossless compression ratio.") and the dance
0.5 MB Method 2 case ("'*' in 'dance': method 2 means that we cannot achieve the target rate solely
by adapting BD. The primitive position rate is larger than the target rate, resulting in attribute
BDs reduced to 0").

## 6. What "size" means in the tables

**Entropy-coded (lossless-coded) bytes of the GPCC bitstream, plus meta data, plus MLP weights.**

- p6: "We use the second implementation in Section 3.2.1 with the 'xyza' mode of the GPCC test
  model v23 (WG7, 2023). The lossless octree and RAHT mode are applied".
- Table 3, p8, itemises "bicycle" at 22.88 MB total: position 2.72 MB (7.36 pre-GPCC), colour
  latent 6.39 MB (7.36), opacity 2.40 MB (2.45), scaling 7.07 MB (7.36), rotation latent 4.30 MB
  (4.91), metadata 4 KB, colour decoder weights 13 KB, rotation decoder weights 4 KB. Primitive
  count 1 286 284. Values in parentheses are the raw quantized size before GPCC.
- "dance" at 1.04 MB total: position 0.16 (0.32), colour 0.28 (0.32), opacity 0.11 (0.11), scaling
  0.31 (0.32), rotation 0.18 (0.22), metadata 4 KB, decoders 13 KB and 4 KB. 56 490 primitives.

So GPCC is lossless. All distortion comes from the training-time quantization and pruning, not from
the codec.

## 7. Runtime cost

Hardware, p12: "All the experiments are tested on Intel Core i9-14900HX, NVIDIA RTX 4090 Laptop."
Coding times are CPU, serial, data I/O excluded (p7).

**Table 2, "Coding time", p8, encode/decode seconds:**

| Method | bicycle | room |
|---|---|---|
| HAC | 85.03 / 80.09 | 17.04 / 15.90 |
| CompGS(MM) | 36.29 / 22.47 | 7.82 / 6.28 |
| HGSC HR | 132.32 / 72.13 | 35.65 / 16.45 |
| HGSC LR | 124.22 / 52.94 | 31.64 / 13.39 |
| HybridGS HR | 1.67 / 1.77 | 0.44 / 0.47 |
| HybridGS LR | 0.66 / 0.92 | 0.26 / 0.46 |
| - position | 0.09 / 0.08 | 0.03 / 0.04 |
| - attribute | 0.57 / 0.84 | 0.23 / 0.32 |

Pre-render preprocessing, p7: "For 'bicycle', color t_deq and t_mlp are around 1s and 0.9s, while
0.6s and 0.001s for rotation."

**Render FPS, Table 8, p18:** 3DGS-30K 154 / 137 / 134 on T&T / DB / MipNeRF360. HybridGS
k_c = 3 HR 207 / 201 / 199, LR 247 / 223 / 220. HybridGS k_c = 6 HR 195 / 191 / 189,
LR 214 / 212 / 210.

**GPU memory: not stated.** **Training time: not stated** beyond the 70 000-iteration count.

## 8. Main quantitative results

**Table 8, "Overall results", p18.** Verified against a rendered image of the page. PSNR and SIZE
in MB and FPS only. **SSIM and LPIPS are not reported anywhere in this paper.** This table has only
one baseline, 3DGS-30K.

| Method | T&T PSNR | T&T SIZE | T&T FPS | DB PSNR | DB SIZE | DB FPS | MipNeRF360 PSNR | MipNeRF360 SIZE | MipNeRF360 FPS |
|---|---|---|---|---|---|---|---|---|---|
| 3DGS-30K | 23.14 | 411.00 | 154 | 29.41 | 676.00 | 137 | 27.21 | 734.00 | 134 |
| HybridGS k_c=3, k_r=2 HR | 22.90 | 8.85 | 207 | 28.51 | 11.52 | 201 | 25.64 | 15.82 | 199 |
| HybridGS k_c=3, k_r=2 LR | 22.66 | 4.27 | 247 | 28.32 | 5.59 | 223 | 25.40 | 7.63 | 220 |
| HybridGS k_c=6, k_r=2 HR | 23.12 | 11.10 | 195 | 29.05 | 16.35 | 191 | 25.97 | 21.73 | 189 |
| HybridGS k_c=6, k_r=2 LR | 22.83 | 5.27 | 214 | 28.82 | 7.92 | 212 | 25.75 | 10.47 | 210 |

**Strongest two baselines** appear only in the per-scene Table 1, p7, on five hand-picked scenes,
with PSNR and SIZE (MB) only. Copied exactly:

| Method | playroom | train | bicycle | room | dance |
|---|---|---|---|---|---|
| 3DGS (anchor) | 30.03 / 550.67 | 21.89 / 256.73 | 24.49 / 1443.84 | 31.55 / 370.14 | 39.83 / 41.34 |
| HAC λ = 0.0005 | 30.84 / 6.86 | 22.73 / 12.26 | 25.00 / 44.07 | 31.89 / 8.23 | 39.20 / 0.45 |
| HAC λ = 0.004 | 30.63 / 3.95 | 22.53 / 7.98 | 24.81 / 26.99 | 31.44 / 5.54 | 36.89 / 0.27 |
| CompGS(MM) λ = 0.001 | 30.11 / 6.31 | 22.08 / 8.06 | 24.74 / 22.09 | 30.90 / 8.99 | 37.43 / 2.84 |
| CompGS(MM) λ = 0.005 | 28.98 / 4.89 | 21.78 / 6.21 | 24.43 / 14.62 | 30.35 / 7.16 | 35.35 / 2.69 |
| HybridGS k_c=3, k_r=2 HR | 29.89 / 12.15 | 21.26 / 4.19 | 24.08 / 22.88 | 29.52 / 6.43 | 39.25 / 1.04 |
| HybridGS k_c=3, k_r=2 LR | 29.49 / 5.88 | 20.96 / 2.03 | 23.53 / 11.01 | 29.23 / 3.14 | 37.65 / 0.51 |
| HybridGS k_c=6, k_r=2 HR | 29.89 / 16.08 | 21.49 / 5.63 | 24.10 / 30.21 | 29.75 / 8.28 | 39.31 / 1.32 |
| HybridGS k_c=6, k_r=2 LR | 29.68 / 7.79 | 21.04 / 2.72 | 23.76 / 14.52 | 29.61 / 4.00 | 37.78 / 0.64 |

Note the paper concedes HAC wins on rate-distortion. p7: "HybridGS does not outperform HAC in terms
of the RD curve, but it offers a notable improvement in encoding and decoding speed".

## 9. Densification and count control

Densification is vanilla 3DGS densification, running from the start to T_d = 15 000, interval 100
(p12). It is **not** informed by the size target. Count is then reduced by two independent
mechanisms:

1. **Uniqueness** (7 500 to 66 000, interval 500). Removes primitives sharing an integer voxel
   position. p12: "For the position that has more than one primitives, we keep the primitive that
   has the largest size based on scaling attributes, other primitives are removed directly (given a
   primitive with scale [Sx, Sy, Sz], the size are defined as V = Sx × Sy × Sz)."
2. **Pruning** from T_p = 36 000 to T = 70 000, interval 2 500, importance from LightGaussian.

The size target influences pruning **only**, and only after T_top. Densification is over by then.
The paper does observe that unbounded densification hurts. p16, Appendix A.6: "it reveals that for
3DGS, more primitives do not necessarily lead to better reconstruction quality. The primitive
densification method proposed in vanilla 3DGS generation can be further improved if the data size
is one of reference." That is a statement of future work, not something HybridGS implements.

## 10. Stated limitations and future work (verbatim)

p9, Section 6:

> "Limitations: The optimal compression efficiency of HybridGS is lower than end-to-end generation
> compression methods using RD loss (Liu et al., 2024) as supervision."

> "Future Work: HybridGS requires hyperparameters like latent feature dimension and BD. An adaptive
> parameter selection algorithm may help HybridGS realize a better quality and size tradeoff. For
> rate control, different from reducing the BD and primitive pruning, decreasing the latent feature
> dimension during 3DGS generation can lead to feature space collapse, as well as training a new
> decoder with extra time. How to achieve smooth latent feature dimension reduction without
> significantly affecting the rendering quality is thus a research topic worthy of investigation.
> Besides 2D rendering loss, explicit 3DGS generation might benefit from new loss functions in 3D
> space such as Chamfer distance to optimize primitive distribution to improve point cloud encoder
> efficiency (Yang et al., 2023)."

p1, abstract, a self-imposed scope limit:

> "At the current stage, HybridGS does not include any modules aimed at improving 3DGS quality
> during generation."

p2:

> "We deliberately do not incorporate any methods to optimize 3DGS generation and improve the
> reconstruction quality. As such, the upper bound for the reconstruction quality of HybridGS is
> the vanilla 3DGS."

## 11. Every sentence touching a size, byte, bitrate, memory or FPS TARGET during training, or "rate control", or "budget" (verbatim, with page)

p1: "A simple and effective rate control scheme is proposed to pivot the interpretable data
compression scheme."

p2: "The primitive pruning proposed in (Fan et al., 2024) is used to control the primitive number,
resulting in two simple but effective rate control strategies."

p2: "Two effective rate control methods are established based on the sparsification strategy."

p3: "Extensive experiments show that HybridGS can provide comparable reconstruction performance
against generative compression methods with greatly decreased encoding and decoding time, as well
as flexible rate control capability."

p3: "It has two parts: a dual-channel sparse representation module to generate explicit compact
3DGS, and a downstream point cloud encoder to realize encoding and rate control."

p4: "After quantization, we can calculate the number of bits per primitive."

p4: "The number of bits per primitive is thus P_bit = 3 · (BD_p + BD_s) + k_c · BD_c + BD_o + k_r ·
BD_r."

p5: "To avoid this, we propose a progressive integer primitive uniqueness and pruning training
strategy, which can reduce quality fluctuations and satisfy the rate control for random access."

p5: "Given the entire training time T, four time nodes are set during 3DGS training: the
densification end point T_d, the uniqueness end point T_u, the pruning start point T_p, and the top
quality point T_top."

p5: "Rate control is important for practical applications. However, with the current generative
compression methods, which use the rate-distortion (RD) optimization as the loss"

p6: "function, the final data rate is difficult to predict. On the contrary, the proposed HybridGS
is easier to realize accurate rate control."

p6: "The bit number of the explicit 3DGS generated in the first step is easy to calculate (see (7)),
since we know the primitive number and the assigned bits for each primitive."

p6: "For the downstream compression, using lossless compression as an example, the compression
ratio of the current point cloud encoder is relatively stable. For GPCC, the lossless compression
ratio is around 3-4× for dense point clouds, 2× for large-scale sparse point clouds. Therefore, we
can perform rate control by controlling the size of the explicit 3DGS with downstream compression
ratios."

p6: "Intuitively, there are three factors that can influence the size of the explicit 3DGS: feature
channels, feature BD, and primitive number. For a given scene, we do not know the vanilla data
volume before T_top, which indicates that we need to change these parameters to satisfy the
bandwidth requirement after T_top."

p6: "Changing feature channels during training results in unstable results and extremely expensive
training costs, while changing feature BD and primitive number are more promising for rate control.
Therefore, we propose two rate control methods here. We only consider lossless mode of the point
cloud encoder for simplicity."

p6: "Method 1 - Controlling Primitive Number: the progressive pruning proposed in Section 3.1.2 can
realize the soft primitive number control. Given the bandwidth limitation as B, the rate control
can be formulated as the following optimization problem: max_n Q(GS), s.t., R(GS) ≤ B and R(GS) =
n · P_bit / L, (8) where GS represents the explicit 3DGS samples, n represents the target primitive
number, Q(·) and R(·) represent the quality and bitstream of GS, and L is the lossless compression
ratio of the downstream encoder."

p6: "The number of primitives in T_top is n_top, the number of primitive need to be pruned is n_p =
n_top − n. Based on the training epoch and the pruning interval I_p, the pruning is performed F_p =
(T − T_p)/I_p times. Each time N′ = N_p/F_p primitives are pruned until the training ends."

p6: "Method 2 - Adapting Feature BD: reducing the feature BD can also lower the bitrate but this is
more complex theoretically."

p6: "The bandwidth savings are ideally derived from the feature that contributes the least to
distortion. Accurate modeling of the distortion sensitivity across different features in 3DGS is
currently absent. Hence, we focus on the most straightforward method: bandwidth savings are
distributed evenly among all features except for the primitive position as (9), where Δpbit is the
bits need to be reduced for each primitive, and Δ is the BD reduced for each features."

p6: "A progressive BD reduction is proposed to mitigate pronounced quality fluctuations: the
difference between target BD_tar and initial BD_ini is Δ = BD_ini − BD_tar. With disabling pruning,
we gradually reduce BD_ini with step size 1 until Δ = 0."

p6: "Over the course of pruning, the explicit compact 3DGS will has fewer primitives, resulting in a
lower bitrate. We select the samples in the 50, 000 and 70, 000 epochs as High and Low (i.e.,
pruning 47% and 75% primitives) Rate (HR, LR) points of HybridGS."

p7: "The quantized metadata, as well as the feature decoders, are very small and can be ignored
during rate control."

p7: "Two different methods for performing rate control are proposed in Section 3.2.2. Four rate
points are chosen for 'train' and 'dance' and the results are presented in Table 4."

p8: "The lossless compression ratio L of GPCC in 3DGS takes values within the range of 1.3 to 1.5
based on preliminary experiments, we set L = 1.3 here."

p8: "We see that: 1) in most cases, method 1 demonstrates a more precise approximation to the
target bitrate compared to method 2; 2) '*' in 'dance': method 2 means that we cannot achieve the
target rate solely by adapting BD. The primitive position rate is larger than the target rate,
resulting in attribute BDs reduced to 0; 3) method 1 reports larger rate error for 10 MB of 'train'
than other rates. The reason is that different primitive densities will influence the lossless
compression ratio."

p8: "An improvement of the proposed rate control strategy is to first calculate the point density
to obtain a more accurate estimation of L"

p8: "We also find after T_p, although we allow optimize primitive position, LQM will not generate
duplicated primitive position at most cases because 3DGS is very sparse, so uniqueness will not
influence rate control methods proposed in Section 3.2.2."

p8: "It indicates that selecting a proper BD can facilitate saving bandwidth."

p8 (Related Work, on CompGS(MM)): "They also formulated a rate-constrained optimization to balance
the quality and bitrate."

p9: "HybridGS reports comparable performance with SOTA methods and faster coding and decoding, as
well as demonstrating characteristics of interpretability, compatibility, and alignment with the
demands of standardization."

p9: "For rate control, different from reducing the BD and primitive pruning, decreasing the latent
feature dimension during 3DGS generation can lead to feature space collapse, as well as training a
new decoder with extra time."

p16: "The primitive densification method proposed in vanilla 3DGS generation can be further improved
if the data size is one of reference."

## 12. Code URL and licence

- p1, printed: `https://github.com/Qi-Yangsjtu/HybridGS`
- p17, demo videos:
  `https://drive.google.com/drive/folders/14KIzFDIPSPdrKpXjtFh1HYUtG-E0Zs5W?usp=sharing`
- Licence: **not stated** anywhere in the PDF.

## HybridGS additional questions

### (a) Is the target size B an input to the whole training run, or only to a post-convergence stage?

**Only to a post-convergence stage.** Verbatim, p6:

> "For a given scene, we do not know the vanilla data volume before T_top, which indicates that we
> need to change these parameters to satisfy the bandwidth requirement after T_top."

Supported by the schedule, p5: "Therefore, we set T_d < T_top < T_p < T_u ≈ T." Pruning starts at
T_p = 36 000 of 70 000 (p6). Note that this is not a fully converged model in the usual sense.
T_p sits inside the training run at 51 % of the schedule, and rendering optimization continues
alongside the shrink. So it is more accurate to call it a **post-peak-quality shrink phase inside a
single training run** than a separate post-hoc stage. The bit depth and the latent widths are still
fixed before iteration 0 and never respond to B.

### (b) Can the rate-control stage ever add primitives or bits, or only remove?

**Only remove.** Method 1, p6:

> "The number of primitives in T_top is n_top, the number of primitive need to be pruned is n_p =
> n_top − n."

Method 2, p6:

> "With disabling pruning, we gradually reduce BD_ini with step size 1 until Δ = 0."

And the schedule forecloses growth, p6: "T_d, T_p, and T_u are set as 15, 000, 36, 000, and 66, 000
epochs." Densification is finished 21 000 iterations before the controller starts.

### (c) Is the size compared against B entropy-coded or an estimate, and what is the measured gap?

**An estimate**, divided by a single hand-set constant. p6:

> "R(GS) = n · P_bit / L ... L is the lossless compression ratio of the downstream encoder."

p8:

> "lossless compression ratio L of GPCC in 3DGS takes values within the range of 1.3 to 1.5 based on
> preliminary experiments, we set L = 1.3 here."

The "Real Rate" column of Table 4 is the actual GPCC bitstream, so the gap is estimate versus
entropy-coded truth. Measured spread over the 15 usable pairs, computed here from Table 4: −45.0 %
(train, Method 2, 4 MB) to +8.0 % (dance, Method 1, 0.5 MB). Method 1 alone spans −14.1 % to
+8.0 %. The paper gives no aggregate error statistic and no per-scene calibration of L.

### (d) What happens when the target is larger than the converged model's size?

**Not stated.** The paper never poses this case. The closest evidence is the train 10 MB row, where
Method 1 reaches only 8.59 MB, which the paper attributes to compression-ratio drift rather than to
having run out of primitives to keep. p8:

> "method 1 reports larger rate error for 10 MB of 'train' than other rates. The reason is that
> different primitive densities will influence the lossless compression ratio. For GPCC, the denser
> the point cloud, the higher the compression ratio."

The only over/under-run the paper does discuss is the opposite direction, the target being too
small for Method 2 (p8): "'*' in 'dance': method 2 means that we cannot achieve the target rate
solely by adapting BD. The primitive position rate is larger than the target rate, resulting in
attribute BDs reduced to 0". Mechanically, since the controller can only prune and can only lower
bit depth, a target above the T_top size would leave both controllers inactive and land the model
at its unconstrained size. The paper does not say this.

### (e) Does the paper claim to be the first to do rate control for 3DGS, and what prior work does it cite?

**No first-of-its-kind claim.** The contribution is worded as p2: "Two effective rate control
methods are established based on the sparsification strategy." The abstract, p1, says only: "A
simple and effective rate control scheme is proposed to pivot the interpretable data compression
scheme." The comparative claim it does make is about predictability, p5 to p6:

> "Rate control is important for practical applications. However, with the current generative
> compression methods, which use the rate-distortion (RD) optimization as the loss function, the
> final data rate is difficult to predict. On the contrary, the proposed HybridGS is easier to
> realize accurate rate control."

The prior work it cites for rate-constrained 3DGS coding is CompGS(MM), Liu et al. 2024, p8:

> "CompGS(MM) (Liu et al., 2024) shared a close concept with Scaffold-GS, in which a group of anchor
> primitives is selected to realize inter-primitive prediction. They also formulated a
> rate-constrained optimization to balance the quality and bitrate."

The only "first" claim in the paper is about a different thing entirely, p9: "GGSC (Yang et al.,
2024) is the first traditional 3DGS compression benchmark".

### (f) Quality at the target compared to the same model compressed without the target?

**The paper does not run this comparison.** No ablation contrasts a rate-controlled run against an
uncontrolled run at matched size.

The comparison can be assembled across two tables, and the following is **computed here, not a
paper claim**. Both HR and LR in Table 1 are fixed-schedule checkpoints (50 000 and 70 000
iterations, no target). Table 4 Method 1 rows are target-driven. Same scenes, same k_c = 3, k_r = 2:

| scene | uncontrolled (Table 1) | target-driven (Table 4, M1) | delta |
|---|---|---|---|
| train | HR 21.26 PSNR at 4.19 MB | 21.13 PSNR at 3.96 MB (target 4 MB) | −0.13 dB at 0.23 MB smaller |
| dance | HR 39.25 PSNR at 1.04 MB | 38.99 PSNR at 1.06 MB (target 1 MB) | −0.26 dB at 0.02 MB larger |
| dance | LR 37.65 PSNR at 0.51 MB | 37.40 PSNR at 0.54 MB (target 0.5 MB) | −0.25 dB at 0.03 MB larger |

Reading: on the two scenes where both are available, hitting a target costs about 0.13 dB to
0.26 dB against the fixed-schedule checkpoint at comparable size. This is an inference across
tables, not something the paper measures or claims.

---

# 2. MesonGS++

## 1. Citation

- Title: "MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter
  Searching"
- First authors: Shuzhao Xie and Junchen Ge, marked "⋆Equal contribution". Corresponding author
  Zhi Wang. Affiliations Tsinghua SIGS, HKUST, HIT Shenzhen, CUHK, UT Austin, Shenzhen Technology
  University, Xiamen University, SFU, Jiangxing Intelligence.
- Venue as printed: the LaTeX header reads "JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST
  2021" and the footer "0000–0000/00$00.00 © 2021 IEEE". **No real venue is stated.** This is an
  unsubmitted or under-review IEEE-journal-format preprint.
- arXiv id: 2604.26799, "arXiv:2604.26799v2 [cs.CV] 7 May 2026" (p1)

## 2. Base representation

**Raw 3DGS Gaussians**, taken as a pre-trained checkpoint. p8: "To obtain the pre-trained 3D
Gaussians for compression, we train 30, 000 iterations and then save the checkpoints for both
datasets. We set the background as white."

Attribute changes: rotation quaternion (4 numbers) replaced by Euler angles (3 numbers); RAHT
applied to opacity, quaternion/Euler angles and degree-0 SH; degree ≥ 1 SH vector-quantized into a
codebook plus index table; geometry voxelized into an octree and coded with G-PCC.

Render-time requirements:
- **MLP: no.** The rotation matrix is rebuilt from Euler angles in closed form (Equation 5, p4).
- **Hash grid: no.**
- **Entropy decoding at render time: no per-frame decoding.** Decoding is a one-off unpack of
  G-PCC, torchac and LZ77, followed by inverse RAHT and de-quantization. The paper does not measure
  decode time.

The method also transfers to Scaffold-GS and 4DGS as inputs (Table VI, p11), which do need an MLP,
but the base method targets raw 3DGS.

## 3. Budget handle

The user supplies a **size budget S_T in bytes or MB**, and it is a **TARGET**. Algorithm 1, p6:
"Input: Size budget S_T and a pre-trained 3DGS model. Output: Hyperparameter set Φ* = {τ*, Q*}".
The MINLP is p5, Equation 9: "minimize_{τ,Q} M(τ, Q), subject to S(τ, Q) ≤ Size Budget, τ ∈ [0, 1],
Q ∈ [1, 32]^(C×B) ∩ Z^(C×B)".

The search returns the reserve ratio τ (fraction of Gaussians kept) and the per-group bit-width
matrix Q. Those two are the internal knobs the target drives.

### Every achieved-versus-requested pair in the paper

**Table IV, "Superiority of 0-1 ILP", p10.** Budget 3 × 10⁷ B for all three rows. Percentages
computed here.

| Method | Budget (B) | Searched (B) | Δsize (B) | Information loss | gap (computed) |
|---|---|---|---|---|---|
| GA | 3 × 10⁷ | 21 833 128 | 8 166 872 | 42 821 038 | −27.2 % |
| Vanilla ILP | 3 × 10⁷ | 28 934 805 | 1 065 195 | 1 258 394 | −3.55 % |
| 0-1 ILP (Our) | 3 × 10⁷ | 29 831 203 | 168 797 | 11 826 | −0.563 % |

**Table V, "Robustness evaluation", p11.** Verified against a rendered image. K is the number of
blocks. Gaps computed here.

| K | Budget (MB) | PSNR | SSIM | LPIPS | Searched Size (MB) | gap (computed) |
|---|---|---|---|---|---|---|
| 40 | 30 | 25.13 | 0.7410 | 0.2684 | 29.85 | −0.50 % |
| 30 | 30 | 25.15 | 0.7411 | 0.2685 | 29.91 | −0.30 % |
| 50 | 30 | 25.14 | 0.7413 | 0.2686 | 29.86 | −0.47 % |
| 40 | 20 | 25.07 | 0.7353 | 0.2752 | 19.83 | −0.85 % |
| 30 | 20 | 25.07 | 0.7357 | 0.2757 | 19.85 | −0.75 % |
| 50 | 20 | 25.12 | 0.7368 | 0.2743 | 19.92 | −0.40 % |

**Table VI, "Quantitative results on 3DGS variants", p11.** Verified against a rendered image.

| Method | Budget | PSNR | Size | CP Time (s) | FT Time (s) |
|---|---|---|---|---|---|
| ScaffoldGS | - | 29.42 | 663.9 MB | - | - |
| Ours+ScaffoldGS | 8 MB | 30.24 | 7.92 MB | 447 | 816 |
| 4DGS | - | 32.06 | 5.10 GB | - | - |
| Ours+4DGS | 200 MB | 32.07 | 198.64 MB | 79 | 28 |

The paper's own summary of Table VI, p10: "The final model sizes obtained by our search deviate
from the prescribed budgets by only 0.8% for ScaffoldGS and 0.5% for 4DGS, demonstrating the
accuracy of the proposed size estimator." Computed here from the rounded table values the deviations
are 1.0 % and 0.68 %, so the stated 0.8 % and 0.5 % come from unrounded numbers not printed.

**Termination tolerance**, Algorithm 1 line 11, p6: `if |S_a − S_T| / |S_T| < 0.05 then break`.
That is a 5 % acceptance band, wider than any deviation actually reported.

**Tables VII to IX, p17**, report five "target-size profiles" named Large, Medium, Small, Very
Small, Tiny with per-scene achieved sizes. **The requested budget behind each profile is never
stated**, so those are not achieved-versus-requested pairs. Averages: Mip-NeRF 360 65.36 / 57.19 /
49.78 / 42.16 / 36.15 MB; Deep Blending 55.45 / 48.42 / 42.25 / 35.90 / 31.21 MB; Tanks and Temples
31.09 / 27.25 / 23.78 / 20.40 / 17.81 MB.

## 4. When it binds, in full detail

**MesonGS++ is a post-training codec. There is no training-time controller.** p1 abstract:
"we propose MesonGS++, a size-aware post-training codec for 3D Gaussian compression." Input is a
30 000-iteration 3DGS checkpoint (p8).

**Pipeline, five stages (p2 to p3, Figure 2):**
1. Prune with joint importance I_g = I_d · I_i, keeping a fraction τ.
2. Octree voxelization of coordinates, depth d, attributes averaged on collision.
3. Quaternion → Euler angle replacement, RAHT on opacity / rotation / degree-0 SH, vector
   quantization of degree ≥ 1 SH with a budget-first retention of top-importance originals.
4. Group-wise mixed-precision quantization of RAHT AC coefficients.
5. Entropy coding: G-PCC for the octree, torchac plus LZ77 for the rest.

**Search loop, Algorithm 1, p6:**
- Initialize Q ← {8}^(C×B), M* ← 0.
- For each candidate τ in a discrete sweep: compress, get actual size S_a. Skip if `2 × S_a < S_T`
  (τ too small, too few points kept). Then loop: S_Δ ← S_a − S_T, re-solve the 0-1 ILP for Q,
  recompress, stop when `|S_a − S_T| / |S_T| < 0.05`. Keep the {τ, Q} with best estimated quality.
- The ILP is solved hierarchically, first per channel Q_c ∈ [1, 16]^C, then per group
  Q_g ∈ [0, 16]^B under the per-channel budget S_c = S_T · Q_{c,i} / Σ Q_{c,i} (p6).
- Q is capped at 16 options: p7, "we set Q as 16 to prune the search space."

**Controller operations:** prune (choose τ), reduce or raise bit depth (choose Q, in [1, 16]),
choose how many original degree ≥ 1 SH vectors to retain in the codebook. It never grows, splits or
clones a primitive, and it never changes a loss weight λ. There is no λ in the method.

**Can the primitive count INCREASE while the controller is active?** No new primitives are ever
created. But τ is swept over a range, so a **later search iteration can retain more primitives than
an earlier one**. p6: "if doubling the bit-width setting under the current configuration (up to a
maximum value of 16) cannot achieve the target size, it indicates that τ is too small, and thus,
more Gaussian points need to be retained." Bit widths can likewise go up as well as down inside the
ILP. This is search over configurations, not growth of a model, and it is bounded above by the
input checkpoint's count.

**What stops the controller:** the 5 % relative tolerance on actual (not estimated) compressed size,
and exhaustion of the τ sweep.

**Optional fine-tuning after the codec.** p7: "we fine-tune the model for multiple epochs after each
of the point pruning and coordinate quantization steps to restore reconstruction quality. In the
final round of fine-tuning, the coordinates are fixed". Iteration counts for fine-tuning are not
stated. Wall-clock is in Table VI: 816 s for ScaffoldGS, 28 s for 4DGS.

## 5. The size model

**Equation 12, p7:**

  S(Q) = Σ_{i,j} P_ij · Q_ij + C + S_Δ

- Q_ij: bit width assigned to quantization group (i, j), i over C channels, j over B blocks
- P_ij: "size of quantization groups", that is the element count of group (i, j), so P_ij · Q_ij is
  that group's bit cost
- C: "the accurate storage consumption of the metadata and the coordinates, which can be obtained by
  storing them to the disk directly" (p7), so the geometry and metadata terms are measured, not
  modelled
- S_Δ: a calibration residual, refreshed each outer iteration from the previous actual size,
  Algorithm 1 line 8, "S_Δ ← S_a − S_T"

**Why this form.** p7: "According to information theory [71], the lower bound of bit consumption can
be calculated by τN × (−Σ_i p_i log₂ p_i). However, such a size estimator is not suitable for the
formulation of ILP. ... Moreover, the relationship between the bit-width settings and the estimated
size is non-linear, which cannot satisfy the linear requirement of the ILP." So the true entropy
model is deliberately abandoned for a linear surrogate that an ILP solver can consume.

**Calibration.** Two mechanisms. The geometry and metadata terms are obtained by actually writing
them to disk. The residual S_Δ is corrected by re-measuring the real compressed size after each ILP
solve, so the estimator is closed-loop against ground truth rather than open-loop.

**Differentiable? No.** It is a linear integer program over discrete bit widths, solved by an ILP
solver. It never enters a gradient.

**Stated error.** p10: "The final model sizes obtained by our search deviate from the prescribed
budgets by only 0.8% for ScaffoldGS and 0.5% for 4DGS, demonstrating the accuracy of the proposed
size estimator." Table IV gives Δsize = 168 797 B on a 3 × 10⁷ B budget (−0.563 %, computed here).
Table V gives −0.30 % to −0.85 % (computed here). Note these are end-to-end search accuracies after
calibration, not the raw error of Equation 12 on a single evaluation. The single-shot estimator
error is never isolated: p7 says only "Of course, such a estimation for the compressed file size is
not accurate. To calibrate it, we update the S_Δ multiple times".

## 6. What "size" means in the tables

**Entropy-coded bytes.** p7: "We compress the octree with G-PCC, and the remained elements are coded
and packed via torchac [55] and LZ77 [72], [73]."

Included, p7: "(1) Octree, i.e., quantized 3D coordinates; (2) DC coefficients and quantized
coefficients; (3) Codebook and the corresponding mapping table, or the preserved original 1+ SH
coefficients under big size budget; (4) Metadata: Min-Max values of each block of quantized
coefficients, octree depth, number of blocks, and bit-width settings." DC coefficients, codebook and
metadata are stored as float; everything else as integers.

Composition, p9: "In the Synthetic-NeRF dataset, the proportions of octree, metadata, important
attributes, and unimportant attributes are: 43%, 0.04%, 34%, 23%, respectively. In the Mip-NeRF 360
dataset, the proportions of the above four elements are: 39%, 0.02%, 47%, 14%, respectively."

**One exception.** The stage-ablation Table I uses a different measure. p9: "We calculate the size
after a zip compression for fair comparison."

## 7. Runtime cost

**Hardware: not stated.** No GPU or CPU model is named anywhere in the paper.

**Encode / search time.** Table I, p9, per-stage latency in seconds:

| Stage | Synthetic-NeRF latency (s) | Mip-NeRF 360 latency (s) |
|---|---|---|
| +Prune | 22.73 | 165.98 |
| +Voxel | 5.98 | 238.78 |
| +Replace | 0.00 | 0.02 |
| +Cluster | 96.75 | 267.66 |
| +RAHT | 0.04 | 0.19 |
| +HS (hyperparameter search) | 111.83 | 253.35 |

Search breakdown, p9: "Computing the estimated quality is very fast due to our parallel group-wise
quantization implementation, only cost 0.49 s. The subsequent calibration (15.10s) and 0-1 ILP
stages (58.52) take comparable time, and together (104.51) provide an efficient way to search the
bit-width configuration under a prescribed size budget."

Table VI, p11: compression time 447 s and fine-tune 816 s for ScaffoldGS; 79 s and 28 s for 4DGS.
p11: "For 4DGS, MesonGS++ reaches the target budget within 51 s and further improves rendering
quality after 28 s of fine-tuning." The 51 s here does not match the 79 s in the table.

**Decode time: not stated. Render FPS: not stated. GPU memory: not stated.**

## 8. Main quantitative results

**Baseline numbers are not tabulated anywhere.** All comparisons against SizeGS, FCGS, MesonGS and
FlexGaussian live only in the rate-distortion curves of Fig. 6, p7, and in the qualitative Fig. 7
captions. So no baseline row can be copied exactly. The reference-model row that is tabulated is
3D-GS in Table I.

**Table VII, Mip-NeRF 360 averages, p17** (nine scenes, five profiles):

| Profile | Size (MB) | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Large | 65.3639 | 27.0555 | 0.803006 | 0.277140 |
| Medium | 57.1948 | 27.0237 | 0.802110 | 0.278507 |
| Small | 49.7811 | 26.9801 | 0.800961 | 0.280256 |
| Very Small | 42.1551 | 26.9230 | 0.799244 | 0.282904 |
| Tiny | 36.1463 | 26.8537 | 0.796964 | 0.286190 |

**Table VIII, Deep Blending averages, p17** (drjohnson, playroom):

| Profile | Size (MB) | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Large | 55.4523 | 29.6487 | 0.902660 | 0.312366 |
| Medium | 48.4188 | 29.6394 | 0.902429 | 0.313077 |
| Small | 42.2545 | 29.6212 | 0.902108 | 0.314164 |
| Very Small | 35.8971 | 29.6082 | 0.902048 | 0.314644 |
| Tiny | 31.2107 | 29.5868 | 0.901607 | 0.315788 |

**Table IX, Tanks and Temples averages, p17** (train, truck):

| Profile | Size (MB) | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Large | 31.0872 | 23.3747 | 0.839182 | 0.222529 |
| Medium | 27.2509 | 23.3591 | 0.838197 | 0.223913 |
| Small | 23.7806 | 23.3228 | 0.837141 | 0.225180 |
| Very Small | 20.3951 | 23.2825 | 0.834568 | 0.228278 |
| Tiny | 17.8052 | 23.2472 | 0.832404 | 0.231129 |

**Reference rows, Table I, p9** (zip-compressed size, see field 6):

| Stage | Synthetic-NeRF PSNR / SSIM / LPIPS / Size | Mip-NeRF 360 PSNR / SSIM / LPIPS / Size |
|---|---|---|
| 3D-GS | 33.37 / 0.9696 / 0.0305 / 68.55 | 28.98 / 0.8647 / 0.1931 / 641.73 |
| +HS (full pipeline) | 29.47 / 0.9476 / 0.0511 / 1.14 | 27.20 / 0.8238 / 0.2402 / 18.43 |

**Nearest thing to baseline numbers**, Fig. 7 caption text, p8, "size/PSNR" pairs per column
(GT, 3DGS, FlexGaussian, FCGS, Ours(Small), Ours(Large)):
- Room: 377 MB / 34.89 dB, 22.0 MB / 33.65 dB, 15.2 MB / 33.67 dB, 15.2 MB / 34.52 dB,
  22.0 MB / 34.70 dB
- Stump: 1174 MB / 27.62 dB, 58.0 MB / 26.47 dB, 56.5 MB / 26.90 dB, 56.3 MB / 27.56 dB,
  57.9 MB / 27.78 dB
- Garden: 1380 MB / 27.62 dB, 89.8 MB / 26.58 dB, 69.0 MB / 27.09 dB, 68.0 MB / 27.28 dB,
  89.7 MB / 28.52 dB
- Drjohnson: 805 MB / 35.30 dB, 29.2 MB / 32.89 dB, 34.8 MB / 33.05 dB, 34.8 MB / 34.70 dB,
  29.2 MB / 34.47 dB

These are train-view PSNRs in a qualitative figure, and the column-to-value assignment comes from
flowed text, so treat them as indicative rather than exact.

## 9. Densification and count control

**No densification.** MesonGS++ never creates a primitive. The count is fixed by the input
checkpoint and then reduced by the reserve ratio τ.

Importance, p3: "For a Gaussian g, we define its importance score I_g as the product of the
view-dependent importance score I_d and the view-independent importance score I_i: I_g = I_d I_i."
I_d = Σ_{p ∈ P} α_i · Π_{j<i} (1 − α_j). I_i = (V_norm)^β where V is the product of the scale
vector, normalized by the 90 % largest and clipped to [0, 1].

Empirical basis, p3 and Fig. 1: "We notice that 40% of the Gaussians contain over 80% of the
importance."

**Does the size target influence the count? Yes, through τ.** τ is one of the two decision variables
the budget-constrained search optimizes. p6: "we propose to traverse the value of reserve ratio τ."
It is a selection from a frozen pool, not a growth policy.

Octree voxelization also merges primitives, p4: "When multiple Gaussians existing within a voxel, we
average the corresponding attributes for deduplication."

## 10. Stated limitations and future work (verbatim)

**MesonGS++ has no Limitations section and no Future Work section.** The conclusion, p12, is purely
positive:

> "We present MesonGS++, a size-aware post-training codec for 3D Gaussian compression. Beyond
> effective compression operators, MesonGS++ tackles a key practical bottleneck: globally
> configuring the many coupled hyperparameters across the pruning–transformation–quantization–
> encoding pipeline under a prescribed budget. The codec integrates importance-based pruning, octree
> geometry coding, attribute transformation, selective vector quantization for higher-degree
> spherical harmonics, and group-wise mixed-precision quantization with entropy coding. A size-aware
> configuration module jointly optimizes the reserve ratio and bit allocation via discrete sampling,
> 0-1 ILP, a linear size estimator, and parallel quantization. An optional piecewise finetuning
> stage further restores quality. Experiments show strong rate-distortion performance, accurate size
> control, and favorable speed-quality trade-offs, with the searched configurations also serving as
> useful priors for other compressed Gaussian representations."

The nearest thing to acknowledged weaknesses, stated in passing:

p7: "Of course, such a estimation for the compressed file size is not accurate. To calibrate it, we
update the S_Δ multiple times, as shown in Algo. 1."

p7 (a hard constraint on the pipeline order): "Consequently, modifying the coordinate values after
quantization may alter the Morton order, disrupting the previously constructed quantization groups
and invalidating the searched bit-width settings."

p7 (a scoping choice): "Note that we do not apply RAHT to the scale vectors when the quantization
bit is 8."

p11 (constraint on prior work, not on itself): "Though some works [31], [33] employed MPQ, their
required retraining for configuring the quantization settings."

## 11. Every sentence touching a size, byte, bitrate, memory or FPS TARGET during training, or "rate control", or "budget" (verbatim, with page)

p1: "Existing post-training compression methods still rely on many coupled hyperparameters across
pruning, transformation, quantization, and entropy coding, making it difficult to control the final
compressed size and fully exploit the rate-distortion trade-off."

p1: "On the configuration side, it treats the reserve ratio and bit-width allocation as the dominant
rate-distortion knobs and jointly optimizes them under a target storage budget via discrete sampling
and 0–1 integer linear programming."

p1: "Extensive experiments show that MesonGS++ achieves over 34× compression while preserving
rendering fidelity, outperforming state-of-the-art post-training methods and accurately meeting
target size budgets."

p1: "As a result, existing post-training methods often rely on manual tuning or locally designed
heuristics, which provide limited global control over the final compressed size and frequently force
overly coarse quantization configurations. This issue becomes particularly problematic when the
target file size is a hard constraint, as in transmission and storage-limited applications."

p2: "On the configuration side, MesonGS++ treats the reserve ratio τ and group-wise mixed-precision
bit allocation as the dominant rate-distortion knobs, and jointly optimizes them under a prescribed
storage budget."

p2: "In this way, MesonGS++ unifies high-quality post-training compression and accurate target-size
control within a single framework."

p2: "We identify hyperparameter configuration as a central bottleneck of post-training 3DGS
compression, and formulate the dominant configuration variables, namely the reserve ratio and
group-wise bit-width allocation, as a budget-constrained optimization problem."

p2: "Extensive experiments demonstrate that MesonGS++ achieves over 34× compression while
maintaining fidelity, surpassing state-of-the-art post-training methods in rate-distortion
performance, with accurate target-size control and favorable speed-quality trade-offs."

p3: "After that, to find the optimal hyperparameter configuration, specifically the reserve ratio
and quantization bit-widths, under a given size constraint, we formulate the search as a Mixed
Integer Nonlinear Programming (MINLP) problem and propose an efficient hierarchical solver, as
illustrated in Fig. 5."

p4: "To mitigate this, we adopt a budget-first allocation strategy. We first estimate the compressed
size of essential components, including octree geometry, VQ indices, base codebook, and quantized
RAHT coefficients. Then, we allocate any remaining budget to preserve the original 1+D SH
coefficients of the top-ranked Gaussians by importance score."

p5: "For a given size budget, there exists an optimal hyperparameter configuration that maximizes
rendering quality, representing the Pareto-optimal trade-off between size and visual quality within
this framework."

p5: "Therefore, a principled search strategy is required to identify the optimal hyperparameter
configuration under a specified size constraint."

p5: "Given the above pipeline, our goal is to search for suitable values of the τ and Q under a
given size constraint, while maximizing visual quality."

p5, Equation 9: "minimize_{τ,Q} M(τ, Q), subject to S(τ, Q) ≤ Size Budget, τ ∈ [0, 1], Q ∈ [1,
32]^(C×B) ∩ Z^(C×B)."

p6, Algorithm 1: "Input: Size budget S_T and a pre-trained 3DGS model"

p6, Algorithm 1: "if |S_a − S_T| / |S_T| < 0.05 then break"

p6: "The input is a target size and a pretrained 3DGS, and the output is the optimal hyperparameters
set Φ* = {τ*, Q*} that can satisfy the size constraint while maximizing visual quality."

p6: "Here, 2 × S_a < S_T refers to the situation where, if doubling the bit-width setting under the
current configuration (up to a maximum value of 16) cannot achieve the target size, it indicates
that τ is too small, and thus, more Gaussian points need to be retained."

p6: "If the stored result size S_a is sufficiently close to the target size S_T, the search is
terminated."

p6: "The objective of solving the ILP is to find the best bit configuration in this search space
that optimally balances quality loss Ω and the a size limit S."

p6, Equation 10: "minimize_Q Ω(Q), subject to S(Q) ≤ Size Budget"

p6: "Then, we calculate the size budget for each channel of attributes based on the Q_c. For
example, the size budget of channel i can be computed by: S_c = S_T · Q_{c,i} / Σ Q_{c,i}."

p7: "Then the challenge of size estimator lies in estimating the size of quantized attributes."

p7: "Besides, the size estimator must be a linear function of the bit-width variables."

p7: "According to information theory [71], the lower bound of bit consumption can be calculated by
τN × (−Σ_i p_i log₂ p_i). However, such a size estimator is not suitable for the formulation of
ILP."

p7: "Hence, an explicit and linear relationship between the size S and the bit-width setting Q must
be established. Thus, we estimate the size by S(Q) = Σ_{i,j} P_ij Q_ij + C + S_Δ."

p7: "Of course, such a estimation for the compressed file size is not accurate. To calibrate it, we
update the S_Δ multiple times, as shown in Algo. 1."

p7: "(3) Codebook and the corresponding mapping table, or the preserved original 1+ SH coefficients
under big size budget"

p8: "We conduct extensive experiments to evaluate rate-distortion performance, target-size control,
and visual fidelity under the post-training setting."

p8: "Compared with existing post-training 3DGS compression methods, MesonGS++ consistently achieves
favorable compression-quality trade-offs across diverse size budgets, while preserving rendering
quality and satisfying target storage constraints more accurately."

p9: "These results verify that combining an effective post-training codec with size-aware
hyperparameter configuration improves both rate-distortion performance and target-size
controllability."

p9: "The subsequent calibration (15.10s) and 0-1 ILP stages (58.52) take comparable time, and
together (104.51) provide an efficient way to search the bit-width configuration under a prescribed
size budget."

p10: "The 0-1 ILP fully utilizes the size budget while minimizing information loss."

p10: "As shown in Tab. V, for varying numbers of blocks and different target sizes, our method
consistently finds appropriate bit-width settings, ensuring that the final file size is close to the
target while maintaining optimal visual quality."

p10: "The final model sizes obtained by our search deviate from the prescribed budgets by only 0.8%
for ScaffoldGS and 0.5% for 4DGS, demonstrating the accuracy of the proposed size estimator."

p11: "For 4DGS, MesonGS++ reaches the target budget within 51 s and further improves rendering
quality after 28 s of fine-tuning."

p11: "Third, qbits converts the searched bit-widths into target quantization steps for the feature,
scaling, and offset groups, and uses them to initialize the bias terms of the three Q-prediction
heads in the grid MLP."

p11: "FCGS [51] is a concurrent feed-forward approach that enables one-shot compression, but it is
limited to fixed target rates for specific pre-trained models. Orthogonally, layered codec designs
[23], [94], [95] address rate adaptation under fluctuating network conditions"

p12: "It combines joint importance-based pruning, attribute transformation, and group-wise
mixed-precision quantization, while jointly optimizing the reserve ratio and bit-width allocation
under a storage budget."

p12: "MesonGS++ tackles a key practical bottleneck: globally configuring the many coupled
hyperparameters across the pruning–transformation–quantization–encoding pipeline under a prescribed
budget."

p12: "Experiments show strong rate-distortion performance, accurate size control, and favorable
speed-quality trade-offs"

## 12. Code URL and licence

- p1, printed as the word "here" with an embedded hyperlink: "Our code is available at here." The
  target of the link is `https://github.com/mmlab-sigs/mesongs_plus`.
- The arXiv landing link embedded in the same page is `https://arxiv.org/abs/2604.26799v2`.
- Licence: **not stated** anywhere in the PDF.

## MesonGS++ additional questions

### How does the size estimator differ from SizeGS's?

**The paper never says.** SizeGS appears exactly three times: as a baseline curve label "SizeGS (MM
25)" in Fig. 6 (p7), in the baselines sentence on p8 ("We compare our method with state-of-the-art
post-training compression methods, including MesonGS [25], FCGS [51], SizeGS [26], and
FlexGaussian [50]"), and in reference [26] ("Sizegs: Size-aware compression of 3d gaussian splatting
via mixed integer programming, in ACM MM, 2025"). There is no methodological comparison of the two
size estimators anywhere in the text, and SizeGS is not discussed in the Related Work section.

What can be said from this paper alone is the shape of its own estimator, described in field 5:
a **linear** function of bit widths, Σ P_ij Q_ij + C + S_Δ, chosen specifically so an ILP solver can
consume it, with C measured by writing geometry and metadata to disk and S_Δ recalibrated from the
real compressed size each outer iteration. The paper's stated reason for rejecting an
entropy-theoretic estimator is on p7: it is too slow (requires re-quantizing and re-histogramming
per iteration) and it is nonlinear, so it "cannot satisfy the linear requirement of the ILP". Note
that both papers share first author Shuzhao Xie and corresponding author Zhi Wang, and both use
mixed integer programming, so MesonGS++ reads as the successor to SizeGS rather than a
differentiated competitor.

### Its reported estimation error

- End-to-end search accuracy, p10: "deviate from the prescribed budgets by only 0.8% for ScaffoldGS
  and 0.5% for 4DGS".
- Table IV, p10, 0-1 ILP row: Δsize 168 797 B on a 3 × 10⁷ B budget, which is −0.563 % (computed
  here). Comparison rows: Vanilla ILP Δ 1 065 195 B (−3.55 %), GA Δ 8 166 872 B (−27.2 %).
- Table V, p11: searched sizes 29.85, 29.91, 29.86 MB against a 30 MB budget and 19.83, 19.85,
  19.92 MB against a 20 MB budget, so −0.30 % to −0.85 % (computed here).
- Search tolerance, Algorithm 1 p6: 5 %.
- **The raw single-shot error of Equation 12, before S_Δ calibration, is never reported.** p7 says
  only "such a estimation for the compressed file size is not accurate".

### Does anything run during training?

**No.** The method is entirely post-training, operating on a saved 30 000-iteration checkpoint
(p8). The only gradient-based component is the **optional** fine-tuning module that runs after
compression stages: p7, "We also provide a fine-tuning module for practical convenience.
Specifically, we directly quantize the coordinates and attributes without applying any
transformations. To improve compression quality, we fine-tune the model for multiple epochs after
each of the point pruning and coordinate quantization steps to restore reconstruction quality." That
fine-tuning restores quality at a size the search already fixed. It does not drive the size.

Its one contact with training is as an **initializer for someone else's training loop**, p11:
"We investigate transferring three types of priors from MesonGS++ to HAC++, a famous online 3DGS
compression method." and "Transferring the compression configuration from MesonGS++ helps HAC++
converge faster without sacrificing final accuracy." That transfer carries a configuration, not a
byte target, into HAC++.

---

# 3. RAVE

## 1. Citation

- Title: "RAVE: RATE-ADAPTIVE VISUAL ENCODING FOR 3D GAUSSIAN SPLATTING"
- First author: Hoang-Nhat Tran (Université Paris-Saclay, CNRS, CentraleSupélec, L2S), co-first
  author Francesco Di Sario (University of Turin). Co-authors Gabriele Spadaro, Giuseppe Valenzise,
  Enzo Tartaglione.
- Venue, p1 footnote: "This article has been accepted for publication at the 2026 IEEE International
  Conference on Acoustics, Speech, and Signal Processing (ICASSP)."
- arXiv id: 2512.07052, "arXiv:2512.07052v2 [cs.CV] 29 Jan 2026" (p1)
- Length: 4 pages plus 1 page of references. **No appendix, no supplementary material.**

## 2. Base representation

**Scaffold-GS anchors plus MLP.** p3: "For the 3DGS model, we adopt Scaffold-GS [20] as our backbone
architecture."

Render-time requirements:
- **MLP: yes**, inherited from Scaffold-GS, which predicts neural Gaussian attributes from anchor
  features. The paper does not describe the MLP or its cost.
- **Hash grid: no.**
- **Entropy decoding: yes, as a transmission step, not per frame.** p3: "the parameters of the
  selected Gaussians are compressed using entropy coding (LZMA in our implementation, chosen for its
  strong compression ratio and practical speed), transmitted, and decoded at the receiver side to
  reconstruct the scene." Whether decoding happens once or at render time is not stated.
- The codec is swappable, p3: "Because our method is codec-agnostic, different compression backends
  could be substituted with no modification to the overall pipeline."

## 3. Budget handle

The user supplies **R_target, a target rate**, and it is a **TARGET** in form. p3: "Our objective is
to generate high-quality scene reconstructions at an arbitrary target rate R_target, determined by
constraints such as network bandwidth, storage budget, or hardware capabilities, without requiring
additional training or model fine-tuning."

R_target is applied at selection time only, and only within the interval spanned by the L
pre-trained anchor levels. Outside that interval it is not defined. p4: "as we reduce the size of
the lowest level G₁, our method is able to achieve arbitrarily low bitrate operating points" makes
extending the range a matter of retraining with a different G₁.

### Every achieved-versus-requested pair in the paper

**None. The paper contains no tables at all.** Every result is a figure. There is not a single
achieved-versus-requested size pair anywhere in the 5 pages, and no numeric statement of how closely
a requested R_target is met.

The only quantities extractable from the figures are axis tick labels, which are ranges rather than
data points, listed here as such:
- Fig. 3, p3, PSNR-Size: Mip-NeRF 360 size axis 10 to 50 MB, PSNR axis 25.0 to 27.5 dB. Tanks and
  Temples size axis 5 to 25 MB, PSNR axis 22.5 to 24.0 dB. Deep Blending size axis 10 to 40 MB,
  PSNR axis 29.4 to 30.4 dB.
- Fig. 3, p3, LPIPS-Size: Mip-NeRF 360 LPIPS axis 0.275 to 0.400. Tanks and Temples 0.22 to 0.30.
  Deep Blending 0.26 to 0.36.
- Fig. 3 legend: LightGaussian, Reduced-3DGS, RDO, HAC, RAVE (Ours).
- Fig. 4, p4, room scene: size axis 4 to 16 MB, PSNR axis 29.5 to 32.5 dB. Legend: Global, Multiple
  Anchors, Context-Aware (Ours). Naive multi-anchor uses "50 levels".

## 4. When it binds, in full detail

**Iteration counts: not stated anywhere.** The paper gives no training schedule, no iteration
budget, no anchor count L, no rate delta ΔR value, and no number of fine-tuning steps. The only
statement is p2: "With marginal extra training costs (in terms of learning iterations), our method
is the first to allow dynamic rate adaptation without retraining."

**Pipeline, p2 to p3:**
1. Start from a **pre-trained** Scaffold-GS model. p2: "Starting from a pre-trained model, a
   hierarchy of gradient-based masks can be generated to progressively select subsets of Gaussians."
2. Build the hierarchy G_l = ∪_{i=1..l} C_i at L fixed rate-distortion levels, ranking Gaussians by
   gradient on the training set. p2: "The Gaussians hierarchy is established by defining a rate
   delta ΔR that translates to a fraction of Gaussians to be added/removed to move from anchor
   l − 1 to l, and their ranking is defined based on the gradient calculated on the training set."
3. **Quantization-aware fine-tuning with stochastic mask sampling.** p2: "After building the
   hierarchy, we can perform quantization-aware fine-tuning using stochastic mask sampling: this
   means that, with uniform probability, we choose to optimize either of the L anchor points." This
   is the only stage where anything is trained under multiple rates, and it optimizes **all L levels
   jointly, not any user-supplied target**.
4. At deployment, given R_target, interpolate (Equation 2) and select (Equation 3), then LZMA-encode.

**When the target is given: after all training is complete.** The whole point of the method is that
R_target arrives at inference. p3: "Crucially, since the ranking is precomputed, this selection
process can be repeated for any number of target rates R_target1, R_target2, . . . without
recomputing gradients".

**Controller operations:** select a subset of Gaussians by gradient rank. That is the entire
controller. It does not prune adaptively, does not change bit depth, does not change λ, does not
grow, split or clone.

**Can the primitive count INCREASE while the controller is active?** Relative to the currently
selected level, **yes, within the pre-built hierarchy**. Equation 2 gives |G_target| = |G_l| +
(R_target − R(G_l)) / (R(G_{l+1}) − R(G_l)) · (|G_{l+1}| − |G_l|), which adds Gaussians from the
context C_{l+1} as R_target rises. p2 says the rate delta ΔR "translates to a fraction of Gaussians
to be added/removed". So RAVE moves both ways along the ladder, unlike HybridGS. But it can never
exceed |G_L|, the top anchor of the frozen hierarchy, and no new primitive is ever created.

**What stops the controller:** it is a single closed-form selection, not a loop. It terminates when
the top-ranked Gaussians fill |G_target|.

## 5. The size model

**Not stated as a formula.** There is no size estimator in this paper. R(G_l) is written as "the
rate for G_l" (p3) with no definition of how it is computed, no unit, and no statement of whether it
is measured after LZMA or predicted.

The only relation given is the **linear interpolation of Gaussian count against rate** between two
adjacent anchors, Equation 2, p3:

  |G_target| = |G_l| + [ (R_target − R(G_l)) / (R(G_{l+1}) − R(G_l)) ] · (|G_{l+1}| − |G_l|)

- G_l: the Gaussian set at anchor level l, G_l = ∪_{i=1..l} C_i
- C_l: the context, "Gaussians to be introduced to move from anchor l − 1 to l" (p2)
- R(·): the rate of a Gaussian set, undefined
- R_target: the requested rate
- l is chosen as l = argmax{R(G_l) ≤ R_target}, so the interval containing R_target

Selection rule, Equation 3, p3: pick the |ΔG| Gaussians of C_{l+1} with largest ‖∂L/∂θ_i‖₂, where
"θ represents the parameters of the Gaussian, and L is the rendering loss function. The gradients
are calculated only once on the training set."

**Differentiable? Not applicable.** The size relation is a linear interpolation used at selection
time. It is never in a loss. The gradient in Equation 3 is the rendering-loss gradient used as an
importance score, not a gradient of size.

**Stated error between predicted and actual encoded bytes: not stated.** The paper reports no
accuracy figure for the interpolation, and the assumption that rate is linear in Gaussian count
between anchors is never validated numerically. The only qualitative caveat is p3: "leveraging the
principle of locality, we know that the denser the anchors for the same bitrate regime, the more
accurate the gradient-based ranking."

## 6. What "size" means in the tables

There are no tables. The figures use "Size [MB]". Given p3, "the parameters of the selected
Gaussians are compressed using entropy coding (LZMA in our implementation)", size is presumably
**LZMA-compressed bytes** of the selected Gaussians' parameters.

**What is included is not stated.** The paper does not say whether the Scaffold-GS MLP weights, the
anchor structure, the hierarchy metadata (which Gaussians belong to which context), or the
gradient-ranking side information are counted. That last item matters, because the receiver needs to
know which subset was selected, and that index set is not obviously free.

## 7. Runtime cost

**Everything in this field is not stated.** No encode time, no decode time, no render FPS, no GPU
memory, and no hardware is named. The paper makes only relative claims:

- p1 abstract: "Our method is computationally lightweight, requires no retraining for any rate"
- p2: "With marginal extra training costs (in terms of learning iterations)"
- p4: "In contrast, our approach achieves a truly continuous RD curve while being faster, more
  lightweight, and consistently yielding higher quality reconstructions."
- p4, on the naive alternative: "it is necessary to compute a new ordering of the Gaussians based on
  gradient evaluation for each new anchor. As the number of levels increases, this calculation grows
  linearly more expensive"

None of these carries a number.

## 8. Main quantitative results

**No numbers are reported.** There is no table on Mip-NeRF 360, Tanks and Temples or Deep Blending,
and no PSNR, SSIM, LPIPS or size value appears in the text. LPIPS is plotted but never tabulated,
and SSIM is not shown at all.

The verbal claims, p4:

> "As shown in Fig. 3, our method achieves competitive performance in both PSNR and LPIPS compared
> to the most recent 3DGS compression techniques."

> "More specifically, our method reaches state-of-the-art performance on Mip-NeRF 360, slightly
> outperforming HAC, while still being competitive on the other datasets."

Baselines in Fig. 3: LightGaussian, Reduced-3DGS, RDO, HAC. No numeric row can be copied for any of
them.

## 9. Densification and count control

Densification is whatever Scaffold-GS does during the original pre-training, described only in the
preliminaries, p2: "These parameters are then optimized through gradient descent, while
simultaneously performing adaptive refinement operations that split, merge, or adjust the primitives
to better match the underlying geometry."

**The size target does not influence densification.** The hierarchy is built after pre-training,
from a frozen set. The count at deployment is decided by Equation 2 from R_target and by the
gradient ranking of Equation 3.

Gradient scores are computed once per anchor level, p3: "For each Gaussian, we compute an importance
score by accumulating its gradient over a single forward pass through the training set. This
operation is performed only once per anchor level and provides a stable ranking of Gaussians based
on their contribution to reconstruction quality."

## 10. Stated limitations and future work (verbatim)

**RAVE has no Limitations section and no Future Work section.** The conclusion, p4, is entirely
positive:

> "We presented a flexible compression framework for 3D Gaussian Splatting that allows continuous
> interpolation between compression rates while maintaining efficient rendering and competitive
> visual quality. Unlike existing approaches constrained to fixed-rate operation, our method adapts
> seamlessly to diverse computational and bandwidth conditions, enabling practical deployment in
> immersive multimedia scenarios. The results show that high-quality rendering can be preserved
> across a wide range of operating points at minimal computational cost, demonstrating that adaptive
> compression of 3DGS is both feasible and effective. This opens the door to more versatile scene
> representations that can dynamically respond to application requirements, paving the way toward
> scalable, interactive, and resource-aware immersive experiences."

The only acknowledged difficulties, all framed as difficulties the method overcomes:

p2: "Notably, achieving such flexibility is not straightforward, as naïve solutions tend to either
degrade quality or impose significant computational overhead."

p4: "In extreme low-bitrate regimes, removing Gaussians significantly degrades the reconstruction
quality, leading to steeper regions in the left-most parts of the RD curves."

p4: "With this global context, however, the ranking is dominated by structures that become relevant
only at higher rates, resulting in poorly aligned intermediate reconstructions."

p4: "This highlights a structural limitation of such an approach: while being able to approximate
continuity by brute force, it becomes both inefficient and unstable as the number of anchors grows."
(this is a limitation of the naive multi-anchor baseline, not of RAVE)

## 11. Every sentence touching a size, byte, bitrate, memory or FPS TARGET during training, or "rate control", or "budget" (verbatim, with page)

p1: "Existing approaches, however, operate at fixed rates, limiting adaptability to varying
bandwidth and device constraints."

p1: "In this work, we propose a flexible compression scheme for 3DGS that supports interpolation at
any rate between predefined bounds."

p1: "Experiments demonstrate that the approach achieves efficient, high-quality compression while
offering dynamic rate control, making it suitable for practical deployment in immersive
applications."

p1, Fig. 1 caption: "RAVE produces a continuous rate–distortion curve in a single, efficient
end-to-end training. This allows seamless adaptation of the bitrate to different constraints without
retraining, offering both state-of-the-art quality and practical deployment."

p2: "However, most existing methods operate at fixed compression rates [10, 11, 12, 13], which
limits their adaptability to different application requirements, such as variable bandwidth and
device capabilities. This rigidity limits the practical deployment of 3DGS across various real-world
scenarios, where dynamic trade-offs between quality, storage, and computation are often required in
real-time."

p2: "RAVE supports interpolation at any rate within a specified range, yielding fine-grained control
over the rate-distortion trade-off through predefined contexts (Fig. 1). With marginal extra
training costs (in terms of learning iterations), our method is the first to allow dynamic rate
adaptation without retraining."

p2, Fig. 2 caption: "For any target rate R_target, the top-ranked Gaussians are selected to form
G_target, then compressed and decoded to reconstruct the scene. This enables multiple operating
points and a continuous rate–distortion curve from a single trained model."

p2: "we can introduce a variable-rate approach by defining anchor points at L fixed rate-distortion
levels."

p2: "The Gaussians hierarchy is established by defining a rate delta ΔR that translates to a
fraction of Gaussians to be added/removed to move from anchor l − 1 to l, and their ranking is
defined based on the gradient calculated on the training set."

p2: "After building the hierarchy, we can perform quantization-aware fine-tuning using stochastic
mask sampling: this means that, with uniform probability, we choose to optimize either of the L
anchor points."

p3: "leveraging the principle of locality, we know that the denser the anchors for the same bitrate
regime, the more accurate the gradient-based ranking. Therefore, given R(G_l) the rate for G_l, the
number of Gaussians for a specific bitrate R_target, for which l = argmax{R(G_l) ≤ R_target}, will
be |G_target| = |G_l| + (R_target − R(G_l))/(R(G_{l+1}) − R(G_l)) (|G_{l+1}| − |G_l|)."

p3: "Among the budget |ΔG| = |G_target| − |G_l|, we select the Gaussians having the highest gradient
from the context C_{l+1}"

p3: "Our objective is to generate high-quality scene reconstructions at an arbitrary target rate
R_target, determined by constraints such as network bandwidth, storage budget, or hardware
capabilities, without requiring additional training or model fine-tuning."

p3: "Given a target rate R_target, we identify the anchor level that contains the desired number of
Gaussians and select the top-ranked ones until the rate budget is satisfied, producing a subset
G_target."

p3: "Crucially, since the ranking is precomputed, this selection process can be repeated for any
number of target rates R_target1, R_target2, . . . without recomputing gradients, enabling the
generation of multiple rate–quality operating points from a single trained model."

p3: "Finally, the parameters of the selected Gaussians are compressed using entropy coding (LZMA in
our implementation, chosen for its strong compression ratio and practical speed), transmitted, and
decoded at the receiver side to reconstruct the scene."

p3, Fig. 3 caption: "Our method is the only one that allows for a continuous rate without requiring
any retraining."

p4: "Remarkably, RAVE is the only approach that supports continuous variable rate compression,
whereas all other single-rate methods require separate, ad hoc training runs to obtain each point on
the rate–distortion curve. Training multiple models for different rates is very expensive and,
hence, not feasible in practice, where we often need to adapt the rate to specific constraints."

p4: "It is particularly noteworthy that as we reduce the size of the lowest level G₁, our method is
able to achieve arbitrarily low bitrate operating points. In extreme low-bitrate regimes, removing
Gaussians significantly degrades the reconstruction quality, leading to steeper regions in the
left-most parts of the RD curves."

p4: "it ensures that the ranking reflects the contribution of the Gaussians added at that specific
target rate, resulting in a smooth RD curve with more coherently evolving intermediate models."

p4: "In principle, with this simple strategy, we can densely populate the rate axis. However, this
comes at a significant computational cost."

p4: "our method adapts seamlessly to diverse computational and bandwidth conditions, enabling
practical deployment in immersive multimedia scenarios."

## 12. Code URL and licence

- p1, printed: `https://github.com/inspiros/RAVE`
- Licence: **not stated** in the PDF.

## RAVE additional questions

### How is any rate between the bounds reached?

By **linear interpolation of the Gaussian count between two adjacent anchor levels, followed by
top-k selection by gradient magnitude from the local context**. Two equations, both on p3:

> "given R(G_l) the rate for G_l, the number of Gaussians for a specific bitrate R_target, for which
> l = argmax{R(G_l) ≤ R_target}, will be |G_target| = |G_l| + (R_target − R(G_l))/(R(G_{l+1}) −
> R(G_l)) (|G_{l+1}| − |G_l|)."

> "Among the budget |ΔG| = |G_target| − |G_l|, we select the Gaussians having the highest gradient
> from the context C_{l+1}: ΔG = ∪_{i=1..|ΔG|} G_i, G_i ∈ C_{l+1}, ‖∂L/∂θ_i‖₂ ≥ ‖∂L/∂θ_j‖₂ ∀ i < j"

The critical design choice is that the ranking is **local to the context C_{l+1}**, not global. p4:
"In our approach, instead, we restrict the ranking and selection to a local context C_{l+1}. This
subtle change is crucial, as it ensures that the ranking reflects the contribution of the Gaussians
added at that specific target rate, resulting in a smooth RD curve with more coherently evolving
intermediate models." Fig. 4 shows the global-ranking variant producing an irregular RD curve.

The bounds are the anchor levels themselves. Below G₁ nothing is defined, and the fix offered is to
retrain with a smaller G₁ (p4).

### Can a size in bytes be requested directly?

**Formally yes, practically unverified.** R_target is named as a bitrate that may come from "network
bandwidth, storage budget, or hardware capabilities" (p3), and the figures plot size in MB, so the
rate axis is a byte axis. But:

- R(·) is never defined. The paper does not say whether R(G_l) is measured post-LZMA, estimated, or
  counted in raw parameters.
- The interpolation assumes rate is linear in Gaussian count between anchors. That assumption is
  never tested, and it will be violated by an entropy coder whose ratio depends on which Gaussians
  are present.
- **No achieved-versus-requested pair is reported anywhere in the paper**, so there is no evidence
  that requesting a specific byte count yields that byte count.

So RAVE gives a continuous rate dial with a byte-shaped label on it, and offers no measurement that
the dial is calibrated.

### Quality gap versus dedicated models

**No numeric gap is reported.** The only statements, p4:

> "our method achieves competitive performance in both PSNR and LPIPS compared to the most recent
> 3DGS compression techniques"

> "our method reaches state-of-the-art performance on Mip-NeRF 360, slightly outperforming HAC,
> while still being competitive on the other datasets"

Read against Fig. 3 axis ranges: on Mip-NeRF 360 RAVE is claimed to be at or slightly above HAC. On
Tanks and Temples and Deep Blending the word is "competitive", which in this literature usually
means at or slightly below. The size of that shortfall in dB is not stated, and cannot be recovered
from the paper without digitizing the figure.

---

# Cross-cutting synthesis

## Where each paper sits on the axis that matters

| | HybridGS | MesonGS++ | RAVE |
|---|---|---|---|
| target in bytes? | yes, B in MB | yes, S_T in bytes/MB | R_target, unit undefined |
| binds during training? | **partly**, from T_p = 36 000 of 70 000 | **no**, post-training only | **no**, at deployment |
| drives primitive growth? | **no**, densification ends at 15 000 | **no**, no densification at all | **no**, hierarchy is frozen |
| drives attribute precision? | yes, Method 2, and only as an alternative to Method 1 | yes, per-group bit widths via ILP | **no**, precision is fixed |
| both jointly? | **no**, the two methods are exclusive | **yes**, τ and Q jointly in one MINLP | no |
| controller can grow? | no | reconfigures, never grows past input | moves up and down a frozen ladder |
| target compared against entropy-coded bytes? | no, against n·P_bit/L | **yes**, actual size measured each loop | not stated |
| hits target? | −45 % to +8 %, and overshoots on dance | −0.3 % to −0.9 % reported | never measured |

## What survives of the claim

The three papers together do not overturn "no published method drives both primitive growth and
attribute precision onto an entropy-coded byte target from the start of training". They do force
three qualifications.

1. **"From the start of training" must be stated explicitly and defended.** HybridGS binds a byte
   target inside a training run, at iteration 36 000 of 70 000. Calling it "post-hoc" would be
   wrong. The correct distinction is that the target arrives after peak quality, after densification
   has finished, and it can only shrink.
2. **"Both growth and precision" is the load-bearing conjunction.** MesonGS++ jointly optimizes
   count (τ) and precision (Q) under one byte constraint, which is more than SizeGS or GETA-3DGS,
   but it does so on a frozen checkpoint with no densification anywhere in the loop. HybridGS has a
   growth phase and a precision knob but never couples them to the target at the same time. Its two
   rate-control methods are mutually exclusive: Method 2 runs "with disabling pruning" (p6).
3. **"Hits the target" is the weakest claim to lean on for HybridGS, and the strongest for
   MesonGS++.** HybridGS overshoots on dance by 3.5 % to 8.0 % and undershoots on train by up to
   14 % with Method 1 and 45 % with Method 2. MesonGS++ lands within 0.3 % to 0.9 %, but only
   because it measures the real compressed size inside its search loop, which is exactly what a
   post-training codec can afford and a training-time controller cannot.

RAVE is not a competitor for this claim at all. It has no size model, no measured target accuracy,
and no numbers.
