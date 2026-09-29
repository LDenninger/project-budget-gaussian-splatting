# K. Count-denominated budget controllers

Reading notes for four count-budget papers. Every number below is **reported** by the paper
unless it is explicitly marked *computed*. Quotes are verbatim from the pymupdf text layer with
the printed page number in parentheses. Hyphenation introduced by line breaks in the PDF has
been rejoined. Table rows quoted raw where the column order was ambiguous.

Source PDFs:
- `references/03_count_budget/constrained_dynamic_gs_zheng_2026_arxiv2602.03538.pdf`
- `references/03_count_budget/controlgs_zhang_2025_arxiv2505.10473.pdf`
- `references/03_count_budget/matryoshka_gs_guo_2026_arxiv2603.19234.pdf`
- `references/03_count_budget/flexgs_liu_cvpr2025_arxiv2506.04174.pdf`

---

## Quick classification

| paper | handle | TARGET or KNOB | budget binds from | overshoot | denominated in |
|---|---|---|---|---|---|
| CDGS | Ntarget, an integer Gaussian count | **TARGET** (hit error ≤ 1.4 % reported) | iteration 501 of 40 000, after a 500-iteration warm-up | none, count rises monotonically to the target from below | count, MB reported separately after compression |
| ControlGS | λα, a loss weight in [1e-7, 1e-6] | **KNOB** (no count is requested, count emerges) | from iteration 0, as an L1 opacity term | yes, uniform 8× splits then prune back | count only, no MB anywhere |
| MGS (Matryoshka) | k, a prefix length chosen at render time | **TARGET at deploy, none at train** | never during training, N is fixed at 5 M by 3DGS-MCMC | not applicable | count (splats) and FPS, no MB |
| FlexGS | e, a fraction of the original Gaussian count | **TARGET at deploy** (0.20 requested → ≈ 19.5 % achieved, then top-k forced) | iteration 15 001 of 35 000, second stage only | not applicable, monotone selection | count as a proxy for "model size", no MB |

---

# 1. Constrained Dynamic Gaussian Splatting (CDGS)

## 1. Citation
- Title as printed: **Constrained Dynamic Gaussian Splatting**
- First author: **Zihan Zheng** (Shanghai Jiao Tong University). Co-authors Zhenlong Wu (byline)
  / Zhenglong Wu (affiliation footnote, spelled differently in the two places), Xuanxuan Wang,
  Houqiang Zhong, Xiaoyun Zhang, Qiang Hu, Guangtao Zhai, Wenjun Zhang.
- Venue as printed in the running head: **IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS FOR VIDEO
  TECHNOLOGY, VOL. XX, NO. XX, MONTH 20XX** (a preprint template, no issue assigned).
- arXiv id from the stamp on page 1: **arXiv:2602.03538v1 [cs.CV] 3 Feb 2026**. File name id
  matches: 2602.03538.

## 2. Base representation and renderer
Anisotropic 3D Gaussians with the standard 3DGS α-blending rasteriser (Kerbl et al. [24]).
gᵢ = {μᵢ, Rᵢ, sᵢ, αᵢ, fᵢ}, Σᵢ = RᵢsᵢsᵢᵀRᵢᵀ, colour from SH.

**Dynamic scenes** (free-viewpoint video, 4D). The Gaussian set is split into a time-invariant
static set S and a time-varying dynamic set D(t), G = S ∪ D(t), S ∩ D(t) = ∅. Dynamic Gaussians
carry a position vector fᵘᵢ, a rotation vector fᴿᵢ interpolated at time t, plus a learnable
activation window [tˢᵢ, tᵉᵢ] outside which opacity decays. Static Gaussians carry only a
lightweight per-frame global transform Tᵢ.

## 3. Budget handle
The user supplies **Ntarget, an absolute integer Gaussian count for the whole sequence**. It is a
hard TARGET, not a knob: the paper's headline claim is adherence with error < 2 %.

Achieved-versus-requested pairs, Table II page 8, "Validation of our precise control capability
over the total number of Gaussians. We report the average PSNR on the coffee martini sequence,
the number of static/dynamic/overall Gaussians, and the error ratio of Gaussian number."
Raw extracted rows, columns are Method / Target / PSNR / Static / Dynamic / Overall / Ratio:

```
Ex4DGS [23]   -      28.79  292.2k  47.7k  339.9k  -
Ours          100k   28.53   72.4k  27.6k   99.9k  0.1%
              200k   28.68  156.8k  41.0k  197.8k  1.1%
              300k   28.81  244.4k  51.4k  295.8k  1.4%
              400k   28.95  316.4k  81.2k  397.6k  0.6%
```

So: requested 100 000 → achieved 99 900 (0.1 % error); requested 200 000 → 197 800 (1.1 %);
requested 300 000 → 295 800 (1.4 %); requested 400 000 → 397 600 (0.6 %). Every achieved count
is **below** the request. No pair overshoots.

Two named operating points used throughout the paper (Table I caption, page 8): "Ours-l and
Ours-s are obtained by setting the target Gaussian numbers to 300,000 and 100,000, respectively."

Ablation of the budget loss, Table VI page 10, the Ratio column is the count error:
```
w/o Importance score  31.91  31.2  1.6%
w/o Fgeom/motion      31.97  31.5  1.3%
w/o Fperceptual       31.99  31.3  1.4%
λgm = 1.5             32.08  31.4  1.3%
λgm = 2.5             32.11  31.5  1.4%
w/o Budget loss       32.15  31.7  4.8%
w/o Adaptive allocation 31.96 33.2 1.3%
Ours full             32.14  31.5  1.3%
```
Dropping the budget loss raises the error from 1.3 % to 4.8 %, and the paper states "Without it,
the Gaussian count error increases by 3.5%." (page 10). Note that "w/o Budget loss" reports the
**highest** PSNR of the whole table, 32.15 dB versus 32.14 dB for the full method, so the budget
loss costs 0.01 dB and buys a 3.5 pp tighter count.

## 4. The control law in full

Constrained problem (Eq. 5, page 4):

    min_G  Lrender(G)    s.t.  |G| ≤ Ntarget

**(a) Differentiable counting.** Each Gaussian gets a continuous activation cᵢ ∈ [0,1] produced by
a temperature-controlled hard sigmoid over a learnable unified importance score Mᵢ (Eq. 8, page 4):

    cᵢ = clamp( (Mᵢ − 0.5)/τ_c + 0.5, 0, 1 )

The soft count (Eq. 6):

    N_p = ∑ᵢ cᵢ

**(b) Quadratic budget penalty** (Eq. 7, page 4):

    L_budget = (N_p − Ntarget)²

The paper calls this "a differentiable, Lagrangian-style penalty" (page 4), but λ_b is a fixed
constant, not a dual variable updated by ascent. λ_b = 1 × 10⁻⁷ for all sequences (page 7).

**(c) Opacity gating couples the gate to the image.** Effective opacity in the splatting equation
(Eq. 9, page 4):

    α̂ᵢ = cᵢ · αᵢ

"This modulation serves as a differentiable bridge: when cᵢ → 0, the Gaussian becomes transparent
and ceases to contribute to the image, physically aligning the soft deletion with the visual
output." (page 4). Gradients from L_render therefore reach Mᵢ through cᵢ.

**(d) Temperature anneal** (Eq. 10, page 4), an exponential decay over the enforcement window
[k_start, k_end]:

    τ_c(k) = τ_init · (τ_end / τ_init)^((k − k_start)/(k_end − k_start))

with τ_init = 1.0 and τ_end = 0.01. "Upon convergence, thresholding cᵢ yields an explicit active
set with size ≈ Ntarget." (page 4)

**(e) Unified importance score** (Eq. 11, page 5), N(·) is min-max normalisation to [0,1]:

    Mᵢ = N( λ_gm · F_geom/motion(gᵢ) + F_perceptual(gᵢ) ),   λ_gm = 2

Geometric and motion cues (Eq. 12, page 5), a weighted vector of five normalised terms:

    F_geom/motion(gᵢ) = w₁ᵀ · N[ ∇ᵘᵢ , α^max_i , d⁻¹ᵢ , λ_max(Σᵢ) , Moᵢ ]
    ∇ᵘᵢ = ‖∇_μ Lᵢ‖₂                      (positional gradient)
    α^max_i = max over t ∈ Tᵢ of αᵢ(t)   (peak, not mean, opacity, to keep transients)
    d⁻¹ᵢ = (1/|V|) ∑_v d⁻¹_{i,v}         (proximity, inverse mean depth)
    λ_max(Σᵢ)                            (spatial extent, largest covariance eigenvalue)
    Moᵢ = ‖Tᵢ‖₂                    if gᵢ ∈ S
        = ‖fᵘᵢ‖₂ + ‖fᴿᵢ‖₂           if gᵢ ∈ D(t)

Perceptual cues (Eq. 13, page 5):

    F_perceptual(gᵢ) = w₂ᵀ · N[ Iᵢ , Areaᵢ , Var⁻¹ᵢ ]
    Iᵢ = ∑_{v ∈ V} ‖I_v − I_v^{\{i}}‖₁     (photometric residual of leave-one-out)
    Areaᵢ = ∑_{v ∈ V} Area_{i,v}           (accumulated projected screen area)
    Var⁻¹ᵢ = Var_{v ∈ V}(c_{i,v})⁻¹        (multi-view consistency)

**(f) Interaction with densification: the count CAN grow while the controller is active.** This is
the closest structural analogue in the four papers to Taming 3DGS. From page 5:

  "The global target Ntarget is dynamically decomposed into a per-iteration sub-target via a
  quadratic schedule, as proposed in [26], throughout the training process. This decomposition
  guides a closed-loop controller for population evolution: (i) densification: Gaussians with
  high Mᵢ are cloned or split with higher probability, ensuring computational resources are
  allocated to perceptually salient regions by injecting detail where needed; (ii) pruning:
  Gaussians with low Mᵢ are targeted for removal via a random sampling strategy, gradually
  eliminating redundant or visually insignificant ones while preserving perceptually critical
  structures."

Reference [26] is Taming 3DGS, so the per-iteration sub-target schedule is **the Taming 3DGS
quadratic ramp**, wrapped around a differentiable importance score instead of a gradient heuristic.

**(g) Schedule and ramp shape (three phases, page 6, iteration counts on page 7).**
Total 40 000 iterations: "our three-phase training strategy consists of 500, 29500, and 10000
iterations, respectively, with Gaussian densification and pruning performed every 500 steps
during the second phase."
- **Phase I, iterations 1 to 500, warm-up.** "We begin with a short warm-up stage that initializes
  the representation without budget constraints." Only L_render. Static Gaussians with per-frame
  global transforms. No budget term at all in this window.
- **Phase II, iterations 501 to 30 000, budget enforcement.** L = L_render + λ_b L_budget + λ_r L_reg
  (Eq. 18, page 7), λ_b = 1e-7, λ_r = 1e-4. Densify/prune every 500 steps. Adaptive static/dynamic
  reallocation runs at the same interval. Gate temperature annealed 1.0 → 0.01 across this window.
- **Phase III, iterations 30 001 to 40 000, stabilisation and fine-tuning.** "Once the effective
  count N_p converges near Ntarget, we enter a stabilization phase. The soft masks cᵢ are
  explicitly binarized, and redundant Gaussians (where cᵢ = 0) are permanently culled to strictly
  satisfy the budget constraint. The remaining compact set is then fine-tuned solely with L_render
  and L_reg." (page 7). This final fine-tune is worth 0.28 dB, Table VII page 11:
  `w/o Fine-tuning 31.86 31.5` versus `Ours full 32.14 31.5`.

**Overshoot: explicitly none.** From page 10, on Fig. 7: "As observed, the total number of
Gaussians monotonically increases and converges exactly to the preset limits without overshooting,
demonstrating the stability and strictness of our budget enforcement strategy." Fig. 7 caption,
page 10: "Evolution of Gaussian counts under varying target budgets. The subplots correspond to
target limits of 1 × 10⁵ (Left), 2 × 10⁵ (Middle), and 3 × 10⁵ (Right). The curves illustrate the
growth trajectories of Static, Dynamic, and Total Gaussians. Under our constrained optimization
framework, the total number steadily increases and converges precisely to the predefined Target
Number."

This is the direct opposite of the "over-densify then sparsify" trajectory of GaussianSpa and
Mini-Splatting. CDGS approaches the budget **from below** and never exceeds it.

**(h) Adaptive static/dynamic split (Eqs. 14–17, page 6),** which decides *where* the budget goes:
histogram of motion magnitudes with B bins, smoothed, peaks found:

    H_T = smooth(hist(Tᵢ, B))
    P_T = { p ∈ B | H_T(p) > H_T(p ± 1) }
    {p_s, p_d} = argmax over pairs {p₁,p₂} ⊆ P_T, p₁ ≠ p₂ of ( H_T(p₁) + H_T(p₂) )
    τ_s = argmin over x ∈ {p_s, p_d} of H_T(x)          (the valley between the two modes)
    α_T = |{Tᵢ | Tᵢ < τ_s}| / |{Tᵢ}| ,  τ_motion = quantile({Tᵢ}, α_T)

Removing this module costs 0.18 dB and adds 1.7 MB (Table VI, page 10:
`w/o Adaptive allocation 31.96 33.2 1.3%` versus `Ours full 32.14 31.5 1.3%`).

## 5. What "size" or "count" means, and whether MB is reported
"Count" = number of Gaussian primitives for the **entire sequence**, static plus dynamic, not per
frame. Table II breaks it into Static / Dynamic / Overall.

**MB is reported**, and separately from the count. From page 7: "we adopt Peak Signal-to-Noise
Ratio (PSNR) and Structural Similarity Index (SSIM) [76] as quality metrics, along with model size
measured in megabytes for the entire sequence." The MB figure is the size **after** the dual-mode
hybrid compression stage (16-bit quantisation for μ, T, fμ; 8-bit for everything else; KD-tree
reorder plus predictive residual coding plus entropy coding for static; H.264/x264 at YUV 4:4:4,
"medium" preset, constant QP 20, I- and P-frames only, 3 reference frames, for dynamic).

The byte size is therefore **not itself controlled**. Only the count is targeted; MB falls out of
the compression stage. Table VII page 11 shows how much of the MB is compression rather than
count control:
```
w/o Compression          32.21   98.3
w/o Static compression   32.19   65.5
w/o Dynamic compression  32.16   64.3
w/o Outlier separation   29.74   26.2
Ours full                32.14   31.5
```
98.3 MB uncompressed → 31.5 MB compressed at the same count. So roughly two thirds of the final
byte figure is decided by the codec, not by Ntarget.

## 6. Runtime cost
Hardware (page 7): "Intel(R) Xeon(R) W-2245 CPU @ 3.90 GHz and an RTX 3090 graphics card."
40 000 iterations total.

Table V page 9, "Complexity comparison of our method with dynamic scene reconstruction and
compression methods", raw row order Encode(s) / Decode(s) / Train(h) / Render(ms), columns
ReRF / TeTriRF / 4DGC / Ex4DGS / Ours:
```
Encode(s)   246    219    810   -     16
Decode(s)   18.3   16.8   28.2  -     0.5
Train(h)    >100   5.2    4.2   1.2   1.0
Render(ms)  497    372    5.6   7.8   5.4
```
Training 1.0 h versus Ex4DGS 1.2 h. Render 5.4 ms per frame. Encode 16 s and decode 0.5 s for 300
frames. FPS in Table I: Ours-l 149 FPS, Ours-s 186 FPS on N3DV. GPU memory during training: **not
stated**.

## 7. Evidence on "budget from the start" versus "explore then sparsify"
CDGS argues **for** binding the budget during training and **against** train-then-prune, but it
never runs the head-to-head ablation that would settle the question. It has no "one training run,
many budgets" claim: each Ntarget is a separate training run.

Arguments made, all on pages 1 and 3:
- "they enforce limits via rigid pruning rather than integrating the budget constraint directly
  into the training loop as a differentiable objective. Consequently, the optimization trajectory
  remains unaware of the capacity limit, leading to suboptimal convergence." (page 1, about
  Taming 3DGS)
- "However, these approaches predominantly operate in a train-then-prune or post-processing
  manner. They lack the mechanism to precisely align the model size with a predefined hardware
  budget during training. Consequently, removing primitives from a converged model often leads to
  suboptimal solutions and unpredictable quality degradation, as the remaining Gaussians are not
  jointly optimized to compensate for the information loss." (page 3)

The only quantitative datum on trajectory shape is the warm-up ablation, Table VII page 11:
`w/o Initialization 30.90 31.8` versus `Ours full 32.14 31.5`. **A 1.24 dB loss from removing the
500-iteration unconstrained warm-up**, the largest single ablation in the paper. The paper
attributes it to the static/dynamic split rather than to the budget trajectory: "the
initialization stage provides a reliable Gaussian translation prior, which is critical for an
accurate static-dynamic allocation. Ablating this stage leads to a sub-optimal allocation that
hinders effective training, resulting in a performance drop of 1.24 dB." (page 11). Still, it is
direct evidence that **some** unconstrained exploration before the budget binds is worth more than
a full dB, which is the same direction as the GaussianSpa / Mini-Splatting finding.

## 8. Main quantitative results
**Mip-NeRF 360, Tanks and Temples and Deep Blending are not evaluated.** CDGS is a dynamic-scene
paper and uses N3DV, MeetRoom and Technicolor.

Table I, N3DV [50], all values **reported**. Columns PSNR (dB) / SSIM / Size (MB) / Render (FPS) /
Controllable. Strongest two baselines by PSNR are Ex4DGS (32.11) and STGS (32.05):
```
4DGS [21]      32.01  0.944  6270  72   ✕
STGS [22]      32.05  0.944  200   107  ✕
Ex4DGS [23]    32.11  0.945  115   128  ✕
Ours-l         32.14  0.946  31.5  149  ✓
Ours-s         31.83  0.944  6.8   186  ✓
```
Also in Table I for byte context: 4DGaussians 31.15 / 0.939 / 34 MB / 147 FPS; GIFStream 31.75 /
0.938 / 10 MB / 95 FPS; RD4DGS 29.66 / 0.917 / 11.1 MB / 100.9 FPS; Swift4D 31.79 / 0.944 / 30 MB
/ 128 FPS.

Table III, MeetRoom [75] and Technicolor [74], strongest two baselines Ex4DGS and STGS:
```
MeetRoom   STGS [22]     29.01  0.929  15.2   159
MeetRoom   Ex4DGS [23]   29.12  0.930  40.2   148
MeetRoom   Ours-l        29.18  0.931  10.4   165
MeetRoom   Ours-s        28.60  0.927   4.5   215
Technicolor STGS [22]    31.96  0.941  58.7   113
Technicolor Ex4DGS [23]  32.38  0.942  170.3  100
Technicolor Ours-l       32.41  0.942   48.0  125
Technicolor Ours-s       31.90  0.940   17.1  144
```

Table IV, BD-PSNR (dB) against 4DGC as anchor:
```
Dataset          ReRF   TeTriRF  4DGCPro  RD4DGS  Ours
N3DV [50]        -1.99  -1.12     0.08     1.33   1.90
MeetRoom [75]    -1.84  -0.86    -0.02     -      1.72
```

Narrative claims, page 8: "4DGS, STGS and Ex4DGS achieve reconstruction quality comparable to that
of our method, but they typically require hundreds to thousands of MB in model size to enable
dynamic scene reconstruction, whereas ours only requires 31.5 MB."

## 9. Stated limitations and future work
**None stated.** The paper has no Limitations section and no Future Work section. Section V
(Conclusion, page 11) is entirely a restatement of contributions. Nothing in the text is framed as
a limitation of CDGS itself.

## 10. Every sentence touching a size, byte, bitrate, memory or FPS target, rate control, budget, or edge/mobile deployment

Page 1 (abstract and introduction):
> "While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets."

> "In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training."

> "Our key insight is to introduce a differentiable budget controller as the core optimization driver."

> "To maximize the utility of this fixed budget, we further decouple the optimization of static and dynamic elements, employing an adaptive allocation mechanism that dynamically distributes capacity based on motion complexity."

> "Coupled with a dual-mode hybrid compression scheme, CDGS not only strictly adheres to hardware constraints (error <2%) but also pushes the Pareto frontier of rate-distortion performance."

> "Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits, achieving over 3× compression compared to state-of-the-art methods."

> "Index Terms—Dynamic Gaussian Splatting, Neural Rendering, Resource-Constrained Rendering, Immersive Media."

> "While offering unprecedented interactivity, the mass deployment of FVV is severely bottlenecked by the heterogeneity of end-user hardware. The strict memory and bandwidth limitations of edge devices stand in sharp contrast to the massive data requirements of high-fidelity volumetric content, making it crucial to maximize reconstruction quality under strict resource constraints."

> "However, dynamic extensions [21]–[23], [25] typically struggle with explosive growth in Gaussian counts, where unconstrained densification defies hardware limits."

> "For instance, Ex4DGS [23] reduces redundancy via decomposition but lacks explicit capacity control, relying on heuristics that yield unpredictable model complexity. Conversely, methods like Taming 3DGS [26] introduce budget control but are restricted to static scenes. More critically, they enforce limits via rigid pruning rather than integrating the budget constraint directly into the training loop as a differentiable objective. Consequently, the optimization trajectory remains unaware of the capacity limit, leading to suboptimal convergence."

> "Our key insight is that optimal constrained reconstruction requires integrating the budget directly into the training loop while dynamically balancing resources between static and dynamic components."

> "Departing from the conventional paradigm of uncontrolled growth, CDGS treats the Gaussian count as a strict budget. By optimizing the spatio-temporal distribution within this user-defined limit, it not only ensures precise controllability but also pushes the Pareto frontier of rate-distortion performance."

> "First, to enforce the target capacity, we introduce a differentiable budget controller."

> "This ensures that visually critical dynamic details are preserved even under tight budgets."

> "Instead of relying on heuristic ratios, we leverage distribution analysis to autonomously identify the natural boundary between static and dynamic components, ensuring that the limited Gaussian budget is invested where it contributes most to the rendering quality."

Page 2:
> "Third, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count (error<2%). Finally, to minimize storage footprint, a dual-mode hybrid compression strategy is tailored specifically for the decomposed static and dynamic streams."

> "Collectively, these innovations enable CDGS to deliver controllable, compact, and high-quality FVV representations tailored to arbitrary hardware specifications. Experimental results demonstrate a 3× model size reduction compared to state-of-the-art methods while maintaining comparable quality."

> "We reinterpret dynamic Gaussian splatting as a budget-constrained optimization problem, enabling controllable model complexity and predictable capacity."

> "We design a budget-consistent three-phase training scheme and a dual-mode hybrid compression pipeline that jointly enforce strict budget adherence while minimizing spatio-temporal redundancy."

> "However, this explicit nature introduces new challenges: the adaptive density control strategy allows the number of primitives to grow unboundedly, resulting in high and variable memory usage that often exceeds consumer-grade hardware capacities. This uncontrolled model complexity and unstable convergence pattern severely hinder the deployment of 3DGS under strict computational or memory budgets, necessitating more robust control mechanisms."

> "While deformation-based methods struggle with large topological changes, 4D spatio-temporal representations often incur significant memory costs and inherit the slow rendering speeds of implicit methods."

> Fig. 1 caption: "Left: Our CDGS leverages differentiable budget control for precise Gaussian number regulation, achieving adaptive static-dynamic allocation across varying target numbers and optimal rendering quality. Middle: Visual comparison with state-of-the-art methods, highlighting advantages in visual quality, model size, and Gaussian count controllability (second row: actual/target counts). Right: Superior rate-distortion performance and precise Gaussian number control of our approach, outperforming all prior works (e.g. 4DGS [21], STGS [22], Ex4DGS [23])."

Page 3:
> "Despite these advances, current dynamic 3DGS methods typically rely on heuristic-driven densification, allowing the Gaussian count to grow arbitrarily. This unconstrained model complexity results in millions of redundant primitives, burdening storage and rendering efficiency without proportional visual gains."

> "Deploying 3DGS on resource-constrained devices faces a fundamental conflict between limited hardware capacities and the standard optimization strategy, which minimizes error via unbounded densification."

> "However, these approaches predominantly operate in a train-then-prune or post-processing manner. They lack the mechanism to precisely align the model size with a predefined hardware budget during training. Consequently, removing primitives from a converged model often leads to suboptimal solutions and unpredictable quality degradation, as the remaining Gaussians are not jointly optimized to compensate for the information loss."

> "The most relevant work to ours is Taming 3DGS [26], which introduces a controlled growth schedule to maintain the Gaussian count near a target level."

> "Distinct from passive compression, our method integrates the target count directly into the training objective. By employing an adaptive static-dynamic allocation strategy and a differentiable budget controller, we ensure the model actively seeks the optimal configuration within the strict Gaussian budget, maximizing rendering fidelity while strictly adhering to the specified capacity limits."

> "given multi-view video inputs and a target Gaussian count Ntarget, our pipeline is driven by a Differentiable Budget Controller (Sec. III-B)."

> "and employs a dual-mode hybrid compression scheme (Sec. III-E) to minimize storage and transmission overhead for efficient deployment."

> Fig. 2 caption: "(Top) Three-Phase Pipeline: The training progresses from a Warm-up phase to establish foundational priors, through a Budget Enforcement phase where constraints are actively applied, to a final Fine-tuning phase that maximizes quality under the fixed count Ntarget."

Page 4:
> "our goal is to reconstruct a compact spatio-temporal Gaussian scene representation G that enables real-time rendering under a fixed capacity budget Ntarget."

> "Unlike prior dynamic Gaussian approaches that freely grow and prune Gaussians post-training, we explicitly constrain the representational capacity during optimization"

> "This constraint directly governs runtime memory, rendering cost, and even streaming bitrate. Our objective is therefore not the unconstrained best reconstruction, but the best reconstruction achievable under a fixed Gaussian-number budget, which forms the foundation for our differentiable population control and adaptive allocation strategies described next."

> "penalizes deviations from the target capacity through a differentiable budget loss."

> "To match the target capacity, we introduce a quadratic budget loss"

Page 5:
> Fig. 3 caption: "This score passes through a hard-sigmoid gate to estimate the effective Gaussian count Np, which is strictly regulated towards the target Ntarget via a quadratic budget loss Lbudget. Guided by this budget constraint, the closed-loop policy performs densification on high-importance Gaussians and pruning on low-importance ones to dynamically optimize capacity."

> "To determine which Gaussians should survive under a strictly constrained budget, we require a metric that evaluates the contribution of each primitive to the final reconstruction."

> "Guided by the importance score Mi and regulated by a predefined budget, our method performs densification and pruning in a unified framework. The global target Ntarget is dynamically decomposed into a per-iteration sub-target via a quadratic schedule, as proposed in [26], throughout the training process."

Page 6:
> "This closed-loop controller dynamically reallocates the Gaussian budget to the most informative spatial and temporal regions while keeping the overall count near Ntarget. Unlike heuristic grow-and-prune pipelines [23], [25], our differentiable control embeds capacity regulation directly into optimization, enabling stable training and precise runtime control over model complexity."

> "To address this and fully utilize the fixed Gaussian budget Ntarget, we introduce an Adaptive Static-Dynamic Allocation strategy designed to maximize efficiency under strict constraints."

> "To enforce temporal sparsity, the opacity αi(t) is designed to decay smoothly outside this interval, ensuring that a dynamic Gaussian only consumes computational resources when it effectively contributes to the rendering."

> "We allocate the Gaussian budget proportionally: regions identified as D(t) or frames exhibiting higher dynamic activity receive a denser representation budget, while static regions S retain compact but stable coverage."

> "Together, this joint mechanism effectively balances reconstruction fidelity and storage efficiency, achieving high-quality dynamic rendering under strict model capacity constraints."

Page 7:
> "We begin with a short warm-up stage that initializes the representation without budget constraints."

> "After warm-up, we activate the population controller (Sec. III-B) and introduce the budget loss Lbudget and regularization Lreg."

> "At each interval, the budget is dynamically redistributed between static and dynamic sets based on the updated motion priors."

> "Once the effective count Np converges near Ntarget, we enter a stabilization phase. The soft masks ci are explicitly binarized, and redundant Gaussians (where ci = 0) are permanently culled to strictly satisfy the budget constraint."

> "To further reduce storage and transmission cost under the fixed Gaussian budget, we adopt a dual-mode hybrid compression strategy tailored to the static and dynamic components introduced in Sec. III-C."

> "we adopt Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) [76] as quality metrics, along with model size measured in megabytes for the entire sequence. For comprehensive rate-distortion performance analysis, we apply Bjontegaard Delta PSNR (BD-PSNR) [77]. Rendering efficiency is gauged by frames per second (FPS)."

Page 8:
> "To validate our method's capacity for Gaussian count control, we further report the number of Gaussians and the error ratio relative to the target."

> "Ours-l and Ours-s are obtained by setting the target Gaussian numbers to 300,000 and 100,000, respectively."

> "Specifically, 4DGS, STGS and Ex4DGS achieve reconstruction quality comparable to that of our method, but they typically require hundreds to thousands of MB in model size to enable dynamic scene reconstruction, whereas ours only requires 31.5 MB."

> "More importantly, we can achieve precise control over the total number of Gaussians. As shown in Tab. II, under different target total numbers of Gaussians, we maintain the error rate within 2% and simultaneously achieve the optimal allocation of static and dynamic Gaussians."

Page 9:
> "Our CDGS exhibits significantly improved computational efficiency: its training time is 1.0 hour, compared to 1.2 hours for Ex4DGS [23], owing to the increased interval of Gaussian expansion. Additionally, experimental results on the N3DV dataset demonstrate that our approach enables high-speed rendering, achieving a rate of 5.4 ms per frame. For encoding and decoding, CDGS achieves times of 16 s and 0.5 s for 300 frames in total, respectively, outperforming all other methods by a significant margin."

Page 10:
> "Our method delivers reconstruction quality comparable to these approaches while operating at a significantly lower model size."

> "For instance, in our lightweight variant (ours-s), reconstructing the flame salmon sequence requires only 10k Gaussians, which is far fewer than the 35k used by Ex4DGS. This underscores that CDGS not only accurately captures dynamic scene content and retains high-fidelity details in complex objects but also achieves a highly compact model size, all while realizing precise regulation of the Gaussian count."

> Fig. 7 caption: "Evolution of Gaussian counts under varying target budgets. The subplots correspond to target limits of 1 × 105 (Left), 2 × 105 (Middle), and 3 × 105 (Right). The curves illustrate the growth trajectories of Static, Dynamic, and Total Gaussians. Under our constrained optimization framework, the total number steadily increases and converges precisely to the predefined Target Number."

> "Conversely, our method (Fig. 8 (d)) acts as a precise filter, concentrating the Gaussian budget solely on perceptually salient object boundaries."

> "Furthermore, the budget loss is proven critical for regulating the score distribution during training, ensuring the Gaussian count aligns with the target. Without it, the Gaussian count error increases by 3.5%. This precise controllability is visually corroborated in Fig. 7. As observed, the total number of Gaussians monotonically increases and converges exactly to the preset limits without overshooting, demonstrating the stability and strictness of our budget enforcement strategy."

Page 11:
> "The necessity of this module is quantitatively confirmed in Tab. VI, whose removal leads to a -0.18 dB drop in PSNR and a slight model size increase of 1.7 MB, underscoring its role in both compactness and fidelity."

> "As summarized in Tab. VII, this approach reduces the model size by 34.0 MB for static and 32.8 MB for dynamic Gaussians, with a marginal impact on rendering quality."

> "Specifically, the 'After' distribution highlights the successful isolation of 'Background' static Gaussians and 'Outlier' dynamic data, ensuring that the majority of the storage budget is precisely allocated to the perceptually critical foreground and motion details, thereby validating our separation strategy."

> "CDGS demonstrates that integrating hardware constraints directly into the training loop is essential for achieving optimal fidelity."

> "Furthermore, coupled with a dual-mode hybrid compression scheme, CDGS ensures that limited capacity is intelligently invested in visually critical motion details while minimizing storage footprint. Extensive experiments validate that CDGS achieves superior rate-distortion performance, offering a robust and practical solution for deploying high-fidelity 4D immersive media on resource-constrained edge devices."

## 11. Code URL and licence
**No code URL and no project page appear anywhere in the PDF.** No licence is stated.

---

# 2. ControlGS

## 1. Citation
- Title as printed: **ControlGS: Consistent Structural Compression Control for Deployment-Aware
  Gaussian Splatting**
- First author: **Fengdi Zhang** (Shenzhen International Graduate School, Tsinghua University, and
  Pengcheng Laboratory). Co-authors Yibao Sun, Hongkun Cao, Ruqi Huang.
- Venue as printed in the running head: **JOURNAL OF LATEX CLASS FILES, VOL. XX, NO. XX, XX 20XX**
  (an unassigned IEEE journal template, footer "0000–0000/00$00.00 © 20XX IEEE"). No accepted venue
  is stated.
- arXiv id from the stamp on page 1: **arXiv:2505.10473v3 [cs.CV] 7 Nov 2025**.

## 2. Base representation and renderer
Vanilla 3DGS, anisotropic Gaussians with SH colour and α-blending, implemented on top of the 3DGS
codebase (page 5): "Our method is implemented on top of the 3DGS framework [1]. We follow default
3DGS settings for data loading, parameter initialization, learning rate scheduling, optimizer
selection, dynamic SH degree promotion, and rendering, with exposure compensation disabled."

**Static scenes only.** Extension to dynamic scenes is listed as future work.

## 3. Budget handle
The user supplies **λα, a single scalar loss weight**, recommended range [1e-7, 1e-6]. This is a
**KNOB, not a TARGET.** No count and no byte size is requested, and none is guaranteed. The paper
is explicit that this is the point: it argues that requesting a count is the wrong interface.

There are therefore **no achieved-versus-requested pairs to copy**, because nothing is requested.
What the paper offers instead is a monotone, cross-scene-consistent map λα ↦ count. From Table II
page 8 (Mip-NeRF360, Tanks & Temples, Deep Blending) and Table III page 8 (NeRF synthetic,
GigaNVS), the Num(M) column at each λα, all **reported**:

| λα | Mip-NeRF360 | Tanks & Temples | Deep Blending | NeRF synthetic | GigaNVS |
|---|---|---|---|---|---|
| 1e-7 | 1.76 M | 1.68 M | 0.90 M | 0.59 M | 3.38 M |
| 2e-7 | 1.10 M | 1.10 M | 0.61 M | 0.35 M | 2.05 M |
| 3e-7 | 0.83 M | 0.85 M | 0.47 M | 0.26 M | 1.50 M |
| 5e-7 | 0.56 M | 0.62 M | 0.33 M | 0.18 M | 0.95 M |
| 1e-6 | 0.31 M | 0.34 M | 0.19 M | 0.11 M | 0.32 M |

The same λα maps to counts spanning 0.11 M to 3.38 M across dataset families, a ratio of roughly
31× (*computed*). Consistency is a claim about the **shape** of the response, not about the
absolute count.

## 4. The control law in full
The motivation section formalises the mechanism as a balance of two objectives (Eqs. 1–3, page 3):

    Ψ(N; λ) = F_D(N) + λ F_P(N)
    N_eq(λ) = argmin over N ∈ [0, N_max] of Ψ(N; λ)

"When λ → 0, densification dominates and N_eq → N_max; when λ is large, pruning dominates and
N_eq → 0. Thus, N_eq(λ) is a continuous and monotonically decreasing function of λ. By the
Intermediate Value Theorem, for any target N⋆ ∈ [N⋆_min, N⋆_max], there exists a unique
λ⋆ ∈ [λ⋆_min, λ⋆_max] such that N_eq(λ⋆) = N⋆." (page 3)

Cross-scene consistency is obtained by making D and P scene-independent (Eq. 3, page 3):

    F_D(N, S) ≈ F_D(N),   F_P(N, S) ≈ F_P(N)

**(a) Densification: uniform octree splitting.** Not a subset selected by gradient heuristics.
Every existing Gaussian is split into 8 children per densification step (Eqs. 6–8, pages 4–5):

    p_child,i = p_parent + R_parent (Δᵢ ⊙ S_parent),   Δᵢ ∈ {±0.25}³
    S_child = S_parent / 1.6
    α_child = 1 − √(1 − α_parent)      solving (1 − α_child)² = 1 − α_parent

R_child and c_child are inherited verbatim from the parent. Splitting is done in batches of
N_batch = 100 000 Gaussians "To avoid memory overflow from splitting too many Gaussians at once"
(page 4), with a pruning pass between batches.

**(b) Pruning: opacity L1 sparsification** (Eq. 9, page 5):

    L_α = ∑ᵢ |αᵢ|

plus a hard threshold: "we further set a small opacity threshold τα and periodically remove
Gaussians with αᵢ < τα, thereby transitioning from soft sparsity (Lα) to hard pruning (α < τα)."
τα = 0.005, pruning every 100 iterations (page 6).

**(c) The single knob** (Eq. 10, page 5):

    L = L_RGB + λα L_α

with L_RGB = (1 − λ_w) L₁ + λ_w L_D-SSIM, λ_w = 0.2.

**(d) Schedule: an event-driven, not iteration-driven, rhythm.** From page 5:

  "we adopt a optimize (with pruning) → split → re-optimize → re-split rhythm. Training begins with
  an SfM-initialized Gaussian set and standard optimization at the current resolution. During
  optimization, we periodically record the number of Gaussians removed due to αᵢ < τα, denoted
  Nremove. If Nremove remains below a threshold τremove, it indicates that redundant Gaussians have
  been pruned and the model has nearly converged at this scale. At this point, we perform uniform
  splitting, resume optimization, and trigger the next split once Nremove < τremove again. This
  iterative process drives the model along a 'first prune, then refine' trajectory, with the rhythm
  determined by sparsification progress rather than scene-specific tuning."

Algorithm 1 (page 5) confirms the details: prune only when `ISPRUNESTEP(t) and t ≥ tuntil`; on a
split, `tuntil ← t + tdelay` delays pruning; once `Nsplit ≥ τsplit`, `λα ← 0` disables the
regulariser for the remainder of training.

Concrete values (page 6): prune every 100 iterations, τα = 0.005, trigger a split when
N_remove < 2000, 100k Gaussians per split batch, pruning delayed 200 iterations after each split,
maximum 6 splitting rounds, then λα is set to zero until the 100k-iteration cap.

**(e) When does the budget bind, and is overshoot allowed?** The λα L_α term is in the loss from
iteration 0. There is no warm-up window. But **overshoot is the whole mechanism**: each uniform
split multiplies the population by 8, and the sparsifier then prunes it back. From page 5: "uniform
splitting provides unbiased expansion, while opacity-based sparsification enforces demand-driven
contraction. Together, under a single hyperparameter λα, they form a stable, scene-independent
closed loop: first expanding the candidate space to capture detail, then retracting redundant
parts." Fig. 7 (page 9) shows this at iterations 399 / 400 / 600 and 799 / 800 / 1000:
"Illustration of the self-correcting mechanism. Uniform splitting may over-split, whereas
opacity-based sparsification prunes non-contributing Gaussians, restoring sparsity."

So ControlGS is squarely an **explore-then-sparsify** method, repeated 6 times. The overshoot
factor is up to 8× the current population per round, bounded in practice by the batch size of 100k
and the interleaved pruning. The paper does not report the peak count.

**(f) The count grows while the controller is active.** Yes, by construction, in 6 discrete
expansion rounds. The final round ends with λα = 0, so the last stretch of training is pure
reconstruction with a frozen population.

## 5. What "size" or "count" means, MB, and "deployment-aware"
"Count" is the number of Gaussian primitives, reported as Num(M) in millions.

**MB is never reported.** There is no byte figure, no storage figure and no bitrate anywhere in the
paper. The word "compression" in the title means **structural** compression only, that is, fewer
primitives. From page 2: "Structural compression [6]–[12], [14], [15] focuses on reducing the
number of Gaussians to fundamentally shrink model size." Attribute compression is listed as future
work: "(1) integrating attribute compression for greater compactness" (page 10). Baselines marked
with `*` in Tables II and III "further employ attribute compression", so ControlGS is compared
against byte-compressed models on a count axis only.

**What "deployment-aware" means concretely.** It does **not** mean the user supplies a device
budget in bytes or ms. It means three things, in this order:
1. Adjusting the trade-off requires **no per-scene tuning**. One λα value works across object,
   indoor, outdoor and large-outdoor scenes. Page 1: "it requires a universal, consistent control
   mechanism that adjusts the trade-off between rendering quality and model compression without
   scene-specific tuning, enabling automated deployment across different device performances and
   communication bandwidths."
2. The control axis lands the model in the "efficient regime", the second of four phases the paper
   identifies in the count-quality curve (underfitting, efficient regime, saturation, overfitting).
   Page 1: "This structure suggests that control is most effectively applied within the efficient
   regime, where the balance between resources and rendering quality is most favorable and
   rendering quality is most responsive to changes in Gaussian count."
3. A **hardware tier is mapped to a λα value once**, not per scene, and the resulting model clears
   an FPS floor. This is the closest the paper gets to binding a device budget.

**What the single hyperparameter controls.** From page 5: "A smaller λα favors retaining more
split candidates and leads to higher rendering quality, while a larger λα enforces stronger
sparsification and yields a more compact representation." It is the relative weight of the opacity
L1 term against the RGB reconstruction loss. It does not set a count, it sets a **price per unit of
opacity**, and the count is whatever equilibrium that price produces.

**Evidence of cross-scene consistency, Table II page 8, quoted raw.** Columns are
PSNR↑ SSIM↑ LPIPS↓ Num(M) repeated for Mip-NeRF360, Tanks & Temples, Deep Blending:
```
ControlGS (Ours)
λα=1e-7   28.15 0.831 0.195 1.76 | 24.64 0.869 0.140 1.68 | 30.08 0.911 0.240 0.90
λα=2e-7   28.08 0.827 0.209 1.10 | 24.41 0.863 0.152 1.10 | 29.96 0.910 0.248 0.61
λα=3e-7   27.90 0.821 0.221 0.83 | 24.35 0.857 0.162 0.85 | 29.81 0.907 0.257 0.47
λα=5e-7   27.70 0.810 0.242 0.56 | 24.15 0.849 0.176 0.62 | 29.64 0.900 0.273 0.33
λα=1e-6   27.10 0.780 0.284 0.31 | 23.61 0.828 0.207 0.34 | 29.14 0.889 0.295 0.19
```
And Table III page 8 (NeRF synthetic PSNR/Num, GigaNVS PSNR/SSIM/LPIPS/Num):
```
λα=1e-7   33.85 0.59 | 19.91 0.763 0.260 3.38
λα=2e-7   33.81 0.35 | 19.72 0.743 0.284 2.05
λα=3e-7   33.61 0.26 | 19.59 0.723 0.307 1.50
λα=5e-7   33.50 0.18 | 19.30 0.688 0.345 0.95
λα=1e-6   33.10 0.11 | 18.63 0.601 0.434 0.32
```
Across all five dataset families the count is **monotonically decreasing in λα** and PSNR is
**monotonically decreasing in λα**, with no inversions in any of the 25 cells. The claimed
behaviour, page 6: "The analysis reveals that ControlGS exhibits consistent and predictable control
behavior across all scenes. As λα increases, it smoothly strengthens the structural compression
with high responsiveness, leading to a monotonic decrease in Gaussian count and a corresponding
decline in rendering quality, without stagnation or abrupt fluctuations."

## 6. Runtime cost
Hardware (page 5): "Experiments are conducted on an Intel Core i9-10980XE CPU and an NVIDIA RTX
3090 GPU." Fig. 3 caption (page 6): "All methods are trained for 100k iterations."

**Training wall-clock time is not stated. GPU memory is not stated.** An OOM is reported only as an
ablation failure mode (page 10): "Replacing opacity-based sparsification with the original
opacity-reset-based pruning causes out-of-memory (OOM) conditions."

Rendering FPS, Table I page 7, measured in a browser client on integrated GPUs. Raw table:
```
GPU Tier / Instance | λα      1e-7   2e-7   3e-7   4e-7   5e-7   6e-7   7e-7
High-end iGPU (AMD Radeon 780M)
  Bicycle (Outdoor)          42.2   56.2   65.8   77.5   78.7   88.6   96.3
  Truck (Outdoor)            43.8   51.2   60.0   63.0   68.0   73.8   77.4
  Bonsai (Indoor)            41.3   51.2   55.1   60.9   67.6   66.9   70.8
  Room (Indoor)              44.5   51.4   57.0   61.0   62.9   65.8   68.8
  Hotdog (Object)            74.0   94.0  106.2  116.0  118.0  >120   >120
  Lego (Object)              55.6   96.4  >120   >120   >120   >120   >120
Mid-range iGPU (Intel UHD Graphics 770)
  Bicycle (Outdoor)          13.9   21.1   26.4   31.7   34.1   39.0   44.0
  Truck (Outdoor)            11.9   17.9   22.5   25.1   26.6   27.7   29.6
  Bonsai (Indoor)            15.6   20.3   23.6   26.2   29.0   29.5   31.8
  Room (Indoor)              17.1   21.3   24.7   27.7   28.1   30.0   31.2
  Hotdog (Object)            37.5   45.9   49.8   54.0   54.7   58.5   59.6
  Lego (Object)              28.3   41.8   50.0   58.5   65.9   73.0   74.9
Entry-level iGPU (AMD Radeon RX Vega 7)
  Bicycle (Outdoor)          17.1   21.5   24.2   28.9   29.7   32.9   37.0
  Truck (Outdoor)            16.1   18.6   21.8   23.1   24.5   26.4   28.2
  Bonsai (Indoor)            14.4   17.8   20.2   21.8   24.3   24.7   26.2
  Room (Indoor)              15.9   18.8   20.6   22.0   22.5   24.2   25.2
  Hotdog (Object)            28.1   35.3   39.2   44.9   45.4   47.2   48.1
  Lego (Object)              28.1   39.3   48.0   56.4   57.1   59.3   60.4
```
Note the mid-range iGPU is **slower than the entry-level one** at λα = 1e-7 on Bicycle (13.9 vs
17.1) and Truck (11.9 vs 16.1). The tier labels do not track the measurements at the heavy end.

**This is the one place in the four papers where a device-side FPS floor is used to pick a control
value.** Page 7: "Results show that assigning λα=1e-7, 4e-7, and 7e-7 to the respective tiers
yields stable rendering above 25 frames per second (FPS), a typical lower bound for perceptually
smooth playback [42], without any scene-specific Gaussian budget tuning." Reading Table I against
that claim: high-end at 1e-7 gives 41.3 to 74.0 FPS, mid-range at 4e-7 gives 25.1 to 58.5 FPS, and
entry-level at 7e-7 gives 25.2 to 60.4 FPS. All ≥ 25. The λα per tier is picked **once, by hand,
offline**, not solved for at training time.

## 7. Evidence on "budget from the start" versus "explore then sparsify"
ControlGS is an explore-then-sparsify method (uniform 8× splitting followed by opacity pruning,
repeated up to 6 rounds), and it beats both the budget-based baseline 3DGS-MCMC and the
non-budget-based LP-3DGS on the count-quality frontier. This is indirect support for the
project's finding that trajectory shape matters, in the same direction as GaussianSpa and
Mini-Splatting.

The paper's own argument against count-targeting, page 1: "Budget-based methods constrain only the
hard-coded Gaussian count budget or similar counterparts, with no guarantee that training converges
to the efficient regime. In practice, this also requires repeated trial-and-error per scene to
determine the appropriate Gaussian budget [8]–[13], which is impractical for automated deployment."
Note that references [8]–[13] include Mini-Splatting, GaussianSpa and Taming 3DGS, so ControlGS
places all three in the same "budget-based" family it argues against.

And page 6: "As a budget-based method, 3DGS-MCMC [13] requires manual specification of the exact
Gaussian count, making cross-scene consistent compression control infeasible. In practice, it
demands per-scene parameter tuning to achieve optimal results; otherwise, it easily falls into
underfitting or overfitting regions, leading to a significant performance drop."

**No "one training run, many budgets" claim.** Each λα is a separate 100k-iteration training run.
ControlGS gives one operating point per run. Ten λα values means ten runs.

The mechanism explanation for why the coarse-to-fine trajectory helps, page 9: "alternating between
uniform splitting and optimization drives a progressive refinement along the frequency dimension:
Large Gaussians generated early in training capture global low-frequency structures, while
later-introduced smaller Gaussians refine local high-frequency details, forming a coarse-to-fine
process that improves Gaussian utilization efficiency. At the same time, each child Gaussian
inherits spatial and appearance parameters from its parent, ensuring cross-stage stability and
naturally forming a top-down hierarchical structure."

## 8. Main quantitative results
Table II page 8, all **reported**. Strongest two baselines by PSNR on each dataset are vanilla
3DGS and LP-3DGS (ρ=0.4 / ρ=0.2), so those are quoted alongside:

**Mip-NeRF360 (Indoor/Outdoor)**, columns PSNR↑ SSIM↑ LPIPS↓ Num(M):
```
3DGS [1]              27.63  0.814  0.222  2.63
LP-3DGS ρ=0.2         27.74  0.814  0.217  2.54
LP-3DGS ρ=0.4         27.74  0.814  0.218  1.90
GoDe-M* LoD6          27.42  0.815  0.263  1.55
ControlGS λα=1e-7     28.15  0.831  0.195  1.76
ControlGS λα=2e-7     28.08  0.827  0.209  1.10
ControlGS λα=3e-7     27.90  0.821  0.221  0.83
```
**Tanks & Temples (Outdoor)**:
```
3DGS [1]              23.70  0.853  0.171  1.58
LP-3DGS ρ=0.4         24.23  0.855  0.167  1.10
LP-3DGS ρ=0.2         24.21  0.854  0.167  1.47
GoDe-M* LoD6          23.97  0.842  0.220  0.94
ControlGS λα=1e-7     24.64  0.869  0.140  1.68
ControlGS λα=2e-7     24.41  0.863  0.152  1.10
```
**Deep Blending (Indoor)**:
```
3DGS [1]              29.88  0.908  0.242  2.48
CompGS* [33]          29.90  0.907  0.251  0.55
EAGLES* [7]           29.83  0.910  0.246  1.20
ControlGS λα=1e-7     30.08  0.911  0.240  0.90
ControlGS λα=2e-7     29.96  0.910  0.248  0.61
```
Summary claim, page 7: "For example, compared to 3DGS, ControlGS achieves 28.08 dB with 1.10M
Gaussians on Mip-NeRF360 (a 58.1% reduction and a 0.45 dB gain), 24.41 dB with 1.10M on Tanks and
Temples (a 30.4% reduction and a 0.71 dB gain), and 30.08 dB with 0.90M on Deep Blending (a 63.7%
reduction and a 0.20 dB gain)."

Ablation, Table IV page 10, Mip-NeRF360 at λα = 3e-7:
```
3DGS [1]                            27.63  0.814  0.222  2.63
w/o Uniform Splitting               27.55  0.803  0.253  0.51
w/o Attribute Inheritance           27.36  0.821  0.202  1.13
w/o Opacity-based Sparsification    OOM
Full                                27.90  0.821  0.221  0.83
```

## 9. Stated limitations and future work
**No Limitations section.** Section VI is titled "DISCUSSION AND CONCLUSIONS". The only
forward-looking statement, verbatim, page 10:

> "Future research directions include: (1) integrating attribute compression for greater
> compactness; (2) extending to dynamic scenes and video reconstruction; and (3) leveraging
> ControlGS as a general framework for broader scene representation methods based on explicit
> primitives."

The nearest thing to an admitted weakness is the ablation finding, page 10:

> "Replacing opacity-based sparsification with the original opacity-reset-based pruning causes
> out-of-memory (OOM) conditions. Without effective pruning, low-contributing Gaussians accumulate
> and, combined with uniform splitting, lead to uncontrolled growth that degrades training
> efficiency, increases memory consumption, and ultimately causes training to fail."

## 10. Every sentence touching a size, byte, bitrate, memory or FPS target, rate control, budget, or edge/mobile deployment

Page 1:
> "3D Gaussian Splatting (3DGS) is a highly deployable real-time method for novel view synthesis. In practice, it requires a universal, consistent control mechanism that adjusts the trade-off between rendering quality and model compression without scene-specific tuning, enabling automated deployment across different device performances and communication bandwidths."

> "Extensive experiments across a wide range of scene scales and types (from small objects to large outdoor scenes) demonstrate that, by adjusting a globally unified control hyperparameter, ControlGS can flexibly generate models biased toward either structural compactness or high fidelity, regardless of the specific scene scale or complexity, while achieving markedly higher rendering quality with the same or fewer Gaussians compared to potential competing methods."

> "Index Terms—Novel view synthesis, 3D Gaussian splatting, structural compression, controllable compression, cross-scene consistency, Gaussian count–rendering quality trade-off."

> "making it a highly deployable method for real-time NVS across a wide range of computing devices, such as smartphones [2], VR/AR headsets [3], [4], and edge devices [5]. However, to cover fine details and preserve fidelity, 3DGS typically requires millions of Gaussians, causing the model size to expand dramatically, which in turn brings storage overhead, computational burden, and deployment challenges."

> "However, in real applications, compression alone is insufficient. Different scenarios, hardware, and deployment goals require different balances between model size and fidelity. Therefore, compared to static compression, a controllable mechanism that allows flexible adjustment between the two is better aligned with practical needs."

> "This structure suggests that control is most effectively applied within the efficient regime, where the balance between resources and rendering quality is most favorable and rendering quality is most responsive to changes in Gaussian count, ensuring that control remains both meaningful and impactful."

> "Budget-based methods constrain only the hard-coded Gaussian count budget or similar counterparts, with no guarantee that training converges to the efficient regime. In practice, this also requires repeated trial-and-error per scene to determine the appropriate Gaussian budget [8]–[13], which is impractical for automated deployment."

Page 2:
> "Structural compression [6]–[12], [14], [15] focuses on reducing the number of Gaussians to fundamentally shrink model size."

> "Attribute compression includes adding neural components [24]–[27], simplifying SH [15], [28]–[31], applying quantization [15], [27]–[29], [31]–[36], and using entropy coding [27], [31], [34] to reduce the storage overhead of each Gaussian's attributes."

> "While 3DGS offers clear advantages in speed and rendering quality, its explicit representation leads to high storage overhead, now a key bottleneck."

> Fig. 2 caption: "A smaller λα retains more candidates for higher rendering quality, whereas a larger λα enforces stronger sparsification with fewer Gaussians, enabling consistent structural compression control across scenes."

Page 3:
> "structural compression control aims to adjust the preference between Gaussian count and rendering quality by tuning hyperparameters during training or post-processing, enabling deployable models tailored to specific resource or application needs. Existing approaches fall into budget-based and non-budget-based categories. Budget-based methods [8]–[13] prune less important Gaussians using a hard-coded Gaussian count budget or similar counterparts, often requiring repeated tuning for specific scenes to select a suitable model. This limits their practicality, as such tuning processes are impractical in automated deployment."

> "By tuning λ within this interval, continuous and predictable preference adjustment can be achieved in the efficient regime."

Page 4:
> "To avoid memory overflow from splitting too many Gaussians at once, we perform the uniform splitting in batches by randomly selecting Nbatch Gaussians without replacement. After each batch is split, a brief optimization phase prunes redundant Gaussians to free memory."

Page 5:
> "A smaller λα favors retaining more split candidates and leads to higher rendering quality, while a larger λα enforces stronger sparsification and yields a more compact representation. Thus, adjusting λα can navigate the Gaussian count–rendering quality trade-off in a scene-agnostic manner, enabling consistent structural compression control across scenes."

> "With a single knob, the model can consistently and predictably transition across scenes between a more compact representation and higher rendering quality."

Page 6:
> Fig. 3 caption: "3DGS-MCMC fits the overall Gaussian count–quality curve with scene-dependent budgets (50k–100k for NeRF Synthetic, 100k–700k for others). ControlGS consistently reaches the efficient regime and delivering comparable or higher PSNR with fewer Gaussians than LP-3DGS, while LP-3DGS saturates early and 3DGS-MCMC requires scene-specific tuning."

> "As a budget-based method, 3DGS-MCMC [13] requires manual specification of the exact Gaussian count, making cross-scene consistent compression control infeasible."

Page 7:
> "We further deployed ControlGS models trained with different λα values to a browser-based client runtime (see our project page) and tested their real-time rendering performance on three widely used integrated GPUs representing high-, mid-, and low-end hardware tiers, as summarized in Table I. Results show that assigning λα=1e-7, 4e-7, and 7e-7 to the respective tiers yields stable rendering above 25 frames per second (FPS), a typical lower bound for perceptually smooth playback [42], without any scene-specific Gaussian budget tuning. These results demonstrate the potential of ControlGS for automated cross-device deployment, enabled by its consistent structural compression control across diverse scenes."

> Table I caption: "Rendering speed (FPS) of our method under different λα on integrated GPUs (iGPUs): high-end (AMD Radeon 780M), mid-range (Intel UHD Graphics 770), and entry-level (AMD Radeon RX Vega 7)."

> "Overall, ControlGS provides a more efficient scene representation and clearly outperforms existing methods in the trade-off between Gaussian count and rendering quality, forming a better Pareto frontier where similar quality is achieved with fewer Gaussians, or higher quality is achieved with a comparable model size."

Page 8:
> Table II caption: "Comparison with non-budget-based structural compression methods on Mip-NeRF360, Tanks & Temples, and Deep Blending datasets using PSNR, SSIM, LPIPS, and Gaussian count in millions."

Page 10:
> "Replacing opacity-based sparsification with the original opacity-reset-based pruning causes out-of-memory (OOM) conditions. Without effective pruning, low-contributing Gaussians accumulate and, combined with uniform splitting, lead to uncontrolled growth that degrades training efficiency, increases memory consumption, and ultimately causes training to fail."

> "In this work, we present ControlGS, a cross-scene consistent structural compression control framework for 3DGS automated deployment."

> "Future research directions include: (1) integrating attribute compression for greater compactness; (2) extending to dynamic scenes and video reconstruction; and (3) leveraging ControlGS as a general framework for broader scene representation methods based on explicit primitives."

> "In summary, ControlGS offers a controllable, broadly applicable, high-performance solution for 3DGS structural compression control. With a simple interface, consistent cross-scene behavior, and strong performance, it enhances the real-world deployment of 3DGS models across varying hardware and bandwidth constraints."

## 11. Code URL and licence
Printed on page 1: **"Project page: https://zhang-fengdi.github.io/ControlGS."** A browser-based
client runtime is referenced on page 7 ("see our project page"). **No licence is stated** other
than the template footer "0000–0000/00$00.00 © 20XX IEEE".

---

# 3. Matryoshka Gaussian Splatting (MGS)

## 1. Citation
- Title as printed: **Matryoshka Gaussian Splatting**
- First author: **Zhilin Guo** (University of Cambridge). Co-authors Boqiao Zhang, Hakan Aktas,
  Kyle Fogarty, Jeffrey Hu, Nursena Koprucu Aslan, Wenzhao Li, Canberk Baykal, Albert Miao, Josef
  Bengtson (Chalmers), Chenliang Zhou, Weihao Xia (corresponding), Cristina Nader Vasconcelos
  (Google), Cengiz Oztireli (Cambridge and Google).
- Venue: **not stated.** The PDF uses the Springer LNCS style with no conference name. Only the
  arXiv stamp appears.
- arXiv id from the stamp on page 1: **arXiv:2603.19234v2 [cs.CV] 20 Mar 2026**.

## 2. Base representation and renderer
Standard 3DGS primitives, gᵢ with mean μᵢ ∈ ℝ³, covariance Σᵢ, opacity σᵢ ∈ [0,1] and SH colour.
Front-to-back α-compositing of depth-sorted Gaussians.

Implementation (page 8): "We implement MGS on gsplat [41] codebase using 3DGS-MCMC [17] training
strategy." So the underlying densification and capacity control is **3DGS-MCMC**, which itself
holds the population at a fixed budget by relocating dead Gaussians rather than by pruning.

**Static scenes only.**

## 3. Budget handle
Two distinct handles, and they are not the same thing:

**(a) At training time: N, the full-scene capacity.** "set the full-scene capacity to N=5M
(Eq. (4))" (page 8), and for ablations "N=1M splats and 50 k training steps" (page 12). This is
inherited from 3DGS-MCMC. The paper describes it, page 5, as "(iii) adopt an capacity control
mechanism to maintain a fixed Gaussian budget during training." No further detail on that
mechanism is given in the body of MGS.

**(b) At deploy time: k, the prefix length**, an integer number of splats. A **TARGET**, and an
exactly-hit one by construction: rendering the first k splats of a sorted array gives exactly k
splats. No error, because it is a truncation, not an optimisation.

**Achieved-versus-requested pairs: none exist and none are needed.** Prefix truncation is exact.
Evaluated prefix ratios (page 7): "are evaluated at the same prefix ratios as MGS: 100%, 90%, ...,
20%, 10%, 5%, and 1%." That is 12 operating points. Claimed granularity, page 12: "MGS produces a
coherent rendering for every integer splat budget k ∈ {1, . . . , N}; in practice, evaluating 12
budget ratios already yields a smooth quality–speed curve (Fig. 3)."

**r_min, the smallest prefix fraction sampled during training, is defined but its value is never
stated.** Page 6: "where rmin ∈ (0, 1] is the smallest prefix fraction seen during training;
rmin = 1 recovers standard 3DGS training." No number is given anywhere. Evaluation goes down to
1 %, which suggests r_min ≤ 0.01, but the paper does not say so.

## 4. The control law in full
There is **no controller in the training loop at all**. MGS does not measure a count against a
target and correct. The count is fixed by 3DGS-MCMC. What MGS adds is a **training objective that
makes every prefix a valid model**, so that the budget can be hit later by truncation.

**(a) Importance score** (Eq. 3, page 6), opacity in descending order:

    s(gᵢ) = σᵢ

Page 6: "The MGS formulation is agnostic to the choice of s(g); any statistic that induces a
meaningful importance ordering is admissible. Empirically, we find that sorting Gaussians by
opacity in descending order yields stable behaviour across budgets."

**(b) Ordering** (Eq. 4, page 6), π a permutation of {1,...,N} with

    s(g_π(1)) ≥ s(g_π(2)) ≥ ... ≥ s(g_π(N))

**(c) Prefix** (Eq. 5, page 6):

    G_≤k := { g_π(1), ..., g_π(k) },   k ∈ {1, ..., N}

Rendering at budget k: Î_≤k = R(G_≤k; c). "Because prefixes are nested (G_≤k ⊂ G_≤k' for k < k')
and rasterization cost scales with the Gaussian count, varying k traces a continuous quality-speed
curve from a single trained model without any per-budget retraining or model switching." (page 6)

**(d) Stochastic budget sampling** (Eq. 6, page 6), the mechanism the paper names as its key idea:

    k = ⌈ r N ⌉,   r ~ Unif(r_min, 1)

"Uniform sampling ensures that every budget in [⌈rminN⌉, N] is visited with equal probability over
training, yielding unbiased coverage of the full budget spectrum." (page 6)

**(e) Training objective** (Eq. 7, pages 6–7), two renders per step:

    ℓ_MGS := ℓ(G_≤k; I, c) + γ · ℓ(G_≤N; I, c)

with ℓ the standard 3DGS reconstruction loss (Eq. 2), ℓ = (1−λ)‖R(G;c) − I‖₁ + λ L_D-SSIM.
γ = 1 by default. "The prefix term pressures the model to produce strong reconstructions from
partial subsets, while the full-set term anchors full-quality performance." (page 7)
"Each step incurs exactly two renders regardless of N." (page 7)

**(f) Dynamic reordering** (Eq. 8, page 7), after **every** training iteration:

    π ← argsort(s),   s = ( s(g₁), ..., s(g_N) )

in non-increasing order. "This ensures every prefix G_≤k always contains the k most important
primitives under the current parameters." (page 7)

**(g) Schedule, ramp shape and overshoot.** There is no ramp and no schedule. The prefix objective
is active from iteration 1 through iteration 50 000, at uniform strength, with the budget resampled
i.i.d. every step. Overshoot above a final budget is meaningless here, because there is no final
budget: the full set N is always maintained and any k below it is available at render time.

**(h) Can the count grow while the controller is active?** The **total** count N is fixed at 5 M by
3DGS-MCMC throughout training. The **rendered** count k changes every iteration by resampling. No
densification decision is driven by any budget signal in MGS itself.

**Prefix/full weight sweep, page 13.** "Sweeping the prefix : full weight ratio from 9:1 to 1:9
reveals a clear trade-off (Fig. 7). At the 10% prefix, prefix-heavy weighting (9:1) attains 22.2 dB
whereas the full-heavy setting (1:9) drops to 21.1 dB. Conversely, at full quality, the 1:9 ratio
achieves the best performance (PSNR 25.5 dB and LPIPS 0.169), surpassing 9:1 by +0.2 dB. For
simplicity, we use 1:1 (5:5) as the default setting, as it achieves comparable performance (25.4 dB
at full quality and 22.1 dB at 10% prefix quality)." Fig. 7 caption (page 14) says the best AUC is
at a different ratio: "While prefix-heavy or full-heavy weighting can marginally improve their
respective endpoints, a 6:4 ratio provides the best area under the curve. For the sake of
simplicity and balanced performance, we utilise a 5:5 ratio for our MGS approach and for all other
ablation experiments."

**This weight sweep is the single most transferable finding for a BudgetGS project**: the
prefix:full ratio is a continuous dial trading tight-budget quality against full-budget quality,
and it costs about 1.1 dB at the 10 % budget to swing from prefix-heavy to full-heavy, versus about
0.2 dB at full budget in the other direction.

## 5. What "size" or "count" means, and whether MB is reported
"Budget" means **number of splats rendered**, and the second axis is **FPS**. Metrics are PSNR,
SSIM, LPIPS plus two area-under-curve summaries. From page 7 and page 8:

    Q̄ = (1/3)( p̃ + s̃ + (1 − ℓ̃) )

"where p̃, s̃, ℓ̃ are PSNR, SSIM, and LPIPS linearly clamped to [0, 1] using fixed ranges (14, 32),
(0.35, 0.92), and (0.06, 0.60), respectively."

AUC_fps clips FPS to [0, 500]; AUC_splat clips budget to [0, 5M]. Both scaled by 100.

**MB is never reported. No byte figure, no storage figure, no bitrate appears anywhere.** MGS binds
splat count and, through it, FPS. It says nothing about file size, and prefix truncation of a
sorted array is not a compression scheme (all N splats must still be stored to serve any k).

## 6. Runtime cost
Hardware (page 8): "All experiments are conducted on identical Ubuntu servers with NVIDIA A100
GPU." Training: 50 000 iterations, N = 5 M for the main results, N = 1 M for ablations.

**Training wall-clock time is not stated.** The training overhead is characterised only structurally:
two renders per iteration instead of one, plus one argsort of N scores per iteration. "This
strategy requires only two forward passes and introduces no architectural modifications." (page 1)

**GPU memory is not stated.**

FPS enters only through AUC_fps and Fig. 3. One absolute FPS figure appears, page 13: "at 10% of
the splat budget, it achieves 22.2 dB PSNR at 493 FPS" (bicycle, N = 1 M).

## 7. Elastic versus dedicated, and "one training run, many budgets"
This is MGS's central claim, and it is measured against **3DGS-MCMC, its own backbone trained only
for the full set**. All values **reported**, from Table 1 page 8 and the discussion on page 12.

Section title, page 12: **"Quality vs 3DGS-MCMC."** Verbatim:

> "Although MGS trains for all possible splat budgets, its full-quality performance closely
> approaches and sometimes exceeds that of stand-alone 3DGS-MCMC [17] backbone, which trains only
> for the full set and provides no LoD capability. On MipNeRF 360 and Tanks & Temples, MGS trails
> 3DGS-MCMC by only 0.20 dB (28.20 vs. 28.40 and 24.56 vs. 24.76, respectively). On Deep Blending
> and BungeeNeRF, MGS surpasses 3DGS-MCMC (28.41 vs. 27.63 dB and 27.13 vs. 27.04 dB), suggesting
> that the stochastic budget objective can act as a beneficial regulariser on certain scene types.
> This confirms that continuous LoD need not come at the cost of full-capacity quality."

**Elastic-versus-dedicated gap at full budget**, all four benchmarks, PSNR in dB:

| benchmark | MGS (elastic) | 3DGS-MCMC (dedicated, full-set only) | gap |
|---|---|---|---|
| MipNeRF 360 | 28.20 | 28.40 | −0.20 (stated in text) |
| Tanks & Temples | 24.56 | 24.76 | −0.20 (stated in text) |
| Deep Blending | 28.41 | 27.63 | +0.78 (*computed*, paper states MGS surpasses) |
| BungeeNeRF | 27.13 | 27.04 | +0.09 (*computed*, paper states MGS surpasses) |

SSIM and LPIPS at full budget, Table 1: MipNeRF 360 MGS 0.841 / 0.130 versus 3DGS-MCMC 0.843 /
0.133; T&T MGS 0.874 / 0.086 versus 0.877 / 0.083; Deep Blending MGS 0.902 / 0.176 versus 0.893 /
0.204; BungeeNeRF MGS 0.906 / 0.088 versus 0.903 / 0.091.

**Important caveat: there is no dedicated-versus-elastic comparison at reduced budgets.** 3DGS-MCMC
is run only at full capacity. Nothing in the paper answers "how much worse is the 10 % prefix of an
MGS model than a 3DGS-MCMC model trained directly at 500 k splats", which is exactly the question a
BudgetGS project cares about.

**One training run, many budgets: yes, explicitly.** Page 3: "At deployment, one adjusts k to match
available resources—no per-budget retraining, no auxiliary data structures." Page 9: "By adjusting
the prefix ratio from 1% to 100%, MGS produces a smooth, dense Pareto frontier of operating points
from a single trained model, without any per-budget retraining or model switching."

**Quality at aggressive prefixes, page 9 (Fig. 4 caption, page 10):** "Under highly constrained
budgets (5–10%), MGS maintains coherent reconstructions with PSNR of 21–28 dB, while both baselines
suffer from severe artifacts and quality collapse (11–17 dB)."

**Failure case, page 9 and Fig. 5 caption page 11:** "the one exception is DrJohnson, where
CLoD-3DGS achieves higher peak PSNR at full budget yet degrades more sharply at reduced budgets."
Fig. 5 caption: "On DrJohnson (bottom), CLoD-3DGS [26] achieves higher peak PSNR at 100% budget
(29.1 vs. 27.7 dB); however, MGS still degrades more gracefully at reduced budgets (5–30%)."

**Internal inconsistency to be aware of.** The page 9 narrative quotes AUC_fps values that do not
match Table 1 on page 8. Page 9: "CLoD-3DGS [26] achieves the next-highest AUCfps on MipNeRF 360
(28.94 vs. 54.46 for MGS)" and "CLoD-GS [4] spans a narrower speed range, resulting in lower
AUCfps (9.78 on MipNeRF 360)" and "Discrete-LoD methods such as Octree-GS [31] and H3DGS [16]
exhibit low AUC scores (AUCfps: 9.96 and 4.81 on MipNeRF 360)". Table 1 on page 8 gives, for
MipNeRF 360 AUC_fps: CLoD-3DGS 46.38, MGS 64.81, CLoD-GS 19.39, Octree-GS 21.61, H3DGS 11.62.
Every one of the four narrative numbers differs from its table entry. **Treat the Table 1 values as
authoritative and do not cite the page 9 AUC numbers.**

A second, smaller inconsistency: the ablation Table 2 (page 13) reports the same default
configuration twice with different values, 25.47 / 0.776 / 0.174 under "Opacity ↓ (ours)" and
25.43 / 0.776 / 0.175 under "Prefix + full (ours)".

## 8. Main quantitative results
Table 1, page 8, **reported at the highest splat budget per baseline**. Best two LoD baselines by
PSNR are Octree-GS and MaskGaussian on Mip-NeRF 360 and T&T; Octree-GS and MaskGaussian again on
Deep Blending and BungeeNeRF. Raw rows, columns PSNR↑ SSIM↑ LPIPS↓ AUCfps↑ AUCsplats↑:

**MipNeRF 360**
```
3DGS-MCMC [17]      28.40  0.843  0.133   –      –
H3DGS [16]          25.41  0.780  0.230  11.62  60.51
FlexGaussian [36]   26.86  0.793  0.237  30.04  62.05
MaskGaussian [22]   27.42  0.815  0.218  34.11  67.10
Octree-GS [31]      27.62  0.813  0.221  21.61  68.43
CLoD-GS [4]         26.55  0.803  0.233  19.39  63.43
CLoD-3DGS [26]      27.44  0.814  0.215  46.38  56.56
MGS (Ours)          28.20  0.841  0.130  64.81  77.79
```
**Tanks & Temples**
```
3DGS-MCMC [17]      24.76  0.877  0.083   –      –
H3DGS [16]          20.41  0.808  0.186   8.23  58.84
FlexGaussian [36]   22.89  0.823  0.198  52.38  64.93
MaskGaussian [22]   23.66  0.846  0.181  45.15  68.89
Octree-GS [31]      24.59  0.865  0.157  23.80  67.81
CLoD-GS [4]         22.78  0.832  0.193  23.39  65.37
CLoD-3DGS [26]      23.51  0.845  0.176  50.97  61.89
MGS (Ours)          24.56  0.874  0.086  70.16  77.54
```
**Deep Blending**
```
3DGS-MCMC [17]      27.63  0.893  0.204   –      –
H3DGS [16]          27.77  0.882  0.277  10.39  73.82
FlexGaussian [36]   29.25  0.899  0.251  43.61  75.53
MaskGaussian [22]   29.69  0.907  0.245  54.96  78.77
Octree-GS [31]      30.35  0.910  0.252  29.54  83.21
CLoD-GS [4]         29.03  0.901  0.249  26.09  77.34
CLoD-3DGS [26]      29.54  0.903  0.247  69.36  72.33
MGS (Ours)          28.41  0.902  0.176  82.63  83.22
```
**BungeeNeRF**
```
3DGS-MCMC [17]      27.04  0.903  0.091   –      –
H3DGS [16]          27.74  0.910  0.114   8.47  62.57
FlexGaussian [36]   25.87  0.872  0.129  12.89  53.58
MaskGaussian [22]   27.80  0.917  0.099  21.58  71.18
Octree-GS [31]      28.23  0.922  0.088  19.69  73.70
CLoD-GS [4]         25.35  0.859  0.148  13.83  65.40
CLoD-3DGS [26]      26.57  0.892  0.131  24.45  29.72
MGS (Ours)          27.13  0.906  0.088  62.99  79.93
```
Note MGS is **not** best on PSNR on Deep Blending (28.41 versus Octree-GS 30.35, a 1.94 dB deficit,
*computed*) or BungeeNeRF (27.13 versus Octree-GS 28.23, a 1.10 dB deficit, *computed*). It wins on
LPIPS everywhere and on both AUC columns everywhere. Page 9: "On Deep Blending [12] and BungeeNeRF
[39], Octree-GS obtains higher PSNR at its single highest-quality level; however, its coarser LoD
levels incur severe quality degradation, yielding far lower AUC scores than MGS."

Ablation, Table 2 page 13, bicycle, N = 1 M, at full budget:
```
Importance Score
  Opacity ↓ (ours)   25.47  0.776  0.174  68.62  61.05
  Opacity ↑          24.91  0.743  0.219  30.53  33.48
  Color variance ↓   25.25  0.770  0.183  51.94  46.26
  SH energy ↓        25.17  0.766  0.187  59.39  47.12
  Volume ↓           25.19  0.764  0.183  52.15  44.73
  Fixed append       25.51  0.773  0.175  46.41  43.09
  Fixed prepend      25.32  0.772  0.176  49.65  42.90
Budget Training
  Prefix + full (ours) 25.43 0.776  0.175  66.87  60.96
  Prefix               24.97 0.734  0.253  56.58  57.09
  MRL                  25.37 0.771  0.183  66.57  60.35
```
At the 10 % prefix, page 13: opacity-descending 22.2 dB at 493 FPS; SH-energy-descending 17.6 dB;
"fixed-random reaches 21.5 dB compared to 22.2 dB for opacity-descending"; and, in a sentence whose
antecedent is garbled in the PDF ("yielding 10.5 dB and 11.5 dB„ respectively, at the 10% of the
splat budget"), two low values are attributed to the two fixed insertion orders.

## 9. Stated limitations and future work
**No Limitations section.** The only forward-looking sentence, verbatim, page 14:

> "Future work could explore distance or view-dependent prefix selection, adaptive budget
> scheduling, and integration with streaming or device-aware rendering systems."

The one acknowledged negative result, Fig. 5 caption, page 11:

> "Additional qualitative results including a failure case. On stump, train, and rome, MGS
> maintains higher fidelity across all budget levels. On DrJohnson (bottom), CLoD-3DGS [26]
> achieves higher peak PSNR at 100% budget (29.1 vs. 27.7 dB); however, MGS still degrades more
> gracefully at reduced budgets (5–30%)."

And in the body, page 9:

> "the one exception is DrJohnson, where CLoD-3DGS achieves higher peak PSNR at full budget yet
> degrades more sharply at reduced budgets."

## 10. Every sentence touching a size, byte, bitrate, memory or FPS target, rate control, budget, or edge/mobile deployment

Page 1:
> "The ability to render scenes at adjustable fidelity from a single model, known as level of detail (LoD), is crucial for practical deployment of 3D Gaussian Splatting (3DGS)."

> "MGS learns a single ordered set of Gaussians such that rendering any prefix, the first k splats, produces a coherent reconstruction whose fidelity improves smoothly with increasing budget."

> "Our key idea is stochastic budget training: each iteration samples a random splat budget and optimises both the corresponding prefix and the full set."

> "Experiments across four benchmarks and six baselines show that MGS matches the full-capacity performance of its backbone while enabling a continuous speed–quality trade-off from a single model."

> "Keywords: 3D Gaussian splatting · continuous level of detail · nested representations · budgeted rendering · stochastic training"

> "Real-time neural rendering is governed by a fundamental tension between scene fidelity and the computational budget available [25, 30]. This budget varies by orders of magnitude across the hardware spectrum, from high-end GPU workstations to mobile devices and mixed-reality headsets [28,37]; and fluctuates dynamically at runtime by viewpoint and scene complexity [9,45]. Level of detail (LoD) techniques address this tension by scaling the rendered representation to match available resources and have long been a cornerstone of interactive graphics [24]."

Page 2:
> Fig. 1 caption: "Continuous LoD need not sacrifice full-capacity quality to enable budget trade-off. Our method, MGS (top), learns an ordered set of Gaussian primitives whose prefixes yield coherent reconstructions at any splat budget. Compared to CLoD-3DGS [26] (mid) and CLoD-GS [4] (bot), MGS achieves the highest fidelity at every operating point with quality degrading gracefully under budget reduction."

> "A coarse set of fixed levels, however, cannot smoothly track a budget that shifts continuously with scene content and viewpoint, and the abrupt transitions between levels produce visible pop-in and pop-out artifacts [9,10]."

> "3D Gaussian Splatting (3DGS) [15] achieves photorealistic novel view synthesis by rasterizing millions of anisotropic Gaussian primitives at real-time frame rates, with computational cost scaling directly with the number of primitives [8]. In principle, this primitive-based nature of the representation offers continuous budget control, since omitting any subset of splats yields an immediate speedup [11,15]. Yet a conventionally trained 3DGS model has no ordering among its primitives, so quality collapses rapidly as splats are removed [21]."

> "Discrete LoD methods [16, 19, 31, 33] build hierarchical structures over Gaussian primitives, exposing a limited number of quality levels but requiring auxiliary index structures and offering only coarse budget granularity. Compression and pruning pipelines [6,22,36] can approximate multi-budget behaviour, but each operating point is typically obtained independently without ensuring that subsets are nested or globally coherent."

Page 3:
> "Rendering at different budgets is then achieved by truncating the ordered sequence, producing a continuous spectrum of quality-speed operating points from a single model."

> "The key mechanism is stochastic budget training: each iteration samples a random splat budget uniformly and optimises both the corresponding prefix and the full set, requiring only two forward passes each training step."

> "At deployment, one adjusts k to match available resources—no per-budget retraining, no auxiliary data structures."

> "We propose stochastic budget training, a simple yet effective model-agnostic training procedure that optimises across a continuous budget range with only two renders per iteration."

Page 4:
> "This formulation enables photorealistic novel view synthesis at real-time frame rates, with rendering cost scaling directly with the number of primitives, making primitive count a natural axis for controlling computational budget."

> "For capacity control, the original densification heuristics offer no direct control over the final primitive count. 3DGS-MCMC [17] addresses this via Langevin-dynamics sampling under a fixed budget. Mini-Splatting [7] further improves densification under constrained capacities."

> "Level-of-detail (LoD) rendering enables adaptive control of rendering cost by varying scene complexity according to available computational budgets."

> "Compression and pruning pipelines [6,22,36] can approximate multi-budget behaviour by applying importance-based pruning or quantization at varying targets, but each operating point is typically obtained independently without ensuring nested or globally consistent primitive subsets [27,29]."

> "However, both incur noticeable quality degradation at full capacity and rapid quality collapse at reduced budgets, making continuous LoD a costly design choice."

Page 5:
> "MGS transfers the nested representation principle to Gaussian scene primitives to enable continuous control over rendering budgets through ordered primitive prefixes within a single model."

> "We construct a nested Gaussian representation that enables continuous control over rendering budgets from a single model. Specifically, we (i) define an importance score to rank Gaussian primitives, (ii) organize them into an ordered prefix representation that supports variable-budget rendering, and (iii) adopt an capacity control mechanism to maintain a fixed Gaussian budget during training."

Page 6:
> "Empirically, we find that sorting Gaussians by opacity in descending order yields stable behaviour across budgets."

> "Because prefixes are nested (G≤k ⊂ G≤k′ for k < k′) and rasterization cost scales with the Gaussian count, varying k traces a continuous quality-speed curve from a single trained model without any per-budget retraining or model switching."

> "Training a single representation to perform well at every budget level is the central challenge of LoD. Optimizing all N possible budgets per step would be prohibitively expensive."

> "where rmin ∈ (0, 1] is the smallest prefix fraction seen during training; rmin = 1 recovers standard 3DGS training. Uniform sampling ensures that every budget in [⌈rminN⌉, N] is visited with equal probability over training, yielding unbiased coverage of the full budget spectrum."

Page 7:
> "Over training, this stochastic procedure estimates the expected multi-budget objective across all budget fractions r ∈ [rmin, 1] and all training views, with prefix membership determined by the ordering in Eq. (4). Each step incurs exactly two renders regardless of N."

> "Continuous LoD methods support rendering at arbitrary splat budgets, including CLoD-GS [4] (distance-dependent opacity decay) and CLoD-3DGS [26] (learned importance ordering), are evaluated at the same prefix ratios as MGS: 100%, 90%, . . ., 20%, 10%, 5%, and 1%."

> "In addition, a key requirement to evaluating LoD methods is comparing quality–speed trade-off at different operating points."

Page 8:
> Table 1 caption: "PSNR, SSIM [38], LPIPS [44] reported at highest splat budget per baseline. AUCfps summarises quality–FPS trade-off frontier, and AUCsplats encapsulates quality/splat efficiency."

> "AUCfps(↑): we construct a monotone envelope of quality versus FPS. Noting that an operating point can be replicated at any lower speed, the envelope extends leftward from every achieved operating point along lowering throughput. We clip FPS to [0, 500] and compute the normalised area under this envelope."

> "AUCsplat(↑): we construct a monotone envelope of quality vs splat count. Noting that any operating point can be replicated with additional splats, the envelope extends rightward from all operating points along increasing splat count. Also noting that quality diminishes as splat count approaches 0, we connect the origin (0, 0) to the lowest-budget operating point. We clip budget to [0, 5M] and compute the normalised area under this envelope."

> "Unless otherwise noted, we order splats by opacity in descending order (Eq. (4)), use equal prefix/full weights (γ=1, Eq. (7)), set the full-scene capacity to N=5M (Eq. (4)), and train for 50 k iterations. All experiments are conducted on identical Ubuntu servers with NVIDIA A100 GPU."

Page 9:
> "MGS outperforms all baselines in both AUCfps and AUCsplats by wide margins across four benchmarks, sustaining high fidelity consistently across varying speed and budget constraints (Fig. 3)."

> "Qualitative comparisons also confirm that MGS preserves coherent scene structure at aggressive budget reductions (5–10% splats) Figs. 4 and 5"

> "By adjusting the prefix ratio from 1% to 100%, MGS produces a smooth, dense Pareto frontier of operating points from a single trained model, without any per-budget retraining or model switching."

> Fig. 3 caption: "Left: Quality (Q̄, Eq. 9) vs. FPS. Right: Quality vs. number of Gaussian splats. Curves trace continuous LoD models across prefix ratios 1%–100%, and trace discrete LoD models at their recommended operating points respectively. MGS (ours, dark blue) achieves the highest quality at every speed and splat budget, while spanning a much wider FPS range than any baseline."

Page 10:
> Fig. 4 caption: "Renderings are shown at 5%, 10%, 30%, 60%, and 100% of the full splat budget. ... Under highly constrained budgets (5–10%), MGS maintains coherent reconstructions with PSNR of 21–28 dB, while both baselines suffer from severe artifacts and quality collapse (11–17 dB)."

Page 12:
> "A key practical advantage of MGS is the density of the operating-point frontier. MGS produces a coherent rendering for every integer splat budget k ∈ {1, . . . , N}; in practice, evaluating 12 budget ratios already yields a smooth quality–speed curve (Fig. 3). In contrast, Octree-GS [31] provides 3–6 LOD levels, H3DGS [16] offers 9 τ-thresholds, and FlexGaussian [36] exposes 2–6 compression targets, each requiring separate configurations. This distinction matters for deployment: MGS allows a system to respond to per-frame or per-device budgets by simply truncating the splat array, with no additional data structures, mode switches, or latency spikes."

> "Although MGS trains for all possible splat budgets, its full-quality performance closely approaches and sometimes exceeds that of stand-alone 3DGS-MCMC [17] backbone, which trains only for the full set and provides no LoD capability."

Page 13:
> "at 10% of the splat budget, it achieves 22.2 dB PSNR at 493 FPS, whereas the next-best score-based ordering (SH-energy descending) reaches only 17.6 dB under the same constraints."

> "MRL nesting restricts the model to a discrete set of budget sizes, sacrificing the fine-grained coverage that stochastic sampling provides."

Page 14:
> "By learning an ordered, prefix-closed set of Gaussian primitives, MGS allows rendering at arbitrary splat budgets by truncating the primitive set, producing a dense spectrum of quality–speed operating points."

> "Future work could explore distance or view-dependent prefix selection, adaptive budget scheduling, and integration with streaming or device-aware rendering systems."

## 11. Code URL and licence
Printed under the author block on page 1: **https://ZhilinGuo.github.io/MGS** (a project page; the
text does not say "code is available"). **No licence is stated.** Built on gsplat [41] and the
3DGS-MCMC training strategy [17].

---

# 4. FlexGS

## 1. Citation
- Title as printed: **FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian
  Splatting**
- First author: **Hengyu Liu** (The Chinese University of Hong Kong), equal contribution with
  Yuehao Wang and Chenxin Li. Co-authors Ruisi Cai (project lead), Kevin Wang, Wuyang Li, Pavlo
  Molchanov, Peihao Wang, Zhangyang Wang. Affiliations: CUHK, UT Austin, Nvidia.
- Venue: **not stated inside the PDF.** The file name records CVPR 2025; the PDF itself shows only
  the arXiv stamp and a CVPR-style two-column layout with a "Supplementary Material" section.
- arXiv id from the stamp on page 1: **arXiv:2506.04174v1 [cs.CV] 4 Jun 2025**.

## 2. Base representation and renderer
Vanilla 3DGS with differentiable splatting. Each Gaussian: position X ∈ ℝ³, scaling s ∈ ℝ³,
rotation q ∈ ℝ³ (written as ℝ³ in the paper although a quaternion is described), opacity α ∈ ℝ,
SH ∈ ℝ^((d+1)²). Σ = R S Sᵀ Rᵀ, Σ' = J W Σ Wᵀ Jᵀ.

**Static scenes only.** The 4D machinery in the method is a spatial-**ratio** field, borrowed from
dynamic-scene work, where the fourth axis is the elastic ratio e rather than time.

## 3. Budget handle
The user supplies **e, an elastic ratio in (0,1]**, the fraction of the original Gaussians to keep.
Described as "an input specifying the desired model size" (page 2). It is a **TARGET at deploy
time**, but the learned selector does not hit it exactly, so a hard top-k is applied on top.

**The one achieved-versus-requested pair in the paper**, appendix B.2, page 11, verbatim:

> "Furthermore, we observe that despite enforcing sparsity supervision on the masks predicted by
> GsNet, the number of activated entries within the predicted mask does not exactly achieve the
> desired ratio. For instance, a target ratio of 0.20 results in approximately selecting 19.5% of
> all Gaussians. Therefore, to attain an accurate elastic ratio, during inference, we employ
> Pytorch's F.gumbel softmax function with its parameter hard=False to output continuous logits and
> select the top ⌊eN⌋ logits out of N."

So: **requested 0.20 → achieved ≈ 0.195** (2.5 % relative error, *computed*) from the soft
mechanism alone. The final inference path bypasses this by forcing exactly ⌊eN⌋ splats. No other
achieved-versus-requested pair appears in the paper, and no absolute counts are ever reported, only
ratios.

Training ratios: E = {0.01, 0.05, 0.10, 0.15} (page 6). Test ratios unseen at training:
{0.02, 0.04, 0.06, 0.08} (page 8).

## 4. The control law in full
Two learned modules, both driven by e, jointly optimised with the Gaussians.

**(a) Problem statement, page 3.** "For a 3DGS with N Gaussians and a target ratio e (representing
the desired percentage of remaining Gaussians), FlexGS aims to predict a binary scaler λᵢ for each
Gaussian. This scaler determines whether the i-th Gaussian is selected (λᵢ = 1) or not (λᵢ = 0),
ensuring minimal performance degradation while satisfying the constraint ∑ᵢ λᵢ = eN."

**(b) GsNet, a ratio-conditioned selector.** Ratio embedding (Eq. 4, page 4) and attribute
embedding (Eq. 5), with Aᵢ = {X, s, q}:

    h_e = σ( σ(e w¹_e) W²_e ),                w¹_e ∈ ℝᴰ,  W²_e ∈ ℝᴰˣᴰ
    hᵃᵢ = σ( σ(ξ W¹_a) W²_a ),  ξ = Concat({a : a ∈ Aᵢ}),   W¹_a ∈ ℝᴴˣᴰ, W²_a ∈ ℝᴰˣᴰ
    z = [h_e , hᵃᵢ] W,                        W ∈ ℝ^(2D×2)

D = 64 (page 11).

**(c) Gumbel-Softmax relaxation** (Eq. 7, page 4):

    P(λᵢ = m | Aᵢ, e) = exp((z_m + g_m)/τ) / ∑_{m=0,1} exp((z_m + g_m)/τ)

g_m ~ Gumbel(0,1). "The temperature parameter τ is exponentially decayed, and as τ approaches 0,
the distribution converges to a one-hot vector, allowing GsNet to make more deterministic
selections." (page 4). The straight-through estimator is derived in appendix A.1 (page 11):
gᵢ,c = −log(−log(Uᵢ,c)), z̃ᵢ,c = (zᵢ,c + gᵢ,c)/τ, Bᵢ = z_hard,ᵢ + stop_gradient(z_soft,ᵢ), and the
gradient ∂M̂/∂z = (1/τ)[ −B[:,0] ⊙ B[:,1] , B[:,1] ⊙ (1 − B[:,1]) ].

**(d) Global Importance, the guidance score** (Eq. 8, page 4):

    GIᵢ = ∑[p = 1 .. QHW] 1(G(Xᵢ), r_p) · γ(sᵢ) · αᵢ · T(i, p)

with (Eqs. 9–10, page 4):

    γ(sᵢ) = max( V(sᵢ)/V_90% , 1 ),   V(s) = |s₁ s₂ s₃| π / 3
    T(i, p) = ∏[t = 1 .. ID(i,p) − 1] (1 − α_t)

Q, H, W are the number of seen views, image height and width; 1(·) is the pixel-overlap indicator.
GI is recomputed every 1000 iterations (page 6).

**(e) Two regularisers, one of which IS the budget control** (Eq. 11, page 5). M^GI is the mask of
the top ⌊eN⌋ Gaussians by GI:

    L_GI = ‖ M̂ − M^GI ‖
    L_spar = | e − ( ∑_{m ∈ M̂} m ) / N |

**L_spar is the count controller: an L1 penalty on the deviation of the realised keep fraction from
the requested fraction e.** Not a Lagrangian, not a projection. A fixed-weight L1 penalty on a
fraction, with weight β₂.

**(f) Gaussian Transform Field**, which is what makes reduced budgets usable. A multi-resolution 4D
volume ψ over (X, e), factorised into 6 planes (Eq. 12, page 5):

    f = F_m(f^m),   f^m = ∪_l BiInterp( ψ_l(X, e) )

Planes {(x,y), (x,z), (y,z), (x,e), (y,e), (z,e)}, resolutions {64, 64, 64, 100} over (x, y, z, e)
(page 11). Multi-head predictor (Eq. 13, page 5), hidden dim 64:

    T^X = φ_X(f),  T^s = φ_s(f),  T^q = φ_q(f)
    X ← X + T^X,   s ← s + T^s,   q ← q + T^q

"This module predicts position and shape transformations for each Gaussian, mapping them from their
original state, where all Gaussians are selected, to the adapted state under a given elastic
ratio." (page 5). Rationale, page 5: "selecting a portion of 3DGS will inevitably lead to holes and
missing details in the rendering results. As visualized in Fig. 3, simply fine-tuning the SH and
opacity attributes is unlikely to compensate for those degraded areas in the masked 3DGS."

**(g) Objective** (Eqs. 14–16, page 5). During training the mask multiplies opacity, so unselected
Gaussians render transparent:

    αᵢ = αᵢ · M̂ᵢ
    L_render = |I^e_s − I_GT| + |I_f − I_GT|,   e ~ E
    L = L_render + β₁ L_GI + β₂ L_spar

Two renders per step, one at the sampled ratio and one at full capacity. This is structurally the
same two-render trick as MGS. Weight values are stated ambiguously, page 6: "In this phase, βs is
set to 1.0 and βd is configured to 0.01." The symbols β_s and β_d do not appear in Eq. 16, which
uses β₁ and β₂. The mapping is not stated.

**(h) Schedule, and when the budget binds.** Page 6: "The training process of FlexGS contains two
stages. The initial stage comprises 15k iterations, replicating the original 3DGS training
procedure. In the subsequent stage, both GsNet and the Transform Field are optimized over 20k
iterations, with the elastic ratio randomly sampled from {0.01, 0.05, 0.10, 0.15} at each
iteration."

So: **iterations 1 to 15 000 are completely unconstrained vanilla 3DGS**, including standard
densification. The budget machinery binds only from iteration 15 001, and even then it is a
sampled-per-step ratio, not a monotone ramp. **FlexGS is an explore-then-select method**, with an
explicit 15k-iteration unconstrained exploration phase and no ramp at all.

**(i) Overshoot and densification interaction.** The full Gaussian set N is whatever vanilla 3DGS
produced by iteration 15 000. It is never reduced. FlexGS never prunes: at inference it "directly
discard[s] the unselected Gaussians" (page 11), but the stored model is the full set plus GsNet
plus the Transform Field. The paper does not state whether densification continues during the
second stage.

## 5. What "size" or "count" means, and whether MB is reported
"Size" is used loosely for **the fraction of the Gaussian count**. The paper repeatedly frames e as
a memory budget ("varying memory budgets at inference time", "the desired model size", "any
user-desired fraction of the memory footprint") but every reported number is a **percentage of
Gaussians**, never bytes.

**MB is never reported.** Not once. There is no table column for size, no storage figure and no
bitrate. The one absolute byte number in the paper is quoted from someone else's work: "deploying
and rendering a city-scale scene (over 20 million Gaussians) requires at least 4 GB of networking
bandwidth and GPU memory" (page 1, citing [29] CityGaussian).

**FlexGS also silently increases the stored model size.** To serve any ratio, one must keep all N
Gaussians plus GsNet (two 64-dim MLPs) plus the Spatial-Ratio Neural Field (six planes at
64×64/64×100 resolutions) plus three prediction heads. A 10 % elastic ratio therefore reduces
render-time work but not on-disk size, unless a single ratio is baked out. The paper does not
discuss this.

**Absolute Gaussian counts N are never reported** for any scene, so the ratios cannot be converted
to counts from the paper alone.

## 6. Runtime cost
Hardware (page 6): "All experiments are conducted on an NVIDIA A100 GPU." Training: 15 000 +
20 000 = 35 000 iterations.

**Training wall-clock time is not stated.** Only a relative claim, page 7: "Despite utilizing less
training time, our method consistently outperforms other approaches in terms of reconstruction
quality across all metrics on almost all elastic ratios (10%, 5%, and 1%)."

**GPU memory is not stated. Rendering FPS is not reported anywhere.**

What is reported is **elastic inference time**, the wall clock to produce a model at a new ratio,
page 8: "Our method achieves the best balance, with an elastic inference time of 0.13 seconds and a
high rendering quality of PSNR 23.0, outperforming other methods like EAGLES (600s) and LightGS
(150s). It also surpasses alternatives such as LightGS* (7.43s) and C3DGS (100s) in speed and
quality." So 0.13 s versus 7.43 s for a training-free prune and 150 s to 600 s for a prune plus
fine-tune. That 0.13 s figure is the paper's real efficiency claim, and it is about **retargeting
cost, not render cost**.

Fig. 8 (page 13) gives load times for incremental scene loading: "Ratio: 1%  Load time: 2s",
"Ratio: 10%  Load time: 10s", "Ratio: 10%  Load time: 2min". The third label reads oddly against
the second (same ratio, 10 s versus 2 min); the figure appears to contrast elastic loading against
loading separate models, but the caption does not say so.

## 7. Elastic versus dedicated, and "one training run, many budgets"
**LightGS (with per-ratio fine-tuning) is the dedicated model**, and LightGS* is the same pruning
without fine-tuning. FlexGS is the elastic model. All values **reported**; the differences are
*computed* from the two reported values.

**Mip-NeRF360, Table 1 page 6, PSNR:**

| ratio | FlexGS (elastic) | LightGS (dedicated, fine-tuned per ratio) | difference |
|---|---|---|---|
| 1 % | 22.730 | 22.265 | +0.465 |
| 5 % | 25.412 | 25.223 | +0.189 |
| 10 % | 26.577 | 26.570 | +0.007 |
| 15 % | 27.035 | 27.239 | **−0.204** |

**Tanks & Temples, Table 2 page 7, PSNR:**

| ratio | FlexGS | LightGS | difference |
|---|---|---|---|
| 1 % | 23.090 | 22.795 | +0.295 |
| 5 % | 26.689 | 25.774 | +0.915 |
| 10 % | 27.069 | 26.820 | +0.249 |
| 15 % | 27.802 | 27.782 | +0.020 |

**Zip-NeRF, Table 2 page 7, PSNR:**

| ratio | FlexGS | LightGS | difference |
|---|---|---|---|
| 1 % | 20.963 | 19.812 | +1.151 |
| 5 % | 23.513 | 22.899 | +0.614 |
| 10 % | 24.691 | 24.223 | +0.468 |
| 15 % | 25.167 | 24.957 | +0.210 |

**Unseen ratios, Zip-NeRF, Table 4 page 8, PSNR:**

| ratio | FlexGS | LightGS | difference |
|---|---|---|---|
| 2 % | 21.636 | 21.081 | +0.555 |
| 4 % | 22.971 | 22.448 | +0.523 |
| 6 % | 24.402 | 23.252 | +1.150 |
| 8 % | 24.720 | 23.805 | +0.915 |

**Reading.** The elastic model **beats** the dedicated fine-tuned model at 11 of 12 seen operating
points and at all 4 unseen ones. The single loss is Mip-NeRF360 at 15 %, by 0.204 dB. The margin
shrinks monotonically as the ratio rises on Mip-NeRF360 and Zip-NeRF: the elastic advantage is
largest at very tight budgets and disappears (or inverts) as the budget loosens. This is exactly
the opposite pattern to MGS, where the elastic model trails at the full budget and wins at low
prefixes. The likely common cause is that the joint prefix objective regularises the tight-budget
regime that a dedicated model would otherwise have to reach by pruning a model that was never
optimised for it.

**Caveat.** The dedicated baseline here is LightGS, an importance-prune-plus-fine-tune pipeline, not
a model trained from scratch at the target count. There is **no comparison in this paper against a
3DGS run trained from the start at eN Gaussians**, which is the comparison the project needs.

**One training run, many budgets: yes, explicitly, and it is the title claim.** Page 7: "Note that
different from all previous methods, which require separate training for each ratio, FlexGS only
requires a single training session to enable flexible inference across different ratios."

**Table 2 and Table 3 disagree on two FlexGS entries for T&T.** Table 2 page 7 gives Ours at 5 %:
26.689 / 0.8764 and at 15 %: 27.802 / 0.9101. Table 3 page 7 gives "Full Model" at 5 %: 26.189 /
0.8769 / 0.1456 and at 15 %: 27.302 / 0.9106 / 0.0934. Same dataset, same method, same ratios,
0.5 dB apart at both points. The page-8 text quotes the Table 3 values ("with optimal results of
27.302 PSNR, 0.9106 SSIM, and 0.0934 LPIPS at 15% sparsity") while the page-7 text quotes the
Table 2 values ("with FlexGS reaching a PSNR of 27.802 at the 15% ratio"). **Cite whichever table
you take a number from and note the discrepancy.**

## 8. Main quantitative results
**Deep Blending is not evaluated.** The datasets are Mip-NeRF360, Tanks & Temples and Zip-NeRF.
(Fig. 4 on page 6 labels a drjohnson crop as "T&T: drjohnson"; drjohnson is a Deep Blending scene,
so that label is wrong, and no drjohnson numbers appear in any table.)

Strongest two baselines throughout are **LightGS** (pruning plus per-ratio fine-tuning) and
**C3DGS**. Table 1, page 6, Mip-NeRF360, all **reported**, columns PSNR / SSIM / LPIPS per ratio:
```
             1%                        5%                        10%                       15%
LightGS* 14.455 0.3741 0.5780 | 17.302 0.5055 0.4369 | 19.595 0.6085 0.3506 | 21.467 0.6805 0.2940
LightGS  22.265 0.6128 0.4384 | 25.223 0.8381 0.3305 | 26.570 0.7831 0.2399 | 27.239 0.8079 0.2070
C3DGS    21.766 0.5440 0.5101 | 24.224 0.7333 0.3452 | 25.609 0.7700 0.2457 | 26.447 0.7807 0.2277
EAGLES   21.761 0.5173 0.5479 | 23.421 0.7389 0.3544 | 24.489 0.6822 0.3111 | 25.078 0.6789 0.2744
Ours     22.730 0.6128 0.4384 | 25.412 0.7475 0.2935 | 26.577 0.7933 0.2320 | 27.035 0.8072 0.2087
```
Note the 1 % row: Ours SSIM 0.6128 and LPIPS 0.4384 are **identical** to LightGS at 1 %, while the
PSNR differs. Likely a transcription error in the table; flag it if quoting. Also note LightGS at
5 % reports SSIM 0.8381, which is higher than its own 10 % (0.7831) and 15 % (0.8079) values, so
that column is not monotone either.

Table 2, page 7, T&T and Zip-NeRF, PSNR↑ SSIM↑ per ratio:
```
            T&T                                              Zip-NeRF
            1%            5%            10%           15%     1%            5%            10%           15%
LightGS* 14.653 0.4674 19.545 0.6705 21.724 0.7705 23.550 0.8271 | 14.744 0.5442 18.772 0.6514 20.913 0.7119 22.839 0.7506
LightGS  22.795 0.7231 25.774 0.8605 26.820 0.8954 27.782 0.9098 | 19.812 0.6527 22.899 0.7356 24.223 0.7788 24.957 0.8040
C3DGS    21.535 0.6930 25.625 0.8460 26.823 0.8853 26.860 0.9013 | 19.955 0.6405 22.105 0.7065 23.425 0.7518 24.053 0.7760
EAGLES   19.558 0.6140 22.738 0.7620 24.058 0.8155 24.808 0.8440 | 18.495 0.6193 20.599 0.6721 21.934 0.7086 22.785 0.7359
Ours     23.090 0.7731 26.689 0.8764 27.069 0.9045 27.802 0.9101 | 20.963 0.6722 23.513 0.7572 24.691 0.7962 25.167 0.8132
```

Ablation, Table 3 page 7, T&T:
```
                                 1%                      5%                      10%                     15%
w/o Adaptive Selector    14.653 0.4674 0.5092 | 19.545 0.6705 0.3082 | 21.724 0.7705 0.2127 | 23.550 0.8271 0.1628
w/o Transform Field      15.117 0.4564 0.6080 | 14.600 0.4441 0.6231 | 14.576 0.4441 0.6273 | 14.569 0.4439 0.6290
Adapt. Sel. w/o Sparsity 23.017 0.7645 0.2894 | 26.022 0.8725 0.1496 | 26.960 0.9002 0.1088 | 27.147 0.9078 0.0965
Adapt. Sel. w/o GI       19.472 0.6325 0.4222 | 22.944 0.7702 0.2685 | 24.075 0.8155 0.2145 | 24.642 0.8360 0.1889
Adapt. Sel. w/o Spars.+GI 18.363 0.5528 0.4619 | 24.323 0.8336 0.1856 | 26.528 0.8919 0.1275 | 26.831 0.9013 0.1152
Full Model               23.090 0.7731 0.2854 | 26.189 0.8769 0.1456 | 27.069 0.9045 0.1035 | 27.302 0.9106 0.0934
```
**The most useful ablation row for a BudgetGS project is `Adapt. Sel. w/o Sparsity`**: removing
L_spar entirely costs only 0.073 / 0.167 / 0.109 / 0.155 dB at 1 / 5 / 10 / 15 % (*computed*).
The count-matching penalty is nearly free in quality terms. Meanwhile `w/o Transform Field`
collapses to 14.57 to 15.12 dB at every ratio, and `w/o GI` costs 3.6 dB at 1 %. So in FlexGS the
budget penalty is the cheap part and the compensation mechanism is the expensive, indispensable
part. Page 8: "Our analysis reveals that Transform Field is the most crucial component, as its
removal causes severe performance degradation (PSNR drops to 14.57-15.12 across all sparsity
ratios)."

## 9. Stated limitations and future work
**No Limitations section and no Future Work section.** Section 5 (Conclusion, page 8) is entirely
positive. The closest to an admitted shortcoming is the inference-time correction in appendix B.2,
page 11, verbatim:

> "Furthermore, we observe that despite enforcing sparsity supervision on the masks predicted by
> GsNet, the number of activated entries within the predicted mask does not exactly achieve the
> desired ratio. For instance, a target ratio of 0.20 results in approximately selecting 19.5% of
> all Gaussians."

and the motivation for the Transform Field, page 5:

> "Merely optimizing Gaussian selection for an elastic ratio may lead to excessive updates of
> crucial Gaussians at lower ratios, thereby inducing overfitting to these ratios and subsequently
> diminishing rendering quality under other ratios. Furthermore, selecting a portion of 3DGS will
> inevitably lead to holes and missing details in the rendering results."

## 10. Every sentence touching a size, byte, bitrate, memory or FPS target, rate control, budget, or edge/mobile deployment

Page 1:
> "However, 3DGS demands relatively significant GPU memory, limiting its use on devices with restricted computational resources."

> "Previous approaches have focused on pruning less important Gaussians, effectively compressing 3DGS but often requiring a fine-tuning stage and lacking adaptability for the specific memory needs of different devices."

> "In this work, we present an elastic inference method for 3DGS. Given an input for the desired model size, our method selects and transforms a subset of Gaussians, achieving substantial rendering performance without additional fine-tuning."

> "We introduce a tiny learnable module that controls Gaussian selection based on the input percentage, along with a transformation module that adjusts the selected Gaussians to complement the performance of the reduced model."

> "First, standard 3DGS models are costly, requiring users to store millions of Gaussian primitives, each with attributes such as position, scaling, rotation, opacity, and color, leading to significant memory overhead. For instance, as reported in [29], deploying and rendering a city-scale scene (over 20 million Gaussians) requires at least 4 GB of networking bandwidth and GPU memory, which is not always affordable to smartphones and laptops with weak GPU."

> "The pruning process is typically followed by an additional fine-tuning stage, which requires around 16.7% of the original training cost to restore rendering quality for each compression."

> "Second, even efficient pruned 3DGS models often struggle to meet the diverse hardware constraints of different deployment environments. For instance, as a potential application of 3DGS, virtual reality city roaming and room tours require the deployment of 3DGS-represented scenes across various devices: PC-connected headsets (e.g., Valve Index®) rely on consumer-grade GPUs in the connected PC; while standalone headsets (e.g., Meta Quest®) feature integrated GPUs with limited memory and processing power. Additionally, different device models (such as those in the iPhone®series) may have varying configurations of computing and graphics units, further increasing the diversity of computational demands."

> Fig. 1 caption: "(a) Traditional case-by-case compression requires separate fine-tuning for each ratio. (b) Our elastic inference enables dynamic compression at any ratio without re-training, adaptively balancing quality and computational costs for diverse deployment scenarios."

Pages 1–2 (sentence spans the page break):
> "Although one can compress an optimized 3DGS for every new device constraint, the process is inefficient as each specific computational budget must be handled separately. A more cost-effective solution is to generate a set of compressed models for popular devices and select the model that best meets the requirements of other devices. However, rendering quality may not reach the optimal level on devices for which the models are not specifically tailored, compromising the user experience."

Page 2:
> "In this work, we introduce FlexGS, which can be trained once and seamlessly adapt to varying computational constraints, eliminating the need for costly retraining or fine-tuning for each configuration / hardware constraint. Given an input specifying the desired model size, our method selects and transforms a subset of Gaussians to meet the memory requirements while maintaining considerable rendering performance."

> "we demonstrate that FlexGS can achieve competitive rendering quality with any user-desired fraction of the memory footprint of the original 3DGS models."

> "FlexGS seamlessly adapts to varying memory budgets at inference time. Unlike previous methods that require separate models or extensive fine-tuning for each target size, FlexGS operates with a single model that can elastically adjust the number of active Gaussians based on real-time input, offering significant flexibility for deployment on devices with different performance budgets."

> "We propose a learnable module that dynamically selects the most significant Gaussians based on an input compression ratio."

> "To address storage efficiency, two main compression paradigms have emerged."

> "However, the method typically requires millions of Gaussian primitives, each storing independent spatial and appearance attributes, resulting in substantial memory requirements."

> "Attribute compression reduces per-Gaussian storage requirements through quantization."

Page 3:
> "While existing methods rely on fixed compression ratios or model-specific optimization, our work introduces a unified framework that dynamically adapts to varying computational constraints. By jointly optimizing scene compactness and spatial relationships, our approach enables efficient deployment across diverse hardware platforms through a single trained model."

> "Given a set of ratios E = {ei}ⁿ_{i=1}, our proposed elastic framework, FlexGS, trains 3DGS once and enables it to render with an arbitrary number of Gaussians without requiring additional fine-tuning."

> "For a 3DGS with N Gaussians and a target ratio e (representing the desired percentage of remaining Gaussians), FlexGS aims to predict a binary scaler λi for each Gaussian."

Page 5:
> "To incorporate this guidance and ensure the number of selected Gaussians aligns with the desired elastic ratio e, we regularize the selection process by closing the gap between M̂ and the guidance mask via LGI, and imposing a sparsity ratio constraint via Lspar"

Page 7:
> "Despite utilizing less training time, our method consistently outperforms other approaches in terms of reconstruction quality across all metrics on almost all elastic ratios (10%, 5%, and 1%)."

> "For example, in the most challenging scenario, retaining only 1% of the Gaussians, our method achieves a PSNR of 22.730 and an SSIM of 0.6128, outperforming LightGS* (PSNR 14.455, SSIM 0.3741) and C3DGS (PSNR 21.766, SSIM 0.5440) by a noticeable margin, indicating the significant potential of our approach for edge devices."

> "Note that different from all previous methods, which require separate training for each ratio, FlexGS only requires a single training session to enable flexible inference across different ratios"

Page 8:
> "Compared to existing Gaussian compression models, our framework offers a key advantage: it achieves high-quality rendering with significantly reduced inference time, which is essential for real-time applications, especially with unseen elastic ratios. While existing methods require separate compression processes for each target ratio, our approach enables dynamic, on-the-fly compression for arbitrary ratios."

> "Our method achieves the best balance, with an elastic inference time of 0.13 seconds and a high rendering quality of PSNR 23.0, outperforming other methods like EAGLES (600s) and LightGS (150s). It also surpasses alternatives such as LightGS* (7.43s) and C3DGS (100s) in speed and quality."

> Fig. 6 caption: "PSNR vs. inference time: Our method achieves the best balance, reducing inference time to 0.13s while maintaining high render quality (PSNR 23.0), outperforming alternatives like LightGS and EAGLES across unseen elastic ratios."

> "This paper presents FlexGS, a flexible 3DGS framework that enables efficient deployment across devices with varying computational constraints. Through our elastic inference method, which combines adaptive Gaussian selection and transformation modules, we achieve efficient model compression without requiring additional fine-tuning. The framework demonstrates strong adaptability by dynamically adjusting the model size according to device-specific requirements while maintaining high rendering quality."

> "Extensive experiments on ZipNeRF, MipNeRF and Tanks&Temples datasets validate our approach's effectiveness, showing that FlexGS successfully bridges the gap between high-quality 3D scene representation and practical deployment constraints."

Page 11:
> "During elastic inference, in contrast to the training phase where the opacity of Gaussians is multiplied by the binary mask values, we directly discard the unselected Gaussians. Furthermore, we observe that despite enforcing sparsity supervision on the masks predicted by GsNet, the number of activated entries within the predicted mask does not exactly achieve the desired ratio. For instance, a target ratio of 0.20 results in approximately selecting 19.5% of all Gaussians. Therefore, to attain an accurate elastic ratio, during inference, we employ Pytorch's F.gumbel softmax function with its parameter hard=False to output continuous logits and select the top ⌊eN⌋ logits out of N."

Pages 12–13 (paragraph spans the page break):
> "How useful is the elastic inference in real application scenarios? In practical applications, the loading and deployment of pre-trained Gaussian models inherently demand a considerable amount of time. Moreover, as shown in Fig 8, the aggregate loading time escalates proportionally with the number of Gaussian models being deployed. Elastic inference enables the rapid deployment of lower-precision, coarse-grained models, while simultaneously maximizing rendering quality within a given resource budget. It can further allow for the incremental loading of higher-precision, detail-rich models. This enhances the user experience of 3D scene deployment scenarios over time, like mobile gaming and online VR shopping."

Page 13:
> Fig. 8 caption and labels: "Ratio: 1%    Load time: 2s" / "Ratio: 10%     Load time: 10s" / "Ratio: 10%     Load time: 2min" / "Figure 8. Potential use of elastic inference in the application scenario of incremental scene loading."

## 11. Code URL and licence
Printed in the abstract, page 1: **"Code is available at https://flexgs.github.io/."**
**No licence is stated.**

---

# Cross-paper synthesis for the BudgetGS project

## Nothing here binds bytes
All four control **count**, not size in bytes.
- **CDGS** is the only one that reports MB at all, and its MB comes from a separate post-hoc
  compression stage (16/8-bit quantisation, KD-tree reorder, predictive plus entropy coding for
  static; H.264 QP 20 for dynamic). Table VII page 11 shows 98.3 MB uncompressed → 31.5 MB at the
  same count, so about two thirds of the byte figure is set by the codec and not by Ntarget. The
  budget loop never sees a byte count. The closest sentence to a byte-denominated claim is page 4:
  "This constraint directly governs runtime memory, rendering cost, and even streaming bitrate."
  That is an assertion, not a measurement.
- **ControlGS, MGS and FlexGS report no byte figure whatsoever.**
- FlexGS uses byte language ("memory budgets", "desired model size", "fraction of the memory
  footprint") but reports only percentages of Gaussians and, because it must store the full set plus
  GsNet plus a 4D field to serve any ratio, its on-disk size does not shrink with e.

**Implication.** No paper in this set demonstrates the byte-target control our project needs. The
mechanisms transfer (a differentiable soft count, an opacity gate, a schedule), but the quantity
being controlled would have to be replaced by an estimator of the encoded byte size, and none of
these papers has one that is differentiable.

## The one FPS-denominated result
ControlGS Table I page 7 is the only measurement in the set that ties a control value to a
device-side timing constraint: three iGPU tiers, λα = 1e-7 / 4e-7 / 7e-7, all six test scenes above
25 FPS. But the mapping tier → λα is chosen by hand after the fact, and Table I itself shows the
mid-range iGPU running slower than the entry-level one at λα = 1e-7 on two scenes. It is an
existence proof, not a controller.

MGS binds FPS only through AUC_fps, an integral over an envelope, with FPS clipped to [0, 500]. It
reports one absolute FPS number, 493 FPS at 10 % of 1 M splats.

FlexGS reports no FPS at all.

## Budget from the start versus explore then sparsify

| paper | unconstrained exploration before budget binds | approach direction | overshoot |
|---|---|---|---|
| CDGS | 500 of 40 000 iterations (1.25 %) | from below, monotone increase to Ntarget | explicitly none |
| ControlGS | none in the loss, but 6 rounds of 8× uniform splitting | oscillating, expand then contract | yes, up to 8× per round |
| MGS | none, N fixed at 5 M by 3DGS-MCMC from the start | not applicable, budget applied at render time | not applicable |
| FlexGS | 15 000 of 35 000 iterations (43 %) | not applicable, selection over a fixed superset | not applicable |

The two methods that beat their dedicated baselines the hardest at low budgets (ControlGS, FlexGS)
are the two with the most exploration. CDGS, the only strict from-below monotone controller, has
the single largest ablation in this reading set for **restoring** exploration: removing its
500-iteration unconstrained warm-up costs **1.24 dB** (Table VII page 11:
`w/o Initialization 30.90 31.8` versus `Ours full 32.14 31.5`). CDGS attributes that to the
static/dynamic split, but the direction agrees with the Taming-3DGS-versus-GaussianSpa finding.

CDGS also reports that its budget loss is nearly free in quality terms and buys count accuracy:
`w/o Budget loss 32.15 31.7 4.8%` versus `Ours full 32.14 31.5 1.3%`, that is **0.01 dB for 3.5
percentage points of count error**. FlexGS reports the same shape: `Adapt. Sel. w/o Sparsity`
costs 0.07 to 0.17 dB across four ratios (*computed*). **In both papers the count-matching penalty
is the cheap component and the compensation mechanism is the expensive one** (CDGS importance score
0.23 dB, adaptive allocation 0.18 dB, fine-tune phase 0.28 dB; FlexGS Transform Field ~12 dB).

## Elastic versus dedicated, side by side

| paper | elastic model | dedicated comparison | gap at full/high budget | gap at tight budget |
|---|---|---|---|---|
| MGS | prefix truncation of a 5 M set | 3DGS-MCMC trained full-set only | −0.20 dB on Mip-NeRF 360 and T&T; +0.78 dB Deep Blending; +0.09 dB BungeeNeRF | **not measured** (no dedicated model at reduced budgets anywhere in the paper) |
| FlexGS | GsNet plus Transform Field | LightGS pruned and fine-tuned per ratio | −0.204 dB at 15 % on Mip-NeRF360, +0.020 dB at 15 % on T&T, +0.210 dB at 15 % on Zip-NeRF | +0.465 / +0.295 / +1.151 dB at 1 % on Mip-NeRF360 / T&T / Zip-NeRF |

The two patterns are opposite at the endpoints and the reason is the objective weighting. MGS uses
γ = 1 (5:5 prefix:full) and its own sweep shows 9:1 buys 22.2 dB at the 10 % prefix while 1:9 gives
21.1 dB there and 25.5 dB at full, so the endpoint it gives up is chosen. FlexGS samples e from
{0.01, 0.05, 0.10, 0.15} only, never near 1.0, so its full-budget behaviour is unconstrained and its
tight-budget behaviour is heavily supervised.

## Directly reusable mechanisms
1. **CDGS soft-count gate.** cᵢ = clamp((Mᵢ − 0.5)/τ_c + 0.5, 0, 1), N_p = ∑ᵢ cᵢ, α̂ᵢ = cᵢ·αᵢ,
   L_budget = (N_p − Ntarget)², τ_c annealed 1.0 → 0.01 exponentially over the enforcement window,
   then hard binarisation and a fine-tune phase. The opacity modulation is the load-bearing part: it
   is what makes the gradient of the image loss reach the gate. Replacing N_p by a differentiable
   byte estimator gives a byte-targeting controller with the same shape.
2. **CDGS reuse of the Taming 3DGS quadratic per-iteration sub-target**, but with an
   importance-weighted densify/prune policy instead of a gradient heuristic. This is the direct
   answer to "can the count grow while the controller is active": yes, and the paper reports a
   monotone approach from below with no overshoot (page 10).
3. **The two-render stochastic objective**, independently arrived at by MGS (ℓ(G_≤k) + γ·ℓ(G_≤N),
   k = ⌈rN⌉, r ~ Unif(r_min,1)) and FlexGS (|I^e_s − I_GT| + |I_f − I_GT|, e ~ E). Both cost exactly
   two forward passes per step. MGS resamples continuously, FlexGS from a 4-element set. MGS's
   prefix:full weight sweep (page 13) quantifies the trade: about 1.1 dB at the 10 % budget against
   about 0.2 dB at full budget.
4. **ControlGS's decoupling of the control knob from scene scale**: uniform 8× octree splitting with
   α_child = 1 − √(1 − α_parent) to preserve composite opacity, plus an opacity L1 whose weight is
   the only knob. The count then emerges. This is the design to copy if the target quantity is
   allowed to be soft; it is the design to avoid if a hard byte budget must be met.
5. **FlexGS's Transform Field.** A ratio-conditioned displacement field over (X, e) that moves and
   reshapes the surviving Gaussians to patch the holes left by removal. Ablating it collapses PSNR to
   14.57 to 15.12 dB. If a BudgetGS method prunes hard at the end of training, something like this
   is what prevents the collapse.

## Things to be careful about when citing these papers
- **MGS page 9 AUC numbers contradict its own Table 1 on page 8** at four separate points. Use
  Table 1.
- **FlexGS Table 2 and Table 3 disagree by about 0.5 dB** on the same T&T operating points at 5 %
  and 15 %.
- **FlexGS Table 1 row "Ours" at 1 %** repeats LightGS's SSIM (0.6128) and LPIPS (0.4384) exactly.
- **FlexGS Fig. 4** labels a Deep Blending scene (drjohnson) as "T&T".
- **CDGS's byline spells its second author two ways** (Zhenlong Wu / Zhenglong Wu).
- **CDGS reports its best PSNR in the ablation row that removes the budget loss** (32.15 versus
  32.14), so the budget mechanism is not a quality win in that paper, it is a controllability win.
