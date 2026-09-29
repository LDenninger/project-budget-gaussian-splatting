# Literature notes C2: MaskGaussian, PUP 3D-GS, SPARE-GS

Read in full with pymupdf on 2026-09-08. Every number below is as reported in the PDF unless
labelled `computed` (a difference or ratio I formed from reported numbers). Page numbers refer to
the PDF page order. Table rows are copied from the extracted text and interpreted with the
surrounding prose. Quoted passages keep their original punctuation.

---

## Paper 1: MaskGaussian

### 1. Citation
- Title: MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks
- First author: Yifei Liu (Shanghai AI Laboratory, Beihang University), co-corresponding authors Zhihang Zhong and Xiao Sun
- Venue as stated in the PDF: not stated in the body. The only stamp is `arXiv:2412.20522v3 [cs.CV] 14 Jun 2025`. The file name says CVPR 2025.
- arXiv id: 2412.20522

### 2. Base representation and renderer
Vanilla 3DGS (Kerbl et al.) with a modified CUDA rasterizer: "We modify the CUDA rasterization code of 3DGS [14] to integrate our masking into the α-blending process" (p. 6). Each Gaussian carries two learnable mask scores. The mask is applied inside alpha blending, never to opacity or scale. Also demonstrated as a plug-in on Taming-3DGS (Table 4, p. 7) and as a post-training fine-tuner on a frozen 3DGS (Table 5, p. 8).

### 3. Budget handle
- User supplies: the mask-loss weight λm and the iteration window in which it is active. Three settings are used (p. 6): "Ours-α: λm=0.1 from 19,000 to 20,000 iterations; Ours-β: λm=0.0005 from 0 to 30,000 iterations; Ours-γ: λm=0.001 from 0 to 30,000 iterations."
- Classification: KNOB. No count or size is requested. The count is an outcome of λm.
- Achieved versus requested: not applicable, nothing is requested. To match a competitor's count the authors hand-tune λm: "We increase mask penalty λm to match its pruning extent" (p. 8, RadSplat-Light comparison). Same λm (Ours-α) yields 1.205 M on Mip-NeRF360, 0.590 M on Tanks&Temples, 0.694 M on Deep Blending (Table 1, reported), i.e. pruning ratios 62.4 %, 67.7 %, 75.3 % (p. 6, reported).
- The only place a requested count appears is Taming-3DGS's own "Target Budget" in Table 4 (p. 7), where Taming alone lands exactly on target (0.700, 2.100, 3.500 M for targets 0.7, 2.1, 3.5 M, and 4.873 M for 4.9 M) while Taming + MaskGaussian ends below target (0.583, 1.581, 2.401, 3.179 M). Computed: the mask undershoots Taming's target by 17 % to 35 %.

### 4. When and how it binds
Binds during training, continuously, through a stochastic mask that is part of the forward pass.

- Mask: two learnable scores per Gaussian, "apply Gumbel-Softmax[13] to sample one differentiable category out of the 2 scores, denoted as Mi ∈ {0, 1}" (p. 4).
- Forward (masked rasterization, p. 4):
  - Eq. 3: c(x) = ∑[i=1..N] Mi · ci · αi · Ti
  - Eq. 4: T_{i+1} = Mi · (1 − αi) · Ti + (1 − Mi) · Ti
  - All Gaussians are splatted normally and pass the α > 0 filter, masked ones included. "When Mi = 0, the Gaussian's contribution to color is masked out, and its transmittance consumption is skipped."
- Backward, gradient estimator (p. 4 and Appendix 6, p. 11):
  - b_{i+1} = ∑[j=i+1..N] Mj · cj · αj · Tj (Eq. 5, colour rendered behind Gaussian i)
  - Eq. 6: ∂L/∂Mi = αi · Ti · (∂L/∂c(x)) · (ci − b_{i+1})
  - Full form with background term, Eq. 18: ∂L/∂Mi = αi Ti (∂L/∂c(x)) (ci − b_{i+1}) + (−αi · T_{N+1} / (1 − αi · Mi)) (∂L/∂c(x)) c_bg
  - Interpretation (p. 4 to 5): the factor αi · Ti "already encompasses αi · Ti, the importance criterion for score-based prunings [6, 28]", and (ci − b_{i+1}) "represents the benefits of using the i-th Gaussians's color over not using it". A masked Gaussian (Mi = 0) still receives this gradient because it stays in the blending loop.
  - Fig. 1 caption (p. 3): "A masked splat Gi (e.g., i=2 in this figure) does not receive a gradient for αi, and thus does not update its normal attributes, but it receives a gradient for mask mi and updates its existence probability."
  - Gumbel-Softmax versus straight-through estimator (Table 6, p. 8, Mip-NeRF360, PSNR / SSIM / LPIPS / #GS in millions, reported). STE 27.30 / 0.808 / 0.232 / 1.026. Gumbel Softmax 27.44 / 0.811 / 0.226 / 1.520. Gumbel Softmax + larger λm 27.42 / 0.809 / 0.228 / 1.171.
- Loss (p. 5): Eq. 7 Lm = ((1/N) ∑[i=1..N] Mi)², Eq. 8 L = L_render + λm · Lm. "We use squared loss to constrain the average number of Gaussians ... and find it empirically superior to L1 loss" (Table 8, p. 12 confirms L2 beats L1 for this method but not for Compact3DGS).
- Hard removal (p. 5): "To prune low-probability Gaussians with near-zero sampling likelihood, we sample each Gaussian 10 times and remove those that are never sampled. This pruning procedure is applied at every densification step, and every 1000 iterations after densification."
- Interaction with densification: the densification rule itself is untouched, "without altering the 3DGS pipeline such as representations (anchors) and the densification scheme" (p. 2). Removal is synchronised with densification steps (above). Fig. 3 (p. 6, Treehill) and p. 7 state that with the mask active from iteration 0 the count stays lower during densification: "This reduction is already significant during the densification stages of training, where the number of Gaussians tends to increase at a lower level than Compact3DGS". Inference, not stated: because a masked splat gets no αi gradient, it also accumulates no view-space positional gradient in that iteration, which lowers its chance of being cloned or split.
- Masking versus opacity multiplication (Table 7, p. 8, Mip-NeRF360, PSNR / SSIM / LPIPS / #GS in millions, reported). Mask × opacity + smaller λm 27.37 / 0.808 / 0.229 / 1.844. Mask × opacity 27.05 / 0.801 / 0.245 / 0.866. Masked-Raster. 27.44 / 0.811 / 0.226 / 1.520. Masked-Raster. + larger λm 27.42 / 0.809 / 0.228 / 1.171. The paper explains opacity multiplication produces a "death spiral" for unsampled Gaussians (p. 8).

### 5. What size or count means, and MB
- "#GS" is the number of Gaussians in millions (Table 1 caption, p. 6).
- MB is reported once, Table 2 (p. 6), column "size (MB)", format not specified (presumably the raw checkpoint): Compact3DGS 362.7 / 228.3 / 311.1, Ours-β 362.5 / 175.4 / 215.2, Ours-γ 274.2 / 129.8 / 134.9 for Mip-NeRF360 / Tanks&Temples / Deep Blending (reported).
- Peak GPU memory in GB is reported per scene in Table 3 (p. 7).

### 6. Runtime cost
- Hardware: "conduct all experiments including performance evaluation on an NVIDIA RTX 4090" (p. 6).
- Render FPS (Table 1, p. 6, reported): 3DGS 187.8 / 254.5 / 201.3, Compact3DGS 281.1 / 358.9 / 366.2, RadSplat 247.8 / 396.4 / 345.3, MaskGaussian-α 384.7 / 558.3 / 637.1 on Mip-NeRF360 / Tanks&Temples / Deep Blending. Speedups stated as 2.05×, 2.19×, 3.16× (p. 2).
- Training time (Table 2, p. 6, mm:ss, reported): Compact3DGS 19:19 / 10:22 / 16:23, Ours-β 19:44 / 09:52 / 15:47, Ours-γ 17:52 / 08:54 / 14:08. Garden scene with Taming-3DGS (Table 4): 3DGS reference 32:32, Taming at 0.7 M 04:47, Taming + Ours 4:58 (reported).
- Peak GPU memory (Table 3, p. 7, reported), scenes counter / drjohnson / room / truck / flowers: 3DGS 5.41 / 9.03 / 5.55 / 5.70 / 7.79 GB, RadSplat 4.77 / 8.98 / 4.76 / 5.18 / 7.38 GB, Compact3DGS 4.99 / 7.40 / 5.16 / 4.90 / 6.30 GB, Ours-β 3.94 / 7.99 / 3.84 / 4.48 / 6.26 GB, Ours-γ 3.53 / 6.45 / 3.65 / 4.41 / 5.94 GB.

### 7. Evidence on training-time budget versus post-hoc pruning
The paper's thesis (abstract, p. 1): prior methods "deterministically remove Gaussians based on a snapshot of the pruning moment, leading to sub-optimized reconstruction performance from a long-term perspective."

(a) Learned mask during training versus one-shot score pruning at the same moment in training, roughly matched count (Tables 9 to 11, p. 11, reported). LightGaussian trained from scratch with pruning at iteration 20,000, MaskGaussian with λm = 0.1 during 19,000 to 20,000. Columns PSNR / SSIM / LPIPS / #GS (M):
- Mip-NeRF360: 3DGS 27.45 / 0.811 / 0.223 / 3.204, LightGaussian 27.10 / 0.800 / 0.246 / 1.090, Ours 27.44 / 0.811 / 0.227 / 1.205
- Tanks&Temples: 3DGS 23.74 / 0.848 / 0.176 / 1.825, LightGaussian 23.04 / 0.822 / 0.222 / 0.625, Ours 23.72 / 0.847 / 0.181 / 0.590
- Deep Blending: 3DGS 29.53 / 0.903 / 0.243 / 2.815, LightGaussian 27.29 / 0.877 / 0.294 / 0.752, Ours 29.69 / 0.907 / 0.244 / 0.694
- Matched count: approximately (counts within 11 % on Mip-NeRF360, MaskGaussian has fewer Gaussians on the other two). Computed PSNR gaps: +0.34, +0.68, +2.40 dB in favour of the mask.

(b) Both methods post hoc on a frozen 30k 3DGS, 5,000 fine-tuning iterations each (Table 5, p. 8, reported):
- Mip-NeRF360: Baseline 3DGS 27.50 / 0.811 / 0.224 / 3.188, LightGaussian 27.50 / 0.809 / 0.232 / 1.084, MaskGaussian 27.49 / 0.811 / 0.228 / 1.052
- Tanks&Temples: 23.62 / 0.847 / 0.176 / 1.841, LightGaussian 23.75 / 0.843 / 0.187 / 0.626, MaskGaussian 23.75 / 0.847 / 0.178 / 0.579
- Deep Blending: 29.54 / 0.904 / 0.244 / 2.820, LightGaussian 29.55 / 0.902 / 0.250 / 0.959, MaskGaussian 29.71 / 0.905 / 0.246 / 0.667
- Matched count: approximately on Mip-NeRF360 and Tanks&Temples, not on Deep Blending. Note that in the post-hoc setting with fine-tuning the PSNR gap on Mip-NeRF360 vanishes (27.50 versus 27.49), which is the closest this paper gets to a "post-hoc with fine-tuning" control.

(c) Count target from the start (Taming-3DGS) versus mask penalty (Table 4, p. 7, Garden scene, reported). Target Budget (M) 0.7 / 2.1 / 3.5 / 4.9:
- Taming-3DGS PSNR 26.736 / 27.350 / 27.481 / 27.617, #GS (M) 0.700 / 2.100 / 3.500 / 4.873, training time 04:47 / 07:28 / 10:37 / 13:47
- Taming-3DGS + Compact3DGS PSNR 26.694 / 27.318 / 27.477 / 27.551, #GS (M) 0.595 / 1.623 / 2.440 / 3.207, training time 04:51 / 07:25 / 10:06 / 12:57
- Taming-3DGS + Ours PSNR 26.717 / 27.391 / 27.522 / 27.616, #GS (M) 0.583 / 1.581 / 2.401 / 3.179, training time 4:58 / 7:31 / 10:14 / 13:10
- 3DGS (as ref.) PSNR: 27.38, #GS (M): 5.81, training time: 32:32
- Matched count: no, the mask variants end with fewer Gaussians than the budget. Paper's reading (p. 7): "using masks to prune Gaussians is a more effective approach to improving efficiency compared to directly controlling the target number of Gaussians in Taming-3DGS."

(d) Full-training mask versus Compact3DGS, same λm (Table 2, p. 6, reported): Mip-NeRF360 Compact3DGS 27.32 / 0.805 / 0.233 / 1.533 M, Ours-β 27.44 / 0.811 / 0.226 / 1.520 M, Ours-γ 27.42 / 0.809 / 0.228 / 1.171 M. Matched count between Compact3DGS and Ours-β on Mip-NeRF360 (1.533 versus 1.520 M).

(e) Fig. 4 (p. 8) compares with RadSplat-Light at counts near the SfM initialisation, numbers only in the figure.

No table in this paper compares a from-the-start count target against post-hoc pruning to the same count.

### 8. Scene dependence
No explicit sentence. The per-scene table shows it: with the single Ours-α setting, Mip-NeRF360 counts range from 0.328 M (counter) to 2.34 M (bicycle) (Table 12, p. 14, reported). Computed pruning ratios from Table 12: bicycle 5.70 to 2.34 M (58.9 %), bonsai 1.25 to 0.353 M (71.8 %), counter 1.16 to 0.328 M (71.7 %), room 1.56 to 0.366 M (76.5 %), stump 4.65 to 1.90 M (59.1 %). Dataset-level ratios at the same λm are 62.4 %, 67.7 %, 75.3 % (p. 6, reported).

### 9. Main quantitative results (Table 1, p. 6, reported)
Columns PSNR / SSIM / LPIPS / #GS (M) / FPS. MaskGaussian-α uses λm = 0.1 from 19,000 to 20,000 iterations.
- Mip-NeRF360: 3DGS† 27.21 / 0.815 / 0.214 / - / -, 3DGS 27.45 / 0.811 / 0.223 / 3.204 / 187.8, Compact3DGS 27.32 / 0.805 / 0.233 / 1.533 / 281.1, RadSplat 27.45 / 0.811 / 0.223 / 2.184 / 247.8, MaskGaussian-α 27.43 / 0.811 / 0.227 / 1.205 / 384.7
- Tanks&Temples: 3DGS† 23.14 / 0.841 / 0.183, 3DGS 23.74 / 0.848 / 0.176 / 1.825 / 254.5, Compact3DGS 23.61 / 0.846 / 0.180 / 0.960 / 358.9, RadSplat 23.61 / 0.847 / 0.178 / 1.053 / 396.4, MaskGaussian-α 23.72 / 0.847 / 0.181 / 0.590 / 558.3
- Deep Blending: 3DGS† 29.41 / 0.903 / 0.243, 3DGS 29.53 / 0.903 / 0.243 / 2.815 / 201.3, Compact3DGS 29.58 / 0.903 / 0.248 / 1.310 / 366.2, RadSplat 29.55 / 0.903 / 0.244 / 1.515 / 345.3, MaskGaussian-α 29.69 / 0.907 / 0.244 / 0.694 / 637.1
- Strongest two baselines: Compact3DGS (learned mask) and RadSplat (score pruning). MB rows are in field 5 (Table 2).

### 10. Densification details and how the budget modifies it
Standard 3DGS densification, unchanged (p. 2). The mask loss adds a global pressure through Lm, and hard removal of never-sampled Gaussians (10 draws) happens at every densification step and every 1000 iterations afterwards (p. 5). With λm active from iteration 0 (Ours-β, Ours-γ) the count is lower throughout densification (Fig. 3, p. 6, and p. 7). With Ours-α the mask loss is only active in 19,000 to 20,000, after densification has ended, so it acts as a late in-training pruning stage rather than a shaping of growth. The paper does not modify the gradient threshold, the interval, or the opacity reset.

### 11. Stated limitations and future work (verbatim)
The paper has no limitations section. The only forward-looking statement:
- p. 2: "Through extensive experiments on various datasets, we show masked-rasterization substantially outperforms the approach of multiplying masks with Gaussian attributes, such as opacity and scale, offering a superior alternative for 3DGS mask application in future work."

### 12. Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, byte budgets, edge or mobile deployment (verbatim)
- p. 1 (abstract): "While 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and real-time rendering, the high memory consumption due to the use of millions of Gaussians limits its practicality."
- p. 1: "This not only reduces training and rendering speed, which could otherwise be faster, but also leads to significant memory consumption."
- p. 7 (Table 4 caption): "Integration with Taming-3DGS. “Budget” refers to Taming-3DGS's hyperparameter that controls the target number of primitives. Scene: Garden."
- p. 7: "As shown in Tab. 4, using masks to prune Gaussians is a more effective approach to improving efficiency compared to directly controlling the target number of Gaussians in Taming-3DGS."
- p. 7 (Table 3 caption): "Peak GPU Memory Requirement. Our method can prune more Gaussians and preserve equal or better quality, thus saving the most GPU memory."
- p. 8: "Fig. 4 shows that even under extreme pruning conditions with a number of Gaussians similar to the initial SFM points, our method more effectively preserves the rendering quality, making it more adaptable to resource-constrained settings."
- Byte target, bitrate, rate control, FPS target, edge or mobile deployment: none found.

### 13. Code
- URL as printed: https://github.com/kaikai23/MaskGaussian (p. 1)
- Licence: not stated

---

## Paper 2: PUP 3D-GS

### 1. Citation
- Title: PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting
- First authors: Alex Hanson and Allen Tu (equal contribution), University of Maryland, College Park
- Venue as stated in the PDF: not stated. Stamp `arXiv:2406.10219v3 [cs.CV] 24 Mar 2025`. The file name says CVPR 2025, and SPARE-GS cites it as "IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 5949–5958".
- arXiv id: 2406.10219

### 2. Base representation and renderer
Any pretrained vanilla 3D-GS model, unchanged training pipeline. "We implement the Hessian computation as a CUDA kernel in the original 3D-GS codebase [11], then adapt the pruning and refining framework from LightGaussian [4]" (p. 5). Also applied to an EAGLES model (Appendix A.5, p. 12). Renderer is the original 3D-GS rasterizer.

### 3. Budget handle
- User supplies: a pruning percentage per round. Chosen schedule: 80 % in round one, then 50 % of the remainder in round two, 90 % cumulative (p. 6).
- Classification: TARGET on a count ratio, met exactly by construction (sort by score, cut). "Note that the sizes of the pruned scenes in this section are identical because exactly 90% of Gaussians were removed from each of them using our two step prune-refine method." (p. 11). Table 2 caption: "The final sizes are identical."
- Achieved versus requested: exact. Sizes 746.46 MB to 74.65 MB (Mip-NeRF 360), 433.24 MB to 43.33 MB (Tanks & Temples), 699.19 MB to 69.92 MB (Deep Blending), each exactly one tenth (Table 2, p. 6, reported). Fig. 1: playroom 2.65M to 0.265M Gaussians (reported).

### 4. When and how it binds
Post hoc, on a converged model, then fine-tuning. Contribution 1 (p. 2): "A post-hoc pruning technique, PUP 3D-GS, that can be applied to any pretrained 3D-GS model without changing its training pipeline."

- Sensitivity score (Sec. 4.1 to 4.2, p. 3 to 4):
  - L₂ = ½ ∑[φ ∈ P_gt] ‖I_G(φ) − I_gt‖₂² (Eq. 5)
  - Hessian, Eq. 6: ∇²_G L₂ = ∑[φ] ∇_G I_G(φ) ∇_G I_G(φ)ᵀ + (I_G(φ) − I_gt) ∇²_G I_G(φ)
  - On a converged model the residual term vanishes, leaving the Fisher information, Eq. 7: ∇²_G L₂ = ∑[φ] ∇_G I_G(φ) ∇_G I_G(φ)ᵀ. "Note that ∇_G I_G is the gradient over only the reconstructed images, so our approximation only depends on the input poses P_gt and not the input images I_gt." (p. 4)
  - Block diagonal per Gaussian, Eq. 8: H_i = ∇_{G_i} I_G ∇_{G_i} I_Gᵀ. Score Eq. 9: Ũ_i = log|H_i|. Final score restricted to mean and scale, Eq. 10: U_i = log|∇_{x_i,s_i} I_G ∇_{x_i,s_i} I_Gᵀ|, a 6 × 6 matrix per Gaussian. Rotation excluded because it "does not induce a change of 3D geometry when invariances are present" and would need 10 × 10 matrices, 2.78× larger (p. 7). Bayesian reading (Appendix A.1, p. 11): the log determinant is the entropy of the Laplace-approximated posterior block, so the score ranks Gaussians by posterior entropy.
  - Patch-wise (Sec. 4.3, p. 4): Fisher computed on 4 × 4 image patches by rendering at lower resolution, summed over all views. Table 6 (p. 11) ablates 2 × 2 (26.46 dB), 4 × 4 (26.67 dB), 8 × 8 (26.53 dB).
  - Cost: "Computing pruning scores across the entire training view set takes seconds." (p. 5). Memory ∝ N × 36 (p. 8).
- Fine-tuning schedule (Sec. 4.4, p. 4): "Similar to LightGaussian [4], we prune the model and then fine-tune it without further Gaussian densification." 5,000 iterations per round, 10,000 in total for two rounds (p. 6).
- Multi-round: two rounds (80 % then 50 %) beat one round of 90 % with the same 10,000 fine-tuning iterations (Table 4, p. 7, Mip-NeRF 360, reported): 3D-GS 27.47 / 0.8123 / 0.2216 / 95.59 FPS / 746.46 MB, Prune 90% + 10K Fine-tune 26.12 / 0.7761 / 0.2807 / 189.95 / 74.65, Prune 80% + 50% (5K Each) 26.67 / 0.7862 / 0.2719 / 204.81 / 74.65. Three rounds: "substantially worse at lower percentages and only slightly better at higher percentages" (p. 7, Fig. 6, bicycle).

### 5. What size or count means, and MB
"Size (MB)" is the "point cloud size" (Table 1 caption, p. 5), i.e. the uncompressed checkpoint. Count appears in millions only in Fig. 1. MB after Vectree Quantization is reported in Table 3 (p. 7): 14.44 MB (Mip-NeRF 360), 8.49 MB (Tanks & Temples), 13.30 MB (Deep Blending), "smaller than the unpruned 3D-GS scenes by an average of 51.70×, 52.56×, and 51.06×" (p. 7, reported).

### 6. Runtime cost
- Hardware: "Rendering speeds are collected using a Nvidia RTXA4000 GPU in frames per second (FPS)." (p. 5). Rotation-inclusive scores do not fit on the 16 GB A4000 and need an RTX A5000 (p. 7).
- FPS (Table 2, p. 6, reported): Mip-NeRF 360 3D-GS 64.07, LightGaussian 162.12, Ours 204.81. Tanks & Temples 97.86 / 329.03 / 391.10. Deep Blending 66.79 / 234.10 / 301.43. Average speed-up 3.56× (p. 6). Note the 3D-GS FPS on Mip-NeRF 360 is 63.88 in Table 1, 64.07 in Table 2, 95.59 in Tables 4 and 5, 83.88 in Table 6, with no explanation given.
- Training time: not stated. Fine-tuning 5,000 iterations per round. Score computation "seconds" (p. 5).
- GPU memory: N × 36 for the Fisher blocks (p. 8), runs on a 16 GB RTX A4000.

### 7. Evidence on training-time budget versus post-hoc pruning
This paper contains no training-time budget method. All its comparisons are post hoc versus post hoc at exactly matched size.

Quality lost at 80 % and 90 % with and without fine-tuning (Table 1, p. 5, Mip-NeRF 360 means, PSNR / SSIM / LPIPS / FPS / Size MB, reported):
- 3D-GS 27.47 / 0.8123 / 0.2216 / 63.88 / 746.46
- + Prune 80% 21.00 / 0.7075 / 0.3075 / 167.93 / 149.29
- + Refine 26.97 / 0.7991 / 0.2444 / 148.28 / 149.29
- + Prune 50% 22.63 / 0.7021 / 0.3316 / 241.97 / 74.65
- + Refine 26.67 / 0.7862 / 0.2719 / 204.81 / 74.65
- Computed: at 80 % without fine-tuning the loss is 6.47 dB, with fine-tuning 0.50 dB. At 90 % (second cut, before its refine) the loss is 4.84 dB versus the original and 4.34 dB versus the refined 80 % model, with fine-tuning 0.80 dB versus the original. One-round 90 % with 10K fine-tuning loses 1.35 dB (26.12, Table 4). No 88 % operating point is reported. SPARE-GS Table III (p. 13) additionally reports PUP without fine-tuning on vanilla 3DGS input on Mip-NeRF 360 at 436.61 MB / 26.94 dB, 311.87 MB / 24.66 dB, 187.12 MB / 20.62 dB (reported there).

Against LightGaussian at identical size (Table 2, p. 6, reported), PSNR / SSIM / LPIPS / FPS / Size MB:
- MipNeRF-360: 3D-GS 27.47 / 0.8123 / 0.2216 / 64.07 / 746.46, LightGaussian 26.28 / 0.7622 / 0.3054 / 162.12 / 74.65, Ours 26.67 / 0.7862 / 0.2719 / 204.81 / 74.65
- Tanks & Temples: 3D-GS 23.77 / 0.8458 / 0.1777 / 97.86 / 433.24, LightGaussian 23.08 / 0.7950 / 0.2634 / 329.03 / 43.33, Ours 22.72 / 0.8013 / 0.2441 / 391.10 / 43.33
- Deep Blending: 3D-GS 28.98 / 0.8816 / 0.2859 / 66.79 / 699.19, LightGaussian 28.51 / 0.8675 / 0.3292 / 234.10 / 69.92, Ours 28.85 / 0.8810 / 0.3015 / 301.43 / 69.92
- Matched count: yes, exactly. LightGaussian wins PSNR on Tanks & Temples by 0.36 (p. 6).

Against a training-time mask at matched ratio (Appendix A.6, p. 12, verbatim): "Compact-3DGS[14] reports that they prune 58.7% of Gaussians from the Mip-NeRF 360 bonsai scene, slightly improving PSNR from 29.87 to 29.91. After pruning 58.7% of Gaussians with a single round of prune-refine, our method produces a much larger PSNR boost from 32.25 to 32.55." Matched ratio, but different base models (29.87 versus 32.25 dB), so it is not a controlled comparison.

### 8. Scene dependence
The ratio is fixed, so every scene keeps exactly 10 % of its own count and the absolute count follows the base model. No statement about count variation. The chosen 80 % then 50 % split was optimal both on bicycle (Fig. 7, p. 8) and on the Mip-NeRF 360 average (Fig. 8, p. 11). Appendix A.9 (p. 13) notes background regions "observed from fewer viewpoints exhibit higher uncertainty than well-observed foreground regions, making them more susceptible to pruning."

### 9. Main quantitative results
Table 2 rows above (field 7) are the main results at 90 % pruning. Strongest baselines: LightGaussian (only pruning baseline at matched size) and the unpruned 3D-GS. With Vectree Quantization (Table 3, p. 7, reported): Mip-NeRF 360 LightGaussian 24.65 / 0.7302 / 0.3341 / 162.34 FPS / 14.44 MB, Ours 24.93 / 0.7584 / 0.2988 / 205.97 / 14.44. Tanks & Temples LightGaussian 21.88 / 0.7679 / 0.2886 / 331.47 / 8.53, Ours 21.61 / 0.7787 / 0.2670 / 389.18 / 8.49. Deep Blending LightGaussian 27.90 / 0.8586 / 0.3403 / 233.28 / 13.38, Ours 28.24 / 0.8735 / 0.3108 / 300.15 / 13.30. Per-scene PSNR / SSIM / LPIPS / FPS are in Tables 7 to 10 (p. 12).

### 10. Densification details and how the budget modifies it
None. Pruning happens after training and fine-tuning runs "without further Gaussian densification" (p. 4). The budget never touches densification.

### 11. Stated limitations and future work (verbatim)
- p. 8, Section 7: "A limitation of our approach is that it assumes that the scene is converged because the Fisher approximation requires a small L1 residual to be accurate. We find that a small residual is preserved even after removing 80% of the Gaussians with prune-refine, allowing for a second round of pruning, as shown in Section 4.4. Another limitation is that the memory requirement for computing the Fisher matrices is proportional to N × 36, where N is the total number of Gaussians. We do not find this size to be prohibitive for our post-hoc pruning pipeline on 16GB Nvidia RTXA4000 or larger GPUs. Appendix A.7 discusses potential directions for future research related to PUP 3D-GS."
- p. 12 to 13, Appendix A.7: "However, our score (1) does not rely on ground-truth data and (2) can be computed when the L1 loss is sufficiently low earlier in the 3D-GS training pipeline. These properties may provide a useful first step for several potential directions for future research."
- p. 13: "Further optimizations and/or sufficiently powerful hardware could allow our sensitivity score to be used to prune the model during training. This is a promising research direction that could potentially lead to even higher compression and rendering speeds."
- p. 13: "Pruning anchor-based approaches like Scaffold-GS [17] and derived works like Octree-GS [26] is a more difficult problem. Directly pruning Gaussians with our current pipeline will not work because a fixed number of Gaussians are produced for each anchor point; pruning anchor points is ill-advised because they are generated in areas of the scene that do not have sufficient geometry."
- p. 13, Appendix A.9: "A potential strategy for mitigating this trade-off is to reweight the background Gaussian scores using masking, incorporating a tunable parameter to control the balance between foreground and background degradation. We leave this exploration to future work."

### 12. Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, byte budgets, edge or mobile deployment (verbatim)
- p. 1 (abstract): "However, complex scenes can consist of millions of Gaussians, resulting in high storage and memory requirements that limit the viability of 3D-GS on devices with limited resources."
- p. 2: "Since they typically consist of millions of Gaussians, 3D-GS scenes often have high storage and memory requirements that limit their viability on devices with limited resources."
- p. 5 (Fig. 3 caption): "The red dots at (80%, 50%) represent our percentages for 90% compression."
- p. 8 (Fig. 6 caption): "The dotted red line denotes our target 90% cumulative percentage; per-round results are ablated in Figure 7."
- Byte target, bitrate, rate control, FPS target during training, edge or mobile by name: none found. The only target is a count ratio applied post hoc.

### 13. Code
- URL as printed: https://pup3dgs.github.io (p. 1, project page)
- Licence: not stated

---

## Paper 3: SPARE-GS

### 1. Citation
- Title: SPARE-GS: Structural Parsimony and Resource Efficiency for 3D Gaussian Splatting
- First author: Zhang Chen (Northwestern Polytechnical University, and City University of Hong Kong)
- Venue as stated in the PDF: not stated. IEEE journal manuscript format (authors carry "Senior Member, IEEE"), stamp `arXiv:2607.16624v1 [cs.CV] 18 Jul 2026`.
- arXiv id: 2607.16624

### 2. Base representation and renderer
Plug-in on five unmodified pipelines and their own rasterizers: vanilla 3DGS, Mip-Splatting, TamingGS, DashGaussian, FastGS (Table I, p. 9). "rather than altering the specific densification and pruning strategies of the underlying methods, we solely applied our modulation to their native heuristics, with no pipeline-specific hyperparameter tuning" (p. 8). It reuses statistics the base pipeline already tracks (accumulated view-space gradient, visibility count, projected radius, opacity).

### 3. Budget handle
- User supplies: nothing that names a count or size. The theory uses a global count budget B(t), "the maximum number of active primitives allowed in step t" (p. 4), but in the method "we dynamically scale the global budget B(t) according to the current number of active primitives N(t), without disrupting the native optimization trajectory of the base 3DGS pipeline" (p. 6). The scaling rule is not stated. The knobs that move the final count are the pruning opacity threshold τα, the grid resolution K, and the termination thresholds (Table V, p. 15).
- Classification: neither TARGET nor a count KNOB. The count is an emergent outcome of regional rebalancing. τα acts as an indirect knob: τα = 0.01 gives 2,230,887 Gaussians, 0.05 (default) 1,693,989, 0.10 gives 1,313,693 on Mip-NeRF 360 (Table V, reported).
- Achieved versus requested: no requested count exists. Achieved reductions are 35.93 % to 54.26 % on vanilla 3DGS (p. 8, reported).

### 4. When and how it binds
Binds during training, at every structural update (densification or pruning trigger), and through adaptive termination.

- Resource model (Sec. III-A, p. 3 to 4): admissible set Ḡ(t) = A(t) ∪ C(t) of active and candidate primitives (Eq. 3), binary activation σ_i ∈ {0, 1} (Eq. 4). MINLP, Eq. 5: min over Ḡ(t), σ(t) of L(G̃(t)(σ(t))) subject to ∑[i=1..N̄(t)] σ_i ≤ B(t). The resource is denominated in PRIMITIVE COUNT. Not bytes, not FLOPs. The paper links count to the other costs only by assertion: the constraint "limits the structural complexity of the scene representation, which dictates memory consumption, rendering time, training overhead, or storage size" (p. 4).
- Regional relaxation (Sec. III-B, p. 4): K disjoint regions Ω_r, n_r active primitives per region, U_r(n_r) the expected local loss reduction from n_r primitives. Eq. 6: max over {n_r} of ∑[r=1..K] U_r(n_r) subject to ∑[r=1..K] n_r ≤ B(t), n_r ≥ 0. Regional contributions assumed independent and additive. Diminishing marginal utility, Eq. 7: ∂U_r/∂n_r ≥ 0 and ∂²U_r/∂n_r² ≤ 0.
- Optimality condition and derivation (Sec. III-C, p. 4): Lagrangian Eq. 8, J(n, λ, ν) = ∑[r=1..K] U_r(n_r) + λ (B(t) − ∑[r=1..K] n_r) + ∑[r=1..K] ν_r n_r. KKT: stationarity Eq. 9, ∂U_r(n_r*)/∂n_r − λ + ν_r = 0. Primal feasibility Eq. 10, ∑ n_r* ≤ B(t), n_r* ≥ 0. Complementary slackness Eq. 11, λ (B(t) − ∑ n_r*) = 0 and ν_r n_r* = 0. The paper asserts "During training, the budget constraint is binding, meaning λ(t) > 0 and ∑ n_r* = B(t)", and for any active region n_r* > 0 so ν_r = 0, giving Eq. 12: ∂U_r(n_r*)/∂n_r = λ(t) for all r with n_r* > 0. In words (p. 4): "in the optimal state, the marginal utility yielded by a unit increase in the structural budget is uniformly equalized across all active regions, converging to the constant λ(t)." Eq. 13 states the initial skew (marginal utilities unequal), Eq. 14 the steering goal ∂U_1/∂n_1 ≈ ... ≈ ∂U_K/∂n_K → λ(t+δt). "although analytically computing the exact λ(t) is intractable, this optimality condition provides a directional guide" (p. 5).
- Marginal utility proxy (Sec. IV-A, p. 5 to 6): bounding box of active Gaussians split into a 32 × 32 × 32 voxel grid, K = number of non-empty voxels. Eq. 15: ∂U_r/∂n_r ≈ D_r = Q(H_r) + Q(V_r), with Q(m_r) = rank(m_r)/K a rank normalisation into [0, 1]. Eq. 16: H_r = (1/n_r) ∑[i=1..N(t)] (a_i/d_i) I[k_i = r], a_i the accumulated view-space positional gradient, d_i the accumulated visibility count, footprint-normalised "to avoid size-induced bias". Eq. 17: V_r = (1/n_r) ∑ d_i I[k_i = r].
- Allocation algorithm (Sec. IV-B, p. 6): capacity regulariser from the mean squared projected radius, Eq. 18: A_r = (1/n_r) ∑ (ρ_i)² I[k_i = r], C_r = A_r / A_ref with A_ref the median projected area over active regions. Capacity-adjusted demand Q_r = D_r / C_r. Eq. 19: l_r = Q_r / ∑[j=1..K] Q_j. Eq. 20: target quota n̂_r = B(t) l_r, "a global reference allocation that guides, rather than strictly enforces, structural growth". Slack, Eq. 21: s_{k_i} = (n̂_{k_i} − n_{k_i}) / (n̂_{k_i} + 1). Weight, Eq. 22: w_i = Φ(s_{k_i}, l̃_r), "Φ(·) is a bounded sigmoid function that maps the structural deviation into a smooth gating signal, and l̃_r denotes the target distribution ratio l_r normalized against the maximum value across all valid regions." The exact form of Φ is not stated.
- Structural modulation (Sec. IV-C, p. 6 to 7): densification score S̃_i = S_i · w_i (Eq. 23), "the native densification threshold is kept unchanged", w_i > 1 accelerates growth in under-budgeted regions and w_i < 1 suppresses it. Pruning, Eq. 24: P̂_i = I[n_{k_i} > n̂_{k_i}] · I[α_i < τα] · I[a_i/d_i < τ_org/2], τα = 0.05, τ_org the base gradient threshold, merged as P̃_i = P_i ∨ P̂_i (Eq. 25). A cooldown equal to the pruning interval after each opacity reset, and new primitives are protected within the same update. Termination, Eq. 26: ΔL(t) = 1 − L̄(t)/L̄(t−Δt), ΔN(t) = N(t)/N(t−Δt) − 1, stop when 0 ≤ ΔL < ε_L and |ΔN| < ε_N for ℓ consecutive intervals, ε_L = 0.008, ε_N = 0.01, ℓ = 2. Algorithm 1 (p. 7) runs Parts 1 and 2 only when a structural update is triggered.
- Overhead (p. 15): "the accumulated budget-estimation time ranges from 0.33 s to 0.93 s over the entire training process. This constitutes a mere 0.073% of the total training time on average".

### 5. What size or count means, and MB
Table I "Gaussians" is the absolute final primitive count. No MB is reported for SPARE-GS models on their own. Tables II and III (p. 12 to 13) report "Size (MB)" for the outputs of downstream compressors and pruners at three operating points, both from vanilla input and from SPARE-GS input, plus BD-Rate and BD-PSNR.

### 6. Runtime cost
- Hardware: "all baselines and our adapted versions were executed on the same hardware setup (Intel i9 CPU, RTX 3090 GPU)" (p. 8).
- Training time (Table I, p. 9, seconds, reported): vanilla 3DGS Mip-NeRF 360 1327.08 to 939.23 (−29.23 %), Deep Blending 1257.89 to 779.11 (−38.06 %), Tanks & Temples 718.15 to 485.66 (−32.37 %), BungeeNeRF 2364.79 to 1478.60 (−37.47 %). Average over all pipelines −23.81 % time, −30.38 % Gaussians.
- Render FPS: not reported. GPU memory: not reported.

### 7. Evidence on training-time budget versus post-hoc pruning
The paper does not compare "train to a count" against "prune to the same count". It compares "regulated training then post-hoc pruning" against "unregulated training then the same post-hoc pruning", at three sizes each, with no fine-tuning ("we evaluated three pruning techniques (PUP [28], REFINE [29], and RAP [30]) without fine-tuning", p. 11).

Table III (p. 13, reported), columns Size (MB) / Time (s) / PSNR / SSIM / LPIPS, three operating points per input, then ΔTime / BD-Rate / BD-PSNR:
- PUP, Mip-NeRF 360, vanilla input: 436.61 / 137.39 / 26.94 / 0.807 / 0.227, 311.87 / 132.38 / 24.66 / 0.781 / 0.248, 187.12 / 130.16 / 20.62 / 0.703 / 0.305. SPARE-GS input: 281.21 / 108.93 / 26.39 / 0.798 / 0.243, 200.87 / 108.40 / 23.05 / 0.753 / 0.275, 120.52 / 108.92 / 18.87 / 0.651 / 0.343. −18.42 % / −22.18 % / +2.04 dB.
- PUP, Deep Blending, vanilla: 410.46 / 110.27 / 29.77 / 0.906 / 0.239, 293.19 / 107.79 / 29.08 / 0.898 / 0.251, 175.91 / 105.55 / 25.67 / 0.860 / 0.296. SPARE-GS: 186.05 / 73.00 / 29.61 / 0.906 / 0.248, 132.90 / 72.25 / 27.65 / 0.884 / 0.275, 79.74 / 71.58 / 21.73 / 0.812 / 0.344. −33.00 % / −39.48 % / +3.57 dB.
- PUP, Tanks & Temples, vanilla: 258.98 / 65.04 / 23.06 / 0.840 / 0.182, 184.98 / 67.11 / 21.38 / 0.813 / 0.207, 110.99 / 62.05 / 18.06 / 0.742 / 0.265. SPARE-GS: 162.26 / 49.02 / 22.96 / 0.834 / 0.196, 115.90 / 50.01 / 20.24 / 0.795 / 0.231, 69.54 / 48.44 / 16.54 / 0.708 / 0.295. −24.06 % / −26.16 % / +2.03 dB.
- REFINE, Mip-NeRF 360: vanilla 436.61 / 5.21 / 27.25, 311.87 / 4.44 / 26.21, 187.12 / 4.48 / 23.55. SPARE-GS 281.06 / 3.46 / 26.65, 200.91 / 3.73 / 25.20, 120.57 / 2.33 / 21.89. BD-Rate −18.55 %, BD-PSNR +1.07 dB.
- RAP, Mip-NeRF 360: vanilla 436.61 / 78.81 / 27.37, 311.87 / 74.15 / 26.47, 187.12 / 71.81 / 24.08. SPARE-GS 281.21 / 47.62 / 27.02, 200.87 / 59.70 / 25.70, 120.52 / 40.91 / 23.12. BD-Rate −21.54 %, BD-PSNR +1.08 dB.
- Matched size: approximately, by reading across operating points. Computed from the PUP Mip-NeRF 360 rows: vanilla pruned to 311.87 MB gives 24.66 dB while SPARE-GS pruned to 281.21 MB gives 26.39 dB (+1.73 dB at 10 % smaller size), vanilla at 187.12 MB gives 20.62 dB while SPARE-GS at 200.87 MB gives 23.05 dB (+2.43 dB at 7 % larger size). Summary sentence (p. 13): "These consistent trends across different pruning strategies suggest that regulating structural growth during training provides a more compact input representation for subsequent post-training simplification."
- Caveat for the hypothesis: these gains are measured without fine-tuning, and the vanilla operating points sit at higher pruning ratios of their larger base model, so part of the gap is the pruner being pushed harder rather than capacity being placed better.

Compression side (Table II, p. 12, reported), first operating point per dataset, Size MB / Time s / PSNR / SSIM / LPIPS: LightGaussian Mip-NeRF 360 vanilla 407.56 / 101.25 / 27.19 / 0.807 / 0.228 versus SPARE-GS 244.08 / 85.89 / 27.05 / 0.802 / 0.244, BD-Rate −27.78 %. FCGS Mip-NeRF 360 vanilla 41.68 / 386.81 / 27.35 / 0.804 / 0.233 versus SPARE-GS 34.08 / 181.79 / 27.28 / 0.802 / 0.239, BD-Rate −24.34 %, BD-PSNR +0.17 dB. Full BD ranges (p. 11 to 13): LightGaussian −27.78 % to −55.38 %, SOG −35.60 % to −55.99 %, MesonGS −4.30 % to −21.35 %, FCGS −24.34 % and −53.99 %, GHAP −9.29 % to −39.07 %.

Fixed global count budget versus regional balancing: on TamingGS, which already trains under "a prescribed global Gaussian budget" (p. 2), SPARE-GS still removes 10.73 % to 55.54 % of primitives with PSNR within ±0.06 dB on Mip-NeRF 360 and Tanks & Temples and +0.14 dB on Deep Blending (p. 8, Table I). Not matched count.

Process evidence (p. 16): "The average reduction in primitive count widens steadily from 8.7% at 1.5K iterations to 26.5% at 6K, ultimately reaching 35.8% by the end of the densification stage (15K)." and "This steady divergence validates that SPARE-GS is not merely a post-hoc pruning filter for an already overgrown model."

### 8. Scene dependence
- p. 8: "This result indicates that SPARE-GS does not enforce uniform compression; instead, it adapts the structural allocation to the demand of the underlying scene and pipeline."
- p. 16: "Scene-wise reductions at 15K iterations remain highly consistent, ranging from 29.7% to 48.0%."
- TamingGS on BungeeNeRF: count change −0.18 %, time +0.08 %, PSNR +0.32 dB (p. 8, reported), so the same setting can leave the count untouched on one dataset and cut it by 55.54 % on another.
- Failure cases (Table VII, p. 16, reported): bonsai ΔGaussians −33.71 %, ΔTime −19.13 %, ΔPSNR −0.28. flowers −32.73 %, −24.97 %, −0.20. garden −33.24 %, −44.77 %, −0.24.

### 9. Main quantitative results (Table I, p. 9, reported)
Columns Time (s) / Gaussians / PSNR / SSIM / LPIPS.
- Vanilla 3DGS, Mip-NeRF 360: Baseline 1327.08 / 2,640,494 / 27.52 / 0.813 / 0.221. + SPARE-GS 939.23 / 1,691,678 / 27.44 / 0.809 / 0.234. Δ −29.23 % / −35.93 % / −0.08 / −0.004 / +0.013.
- Vanilla 3DGS, Deep Blending: Baseline 1257.89 / 2,458,697 / 29.63 / 0.902 / 0.241. + SPARE-GS 779.11 / 1,124,510 / 29.77 / 0.906 / 0.246. Δ −38.06 % / −54.26 % / +0.14 / +0.004 / +0.005.
- Vanilla 3DGS, Tanks & Temples: Baseline 718.15 / 1,572,798 / 23.83 / 0.850 / 0.171. + SPARE-GS 485.66 / 973,619 / 24.01 / 0.849 / 0.181. Δ −32.37 % / −38.10 % / +0.18 / −0.001 / +0.010.
- Strongest baseline 1, TamingGS: Mip-NeRF 360 Baseline 871.08 / 1,416,946 / 27.82 / 0.805 / 0.240, + SPARE-GS 716.64 / 1,052,388 / 27.77 / 0.808 / 0.238. Deep Blending 597.56 / 1,178,605 / 29.97 / 0.907 / 0.246, + SPARE-GS 550.27 / 1,052,175 / 30.11 / 0.911 / 0.243. Tanks & Temples 848.80 / 2,206,907 / 24.47 / 0.863 / 0.156, + SPARE-GS 518.31 / 981,097 / 24.41 / 0.860 / 0.167.
- Strongest baseline 2, FastGS: Mip-NeRF 360 Baseline 269.08 / 558,202 / 27.29 / 0.800 / 0.254, + SPARE-GS 248.83 / 508,898 / 27.26 / 0.797 / 0.259. Deep Blending 168.76 / 316,890 / 30.14 / 0.908 / 0.254, + SPARE-GS 158.02 / 281,643 / 30.09 / 0.908 / 0.256. Tanks & Temples 151.04 / 247,905 / 23.48 / 0.837 / 0.212, + SPARE-GS 139.74 / 221,914 / 23.46 / 0.834 / 0.216.
- Average over all rows: Δ −23.81 % time, −30.38 % Gaussians, +0.01 PSNR, 0.000 SSIM, +0.005 LPIPS.
- No MB for these models. Ablation (Table IV, p. 14, Mip-NeRF 360): Baseline 2.64M / 1314 s / 27.51, w/o BMD 1.88M / 1091 s / 27.50, w/o RAP 2.32M / 1161 s / 27.48, w/o EDT 1.68M / 964 s / 27.45, Full 1.69M / 928 s / 27.46 (reported).

### 10. Densification details and how the budget modifies it
The base pipeline's densification criterion and threshold stay as they are. SPARE-GS multiplies the native per-primitive score by w_i (Eq. 23), adds a conservative extra pruning mask that fires only in over-budgeted regions for low-opacity, low-gradient primitives (Eq. 24 to 25), and stops training early on loss and count stability (Eq. 26). Regional statistics and quotas are recomputed only when a structural update triggers (Algorithm 1). Fig. 7 (p. 16) shows the count curve over 0 to 15k iterations per Mip-NeRF 360 scene lying below the baseline throughout. Ablation attribution (p. 14): removing BMD raises the count from 1.69M to 1.88M and time from 928.40 s to 1090.97 s, removing RAP raises the count to 2.32M and time to 1160.72 s, removing EDT raises time to 964.34 s with the count unchanged.

### 11. Stated limitations and future work (verbatim)
- p. 16: "Failure Cases and Limitations. Table VII presents representative cases where the efficiency gains of SPARE-GS are accompanied by more noticeable quality degradation. Across the three scenes, SPARE-GS reduces the Gaussian count by approximately 33% and the training time by 19.13% to 44.77%, while PSNR decreases by 0.20 dB to 0.28 dB. These cases may reflect limitations of the current heuristic demand estimation and capacity calibration in scenes with complex occlusions, fine details, or uneven view coverage. Future work could improve robustness by incorporating uncertainty-aware, occlusion-aware, or learned demand estimation, together with more conservative budget control in low-confidence regions."
- p. 16 (conclusion): "Future work will explore more accurate uncertainty-aware or learned demand estimation to improve robustness in structurally complex scenes."
- p. 5: "Therefore, although analytically computing the exact λ(t) is intractable, this optimality condition provides a directional guide."

### 12. Sentences touching a size, byte, bitrate, memory or FPS target during training, rate control, byte budgets, edge or mobile deployment (verbatim)
- p. 1: "This dense primitive count increases model size, posing challenges for mobile deployment and large-scale scene modeling, while raising memory consumption and training time [11]–[14]."
- p. 1: "Consequently, advancing the practicality of 3DGS becomes a problem of comprehensive cost control across training time, computational resources, memory footprint, and storage capacity, a challenge that has driven recent efforts to explore various optimization strategies [15], [16]."
- p. 2: "Each Gaussian primitive represents a unit of computational and memory cost, while different spatial regions exhibit varying representational demands."
- p. 2: "GaussianPro [22] progressively propagates Gaussians using multi-view stereo priors, whereas TamingGS [32] performs constructive densification under a prescribed global Gaussian budget."
- p. 3: "Representation compression targets the storage, transmission, and deployment costs of 3DGS."
- p. 4 (after Eq. 5, sum written out because the extraction garbles it): "where B(t) is the maximum number of active primitives allowed in step t in the scene, and ∑[i=1..N̄(t)] σ_i(t) ≤ B(t) limits the structural complexity of the scene representation, which dictates memory consumption, rendering time, training overhead, or storage size."
- p. 4: "Compared with Eq. (2), Eq. (5) models Gaussian primitives as finite resources bounded by B(t)."
- p. 6: "Target Quota Allocation. To regulate structural evolution globally, we dynamically scale the global budget B(t) according to the current number of active primitives N(t), without disrupting the native optimization trajectory of the base 3DGS pipeline."
- p. 8: "For downstream tasks, we further reported the Bjøntegaard Delta (BD) metrics to measure rate-distortion efficiency, alongside coding times to quantify the associated computational overhead."
- p. 16: "This steady divergence validates that SPARE-GS is not merely a post-hoc pruning filter for an already overgrown model."
- Byte target, bitrate target, rate control in the coding sense, FPS target during training, edge deployment by name: none found. "Budget" is always a primitive count, and B(t) is never set by the user.

### 13. Code
- URL as printed: https://zhangchen2022.github.io/SPARE-GS.github.io/ (p. 1, "for the source code and more visual results"). No repository URL is printed in the PDF and the PDF carries no hyperlink annotations.
- Licence: not stated

---

## Cross-paper reading for the project hypothesis

Hypothesis under test: training with the budget known from the start places capacity better than post-hoc pruning to the same count.

- Direct matched-count test of "from-the-start target versus post-hoc": absent in all three papers.
- Closest support: SPARE-GS Table III (regulated growth then prune, versus unregulated growth then the same prune, without fine-tuning) with BD-PSNR +2.03 to +3.57 dB for PUP, +1.07 to +2.63 dB for REFINE, +1.08 to +1.46 dB for RAP. MaskGaussian Tables 9 to 11 (learned probabilistic mask in 19k to 20k versus one-shot LightGaussian at 20k) with +0.34, +0.68, +2.40 dB at counts within about 10 %.
- Closest counter-evidence: MaskGaussian Table 5, where LightGaussian and MaskGaussian both run post hoc with 5,000 fine-tuning iterations and tie on Mip-NeRF 360 PSNR (27.50 versus 27.49 at 1.084 versus 1.052 M). PUP Table 1, where post-hoc pruning to 80 % with 5,000 fine-tuning iterations costs only 0.50 dB, and to 90 % 0.80 dB. MaskGaussian Table 4, where Taming-3DGS's from-the-start count budget at 2.1 M (27.350 dB) is beaten by mask pruning ending at 1.581 M (27.391 dB) on Garden.
- No paper trains toward a byte, MB, bitrate or FPS target. Taming-3DGS, cited by two of them, is the one method with a count target fixed before training, and MaskGaussian Table 4 shows it lands exactly on that count.
