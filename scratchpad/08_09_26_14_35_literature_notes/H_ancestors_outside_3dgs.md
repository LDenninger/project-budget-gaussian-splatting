# H. Ancestors outside 3DGS: rate control, variable-rate fields, hardware-aware NAS, budgeted compression, constrained optimisation

Date: 2026-09-08. Purpose: identify and verify the papers outside 3DGS that a BudgetGS
training method descends from, so the paper cites them correctly, and record the exact mechanism
each one provides.

Our setting, restated for the mapping paragraphs: the model is a per-scene 3DGS point set with N
primitives. The rate R is the total byte count of the encoded scene. R can be measured exactly at
any time by running the encoder. N changes during training through densification and pruning.
Every 3DGS compression method trains D + λ·R with λ a fixed knob. We want the mechanism that turns
the knob into a target: R → R_target during training, with quality maximised.

## 0. Verification protocol

- arXiv papers: every id was resolved through `https://export.arxiv.org/api/query?id_list=...`
  and the title, author list, submission date, comment and abstract were read from the Atom feed.
  A title stated below is the title returned by the API.
- Non-arXiv papers: DOI resolved through `https://doi.org/<doi>` with CSL-JSON content
  negotiation, and cross-checked through Crossref and OpenAlex. Abstracts for the IEEE papers came
  from OpenAlex and Europe PMC.
- Venues: from the arXiv comment field where the authors state it, otherwise from OpenAlex,
  Crossref, Semantic Scholar, or the proceedings page itself (PMLR, JMLR, papers.nips.cc,
  iclr.cc, neurips.cc).
- Mechanism details beyond the abstract: extracted from the PDF text of the paper (pypdf), from
  arXiv HTML, or from the reference software source. Each such case says so.
- Anything that could not be confirmed is in Section 8, with the reason.

## 1. The six named citations

| label | phrase | citation | id |
|---|---|---|---|
| (a) | λ as a controlled variable driven by measured rate | Bin Li, Houqiang Li, Li Li, Jinlei Zhang, "λ Domain Rate Control Algorithm for High Efficiency Video Coding", IEEE TIP 23(9), 2014 | DOI 10.1109/TIP.2014.2336550 |
| (b) | one model, many rates | Yoojin Choi, Mostafa El-Khamy, Jungwon Lee, "Variable Rate Deep Image Compression With a Conditional Autoencoder", ICCV 2019. Field-side instance: Towaki Takikawa et al., "Variable Bitrate Neural Fields", SIGGRAPH 2022. Architecture-side instance: Han Cai et al., "Once-for-All", ICLR 2020 | arXiv 1909.04802, 2206.07707, 1908.09791 |
| (c) | latency predictor as a differentiable resource constraint | Han Cai, Ligeng Zhu, Song Han, "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware", ICLR 2019. Co-cite Bichen Wu et al., "FBNet", CVPR 2019 for the lookup-table form | arXiv 1812.00332, 1812.03443 |
| (d) | budget-aware regularisation for pruning | Carl Lemaire, Andrew Achkar, Pierre-Marc Jodoin, "Structured Pruning of Neural Networks with Budget-Aware Regularization", CVPR 2019 | arXiv 1811.09332, DOI 10.1109/cvpr.2019.00932 |
| (e) | memory budget in bytes inside a training loss | Stefan Uhlich et al., "Mixed Precision DNNs: All you need is a good parametrization", ICLR 2020, Eq. (8a) and (10). Earlier general-cost form: Tom Veniat, Ludovic Denoyer, "Learning Time/Memory-Efficient Deep Architectures with Budgeted Super Networks", CVPR 2018, Eq. (2) | arXiv 1905.11452, 1706.00046 |
| (f) | dual ascent on a Lagrange multiplier during SGD | Jose Gallego-Posada et al., "Controlled Sparsity via Constrained Optimization or: How I Learned to Stop Tuning Penalties and Love Constraints", NeurIPS 2022, for the applied form with a resource target. Theory with non-differentiable constraints: Andrew Cotter, Heinrich Jiang, Karthik Sridharan, "Two-Player Games for Efficient Non-Convex Constrained Optimization", ALT 2019 (PMLR 98). Origin: John C. Platt, Alan H. Barr, "Constrained Differential Optimization", NIPS 1987 | arXiv 2208.04425, 1804.06500, papers.nips.cc 1987 |

Reasoning for each choice is in the family sections. Two corrections to the brief: the Cotter
two-player paper is ALT 2019 (PMLR volume 98, pages 300 to 332), not ICML 2019, and no paper titled
"Constrained optimization for deep networks with PID control" exists under that name, the nearest
verified works being Stooke et al. 2020 and Sohrabi et al. 2024 (Section 6).

## 2. Family 1: rate control in video and image coding

### 2.1 Li, Li, Li, Zhang 2014, the HEVC R-λ model (citation (a))

- DOI 10.1109/TIP.2014.2336550. Title: "λ Domain Rate Control Algorithm for High Efficiency Video
  Coding". First author: Bin Li. IEEE Transactions on Image Processing, vol. 23, no. 9, pp.
  3841 to 3854, September 2014. Verified through doi.org, OpenAlex, Europe PMC (PMID 25020096).
- Abstract, verbatim excerpt: "Most of existing rate control algorithms are based on the R-Q model
  ... we find that there exists a more robust correspondence between R and the Lagrange multiplier
  λ. Therefore, in this paper, we propose a novel λ-domain rate control algorithm based on the R-λ
  model, and implement it in the newest video coding standard high efficiency video coding (HEVC).
  ... The proposed λ-domain rate control algorithm has already been adopted by Joint Collaborative
  Team on Video Coding and integrated into the HEVC reference software."
- Mechanism. R is modelled per picture as a power law in λ, and λ is the variable the controller
  sets. The reference implementation (HM, `source/Lib/TLibEncoder/TEncRateCtrl.cpp`, fetched from
  `vcgit.hhi.fraunhofer.de/jvet/HM`, master) carries the exact law. Estimation:
  `estLambda = alpha * pow(bpp, beta)`, so λ = α·bpp^β with bpp the target bits per pixel for the
  picture. Target bits come from a bit budget: `m_targetBits = totalFrames * targetRate /
  frameRate`, decremented by the actual bits of each coded picture (`m_bitsLeft -= bits`), and
  redistributed over the remaining pictures. After coding a picture the model is corrected from the
  measured rate: `calLambda = α·bpp_actual^β`, clipped to [λ_in/10, 10·λ_in], then
  `alpha += δ_α · (ln λ_in − ln calLambda) · alpha` and
  `beta += δ_β · (ln λ_in − ln calLambda) · ln(bpp_actual)` with ln bpp clipped to [−5, −0.1],
  followed by clipping of α and β to fixed ranges. The step sizes δ_α, δ_β are presets from
  (0.01, 0.005) to (0.4, 0.2) selected by an adaptation-speed option. QP follows from λ by
  `QP = round(4.2005·ln λ + 13.7122)`. The update is a multiplicative correction in the log domain
  driven by the mismatch between the λ that was used and the λ the model would have predicted for
  the bits actually produced. It is an integral-type feedback on ln λ with a one-parameter-family
  model in between.
- Mapping to 3DGS. The picture becomes a training window (a fixed number of iterations, or the
  interval between two densification events), bpp becomes bytes per primitive r = R/N, and the
  encoder gives R exactly instead of through an entropy estimate. The power law λ = α·r^β is
  plausible for the per-primitive byte cost at fixed N, and the log-domain update transfers
  unchanged. What does not transfer is the assumption that the pixel count is constant: R = N·r,
  and N moves with densification, so the model must be written on r with N tracked separately, and
  α, β must be re-estimated after each densification step in the way HEVC re-estimates after a
  scene change.

### 2.2 He and Mitra 2002, ρ-domain rate control

- DOI 10.1109/TCSVT.2002.804883. Title: "Optimum bit allocation and accurate rate control for video
  coding via ρ-domain source modeling". First author: Zhihai He. IEEE TCSVT vol. 12, no. 10, pp.
  840 to 849, October 2002. Verified through doi.org and OpenAlex.
- DOI 10.1109/TCSVT.2002.805511. Title: "A linear source model and a unified rate control algorithm
  for DCT video coding". First author: Zhihai He. IEEE TCSVT vol. 12, no. 11, pp. 970 to 982,
  November 2002. Verified the same way.
- Abstract, verbatim excerpts: "the coding rate R and distortion D are considered as functions of
  ρ which is the percentage of zeros among the quantized transform coefficients ... the rate
  function R(ρ) is approximately linear" (804883). "there is always a linear relationship between
  the coding bit rate R and the percentage of zeros among the quantized transform coefficients ...
  The physical meaning of the model parameter is also discussed. We show that it is directly
  related to the image content and is a measure of picture complexity. In video coding, we propose
  an adaptive estimation scheme to estimate this model parameter" (805511).
- Mechanism. Instead of modelling R as a function of the quantiser, model it as a linear function
  of a count, the fraction ρ of coefficients quantised to zero. The slope is one scalar per picture,
  re-estimated from the previous coded pictures. To hit a target T, invert the line to get the
  target ρ, then pick the quantiser that produces that ρ (a monotone table computed from the
  coefficient histogram). Rate control becomes a one-dimensional root find on a count.
- Mapping to 3DGS. The count that drives bytes is N (and, inside a primitive, the number of
  non-zero or non-pruned attributes). A linear model R ≈ θ·N_active with θ re-estimated after each
  encoder call is the ρ-domain idea transplanted, and it is the natural model when the controlled
  variable is a pruning threshold rather than λ. The adaptation needed is the same as in 2.1: N is
  moved by densification as well as by the controller, so the slope θ (bytes per active primitive)
  has to be estimated on r = R/N.

### 2.3 MPEG-2 Test Model 5 (primary document not fetched, see Section 8)

- The TM5 document itself (ISO/IEC JTC1/SC29/WG11, "Test Model 5", 1993) is a committee document
  with no DOI, and no primary copy was reachable (the mpeg.org MSSG page returns 404, the Wayback
  Machine is blocked in this environment). The description below is taken from a secondary source
  that was fetched and read: Zongze Wu, Shengli Xie, Kexin Zhang, Rong Wu, "Rate Control in Video
  Coding", chapter 4 of "Recent Advances on Video Coding", ed. Javier Del Ser Lorente, InTech, June
  2011, ISBN 978-953-307-181-7. Its reference list cites TM5 as "ISO/IEC JTC1/SC29/WG11, 'Test
  Model 5', ... MPEG 93/457, Document AVC-491, April 1993".
- Mechanism, from the chapter: "the rate control model comprises the following three steps: 1.
  Target bit allocation ... allocates bits for given Group of Pictures (GOP) based on the target bit
  rate and the number of frames in the GOP. Then before encoding of each frame, it allocates bits
  for that frame based on the frame type (I, P or B), the complexity measure, the remaining number
  of bits in the current GOP. 2. Rate control ... a quantization parameter Q is computed for the
  macroblock j under consideration based on the difference between the allocated bits and the
  actually generated bits till the encoding of previous macroblock in this picture. 3. Adaptive
  quantization ... an 'activity measure' of the macroblock is found using variance of the four
  sub-blocks". The complexity measure of a picture type is the product of its last bit count and its
  last average quantiser, and the reference quantiser is proportional to the fullness of a virtual
  buffer that accumulates bits spent minus bits allocated.
- Mapping to 3DGS. TM5 is the ancestor of every "measure, compare to allocation, correct the knob
  proportionally" loop. Its complexity measure X = S·Q (bits times quantiser) is the cheapest
  possible rate model, one product per unit. In 3DGS the analogue is X = R·λ per training window,
  and the virtual buffer is the running difference between bytes reached and the byte schedule.

### 2.4 x264: ABR, CRF and two-pass

- Source: Loren Merritt, "A qualitative overview of x264's ratecontrol methods", file
  `doc/ratecontrol.txt` in the x264 repository (read from the GitHub mirror `mirror/x264`, master).
  Not a paper, no DOI. The document opens with a historical note that MB-tree and per-row VBV have
  since replaced parts of it.
- Mechanism, verbatim excerpts. Two-pass: "(1) Before starting the 2nd pass, select the relative
  number of bits to allocate between frames ... The default formula ... is 'complexity ** 0.6',
  where complexity is defined to be the bit size of the frame at a constant QP (estimated from the
  1st pass). (2) Scale the results of (1) to fill the requested total size. ... this is an iterative
  process. (3) ... After each frame, update future QPs to compensate for mispredictions in size. If
  the 2nd pass is consistently off from the predicted size ... then we multiply all future frames'
  qscales by the reciprocal of the error." One-pass ABR: "The scaling factor is chosen to be the one
  that would have resulted in the desired bitrate if it had been applied to all frames so far. (3)
  Overflow compensation is the same as in 2pass." CRF: "(2) The scaling factor is a constant based
  on the --crf argument. (3) No overflow compensation is done."
- Mapping to 3DGS. CRF is exactly the current 3DGS practice: a constant λ knob and whatever size
  results. ABR is the target-size mode: the same allocation rule, but the global scale is re-solved
  from the bytes measured so far, and a multiplicative correction is applied to all future
  quantisers by the reciprocal of the size error. Two-pass is "train once at fixed λ to measure the
  R(λ) curve of this scene, then train again with the scale that fills the budget", and it is the
  simplest baseline for a BudgetGS method. The reciprocal-error correction transfers as is,
  applied to λ (or to the pruning threshold) between training windows.

### 2.5 Rate control for learned image and video compression

Verified through the arXiv API unless stated otherwise.

- arXiv 1909.04802. "Variable Rate Deep Image Compression With a Conditional Autoencoder". Yoojin
  Choi. ICCV 2019. Citation (b). Abstract: "We provide two rate control parameters, i.e., the
  Lagrange multiplier and the quantization bin size, which are given as conditioning variables to
  the network. Coarse rate adaptation to a target is performed by changing the Lagrange multiplier,
  while the rate can be further fine-tuned by adjusting the bin size". Mechanism: one network is
  trained over a set of λ values with λ as an input, so at test time λ is a continuous input and a
  target rate is reached by searching λ. Mapping: a 3DGS scene could in principle be trained once
  with a λ-conditioned entropy model and decoded at several rates, but the per-scene point set has
  no shared weights across scenes, so the transferable idea is narrower: keep λ an input that can be
  moved during training rather than a compile-time constant.
- arXiv 2003.02012. "Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation". Ze
  Cui. CVPR 2021. Mechanism: gain units (per-channel scaling of the latent before quantisation) give
  discrete rates in one model, exponential interpolation of gain vectors gives continuous rates.
  Mapping: the per-attribute quantisation step of a 3DGS encoder is the direct analogue of a gain
  vector, and interpolating it is the cheapest continuous rate knob available after training.
- arXiv 1511.06085. "Variable Rate Image Compression with Recurrent Neural Networks". George
  Toderici. arXiv 2015 (submitted to ICLR 2016 per the comment). Mechanism: progressive coding, "the
  more bits are sent, the more accurate the image reconstruction". Mapping: the earliest "one model,
  many rates" in learned compression, and the progressive prefix idea reappears in VBNF (Section 3).
- arXiv 2103.15726. "Slimmable Compressive Autoencoders for Practical Neural Image Compression".
  Fei Yang. CVPR 2021. Mechanism: one autoencoder trained at several widths, each width a rate and
  a complexity. Mapping: rate and compute budget handled by one nested model, the NAS-side view of
  (b).
- arXiv 2303.05744. "QVRF: A Quantization-error-aware Variable Rate Framework for Learned Image
  Compression". Kedeng Tong. 2023. Mechanism: a univariate quantisation regulator coupled with
  predefined λ values gives wide-range variable rate in one model.
- "Neural Rate Control for Learned Video Compression". Yiwei Zhang, Guo Lu, Yunuo Chen, Shen Wang,
  Yibo Shi, Jing Wang, Li Song. ICLR 2024. Verified through the ICLR virtual page
  (`iclr.cc/virtual/2024/poster/19481`), OpenReview id 42lcaojZug. No arXiv id was found for it.
  Abstract: "we first design a rate allocation model to assign optimal bitrates to each frame ...
  Then, we propose a deep learning-based rate implementation network to perform the rate-parameter
  mapping, precisely predicting coding parameters for a given rate." Mechanism: a learned inverse
  model rate → λ replaces the analytic R-λ model. Mapping: in 3DGS the inverse map could be learned
  across scenes offline, but with an exact encoder available per scene the analytic model with online
  correction (2.1) is cheaper and needs no training set.
- arXiv 2412.18834. "Adaptive Rate Control for Deep Video Compression with Rate-Distortion
  Prediction". Bowen Gu. 2024. Mechanism (from the arXiv HTML): restates the conventional model as
  R = α₁·λ^β₁ (its Eq. 3) and replaces the online parameter update by two neural predictors of R(λ)
  and D(λ) per frame, then solves for λ per frame for a target and uses binary search over a
  mini-GOP target distortion.
- arXiv 2409.01009. "Accelerating block-level rate control for learned image compression". Muchen
  Dong. 2024. Mechanism: a D-λ model per block plus block-wise R-D prediction, "more than 98%
  accuracy" of the target. Mapping: block-level allocation is the analogue of distributing a scene
  budget over spatial regions of the point set.
- arXiv 2508.20709. "Learned Rate Control for Frame-Level Adaptive Neural Video Compression via
  Dynamic Neural Network". Chenhao Zhang. 2025. Mechanism: variable coding routes, a Rate Control
  Agent that estimates each route's bitrate and switches route at run time, average bitrate error
  1.66 %.
- arXiv 2604.20104. "Feedback-Driven Rate Control for Learned Video Compression". Zhiheng Xu. 2026.
  The closest published mechanism to our idea. From the arXiv HTML: error e_t = log(r̂_t / r_t)
  between estimated and target rate, integral state I_t = clip(I_{t−1} + e_t, −I_max, I_max),
  increment Δ log λ_t = −(k_p·e_t + k_i·I_t + k_d·d_t), clipped to ±Δ_max, and
  λ_{t+1} = clip(λ_t · exp(Δ log λ_t), λ_min, λ_max). They use k_p = 0.9, k_i = 0.05, k_d = 0,
  report that the derivative term amplifies instability, and feed back "the entropy-model-estimated
  bitrate". Average bitrate errors 2.88 % and 2.95 % on DCVC and DCVC-TCM, 2.13 % and 2.24 % with an
  extra GRU adjustment controller. Mapping: this is λ as a controller state driven by a rate error,
  in the log domain, exactly what we want, and our advantage is that the feedback can be the exact
  encoder byte count instead of an entropy estimate. Their integral clipping is the anti-windup we
  will need while densification makes the violation transient.
- arXiv 2601.19293. "Reinforced Rate Control for Neural Video Compression via Inter-Frame
  Rate-Distortion Awareness". Wuyang Cong. AAAI 2026. Mechanism: an RL agent picks per-frame coding
  parameters with a reward on R-D and bitrate adherence, relative bitrate error 1.20 %.
- arXiv 2601.14240. "LRC-DHVC: Towards Local Rate Control in Neural Video Compression". Marc
  Windsheimer. 2026. Title and authors verified, abstract not studied.
- arXiv 2012.05339. "Neural Rate Control for Video Encoding using Imitation Learning". Hongzi Mao.
  2020. Mechanism: rate control of libvpx two-pass VBR as a POMDP, policy learned by imitation of
  evolution-strategy trajectories, 8.5 % median bitrate reduction. Mapping: learned controllers exist
  for conventional codecs too, but they need a corpus of episodes, which per-scene 3DGS lacks.

## 3. Family 2: variable-rate and target-rate neural fields, NeRF compression

None of these takes a byte target as an input to training. The size is set either by an
architectural choice made before training (hash table size, grid resolution, binary features) or
by a λ knob on an entropy term, and the "budget" figures in abstracts are reporting brackets. The
one true variable-rate representation is VBNF, which gives a ladder of sizes after training by
prefix truncation.

- arXiv 2206.07707. "Variable Bitrate Neural Fields". Towaki Takikawa. SIGGRAPH 2022. Citation (b),
  field-side. Abstract: "We present a dictionary method for compressing such feature grids, reducing
  their memory consumption by up to 100x and permitting a multiresolution representation which can
  be useful for out-of-core streaming. We formulate the dictionary optimization as a
  vector-quantized auto-decoder problem". From the PDF: "Our representation enables progressive,
  variable bitrate streaming of data by being able to scale the quality according to the available
  bandwidth or desired level of detail", "we aim to achieve variable bitrate via streaming level of
  detail", and the codebook indices "may be as small as 4 bits" where the replaced feature vectors
  took 512 bits. Mechanism: a multiresolution octree of learned codebook indices, trained with
  index learning (a VQ auto-decoder), and bitrate varies by the number of tree levels streamed. The
  index bit-width is chosen before training. Mapping: importance-ordered primitives form the 3DGS
  analogue of a prefix-truncatable representation, which is how "one model, many rates" would look
  for a point set. It does not give a target, it gives a ladder.
- DOI 10.1145/3592426, arXiv 2302.12249. "MERF: Memory-Efficient Radiance Fields for Real-time View
  Synthesis in Unbounded Scenes". Christian Reiser. ACM TOG / SIGGRAPH 2023. Mechanism: a sparse
  feature grid plus high-resolution 2D feature planes, baked losslessly for browser rendering.
  Memory is fixed by the grid and plane resolutions chosen before training, no target and no
  penalty.
- DOI 10.1145/3658193, arXiv 2312.07541. "SMERF: Streamable Memory Efficient Radiance Fields for
  Real-Time Large-Scene Exploration". Daniel Duckworth. ACM TOG / SIGGRAPH 2024. Abstract: "a
  hierarchical model partitioning scheme, which increases model capacity while constraining compute
  and memory consumption". Mechanism: capacity is scaled by partitioning into submodels of fixed
  per-submodel memory, so the device budget is respected by construction, not by optimisation.
- arXiv 2212.09069. "Masked Wavelet Representation for Compact Neural Radiance Fields". Daniel Rho.
  CVPR 2023. Abstract: "a novel trainable masking approach ... we achieved state-of-the-art
  performance within a memory budget of 2 MB". Mechanism: wavelet-domain grid coefficients with a
  learned binary mask driven by a sparsity penalty (a λ knob). The 2 MB is the bracket the results
  are reported in, not an input.
- arXiv 2306.07581. "Binary Radiance Fields". Seungjoo Shin. NeurIPS 2023. Mechanism: features
  encoded as ±1, storage fixed by the grid layout, 0.5 MB reported.
- arXiv 2406.04101. "How Far Can We Compress Instant-NGP-Based NeRF?". Yihang Chen. CVPR 2024 (CNC).
  Mechanism: level-wise and dimension-wise context models for entropy coding of the hash grid,
  trained with rendering loss plus λ times estimated bits. The direct ancestor of HAC for 3DGS. Knob
  only.
- arXiv 2404.02185. "NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for
  Memory-Efficient Scene Representation". Sicheng Li. CVPR 2024. Abstract: "train the full pipeline
  via supervision of rendering loss and entropy loss, yielding the rate-distortion balance by
  updating the content-specific parameters ... enabling high-quality novel view synthesis with a
  memory budget of 0.5 MB". Knob only, budget is a reporting bracket.
- DOI 10.1109/cvpr52729.2023.00411, arXiv 2211.16386. "Compressing Volumetric Radiance Fields to 1
  MB". Lingzhi Li. CVPR 2023 (VQRF). Mechanism: importance-based voxel pruning, trainable vector
  quantisation, joint tuning and post-processing, 100× to about 1 MB. No target.
- DOI 10.1145/3610548.3618167, arXiv 2312.17241. "Compact Neural Graphics Primitives with Learned
  Hash Probing". Towaki Takikawa. SIGGRAPH Asia 2023. Mechanism: hash table with learned probes,
  size set by table sizes chosen before training, "Pareto optimal compression and speed".
- arXiv 2201.05989. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding".
  Thomas Müller. ACM TOG 41(4), SIGGRAPH 2022. From the PDF, Table 1 and Section 4: "Only the hash
  table size T and max. resolution N_max need to be tuned to the task", T ranges from 2^14 to 2^24,
  and "Choosing the hash table size T provides a trade-off between performance, memory and quality.
  Higher values of T result in higher quality and lower performance." Mechanism: the memory knob is
  a discrete architectural parameter fixed before training, memory = L·T·F entries. Mapping: the
  3DGS analogue is a hard cap on N, which is a projection, not a loss term.
- arXiv 2406.08943. "Neural NeRF Compression". Tuan Pham. ICML 2024. Mechanism: non-linear transform
  coding of feature grids, encoder-free per-scene optimisation, importance-weighted R-D objective
  with a sparse entropy model. Knob only.
- arXiv 2309.15848. "SHACIRA: Scalable HAsh-grid Compression for Implicit Neural Representations".
  Sharath Girish. ICCV 2023 (per Semantic Scholar). Mechanism: quantised latent reparameterisation
  of feature grids plus entropy regularisation. Knob only.
- arXiv 2311.14208. "ECRF: Entropy-Constrained Neural Radiance Fields Compression with Frequency
  Domain Optimization". Soonbin Lee. IEEE MMSP 2023 (per Semantic Scholar). Mechanism: DCT of
  tensorial grids, entropy parameterisation in the frequency domain, trained end to end. Despite
  the name, "entropy-constrained" means an entropy penalty, not a target.
- arXiv 2310.14695. "CAwa-NeRF: Instant Learning of Compression-Aware NeRF Features". Omnia Mahmoud.
  2023. Mechanism: compression-aware training so that zip-compressed grids export without post
  processing. Sizes reported (1.2 MB, 0.53 MB), no target.
- arXiv 2409.15689. "Plenoptic PNG: Real-Time Neural Radiance Fields in 150 KB". Jae Yong Lee. 3DV
  2024 (per Semantic Scholar). Mechanism: sinusoid-indexed dense volumes plus spatial decomposition,
  size fixed by design.
- arXiv 2210.12782. "Compressing Explicit Voxel Grid Representations: fast NeRFs become also small".
  Chenxi Lola Deng. WACV 2023 (Re:NeRF). Mechanism: pruning of explicit voxel grids for storage.
- arXiv 2412.11362. "VRVVC: Variable-Rate NeRF-Based Volumetric Video Compression". Qiang Hu. AAAI
  (per Semantic Scholar). Abstract: "enables variable bitrates through the utilization of predefined
  Lagrange multipliers to manage the quantization error of all latent representations ... a
  multi-rate-distortion loss function". One model, a discrete set of λ, again a ladder.
- arXiv 2411.05322. "Rate-aware Compression for NeRF-based Volumetric Video". Zhiyu Zhang. ACM MM
  2024. Mechanism: an implicit entropy model estimates the bitrate during training and a rate loss
  is added, with a learned quantisation step. Knob only.
- arXiv 2212.05589. "Learning Neural Volumetric Field for Point Cloud Geometry Compression". Yueyu
  Hu. PCS 2022. Mechanism: entropy of network parameters and latent codes in the loss with
  distortion, "an R-D optimal representation". Knob only.
- arXiv 2103.03123. "COIN: COmpression with Implicit Neural representations". Emilien Dupont. 2021.
  Rate set by MLP architecture and weight quantisation, no target.
- arXiv 2112.04267. "Implicit Neural Representations for Image Compression". Yannick Strümpler.
  2021. Rate operating points set by MLP width.
- arXiv 2502.11729. "On Quantizing Neural Representation for Variable-Rate Video Coding". Junqi
  Shi. ICLR 2025 (per the comment). Mechanism: post-training mixed-precision quantisation of a
  neural video representation, framed as fine-grained rate control after training.
- "Compressing neural fields with a size budget": no paper with this title, or with a size budget
  as a training input, was found (Section 8).

## 4. Family 3: hardware-aware NAS with latency or memory predictors

Resource model and constraint entry point for each. Verified through the arXiv API, with venue
from the comment field or OpenAlex, and mechanism details from the PDF text where cited.

- arXiv 1807.11626. "MnasNet: Platform-Aware Neural Architecture Search for Mobile". Mingxing Tan.
  CVPR 2019. Resource model: direct measurement, "our approach directly measures real-world inference
  latency by executing the model on mobile phones". Constraint entry: a multi-objective reward for an
  RL controller (PPO), of the form Reward(M) = ACC(M)·(LAT(M)/T)^w with an exponent that switches at
  the target T, and the paper uses α = β = −0.07 for a soft constraint (from the PDF). No gradient
  flows through the resource.
- arXiv 1812.00332. "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware".
  Han Cai. ICLR 2019. Citation (c). From the PDF, Section 3.3.1 "Making latency differentiable":
  each candidate op o_j^i of block i has a path probability p_j^i, "E[latency_i] = Σ_j p_j^i · F(o_j^i)"
  where "F(·) denotes the latency prediction model", the network latency is the sum over blocks
  "E[latency] = Σ_i E[latency_i]" because blocks run sequentially, and the final loss is
  "Loss = Loss_CE + λ₁‖w‖²₂ + λ₂·E[latency]" with λ₂ > 0 controlling the accuracy-latency trade-off.
  The gradient ∂E[latency_i]/∂p_j^i = F(o_j^i). A REINFORCE alternative is given for the
  non-differentiable case. Resource model: per-op lookup (Appendix B). Constraint entry: fixed-λ
  penalty on the expected resource.
- arXiv 1812.03443, DOI 10.1109/cvpr.2019.01099. "FBNet: Hardware-Aware Efficient ConvNet Design via
  Differentiable Neural Architecture Search". Bichen Wu. CVPR 2019. From the PDF: "we measure the
  latency of each operator in the search space and use a lookup table model to compute the overall
  latency by adding up the latency of each operator ... it makes the latency differentiable with
  respect to layer-wise block choices", and the loss "L(a, w_a) = CE(a, w_a) · α·log(LAT(a))^β"
  (Eq. 2) with LAT(a) = Σ_l lookup(block choice of layer l), latency in microseconds. Resource
  model: additive lookup table. Constraint entry: multiplicative penalty, fixed α, β.
- arXiv 1908.09791. "Once-for-All: Train One Network and Specialize it for Efficient Deployment".
  Han Cai. ICLR 2020. Citation (b), architecture side. Mechanism: progressive shrinking trains a
  supernet whose sub-networks (depth, width, kernel, resolution) all work without retraining, "more
  than 10^19" of them, and deployment selects a sub-network under a latency constraint by search.
  Resource model: measured or predicted per device at search time. Constraint entry: at selection,
  not in training.
- arXiv 2005.14187. "HAT: Hardware-Aware Transformers for Efficient Natural Language Processing".
  Hanrui Wang. ACL 2020. From the PDF: "Collect (SubTransformer architecture, latency) data pairs on
  the target hardware. Train a latency predictor for each hardware", "The latency predictor is very
  accurate, with an average prediction error (RMSE) of 0.1s", and an evolutionary search "with
  hardware latency constraint". Resource model: learned MLP predictor from architecture features.
  Constraint entry: search-time constraint.
- DOI 10.1145/3458864.3467882. "nn-Meter: towards accurate latency prediction of deep-learning model
  inference on diverse edge devices". First author Li Lyna Zhang. MobiSys 2021, pp. 81 to 93. No
  arXiv version found. Abstract: "dividing a whole model inference into kernels, i.e., the execution
  units on a device, and conducting kernel-level prediction ... (i) kernel detection ... and (ii)
  adaptive sampling ... evaluated using a large dataset of 26,000 models". Resource model: learned
  kernel-level predictors composed per model. Not tied to any training loss.
- arXiv 2007.08668. "BRP-NAS: Prediction-based NAS using GCNs". Łukasz Dudziak. NeurIPS 2020.
  Resource model: a graph convolutional network predicts end-to-end latency (and accuracy) from the
  architecture graph, plus LatBench, a latency dataset of NAS-Bench-201 models across devices.
  Constraint entry: predictor-guided search.
- arXiv 1706.00046. "Learning Time/Memory-Efficient Deep Architectures with Budgeted Super
  Networks". Tom Veniat. CVPR 2018. Secondary citation for (e). From the PDF: "learn a neural network
  able to predict well in less than 100 milliseconds or learn an efficient model that fits in a 50
  Mb memory", the "soft constrained budgeted learning problem" of Eq. (2) adds
  "λ max(0, C(H⊙E) − 𝐂)" to the expected loss, where C is the measured cost of the sampled
  architecture H⊙E and 𝐂 "the maximum cost the user would allow", "Note that the only required
  property of C(H⊙E) is that this cost can be measured during training", and the objective is
  optimised with a policy-gradient-inspired algorithm over the edge-sampling distribution Γ.
  Resource model: measured, arbitrary units (ms, Mb). Constraint entry: hinge penalty with fixed λ.
- arXiv 1711.06798, DOI 10.1109/cvpr.2018.00171. "MorphNet: Fast & Simple Resource-Constrained
  Structure Learning of Deep Networks". Ariel Gordon. CVPR 2018. Mechanism: "shrinking via a
  resource-weighted sparsifying regularizer on activations and expanding via a uniform
  multiplicative factor on all layers", iterated until the resource constraint (FLOPs or model size)
  holds. Resource model: analytic per-layer cost weights. Constraint entry: penalty plus an outer
  rescaling loop that lands on the constraint.
- arXiv 1804.03230. "NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications".
  Tien-Ju Yang. ECCV 2018. Mechanism: "automatically and progressively simplifies a pre-trained
  network until the resource budget is met", with direct metrics "evaluated using empirical
  measurements". Resource model: measured lookup. Constraint entry: greedy outer loop to the budget.

Mapping of the family to 3DGS. The per-op lookup tables of ProxylessNAS and FBNet work because
latency is additive over sequential blocks. Render time of a 3DGS scene is not additive in N: it
depends on screen coverage, tile overlap, sorting and the depth distribution, so a render-time
budget needs a learned predictor in the nn-Meter or HAT style, a regression from (N, coverage
statistics, resolution) to milliseconds on the target device, fit from measurements. Once such a
predictor exists it can enter training exactly as in ProxylessNAS, as a differentiable penalty on
the expected cost, or as a constraint under dual ascent (Section 6). The byte budget needs no
predictor at all because the encoder is exact, which is a simplification these papers never had.

## 5. Family 4: budget-constrained pruning and quantisation

- arXiv 1811.09332, DOI 10.1109/cvpr.2019.00932. "Structured Pruning of Neural Networks with
  Budget-Aware Regularization". Carl Lemaire. CVPR 2019. Citation (d). From the PDF, Section 3.3:
  "a budget is the maximum number of neurons a 'hard-pruned' network is allowed to have", the
  network size is "the total activation volume of the structurally 'hard-pruned' network" V,
  computed from the expected values of the learnable dropout masks, "A budget constraint imposes on V
  to be smaller than the allowed budget b", and instead of a log barrier (which needs a feasible
  start) they use a barrier function f(V, a, b) that "has an infinite value when the volume used by
  a network exceeds the budget, i.e. V > b" and "a value of zero when the budget is comfortably
  respected, i.e. V < a", and "instead of having a fixed budget b and a parameter t that grows at
  each iteration as required by the barrier method, we eliminate the hardness parameter t and
  instead decrease the budget constraint at each iteration". Resource model: analytic count of
  surviving feature-map volume. Constraint entry: barrier term with a scheduled budget, plus
  knowledge distillation. Mapping: this is the pruning-side twin of our problem. Their V is our N
  (or our byte count), their scheduled budget is a byte schedule over training, and their barrier
  is one candidate for the loss term. The difference is that V in their setting only falls, while
  N in ours also rises through densification, so the schedule must allow growth phases (Section 7).
- arXiv 1905.11452. "Mixed Precision DNNs: All you need is a good parametrization". Stefan Uhlich.
  ICLR 2020. Citation (e). From the PDF, Section 3 "Training quantized DNNs with memory
  constraints": total weight memory S^w(θ^w) = Σ_l S^w_l(θ^w_l), with, from Table 2,
  S^w_l = M_l·(M_{l−1} + 1)·b^w_l bits for a fully connected layer and
  S^w_l = M_l·(M_{l−1}·K_l² + 1)·b^w_l for a convolution, where b^w_l is the learned bit-width.
  Constraint (8a): g₁ = S^w − S^w₀ ≤ 0 with S^w₀ "a certain maximum weight memory size". (8b)
  and (8c) do the same for total and maximum activation memory. Problem (9) is min E[J] s.t.
  g_j ≤ 0, solved by "the penalty method (Bertsekas, 2014)" as (10):
  min E[J] + Σ_j λ_j·max(0, g_j)², "where λ_j ∈ ℝ⁺ are individual weightings for the penalty
  terms". They state the limit: "the optimization problem (10) does not necessarily give a quantized
  DNN which fulfills the memory constraints. The probability to fulfill the constraint g_j depends
  on the choice of λ_j ... we choose λ_j such that the initial loss and the penalty term have
  approximately the same magnitude". Sizes in the experiments are in kB and MB (for example a
  MobileNetV2 at 1.55 MB of weights). Resource model: analytic bits from bit-widths and parameter
  counts. Constraint entry: squared hinge penalty with a hand-chosen fixed weight. Mapping: the
  closest thing in the literature to "bytes inside the loss". The 3DGS version replaces the
  analytic bit count by the entropy estimate as a differentiable proxy and by the exact encoder
  count for the feedback, and replaces the hand-chosen λ_j by a multiplier under dual ascent so
  that the constraint is met rather than approximated.
- arXiv 1811.08886. "HAQ: Hardware-Aware Automated Quantization with Mixed Precision". Kuan Wang.
  CVPR 2019. From the PDF: a DDPG agent outputs a bit-width per layer, "we employ a hardware
  simulator to generate direct feedback signals (latency and energy)", and "We encourage our agent to
  meet the computation budget by limiting the action space. After our RL agent gives actions {a_k}
  to all layers, we measure the amount of resources" and adjust the bit-widths until the budget
  holds. Resource model: simulator or measurement. Constraint entry: projection of the action
  sequence onto the budget, outside the gradient.
- arXiv 2005.07093. "Bayesian Bits: Unifying Quantization and Pruning". Mart van Baalen. NeurIPS 2020
  (per Semantic Scholar). Mechanism: stochastic gates on successive residual bit doublings, with a
  0-bit option unifying pruning, and "prior distributions that encourage most of them to be switched
  off". Resource model: expected bits. Constraint entry: prior strength, a knob.
- arXiv 2104.09987. "Differentiable Model Compression via Pseudo Quantization Noise". Alexandre
  Défossez. TMLR (per the comment). Mechanism: pseudo-quantisation noise makes the loss
  differentiable in the number of bits, "Given a single hyper-parameter balancing between the
  quantized model size and accuracy, DiffQ optimizes the number of bits used per individual weight
  or groups of weights". Resource model: analytic bits. Constraint entry: fixed-λ penalty on size.
- DOI 10.1109/cvpr.2018.00890. "'Learning-Compression' Algorithms for Neural Net Pruning". First
  author Miguel A. Carreira-Perpiñán. CVPR 2018, pp. 8532 to 8541. Companion reports: arXiv
  1707.01209 "Model compression as constrained optimization, with application to neural nets. Part I:
  general framework" (Carreira-Perpiñán, 2017) and arXiv 1707.04319 "Part II: quantization"
  (Carreira-Perpiñán and Idelbayev, 2017). Abstract of the CVPR paper: "We formulate pruning as an
  optimization problem of finding the weights that minimize the loss while satisfying a pruning cost
  condition. We give a generic algorithm to solve this which alternates 'learning' steps that
  optimize a regularized, data-dependent loss and 'compression' steps that mark weights for pruning
  in a data-independent way ... Using a single pruning-level user parameter". Part I: "a general
  algorithm to optimize this nonconvex problem based on the augmented Lagrangian and alternating
  optimization". Resource model: the compression step is a projection onto the set of models with
  the prescribed size (the count of surviving weights, a codebook of K entries). Constraint entry:
  augmented Lagrangian with alternation, the constraint is met exactly by the C step. Mapping: an LC
  scheme for 3DGS would alternate SGD on the primitives with a projection that prunes or quantises
  to the byte budget. The projection is available (rank primitives, keep a prefix that fits), and
  the augmented Lagrangian term pulls the training model toward the projected one. This is the
  cleanest way to get exact feasibility at the end, and it composes with dual ascent.
- arXiv 1802.03494, DOI 10.1007/978-3-030-01234-2_48. "AMC: AutoML for Model Compression and
  Acceleration on Mobile Devices". Yihui He. ECCV 2018. From the PDF: "We achieve
  resource-constrained compression by constraining the search space, in which the action space
  (pruning ratio) is constrained such that the model compressed by the agent is always below the
  resources budget", and "as the agent receives no incentive for going below the budget, it can
  precisely arrive at the target compression ratio". Resource model: analytic (model size, FLOPs)
  or measured latency. Constraint entry: projection of the per-layer actions.
- arXiv 1905.08318 (ICML 2019 workshop version) and arXiv 1907.11900 (journal version). "DeepCABAC:
  Context-adaptive binary arithmetic coding for deep neural network compression" and "DeepCABAC: A
  Universal Compression Algorithm for Deep Neural Networks". Simon Wiedemann. Mechanism: "quantizes
  each weight parameter by minimizing a weighted rate-distortion function" then CABAC. Knob only.
- arXiv 1812.07520. "Entropy-Constrained Training of Deep Neural Networks". Simon Wiedemann. 2018.
  Mechanism: "an expression for the entropy of a neural network, which measures its complexity
  explicitly in terms of its bit-size", an "entropy-constrained optimization objective" and "a
  continuous relaxation of the objective". The constraint is a penalty weight, not a target.
- arXiv 1810.06401. "Rate Distortion For Model Compression: From Theory To Practice". Weihao Gao.
  ICML 2019 (per Semantic Scholar). Mechanism: rate-distortion theory applied to weights, a new
  objective for pruning and quantisation. Knob only.
- arXiv 2409.20138. "Constraint Guided Model Quantization of Neural Networks". Quinten Van Baelen.
  2024. Abstract: "uses an upper bound on the computational resources ... CGMQ does not require the
  tuning of a hyperparameter to result in a mixed precision neural network that satisfies the
  predefined computational cost constraint". Constraint entry: a sampling rule keeps bit-widths
  within the budget during training, a projection.
- arXiv 2502.16638. "Automatic Joint Structured Pruning and Quantization for Efficient Neural
  Network Training and Compression". Xiaoyi Qu. 2025 (GETA). Abstract: "a partially projected
  stochastic gradient method that guarantees layerwise bit constraints are satisfied". Constraint
  entry: projection inside SGD.
- arXiv 2210.15623. "Neural Networks with Quantization Constraints". Ignacio Hounie. 2022.
  Mechanism: quantisation-aware training as constrained learning, "the resulting problem is strongly
  dual", and "dual variables indicate the sensitivity of the objective with respect to constraint
  perturbations", used to pick which layers to quantise. Constraint entry: dual ascent.
- arXiv 2110.02550. "CBP: Backpropagation with constraint on weight precision using a
  pseudo-Lagrange multiplier method". Guhyun Kim. NeurIPS 2021. Constraint entry: Lagrangian
  objective for precision constraints, post-training.

## 6. Family 5: constrained optimisation machinery for deep learning

- "Constrained Differential Optimization". John C. Platt, Alan H. Barr. Neural Information
  Processing Systems 0 (NIPS 1987), papers.nips.cc, no DOI. Verified through the proceedings
  abstract page and the PDF. Abstract: "we present the basic differential multiplier method (BDMM),
  which satisfies constraints exactly; we create forces which gradually apply the constraints over
  time, using 'neurons' that estimate Lagrange multipliers. The basic differential multiplier method
  is a differential version of the method of multipliers from Numerical Analysis. We prove that the
  differential equations locally converge to a constrained minimum." Mechanism: gradient descent on
  x coupled with gradient ascent on λ, ẋ = −∇f − λ·∇g and λ̇ = g(x), which is a damped oscillator
  whose damping matrix must be positive definite for convergence. Section 5 gives the modified
  differential method of multipliers (MDMM), which adds the penalty force: from the PDF,
  ẋᵢ = −∂f/∂xᵢ − λ·∂g/∂xᵢ − c·g·∂g/∂xᵢ, "which is a modification of the BDMM with more robust
  convergence properties". Mapping: the multiplier is a state updated from the measured constraint
  value, which is exactly "λ as a controller state". The ascent λ̇ = g(x) is pure integral action,
  the MDMM penalty is proportional action on the state, and 3DGS training can run this with g = R
  measured by the encoder minus R_target.
- arXiv 1804.06500. "Two-Player Games for Efficient Non-Convex Constrained Optimization". Andrew
  Cotter, Heinrich Jiang, Karthik Sridharan. Proceedings of the 30th International Conference on
  Algorithmic Learning Theory, PMLR 98, pp. 300 to 332, 2019 (verified on proceedings.mlr.press).
  Not ICML 2019 as stated in the brief. Abstract: "A natural approach to constrained optimization is
  to optimize the Lagrangian, but this is not guaranteed to work in the non-convex setting, and, if
  using a first-order method, cannot cope with non-differentiable constraints ... We propose a
  non-zero-sum variant of the Lagrangian formulation that can cope with non-differentiable, even
  discontinuous, constraints, which we call the 'proxy-Lagrangian'. The first player minimizes
  external regret in terms of easy-to-optimize 'proxy constraints', while the second player enforces
  the original constraints by minimizing swap regret ... a distribution over no more than m+1
  models". Mechanism: the model player descends on a Lagrangian built from differentiable proxy
  constraints, the multiplier player ascends on the true, non-differentiable constraint values.
  Mapping: this is our exact situation. The differentiable proxy is the entropy estimate of R, the
  true constraint is the encoder's byte count, which is exact and non-differentiable, and the
  multiplier is updated from the true count. The stochastic-classifier part (a mixture of m+1
  models) has no analogue in a per-scene point set, so only the update structure transfers.
- arXiv 1809.04198. "Optimization with Non-Differentiable Constraints with Applications to
  Fairness, Recall, Churn, and Other Goals". Andrew Cotter, Heinrich Jiang, Maya Gupta, Serena Wang,
  Taman Narayan, Seungil You, Karthik Sridharan. Journal of Machine Learning Research 20(172), pp.
  1 to 59, 2019 (verified on jmlr.org). Same proxy-Lagrangian machinery, applied to "rate
  constraints" on predictions, with the practical algorithms and experiments.
- arXiv 2208.04425. "Controlled Sparsity via Constrained Optimization or: How I Learned to Stop
  Tuning Penalties and Love Constraints". Jose Gallego-Posada, Juan Ramirez, Akram Erraqabi, Yoshua
  Bengio, Simon Lacoste-Julien. NeurIPS 2022. Citation (f), applied form. Abstract: "Existing methods
  based on sparsity-inducing penalties involve expensive trial-and-error tuning of the penalty
  factor, thus lacking direct control of the resulting model sparsity. In response, we adopt a
  constrained formulation: using the gate mechanism proposed by Louizos et al. (2018), we formulate a
  constrained optimization problem where sparsification is guided by the training objective and the
  desired sparsity target". From the PDF: constraints of the type ‖θ_g‖₀ ≤ K per parameter group,
  written as an expected L0 density ≤ ε_g, "the constraint level ϵ_g has straightforward and
  interpretable semantics ... i.e. the percentage of active parameters", solved by simultaneous
  gradient descent on the model and gradient ascent on the multipliers, "The gradient update for
  λ_co matches the value of the violation of each constraint. When a constraint is satisfied, the
  gradient for its corresponding Lagrange multiplier is non-positive, leading to a reduction in the
  value of the multiplier", with a "dual restart" scheme that resets a multiplier to zero once its
  constraint is satisfied, because "constraint violations accumulate in the value of the Lagrange
  multipliers throughout the optimization, and continue to affect the optimization dynamics, even
  after a constraint has been satisfied". Their Figure 1 shows the penalty approach jumping from
  more than 80 % density to less than 40 % across orders of magnitude of the penalty factor while the
  constrained approach lands on the target. Mapping: replace "L0 density of a layer" by "bytes of the
  scene" and this is the method. The dual restart matters for us because densification produces
  transient violations that would otherwise wind up the multiplier.
- arXiv 2504.01212. "Cooper: A Library for Constrained Optimization in Deep Learning". Jose
  Gallego-Posada, Juan Ramirez, Meraj Hashemizadeh, Simon Lacoste-Julien. arXiv 2025, listed on
  neurips.cc under the NeurIPS 2025 workshop "Constrained Optimization for Machine Learning".
  Abstract: "Cooper implements several Lagrangian-based first-order update schemes ... suitable for
  general non-convex continuous constrained optimization." The implementation home for (f).
- arXiv 2007.03964. "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods". Adam
  Stooke, Joshua Achiam, Pieter Abbeel. ICML 2020. Abstract: "the traditional Lagrange multiplier
  update behaves as integral control; our terms introduce proportional and derivative control,
  achieving favorable learning dynamics through damping and predictive measures". From the PDF:
  "λ̇ = α·g(x) + β·ġ(x)" for the proportional-integral form, and a further term in the second
  derivative of the constraint for the derivative part. Mapping: the controls reading of dual
  ascent. Our rate error is measured exactly, so a PI update of λ on the byte error is the same
  algorithm as their PI multiplier, and their scale-invariance trick (normalising by the magnitudes
  of objective and cost) removes the λ-scale problem between the photometric loss and bytes.
- arXiv 2406.04558. "On PI Controllers for Updating Lagrange Multipliers in Constrained
  Optimization". Motahareh Sohrabi, Juan Ramirez, Tianyue H. Zhang, Simon Lacoste-Julien, Jose
  Gallego-Posada. ICML 2024. Mechanism: the νPI update for multipliers, "extending the work of
  Stooke, Achiam and Abbeel (2020)", with the finding that momentum on the dual variable does not
  fix gradient descent-ascent oscillations while νPI does.
- arXiv 2509.22500. "Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in
  Disguise". Juan Ramirez, Simon Lacoste-Julien. ICLR 2026 (per the comment). Mechanism: "dual
  optimistic ascent on the Lagrangian is equivalent to gradient descent-ascent on the Augmented
  Lagrangian", which transfers ALM's linear local convergence to the PI multiplier update and gives
  a rule for the optimism (proportional gain) hyperparameter. This closes the loop between Platt and
  Barr's MDMM, Stooke's PI multiplier and the augmented Lagrangian.
- arXiv 1810.00597. "Taming VAEs". Danilo Jimenez Rezende, Fabio Viola. arXiv 2018 (GECO).
  Mechanism: minimise the KL term subject to expectation constraints on reconstruction,
  E[C(x, g(z))] ≤ 0, "by using a standard method of Lagrange multipliers ... via a min-max
  optimization scheme", and the β of a β-VAE is read as the multiplier of such a constraint. The
  point for us: a constraint stated in the data domain (a reconstruction target) replaces a
  hand-tuned trade-off weight. Our constraint is stated in the byte domain, which is the mirror
  image, with the multiplier attached to R rather than to D.
- arXiv 2006.05487. "Probably Approximately Correct Constrained Learning". Luiz F. O. Chamon,
  Alejandro Ribeiro. arXiv 2020. Mechanism: the empirical dual problem of constrained learning is
  itself a PAC constrained learner, giving a generalisation justification for dual ascent on
  mini-batch losses.
- arXiv 2001.09394, DOI 10.1007/978-3-030-67670-4_8. "Lagrangian Duality for Constrained Deep
  Learning". Ferdinando Fioretto. ECML PKDD 2020, LNCS. Dual ascent with SGD for physical and
  fairness constraints.
- arXiv 1904.04205. "Constrained Deep Networks: Lagrangian Optimization via Log-Barrier Extensions".
  Hoel Kervadec. EUSIPCO 2022. Mechanism: log-barrier extensions that approximate Lagrangian
  optimisation with implicit dual variables and need no feasible start. Relevant as the barrier-type
  alternative to Lemaire's f(V, a, b).
- "A Primal Dual Formulation For Deep Learning With Constraints". Yatin Nandwani. NeurIPS 2019 per
  OpenAlex, no arXiv id found, PDF not read (Section 8).
- arXiv 2501.15217. "Predictive Lagrangian Optimization for Constrained Reinforcement Learning".
  Tianqi Zhang. 2025. Mechanism: the multiplier is the control input of a feedback system whose
  state is the policy parameters, PID Lagrangian is the special case with a PID controller, and MPC
  is used instead.
- arXiv 2604.06511. "Feedback control of Lagrange multipliers for non-smooth constrained
  optimization". V. Cerone. 2026. Control-theoretic multiplier updates on a proximal augmented
  Lagrangian, with exponential convergence under strong convexity.
- arXiv 2412.12092. "No More Tuning: Prioritized Multi-Task Learning with Lagrangian Differential
  Multiplier Methods". Zhengxing Cheng. AAAI 2025. Platt and Barr's differential multipliers applied
  to prioritised multi-task learning.
- arXiv 2010.06313. "Controllable Pareto Multi-Task Learning". Xi Lin. arXiv 2020. Mechanism: a
  hypernetwork conditioned on a preference vector generates the weights, so one model covers the
  trade-off front. Another "one model, many trade-offs" instance, not a target mechanism.
- "Constrained optimization for deep networks with PID control": no paper with this title was
  found. The PID multiplier line is Stooke et al. 2020, Sohrabi et al. 2024 and Ramirez and
  Lacoste-Julien 2025 above.

## 7. Transfer to BudgetGS

Transfers unchanged:

- The log-domain multiplicative update of λ from a measured rate error (HM R-λ, x264 ABR, Xu et al.
  2026 PI). Bytes are measured exactly by the encoder, so the feedback signal is cleaner than in any
  of the ancestors.
- Dual ascent on a multiplier attached to a resource constraint (Platt and Barr 1987, Cotter et al.
  2019, Gallego-Posada et al. 2022), including the proxy split: the differentiable entropy estimate
  is the proxy constraint for the primal step, the encoder count is the true constraint for the dual
  step.
- Dual restarts (Gallego-Posada et al. 2022) and integral clipping (Xu et al. 2026) as anti-windup.
- PI damping and scale normalisation of the multiplier (Stooke et al. 2020, Sohrabi et al. 2024),
  with the ALM equivalence of Ramirez and Lacoste-Julien 2025 as the convergence argument.
- Hinge or squared-hinge penalties on a byte count (Uhlich et al. 2020, Veniat and Denoyer 2018) as
  the baseline that a controlled multiplier improves on.
- Two-pass (x264): a first training run at fixed λ measures the scene's R(λ), a second run applies
  the corrected λ. The simplest baseline.
- Exact-feasibility projection (Carreira-Perpiñán and Idelbayev 2018, AMC, HAQ, GETA): rank
  primitives and keep the prefix that fits the budget, as the final step or alternated with SGD.

Needs adaptation because N is not fixed:

- Every rate model above assumes a fixed unit count: pixels per picture (R-λ, ρ-domain, TM5),
  weights per layer (Uhlich, Lemaire), ops per block (ProxylessNAS, FBNet). R = N·r(λ) in 3DGS, and
  densification changes N by construction. The model parameters must be fit on r = R/N and N tracked
  separately, and re-estimated after each densification event the way HEVC re-estimates after a
  scene cut.
- Budget schedules (Lemaire's decreasing budget, TM5's per-GOP allocation) assume the resource
  only falls. A 3DGS schedule needs growth phases during densification and a path back to the target
  after it, or the constraint must be evaluated on the projected post-densification size.
- Multiplier windup: densification makes the violation transient and large, so dual restarts or
  integral clipping are required rather than optional.
- Additive latency lookup tables do not apply to rasterised splatting, whose time is not additive
  in N. A render-time budget needs a learned device predictor (nn-Meter, HAT, BRP-NAS style) fit
  from measurements of (N, coverage, resolution), before it can enter training the ProxylessNAS way.
- "One model, many rates" for a point set means an importance ordering with prefix truncation
  (VBNF's level-of-detail prefix is the template), which interacts with densification because new
  primitives must be inserted into the ordering.

## 8. Unverified or partially verified

- MPEG-2 Test Model 5 (ISO/IEC JTC1/SC29/WG11, "Test Model 5", MPEG 93/457, Document AVC-491, April
  1993). No primary copy fetched: the mpeg.org MSSG chapter returns 404 and web.archive.org is not
  reachable from this environment. Identity and mechanism taken from the InTech 2011 chapter by Wu
  et al. (fetched and read) and from search summaries. The exact TM5 formulas (Q proportional to
  virtual buffer fullness, the reaction parameter) are therefore not quoted here.
- Li et al. 2014 R-λ update law. The paper's abstract is verified, but the update equations quoted
  in 2.1 come from the HM reference software source (`TEncRateCtrl.cpp`), not from the paper text,
  which sits behind IEEE Xplore. The paper states its algorithm was integrated into HM, so the source
  is the normative implementation, but a citation of specific constants should say "HM".
- "Neural Rate Control for Learned Video Compression" (Zhang et al., ICLR 2024): verified through
  iclr.cc and OpenReview id 42lcaojZug, no arXiv id found. OpenReview itself blocked the fetch.
- "Compressing neural fields with a size budget": no paper with this title or with a byte target as
  a training input was found across arXiv title search and web search. If the phrase refers to a
  specific paper, its id is needed.
- "Constrained optimization for deep networks with PID control": no paper with this title found.
- Nandwani et al., "A Primal Dual Formulation For Deep Learning With Constraints", NeurIPS 2019:
  record found on OpenAlex only, no arXiv id, PDF not read.
- LRC-DHVC (arXiv 2601.14240): title and authors verified, mechanism not read.
- GECO's exact multiplier update rule (a moving-average form) could not be quoted: pypdf stopped
  extracting the page that carries the algorithm. The Lagrangian min-max formulation is quoted from
  the pages that were extracted.
- Venues taken from a single secondary source (Semantic Scholar or OpenAlex) rather than the
  proceedings page: SHACIRA (ICCV 2023), ECRF (MMSP 2023), Plenoptic PNG (3DV 2024), VRVVC (AAAI),
  Bayesian Bits (NeurIPS 2020), Gao et al. (ICML 2019), Nandwani et al. (NeurIPS 2019). Cite the
  arXiv id alongside.
- Instant-NGP's T range, MnasNet's reward form, HAT's predictor error, ProxylessNAS's loss and
  FBNet's loss were read from PDF text extraction, which can drop symbols. The equations were
  cross-checked against the surrounding sentences, but the extracted glyphs for exponents are not a
  substitute for reading the typeset paper.
