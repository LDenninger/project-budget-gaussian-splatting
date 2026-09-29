# F2 Rate-control theory: Ballé 2018, Choi 2019, Cui 2021

Reading notes for the three learned-image-compression papers that the 3DGS compression literature
inherits its D + λ·R objective and its "one model, many rates" mechanisms from. Every number is
tagged with its provenance:

- `(reported, p. N)`: stated in the paper text, a caption or a table on that PDF page.
- `(read off Fig. N)`: not stated as a number. Read from the vector data of the named figure by
  extracting the marker and polyline coordinates with pymupdf and calibrating the axes on the tick
  label positions. Reading uncertainty is about ±0.01 bpp and ±0.1 dB, from marker size and line
  width. These values are measurements of the plot, not statements by the authors.
- `not stated`: the paper does not say it.

Math is written in plain Unicode. Tildes mark noisy training-time quantities, hats mark quantized
test-time quantities, following the papers.

---

## 1. Ballé et al. 2018, Variational image compression with a scale hyperprior

### 1.1 Citation

- Title: Variational Image Compression with a Scale Hyperprior
- First author: Johannes Ballé (with David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston,
  Google)
- Venue and year as stated: "Published as a conference paper at ICLR 2018" (p. 1)
- arXiv id from the file name: 1802.01436 (PDF is v2, 1 May 2018)

### 1.2 Objective

The paper starts from the Shannon cross entropy between the true marginal m(ŷ) of the quantized
latent and the entropy model p_ŷ (p. 1, eq. 1):

    R = E_{ŷ∼m}[ −log₂ p_ŷ(ŷ) ]

and the operational rate of the transform code x → y = g_a(x; φ_g) → ŷ = Q(y) → x̂ = g_s(ŷ; θ_g)
(p. 3, eq. 2):

    R = E_{x∼p_x}[ −log₂ p_ŷ( Q(g_a(x; φ_g)) ) ]

"Various compression methods can be viewed as minimizing a weighted sum of these two quantities.
Formally, we can parameterize the problem by λ, a weight on the distortion term. Different
applications require different trade-offs, and hence different values of λ." (p. 3). Note the
convention: λ weights distortion in this paper, whereas Choi weights the rate.

The whole thing is then rewritten as a variational autoencoder. The loss is the expected KL
divergence between the inference density q and the true posterior (p. 4, eq. 3):

    E_{x∼p_x} D_KL[ q ‖ p_{ỹ|x} ]
      = E_{x∼p_x} E_{ỹ∼q}[ log q(ỹ | x)  −  log p_{x|ỹ}(x | ỹ)  −  log p_ỹ(ỹ) ] + const

with the three terms being (i) zero, (ii) the weighted distortion and (iii) the rate.

**How the rate is made differentiable: additive uniform noise.** "Approximations that have been
investigated include substituting the gradient of the quantizer (Theis et al., 2017), and
substituting additive uniform noise for the quantizer itself during training (Ballé et al.,
2016b). Here, we follow the latter method, which switches back to actual quantization when
applying the model as a compression method." (p. 3). The inference model is (p. 4, eq. 4):

    q(ỹ | x, φ_g) = ∏[i] U( ỹᵢ | yᵢ − ½, yᵢ + ½ ),   y = g_a(x; φ_g)

Because every factor has unit width, log q(ỹ | x) = 0 and the first KL term drops out. The
likelihood is taken Gaussian with precision 2λ (p. 4, eq. 5):

    p_{x|ỹ}(x | ỹ, θ_g) = N( x | x̃, (2λ)⁻¹ 1 ),   x̃ = g_s(ỹ; θ_g)

so −log p_{x|ỹ} = λ‖x − x̃‖² + const, i.e. squared error weighted by λ. The third term is the
differential cross entropy between m(ỹ) and the prior p_ỹ, "as opposed to a Shannon (discrete)
entropy as in eq. (2), due to the uniform noise approximation. Under the given assumptions,
however, they are close approximations of each other (for an empirical evaluation of this
approximation, see Ballé et al., 2017)." (p. 4).

With the hyperprior the loss becomes (p. 5, eq. 10):

    E_{x∼p_x} D_KL[ q ‖ p_{ỹ,z̃|x} ]
      = E_{x∼p_x} E_{ỹ,z̃∼q}[ log q(ỹ, z̃ | x) − log p_{x|ỹ}(x | ỹ)
                              − log p_{ỹ|z̃}(ỹ | z̃) − log p_z̃(z̃) ] + const

"Again, the first term is zero, since q is a product of uniform densities of unit width. The
second term (the likelihood) encapsulates the distortion, as before. The third and fourth term
represent the cross entropies encoding ỹ and z̃, respectively. In analogy to traditional transform
coding, the fourth term can be seen as representing side information." (p. 5). In plain form the
training objective is therefore

    L = λ · E‖x − x̃‖²  +  E[ −log p_{ỹ|z̃}(ỹ | z̃) ]  +  E[ −log p_z̃(z̃) ]

with ỹ = y + u, z̃ = z + u', u, u' ∼ U(−½, ½) elementwise, and no straight-through estimator.

### 1.3 Entropy model

**Factorized prior** (p. 4, eq. 6): each latent element gets its own univariate non-parametric
density convolved with a unit uniform,

    p_{ỹ|ψ}(ỹ | ψ) = ∏[i] ( p_{yᵢ|ψ⁽ⁱ⁾}(ψ⁽ⁱ⁾) ∗ U(−½, ½) )(ỹᵢ)

The univariate density is defined through its cumulative c = f_K ∘ ... ∘ f₁ with
f_k(x) = g_k(H⁽ᵏ⁾x + b⁽ᵏ⁾) for k < K, f_K(x) = sigmoid(H⁽ᴷ⁾x + b⁽ᴷ⁾), g_k(x) = x + a⁽ᵏ⁾ ⊙ tanh(x),
H⁽ᵏ⁾ = softplus(Ĥ⁽ᵏ⁾), a⁽ᵏ⁾ = tanh(â⁽ᵏ⁾), and p = c' by the chain rule (pp. 14–15, eqs. 12–23).
"For all experiments in this paper, we used K = 4, with the dimensionalities r₁ = r₂ = r₃ = 3."
(p. 15, reported). K = 1 reduces to a logistic (eq. 24–25). "To maintain translation invariance
across the model, all elements of z with the same channel index are assumed to follow the same
univariate distribution." (p. 6).

**Scale hyperprior** (p. 5, eqs. 7–9): a second latent z̃ = h_a(y) + noise is coded with the
factorized non-parametric prior above, and it predicts a per-element standard deviation

    p_{ỹ|z̃}(ỹ | z̃, θ_h) = ∏[i] ( N(0, σ̃ᵢ²) ∗ U(−½, ½) )(ỹᵢ),   σ̃ = h_s(z̃; θ_h)

    q(ỹ, z̃ | x, φ_g, φ_h) = ∏[i] U(ỹᵢ | yᵢ − ½, yᵢ + ½) · ∏[j] U(z̃ⱼ | zⱼ − ½, zⱼ + ½),
        y = g_a(x; φ_g),  z = h_a(y; φ_h)

    p_{z̃|ψ}(z̃ | ψ) = ∏[i] ( p_{zᵢ|ψ⁽ⁱ⁾}(ψ⁽ⁱ⁾) ∗ U(−½, ½) )(z̃ᵢ)

The Gaussians are zero-mean (a mean is only added later by Minnen et al. 2018).

**Likelihood of a quantized symbol** (p. 7, eq. 11): the same function that gives the noisy
density gives the probability mass at integers, which is what lets the arithmetic coder be driven
directly by the trained prior:

    p_ŷᵢ(ŷᵢ | σ̂ᵢ) = p_ỹᵢ(ŷᵢ | σ̂ᵢ) = ( N(0, σ̂ᵢ) ∗ U(−½, ½) )(ŷᵢ) = ∫[ŷᵢ−½ .. ŷᵢ+½] N(y | 0, σ̂ᵢ) dy

"which can be evaluated in closed form." For any underlying density with cumulative c the
convolution is (p. 16, eq. 29):

    p_ỹ(ỹ) = ( p ∗ U(−½, ½) )(ỹ) = c(ỹ + ½) − c(ỹ − ½)

Why the convolution with the uniform: so that a collapsed latent dimension (g_a outputs a constant
c independent of x) gets zero bits, the prior must be able to become a unit-width uniform, which
the smooth density models cannot do on their own (pp. 15–16, Fig. 9).

**Hyperprior architecture** (Fig. 4, p. 6, reported): g_a = 3 × [conv N×5×5/2↓, GDN] then
conv M×5×5/2↓. g_s mirrors it with IGDN and a final conv 3×5×5/2↑. h_a = abs, conv N×3×3/1,
ReLU, conv N×5×5/2↓, ReLU, conv N×5×5/2↓. h_s = conv N×5×5/2↑, ReLU, conv N×5×5/2↑, ReLU,
conv M×3×3/1, ReLU. "N and M were chosen dependent on λ, with N = 128 and M = 192 for the 5 lower
values, and N = 192 and M = 320 for the 3 higher values." The factorized-prior baseline uses the
identical g_a, g_s. Arithmetic coding: "a simple non-adaptive binary arithmetic coder. Each
element of ŷ and ẑ is independently converted to its representation as a binary integer and
arithmetically encoded from the most significant to the least significant bit." (p. 6).

**Gap between estimated entropy and actual arithmetic-coded bits:** not stated in this paper. It
defers to the earlier work twice: "for an empirical evaluation of this approximation, see Ballé
et al., 2017" (p. 4) and "While this method doesn't offer a bound for the approximation, it
establishes a direct relationship between the discrete and continuous prior distributions p_ŷ
and p_ỹ, which enables direct evaluation of the discrete prior as a function of the latents ẑ as
in eq. (11), and hence makes use of a hyperprior feasible in practice. The quality of the
approximation is verified empirically by Ballé et al. (2017)." (p. 11). Whether the plotted bpp
are actual coded sizes or entropy estimates is also not stated.

**Side-information cost** (reported, p. 10, Fig. 7): "The amount of side information grows with
the total bit rate, but stays far below 0.1 bpp, even for the highest total bit rates." The Fig. 7
side-information axis runs from 0 to 0.02 bpp against a total-rate axis of 0 to 1.6 bpp.

### 1.4 Variable-rate mechanism

None. One model per operating point: "we trained a total of 32 separate models: half of the
models with a hyperprior and half without; half of the models with mean squared error as the
distortion metric ..., and half on the MS-SSIM distortion index ...; finally, each of these
combinations with 8 different values of λ in order to cover a range of rate–distortion tradeoffs."
(p. 7, reported). The λ values themselves are not stated. Training: ≈ 1 million web images,
256×256 crops, minibatches of 8, Adam, learning rate 10⁻⁴, no batch norm or learning-rate decay
(p. 7, reported). Capacity is tied to λ: "for a given λ, there exist a certain number of filters
per layer at which performance saturates ... The optimal number of filters increases with λ"
(p. 17).

### 1.5 Statements about hitting a target rate

None found for a target rate, a rate control or a search procedure. The closest statements:

- "Different applications require different trade-offs, and hence different values of λ." (p. 3)
- "Note that the architecture of the models does not explicitly constrain the bit rates in any
  way. The illustrated trade-off in allocating bits for encoding ẑ vs. ŷ is simply the result of
  optimizing the loss function given in eq. (10)." (p. 10)
- "Note that the PSNR plot aggregates curves over equal values of λ, and the MS-SSIM plot
  aggregates over equal rates (with interpolation)" (Fig. 5 caption, p. 8), i.e. equal-rate points
  are only obtained by interpolating between trained models.

### 1.6 Stated limitations and future work

There is no explicit future-work section. Limitation statements, verbatim:

- "Our model, when trained on the appropriate loss, has the capacity to surpass the state of the
  art on MS-SSIM, but does not quite reach the performance of a heavily optimized traditional
  method such as BPG on PSNR (while outperforming all other methods based on ANNs). This
  discrepancy may indicate that methods based on ANNs have not yet reached the expressive power
  of traditional methods." (p. 12)
- "It is easy to see that for increasing values of λ, the rate term containing the factorized
  prior becomes less and less important. Hence, it is questionable whether rate–distortion
  optimality implies full independence of the representation, at least for arbitrary values of
  λ." (p. 12)
- "It is important to note that neither distortion metric is sophisticated enough to capture
  image semantics, which makes the choice of distortion loss a difficult one." (p. 10)
- "However, it is possible that dependencies remain simply because the analysis and synthesis
  transforms g_a and g_s do not have enough capacity to factorize the image representation, or
  because the training algorithm did not succeed in finding the global optimum." (pp. 16–17)
- "Thus, these measurements represent proof that the method is feasible, but their utility for
  meaningful comparisons with other methods is limited." (p. 17, on runtimes)

### 1.7 Reported results on Kodak

- Axis ranges of Fig. 5 (p. 8): 0 to 2.0 bpp, 25 to 40 dB PSNR.
- Hyperprior optimized for MSE, 8 λ points (read off Fig. 5): (0.12 bpp, 27.1 dB), (0.19, 28.7),
  (0.30, 30.6), (0.47, 32.6), (0.69, 34.6), (0.97, 36.7), (1.31, 38.8), (1.73, 40.8). Operating
  range ≈ 0.12–1.73 bpp, ≈ 27.1–40.8 dB.
- Factorized prior optimized for MSE, 8 points (read off Fig. 5): (0.12, 26.8), (0.19, 28.4),
  (0.32, 30.0), (0.48, 31.7), (0.72, 33.7), (1.06, 35.8), (1.46, 38.0), (1.96, 40.1).
- Single-image example (reported, Fig. 6 caption, p. 9): MS-SSIM-trained hyperprior at 0.1864 bpp
  gives PSNR 27.99 and MS-SSIM 0.9803. MSE-trained hyperprior at 0.1932 bpp gives PSNR 32.26 and
  MS-SSIM 0.9713.
- Qualitative claims (reported, p. 7): the MSE-trained hyperprior "get[s] close to BPG
  performance, with better results at higher bit rates than lower ones, but still substantially
  outperforming all published ANN-based methods." MS-SSIM-trained hyperprior "consistently
  surpassing the state of the art" (Rippel and Bourdev 2017).
- Runtime (reported, Table 1, p. 17): N = 128, CPU, Kodak: encode 331.54 ms, decode 334.21 ms.
  "The average increase in runtime for the hyperprior model compared to the factorized-prior model
  was between 20% and 50%."

---

## 2. Choi et al. 2019, Variable rate deep image compression with a conditional autoencoder

### 2.1 Citation

- Title: Variable Rate Deep Image Compression With a Conditional Autoencoder
- First author: Yoojin Choi (with Mostafa El-Khamy, Jungwon Lee, Samsung Semiconductor)
- Venue and year as stated: the PDF carries no venue line (arXiv 1909.04802v1, 11 Sep 2019). The
  file name labels it ICCV 2019, and Cui et al. cite it as "ICCV, 2019".
- arXiv id from the file name: 1909.04802

### 2.2 Objective

Notation: encoder f_φ, decoder g_θ, quantized latent z = round_Δ(f_φ(x)) with
round_Δ(x) = Δ · round(x/Δ). Deterministic form (p. 3, eq. 1–2):

    R_φ = ∑[z] −P_φ(z) log₂ P_φ(z),   D_{φ,θ} = E_{p(x)}[ ‖x − g_θ(round_Δ(f_φ(x)))‖₂² ]

    min[φ,θ] { D_{φ,θ} + λ R_φ },   λ > 0

Here λ weights the rate (the opposite convention from Ballé).

**How the rate is made differentiable: universal quantization with a straight-through gradient.**
"It was proposed in [3] to model the quantization error as additive uniform stochastic noise to
relax the optimization of (2). ... In this paper, we instead propose employing universal
quantization [29, 30] to relax the problem" (p. 3). Eq. 3:

    z = round_Δ( f_φ(x) + u ) − u,   u = [U, U, ..., U],   U ∼ Uniform[−Δ/2, Δ/2]

"the dithering vector u consists of repetitions of a single uniform random variable U ... In each
dimension, universal quantization is effectively identical in distribution to adding uniform noise
independent of the source, although the noise induced from universal quantization is dependent
across dimensions. Note that universal quantization is approximated as a linear function of the
unit slope (of gradient 1) in the backpropagation of the network training." (p. 3). So the forward
pass quantizes (dithered), the backward pass is identity. "We observed from our experiments that
our relaxation with universal quantization provides some gain over the conventional method of
adding independent uniform noise (see Figure 3)." (Remark 1, p. 3, no number given).

Relaxed rate and distortion (p. 3, eq. 4) and the tractable upper bound (eq. 5):

    R_φ = E_{p(x) p_φ(z|x)}[ −log₂ p_φ(z) ],   D_{φ,θ} = E_{p(x) p_φ(z|x)}[ ‖x − g_θ(z)‖₂² ]

    R_φ = E[ −log₂ q_θ(z) ] − KL( p_φ(z) ‖ q_θ(z) ) ≤ E_{p(x) p_φ(z|x)}[ −log₂ q_θ(z) ] ≜ R_{φ,θ}

Single-rate objective (p. 3, eq. 6):

    min[φ,θ] E_{p(x) p_φ(z|x)}[ ‖x − g_θ(z)‖₂² − λ log₂ q_θ(z) ]

Variable-rate objective over a finite set Λ (p. 4, eq. 7) and with mixed bin sizes (p. 5,
eqs. 10–11):

    min[φ,θ] ∑[λ∈Λ] ( D_{φ,θ}(λ) + λ R_{φ,θ}(λ) )

    min[φ,θ] ∑[λ∈Λ] E_{p(Δ)}[ D_{φ,θ}(λ, Δ) + λ R_{φ,θ}(λ, Δ) ]

    R_{φ,θ}(λ, Δ) = E_{p(x) p_φ(z|x,λ,Δ)}[ −log₂ q_θ(z | λ, Δ) ],
    D_{φ,θ}(λ, Δ) = E_{p(x) p_φ(z|x,λ,Δ)}[ ‖x − g_θ(z, λ)‖₂² ]

"In training, we compute neither the summation over λ ∈ Λ nor the expectation over p(Δ) in (10).
Instead, we randomly select λ uniformly from Λ and draw Δ from p(Δ) for each image to compute its
individual R-D cost, and then we use the average R-D cost per batch as the loss for gradient
descent, which makes the training scalable." (Remark 4, p. 5).

### 2.3 Entropy model

Two-level hierarchical model with a hyper-latent w and autoregressive (masked-convolution)
context, all conditioned on λ and Δ (p. 5–6, eq. 12):

    q_θ(w | λ, Δ) = ∏[i] q_θ(wᵢ | w_{<i}, λ, Δ),   q_θ(z | w, λ, Δ) = ∏[i] q_θ(zᵢ | z_{<i}, w, λ, Δ)

Likelihood of a quantized symbol, Gaussian conditional integrated over a bin of width Δ (p. 6,
eq. 13):

    q_θ(zᵢ | z_{<i}, w, λ, Δ) = ∫[zᵢ−Δ/2 .. zᵢ+Δ/2] (1/(Δ σᵢ)) f_N( (x − μᵢ)/σᵢ ) dx,
    μᵢ = μ_θ(z_{<i}, w, λ),  σᵢ² = σ_θ²(z_{<i}, w, λ)

with f_N the standard normal density. The hyper-latent uses the same form with a learned
univariate density f_ψ "as described in [4, Appendix 6.1]" (eq. 14), i.e. Ballé's non-parametric
cumulative model. Note the 1/Δ factor makes q a density. The bits actually spent per element are
"−log₂(Δ q_θ(zᵢ | z_{<i}, w)) and −log₂(Δ q_θ(wᵢ | w_{<i}))" (p. 7). Table 1 (p. 10) lists the
differences to Minnen et al. 2018: every factor gains the (λ, Δ) conditioning, and the decoder
takes both z and w. Figure 7 (p. 7) gives the full architecture. All convolutions, including the
masked ones, are conditional convolutions.

### 2.4 Variable-rate mechanism

**Knob 1, conditional convolution on λ** (p. 4, eqs. 8–9, Fig. 4). For input feature map Xᵢ,
output Yⱼ and kernel W_{i,j}:

    Yⱼ = sⱼ(λ) · ∑[i] Xᵢ ∗ W_{i,j} + bⱼ(λ)

    sⱼ(λ) = softplus( uⱼᵀ onehot_Λ(λ) ),   bⱼ(λ) = vⱼᵀ onehot_Λ(λ)

"where uⱼ and vⱼ are the fully-connected layer weight vectors of length |Λ| for output channel
j". A per-output-channel scale and bias looked up from a one-hot code of λ, applied in every
layer of encoder, decoder and entropy model. "The proposed conditional convolution is similar to
the one proposed by conditional PixelCNN [26] ... A gated-convolution structure is presented in
[26], but we develop a simpler structure so that the additional computational cost of
conditioning is marginal." (Remark 3, p. 4). Continuous λ input was tried and rejected: "The
conditioning part can be modified to take continuous λ values, which however did not produce
good results in our trials." (footnote 1, p. 4).

**Discrete λ set used in training** (p. 6, eq. 15, reported):

    Λ = { 10⁻¹·⁵, 10⁻²·⁰, 10⁻²·⁵, 10⁻³·⁰, 10⁻³·⁵ }

i.e. 5 values spaced by a factor 10⁰·⁵ ≈ 3.16.

**Knob 2, mixed quantization bin sizes** (p. 4–5): "only finite discrete points in the R-D curve
can be obtained from it, since λ is selected from a pre-determined finite set Λ. To extend the
coverage to the whole continuous range of the R-D curve, we develop another (continuous) knob to
control the rate, i.e., the quantization bin size." At inference "the larger the bin size, the
lower the rate. However, the performance naturally suffers from mismatched bin sizes in training
and inference. For a trained network to be robust and accurate for varying bin sizes, we propose
training (or fine-tuning) it with mixed bin sizes." Mixing distribution (reported, p. 5): Δ = 2ᵇ
with b uniform on [−1, 1], so Δ ∈ [0.5, 2]. Wider ranges were tested: "The larger the range of b,
we optimize a network for a broader range of the R-D curve, but the performance also degrades.
... We found that mixing bin sizes in Δ ∈ [0.5, 2] yields the best performance, although the
coverage is limited" (p. 5, Fig. 5c compares [2⁻¹, 2¹], [2⁻², 2²], [2⁻³, 2³]).

**Training schedule** (reported, p. 6): ImageNet plus the CLIC training set, 256×256 patches,
Adam, 50 epochs of 40k batches of 8, learning rate 10⁻⁴ decreased to 10⁻⁵ at epoch 20 and 10⁻⁶ at
epoch 40. First a conditional model is pre-trained on Λ with fixed Δ = 1 and MSE, then re-trained
with mixed bin sizes (with MSE, MS-SSIM or MSE+MS-SSIM). Fixed-rate baselines were trained for
fixed λ ∈ Λ and Δ = 1.

**Are rates between training points reachable?** Yes, continuously, through Δ: "Given a user's
target rate, large-scale discrete rate adaptation is achieved by changing λ, while fine continuous
rate adaptation can be performed by adjusting Δ for fixed λ. When the R-D curves overlap at the
target rate (e.g., see 0.5 BPP in Figure 5(a)), we select the combination of λ and Δ that produces
better performance." (p. 5). "our model can produce any point in the R-D curve with infinitely
fine resolution by tuning the continuous rate-adaptive parameter, the quantization bin size Δ"
(p. 7). Each λ curve spans about a factor 4.3 in rate for Δ ∈ [0.5, 2] and consecutive λ curves
overlap (read off Fig. 9, see 2.7). Both conditioning values travel with the bitstream: "we
additionally need to store the values of the conditioning variables, λ and Δ, used in encoding."
(p. 5). Compression uses "regular deterministic quantization on the encoded representation with
the selected quantization bin size Δ" (p. 5), decompression multiplies the decoded integers by Δ.

**Cost of the mechanism** (from Cui et al. 2021, Table 1, p. 6, reported there): the conditional
convolution adds 0.315 % parameters and 0.0428 % FLOPs on the Ballé 2018 backbone and 0.140 % /
0.0193 % on the Minnen 2018 backbone.

### 2.5 Statements about hitting a target rate

Verbatim, with page:

- "Coarse rate adaptation to a target is performed by changing the Lagrange multiplier, while the
  rate can be further fine-tuned by adjusting the bin size used in quantizing the encoded
  representation." (abstract, p. 1)
- "We adapt the rate by changing the Lagrange multiplier λ and the quantization bin size Δ to
  match the rate of BPG." (Fig. 2 caption, p. 2)
- "Given a user's target rate, large-scale discrete rate adaptation is achieved by changing λ,
  while fine continuous rate adaptation can be performed by adjusting Δ for fixed λ. When the R-D
  curves overlap at the target rate (e.g., see 0.5 BPP in Figure 5(a)), we select the combination
  of λ and Δ that produces better performance." (p. 5)
- "In practice, one can make a set of pre-selected combinations of λ and Δ, similar to the set of
  quality factors in JPEG or BPG." (footnote 2, p. 5)
- "We adapt and match the compression rate of our variable-rate network to the rate of BPG by
  adjusting the Lagrange multiplier λ and the quantization bin size Δ." (Appendix B, p. 10)

How the (λ, Δ) pair that meets a given target is found (a search, a table, or trial encoding) is
not stated.

### 2.6 Stated limitations and future work

- "The conditioning part can be modified to take continuous λ values, which however did not
  produce good results in our trials." (footnote 1, p. 4)
- "However, the performance naturally suffers from mismatched bin sizes in training and
  inference." (p. 4)
- "The larger the range of b, we optimize a network for a broader range of the R-D curve, but the
  performance also degrades." (p. 5)
- "We found that mixing bin sizes in Δ ∈ [0.5, 2] yields the best performance, although the
  coverage is limited, which is not a problem since we can cover large-scale rate adaptation by
  changing the input Lagrange multiplier in our conditional model" (p. 5)
- Future work, the only forward-looking sentence: "We finally note that the proposed conditional
  neural network can be adopted in deep learning not only for image compression but also in
  general to solve any optimization problem that can be formulated with the method of Lagrange
  multipliers." (p. 8)

### 2.7 Reported results on Kodak

- Axis ranges of Fig. 8 and Fig. 9 (pp. 7–8): 0 to 2.0 bpp, 24 to 42 dB PSNR.
- Variable-rate model optimized for MSE, one curve per λ, each swept over Δ ∈ [0.5, 2] (read off
  Fig. 9, p. 8):
  - λ = 10⁻³·⁵: 0.083 → 0.355 bpp, 26.8 → 31.0 dB
  - λ = 10⁻³·⁰: 0.179 → 0.639 bpp, 29.2 → 34.1 dB
  - λ = 10⁻²·⁵: 0.370 → 1.120 bpp, 31.9 → 37.5 dB
  - λ = 10⁻²·⁰: 0.698 → 1.841 bpp, 35.0 → 40.7 dB
  - λ = 10⁻¹·⁵: 1.187 bpp / 38.3 dB up to the plot edge at 2.16 bpp / 42.0 dB (clipped)
  Operating range of the single model ≈ 0.08 bpp / 26.8 dB to beyond 2.0 bpp / 42 dB.
- Fixed-rate MSE networks, 7 markers in Fig. 9 (read off): (0.165 bpp, 29.15 dB), (0.256, 30.74),
  (0.445, 33.08), (0.662, 35.08), (0.961, 37.14), (1.508, 39.85), (2.076, 41.96). The text names
  5 λ values for these baselines, the figure carries 7 markers.
- **Variable-rate versus dedicated models.** The paper's statement: "Observe that our
  variable-rate network performs very near to the ones individually optimized for fixed λ and Δ."
  (p. 7). No number is given. Read off Fig. 9, dedicated minus best variable-rate curve at the
  dedicated model's rate: +0.04, +0.06, +0.18, +0.16, +0.12, −0.03, +0.14 dB at the seven rates
  above, i.e. the single model is within about 0.2 dB PSNR of the seven separately trained models.
- Single-image operating points (reported, Fig. 10, p. 8, one Kodak image). The extracted table
  rows appear in reverse order to the header row, the pairing below follows the monotone
  rate-in-Δ relation:

  | λ | Δ | bpp | PSNR (dB) | MS-SSIM |
  |---|---|---|---|---|
  | 10⁻³·⁵ | 1.0 | 0.1326 | 29.2833 | 0.9249 |
  | 10⁻²·⁵ | 1.5 | 0.4132 | 33.1478 | 0.9737 |
  | 10⁻²·⁵ | 1.0 | 0.6006 | 34.8283 | 0.9819 |
  | 10⁻²·⁵ | 0.7 | 0.8086 | 36.2535 | 0.9863 |
  | 10⁻¹·⁵ | 1.0 | 1.8027 | 41.3656 | 0.9951 |
- Versus BPG (reported): Fig. 2, p. 2: 0.1697 bpp, PSNR 32.2332 vs BPG 31.9404 ("0.3 dB PSNR
  gain"). Fig. 11, p. 11, kodim04: 0.2078 bpp, 32.4296 dB vs BPG 32.0406. Fig. 13, p. 13,
  kodim23: 0.1289 bpp, 34.4543 dB vs BPG 33.3546.
- Claims (reported, p. 6): "Our variable-rate model outperforms BPG in both PSNR and MS-SSIM
  measures. It also performs comparable overall and better in some cases than the state-of-the-art
  learned image compression models [14, 16]".

---

## 3. Cui et al. 2021, Asymmetric gained deep image compression with continuous rate adaptation

### 3.1 Citation

- Title: Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation
- First author: Ze Cui (with Jing Wang, Shangyin Gao, Tiansheng Guo, Yihui Feng, Bo Bai, Huawei)
- Venue and year as stated: the PDF carries no venue line (arXiv 2003.02012v3, 2 Aug 2022). The
  file name labels it CVPR 2021.
- arXiv id from the file name: 2003.02012

### 3.2 Objective

Baseline VAE objective (p. 2, eq. 1), with β weighting distortion (Ballé's convention):

    min[θ,φ,ϕ] R_ϕ( Q(f_θ(x)) ) + β · D( x, g_φ(Q(f_θ(x))) )

"where R_ϕ(·) represents the expected code length (bit rate) of the quantized latent
representation and D(·) measures the distortion ... The Lagrange multiplier β is a constant in the
training process to specify the R-D tradeoff of the trained model [24]. Therefore, the VAE-based
image compression methods need to use multiple fixed-rate models trained under different β".

Discrete variable-rate (DVR) objective with gain units (p. 3, eq. 6):

    min[θ,φ,ϕ,ψ] ∑[s=0..n−1] R_ϕ( Q(G_ψ(f_θ(x), s)) )
                          + β_s · D( x, g_φ( IG_τ( Q(G(f_θ(x), s)), s ) ) )

"each pair of gain vectors {m_s, m'_s} corresponds to a specific Lagrange multiplier β_s from the
predefined finite set of the Lagrange multipliers, B ∈ ℝⁿ. The gain vector, inverse-gain vector,
and Lagrange multiplier are bound together with the subscript s." (p. 3).

**How the rate is made differentiable:** the loss is written with the hard rounding Q. The
relaxation is named but not derived: the architecture uses "UnivQuan" blocks and the text says
"we also adopt some optimization methods such as the attention module [36], Universal
quantization [43, 35], parallel context models [23]" (p. 4) and "UnivQuant represents universal
quantization [43, 35]" (Fig. 6 caption, p. 5). The gradient rule is not stated. The rate term is
"the expected bit rate of the quantized gained latent representation" (p. 3), with no −log₂ p
formula written out.

### 3.3 Entropy model

Mean-and-scale hyperprior inherited from Minnen et al. 2018 (p. 4, eq. 9), replaced by an
asymmetric Gaussian (eq. 10):

    p_{ŷ|ẑ}(ŷ | ẑ) ∼ N(μ, σ²)          (symmetric, baseline)
    p_{ŷ|ẑ}(ŷ | ẑ) ∼ N(μ, σ_l², σ_r²)   (asymmetric, proposed)

"where σ_l² and σ_r² represent the estimated left-scale and right-scale parameter of the latent
representation ... all the parameter, including μ, σ_l² and σ_r², are learnable during the
training process so that to the extreme that σ_l² and σ_r² are the same, the asymmetric Gaussian
model could degrade to the symmetric Gaussian model." (p. 4). The hyper-latent z uses "a
non-parametric, fully factorized entropy model" (p. 4). Architecture (p. 4, Fig. 6): backbone of
Minnen et al. 2018, 192 latent channels, 3×3 kernels, 192 × 3 output channels for (μ, σ_l, σ_r),
attention modules, masked convolutions of 3×3, 5×5 and 7×7 for a parallel context model. How the
bin probability is computed from the asymmetric Gaussian is not written out. Ablation (Fig. 12,
p. 7, reported qualitatively): AG-VAE beats the symmetric SG-VAE on both metrics in the 0.4 to 0.6
bpp window.

### 3.4 Variable-rate mechanism: gain and inverse-gain vectors

**Motivation** (p. 3, Fig. 2): zeroing single latent channels of kodim20 degrades PSNR unevenly,
and scaling one channel down degrades PSNR monotonically, "We can conclude that the channels'
importance varies and can be scaled to control the reconstruction quality."

**Gain unit** (p. 3, eqs. 2–4). Latent y ∈ ℝ^(c×h×w), gain matrix M ∈ ℝ^(c×n) with n gain
vectors m_s = {m_{s,0}, ..., m_{s,c−1}}:

    ȳ_{s,i} = yᵢ × m_{s,i}         (per-channel scaling)
    ȳ_s = G_ψ(y, s) = y ⊙ m_s
    ŷ_s = Q(ȳ_s) = round(ȳ_s)

"In this way, the quantization loss of the latent representation can be finely adjusted by the
gain vector channel-wisely. Therefore, the network is guided to allocate more bit rates for the
channels, which influence the reconstruction quality significantly." A gain of m on a channel
is a quantization step of 1/m on that channel.

**Inverse gain unit** (p. 3, eq. 5), a second trainable matrix M' rather than the reciprocal:

    y'_s = IG_τ(ŷ_s, s) = ŷ_s ⊙ m'_s

"[28] limits the scale and inverse-scale operation to be strictly reciprocal. However, they ignore
that the latent representation can not be mapped to the same numeral intervals due to
quantization operation by reciprocal inverse scale operation. Here, we adopt another trainable
gain unit before the decoder to adaptively rescale ŷ_s, named as the inverse gain unit."

**How they are trained** (p. 3 and p. 5, reported): "the gain matrix is trained jointly with the
autoencoder network to ensure compatibility between them." Sets of n = 6 multipliers,
B_msssim = {0.07, 0.03, 0.007, 0.003, 0.001, 0.0006} and
B_mse = {0.05, 0.03, 0.007, 0.003, 0.001, 0.0003}. "In the training process, we randomly select s
from 1 to 6 in each iteration to obtain the gain vector m_s, inverse-gain vector m'_s and Lagrange
multiplier β_s from gain matrix M, inverse-gain matrix M' and B_msssim/mse. The selected
gain/inverse-gain vector will be optimized jointly with the entire framework under the
corresponding Lagrange multiplier." Data: 5 000 self-collected images sampled to 2 000 × 2 000 plus
CLIC2020, 2 million 256×256 patches, Adam, 12 epochs, batch 8, learning rate 10⁻⁴ halved at epoch
6 (p. 5, reported).

**Continuous rate adaptation by exponential interpolation** (p. 4, eqs. 7–8). The pair keeps the
numerical interval of y, which is posed as a constant vector C ∈ ℝᶜ shared by all pairs:

    m_t · m'_t = m_r · m'_r = C

    (m_r · m'_r)ˡ · (m_t · m'_t)^(1−l) = C
    [ (m_r)ˡ · (m_t)^(1−l) ] · [ (m'_r)ˡ · (m'_t)^(1−l) ] = C

    m_v = (m_r)ˡ · (m_t)^(1−l),   m'_v = (m'_r)ˡ · (m'_t)^(1−l)

"where {m_v, m'_v} is the generated gain vector pair and l ∈ ℝ is an interpolation coefficient,
which controls the corresponding bit rate of the generated gain vector pair. Since l is a real
number, utilizing the exponent interpolation of the gain vector pairs could achieve an arbitrary
bit rate between t and r. And when l is equal to 0 or 1, it represents {m_t, m'_t} or {m_r, m'_r}
respectively. Without an extra training process and supplementary blocks, we apply the exponent
interpolation formula between the adjacent gain vector pairs in the inference process to obtain
the Continuously Variable Rate (CVR) method." Inference (p. 5): "large-scale discrete rate
adaptation by selecting the index s, while adjusting the interpolation coefficient l to achieve
fine continuous rate adaptation. The bit rate increases as the values of s and l increase. When l
is equal to 0 or 1, the discrete rate at s or s + 1 can be achieved. In practical use, the
parameters s and l are also arithmetically encoded and decoded along with the latent
representation." Interpolation is stated only between adjacent trained pairs, so the reachable
range is [β_min, β_max] of B.

**Hyperprior variant (HCVR)** (p. 4, Fig. 5): a second gain pair on the hyper-latent z. "methods
with the HCVR method could achieve slightly better R-D performance than the counterpart of
methods with the CVR method in the whole bit-rate range." (p. 7, no number).

**Cost** (Table 1, p. 6, reported, percentages of additional parameters and FLOPs, n_hp = 6):
HCVR on Ballé 2018: 0.076 % / 0.0004 %. HCVR on Minnen 2018: 0.040 % / 0.0002 %. Theis 2017
bottleneck scaling: 0.023 % / 0.0004 % and 0.010 % / 0.0002 %. Choi 2019 conditional convolution:
0.315 % / 0.0428 % and 0.140 % / 0.0193 %. "the additional parameter percentages of our HCVR
method and the bottleneck-scaling method [28] are nearly seven times smaller. The additional FLOPs
percentages of our HCVR method is nearly 100 times smaller." (p. 6).

**Reported quality loss versus separately trained models.** The paper gives no number, only:
"our variable-rate networks in a single model obtain similar R-D performance with those of the
multiple fixed-rate models individually optimized for several discrete fixed Lagrange
multipliers." (p. 6, Fig. 10) and for CVR versus DVR "the CVR method extends the coverage from
finite discrete points to the whole continuous range of the R-D curve while R-D performance not
degrades." (p. 4, Fig. 4). Read off Fig. 10 (p. 7, PSNR panels, dedicated minus variable-rate
curve at the dedicated model's rate, dedicated models are Cui's own reproductions):

| backbone | dedicated point (bpp, dB) | CVR at same rate (dB) | dedicated − CVR (dB) |
|---|---|---|---|
| Ballé 2018 | (0.134, 27.37) | 27.00 | +0.36 |
| Ballé 2018 | (0.319, 30.61) | 30.59 | +0.02 |
| Ballé 2018 | (0.517, 32.85) | 33.09 | −0.24 |
| Ballé 2018 | (0.974, 36.33) | 36.58 | −0.25 |
| Ballé 2018 | (1.102, 37.03) | 37.13 | −0.10 |
| Minnen 2018 | (0.123, 28.16) | 27.73 | +0.42 |
| Minnen 2018 | (0.300, 31.17) | 31.10 | +0.07 |
| Minnen 2018 | (0.476, 33.13) | 33.27 | −0.14 |
| Minnen 2018 | (0.896, 36.34) | 36.42 | −0.08 |

The lowest dedicated points, (0.072 bpp, 23.0 dB) for Ballé and (0.024 bpp, 24.1 dB) for Minnen,
lie below the start of the CVR curves (0.077 bpp / 25.0 dB and 0.052 bpp / 25.1 dB). So the single
gained model is within ≈ 0.4 dB at the lowest shared rate and matches or beats the dedicated
models by up to 0.25 dB above 0.3 bpp. Fig. 9 (p. 6, Minnen backbone, PSNR) adds the competing
mechanisms: HCVR gives +0.38, 0.00, −0.22, −0.14 dB (dedicated minus variable) at 0.124, 0.301,
0.477 and 0.897 bpp. Cui's re-implementation of Choi's conditional convolution scatters from
+0.24 dB to −1.14 dB relative to the dedicated Minnen models, with the largest deficits at
0.93–1.10 bpp (−0.6 to −1.1 dB) and at the overlaps of adjacent λ curves (−0.57 to −0.68 dB at
0.20, 0.32, 0.42 and 0.67 bpp). The paper's wording: "the method [10] suffers from performance
degradation in high-rate segmentations of the R-D curve and intersection of different bit rate
areas of the R-D curve." (p. 6). Theis-style bottleneck scaling loses 2.4 dB at 0.44 bpp and
0.65 dB at 0.53 bpp, and is within 0.2 dB above 0.65 bpp (read off Fig. 9).

### 3.5 Statements about hitting a target rate

Verbatim, with page:

- "The bit rate could be adjusted continuously with the change of the gain vector index s and the
  interpolation coefficient l." (Fig. 1 caption, p. 1)
- "We propose the exponent interpolation, which can generate gain vectors at the arbitrary bit
  rate." (p. 2)
- "Given the target image and the target rate, we can obtain large-scale discrete rate adaptation
  by selecting the index s, while adjusting the interpolation coefficient l to achieve fine
  continuous rate adaptation. The bit rate increases as the values of s and l increase." (p. 5)

A procedure that solves for (s, l) given a target rate is not stated. The monotonicity claim
above is the only handle offered.

### 3.6 Stated limitations and future work

- "By this means, we can obtain the desired compression performance limited to several discrete
  points of the R-D curve. The the R-D curve range depends on the number and value of Lagrange
  multiplier β_s ∈ B." (pp. 3–4, on DVR)
- "methods with the HCVR method could achieve slightly better R-D performance than the
  counterpart of methods with the CVR method in the whole bit-rate range." (p. 7)
- Future work: "We also want to utilize the AG-VAE framework on MindSpore [39], which is a new
  deep learning computing framework. These works will be finished in the future." (p. 7)

No other limitation of the gain mechanism is stated.

### 3.7 Reported results on Kodak

- Axis ranges of Fig. 7 (p. 5): 0 to 1.4 bpp, 20 to 40 dB PSNR.
- AG-VAE optimized for MSE, single model, continuous curve (read off Fig. 7): from (0.03 bpp,
  24.1 dB) to (1.01 bpp, 38.0 dB), passing 27.7 dB at 0.10 bpp, 31.1 dB at 0.25 bpp, 34.4 dB at
  0.50 bpp, 36.4 dB at 0.75 bpp and 37.9 dB at 1.00 bpp.
- DVR with n = 5 (Fig. 3, p. 3): 5 discrete points, axis 0.2 to 1.1 bpp and 28 to 38 dB. Values
  not stated.
- Single image kodim04 at ≈ 0.1 bpp (reported, Fig. 8, p. 6): AG-VAE [MSE] 0.107 bpp, 30.68 dB,
  MS-SSIM 0.93. AG-VAE [MS-SSIM] 0.109 bpp, 27.76 dB, 0.940. VTM 0.111 bpp, 31.11 dB, 0.931. BPG
  0.105 bpp, 30.00 dB, 0.914.
- Claims (reported, p. 6): "With a single model, AG-VAE achieves better R-D performance than those
  of multiple-networks methods [16, 20, 9] in PSNR" and "obtains better results than the widely
  used classical image codec BPG [8] and yields competitive results with VTM [38] in PSNR".

---

## 4. What transfers to 3DGS

Setting assumed below: a scene is a set of N primitives (Gaussians, or anchors that decode into
Gaussians) with attribute vectors yᵢ (position, scale, rotation, opacity, colour or feature
channels). The deliverable is one file whose total size in bytes must meet a budget B*. Every
3DGS compression method in the survey trains D + λ·R with a fixed λ and reaches a size only by
sweeping λ across runs. The three papers supply four mechanisms.

**Noise relaxation (Ballé eqs. 4, 6, 11, 29 and Choi eq. 3).** The device transfers unchanged:
during training replace round(yᵢ/Δ) by yᵢ/Δ + u, u ∼ U(−½, ½), score it under a density that is
an underlying model convolved with a unit uniform, p̃(ỹ) = c(ỹ + ½) − c(ỹ − ½), and at test time
evaluate the identical function at the integers to get the probability mass the arithmetic coder
needs. Choi's variant replaces the noise by dithered quantization with an identity gradient, and
reports a PSNR gain over plain noise (no number). What differs in 3DGS: the rate term is a plain
sum ∑[i=1..N] ∑[k] −log₂ p(ỹ_{i,k}) over the primitives of one scene, not an expectation over a
data distribution, so it is already the code length of the object being shipped. The surrogate
gap that Ballé leaves to an earlier paper can be measured per scene at any iteration by running
the coder on the current hard-quantized attributes and comparing to the noisy estimate, and the
measured ratio can be folded into the budget. A second difference: the image codec's rate is only
the latent, whereas the scene file also carries the primitive count, the decoder MLPs, the
hyperprior parameters and headers, which must be added to R as constant or piecewise-constant
terms if the byte budget is to be met rather than approached.

**Hyperprior (Ballé eqs. 7 to 11).** The structure transfers: model each attribute channel of
primitive i as N(μ_{i,k}, σ_{i,k}²) ∗ U(−½, ½) with (μ, σ) predicted from side information, code
the side information with a factorized prior, and count its bits. The convolutional h_a / h_s
that exploit 2D neighbourhoods must be replaced by a function of 3D position: a hash grid queried
at the primitive position, a per-anchor latent, or a context of already-decoded neighbours, which
is what the anchor-based 3DGS codecs already do. Two things change. First, the side-information
cost is no longer "far below 0.1 bpp" of a per-pixel rate but a fixed per-scene block whose size
does not shrink with the budget, so at small budgets it can dominate and must be part of the
budget accounting from the first iteration. Second, because the scene is one sample, the
hyperprior is allowed to overfit it. There is no generalization gap to protect against, only the
transmitted parameter count, so the hyperprior capacity is a rate knob in its own right.

**Conditional-on-λ layers (Choi eqs. 7 to 9).** The one-hot per-channel affine modulation
Yⱼ = sⱼ(λ)·(∑ Xᵢ ∗ W_{ij}) + bⱼ(λ) can be dropped into the attribute-decoding MLPs and the
(μ, σ) predictor of an anchor-based scene, giving one trained scene that decodes at |Λ| budgets,
which is a level-of-detail feature rather than rate control. The stronger lesson is the two-knob
structure: λ moves the operating point in coarse steps of about 3× in rate, and the bin size Δ
moves it continuously by a factor of about 4 per λ without retraining, at a quality cost that
Choi keeps below ≈ 0.2 dB only by training with Δ drawn from [0.5, 2]. In a per-scene setting the
motivation for a conditional network disappears, since the scene is trained once for one
deployment and the rate is measurable exactly, so the knob that remains useful is Δ: quantization
step sizes per attribute channel are instant, exact rate controls, and a scene can be fine-tuned
at the Δ that meets the budget rather than made robust to all Δ. Choi's negative results set the
guard rails: a continuous λ input did not train well, and widening the Δ mixing range degraded
quality, so a controller that changes λ or Δ during training should move them slowly.

**Gain vectors with interpolation (Cui eqs. 2 to 8).** A gain m_{s,k} on attribute channel k is a
per-channel quantization step 1/m_{s,k} that the R-D loss learns jointly with the representation,
and the trainable inverse gain m'_{s,k} absorbs the quantization bias that a reciprocal would not.
This is the mechanism with the least friction for 3DGS: attribute channels have very uneven
importance (position, opacity, low-order colour, high-order colour, scale, rotation), so a
learned per-channel step vector is the natural first rate knob, and it costs c × n scalars. The
exponential interpolation m_v = m_rˡ · m_t^(1−l) gives a continuous rate control between two
trained operating points with no retraining, and Cui reads its rate as monotone in l. For a
per-scene method this suggests a post-training exact fit: train gain pairs for two λ values that
bracket the budget, then bisect on l against the measured byte count until the file meets B*.
The cost Cui measures for this knob is 0.04 % to 0.08 % parameters. The difference from the
paper's use: Cui trains six (β_s, m_s, m'_s) triples in one network by sampling s per iteration,
whereas a per-scene method needs only the bracketing pair, and can even let the gain vector itself
be driven by the measured rate during training rather than by a fixed β.

**What is different across all four.** A scene is one sample, so E_{x∼p_x} collapses and the
objective is the exact code length plus λ times the reconstruction error of that one scene, which
turns D + λR from a training-time surrogate for a dataset into a per-object constrained problem,
min D subject to R ≤ B*, in which λ is a dual variable that can be updated during training from
the measured gap R − B* instead of a hyperparameter fixed before training. The rate can be
measured exactly at any time, by summing −log₂ P over the hard-quantized attributes under the
current entropy model or by running the coder, so closed-loop control replaces the λ sweep. The
budget is a total byte count, so the primitive count N is a rate knob without image-codec
analogue (pruning and densification change R in steps that quantization steps cannot), and the
fixed costs (MLPs, hash grid, headers, the count itself) belong inside R. The instant knobs are
Δ, gain vectors and their interpolation coefficient. The slow knobs are λ, N and entropy-model
capacity, and Choi's two negative results say the slow knobs must move gradually.
