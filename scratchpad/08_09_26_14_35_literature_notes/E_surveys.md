# E. Surveys: 3DGS.zip (Bagdasarian et al.) and Ali et al. (2025)

Sources, both read in full with pymupdf (page numbers below are PDF page numbers, which match the printed page numbers in both files):

- `references/05_surveys/3dgs_zip_bagdasarian_cgf2025_arxiv2407.09510.pdf`, 28 pages, arXiv v5 dated 5 Mar 2025. "3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods", M. T. Bagdasarian, P. Knoll, Y. Li, F. Barthel, A. Hilsmann, P. Eisert, W. Morgenstern (Fraunhofer HHI, HU Berlin, TU Berlin).
- `references/05_surveys/compression_3dgs_survey_ali_2025_arxiv2502.19457.pdf`, 17 pages, arXiv v1 dated 26 Feb 2025. "Compression in 3D Gaussian Splatting: A Survey of Methods, Trends, and Future Directions", M. S. Ali, C. Zhang, M. Cagnazzo, G. Valenzise, E. Tartaglione, S.-H. Bae.

Every number in this file is "as reported" by the survey that is named next to it. No number was re-measured. Quotes are verbatim from the extracted text, with PDF line-break hyphenation removed.

---

## 1. 3DGS.zip: results tables

### 1.1 Table 1 (p13): compression methods, four datasets

Caption as printed (p13): "Performance comparison of 3DGS compression methods across four datasets: Tanks and Temples, Mip-NeRF 360, Deep Blendingand, Synthetic NeRF. The included metrics are PSNR, SSIM, LPIPS and, model size in MB. The best methods in each category are highlighted ( first , second , third ). The rank represents the average rankings of the methods across all available datasets. The method highlighted in bold: 3DGS-30K is the original 3DGS method."

Column alignment was verified with `page.find_tables()` and word x-coordinates. An empty cell means the survey printed nothing for that method on that dataset. Size is in MB with 1 MB = 1000² bytes (survey protocol, see 2.2). No Gaussian counts are given in Table 1.

Rank formula as printed (p12): ranks = rank(PSNR)/6 + rank(SSIM)/6 + rank(LPIPS)/6 + rank(Size[MB])/2, averaged over the datasets a method reports.

**Tanks and Temples (as reported, 3DGS.zip Table 1)**

| Method | Rank | PSNR | SSIM | LPIPS | Size MB |
|---|---|---|---|---|---|
| ContextGS_lowrate | 4.3 | 24.12 | .849 | .186 | 9.9 |
| HAC-highrate | 4.4 | 24.40 | .853 | .177 | 11.8 |
| CodecGS | 4.8 | 23.63 | .841 | .192 | 7.8 |
| HAC-lowrate | 5.0 | 24.04 | .846 | .187 | 8.5 |
| gsplat-1.00M | 5.3 | 24.03 | .857 | .163 | 16.1 |
| ContextGS_highrate | 5.8 | 24.29 | .855 | .176 | 12.4 |
| Compact3D 32K | 9.3 | 23.44 | .838 | .198 | 13.0 |
| Compact3D 16K | 9.7 | 23.39 | .836 | .200 | 12.0 |
| RDO-Gaussian | 9.8 | 23.34 | .835 | .195 | 12.0 |
| CompGS | 9.8 | 23.70 | .837 | .208 | 10.1 |
| Reduced3DGS | 10.4 | 23.57 | .840 | .188 | 14.0 |
| SOG w/o SH | 10.5 | 23.15 | .828 | .198 | 9.3 |
| MesonGS c3 | 12.0 | 23.29 | .835 | .197 | 17.4 |
| Compressed3D | 12.4 | 23.32 | .832 | .194 | 17.3 |
| MesonGS c1 | 12.6 | 23.31 | .835 | .196 | 18.5 |
| SOG | 12.7 | 23.56 | .837 | .186 | 22.8 |
| Compact3DGS+PP | 13.3 | 23.32 | .831 | .202 | 20.9 |
| EAGLES | 14.4 | 23.37 | .84 | .20 | 29.0 |
| Scaffold-GS | 14.6 | 23.96 | .853 | .177 | 87.0 |
| Compact3DGS | 14.9 | 23.32 | .831 | .201 | 39.4 |
| LightGaussian | 15.3 | 23.11 | .817 | .231 | 22.0 |
| 3DGS-30K | 15.3 | 23.14 | .841 | .183 | 411.0 |
| EAGLES-Small | 17.2 | 23.10 | .82 | .22 | 19.0 |

**Mip-NeRF 360 (as reported, 3DGS.zip Table 1)**

| Method | Rank | PSNR | SSIM | LPIPS | Size MB |
|---|---|---|---|---|---|
| ContextGS_lowrate | 4.3 | 27.62 | .808 | .237 | 13.3 |
| HAC-highrate | 4.4 | 27.77 | .811 | .230 | 22.9 |
| CodecGS | 4.8 | 27.30 | .810 | .236 | 10.3 |
| HAC-lowrate | 5.0 | 27.53 | .807 | .238 | 16.0 |
| gsplat-1.00M | 5.3 | 27.29 | .811 | .229 | 16.0 |
| ContextGS_highrate | 5.8 | 27.75 | .811 | .231 | 19.3 |
| Compact3D 32K | 9.3 | 27.12 | .806 | .240 | 19.0 |
| Compact3D 16K | 9.7 | 27.03 | .804 | .243 | 18.0 |
| RDO-Gaussian | 9.8 | 27.05 | .802 | .239 | 23.5 |
| CompGS | 9.8 | 27.26 | .803 | .239 | 17.3 |
| Reduced3DGS | 10.4 | 27.10 | .809 | .226 | 29.0 |
| SOG w/o SH | 10.5 | 26.56 | .791 | .241 | 16.7 |
| MesonGS c3 | 12.0 | 26.99 | .797 | .246 | 25.9 |
| Compressed3D | 12.4 | 26.98 | .801 | .238 | 28.8 |
| MesonGS c1 | 12.6 | 26.99 | .796 | .247 | 28.5 |
| SOG | 12.7 | 27.08 | .799 | .230 | 40.3 |
| Compact3DGS+PP | 13.3 | 27.03 | .797 | .247 | 29.1 |
| EAGLES | 14.4 | 27.23 | .81 | .24 | 54.0 |
| Scaffold-GS | 14.6 | 27.50 | .806 | .252 | 156.0 |
| Compact3DGS | 14.9 | 27.08 | .798 | .247 | 48.8 |
| LightGaussian | 15.3 | 27.28 | .805 | .243 | 42.0 |
| 3DGS-30K | 15.3 | 27.21 | .815 | .214 | 734.0 |
| EAGLES-Small | 17.2 | 26.94 | .80 | .25 | 47.0 |

**Deep Blending (as reported, 3DGS.zip Table 1)**

| Method | Rank | PSNR | SSIM | LPIPS | Size MB |
|---|---|---|---|---|---|
| ContextGS_lowrate | 4.3 | 30.09 | .907 | .265 | 3.7 |
| HAC-highrate | 4.4 | 30.34 | .906 | .258 | 6.7 |
| CodecGS | 4.8 | 29.81 | .906 | .251 | 9.0 |
| HAC-lowrate | 5.0 | 29.98 | .902 | .269 | 4.6 |
| gsplat-1.00M | 5.3 | (not reported) | | | |
| ContextGS_highrate | 5.8 | 30.41 | .909 | .259 | 6.9 |
| Compact3D 32K | 9.3 | 29.90 | .907 | .251 | 13.0 |
| Compact3D 16K | 9.7 | 29.90 | .906 | .252 | 12.0 |
| RDO-Gaussian | 9.8 | 29.63 | .902 | .252 | 18.0 |
| CompGS | 9.8 | 29.69 | .901 | .279 | 9.2 |
| Reduced3DGS | 10.4 | 29.63 | .902 | .249 | 18.0 |
| SOG w/o SH | 10.5 | 29.12 | .892 | .270 | 5.7 |
| MesonGS c3 | 12.0 | 29.48 | .903 | .252 | 29.0 |
| Compressed3D | 12.4 | 29.38 | .898 | .253 | 25.3 |
| MesonGS c1 | 12.6 | 29.50 | .903 | .251 | 31.1 |
| SOG | 12.7 | 29.26 | .894 | .268 | 17.7 |
| Compact3DGS+PP | 13.3 | 29.73 | .900 | .258 | 23.8 |
| EAGLES | 14.4 | 29.86 | .91 | .25 | 52.0 |
| Scaffold-GS | 14.6 | 30.21 | .906 | .254 | 66.0 |
| Compact3DGS | 14.9 | 29.79 | .901 | .258 | 43.2 |
| LightGaussian | 15.3 | (not reported) | | | |
| 3DGS-30K | 15.3 | 29.41 | .903 | .243 | 676.0 |
| EAGLES-Small | 17.2 | 29.92 | .90 | .25 | 33.0 |

**Synthetic NeRF (as reported, 3DGS.zip Table 1)**

| Method | Rank | PSNR | SSIM | LPIPS | Size MB |
|---|---|---|---|---|---|
| HAC-highrate | 4.4 | 33.71 | .968 | .034 | 2.0 |
| HAC-lowrate | 5.0 | 33.24 | .967 | .037 | 1.2 |
| RDO-Gaussian | 9.8 | 33.12 | .967 | .035 | 2.3 |
| SOG w/o SH | 10.5 | 31.37 | .959 | .043 | 2.0 |
| MesonGS c3 | 12.0 | 32.96 | .968 | .033 | 3.5 |
| Compressed3D | 12.4 | 32.94 | .967 | .033 | 3.7 |
| MesonGS c1 | 12.6 | 32.94 | .968 | .033 | 3.9 |
| SOG | 12.7 | 33.23 | .966 | .034 | 4.1 |
| Compact3DGS+PP | 13.3 | 32.88 | .968 | .034 | 2.8 |
| Compact3DGS | 14.9 | 33.33 | .968 | .034 | 5.8 |
| LightGaussian | 15.3 | 32.72 | .965 | .037 | 7.8 |
| 3DGS-30K | 15.3 | 33.32 | (blank) | (blank) | (blank) |

Methods absent from the Synthetic NeRF block: ContextGS_lowrate, ContextGS_highrate, CodecGS, gsplat-1.00M, Compact3D 16K/32K, CompGS, Reduced3DGS, EAGLES, EAGLES-Small, Scaffold-GS.

Rate points: the survey prints exactly the variants listed above (lowrate/highrate for HAC and ContextGS, 16K/32K for Compact3D, c1/c3 for MesonGS, EAGLES/EAGLES-Small, SOG/SOG w/o SH, Compact3DGS/Compact3DGS+PP). It states on p12: "Some approaches have additional data points, which for clarity were not included in the table." Those extra points appear only as curves in Figure 12 (PSNR vs size), Figure 16 (SSIM) and Figure 17 (LPIPS) and are not numerically extractable. Figures 16 and 17 carry a legend entry "IGS" that does not appear in any table or in the text.

### 1.2 Table 2 (p16): compaction methods, three datasets

Caption as printed: "Performance comparison of 3DGS compaction methods across three datasets: Tanks and Temples, Mip-NeRF 360, and Deep Blending. The included metrics are PSNR, SSIM, LPIPS and, number of Gaussians." The count column is "k Gauss" (thousands of Gaussians). Rank formula (p15): rank_g = rank(PSNR)/6 + rank(SSIM)/6 + rank(LPIPS)/6 + rank(k Gaussians)/2.

**Tanks and Temples (as reported, 3DGS.zip Table 2)**

| Method | Rank | PSNR | SSIM | LPIPS | k Gauss |
|---|---|---|---|---|---|
| Octree-GS | 2.7 | 24.68 | .866 | .153 | 443 |
| Mini-Splatting | 3.4 | 23.18 | .835 | .202 | 200 |
| Taming3DGS | 4.8 | 23.89 | .835 | .207 | 290 |
| Taming3DGS (Big) | 4.8 | 24.04 | .851 | .170 | 1,840 |
| AtomGS | 4.9 | 23.70 | .849 | .166 | 1,480 |
| GaussianPro | 5.0 | 24.09 | .862 | .185 | 1,441 |
| Color-cued GS | 5.5 | 23.18 | .830 | .198 | 370 |
| Mini-Splatting-D | 5.7 | 23.23 | .853 | .140 | 4,280 |
| 3DGS-30K | 6.6 | 23.14 | .841 | .183 | 1,783 |

**Mip-NeRF 360 (as reported, 3DGS.zip Table 2)**

| Method | Rank | PSNR | SSIM | LPIPS | k Gauss |
|---|---|---|---|---|---|
| Octree-GS | 2.7 | 28.05 | .819 | .217 | 657 |
| Mini-Splatting | 3.4 | 27.34 | .822 | .217 | 490 |
| Taming3DGS | 4.8 | 27.29 | .799 | .253 | 630 |
| Taming3DGS (Big) | 4.8 | 27.79 | .822 | .205 | 3,310 |
| AtomGS | 4.9 | 27.38 | .816 | .211 | 3,140 |
| GaussianPro | 5.0 | 27.43 | .813 | .219 | 3,403 |
| Color-cued GS | 5.5 | 27.07 | .797 | .249 | 646 |
| Mini-Splatting-D | 5.7 | 27.51 | .831 | .176 | 4,690 |
| 3DGS-30K | 6.6 | 27.21 | .815 | .214 | 3,362 |

**Deep Blending (as reported, 3DGS.zip Table 2)**

| Method | Rank | PSNR | SSIM | LPIPS | k Gauss |
|---|---|---|---|---|---|
| Octree-GS | 2.7 | 30.49 | .912 | .241 | 112 |
| Mini-Splatting | 3.4 | 29.98 | .908 | .253 | 350 |
| Taming3DGS | 4.8 | 27.79 | .822 | .263 | 270 |
| Taming3DGS (Big) | 4.8 | 30.14 | .907 | .235 | 2,810 |
| AtomGS | 4.9 | (not reported) | | | |
| GaussianPro | 5.0 | 29.79 | .913 | .222 | 2,582 |
| Color-cued GS | 5.5 | 29.71 | .902 | .255 | 644 |
| Mini-Splatting-D | 5.7 | 29.88 | .906 | .211 | 4,630 |
| 3DGS-30K | 6.6 | 29.41 | .903 | .243 | 2,975 |

Note (p12): "Note that some methods mentioned in this section are not included in Table 2 due to incomplete evaluations, particularly in the number of Gaussians, which excludes them from the comparison." Named in the text but absent from Table 2: FreGS, Pixel-GS, Revising Densification (RDGS), MVG-Splatting, 3DGS-MCMC, SUNDAE, and the pruning components of Compact3DGS, RDO-Gaussian, HAC, LightGaussian, EAGLES, Reduced3DGS.

### 1.3 Table 3 (p18): training resolution versus evaluation resolution, PSNR on Mip-NeRF 360, trained with gsplat (as reported)

| train. res. | eval full, 360 K Gaussians | eval full, 1 M | eval 1600px, 360 K | eval 1600px, 1 M |
|---|---|---|---|---|
| full | 26.38 | 26.97 | 26.33 | 26.67 |
| scaling factor 2/4 | 25.58 | 25.84 | 26.40 | 27.02 |
| 1600px | 25.82 | 26.05 | 26.69 | 27.33 |

Conclusion drawn by the survey (p18): "Consequently, it is advised to carefully select training and evaluation sizes, acknowledging that this decision affects the final PSNR outcome as well as the comparative analysis of different compression approaches."

---

## 2. 3DGS.zip: taxonomy, size protocol, datasets, website

### 2.1 Taxonomy (exact wording)

Two top-level categories (abstract, p1): "compression, which focuses on reducing file size, and compaction, which aims to minimize the number of Gaussians. Both methods aim to maintain or improve quality, each by minimizing its respective attribute: file size for compression and Gaussian count for compaction."

Section 4 structure (p9, p11 to p15):

- 4.1 Compression
  - 4.1.1 Efficient Representation of Gaussian Attributes ("attribute compression"): vector quantization with K-means codebooks (LightGaussian, Compact3DGS, Compact3D, Compressed3D, Reduced3DGS), sensitivity-aware VQ (Compressed3D), entropy-constrained VQ (RDO-Gaussian), adaptive quantization for entropy coding (HAC), latent quantization with an MLP decoder and STE (EAGLES), SH pruning and adaptive SH bands (Reduced3DGS, SOG w/o SH, LightGaussian distillation), lower bit-depth (fp16, SOG clipping and rounding with q = 2¹⁴ for coordinates, 2⁶ for scale, opacity, rotation, 2⁵ for SH), packing with LZ77 (MesonGS) or JPEG XL (SOG).
  - 4.1.2 Structured Representation of Gaussians: anchor-based (Scaffold-GS, ContextGS with hierarchical anchor levels, HAC with hash-grid context, CompGS with anchor and coupled primitives), hash-grid plus MLP for color (Compact3DGS), 2D grid sorting (SOG, gsplat), tri-planes plus video codec (CodecGS), sorting by a quantized index plus RLE (Compact3D), octrees for positions (MesonGS).
  - 4.1.3 Quantitative comparison (Table 1).
- 4.2 Compaction, via Adaptive Density Control (ADC)
  - 4.2.1 Densification: Color-cued GS, FreGS, Pixel-GS, RDGS, GaussianPro, MVG-Splatting, Mini-Splatting, Taming 3DGS, AtomGS, Octree-GS, MCMC.
  - 4.2.2 Pruning: mask-based (Compact3DGS, RDO-Gaussian, HAC), importance-score based (LightGaussian, EAGLES), overlap-based (Papantonakis et al.), graph-based (SUNDAE).
  - 4.2.3 Quantitative comparison (Table 2).

Figure 11 (p10) is a Venn diagram: "Compression approaches notably Attribute Compression and Structured Representions of Gaussians intesect with Compaction approaches."

Section 3 "Fundamentals" is the toolkit behind the taxonomy: 3.1 Vector Quantization, 3.2 Structuring and Dimensionality Reduction (3.2.1 Octrees, 3.2.2 Anchor-based Representations, 3.2.3 Multi-Resolution Hash Grids, 3.2.4 Z-order Curves, 3.2.5 Tri-planes and K-planes, 3.2.6 Self-Organizing Gaussians, 3.2.7 RAHT, 3.2.8 Discussion), 3.3 Attribute Pruning, 3.4 Compaction (clone, split, prune).

### 2.2 Size measurement protocol

- Unit (p16, Sec 5.2): "Model Size is represented either as the file size in megabytes (MB, with 1 MB = 1000² Bytes) or the total number of Gaussians in the model."
- Source of numbers (p16, Sec 5.3): "Initially, our methodology for data compilation for this survey involved parsing various tables from numerous 3DGS compression publications. However, we have then revised our strategy and asked all authors of the publications referenced in this report to provide us with data / results in a standardized format." So the numbers are author-supplied, not re-measured by the survey, but normalised to the survey's unit and protocol.
- What counts as size: the survey never states whether MLP weights, hash grids or codebooks are counted in the reported MB. It only describes the file-size unit. Its Sec 3.2.8 (p8) discusses that hash grids and tri-planes "store features, which need to be decoded into attributes with an MLP", and that their VRAM "usage is constant", but this is about runtime memory, not the reported file size.
- Evidence of unit normalisation: for HAC and ContextGS, the 3DGS.zip sizes are 1.045 to 1.049 times the sizes printed in Ali et al. (who copy paper values), which matches the MiB to MB factor 1024²/1000² = 1.0486 (for example HAC-lowrate Mip-NeRF 360: 16.0 MB in 3DGS.zip versus 15.3 in Ali et al.). This is my observation, not a statement by either survey.
- Testing protocol requested from authors (p16, Sec 5.3): all 9 Mip-NeRF 360 scenes including "flowers" and "treehill", only "train" and "truck" from Tanks and Temples, full-resolution images up to a maximum side length of 1600 px (larger images downscaled with PIL `.resize()` bicubic following the 3DGS method), every 8th image for testing (index i with i mod 8 ≡ 0) on the three COLMAP datasets, and the predefined train/evaluation split for Synthetic NeRF.
- Table 1 has no FPS, training-time, decode-time or VRAM column. Table 2 has no such column either.

### 2.3 Datasets and splits (p15 to p17)

- Tanks and Temples: "truck", "train" (unbounded outdoor, centered viewpoint).
- Mip-NeRF 360: "bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill" (9 scenes).
- Deep Blending: "Dr Johnson", "Playroom".
- Synthetic NeRF: "chair", "drums", "ficus", "hotdog", "lego", "material", "mic", "ship".
- p16: "These scenes align with those used in the 3D Gaussian Splatting (3DGS) [21] publication, making them particularly useful for comparing compression methods, as most authors have benchmarked against them. While it would be beneficial to explore larger or more specialized scenes in future work, the lack of accessible data for such comparisons currently limits our scope."

### 2.4 Leaderboard website

Printed in the abstract (p1): `https://w-m.github.io/3dgs-compression-survey/` ("we maintain a dedicated website that will be regularly updated with new techniques and revisions of existing findings").

---

## 3. 3DGS.zip: verbatim limitation statements

Grouped by the look-for topics. Page numbers are PDF pages.

**Comparability of numbers across papers, benchmarks, reproducibility**

- p1 (abstract): "Specifically, since these methods have been developed in parallel and over a short period of time, currently, no comprehensive comparison exists."
- p11: "Furthermore some approaches were missing the necessary data for quantitative comparison, approaches that we consider worth mentioning nevertheless are included in the survey but not in the tables."
- p12: "The objectives for compressing 3DGS vary depending on the application, with some requiring minimal model size, while others prioritize a smaller size alongside optimal perceptual quality. As there is no definitive winner that excels across all categories, we introduce a simple rank, reflecting the average rankings of the methods across all available datasets, thereby offering general guidance on the overall performance of the approaches."
- p12: "While our proposed ranking puts ContextGS-lowrate [50] on the first place, high variations between datasets and quality metrics show that there is not one winning compression strategy. Depending on the application and goals, different approaches should be considered."
- p12: "It is noteworthy that the original 3DGS-30K [21] exhibits the best LPIPS values on both the Mip-NeRF 360 and the Deep Blending dataset, at the cost of being up to 70 times larger in size."
- p12: "Some approaches have additional data points, which for clarity were not included in the table."
- p12: "Note that some methods mentioned in this section are not included in Table 2 due to incomplete evaluations, particularly in the number of Gaussians, which excludes them from the comparison."
- p15: "A closer analysis reveals that even the highest-ranked method in terms of efficiency, Octree-GS, does not achieve the best performance across all datasets."
- p16: "While it would be beneficial to explore larger or more specialized scenes in future work, the lack of accessible data for such comparisons currently limits our scope."
- p16: "Initially, our methodology for data compilation for this survey involved parsing various tables from numerous 3DGS compression publications. However, we have then revised our strategy and asked all authors of the publications referenced in this report to provide us with data / results in a standardized format."
- p18: "Consequently, it is advised to carefully select training and evaluation sizes, acknowledging that this decision affects the final PSNR outcome as well as the comparative analysis of different compression approaches."
- p18: "Another key challenge is the lack of standard benchmarks for evaluating 3DGS compression methods. Current studies use different datasets, metrics, and experimental conditions, making direct comparisons difficult. In this survey we propose basic guidelines to enable comparison but establishing a unified benchmarking framework with standardized datasets and evaluation protocols would enhance reproducibility and facilitate progress in the field. In addition, methods should be evaluated not only on file size and reconstruction quality but also on real-time performance and energy efficiency, particularly for deployment on mobile and embedded systems."

**Targeting a size, controllability, rate control**

- p4: "This can be exploited in considering additional losses during training and scene optimization to limit bit-rate while keeping visual quality."
- p15 (on MCMC): "This way, the method improves the quality of regions with few initial Gaussian primitives and the number of Gaussians can directly be controlled."
- p18: "Determining the optimal trade-off between reducing attributes and minimizing Gaussian count remains an important area of study. Some compression methods prioritize extreme size reduction through entropy coding and quantization, while others focus on preserving fidelity using hierarchical encoding. Hybrid approaches that dynamically adjust compression levels based on scene complexity could be highly beneficial."
- p18: "Finally, the trade-off between compression efficiency and visual quality remains a central challenge. Approaches like HAC-highrate excel in achieving high-quality compression but at the cost of higher computational overhead. On the other hand, more aggressive compression techniques such as SOG w/o SH reduce memory usage significantly but may result in noticeable quality degradation, especially in scenes requiring fine detail. This trade-off highlights the need for flexible solutions that can be tuned based on the specific requirements of the application, whether it's focused on minimizing storage or maximizing visual realism. As the field advances, hybrid approaches that integrate machine learning-driven optimizations may offer new opportunities. Deep learning models could be trained to predict optimal compression strategies based on scene characteristics, dynamically selecting techniques that maximize efficiency while preserving quality."
- p18: "Most current 3DGS compression techniques are designed for static scenes. However, real-world applications increasingly demand dynamic and interactive capabilities. Existing methods rely on pre-trained models, limiting adaptability. Developing real-time adaptive compression strategies that adjust based on scene complexity such as changing objects or lighting conditions over time, or changing hardware constraints would be a valuable research direction."
- p19 (CompGS summary): "A rate-constrained optimization scheme further enhances compactness by jointly minimizing both rendering distortion and bit rate. The bit rate of both anchor and coupled primitives is modeled by entropy estimation."
- p19 to p20 (RDO-Gaussian summary): "The authors achieve flexible, continuous rate control by formulating 3D Gaussian representation learning as a joint optimization of rate and distortion. Rate-distortion optimization is realized through dynamic pruning and entropy-constrained vector quantization (ECVQ)."
- Observation (mine, checkable in Table 1): every multi-rate method is labelled by a variant name (lowrate/highrate, 16K/32K, c1/c3, Small, w/o SH, 1.00M), never by a byte target. The only budget-labelled entry is gsplat-1.00M, which is a Gaussian-count cap via MCMC, not a byte cap. The survey nowhere mentions a method that takes a size in bytes as input.

**Decode time, render speed of compressed models, cost of MLP or hash grid at render time**

- p6: "This makes this a hybrid explicit/implicit method, and introduces a small computational overhead for the feature decoding."
- p8: "Hash grids and tri-planes both store features, which need to be decoded into attributes with an MLP. While explicit methods allow direct access to Gaussian attributes, this implicit feature storage introduces compactness at the cost of requiring additional decoding."
- p8: "All discussed methods are designed for efficient decoding, making them suitable for real-time applications. However, encoding complexity varies: Self-Organizing Gaussians require a computationally expensive optimization to determine element order, anchor-based methods involve clustering and optimization to define representative points, and feature-based methods like tri-planes and hash grids require a training process to establish features and MLP weights for decoding. This feature decoding adds a small computational overhead at decoding and rendering time."
- p8: "These approaches may use far fewer attributes then third-degree spherical harmonics, but on the downside require custom, and potentially slower, rendering."
- p18: "Compression techniques must balance reducing file size with maintaining rendering accuracy. Neural feature encoding techniques, such as tri-planes or hash grids, significantly reduce storage needs but require additional computation during rendering. In contrast, explicit representations with precomputed attributes demand more storage but enable real-time rendering with minimal overhead. Lossy compression techniques like pruning or aggressive quantization can introduce artifacts, particularly in view-dependent rendering."
- p18: "Approaches like HAC-highrate excel in achieving high-quality compression but at the cost of higher computational overhead."

**Memory at render time, edge and mobile deployment**

- p1: "Despite its advantages in rendering speed and image fidelity, 3DGS is limited by its significant storage and memory demands. These high demands make 3DGS impractical for mobile devices or headsets, reducing its applicability in important areas of computer graphics."
- p8: "Memory consumption at decoding and rendering time depends on the representation. For explicit methods, VRAM (GPU memory) usage is directly tied to the number of primitives, meaning that reducing their count through compaction (as discussed in the following sections) directly lowers memory requirements. In contrast, methods like hash grids, tri-planes, and vector quantization can use less memory overall by storing features in compact structures such as feature planes, hash tables, or codebooks. However, their memory usage is constant, meaning that reducing the number of primitives does not further decrease VRAM consumption. Depending on whether a system is memory- or compute-constrained, different trade-offs may be preferable. Large scenes often demand significant VRAM, making compact representations beneficial, while compute-constrained devices, such as VR headsets, may have sufficient memory but benefit from pre-decoding implicit representations into explicit attributes to reduce runtime computation."
- p18: "This is vital for applications in resource-constrained environments, such as mobile devices or VR headsets, where storage, memory and processing power are limited but visual fidelity matters to the user."
- p18: "However, its practicality remains constrained by storage and computational costs. Scalability, ease of use, and adaptability across different platforms can still be improved."
- p19 (Compact3D summary): "Only a small codebook is stored along with the index of the code for each Gaussian, resulting in a large reduction in the storage of the learned radiance fields and a reduction of the memory footprint at rendering time."
- p20 (Compressed3D summary): "Furthermore, a renderer for the compressed scenes utilizing GPU-based sorting and rasterization is proposed, enabling real-time novel view synthesis on low-end devices."

**Quality cost of attribute pruning**

- p10: "It should be noted, that whenever a band or degree of spherical harmonics is removed, the higher frequency components of the color representation are also eliminated, which inherently leads to a loss of fine detail in view-dependent effects. This reduction, while beneficial for storage efficiency, inevitably sacrifices some information that would otherwise capture subtle variations in lighting and shading across different viewpoints."

**Interplay of techniques, LOD, quantization-aware training**

- p18: "While various compression techniques have been developed independently, their interplay remains an open research area."
- p18 to p19: "Another improvement could be the creation of multi resolution models with support for level-of-detail (LOD) scaling. Such models would optimize performance by allowing different parts of a scene to be rendered with varying levels of detail, depending on real-time requirements. Octrees or feature grids, provide scalable solutions, but ensuring seamless transitions between different levels of detail without introducing artifacts remains a challenge. Quantization-aware training and the development of shared codebooks across scenes or applications could further improve compression efficiency, allowing for reduced redundancy and more effective memory usage."

**Training time**: the survey contains no statement about training time as a limitation and no training-time column. Encoding cost is mentioned only qualitatively (p8, quoted above).

---

## 4. Ali et al. (2025): taxonomy, tables, future directions

### 4.1 Taxonomy (Table I, p2, and Figure 1, p2)

Two top-level categories (p6): "i) Unstructured Compression: Those primarily concerned with compressing the "value" of model parameters N, utilizing methods like pruning [20], [21], quantization [20]–[22], [36], and entropy constraints [23] without considering the relationship between the Gaussians." and "ii) Structured Compression: those exploring compression techniques that consider the relationships between Gaussians [41], [42]."

Table I as printed (Category, Sub-Type, Representative Publications):

| Category | Sub-Type | Representative Publications |
|---|---|---|
| Unstructured Compression | Pruning | LightGaussian [21], Compact 3D [20], CompGS [22], EAGLES [23], Papantonakis et al. [24], Trimming the Fat [25], SafeguardGS [26], EfficientGS [27], RDO-Gaussian [28], Pup 3D-GS [29], Kim et al. [30], RTGS [31], Kheradmand et al. [32], LP-3DGS [33], Mini-Splatting [34], GoDe [35], ELMGS [18] |
| Unstructured Compression | Quantization | LightGaussian [21], Compact 3D [20], CompGS [22], Niedermayr et al. [36], EAGLES [23], Papantonakis et al. [24], RDO-Gaussian [28], Morgenstern et al. [37], GoDe [35], ELMGS [18] |
| Unstructured Compression | Entropy Coding | Compact 3D [20], CompGS [22], Niedermayr et al. [36], EAGLES [23], Morgenstern et al. [37], ELMGS [18] |
| Structured Compression | Graph-Based | SUNDAE [38], Gaussian-Forest (GF) [39], Liu et al. [40] |
| Structured Compression | Anchor-Based | Scaffold-GS [41], HAC [42], Context-GS [43] |
| Structured Compression | Contextual/AR Modeling | Context-GS [43] |
| Structured Compression | Factorization Approach | F-3DGS [44], Radsplat [45] |

Figure 1 leaves: Unstructured → Pruning (Spatial, Opacity Based, Gradient Based, Size Based), Quantization (Scalar Quantization, Vector Quantization), Entropy Encoding (Hierarchal Encoding, Hash Grids). Structured → Graph Based, Anchor Based, Contextual/AR Modelling, Factorization Approach. Section IV.A additionally names "significance-scoring-based" pruning (LightGaussian, EAGLES, Papantonakis, SafeguardGS, LP-3DGS).

Reference-key resolution (important because the naming collides with 3DGS.zip): in Ali et al., "Compact 3D [20]" / "Compact3D [20]" is Lee et al. "Compact 3d gaussian representation for radiance field" (3DGS.zip's Compact3DGS). "CompGS [22]" / "CompGS-16K/32K" is Navaneet et al. "Compgs: Smaller and faster gaussian splatting with vector quantization" (ECCV 2024, 3DGS.zip's Compact3D). "CompGS (highrate/lowrate) [40]" is Liu et al. "Compgs: Efficient 3d scene representation via compressed gaussian splatting" (3DGS.zip's CompGS). "Niedermayr et al. [36]" is Compressed3D. "Morgenstern et al. [37]" is SOG. "Papantonakis et al. [24]" is Reduced3DGS. "Kim et al. [30]" is Color-cued GS. "Kheradmand et al. [32]" is 3DGS-MCMC.

### 4.2 Protocol (p6, Sec III)

- "Evaluation Metrics: The compression cost is quantified either as the bit count of the compressed representation or its compression ratio compared to the baseline 3DGS-30k (trained for 30, 000 iterations). The performance evaluation of 3DGS compression relies on image-based fidelity metrics such as PSNR ..., SSIM ..., and LPIPS ..., and its computational efficiency is measured in terms of Frames per second (FPS)."
- "Datasets: ... 9 scenes from Mip-NeRF360 [97], which includes both indoor and outdoor scenes, two scenes from Tanks&Temples [95], and the Deep Blending [96] dataset. ... studies typically adhere to the train-test split used in Mip-NeRF360 [97] and 3DGS where every 8th image must be selected for testing."
- Table captions (p7, p11): "All values are sourced from their respective papers. Memory size is reported in megabytes (MB)." No unit convention (MB versus MiB) is stated, and nothing is said about whether MLP or hash-grid parameters are counted. "Comp." is the compression ratio against 3DGS-30k.
- FPS (p9, Fig. 7 and p13, Fig. 10): "The FPS for all the techniques is evaluated on a single NVIDIA A40 GPU." FPS appears only in figures, not in any table, so no FPS numbers are transcribable except the two quoted in the text (ELMGS "surpassing 400 FPS on Deep Blending and 600 FPS on Tanks&Temples", p9).
- No Synthetic NeRF results, no Gaussian counts in the tables, no training-time column.

### 4.3 Table II (p7): unstructured methods, Mip-NeRF360 and Tanks&Temples (as reported)

| Model | Mip SSIM | Mip PSNR | Mip LPIPS | Mip Mem. MB | Mip Comp. | T&T SSIM | T&T PSNR | T&T LPIPS | T&T Mem. MB | T&T Comp. |
|---|---|---|---|---|---|---|---|---|---|---|
| 3DGS-30k [93] | 0.815 | 27.21 | 0.214 | 734.0 | 1× | 0.841 | 23.14 | 0.183 | 411.0 | 1× |
| LightGaussian [21] | 0.805 | 27.28 | 0.243 | 42.0 | 18× | 0.817 | 23.11 | 0.231 | 22.0 | 19× |
| Compact3D [20] | 0.798 | 27.08 | 0.247 | 48.8 | 15× | 0.831 | 23.32 | 0.201 | 39.4 | 10× |
| CompGS-16K [22] | 0.804 | 27.03 | 0.243 | 18.0 | 41× | 0.836 | 23.39 | 0.200 | 12.0 | 34× |
| CompGS-32K [22] | 0.806 | 27.12 | 0.240 | 19.0 | 39× | 0.838 | 23.44 | 0.198 | 13.0 | 32× |
| Niedermayr et al. [36] | 0.801 | 26.98 | 0.238 | 28.8 | 26× | 0.832 | 23.32 | 0.194 | 17.3 | 24× |
| EAGLES [23] | 0.810 | 27.23 | 0.240 | 54.0 | 14× | 0.840 | 23.37 | 0.200 | 29.0 | 14× |
| Papantonakis et al. [24] | 0.809 | 27.10 | 0.226 | 29.0 | 25× | 0.840 | 23.57 | 0.188 | 14.0 | 29× |
| Trimming the Fat [25] | 0.798 | 27.13 | 0.248 | 20.1 | 37× | 0.831 | 23.69 | 0.210 | 8.6 | 48× |
| Efficientgs [27] | 0.817 | 27.38 | 0.216 | 98.0 | 8× | 0.837 | 23.45 | 0.197 | 33.0 | 13× |
| RDO-Gaussian [28] | 0.802 | 27.05 | 0.239 | 23.5 | 31× | 0.835 | 23.34 | 0.195 | 12.0 | 34× |
| PUP 3D-GS [29] | 0.792 | 26.83 | 0.268 | 86.3 | 9× | 0.807 | 23.03 | 0.245 | 50.1 | 8× |
| Kim et al. [30] | 0.797 | 27.07 | 0.249 | 73.0 | 10× | 0.830 | 23.18 | 0.198 | 42.0 | 10× |
| ELMGS-medium [18] | 0.792 | 27.31 | 0.264 | 38.6 | 19× | 0.838 | 24.08 | 0.191 | 18.8 | 22× |
| ELMGS-small [18] | 0.779 | 27.00 | 0.286 | 25.8 | 28× | 0.825 | 23.90 | 0.233 | 11.6 | 35× |

### 4.4 Table III (p7): unstructured methods, Deep Blending (as reported)

| Model | SSIM | PSNR | LPIPS | Mem. MB | Comp. |
|---|---|---|---|---|---|
| 3DGS-30k [93] | 0.903 | 29.41 | 0.243 | 676.0 | 1× |
| Compact3D [20] | 0.901 | 29.79 | 0.258 | 43.2 | 16× |
| CompGS-16K [22] | 0.906 | 29.90 | 0.252 | 12.0 | 56× |
| CompGS-32K [22] | 0.907 | 29.90 | 0.251 | 13.0 | 52× |
| Niedermayr et al. [36] | 0.898 | 29.38 | 0.253 | 25.3 | 27× |
| EAGLES [23] | 0.910 | 29.86 | 0.250 | 52.0 | 13× |
| Papantonakis et al. [24] | 0.902 | 29.63 | 0.249 | 18.0 | 38× |
| Trimming the Fat [25] | 0.897 | 29.43 | 0.267 | 12.5 | 54× |
| Efficientgs [27] | 0.903 | 29.63 | 0.251 | 40.0 | 17× |
| RDO-Gaussian [28] | 0.902 | 29.63 | 0.252 | 18.0 | 38× |
| PUP 3D-GS [29] | 0.881 | 28.61 | 0.305 | 80.8 | 8× |
| Kim et al. [30] | 0.902 | 29.71 | 0.255 | 72.0 | 9× |
| ELMGS-medium [18] | 0.897 | 29.48 | 0.261 | 23.5 | 29× |
| ELMGS-small [18] | 0.894 | 29.24 | 0.273 | 12.3 | 55× |

LightGaussian has no Deep Blending row in Ali et al., consistent with 3DGS.zip.

### 4.5 Table IV (p11): structured methods, Mip-NeRF360 and Tanks&Temples (as reported)

| Model | Mip SSIM | Mip PSNR | Mip LPIPS | Mip Mem. MB | Mip Comp. | T&T SSIM | T&T PSNR | T&T LPIPS | T&T Mem. MB | T&T Comp. |
|---|---|---|---|---|---|---|---|---|---|---|
| 3DGS-30k [93] | 0.815 | 27.21 | 0.214 | 734.0 | 1× | 0.841 | 23.14 | 0.183 | 411.0 | 1× |
| ScaffoldGS [41] | 0.848 | 28.84 | 0.220 | 102.0 | 7× | 0.853 | 23.96 | 0.177 | 87.0 | 5× |
| GF Large [39] | 0.803 | 27.45 | 0.212 | 85.0 | 9× | 0.839 | 23.67 | 0.188 | 45.0 | 9× |
| GF Small [39] | 0.797 | 27.33 | 0.219 | 50.0 | 15× | 0.836 | 23.56 | 0.194 | 38.0 | 11× |
| SUNDAE (30%) [38] | 0.826 | 27.24 | 0.228 | 279.0 | 3× | 0.817 | 23.46 | 0.242 | 148.0 | 3× |
| SUNDAE (1%) [38] | 0.716 | 24.70 | 0.375 | 38.0 | 19× | 0.703 | 20.49 | 0.375 | 33.0 | 13× |
| HAC-lowrate [42] | 0.807 | 27.53 | 0.238 | 15.3 | 48× | 0.846 | 24.04 | 0.187 | 8.1 | 51× |
| HAC-highrate [42] | 0.811 | 27.77 | 0.230 | 21.9 | 34× | 0.853 | 24.40 | 0.177 | 11.2 | 37× |
| ContextGS (lowrate) [43] | 0.808 | 27.62 | 0.237 | 12.7 | 58× | 0.852 | 24.20 | 0.184 | 7.1 | 58× |
| ContextGS (highrate) [43] | 0.811 | 27.75 | 0.231 | 18.4 | 40× | 0.855 | 24.29 | 0.176 | 11.8 | 35× |
| CompGS (highrate) [40] | 0.800 | 27.26 | 0.240 | 16.5 | 45× | 0.840 | 23.70 | 0.210 | 9.6 | 43× |
| CompGS (lowrate) [40] | 0.780 | 26.37 | 0.280 | 8.8 | 83× | 0.810 | 23.11 | 0.240 | 5.9 | 70× |

### 4.6 Table V (p11): structured methods, Deep Blending (as reported)

| Model | SSIM | PSNR | LPIPS | Mem. MB | Comp. |
|---|---|---|---|---|---|
| 3DGS-30k [93] | 0.903 | 29.41 | 0.243 | 676.0 | 1× |
| ScaffoldGS [41] | 0.906 | 30.21 | 0.254 | 66.0 | 10× |
| GF Large [39] | 0.908 | 30.18 | 0.215 | 98.0 | 7× |
| GF Small [39] | 0.905 | 30.11 | 0.223 | 64.0 | 11× |
| SUNDAE (30%) [38] | 0.899 | 29.40 | 0.248 | 203.0 | 3× |
| SUNDAE (1%) [38] | 0.861 | 26.57 | 0.355 | 36.0 | 19× |
| HAC-lowrate [42] | 0.902 | 29.98 | 0.269 | 4.4 | 154× |
| HAC-highrate [42] | 0.906 | 30.34 | 0.258 | 6.4 | 106× |
| ContextGS (lowrate) [43] | 0.907 | 30.11 | 0.265 | 3.6 | 188× |
| ContextGS (highrate) [43] | 0.909 | 30.39 | 0.258 | 6.6 | 102× |
| CompGS (highrate) [40] | 0.900 | 29.69 | 0.280 | 8.8 | 77× |
| CompGS (lowrate) [40] | 0.900 | 29.30 | 0.290 | 6.0 | 113× |

Figure 4 (p6) is a pruning illustration with printed numbers for one unnamed scene (as reported):

| Pruning | Gaussians | PSNR | Size |
|---|---|---|---|
| Opacity Based | 2.65M | 25.1 dB | 629 MB |
| Opacity Based | 2.01M | 24.9 dB | 478 MB |
| Opacity Based | 1.14M | 24.0 dB | 270 MB |
| Opacity Based | 0.62M | 23.1 dB | 148 MB |
| Opacity + Gradient Based | 1.93M | 25.1 dB | 526 MB |
| Opacity + Gradient Based | 1.19M | 25.0 dB | 326 MB |
| Opacity + Gradient Based | 0.74M | 24.7 dB | 203 MB |
| Opacity + Gradient Based | 0.40M | 24.1 dB | 110 MB |

### 4.7 Verbatim future-direction and challenge statements in Ali et al.

**Deployment, edge, mobile, memory**

- p1: "However, despite its advantages, 3DGS suffers from substantial memory and storage requirements, posing challenges for deployment on resource-constrained devices."
- p2: "Compared with NeRFs, 3DGS has the advantage of faster rendering speed but at the cost of higher demand of memory with the need to store millions of Gaussians. This limits their application in resource-constrained devices, like VR/AR or game environments."
- p3: "(2) We systematically analyze existing compression techniques, highlighting key challenges such as scalability constraints in large-scale scenes, dependence on vector quantization, suboptimal loss function designs, and limitations in deploying 3DGS on resource-constrained hardware."
- p6: "3DGS faces significant scalability challenges compared to NeRFs. While NeRFs require only the storage of weight parameters for a multilayer perceptron (MLP), 3DGS necessitates storing the parameters of millions of Gaussians per scene. This issue becomes especially critical in large, complex scenes, where computational and memory demands increase significantly. The number of Gaussians is directly proportional to storage and computational complexity and inversely proportional to rendering efficiency. Therefore, optimizing memory usage and computational efficiency for both storage and rendering is essential to improve the scalability of 3DGS-based methods."
- p8: "However, even after removing redundant Gaussians through pruning, millions of Gaussians are still required for accurate scene reconstruction."
- p10: "Despite substantial progress in compressing Gaussian splats, the majority of research has primarily concentrated on reducing the storage footprint of 3DGS. A critical challenge that remains unaddressed is the densification of 3DGS, particularly for large scenes like those in the Mip-NeRF dataset [97], such as the garden and bicycle scenes, which can demand up to 15GB of GPU memory for training and rendering. This level of scaling is unsustainable for large-scale scenes, indicating a pressing need for unstructured 3DGS compression techniques to tackle this issue."
- p10: "Furthermore, most of the existing compression methods rely on vector quantization. However, recent advances in model compression for deep learning have demonstrated that scalar quantization is more favorable and easier to implement in hardware, especially for low-powered edge devices [104], [106]. Therefore, developing specialized scalar quantization techniques for efficiently rendering Gaussian splats on such devices should be a priority."
- p13: "Challenges and Future Direction: Despite these advancements, challenges remain. Vector quantization, while effective for compression, is computationally expensive compared to scalar quantization [18]. Additionally, optimizing structured methods for low-power edge devices remains underexplored. Adaptive rasterization strategies, such as those proposed by Niedermayr et al. [36], could enhance rendering efficiency while refining loss functions could further improve 3DGS quality [107]. Addressing these issues will enable more efficient and high-quality 3DGS compression techniques, benefiting both structured and unstructured approaches."
- p14: "Despite these advances, the scalability of 3DGS for large and complex scenes remains a significant challenge, especially for resource-constrained environments like mobile AR/VR devices. The field also lacks a unified framework that integrates the strengths of structured and unstructured compression methods, limiting the adaptability and standardization of 3DGS models. Furthermore, existing research has not sufficiently explored the potential of novel scalar quantization techniques or the optimization of loss functions tailored specifically for 3DGS. Addressing these gaps could unlock higher compression efficiencies, improved rendering speeds, and enhanced fidelity, making 3DGS more practical for real-world deployment."
- p14: "Looking forward, the future of 3DGS lies in developing hybrid compression frameworks, optimizing models for edge devices, and expanding its applications across diverse domains. Scalar quantization techniques, loss function innovations, and hardware-aware optimizations will be critical to enabling real-time rendering on low-power devices."

**Render speed of compressed models, decode cost, cost of the MLP or hash grid**

- p9: "Entropy encoding serves as a crucial final step in 3DGS compression, significantly enhancing compression efficiency by further reducing storage redundancy. However, it introduces additional computational complexity during decoding. Methods such as RDO-Gaussian, which integrate entropy encoding with quantization and pruning, offer the most efficient end-to-end compression pipelines, balancing compression ratio, and computational complexity."
- p9: "FPS: Frame rate is a crucial factor in real-time rendering applications, ensuring whether a method is viable for interactive environments such as virtual reality and gaming or low-power edge devices. The rendering speeds are calculated on a single NVIDIA A40 GPU. The FPS vs. PSNR plots in Figure 7 highlight a clear trade-off between rendering speed and reconstruction accuracy."
- p11: "However, HAC encodes all anchors simultaneously, leaving room for further optimization to reduce spatial redundancy."
- p12: "FPS: Structured compression methods, while achieving high compression rates, often maintain FPS similar to or lower than the 3DGS-30k baseline, as shown in Figure 10. GF and HAC, despite their efficient compression, exhibit slower rendering speeds due to high encoding/decoding complexity, which improves compression but reduces rendering efficiency. CompGS, however, demonstrates better rendering speeds than 3DGS-30k, HAC, and GF across all benchmark datasets, making it a more balanced approach between compression efficiency and rendering speed."
- p12: "Although structured compression techniques offer superior storage efficiency compared to unstructured methods, there remains a trade-off between extreme compression and rendering speeds."
- p12: "Challenges and Future Direction: Structured compression organizes sparse Gaussians, offering significant memory footprint reduction as discussed in earlier sections. However, the added complexity from hashing [42], offsetting [41], encoding, and decoding [40] introduces computational overhead, which impacts rendering efficiency. Unlike unstructured compression, where compression gains typically lead to faster rendering, structured methods do not inherently translate compression efficiency into rendering efficiency."
- p12: "One challenge with structured compression lies in the diversity of methodologies employed by different approaches, which complicates their scalability and standardization. Unlike unstructured compression methods, structured techniques cannot be easily integrated into the baseline 3DGS as plug-and-play solutions."
- p12: "To advance the field, future research should aim to merge insights from unstructured compression into structured approaches, particularly to enhance rendering speed. Given that techniques inspired by LIC have already influenced 3DGS, it is likely that future work will delve deeper into exploring relationships among Gaussians, developing strategies for effective grouping and combining of Gaussians, and optimizing their encoding and decoding processes for greater efficiency and effectiveness."
- p13: "FPS: Structured methods remain comparable to the 3DGS-30K baseline, whereas unstructured techniques achieve 4× to 5× FPS improvements, as shown in Figure 10. This makes unstructured compression preferable for real-time and low-power applications, where efficiency is critical. While structured methods introduce additional parameters that can slightly affect rendering speed, they still maintain a high FPS suitable for interactive applications."
- p14 (Conclusion): "Structured compression methods leverage the relationships between Gaussians to achieve compression ratios of up to 100× compared to baseline 3DGS while maintaining high fidelity and perceptual quality. However, this comes at the cost of rendering speed, with FPS comparable to baseline 3DGS-30k. In contrast, unstructured compression methods achieve compression ratios of up to 50×, preserving fidelity similar to baseline 3DGS-30k, while significantly improving rendering speeds by up to 7×."

**Rate control, budgets, adaptivity, targeting**

- p9: "RDO-Gaussian [28] is among the first to introduce an end-to-end rate-distortion framework, dynamically adjusting compression based on a quality-loss trade-off."
- p9: "Although VQ is widely used in 3DGS-based methods, recent advances suggest that SQ offers better hardware efficiency, particularly for deployment in edge devices [104]. ELMGS [18] used learned step-size-based uniform quantization [105] for quantization. However, uniform SQ suffers from a performance drop at lower bit-depths due to the non-uniform distribution of 3DGS attributes such as opacity as seen in Figure 5 [105]. The distribution of opacity in 3DGS is highly non-uniform, with peaks at both lower and higher opacity levels. This underscores the necessity for non-uniform SQ methods specifically tailored for 3DGS quantization, ensuring better preservation of scene fidelity even at extremely low bit-depths."
- p10: "These findings indicate that no single method dominates all aspects, highlighting the importance of application-specific selection when choosing an appropriate unstructured compression technique for 3DGS."
- p10: "Additionally, none of the current compression works have investigated the behavior of loss functions in relation to different compression methods. In image compression, it has been shown that altering the loss function can significantly impact compression performance [107]. Thus, future research should explore the effects of various loss functions, such as perceptual loss and edge-based loss, on the performance of 3DGS compression. This could lead to more efficient and perceptually optimized compression techniques for 3DGS."
- p13: "This paradigm suggests that 3DGS compression could similarly benefit from end-to-end deep learning techniques that directly optimize 3D Gaussian representations, rather than relying merely on traditional data compression methods."
- p13: "The concept of progressive compression, where data is encoded in multiple stages to maintain high fidelity in key regions while aggressively compressing less critical data, could offer a path forward for improving 3DGS techniques, especially in complex 3D scenes where different regions (eg., background vs. foreground) require different levels of precision."
- p14: "Furthermore, techniques like adaptive quantization and neural compression employed in NeRFs open the door for utilizing machine learning models to optimize the encoding of Gaussian components. Such models can automatically adjust the precision of 3D Gaussian parameters based on the local geometric complexity and importance of the regions being encoded, or simply depth, concept explored in [117]. By adopting similar principles, 3DGS compression could be enhanced to dynamically adjust encoding strategies based on the structure of 3D data."
- p14: "To achieve a favorable compression ratio, it is critical to focus on lossy compression methods and address a key question: what properties of point clouds should be preserved within a limited bitrate budget?"
- p14: "Despite the fact that the output of 3DGS is a point cloud, there has been little focus on combining 3DGS compression with dedicated point cloud compression techniques. While the compression methods used in 3DGS result in a compressed point cloud, this is more of a byproduct of those methods rather than a direct effort to compress the point cloud itself."
- p14: "Inspired by recent advancements in point cloud compression [118], [119], it is worth exploring how these techniques can be integrated into the end-to-end training process of 3DGS or developed as a separate module specifically tailored for compressing the output of 3DGS."

**Comparability**: Ali et al. do not discuss comparability of numbers across papers. Their only protocol statement is "All values are sourced from their respective papers" (p7, p11). **Training time**: no statement beyond the 15 GB GPU memory figure for training and rendering on garden and bicycle (p10).

---

## 5. Methods named in either survey that are not in the known list

Known list (given): 3DGS, 2DGS, Mip-Splatting, Scaffold-GS, 3DGS-MCMC, SizeGS, GETA-3DGS, FlexGaussian, KISS-GS, RDO-Gaussian, HAC, HAC++, ContextGS, HEMGS, CAT-3DGS, PCGS, SALVQ, FCGS, CompGS, MesonGS, Taming 3DGS, Mini-Splatting, GaussianSpa, MaskGaussian, PUP-3DGS, SPARE-GS, Smart target point control, LightGaussian, Compact3DGS (Lee), Self-Organizing Gaussians, Compressed 3DGS (Niedermayr).

Two names in the known list are ambiguous against the surveys: "CompGS" matches both Navaneet et al. (called Compact3D in 3DGS.zip, CompGS-16K/32K in Ali) and Liu et al. (called CompGS in 3DGS.zip, CompGS (highrate/lowrate) in Ali). Both are listed below so the caller can decide. Mip-NeRF 360 numbers are PSNR / SSIM / LPIPS / size, as reported by the named survey.

### 5.1 Compression methods

| Name as printed | First author, year (as printed) | What it does (one line) | Mip-NeRF 360 (as reported) |
|---|---|---|---|
| CodecGS (3DGS.zip) | S. Lee, 2025 (arXiv 2501.03399) | Progressive tri-planes predict Gaussian attributes from positions, block DCT entropy modeling, planes stored as 16-bit YUV and encoded with a standard video codec | 27.30 / .810 / .236 / 10.3 MB (3DGS.zip) |
| gsplat, gsplat-1.00M (3DGS.zip) | V. Ye, M. Turkulainen and the Nerfstudio team (no year) | 3DGS-MCMC training with a 1.00 M Gaussian cap, SOG-style 2D grid sorting, SH clustering, FP16 storage, in the gsplat library | 27.29 / .811 / .229 / 16.0 MB (3DGS.zip) |
| Compact3D 16K, Compact3D 32K (3DGS.zip), CompGS-16K/32K [22] (Ali) | K. Navaneet, 2024 (arXiv 2311.18159, ECCV 2024) | K-means VQ of Gaussian parameters into a codebook (16K or 32K entries), indices sorted and RLE-coded, opacity regularizer before pruning | 32K: 27.12 / .806 / .240 / 19.0 MB. 16K: 27.03 / .804 / .243 / 18.0 MB (both surveys agree) |
| CompGS (3DGS.zip), CompGS (highrate/lowrate) [40] (Ali) | X. Liu, 2024 (ACM MM 2024) | Anchor primitives plus coupled primitives with residual embeddings, rate-constrained optimization with entropy-estimated bit rate | 3DGS.zip: 27.26 / .803 / .239 / 17.3 MB. Ali highrate: 27.26 / 0.800 / 0.240 / 16.5 MB. Ali lowrate: 26.37 / 0.780 / 0.280 / 8.8 MB |
| Reduced3DGS (3DGS.zip), Papantonakis et al. [24] (Ali) | P. Papantonakis, 2024 | Scale- and resolution-aware redundant primitive removal, adaptive SH band reduction per Gaussian, K-means codebook plus 16-bit half floats | 27.10 / .809 / .226 / 29.0 MB (both surveys agree) |
| EAGLES, EAGLES-Small (3DGS.zip and Ali) | S. Girish, 2024 (ECCV 2024) | Latent quantization of color and rotation decoded by an MLP with STE, opacity quantization, pruning by a transmittance-and-opacity influence score | EAGLES: 27.23 / .81 / .24 / 54.0 MB (3DGS.zip), 27.23 / 0.810 / 0.240 / 54.0 MB (Ali). EAGLES-Small: 26.94 / .80 / .25 / 47.0 MB (3DGS.zip) |
| Compact3DGS+PP (3DGS.zip) | J. C. Lee, 2024 (variant) | Compact3DGS with post-processing quantization and entropy coding | 27.03 / .797 / .247 / 29.1 MB (3DGS.zip) |
| SOG w/o SH (3DGS.zip) | W. Morgenstern, 2024 (variant) | Self-Organizing Gaussians trained without higher-degree SH | 26.56 / .791 / .241 / 16.7 MB (3DGS.zip) |
| MesonGS c1, MesonGS c3 (3DGS.zip) | S. Xie, 2024 (variants) | Two rate configurations of MesonGS | c3: 26.99 / .797 / .246 / 25.9 MB. c1: 26.99 / .796 / .247 / 28.5 MB (3DGS.zip) |
| Trimming the Fat (Ali) | M. Salman Ali, 2024 | Gradient-based plus opacity-based pruning removing 75 % of Gaussians, high FPS | 27.13 / 0.798 / 0.248 / 20.1 MB / 37× (Ali) |
| ELMGS, ELMGS-medium, ELMGS-small (Ali) | M. S. Ali, 2024 (arXiv 2410.23213) | Pruning plus learned step-size uniform scalar quantization plus entropy coding, reported above 400 FPS on Deep Blending and 600 FPS on Tanks&Temples | medium: 27.31 / 0.792 / 0.264 / 38.6 MB / 19×. small: 27.00 / 0.779 / 0.286 / 25.8 MB / 28× (Ali) |
| EfficientGS, Efficientgs (Ali) | W. Liu, 2024 (arXiv 2404.12777) | Cumulative gradient analysis to halt unnecessary densification plus pruning, aimed at large-scale high-resolution scenes | 27.38 / 0.817 / 0.216 / 98.0 MB / 8× (Ali) |
| SafeguardGS (Ali) | Y. Lee, 2024 (arXiv 2405.17793) | Pruning score function that avoids catastrophic scene destruction | not tabulated |
| LP-3DGS (Ali) | Z. Zhang, 2024 (arXiv 2405.18784) | Trainable binary mask with Gumbel-Sigmoid that learns the pruning ratio automatically | not tabulated |
| RTGS (Ali) | W. Lin, 2024 (arXiv 2407.00435) | Efficiency-guided pruning plus foveated rendering by pixel eccentricity for real-time GS on mobile devices | not tabulated |
| GoDe (Ali) | F. Di Sario, 2025 (arXiv 2501.13558) | Gaussians on demand, progressive level of detail and scalable compression | not tabulated |
| GDGS (Ali) | Y. Gong, 2024 (arXiv 2405.05446) | Gradient domain Gaussian splatting for a sparse representation | not tabulated |
| SUNDAE (3DGS.zip text only, Ali tables) | R. Yang, 2024 (arXiv 2405.00676) | Graph-based spectral pruning with a band-limited graph filter plus a CNN neural compensation head (feature splatting) | SUNDAE (30%): 27.24 / 0.826 / 0.228 / 279.0 MB / 3×. SUNDAE (1%): 24.70 / 0.716 / 0.375 / 38.0 MB / 19× (Ali) |
| Gaussian-Forest, GF Large, GF Small (Ali) | F. Zhang, 2024 (arXiv 2406.08759) | Tree-structured hierarchical hybrid representation, explicit attributes at leaves and implicit shared attributes at higher levels, adaptive growth and pruning | Large: 27.45 / 0.803 / 0.212 / 85.0 MB / 9×. Small: 27.33 / 0.797 / 0.219 / 50.0 MB / 15× (Ali) |
| F-3DGS (Ali) | X. Sun, 2024 (arXiv 2405.17083) | Matrix and tensor factorization of coordinates and attributes into a few basis components per axis | not tabulated |
| Radsplat, RadSplat (Ali) | M. Niemeyer, 2024 (arXiv 2403.13806) | Radiance-field-informed Gaussian splatting for robust real-time rendering at 900+ FPS, listed under "Factorization Approach" | not tabulated |
| Kim et al. [30] (Ali) | S. Kim, 2024 (CVPR Workshops) | Same paper as Color-cued GS, used in Ali as a pruning method with SH gradients | 27.07 / 0.797 / 0.249 / 73.0 MB / 10× (Ali) |
| IGS (3DGS.zip, Figures 16 and 17 legend only) | not given | Appears only as a legend entry in the SSIM and LPIPS versus size plots, no table row, no description | none |

### 5.2 Compaction methods (3DGS.zip Section 4.2 and Table 2)

| Name as printed | First author, year (as printed) | What it does (one line) | Mip-NeRF 360 (as reported, 3DGS.zip) |
|---|---|---|---|
| Octree-GS | K. Ren, 2024 (arXiv 2403.17898) | Anchor Gaussians organized in an octree with levels of detail, LOD selected from the viewing distance | 28.05 / .819 / .217 / 657 k Gaussians |
| AtomGS | R. Liu, 2024 (arXiv 2405.12369) | Isotropic uniformly sized "Atom Gaussians" for detail, merged large Gaussians for smooth areas, edge-aware normal loss and multi-scale SSIM | 27.38 / .816 / .211 / 3,140 k |
| GaussianPro | K. Cheng, 2024 (ICML 2024) | Progressive propagation of depth and normal maps by patch matching to place new Gaussians, planar loss | 27.43 / .813 / .219 / 3,403 k |
| Color-cued GS | S. Kim, 2024 (CVPR Workshops) | Uses the 0th-order SH coefficient gradient as a color cue for densification | 27.07 / .797 / .249 / 646 k |
| Mini-Splatting-D | G. Fang, 2024 (variant) | Densification-only variant of Mini-Splatting | 27.51 / .831 / .176 / 4,690 k |
| Taming3DGS (Big) | Mallick and Goel, 2024 (variant) | High-budget setting of Taming 3DGS | 27.79 / .822 / .205 / 3,310 k (Taming3DGS: 27.29 / .799 / .253 / 630 k) |
| FreGS | J. Zhang, 2024 (CVPR 2024) | Progressive frequency regularization of rendered images against over-reconstruction | not tabulated |
| Pixel-GS | Z. Zhang, 2024 (ECCV 2024) | Pixel-aware gradient for density control | not tabulated |
| Revising Densification in Gaussian Splatting (RDGS) | S. Rota Bulò, 2024 (ECCV 2024) | Per-Gaussian error-based (SSIM-derived) criterion for splitting instead of positional gradient | not tabulated |
| MVG-Splatting | Z. Li, 2024 (arXiv 2407.11840) | Multi-view guided densification with quantile-based geometric consistency in near and far regions | not tabulated |

### 5.3 Other named works that are not 3DGS compression methods

Cited as background or tools, listed so nothing named is dropped: NeRF, Instant-NGP, Zip-NeRF, PlenOctrees, K-Planes, EG3D tri-planes, StyleGAN, EWA splatting, GaussianShader (Y. Jiang, 2024, cited in 3DGS.zip p8 as shading instead of SH), RAHT (de Queiroz and Chou, 2016), Hilbert-curve point cloud coding, 3DGS surveys by Fei et al., Wu et al., Chen and Wang, Bao et al., Block-NeRF, HPC, Boost Your NeRF, "Compressing explicit voxel grid representations", Depth-supervised NeRF, TensoRF, FastNeRF, NGLOD, ChARM and Minnen et al. context models, Learned Step Size Quantization, point cloud compression by He et al. 2022 and Song et al. 2023, VR-GS, SplatLoc, DrivingGaussian, GS-SLAM family, avatar and endoscopy applications.

---

## 6. Cross-survey discrepancies for the same method (both surveys say numbers come from the papers)

| Method | Dataset | 3DGS.zip (author-supplied, MB = 10⁶ B) | Ali et al. (copied from paper) |
|---|---|---|---|
| HAC-lowrate | Mip-NeRF 360 | 27.53 / .807 / .238 / 16.0 | 27.53 / 0.807 / 0.238 / 15.3 |
| HAC-lowrate | T&T | 24.04 / .846 / .187 / 8.5 | 24.04 / 0.846 / 0.187 / 8.1 |
| HAC-lowrate | Deep Blending | 29.98 / .902 / .269 / 4.6 | 29.98 / 0.902 / 0.269 / 4.4 |
| HAC-highrate | Mip-NeRF 360 | 27.77 / .811 / .230 / 22.9 | 27.77 / 0.811 / 0.230 / 21.9 |
| HAC-highrate | T&T | 24.40 / .853 / .177 / 11.8 | 24.40 / 0.853 / 0.177 / 11.2 |
| HAC-highrate | Deep Blending | 30.34 / .906 / .258 / 6.7 | 30.34 / 0.906 / 0.258 / 6.4 |
| ContextGS lowrate | Mip-NeRF 360 | 27.62 / .808 / .237 / 13.3 | 27.62 / 0.808 / 0.237 / 12.7 |
| ContextGS lowrate | T&T | 24.12 / .849 / .186 / 9.9 | 24.20 / 0.852 / 0.184 / 7.1 |
| ContextGS lowrate | Deep Blending | 30.09 / .907 / .265 / 3.7 | 30.11 / 0.907 / 0.265 / 3.6 |
| ContextGS highrate | Mip-NeRF 360 | 27.75 / .811 / .231 / 19.3 | 27.75 / 0.811 / 0.231 / 18.4 |
| ContextGS highrate | T&T | 24.29 / .855 / .176 / 12.4 | 24.29 / 0.855 / 0.176 / 11.8 |
| ContextGS highrate | Deep Blending | 30.41 / .909 / .259 / 6.9 | 30.39 / 0.909 / 0.258 / 6.6 |
| CompGS (Liu) | Mip-NeRF 360 | 27.26 / .803 / .239 / 17.3 | highrate 27.26 / 0.800 / 0.240 / 16.5 |
| Scaffold-GS | Mip-NeRF 360 | 27.50 / .806 / .252 / 156.0 | 28.84 / 0.848 / 0.220 / 102.0 |
| Scaffold-GS | T&T | 23.96 / .853 / .177 / 87.0 | 23.96 / 0.853 / 0.177 / 87.0 |
| Scaffold-GS | Deep Blending | 30.21 / .906 / .254 / 66.0 | 30.21 / 0.906 / 0.254 / 66.0 |

The HAC and ContextGS size ratios (1.045 to 1.049) match a MiB to MB conversion. The ContextGS lowrate Tanks and Temples row (9.9 versus 7.1 MB, 24.12 versus 24.20 dB) and the Scaffold-GS Mip-NeRF 360 row (27.50 versus 28.84 dB) are not explained by units, so those are different runs or a transcription difference. This is the concrete evidence for the "comparability across papers" open problem.

---

## 7. Scoreboard extracts for the project (Mip-NeRF 360)

All rate points from both surveys pooled, using the 3DGS.zip value where both surveys list a method (3DGS.zip normalises the unit), Ali et al. otherwise. "Under 20 MB" is strict.

**Top 8 by PSNR at under 20 MB**

| Rank | Method (as printed) | PSNR | Size MB | Source |
|---|---|---|---|---|
| 1 | ContextGS_highrate | 27.75 | 19.3 | 3DGS.zip T1 (Ali: 18.4) |
| 2 | ContextGS_lowrate | 27.62 | 13.3 | 3DGS.zip T1 (Ali: 12.7) |
| 3 | HAC-lowrate | 27.53 | 16.0 | 3DGS.zip T1 (Ali: 15.3) |
| 4 | CodecGS | 27.30 | 10.3 | 3DGS.zip T1 |
| 5 | gsplat-1.00M | 27.29 | 16.0 | 3DGS.zip T1 |
| 6 | CompGS (Liu) | 27.26 | 17.3 | 3DGS.zip T1 (Ali highrate: 16.5) |
| 7 | Compact3D 32K (= Ali CompGS-32K) | 27.12 | 19.0 | both |
| 8 | Compact3D 16K (= Ali CompGS-16K) | 27.03 | 18.0 | both |

Just outside: HAC-highrate 27.77 dB at 22.9 MB (3DGS.zip) or 21.9 MB (Ali). Trimming the Fat 27.13 dB at 20.1 MB (Ali).

**Under 10 MB**

Only one Mip-NeRF 360 rate point under 10 MB exists in either survey: CompGS (lowrate) [40] (Liu), 26.37 dB at 8.8 MB (Ali Table IV). The nearest points above the threshold are CodecGS 27.30 dB at 10.3 MB and ContextGS_lowrate 27.62 dB at 13.3 MB (12.7 in Ali). The surveys tabulate no other Mip-NeRF 360 entry below 10 MB. Sub-10 MB entries do exist for the other datasets (Tanks and Temples: CodecGS 7.8, HAC-lowrate 8.5, SOG w/o SH 9.3, ContextGS_lowrate 9.9, Trimming the Fat 8.6 (Ali), CompGS lowrate 5.9 (Ali), ContextGS lowrate 7.1 (Ali), HAC-lowrate 8.1 (Ali), CompGS highrate 9.6 (Ali). Deep Blending: ContextGS_lowrate 3.7, HAC-lowrate 4.6, SOG w/o SH 5.7, HAC-highrate 6.7, ContextGS_highrate 6.9, CodecGS 9.0, CompGS 9.2, plus the Ali values 3.6, 4.4, 6.0, 6.4, 6.6, 8.8).

**What neither survey contains**: a method that takes a byte target as input, a column for decode time, training time or peak VRAM, and any FPS number in a table (Ali has FPS only as figure points measured on an A40).
