# I. Code availability of the reference papers

Checked 2026-09-08. Every repository below was confirmed through the GitHub REST API (`gh api repos/<owner>/<repo>`,
top-level `contents`, `.gitmodules`, the licence file head, README). URL discovery: `github.com` strings grepped from
the PDF text with pymupdf, then the project page or arXiv abstract page (WebFetch) and GitHub repository search for
the papers that print no URL. "Inria" below means the custom "Gaussian-Splatting License" of Inria and MPII
(`LICENSE.md` of graphdeco-inria/gaussian-splatting): non-commercial research and evaluation use only, no
redistribution for commercial purposes. GitHub reports it as `NOASSERTION`.

Column key: **Train** = training or fine-tuning code shipped. **Coder / size** = entropy coder shipped, and whether
a size estimator or size search exists (compression papers only). **Rasterizer** = how the CUDA rasterizer is
obtained (git submodule, vendored copy, zip archive, pip package) and whether it is a modified kernel.

## Papers

| # | Paper | Repository | Exists | Licence | Last push | Stars | Lang | Base framework | Train | Coder / size | Rasterizer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3DGS, Kerbl, SIGGRAPH 2023 | https://github.com/graphdeco-inria/gaussian-splatting | yes | Inria (origin of the licence) | 2025-10-17 | 23760 | Python | original graphdeco | yes | n/a | git submodules: diff-gaussian-rasterization (github), simple-knn and SIBR (gitlab.inria.fr), fused-ssim |
| 2 | 2DGS, Huang, SIGGRAPH 2024 | https://github.com/hbb1/2d-gaussian-splatting | yes | Inria (inherited) | 2026-08-25 | 3287 | Python | original graphdeco | yes | n/a | custom: git submodule hbb1/diff-surfel-rasterization, simple-knn |
| 3 | Mip-Splatting, Yu, CVPR 2024 | https://github.com/autonomousvision/mip-splatting | yes | Inria (inherited) | 2024-12-17 | 1470 | Python | original graphdeco | yes | n/a | custom (3D/2D filter): vendored copy under `submodules/`, no `.gitmodules` |
| 4 | Scaffold-GS, Lu, CVPR 2024 | https://github.com/city-super/Scaffold-GS | yes | Inria (inherited) | 2024-09-26 | 1279 | C++ (SIBR) | original graphdeco, anchor model | yes | n/a | vendored copy of diff-gaussian-rasterization and simple-knn, no `.gitmodules` |
| 5 | 3DGS-MCMC, Kheradmand, NeurIPS 2024 | https://github.com/ubc-vision/3dgs-mcmc | yes | Inria (inherited, GitHub fork of graphdeco) | 2025-01-02 | 680 | Python | original graphdeco | yes | n/a | custom: git submodule shakibakh/diff-gaussian-rasterization, simple-knn, SIBR |
| 6 | SizeGS, Xie, ACM MM 2025 | https://github.com/mmlab-sigs/SizeGS | yes | no licence file (built on Scaffold-GS and MesonGS code, so Inria terms apply de facto) | 2025-10-28 | 16 | C++ (SIBR) | Scaffold-GS checkpoints, MesonGS RAHT code | post-training on pretrained Scaffold-GS (`sizegs.sh`, `meson.py`), no reconstruction training | yes: torchac (`size_ac_*.sh`) or zip/LZ77 (`size_*.sh`). Size estimator: yes, the MIP bit-width and reserve-ratio solver (pulp, LPMP/BDD) is the paper's core | vendored diff-gaussian-rasterization and simple-knn, plus custom CUDA modules `octree`, `quant`, `seg-quant`. Only `submodules/BDD` is a real git submodule |
| 7 | GETA-3DGS, Zhang, 2026 | none found | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 8 | KISS-GS, Morgenstern, ECCV 2026 | https://github.com/fraunhoferhhi/KISS-GS (project page only) and https://github.com/w-m/ffsplat (code link in that README) | yes, partially | KISS-GS page repo: no licence. ffsplat: Apache-2.0 | KISS-GS page 2026-09-03, ffsplat 2025-08-04 (last commit 2025-07-10) | 6 and 44 | ffsplat: Python | gsplat (`gsplat==1.4.0` pip dependency) | no: ffsplat CLI is `convert`, `eval`, `live`, `view`. Reconstruction is done externally with gsplat MCMC. The paper's encoding-aware fine-tuning is not in the public code | image-format encoder (SOG/PlayCanvas WebP, PLAS sorting), no arithmetic coder. SOG-XT from the paper is absent (code search: 0 hits for `sogxt`, `finetune`, `jxl`). No size estimator | none: gsplat via pip, no `.gitmodules` entries |
| 9 | FlexGaussian, Tian, ACM MM 2025 | https://github.com/Supercomputing-System-AI-Lab/FlexGaussian | yes | Inria (inherited) | 2025-07-21 | 11 | Python | original graphdeco plus LightGaussian importance score | no, training-free (`compress.py`, `prune.py`, no fine-tune) | no entropy coder (`np.savez` npz). Search: yes, `Searcher` enumerates pruning and SH ratios against a target PSNR drop (`--quality_target_diff`) and reports output bytes, so it is quality-targeted, not byte-targeted | custom: vendored `submodules/LightGS-rasterizer` (LightGaussian's compress-diff-gaussian-rasterization), simple-knn, no `.gitmodules` |
| 10 | RDO-Gaussian, Wang, ECCV 2024 | https://github.com/USTC-IMCL/RDO-Gaussian | yes | Inria (inherited) | 2024-09-25 | 29 | Python | original graphdeco | yes, end-to-end, 6 rate points per run script | yes: vendored `submodules/imcl-compression` (C++ entropy coder extensions) plus `vq/` (ECVQ, k-means). Rate estimated in training, no size target | vendored diff-gaussian-rasterization and simple-knn, no `.gitmodules` |
| 11 | HAC, Chen, ECCV 2024 | https://github.com/YihangChen-ee/HAC | yes | Inria (inherited) | 2025-11-09 | 336 | Python | Scaffold-GS | yes (train, encode, decode, test in one run) | yes: own CUDA arithmetic codec (`submodules/arithmetic.zip`, torchac-derived). Rate from the entropy model, controlled by `lmbda`, no size target | zip archives to unzip: diff-gaussian-rasterization.zip, simple-knn.zip, gridencoder.zip. No `.gitmodules` |
| 12 | HAC++, Chen, TPAMI 2025 | https://github.com/YihangChen-ee/HAC-plus | yes | Inria (inherited) | 2025-11-13 | 212 | Python | Scaffold-GS, HAC | yes | yes: arithmetic.zip CUDA codec. `lmbda` controls rate | zip archives as HAC |
| 13 | ContextGS, Wang, NeurIPS 2024 | https://github.com/wyf0912/ContextGS | yes | Inria (inherited) | 2025-01-09 | 93 | Python | Scaffold-GS, HAC | yes | yes: compressai entropy models plus torchac (`utils/entropy_models.py`). `lmbda` controls rate | none shipped: README says `unzip diff-gaussian-rasterization.zip`, but no `submodules/` or zip is in the repository. Take the rasterizer from HAC or Scaffold-GS |
| 14 | HEMGS, Liu, 2024 | none | no | n/a | n/a | n/a | n/a | Scaffold-GS (paper) | n/a | n/a | n/a |
| 15 | CAT-3DGS, Zhan, ICLR 2025 | https://github.com/NYCU-MAPL/CAT-3DGS | yes | Inria (inherited) | 2025-04-04 | 19 | Python | Scaffold-GS, HAC | yes | yes: torchac (`utils/entropy_models.py`, `utils/gc/`) | zip archives: diff-gaussian-rasterization.zip, simple-knn.zip |
| 16 | PCGS, Chen, AAAI 2026 | https://github.com/YihangChen-ee/PCGS | yes | Inria (inherited) | 2026-01-23 | 47 | Python | Scaffold-GS, HAC++ | yes, several rate points in one run | yes: arithmetic.zip CUDA codec, progressive bitstreams | zip archives as HAC |
| 17 | SALVQ, Xu, TIP 2025 | https://github.com/hxu160/SALVQ | yes | no licence file, README says follow the 3DGS licence (Inria) | 2025-10-02 | 6 | Python | HAC (Scaffold-GS) | yes (`train.py`, `train_script/run_shell_*.py`) | not shipped: relies on the HAC installation and codec. Lattice scaling gives multiple rates from one model | none shipped: install per HAC instructions |
| 18 | FCGS, Chen, ICLR 2025 | https://github.com/YihangChen-ee/FCGS | yes | custom: Apache-2.0 text plus "commercial use prohibited without written permission" | 2025-11-09 | 246 | Python | original graphdeco `.ply` input, feed-forward | no training script, encode and decode with released checkpoints | yes: CUDA arithmetic codec (`submodules/arithmetic`, torchac-based), external `tmc3` (G-PCC) for positions | git submodule graphdeco diff-gaussian-rasterization, vendored simple-knn, gridencoder, freqencoder, gridcreater |
| 19 | CompGS, Liu, 2024 | https://github.com/LiuXiangrui/CompGS | yes | GPL-3.0 | 2024-11-06 | 49 | Python | Scaffold-GS ideas, own rewrite (`Modules/TrainerCompGS.py`) | yes | yes: compressai entropy models and range coder, external G-PCC `tmc3` for anchors, bitstream size reported in bytes | vendored `submodules/diff-gaussian-rasterization` and `knn_dist`, no `.gitmodules` |
| 20 | MesonGS, Xie, ECCV 2024 | https://github.com/ShuzhaoXie/MesonGS | yes | Inria (inherited) | 2024-10-24 | 28 | Python | original graphdeco | post-training, optional fine-tune (`--iteration 0` disables it) | RAHT plus VQ plus zip (`bins.zip`), no arithmetic coder. Block-count config, no size target | vendored diff-gaussian-rasterization, simple-knn, `weighted_distance` CUDA |
| 21 | Taming 3DGS, Mallick, SIGGRAPH Asia 2024 | https://github.com/humansensinglab/taming-3dgs (paper URL nullptr81/3dgs-accel redirects here) | yes | MIT for the modifications, `LICENSE_ORIGINAL.md` Inria for the base | 2025-06-16 | 358 | Python | original graphdeco | yes, `--budget` and `--mode` give a Gaussian-count target | n/a | custom accelerated kernels vendored under `submodules/diff-gaussian-rasterization`, simple-knn, fused-ssim as git submodule |
| 22 | Mini-Splatting, Fang, ECCV 2024 | https://github.com/fatPeter/mini-splatting | yes | no licence file (built on Inria code) | 2024-10-12 | 227 | Python | original graphdeco | yes (`ms/`, `ms_d/`) | `ms_c/` Mini-Splatting-C: Haar3D transform, no arithmetic coder | vendored diff-gaussian-rasterization plus custom `diff-gaussian-rasterization_ms`, simple-knn |
| 23 | GaussianSpa, Zhang, CVPR 2025 | https://github.com/noodle-lab/GaussianSpa | yes | MIT (`LICENSE`) next to Inria (`LICENSE.md`), both present | 2025-04-05 | 104 | Python | original graphdeco plus Mini-Splatting rasterizer | yes (`train_op.py`, `train_opacity.py`, `train_imp_score.py`) | n/a | vendored diff-gaussian-rasterization, custom `diff-gaussian-rasterization_ms`, simple-knn |
| 24 | MaskGaussian, Liu, CVPR 2025 | https://github.com/kaikai23/MaskGaussian | yes | Inria (inherited) | 2026-05-12 | 101 | Python | original graphdeco | yes, plus post-training `prune_finetune.py` | n/a | git submodules: graphdeco diff-gaussian-rasterization, custom kaikai23/mask-diff-gaussian-rasterization (registered with an ssh `git@github.com:` URL, rewrite before cloning), simple-knn, SIBR |
| 25 | SPARE-GS, Chen, 2026 | https://github.com/ZhangChen2022/SPARE-GS-code (linked from the project page) | yes | no licence file (built on Inria code) | 2026-07-15 | 2 | C++ (SIBR) | original graphdeco | yes (`train_3dgs_ours.py`, `run_3dgs+SPAREGS.py`) | n/a | git submodules: graphdeco diff-gaussian-rasterization, simple-knn, fused-ssim, SIBR |
| 26 | Smart target point control, Bisht, 2026 | none found | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 27 | PUP 3D-GS, Hanson, CVPR 2025 | https://github.com/j-alex-hanson/gaussian-splatting-pup | yes | Inria (inherited) | 2025-11-22 | 153 | Python | original graphdeco | post-training prune and fine-tune (`prune.py`, `prune_finetune.py`), `train.py` present | n/a | git submodules: graphdeco diff-gaussian-rasterization, custom `rasterization_and_pup_fisher`, `tuallen/compress-diff-gaussian-rasterization`, simple-knn, SIBR |
| 28 | LightGaussian, Fan, NeurIPS 2024 | https://github.com/VITA-Group/LightGaussian | yes | Inria (inherited) | 2024-12-30 | 819 | Python | original graphdeco | yes: prune and fine-tune, distillation, and train-from-scratch variants | VQ (`vectree/`) plus `np.savez_compressed`, no arithmetic coder | custom: git submodule Kevin-2017/compress-diff-gaussian-rasterization, simple-knn |
| 29 | Compact-3DGS, Lee, CVPR 2024 | https://github.com/maincold2/Compact-3DGS | yes | Inria (inherited) | 2024-09-27 | 500 | Python | original graphdeco plus tiny-cuda-nn hash grid | yes | post-processing: 8-bit quantization, Huffman on opacity and hash grid, Morton sort, npz. A `Storage` estimate file, no size target | git submodules: graphdeco diff-gaussian-rasterization, camenduru/simple-knn |
| 30 | Compressed 3DGS, Niedermayr, CVPR 2024 | https://github.com/KeKsBoTer/c3dgs | yes | Inria (inherited) | 2025-09-17 | 403 | Python | original graphdeco | post-training: `compress.py` plus `finetune.py`, no `train.py` | sensitivity-aware VQ and quantization-aware fine-tuning, npz (deflate), no arithmetic coder | vendored diff-gaussian-rasterization (only its glm is a git submodule), `weighted_distance` CUDA |
| 31 | Self-Organizing Gaussians, Morgenstern, ECCV 2024 | https://github.com/fraunhoferhhi/Self-Organizing-Gaussians | yes | Inria (inherited, fork of graphdeco) | 2025-01-31 | 395 | Python | original graphdeco | yes | image codecs: JPEG XL (imagecodecs 2023.9.18 pinned), PNG, EXR, npz (`compression/codec.py`), no arithmetic coder | git submodules: graphdeco diff-gaussian-rasterization, simple-knn, fraunhoferhhi/PLAS, SIBR |
| 32 | 3DGS.zip survey, Bagdasarian, CGF 2025 | https://github.com/w-m/3dgs-compression-survey | yes | MIT | 2026-07-14 | 242 | JavaScript | leaderboard, no framework | no | no, result CSVs and plots only | none |
| 33 | Compression in 3DGS survey, Ali, 2025 | none | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 34 | Scale hyperprior, Ballé, ICLR 2018 | https://github.com/tensorflow/compression | yes | Apache-2.0 | 2026-04-17 | 923 | Python | TensorFlow | yes (`models/bmshj2018.py`) | yes: range coder and entropy model layers | none |
| 35 | Variable-rate conditional AE, Choi, ICCV 2019 | none official | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 36 | AG-VAE gained compression, Cui, CVPR 2021 | none official. Unofficial: https://github.com/mmSir/GainedVAE | no (official), yes (unofficial) | unofficial repo has no licence file | 2023-04-06 | 78 | Python | CompressAI | yes | yes (CompressAI entropy bottleneck), asymmetric Gaussian model not implemented | none |

## Non-paper repositories

| Repository | Exists | Licence | Last push | Stars | Lang | Notes |
|---|---|---|---|---|---|---|
| https://github.com/nerfstudio-project/gsplat | yes | Apache-2.0 | 2026-09-03 | 5653 | Python | pip package, own CUDA kernels (glm and googletest as git submodules). Training: `examples/simple_trainer.py` with MCMC densification. Compression: `gsplat/compression/png_compression.py` (PLAS sort plus PNG, from Self-Organizing Gaussians), no arithmetic coder |
| https://github.com/graphdeco-inria/gaussian-splatting | yes | Inria | 2025-10-17 | 23760 | Python | see row 1 |
| https://github.com/graphdeco-inria/diff-gaussian-rasterization | yes | Inria, same `LICENSE.md` text as the trainer | 2024-10-21 | 1492 | Cuda | glm as git submodule. Every repository above that vendors, zips or submodules this rasterizer inherits the Inria terms for that component |
| https://github.com/w-m/3dgs-compression-survey | yes | MIT | 2026-07-14 | 242 | JavaScript | leaderboard, see row 32. Issue #16 documents that HEMGS results came by e-mail with no repository |

## Notes

- **Inria licence inheritance.** The `LICENSE.md` of rows 1 to 5, 9 to 13, 15, 16, 20, 23, 24, 27 to 31 and of
  diff-gaussian-rasterization is the Inria and MPII Gaussian-Splatting licence (GitHub `NOASSERTION`). Row 21
  keeps it as `LICENSE_ORIGINAL.md` beside an MIT file for the modifications. Rows 6, 17, 22 and 25 ship no licence
  file at all but are built on that code, so the same terms hold in practice. Only gsplat, ffsplat, CompGS (GPL-3.0),
  tensorflow/compression, the survey repo and the MIT half of GaussianSpa and Taming carry an OSI licence, and FCGS
  adds a non-commercial clause on top of Apache-2.0.
- **Submodule usability.** Clean for a git submodule or pip dependency: gsplat, ffsplat, tensorflow/compression,
  CompGS, graphdeco, SPARE-GS-code, Self-Organizing-Gaussians, Compact-3DGS, LightGaussian, PUP 3D-GS, 2DGS,
  3DGS-MCMC. Awkward: HAC, HAC++, PCGS, CAT-3DGS ship the CUDA extensions as zip archives that a build step must
  unpack. MaskGaussian registers its custom rasterizer with an ssh URL. ContextGS and SALVQ ship no rasterizer and
  expect the HAC or Scaffold-GS install. SizeGS mixes one real submodule (BDD) with vendored CUDA modules.
- **Byte-budget relevance.** Only SizeGS contains an explicit size-to-hyperparameter solver. FlexGaussian searches a
  discrete configuration grid against a PSNR-drop target and only reports the resulting bytes. HAC, HAC++, PCGS,
  ContextGS, CAT-3DGS, CompGS, RDO-Gaussian and SALVQ control rate through a Lagrangian `lmbda` and have no size
  target. Taming 3DGS and 3DGS-MCMC (`--cap_max`) take a Gaussian-count target, not bytes.
- **KISS-GS.** `fraunhoferhhi/KISS-GS` holds the built project page only ("page source is maintained internally").
  Its README links `w-m/ffsplat` as the code, but that repository has not moved since 2025-07-10 and has none of the
  SOG-XT encoder, the POPSpa compaction or the encoding-aware fine-tuning from the paper. Treat the KISS-GS method as
  not yet released.
- **CAT-3DGS.** The SALVQ paper states that the public CAT-3DGS release has implementation issues that prevented a
  reliable evaluation, so budget time for debugging if it is a baseline.
- **AG-VAE and Choi.** Both rate-control papers come from industrial labs (Huawei, Samsung) without official code.
  `mmSir/GainedVAE` is a CompressAI reimplementation that states it is not official and omits the asymmetric
  Gaussian entropy model.

## Papers with no code found

- GETA-3DGS, Zhang, 2026 (arXiv 2605.02086): no URL in the PDF, none on arXiv, no repository on GitHub search.
- HEMGS, Liu, 2024 (arXiv 2411.18473): confirmed unreleased by the survey maintainers (issue #16) and by the SALVQ paper.
- Smart target point control, Bisht, 2026 (arXiv 2605.16158): no URL, no arXiv link, no repository found.
- Compression in 3DGS survey, Ali, 2025 (arXiv 2502.19457): survey without a companion repository.
- Variable-rate conditional autoencoder, Choi, ICCV 2019: no official release (Samsung), unofficial ports only.
- AG-VAE gained compression, Cui, CVPR 2021: no official release (Huawei), unofficial `mmSir/GainedVAE`.
- KISS-GS, Morgenstern, ECCV 2026: project page exists, the method's code is not in the linked public repository.
