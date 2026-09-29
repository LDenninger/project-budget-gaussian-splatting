# references/results

Baseline evaluation numbers transcribed from the PDFs in `references/`, one CSV per source paper
plus a consolidated `all_results.csv`.

Every number here is **reported**, not measured in this repository. Nothing in this directory has
been re-run on a GPU. A row records what a specific table on a specific page of a specific paper
printed, so a value can always be traced back and checked.

## Table of Contents

- [What is covered](#what-is-covered)
- [Layout](#layout)
- [Schema](#schema)
- [How the numbers were extracted](#how-the-numbers-were-extracted)
- [Regenerating and validating](#regenerating-and-validating)
- [Reading the data without being misled](#reading-the-data-without-being-misled)
- [Known conflicts between papers](#known-conflicts-between-papers)
- [What was deliberately left out](#what-was-deliberately-left-out)

## What is covered

31 papers, 3 787 rows, 409 distinct method labels, across Mip-NeRF 360, Tanks and Temples, Deep
Blending, NeRF Synthetic, BungeeNeRF, OMMO, N3DV and DL3DV-GS.

| directory | papers | what the rows are |
|---|---|---|
| `00_foundations/` | 3 | 3DGS, Scaffold-GS and 3DGS-MCMC, the uncompressed anchors every other table compares against |
| `01_size_targeted/` | 4 | SizeGS, GETA-3DGS, FlexGaussian, KISS-GS — the methods that take a size in MB as input |
| `02_rate_distortion/` | 11 | the HAC line, ContextGS, HEMGS, CAT-3DGS, PCGS, SALVQ, FCGS, CompGS, MesonGS, RDO-Gaussian |
| `03_count_budget/` | 7 | Taming 3DGS, Mini-Splatting, GaussianSpa, MaskGaussian, PUP 3D-GS, SPARE-GS, Target Point Control |
| `04_compaction_pruning/` | 4 | LightGaussian, Compact 3DGS, Self-Organizing Gaussians, Compressed3D |
| `05_surveys/` | 2 | the 3DGS.zip leaderboard and the Ali et al. compression survey |
| `special/` | 1 | a table whose shape does not fit the common schema, see below |

1 890 rows are per-scene rather than dataset averages. 2 698 rows carry a stored size, 464 carry a
primitive count instead. Papers that budget by primitive count (`03_count_budget/`, most of
`00_foundations/`) report no MB column at all, which is the gap this project exists to close.

## Layout

```
references/results
├── 00_foundations/             # one CSV per paper, named after its manifest slug
├── 01_size_targeted/
├── 02_rate_distortion/
├── 03_count_budget/
├── 04_compaction_pruning/
├── 05_surveys/
├── special/                    # tables with their own shape, excluded from all_results.csv
├── all_results.csv             # every per-paper CSV concatenated, generated
├── build_all_results.py        # generates and validates all_results.csv
├── SCHEMA.csv                  # column dictionary
└── README.md
```

## Schema

`SCHEMA.csv` is the machine-readable column dictionary. The columns, in order:

    source_paper         | slug matching references/manifest.txt
    source_arxiv         | arXiv id of the PDF
    source_table         | table label as printed
    source_page          | 1-based PDF page, as a viewer shows it
    table_kind           | main | ablation | supplement
    method               | method name as printed in the table
    base_model           | 3DGS | Scaffold-GS | 3DGS-MCMC | 2DGS | 4DGS | NeRF | ...
    dataset              | Mip-NeRF 360 | Tanks and Temples | Deep Blending | NeRF Synthetic |
                         |   BungeeNeRF | OMMO | N3DV | DL3DV-GS | Other
    scene                | scene name, or `average` for a dataset-average row
    psnr_db, ssim, lpips | quality metrics
    size_mb              | stored size, as printed when the paper prints MB or MiB
    size_unit_printed    | MB | MiB | GB | KB, the unit the paper actually printed
    num_gaussians        | primitive count (anchors where the notes say so)
    fps, train_time_s, encode_time_s, decode_time_s, total_time_s
    notes                | everything needed to read the row correctly

Two conventions matter:

- `size_mb` holds the printed number unchanged when the paper prints MB or MiB, and a converted
  value when it prints GB or KB. `size_unit_printed` says which. MiB is 1.048576 MB, and papers in
  this field use the two interchangeably, so a byte-target controller must not silently mix them.
  107 rows are MiB (all FlexGaussian), 9 are converted from GB.
- `num_gaussians` is a Gaussian count unless the `notes` say it is an anchor count. HAC++ Tables
  IV-IX report anchors, and each anchor carries K = 10 Gaussians.

Empty means the paper did not print that value.

## How the numbers were extracted

`pymupdf` word positions, clustered into rows by baseline and sorted by x. That reconstructs the
column structure of a two-column paper reliably, where a plain text extraction interleaves table
rows with body text.

Regular per-scene tables (the λ sweeps in HAC, HAC++, CAT-3DGS, PCGS, SALVQ, SPARE-GS) were parsed
programmatically. Irregular comparison tables were read by hand from the reconstructed layout.

The parser was validated against a hand transcription: all 160 per-scene HAC rows (5 λ values ×
5 datasets) matched the parser exactly, field for field. Two further cross-checks agreed
independently — CAT-3DGS Table 15 reproduces HAC appendix Table H for Scaffold-GS, and Scaffold-GS
Tables 14-15 reproduce the 3DGS Table 2 row exactly.

Where the PDF text layer is damaged, the row says so in `notes` rather than being silently
repaired. Three cases:

- PCGS Table VII renders the Compressed3D row shifted one column left, with a stray `3.68` at its
  head and the Tanks and Temples size dropped. Recovered by column position and confirmed against
  HAC++ Table II.
- KISS-GS Tables 2 and 6 print only deltas against the full pipeline. The absolute quality columns
  are the full-pipeline values plus the printed delta, and the notes carry the raw delta.
- GaussianSpa Table 5 prints its two playroom rows with identical values. The two Deep Blending
  averages imply they differ by about 0.001 M primitives, so the duplication is plausible.

## Regenerating and validating

```bash
python references/results/build_all_results.py
```

Concatenates every per-paper CSV into `all_results.csv` and validates each row: known dataset,
known `table_kind`, numeric metric columns, `size_mb` and `size_unit_printed` present together,
and PSNR in [5, 45], SSIM and LPIPS in [0, 1]. It exits non-zero on any problem. Files under
`special/` are excluded, because they carry their own columns.

`special/kiss_gs_morgenstern_eccv2026_table3_reduction_factors.csv` holds KISS-GS Table 3, which
reports a file-size reduction factor at matched quality rather than a size and a metric, so it has
its own columns.

## Reading the data without being misled

**Do not compare PSNR across papers without checking the protocol.** Three things move it by more
than most reported improvements:

1. **Scene subset.** Scaffold-GS evaluates Mip-NeRF 360 on seven scenes, dropping `flowers` and
   `treehill`, which is why 3DGS reads 28.69 dB in its tables and 27.21 dB everywhere else.
   Target Point Control uses a different seven. MesonGS uses another subset again (3DGS at
   28.98 dB). Filter on `scene` and recompute the average yourself when in doubt.
2. **Training and evaluation resolution.** 3DGS.zip Table 3 measures this directly: on Mip-NeRF 360
   the same model scores 25.58 dB or 27.02 dB depending only on whether the evaluation resolution
   matches the training resolution. That is larger than the gap between most competing methods.
3. **Whose run it is.** Many rows are a baseline re-run by the authors of a competing method rather
   than the baseline's own published number. The `notes` say which.

**A size in MB and a primitive count are not interchangeable.** They are separate columns for that
reason. Rows with a count and no size come from methods whose budget is denominated in primitives.

**`table_kind`** separates the headline comparison (`main`) from ablations (`ablation`) and
appendix per-scene tables (`supplement`). Filtering to `main` gives the frontier; ablations sit at
operating points chosen to isolate a component, not to be competitive.

## Known conflicts between papers

Recorded here because they are load-bearing when picking a baseline, and each is flagged in the
`notes` of the affected row.

| conflict | detail |
|---|---|
| ContextGS Mip-NeRF 360 high rate | ContextGS Table 1 reports 27.75 dB at 18.41 MB, its own appendix Table 9 reports 27.72 dB at 21.58 MB. HEMGS quotes the appendix value, HAC++ and PCGS quote Table 1, 3DGS.zip lists 19.3 MB and the Ali et al. survey 18.4 MB. |
| Scaffold-GS on Mip-NeRF 360 | Three different pairs. 28.84 dB / 156 MB in the Scaffold-GS paper itself (7 scenes, its own Tables 1 and 2), 27.50 dB / 253.9 MB in HAC, HAC++, ContextGS, CAT-3DGS, HEMGS, PCGS and SALVQ (9 scenes, their own re-run), and 28.84 dB / 102 MB in the Ali et al. survey, which pairs the 7-scene PSNR with a size matching neither. The 3DGS.zip survey uses 27.50 dB / 156.0 MB, mixing the two sources. |
| 3DGS baseline size on Mip-NeRF 360 | 693 MB (Scaffold-GS), 734 MB (3DGS Table 1), 741.12 MB (FCGS), 744.7 MB (HAC), 746.46 MB (PUP 3D-GS), 750.9 MB (HAC++), 782 MB (LightGaussian re-run), 785 MB (SOG), 788.98 MB (CompGS), 795.26 MB (Compressed3D and FlexGaussian). Every one of these is a vanilla 30k 3DGS checkpoint on the same nine scenes at 27.2-27.6 dB, and they span 15 %. A byte-target controller cannot be evaluated against "the 3DGS size" without saying whose. |
| HAC++ reported against recomputed | KISS-GS Table 5 re-runs HAC++ and reaches 27.55 dB at 8.84 MB against the reported 27.60 dB at 8.34 MB, and shows quality falling at 40k and 44k iterations while size converges. |
| FlexGaussian's 3D-GS baseline | 795.26 MiB on Mip-NeRF 360, because it compresses large Grendel-GS models rather than standard 30k 3DGS checkpoints. Its compression ratios are not comparable to the rest. |

One finding worth carrying into the project: **GETA-3DGS's storage budget does not bind.** Its own
Fig. 2 and Fig. 6 disclose that all eight target sizes B in {3, 5, 8, 10, 12, 30, 60, 120} MB land
in 4.0–4.2 MB, because B enters as a sparsity penalty rather than a hard byte cap. Those rows are in
`01_size_targeted/geta_3dgs_zhang_2026.csv`, marked `Fig. 2 / Fig. 6`.

## What was deliberately left out

- **`06_rate_control_theory/`** — Ballé, Choi and Cui report bpp on Kodak, so no row of theirs is
  comparable to a 3DGS size-quality table.
- **BD-rate and BD-PSNR** are not columns. They are a property of a curve, not of a row, so where a
  table reports them they are recorded in `notes` (CAT-3DGS Table 1, HEMGS Table 2, SALVQ Tables I,
  II and XI, SPARE-GS Tables II and III).
- **Figures.** Rate-distortion curves were not digitised. The two exceptions are the GETA-3DGS
  budget sweep above and KISS-GS Figure 1, because the non-binding-budget finding appears only
  there.
- **The 31 references added to `references/manifest.txt` after this transcription began**
  (`hybridgs`, `mesongs_plusplus`, `rave`, `smol_gs`, `gsico`, `codecgs`, the `03_count_budget`
  additions, `eagles`, `compact3d`, and all of the new `07_runtime_cost` category). Their PDFs are
  on disk and their tables are not transcribed here.
