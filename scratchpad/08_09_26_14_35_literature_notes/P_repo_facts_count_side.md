# Facts for the count-side implementation plan

Five repositories, shallow-cloned into
`<session scratchpad>/repos/`.
Every path below is relative to a repository root unless stated otherwise. Every line number refers
to the commit recorded in the table.

| repo | commit | date | clone command used |
|---|---|---|---|
| gsplat | `28e794ca44a4c25ffc39175370c5ee7b38bfcc36` | 2026-09-03 | `git clone --depth 1 https://github.com/nerfstudio-project/gsplat` |
| ffsplat | `3fb6444c5faed6de9aa55fa2313df6c2a4104193` | 2025-07-10 | `git clone --depth 1 https://github.com/w-m/ffsplat` |
| taming-3dgs | `fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16` | 2025-06-16 | `git clone --depth 1 https://github.com/humansensinglab/taming-3dgs` |
| gaussian-splatting-pup | `e971ea4802908c69eab7e8601bb1eddd2e443754` | 2025-11-22 | `git clone --depth 1 https://github.com/j-alex-hanson/gaussian-splatting-pup` |
| GaussianSpa | `abdfaf6401834bbf9348079f7bd380f7514822a5` | 2025-04-04 | `git clone --depth 1 https://github.com/noodle-lab/GaussianSpa` |

None of the clones used `--recurse-submodules`, so submodule directories in
gaussian-splatting-pup are empty. taming-3dgs and GaussianSpa carry their rasterizers as checked-in
copies, which are present.

---

## 1. gsplat (nerfstudio-project/gsplat)

### a. Licence and Inria rasterizer

- Licence file name: `LICENSE`.
- First 5 lines verbatim (`LICENSE:1-5`):

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
```

- Inria `diff-gaussian-rasterization`: **not vendored**. `find` over the tree returns no path
  matching `*diff*gaussian*`. gsplat ships its own CUDA rasterizer under `gsplat/cuda/csrc/`.
- `.gitmodules` declares only two submodules:
  - `gsplat/cuda/csrc/third_party/glm` from `https://github.com/g-truc/glm.git`
  - `third_party/googletest` from `https://github.com/google/googletest.git`

### b. Installation

README commands, verbatim:

- `README.md:63` — `pip install gsplat`
  - `README.md:60` states this path builds the CUDA code **on the first run (JIT)**.
- `README.md:69` — `pip install git+https://github.com/nerfstudio-project/gsplat.git`
  - `README.md:66` states this path builds the CUDA code **during installation**.
- `README.md:74-75` (pre-compiled wheels):

```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

- Evaluation block, `README.md:85-91`:

```bash
python -m pip install -e .
cd examples
python -m pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

- Examples extras, `README.md:97` — `pip install -r examples/requirements.txt --no-build-isolation`

Versions named:

- `pyproject.toml:21` — build requires `torch>=2.7`.
- `setup.py:141` — install requires `torch>=2.7`.
- `setup.py:281` — `python_requires=">=3.7"`.
- `README.md:29` — "**PyTorch 2.7+ is now required**, with improved CUDA 12.8 and 13.2 build compatibility."
- `examples/requirements.txt:4` — `torch==2.9.1`; `examples/requirements.txt:12` — `torchvision==0.24.1`;
  `examples/requirements.txt:20` — `torchmetrics==1.8.2`; `examples/requirements.txt:16` — `numpy>=2.0,<3.0`.

JIT vs install-time build: `gsplat/cuda/_backend.py:30-35` tries to import the compiled module first
and falls back to `build_and_load_gsplat()` (JIT). `setup.py:284` adds `ext_modules=get_extensions()`
unless `BUILD_NO_CUDA=1` (`setup.py:40`), which is the install-time build path.

### c. Training entry point

Script: `examples/simple_trainer.py`. It is a tyro subcommand CLI with two configs, `default` and
`mcmc`, registered at `examples/simple_trainer.py:1576-1593` and dispatched at
`examples/simple_trainer.py:1594` (`tyro.extras.overridable_config_cli(configs)`) and
`examples/simple_trainer.py:1617` (`cli(main, cfg, verbose=True)`).

Docstring usage lines, `examples/simple_trainer.py:1567` and `:1570`:

```
CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default
CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25
```

The full COLMAP command that the benchmark scripts actually run.

Default strategy, `examples/benchmarks/basic.sh:22-25`:

```bash
CUDA_VISIBLE_DEVICES=0 python $EXAMPLES_DIR/simple_trainer.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
    --render_traj_path $RENDER_TRAJ_PATH \
    --data_dir data/360_v2/$SCENE/ \
    --result_dir $RESULT_DIR/$SCENE/
```

Eval-only re-run from a checkpoint, `examples/benchmarks/basic.sh:30-34`:

```bash
CUDA_VISIBLE_DEVICES=0 python $EXAMPLES_DIR/simple_trainer.py default --disable_viewer --data_factor $DATA_FACTOR \
    --render_traj_path $RENDER_TRAJ_PATH \
    --data_dir data/360_v2/$SCENE/ \
    --result_dir $RESULT_DIR/$SCENE/ \
    --ckpt $CKPT
```

MCMC strategy with a count cap, `examples/benchmarks/mcmc.sh:24-28`:

```bash
CUDA_VISIBLE_DEVICES=0 python $EXAMPLES_DIR/simple_trainer.py mcmc --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
    --strategy.cap-max $CAP_MAX \
    --render_traj_path $RENDER_TRAJ_PATH \
    --data_dir $SCENE_DIR/$SCENE/ \
    --result_dir $RESULT_DIR/$SCENE/
```

`examples/benchmarks/mcmc.sh:11` sets `CAP_MAX=1000000`. `examples/benchmarks/basic.sh:13-17` and
`examples/benchmarks/mcmc.sh:15-19` set `DATA_FACTOR=2` for bonsai, counter, kitchen, room and
`DATA_FACTOR=4` for the rest.

Flag mapping (all in the `Config` dataclass, `examples/simple_trainer.py:79`):

| purpose | flag | type | default | file:line |
|---|---|---|---|---|
| data path | `--data_dir` | `str` | `"data/360_v2/garden"` | `examples/simple_trainer.py:92` |
| image downscale | `--data_factor` | `int` | `4` | `examples/simple_trainer.py:94` |
| output directory | `--result_dir` | `str` | `"results/garden"` | `examples/simple_trainer.py:96` |
| iterations | `--max_steps` | `int` | `30_000` | `examples/simple_trainer.py:141` |
| test split period | `--test_every` | `int` | `8` | `examples/simple_trainer.py:98` |
| eval schedule | `--eval_steps` | `List[int]` | `[7_000, 30_000]` | `examples/simple_trainer.py:143` |
| checkpoint schedule | `--save_steps` | `List[int]` | `[7_000, 30_000]` | `examples/simple_trainer.py:145` |
| ply export toggle | `--save_ply` | `bool` | `False` | `examples/simple_trainer.py:147` |
| ply schedule | `--ply_steps` | `List[int]` | `[7_000, 30_000]` | `examples/simple_trainer.py:149` |
| seed | not found (no CLI flag) | | | see (i) |

### d. Count and budget flags

**MCMC `cap_max`** — `gsplat/strategy/mcmc.py:82`, `cap_max: int = 1_000_000`. Docstring at
`gsplat/strategy/mcmc.py:52`: "Maximum number of GSs. Default to 1_000_000." Exposed on the CLI by
tyro as `--strategy.cap-max` because `Config.strategy` is a `Union[DefaultStrategy, MCMCStrategy]`
field (`examples/simple_trainer.py:179-181`) and the `mcmc` subcommand binds `MCMCStrategy`
(`examples/simple_trainer.py:1590`).

What it does in code, `gsplat/strategy/mcmc.py:226-238`, inside `_add_new_gs`:

```python
current_n_points = len(params["means"])
n_target = min(self.cap_max, int(1.05 * current_n_points))
n_gs = max(0, n_target - current_n_points)
```

Each refine step grows the population by at most 5 %, and never above `cap_max`. It is a hard
ceiling on primitive count, not a target the run is driven to reach from above.

Other MCMC strategy knobs, all `gsplat/strategy/mcmc.py`:

| field | type | default | line |
|---|---|---|---|
| `noise_lr` | `float` | `5e5` | `:83` |
| `refine_start_iter` | `int` | `500` | `:84` |
| `refine_stop_iter` | `int` | `25_000` | `:85` |
| `noise_injection_stop_iter` | `int` | `-1` (never stop) | `:86` |
| `refine_every` | `int` | `100` | `:87` |
| `min_opacity` | `float` | `0.005` | `:88` |
| `verbose` | `bool` | `False` | `:89` |
| `noise_opacity_t` | `float` | `DEFAULT_MCMC_OPACITY_T` | `:90` |
| `noise_opacity_k` | `float` | `DEFAULT_MCMC_OPACITY_K` | `:91` |

`min_opacity` also drives the relocation of dead Gaussians at `gsplat/strategy/mcmc.py:203-216`
(`_relocate_gs` teleports every Gaussian whose sigmoid opacity is at or below `min_opacity`).

`DefaultStrategy` has **no count cap**. Its knobs, `gsplat/strategy/default.py`:

| field | type | default | line |
|---|---|---|---|
| `prune_opa` | `float` | `0.005` | `:99` |
| `grow_grad2d` | `float` | `0.0002` | `:100` |
| `grow_scale3d` | `float` | `0.01` | `:101` |
| `grow_scale2d` | `float` | `0.05` | `:102` |
| `prune_scale3d` | `float` | `0.1` | `:103` |
| `prune_scale2d` | `float` | `0.15` | `:104` |
| `refine_scale2d_stop_iter` | `int` | `0` | `:105` |
| `refine_start_iter` | `int` | `500` | `:106` |
| `refine_stop_iter` | `int` | `15_000` | `:107` |
| `reset_every` | `int` | `3000` | `:108` |
| `refine_every` | `int` | `100` | `:109` |
| `pause_refine_after_reset` | `int` | `0` | `:110` |
| `absgrad` | `bool` | `False` | `:111` |
| `revised_opacity` | `bool` | `False` | `:112` |
| `verbose` | `bool` | `False` | `:113` |
| `key_for_gradient` | `Literal["means2d", "gradient_2dgs"]` | `"means2d"` | `:114` |

Initial count knob, only used when `init_type != "sfm"`: `--init_num_pts`, `int`, default
`100_000`, `examples/simple_trainer.py:156`. `--init_type` defaults to `"sfm"`
(`examples/simple_trainer.py:154`), so the initial count comes from the COLMAP point cloud.

**Fixed seed flag: not found.** There is no seed field in `Config`. The seed is hardcoded at
`examples/simple_trainer.py:387` — `set_random_seed(42 + local_rank)`.

### e. Checkpoint format

- Path pattern: `{result_dir}/ckpts/ckpt_{step}_rank{world_rank}.pt`, format `.pt`
  (`torch.save`). Written at `examples/simple_trainer.py:1070-1072`. The compression benchmark
  reads back `${RESULT_DIR}/${SCENE}/ckpts/ckpt_29999_rank0.pt`
  (`examples/benchmarks/compression/mcmc.sh:45`).
- Directory creation: `self.ckpt_dir = f"{cfg.result_dir}/ckpts"` at
  `examples/simple_trainer.py:399-400`.
- The saved dict contains `"step"`, `"scene_id"` and `"splats"` (the `ParameterDict` state dict),
  plus optional `pose_adjust`, `app_module`, `post_processing`
  (`examples/simple_trainer.py:1055-1069`).
- PLY export path: `{result_dir}/ply/point_cloud_{step}.ply`, written by
  `gsplat.export_splats(..., format="ply", save_to=...)` at
  `examples/simple_trainer.py:1096-1105`, only when `cfg.save_ply` is set
  (`examples/simple_trainer.py:1073-1075`). `self.ply_dir` at `examples/simple_trainer.py:405-406`.
- The exporter: `export_splats` at `gsplat/exporter.py:588`, signature accepts
  `format: Literal["ply", "splat", "ply_compressed"] = "ply"` and `save_to: Optional[str] = None`
  (`gsplat/exporter.py:595-596`). Re-exported as `gsplat.export_splats`
  (`gsplat/__init__.py:64`, `:154`).
- **Loading a ply back**: `load_ply_to_splats(path: str) -> dict` at `gsplat/exporter.py:435`. It
  requires `plyfile` (`gsplat/exporter.py:462-468`) and returns `means (N,3)`, `scales (N,3)`,
  `quats (N,4)`, `opacities (N,)`, `sh0 (N,1,3)`, `shN (N,K-1,3)`
  (`gsplat/exporter.py:452-460`). Used at `examples/simple_viewer.py:48` and
  `examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py:211`.
- **Loading a checkpoint back**: `examples/simple_trainer.py:1532-1546` in `main()`, which
  `torch.load`s each `--ckpt` path, concatenates `ckpt["splats"][k]` across ranks into
  `runner.splats[k].data`, and reads `step = ckpts[0]["step"]`.

### f. Compression and size

- Module path: `gsplat/compression/`, exporting one class.
  `gsplat/compression/__init__.py:16` — `from .png_compression import PngCompression`.
- Class name: `PngCompression` at `gsplat/compression/png_compression.py:31`. Fields
  `use_sort: bool = True` (`:59`) and `verbose: bool = True` (`:60`).
- CLI flag to enable it: `--compression png`. The field is
  `compression: Optional[Literal["png"]] = None` at `examples/simple_trainer.py:85`. The
  compression benchmark passes `--compression png` at `examples/benchmarks/compression/mcmc.sh:44`.
  Enabling it requires `plas` and `torchpq` (`examples/simple_trainer.py:1600-1609`).
- Entry: `Runner.run_compression` at `examples/simple_trainer.py:1412`. Output directory
  `compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"`
  (`examples/simple_trainer.py:1417`). It compresses, decompresses, overwrites `self.splats` with
  the decompressed values, then calls `self.eval(step=step, stage="compress")`
  (`examples/simple_trainer.py:1420-1426`).
- What lands on disk, from `PngCompression.compress` (`gsplat/compression/png_compression.py:90`)
  and its per-parameter dispatch (`:62-74`):
  - `means_l.png` and `means_u.png` — 16-bit split, `_compress_png_16bit`
    (`gsplat/compression/png_compression.py:262-267`, name pattern `f"{param_name}_l.png"` and
    `f"{param_name}_u.png"`).
  - `scales.png`, `quats.png`, `opacities.png`, `sh0.png` — `_compress_png`
    (`gsplat/compression/png_compression.py:189`, pattern `f"{param_name}.png"`).
  - `shN.npz` — k-means, `_compress_kmeans`
    (`gsplat/compression/png_compression.py:397`, pattern `f"{param_name}.npz"`).
  - any other field — `{param_name}.npz` via `_compress_npz`
    (`gsplat/compression/png_compression.py:318-320`).
  - `meta.json` — written last at `gsplat/compression/png_compression.py:125-126`.
  - Warning in the class docstring at `gsplat/compression/png_compression.py:40-42`: it drops the
    lowest-opacity splats when the count is not a perfect square (`:102-109`).
- **Total size reporting**: `examples/benchmarks/compression/summarize_stats.py`. It zips the
  `compression/` directory to `{scene_dir}/compression.zip`
  (`examples/benchmarks/compression/summarize_stats.py:34-37`), reads the byte count with
  `stat -c%s` (`:38-41`), appends it under the key `"size"` (`:42`), merges the stats from
  `stats/compress_step29999.json` (`:44-47`), averages across scenes (`:49-50`) and writes
  `{results_dir}/compress_summary.json` (`:53-54`). Invoked from
  `examples/benchmarks/compression/mcmc.sh:52`.

### g. Evaluation

- Function: `Runner.eval(self, step: int, stage: str = "val")` at
  `examples/simple_trainer.py:1201`.
- Metric objects built at `examples/simple_trainer.py:606-615`:
  `StructuralSimilarityIndexMeasure(data_range=1.0)` (`:606`),
  `PeakSignalNoiseRatio(data_range=1.0)` (`:607`),
  `LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)` (`:610-612`) or
  `net_type="vgg"` when `--lpips_net vgg` (`:613-615`). Flag
  `lpips_net: Literal["vgg", "alex"] = "alex"` at `examples/simple_trainer.py:257`. The
  compression benchmark forces vgg with `--lpips_net vgg`
  (`examples/benchmarks/compression/mcmc.sh:43`) "to align with other benchmarks" (`:38`).
- Metric accumulation: `examples/simple_trainer.py:1257-1259`.
- Results written as JSON: `f"{self.stats_dir}/{stage}_step{step:04d}.json"` at
  `examples/simple_trainer.py:1295-1296`, with `self.stats_dir = f"{cfg.result_dir}/stats"`
  (`examples/simple_trainer.py:401-402`). So `results/<scene>/stats/val_step29999.json` for
  validation and `.../compress_step29999.json` for the compressed pass.
- The stats dict also carries `"ellipse_time"` (seconds per image) and `"num_GS"`
  (`examples/simple_trainer.py:1275-1280`).
- Training stats are written separately to
  `f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json"`
  (`examples/simple_trainer.py:1049-1052`).
- Rendered comparison images: `f"{self.render_dir}/{stage}_step{step}_{i:04d}.png"`
  (`examples/simple_trainer.py:1250-1253`), `render_dir` at `examples/simple_trainer.py:403-404`.
- **Test split rule, every 8th image, confirmed**: `examples/datasets/colmap.py:457-461`:

```python
indices = np.arange(len(self.parser.image_names))
if split == "train":
    self.indices = indices[indices % self.parser.test_every != 0]
else:
    self.indices = indices[indices % self.parser.test_every == 0]
```

  with `test_every: int = 8` at `examples/datasets/colmap.py:128`, plumbed from
  `Config.test_every` (`examples/simple_trainer.py:98`).
- **Resize rule**: `_resize_image_folder` at `examples/datasets/colmap.py:49`, which divides both
  sides by `factor` and uses `Image.BICUBIC` (`examples/datasets/colmap.py:63-70`). It only fires
  when `factor > 1` and the discovered images are `.jpg`
  (`examples/datasets/colmap.py:234-238`), writing PNGs into `images_{factor}_png`. Intrinsics are
  divided by `factor` at `examples/datasets/colmap.py:169` and image sizes at `:176`.

### h. Rendering speed

- `Runner.eval` measures per-image render time with `torch.cuda.synchronize()` around the render
  call (`examples/simple_trainer.py:1224-1241`), averages it
  (`examples/simple_trainer.py:1272`) and reports it as `"ellipse_time"` in seconds per image
  (`examples/simple_trainer.py:1277`, printed at `:1285` and `:1291`). No FPS flag, this is always on.
- A dedicated FPS benchmark exists:
  `examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py`.
  It converts a median millisecond timing into FPS at `:319` (`fps = 1000.0 / median if median > 0 else 0.0`),
  prints it at `:330` and in a table at `:1006` and `:1028-1031`. The README points at it at
  `README.md:113` and names the viewer flag `--use_gaussian_render_inference_scene`.

### i. Determinism

- Seed setting: `set_random_seed(42 + local_rank)` at `examples/simple_trainer.py:387`, inside
  `Runner.__init__`. The helper is `examples/utils.py:168-171`:

```python
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

  It does not touch `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic` or
  `torch.use_deterministic_algorithms`.
- No CLI seed flag exists (**not found**).
- README note on non-determinism: **not found**. The nearest statement is `README.md:82`, which
  claims the trainer "reproduces the official Gaussian Splatting with exactly the same performance
  on PSNR, SSIM, LPIPS, and converged number of Gaussians".

### j. Datasets and directory layout

- README names the benchmark data as `mipnerf_360` downloaded via
  `python datasets/download_dataset.py` (`README.md:89`), and the scripts use `data/360_v2`
  (`examples/benchmarks/basic.sh:6`, `examples/benchmarks/mcmc.sh:6`,
  `examples/benchmarks/compression/mcmc.sh:1`).
- Scene list, `examples/benchmarks/basic.sh:8` — `garden bicycle stump bonsai counter kitchen room`
  with `treehill flowers` commented out. The compression script uses all nine
  (`examples/benchmarks/compression/mcmc.sh:3`).
- COLMAP layout, `examples/datasets/colmap.py:137-142`: it looks for `sparse/0/` first and falls
  back to `sparse`, then asserts the directory exists. It parses with
  `pycolmap.Reconstruction(colmap_dir)` (`:144`).
- Image directories, `examples/datasets/colmap.py:220-226`: `images` always, plus
  `images_{factor}` when `factor > 1` and `no_factor_suffix` is false. Both must exist (`:226`).
- Download URLs, `examples/datasets/download_dataset.py:38-51`, target directory names at `:57-60`
  (`mipnerf360` → `360_v2`).
- README also lists an NCore v4 capture backend, selected by `--data_type ncore`
  (`examples/simple_trainer.py:90-92`).

---

## 2. ffsplat (w-m/ffsplat)

This is an encoder and container tool, not a trainer. There is **no training entry point**
(not found). Its role is encode a ply into a compact container and decode it back.

### a. Licence and Inria rasterizer

- Licence file name: `LICENSE`.
- First 5 lines verbatim (`LICENSE:1-5`, note the leading blank line):

```

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

```

- Inria `diff-gaussian-rasterization`: **not vendored**, and there is no `.gitmodules`. Rendering
  goes through gsplat: `pyproject.toml:18` pins `gsplat==1.4.0` and
  `src/ffsplat/cli/eval.py:8` does `from gsplat import rasterization`.

### b. Installation

README commands, verbatim:

- Quick start, `README.md:17-22`:

```
- Install CUDA 12.x
- Install uv
uvx --from git+https://github.com/w-m/ffsplat@main ffsplat live --help
```

- Dev environment, `README.md:31-32` then `README.md:38`:

```
git clone https://github.com/w-m/ffsplat.git
cd ffsplat
```

```bash
make install
```

  `Makefile:2-7` expands `make install` into `uv sync`, `uv pip install -e .` and
  `uv run pre-commit install`.

Versions named:

- `pyproject.toml:8` — `requires-python = ">=3.12,<4.0"`.
- `pyproject.toml:19` — `torch==2.6.0`; `pyproject.toml:18` — `gsplat==1.4.0`;
  `pyproject.toml:33` — `torchvision>=0.21.0,<1.0.0`.
- `pyproject.toml:42-45` declares a `pytorch-cu126` index.

CUDA extension: ffsplat compiles nothing of its own. It inherits gsplat's JIT path, and
`README.md:24` warns "On the first launch, it will take a few minutes to compile gsplat kernels
before a rendered image shows up."

### c. Training entry point

**Not found.** ffsplat has four subcommands only: `convert`, `view`, `live`, `eval`
(`src/ffsplat/cli/__main__.py:55-60`, usage text at `:42-45`).

The nearest thing the README gives is the live encoding session, `README.md:44-48`:

```bash
uv run ffsplat live \
    --input /data/gaussian_splatting/mini-splatting2/truck_sparse/point_cloud/iteration_18000/point_cloud.ply \
    --input-format=3DGS-INRIA.ply \
    --dataset-path /data/gaussian_splatting/tandt_db/tandt/truck/ \
    --verbose
```

### d. Count and budget flags

**Not found.** ffsplat has no primitive count or budget flag. The closest thing is a `prune_by`
key inside the PLAS sort block of a format template
(`src/ffsplat/conf/format/SOG-web.yaml:49`, `prune_by: opacities`), which crops the point set to a
square grid rather than to a target count.

### e. Checkpoint format

**Not applicable, no training.** What it reads and writes:

- Input formats accepted by `decode_gaussians`
  (`src/ffsplat/coding/scene_decoder.py:103`): `"3DGS-INRIA.ply"` (`:106`),
  `"3DGS-INRIA-nosh.ply"` (`:116`), `"smurfx"` for a container directory (`:126`).
- Output format templates live in `src/ffsplat/conf/format/` and their file stems are the accepted
  `--output-format` values (`EncodingParams.from_template_yaml` at
  `src/ffsplat/coding/scene_encoder.py:150-155`):
  `3DGS-INRIA-nosh-ply`, `3DGS-INRIA-ply`, `SOG-PlayCanvas`, `SOG-web-nosh`, `SOG-web-png`,
  `SOG-web-sh-split`, `SOG-web`.

### f. Compression and size

- **Encode a ply**, CLI entry point: `ffsplat convert`, `src/ffsplat/cli/convert.py:10`
  (`def main()`), dispatched from `src/ffsplat/cli/__main__.py:56`. Console script names at
  `pyproject.toml:53` (`ffsplat-convert`) and `pyproject.toml:59` (`ffsplat`).
- Flags, all in `src/ffsplat/cli/convert.py`:

| flag | type | default | line |
|---|---|---|---|
| `--input` | `Path` | required | `:19` |
| `--input-format` | `str` | required | `:21-26` |
| `--output` | `Path` | required | `:27` |
| `--output-format` | `str` | required | `:28` |
| `--device` | `str` | `"cuda:0"` | `:29` |
| `--verbose` | flag | `False` | `:30-34` |

- README docstring example of both directions, `src/ffsplat/cli/__main__.py:20-23`:

```
>>> ffsplat convert --input scene.ply --input-format ply \
                   --output scene.sog --output-format SOG-web

>>> ffsplat view --input scene.sog --input-format SOG-web
```

  Note `--input-format ply` in that docstring is **not** one of the accepted strings at
  `src/ffsplat/coding/scene_decoder.py:106-131`. Use `3DGS-INRIA.ply`, as the README's live
  example does (`README.md:46`).

- **Decode back to a ply**: same `convert` subcommand, reversed, with
  `--input-format smurfx --output-format 3DGS-INRIA-ply`. The smurfx branch is
  `src/ffsplat/coding/scene_decoder.py:126-129` and requires `--input` to be a directory. The ply
  encoder template is `src/ffsplat/conf/format/3DGS-INRIA-ply.yaml`.
- Flow inside convert: `decode_gaussians(...)` then `.to(cfg.device)` then `encode_gaussians(...)`
  at `src/ffsplat/cli/convert.py:54-57`.
- What it writes for `SOG-web`:
  - One image per field, named `f"{field_name}.{codec}"` at
    `src/ffsplat/models/transformations.py:1089-1090`, with `codec: avif` from the template. The
    fields written by `SOG-web.yaml` are `opacities` (`:182-190`), `scales` (`:192-200`),
    `means_bytes_0` and `means_bytes_1` (`:202-211`), `quaternions` (`:213-221`), `f_dc`
    (`:223-231`), `f_rest_labels_bytes_0` (`:233-241`), `f_rest_labels_bytes_1` (`:243-251`),
    `f_rest_centroids` (`:253-261`).
  - `meta.json` for the canvas metadata block,
    `src/ffsplat/models/transformations.py:1111-1113` and `:1153`.
  - `container_meta.yaml`, always, at `src/ffsplat/coding/scene_encoder.py:236-237`.
- **Reading the resulting size**: `get_directory_size(path: Path) -> int` at
  `src/ffsplat/cli/eval.py:110`, which walks the directory and sums `os.path.getsize`
  (`:115-118`). It is called at `src/ffsplat/cli/eval.py:141` and the value goes into the stats
  dict under `"size"` (`:147`), printed as MB at `:153`:
  `f"Size: {stats["size"] / 1024 / 1024:.3f}MB "`.

### g. Evaluation

- CLI entry point: `ffsplat eval`, `src/ffsplat/cli/eval.py:157` (`def main()`), dispatched from
  `src/ffsplat/cli/__main__.py:59`. Console script `ffsplat-eval` at `pyproject.toml:57`.
- Flags: `--dataset-path` (`Path`), `--data-path` (`Path`), `--results-path` (`Path`),
  `src/ffsplat/cli/eval.py:160-162`. There are no defaults. `--data-path` is the encoded container
  directory, always decoded as `"smurfx"` (`:166`).
- Metric function: `evaluation(gaussians, valset, data_path, results_path)` at
  `src/ffsplat/cli/eval.py:122`, building
  `StructuralSimilarityIndexMeasure(data_range=1.0)` (`:126`),
  `PeakSignalNoiseRatio(data_range=1.0)` (`:127`) and
  `LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False)` (`:128`). Metrics are
  appended per view at `src/ffsplat/cli/eval.py:104-106`.
- **Results are printed to stdout only** (`src/ffsplat/cli/eval.py:149-154`). No json or txt file
  is written (**not found**). Only per-view comparison PNGs land on disk:
  `results_path / f"eval_{step:04d}.png"` at `src/ffsplat/cli/eval.py:100`.
- **Test split rule, every 8th image, confirmed**:
  `src/ffsplat/datasets/colmapparser.py:203-204`:

```python
self.train_indices = indices[indices % test_every != 0]
self.test_indices = indices[indices % test_every == 0]
```

  with `test_every: int = 8` at `src/ffsplat/datasets/colmapparser.py:56`. `Dataset` defaults to
  `split: str = "eval"` and picks `test_indices` for anything but `"train"`
  (`src/ffsplat/datasets/dataset.py:17-23`).
- **Resize rule**: `_resize_image_folder` at `src/ffsplat/datasets/colmapparser.py:27`, dividing
  both sides by `factor` with `Image.BICUBIC` (`:39-44`). Two triggers:
  - `factor > 1` and the images are `.jpg`, writing into `image_dir + "_png"`
    (`src/ffsplat/datasets/colmapparser.py:140-141`).
  - `factor == -1` (default, `:54`) and the longest side exceeds 1600 px, writing into
    `image_dir + "_1600px"` (`src/ffsplat/datasets/colmapparser.py:151-159`). The comment at
    `:152` cites the Inria `camera_utils.py` line this replicates.

### h. Rendering speed

- `eval_step` brackets the render with `torch.cuda.Event(enable_timing=True)` and reports the
  elapsed time in seconds (`src/ffsplat/cli/eval.py:72-88`, division by 1000 at `:88`). It is
  averaged at `:139` and reported as `"elapsed_time"` seconds per image (`:145`, printed at `:151`).
- No FPS flag or FPS-reporting script (**not found**).

### i. Determinism

- One seed, hardcoded, in the PLAS sort shuffle: `torch.manual_seed(42)` at
  `src/ffsplat/models/transformations.py:892`, guarded by `if plas_cfg.shuffle:` (`:891`). The
  `SOG-web` template turns that on at `src/ffsplat/conf/format/SOG-web.yaml:52` (`shuffle: true`).
- No seed CLI flag (**not found**).
- README note on non-determinism: **not found**. The README does carry a stability warning at
  `README.md:11`: "**NOTE: this code is pre-alpha, under heavy development, a community developer
  preview.**"

### j. Datasets and directory layout

- README gives dataset paths only through the live example, `README.md:47`
  (`--dataset-path /data/gaussian_splatting/tandt_db/tandt/truck/`). It documents no layout in prose.
- COLMAP layout, `src/ffsplat/cli/eval.py:170-178`: it checks `sparse/0/`, falls back to `sparse`,
  and if neither exists it looks for `transforms_train.json` to pick the Blender parser, otherwise
  raises `ValueError("could not identify type of dataset")`.
- The parser repeats the same check at `src/ffsplat/datasets/colmapparser.py:66-68`, and builds
  `images` plus `images_{factor}` at `:129`.

---

## 3. taming-3dgs (humansensinglab/taming-3dgs)

### a. Licence and Inria rasterizer

- Two licence files. `LICENSE.md` covers the new work, `LICENSE_ORIGINAL.md` carries the Inria
  terms.
- `LICENSE.md:1-5` verbatim:

```
MIT License

Copyright (c) 2024 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
```

- `LICENSE_ORIGINAL.md:1-5` verbatim:

```
Gaussian-Splatting License  
===========================  

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.  
The *Software* is in the process of being registered with the Agence pour la Protection des  
```

- Inria `diff-gaussian-rasterization`: **vendored as a checked-in copy** at
  `submodules/diff-gaussian-rasterization/` (10 entries present in a plain clone, package name
  `diff_gaussian_rasterization` at `submodules/diff-gaussian-rasterization/setup.py:18`, and its own
  `submodules/diff-gaussian-rasterization/LICENSE.md` starts with "Gaussian-Splatting License").
  `submodules/simple-knn/` is likewise a checked-in copy.
- `.gitmodules` declares exactly one submodule:

```
[submodule "submodules/fused-ssim"]
	path = submodules/fused-ssim
	url = https://github.com/rahul-goel/fused-ssim.git
```

- `README.md:67` notes the drop-in performance optimizations live on the `rasterizer` branch and
  are "released under the MIT License".

### b. Installation

- `README.md:16` — `git clone https://github.com/humansensinglab/taming-3dgs.git --recursive`
- `README.md:18` — "Follow the instructions in the original 3DGS repository to setup the
  environment." No pip line is given in this README.
- `environment.yml` is the concrete recipe:

| item | value | line |
|---|---|---|
| env name | `taming_3dgs` | `environment.yml:1` |
| CUDA | `cudatoolkit=11.6` | `environment.yml:7` |
| Python | `python=3.7.13` | `environment.yml:9` |
| torch | `pytorch=1.12.1` | `environment.yml:11` |
| torchvision | `torchvision=0.13.1` | `environment.yml:13` |
| extensions | `submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, `submodules/fused-ssim` | `environment.yml:17-19` |

- CUDA extension builds **at install** through those three `pip:` entries. There is no JIT path.

### c. Training entry point

Script: `train.py`. Command line from `train.sh:1` (Mip-NeRF 360 bicycle, budgeted setting):

```bash
python train.py -s data//bicycle -i images_4 -m ./eval/bicycle_budget --quiet --eval --test_iterations -1  --optimizer_type default --budget 15  --densification_interval 500 --mode multiplier
```

Unconstrained setting, `train.sh:40`:

```bash
python train.py -s data//bicycle -i images_4 -m ./eval/bicycle_big --quiet --eval --test_iterations -1  --optimizer_type default --budget 5987095  --densification_interval 100 --mode final_count
```

Reproduction driver from the README, `README.md:46-51`:

```bash
python full_eval.py \
    -m360 <MipNeRF360 dataset path> \
    -tat <TanksAndTemples dataset path> \
    -db <DeepBlending dataset path> \
    --sh_lower \
    --mode ${MODE}
```

with `MODE={budget|big}` (`README.md:45`, `full_eval.py:59`) and `--dry_run` to print without
running (`README.md:57`, `full_eval.py:78-80`). `full_eval.py:83-84` builds the shared arguments
`" --quiet --eval --test_iterations -1 "` plus `--optimizer_type`, adds `--sh_lower` at `:87`, and
picks `" --densification_interval 100 --mode final_count"` for `big` (`:90`) or
`" --densification_interval 500 --mode multiplier"` for `budget` (`:121`).

Flag mapping:

| purpose | flag | type | default | file:line |
|---|---|---|---|---|
| data path | `-s` / `--source_path` | `str` | `""` | `arguments/__init__.py:50` |
| image subfolder | `-i` / `--images` | `str` | `"images"` | `arguments/__init__.py:52` |
| downscale | `-r` / `--resolution` | `int` | `-1` | `arguments/__init__.py:53` |
| output directory | `-m` / `--model_path` | `str` | `""` | `arguments/__init__.py:51` |
| iterations | `--iterations` | `int` | `30_000` | `arguments/__init__.py:74` |
| test split on | `--eval` | flag | `False` | `arguments/__init__.py:56` |
| seed | not found (no flag) | | | see (i) |

`-i images_4` is the Mip-NeRF 360 4x downscale for outdoor scenes and `-i images_2` for indoor
(`train.sh:1-9`, `full_eval.py:95` and `:101`).

### d. Count and budget flags

All parsed in `train.py`:

| flag | type | default | line |
|---|---|---|---|
| `--budget` | `float` | `20` | `train.py:318` |
| `--mode` | `str`, `choices=["multiplier", "final_count"]` | `"multiplier"` | `train.py:319` |
| `--cams` | `int` | `10` | `train.py:317` |
| `--ho_iteration` | `int` | `15000` | `train.py:321` |
| `--sh_lower` | flag | `False` | `train.py:322` |
| `--benchmark_dir` | `str` | `None` | `train.py:323` |
| `--websockets` | flag | `False` | `train.py:320` |
| `--densification_interval` | `int` | `100` | `arguments/__init__.py:86` |
| `--densify_from_iter` | `int` | `500` | `arguments/__init__.py:88` |
| `--densify_until_iter` | `int` | `15_000` | `arguments/__init__.py:89` |

README descriptions: `--cams` at `README.md:23-24`, `--budget` at `README.md:25-26`, `--mode` at
`README.md:27-29`, `--websockets` at `README.md:30-31`, `--ho_iteration` at `README.md:32-33`,
`--sh_lower` at `README.md:34-35`, `--benchmark_dir` at `README.md:36-37`.

**How the schedule is computed.** `get_count_array(start_count, multiplier, opt, mode)` at
`utils/taming_utils.py:100`, imported at `train.py:34` and called once at `train.py:90`:

```python
counts_array = get_count_array(len(scene.gaussians.get_xyz), args.budget, opt, mode=args.mode)
```

The body, `utils/taming_utils.py:101-117`, commented "Eq. (2) of taming-3dgs":

```python
if mode == "multiplier":
    budget = int(start_count * float(multiplier))
elif mode == "final_count":
    budget = multiplier

num_steps = ((opt.densify_until_iter - opt.densify_from_iter) // opt.densification_interval)
slope_lower_bound = (budget - start_count) / num_steps

k = 2 * slope_lower_bound
a = (budget - start_count - k*num_steps) / (num_steps*num_steps)
b = k
c = start_count

values = [int(1*a * (x**2) + (b * x) + c) for x in range(num_steps)]
```

So the per-step target count is a downward-opening parabola through `start_count` at step 0, with
its slope at step 0 set to twice the average slope. The count reaches the budget at the end of
densification. `start_count` is the SfM point count, so `multiplier` mode makes the final count
scene-dependent and `final_count` mode makes it exact.

The array is consumed at `train.py:175-181`, which passes `budget=counts_array[densify_iter_num+1]`
into `gaussians.densify_with_score(...)` at each densification step, with the counter incremented at
`train.py:182`. The step fires when `iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0`
(`train.py:159`) and only while `iteration < opt.densify_until_iter` (`train.py:154`).

Scene-specific values used in the paper: `full_eval.py:21` (`big_budgets`, final counts) and
`full_eval.py:38` (`budget_multipliers`).

### e. Checkpoint format

- Primary artifact is a **ply**: `{model_path}/point_cloud/iteration_{iteration}/point_cloud.ply`,
  written by `Scene.save` at `scene/__init__.py:85-87`:

```python
def save(self, iteration):
    point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
```

  The writer is `GaussianModel.save_ply(self, path)` at `scene/gaussian_model.py:225`. The reader
  is `GaussianModel.load_ply(self, path)` at `scene/gaussian_model.py:249`, called from
  `scene/__init__.py:79-81`.
- `scene.save(iteration)` is called at `train.py:151` on each `--save_iterations` step and
  unconditionally once more at `train.py:234` after the loop.
- Optional torch checkpoint: `{model_path}/chkpnt{iteration}.pth` at `train.py:225`
  (`torch.save((gaussians.capture(), iteration), ...)`), on `--checkpoint_iterations`
  (default `[30_000]`, `train.py:315`). Restored at `train.py:44-45` via
  `gaussians.restore(model_params, opt)`. `capture` at `scene/gaussian_model.py:80`, `restore` at
  `scene/gaussian_model.py:97`.
- `--save_iterations` default `[7_000, 30_000]` (`train.py:313`), with `args.iterations` appended
  at `train.py:325`.

### f. Compression and size

**Not found.** taming-3dgs has no compression module and no size reporting. Model size is whatever
the ply weighs on disk.

### g. Evaluation

Two-step, the standard Inria pipeline.

1. Render the held-out views: `render.py`. `render_set` at `render.py:24` writes
   `{model_path}/{name}/ours_{iteration}/renders/{idx:05d}.png` (`render.py:25`, `:34`) and the
   ground truth to `.../gt/{idx:05d}.png` (`render.py:26`, `:35`). Flags `--iteration`
   (`int`, `-1`), `--skip_train`, `--skip_test`, `--quiet` at `render.py:56-59`. Called as
   `python render.py -m ./eval/bicycle_budget` (`train.sh:14`).
2. Compute metrics: `metrics.py`. `evaluate(model_paths)` at `metrics.py:36`, computing
   `ssim(renders[idx], gts[idx])` (`metrics.py:72`), `psnr(renders[idx], gts[idx])`
   (`metrics.py:73`) and `lpips(renders[idx], gts[idx], net_type='vgg')` (`metrics.py:74`).
   Implementations imported at `metrics.py:17` (`utils.loss_utils.ssim`), `:18`
   (`lpipsPyTorch.lpips`), `:21` (`utils.image_utils.psnr`).
   Called as `python metrics.py -m ./eval/bicycle_budget` (`train.sh:27`).

Results are written to two JSON files at `metrics.py:88-91`:

- `{scene_dir}/results.json` with keys `SSIM`, `PSNR`, `LPIPS` (`metrics.py:81-83`, `:88-89`)
- `{scene_dir}/per_view.json` with the per-image values (`metrics.py:84-86`, `:90-91`)

**Test split rule, every 8th image, confirmed**: `scene/dataset_readers.py:132` declares
`def readColmapSceneInfo(path, images, eval, llffhold=8)` and `:149-150`:

```python
train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
```

It only splits when `eval` is set (`scene/dataset_readers.py:148`), which is why every train.sh
line passes `--eval`.

**Resize rule**: `loadCam` at `utils/camera_utils.py:19`. `--resolution` in `{1, 2, 4, 8}` divides
both sides by that integer (`utils/camera_utils.py:22-23`). `--resolution -1` (the default) caps
the width at 1600 px: `if orig_w > 1600` at `:26`, `global_down = orig_w / 1600` at `:32`,
otherwise `global_down = 1` at `:34`, then `scale = float(global_down) * float(resolution_scale)`
at `:38` and `resolution = (int(orig_w / scale), int(orig_h / scale))` at `:39`.

### h. Rendering speed

- `--benchmark_dir` (`str`, default `None`, `train.py:323`). When set, per-iteration CUDA event
  timings are recorded for forward (`train.py:111-126`), backward (`train.py:128-134`) and
  optimizer step (`train.py:210-212`), then written as three numpy arrays at
  `train.py:227-231`: `forward.npy`, `backward.npy`, `step.npy` inside that directory. These are
  training-step timings in milliseconds, not render FPS.
- Total wall-clock training time printed at `train.py:235`.
- No FPS flag or FPS output file (**not found**).

### i. Determinism

- Seed set inside `safe_state(args.quiet)`, called at `train.py:330`. The seeds are hardcoded in
  `utils/general_utils.py:133-135`:

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

  followed by `torch.cuda.set_device(torch.device("cuda:0"))` at `utils/general_utils.py:136`.
- No seed CLI flag (**not found**).
- README claim on determinism: `README.md:11` — "We improve the densification process to make the
  primitive count deterministic and implement several low-level optimizations for fast
  convergence." That is a claim about the count, not about bit-exact reproducibility. No
  non-determinism warning appears (**not found**).

### j. Datasets and directory layout

- README names three datasets through `full_eval.py` flags at `README.md:47-49`: MipNeRF360
  (`-m360`), TanksAndTemples (`-tat`), DeepBlending (`-db`).
- No directory layout is written out in prose (**not found in README**). The layout is the Inria
  COLMAP one, enforced in code: `scene/dataset_readers.py:132` and the `-i`/`--images` subfolder
  (`arguments/__init__.py:52`), with `images_4` and `images_2` used by `train.sh` and
  `full_eval.py`.
- Hardware note at `README.md:43`: 24 vCPUs, 512 GB RAM, 1 x NVIDIA RTX A4500 20G.

---

## 4. gaussian-splatting-pup (j-alex-hanson/gaussian-splatting-pup)

This is a **post-hoc pruning pipeline**, not a trainer from scratch. It needs a pretrained 3DGS
model.

### a. Licence and Inria rasterizer

- Licence file name: `LICENSE.md` (the Inria terms, unmodified).
- First 5 lines verbatim (`LICENSE.md:1-5`):

```
Gaussian-Splatting License  
===========================  

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.  
The *Software* is in the process of being registered with the Agence pour la Protection des  
```

- Inria `diff-gaussian-rasterization`: **a git submodule**, not a copy.
  `.gitmodules` lists five:

```
[submodule "submodules/simple-knn"]
	path = submodules/simple-knn
	url = https://gitlab.inria.fr/bkerbl/simple-knn.git
[submodule "submodules/diff-gaussian-rasterization"]
	path = submodules/diff-gaussian-rasterization
	url = https://github.com/graphdeco-inria/diff-gaussian-rasterization
[submodule "SIBR_viewers"]
	path = SIBR_viewers
	url = https://gitlab.inria.fr/sibr/sibr_core.git
[submodule "submodules/rasterization_and_pup_fisher"]
	path = submodules/rasterization_and_pup_fisher
	url = https://github.com/j-alex-hanson/rasterization_and_pup_fisher
[submodule "submodules/compress-diff-gaussian-rasterization"]
	path = submodules/compress-diff-gaussian-rasterization
	url = https://github.com/tuallen/compress-diff-gaussian-rasterization
```

  A `--depth 1` clone without `--recurse-submodules` leaves all five directories empty. Clone this
  repository with `git clone --depth 1 --recurse-submodules`.

### b. Installation

README commands, verbatim (`README.md:22-23`):

```shell
conda env create --file environment.yml
conda activate gaussian_splatting_pup
```

`README.md:19` says to follow the original 3DGS setup "including recursively cloning the repository
submodules", and `README.md:26` adds: "We created an additional submodule for CUDA Fisher
computation: `rasterization_and_pup_fisher`. Ensure that it is also cloned and installed."

Versions from `environment.yml`:

| item | value | line |
|---|---|---|
| env name | `gaussian_splatting_pup` | `environment.yml:1` |
| CUDA | `cudatoolkit=11.8` | `environment.yml:7` |
| Python | `python=3.8` | `environment.yml:9` |
| torch | `pytorch=2.4.1` | `environment.yml:11` |
| torchvision | `torchvision=0.19.1` | `environment.yml:12` |

CUDA extensions build **at install**, from the four pip entries at `environment.yml:15-18`:
`submodules/compress-diff-gaussian-rasterization`, `submodules/diff-gaussian-rasterization`,
`submodules/simple-knn`, `submodules/rasterization_and_pup_fisher`. No JIT path.

### c. Training entry point

There is a `train.py` (the stock 3DGS trainer), but the README documents only the pruning pipeline.

README, `README.md:35`:

```shell
bash scripts/full_pruning_pipeline.sh
```

driven by three environment variables (`README.md:32`, described at `:38-42`):

- `SCENE_DATA_PATH` — "the path to the COLMAP or NeRF Synthetic dataset"
- `SCENE_MODEL_PATH` — "the path to the pretrained 3D-GS model"
- `SCENE_NAME` — "unique name for the scene"

The actual command that script runs, `scripts/full_pruning_pipeline.sh:17-30` (round 1, 80 %
pruning):

```bash
CUDA_VISIBLE_DEVICES=0 python prune_finetune.py \
    -s $source_path \
    -m $directory1 \
    --start_pointcloud ${orig_path}/point_cloud/iteration_30000/point_cloud.ply \
    --eval \
    --prune_percent 0.$round1 \
    --position_lr_max_steps 35000 \
    --iterations 35000 \
    --save_iterations 35000 \
    --checkpoint_iterations 0 \
    --test_iterations 0 \
    --prune_type $prune_type \
    --fisher_resolution 4 \
    --port 6071
```

Round 2 (a further 50 %, giving 90 % total) is `scripts/full_pruning_pipeline.sh:37-51`, identical
except `--start_pointcloud ${directory1}/point_cloud/iteration_35000/point_cloud.ply`,
`--prune_percent 0.$round2` and the added `--first_iter 30000`.

Script variables: `start_iteration=30000` (`:4`), `prune_type=fisher` (`:7`), `round1=80` (`:8`),
`round2=50` (`:9`), output directories `./experiments/${scene_name}_$round1` (`:13`) and
`$directory1\_$round2` (`:33`). It copies `cameras.json` and `cfg_args` from the pretrained model
into each output directory (`:15-16`, `:35-36`).

The Fisher-precompute script, `README.md:52` — `bash scripts/create_fishers.sh`, running
`scripts/create_fishers.sh:4-9`:

```bash
CUDA_VISIBLE_DEVICES=0 python fisher_pool_xyz_scaling.py \
  -s ${source_path} \
  -m ${orig_path} \
  --iteration 30_000 \
  --pool-resolution 1 \
  --fisher-via-cuda
```

Flags: `--iteration` (`int`, `-1`, `fisher_pool_xyz_scaling.py:174`), `--quiet` (`:175`),
`--pool-resolution` (`int`, `1`, `:176`), `--fisher-via-cuda` (flag, `:177`).

No image-downscale flag is passed by the pipeline scripts. `-i`/`--images` and `-r`/`--resolution`
exist with the stock defaults `"images"` and `-1` (`arguments/__init__.py:52-53`).

### d. Count and budget flags

**The pruning script is `prune_finetune.py`.** There is also a `prune.py`, but it has no argparse
(**no `add_argument` calls**); it is a helper module providing `prune_list` and
`calculate_v_imp_score`, imported at `prune_finetune.py:37`.

**The ratio flag is `--prune_percent`.** It is a list, one entry per pruning round:

| flag | type | default | file:line |
|---|---|---|---|
| `--prune_percent` | `float`, `nargs="+"` | `[0.8, 0.5]` | `prune_finetune.py:302` |
| `--prune_iterations` | `int`, `nargs="+"` | `[30_001, 35_001]` | `prune_finetune.py:298` |
| `--prune_type` | `str` | `"fisher"` | `prune_finetune.py:301` |
| `--fisher_resolution` | `int` | `1` | `prune_finetune.py:303` |
| `--v_pow` | `float` | `0.1` | `prune_finetune.py:305` |
| `--first_iter` | `int` | `None` | `prune_finetune.py:304` |
| `--start_pointcloud` | `str` | `None` | `prune_finetune.py:300` |
| `--start_checkpoint` | `str` | `None` | `prune_finetune.py:299` |

What `--prune_percent` does in code: at `prune_finetune.py:193-195` the loop checks
`if iteration in args.prune_iterations`, reads `prune_percent = args.prune_percent[prune_idx]` and
advances `prune_idx`. For `--prune_type fisher` (`:197`) it accumulates a per-Gaussian 6x6 Fisher
over every training camera (`:199-209`), saves it to `{model_path}/fisher_iter{iteration}.pt`
(`:210`), takes the log determinant via singular values (`:212-213`) and calls
`gaussians.prune_gaussians(prune_percent, fishers_log_dets)` (`:214-217`). For
`--prune_type v_important_score` (`:219`) it uses the LightGaussian score instead (`:220-225`).
Anything else raises (`:226-227`). The new count is printed at `:229`.

The removal itself, `scene/gaussian_model.py:405-410`:

```python
def prune_gaussians(self, percent, import_score: list):
    sorted_tensor, _ = torch.sort(import_score, dim=0)
    index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    value_nth_percentile = sorted_tensor[index_nth_percentile]
    prune_mask = (import_score <= value_nth_percentile).squeeze()
    self.prune_points(prune_mask)
```

So `--prune_percent 0.8` removes the bottom 80 % by score. It is a **fraction, not a target count**.
There is no absolute count flag (**not found**).

**Fine-tuning iteration flags.** The number of iterations comes from the stock
`--iterations` (`int`, default `30_000`, `arguments/__init__.py:73`), which the pipeline overrides
to `35000` (`scripts/full_pruning_pipeline.sh:24` and `:44`), together with
`--position_lr_max_steps 35000` (`:23`, `:43`) whose default is `30_000`
(`arguments/__init__.py:77`). The loop runs `range(first_iter, opt.iterations + 1)` at
`prune_finetune.py:83`, where `first_iter` is parsed out of the `--start_pointcloud` path at
`prune_finetune.py:59` and can be overridden by `--first_iter` at `:64-65`. So the fine-tune
length is `iterations - first_iter`, that is 5000 steps per round in the shipped pipeline.

Other iteration lists, all `prune_finetune.py`:

| flag | type | default | line |
|---|---|---|---|
| `--test_iterations` | `int`, `nargs="+"` | `[30_000, 30_001, 35_000, 35_001, 40_000]` | `:288-290` |
| `--save_iterations` | `int`, `nargs="+"` | `[35_000, 40_000]` | `:291-293` |
| `--checkpoint_iterations` | `int`, `nargs="+"` | `[35_000, 40_000]` | `:295-297` |

**Does it need a pretrained model path? Yes.** Two ways in:

- `--start_pointcloud <path to point_cloud.ply>` at `prune_finetune.py:300`, loaded with
  `gaussians.load_ply(args.start_pointcloud)` at `prune_finetune.py:58`. The starting iteration is
  parsed from the path with `int(args.start_pointcloud.split('/')[-2].split('_')[-1])`
  (`prune_finetune.py:59`), which is why the path must keep the `iteration_30000` directory name.
- `--start_checkpoint <path to .pth>` at `prune_finetune.py:299`, loaded via
  `torch.load(checkpoint)` and `gaussians.restore(...)` (`prune_finetune.py:55`).

The pipeline script uses `--start_pointcloud`, and `README.md:40` states `SCENE_MODEL_PATH` "is the
path to the pretrained 3D-GS model".

### e. Checkpoint format

- **ply**, at `{model_path}/point_cloud/iteration_{iteration}/point_cloud.ply`, written by
  `Scene.save` at `scene/__init__.py:85-87`, called from `prune_finetune.py:166`. The README
  confirms the pipeline writes into `point_cloud/iteration_35000` (`README.md:44`).
- Torch checkpoint `{model_path}/chkpnt{iteration}.pth` at `prune_finetune.py:172-174`. The
  pipeline disables it with `--checkpoint_iterations 0` (`scripts/full_pruning_pipeline.sh:26`,
  `:46`).
- Fisher tensors: `{model_path}/fisher_iter{iteration}.pt` written mid-run at
  `prune_finetune.py:210`, and the standalone precompute writes into
  `{model_path}/fisher-pool-{iteration}/fishers_xyz_scaling/`
  (`fisher_pool_xyz_scaling.py:149`, `:154-155`, `torch.save` at `:165`). This matches
  `README.md:59`.

### f. Compression and size

**Not found.** No compression module. The repo does vendor `compress-diff-gaussian-rasterization`
as a submodule (`.gitmodules`, installed at `environment.yml:15`), but that is a rasterizer, and no
script in this repo reports a model size in bytes.

### g. Evaluation

Same two-step Inria pipeline, driven from `scripts/full_pruning_pipeline.sh:54-57`:

```bash
python render.py \
    --source_path $source_path --skip_train \
    -m $directory2
python metrics.py --model_paths $directory2
```

- `render.py` writes `{model_path}/{name}/ours_{iteration}/renders/{idx:05d}.png` and
  `.../gt/{idx:05d}.png` (`render.py:28-29`, `:38-39`). Flags `--iteration` (`int`, `-1`),
  `--skip_train`, `--skip_test`, `--quiet` at `render.py:86-89`.
- `metrics.py` computes `ssim` (`metrics.py:72`), `psnr` (`:73`) and
  `lpips(..., net_type='vgg')` (`:74`), then writes `{scene_dir}/results.json`
  (`metrics.py:89-90`) and `{scene_dir}/per_view.json` (`metrics.py:91-92`). Flag
  `--model_paths` / `-m`, `nargs="+"`, required, at `metrics.py:102`. `README.md:44` confirms
  "image quality metrics of the test renders will be computed and saved in `results.json`".

**Test split rule, every 8th image, confirmed**: `scene/dataset_readers.py:132`
(`llffhold=8`) and `:149-150`, identical to taming-3dgs. Gated on `--eval`, which the pipeline
passes (`scripts/full_pruning_pipeline.sh:21`, `:41`).

**Resize rule**: `utils/camera_utils.py:22-23` for `--resolution` in `{1, 2, 4, 8}`, and the
1600 px width cap for `--resolution -1` at `utils/camera_utils.py:26`, `:32`, `:34`, `:38`.

### h. Rendering speed

**Present, and always on in `render.py`.** `render.py:53-56` brackets each render call with
`timeit.default_timer` (imported at `render.py:24`) and accumulates `total_time`. Then
`render.py:61-64`:

```python
avg_fps = num_frames / total_time
print("Average FPS:", avg_fps)
with open(os.path.join(model_path, f'fps_{iteration}.txt'), 'w') as f:
    f.write(str(avg_fps))
```

**Discrepancy to note in the plan**: `README.md:44` says the FPS lands in `fps.txt`, but the code
writes `fps_{iteration}.txt` (`render.py:63`). Use the code's name.

### i. Determinism

- Seed set inside `safe_state(args.quiet)` at `prune_finetune.py:311`. Hardcoded at
  `utils/general_utils.py:130-132`:

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

- No seed CLI flag (**not found**).
- README note on non-determinism: **not found**.

### j. Datasets and directory layout

- `README.md:38` — `SCENE_DATA_PATH` "is the path to the COLMAP or NeRF Synthetic dataset". No
  layout tree is drawn (**not found in README**).
- In code, the COLMAP layout is the stock Inria one: `readColmapSceneInfo` at
  `scene/dataset_readers.py:132`, with the image subfolder from `-i`/`--images`
  (`arguments/__init__.py:52`).
- Datasets named in the abstract, `README.md:16`: Mip-NeRF 360, Tanks & Temples, Deep Blending.

---

## 5. GaussianSpa (noodle-lab/GaussianSpa)

### a. Licence and Inria rasterizer

- Two licence files. `LICENSE` is the new work, `LICENSE.md` carries the Inria terms.
- `LICENSE:1-5` verbatim:

```
MIT License

Copyright (c) 2024 Miao Yin

Permission is hereby granted, free of charge, to any person obtaining a copy
```

- `LICENSE.md:1-5` verbatim:

```
Gaussian-Splatting License  
===========================  

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.  
The *Software* is in the process of being registered with the Agence pour la Protection des  
```

- Inria `diff-gaussian-rasterization`: **vendored as a checked-in copy**. There is **no
  `.gitmodules`** at all, and a plain `--depth 1` clone already contains:
  - `submodules/diff-gaussian-rasterization/` (package `diff_gaussian_rasterization`,
    `submodules/diff-gaussian-rasterization/setup.py:18`)
  - `submodules/diff-gaussian-rasterization_ms/` (package `diff_gaussian_rasterization_ms`,
    `submodules/diff-gaussian-rasterization_ms/setup.py:18`, the Mini-Splatting variant)
  - `submodules/simple-knn/`

  `README.md:18` explains the reason: "The repository contains submodules which are not compatible
  with newest 3DGS, thus please check it out with" `--recursive`.

### b. Installation

README commands, verbatim (`README.md:21`, then `:25-28`):

```shell
git clone https://github.com/noodle-lab/GaussianSpa.git --recursive
```

```shell
conda create -n gaussian_spa python=3.7
conda activate gaussian_spa
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Versions named: Python 3.7 (`README.md:25`), `torch==1.12.1+cu116` and `torchvision==0.13.1+cu116`
(`README.md:27`).

CUDA extensions build **at install**, from `requirements.txt:3-5`:
`submodules/diff-gaussian-rasterization`, `submodules/diff-gaussian-rasterization_ms`,
`submodules/simple-knn`. No JIT path.

### c. Training entry point

Three scripts, one per criterion.

| script | driver | README |
|---|---|---|
| `train_op.py` | `train_op.sh` (line 4 sets `PYTHON_SCRIPT="./train_op.py"`) | vanilla 3DGS, opacity criterion, `README.md:58-59` |
| `train_opacity.py` | `train_opacity.sh` (`:4`) | Mini-Splatting, opacity criterion, `README.md:74-75` |
| `train_imp_score.py` | `train_imp_score.sh` (`:4`) | Mini-Splatting, importance-score criterion, `README.md:78-79` |

README invocations, `README.md:58-59`, `:74-75`, `:78-79`:

```shell
chmod +x train_opa.sh
bash train_opa.sh

chmod +x train_opacity.sh
bash train_opacity.sh

chmod +x train_imp_score.sh
bash train_imp_score.sh
```

**Discrepancy to note**: `README.md:58-59` names `train_opa.sh`, but the file in the repository is
`train_op.sh`. Use `train_op.sh`.

The command each script actually runs, for a single COLMAP scene.

`train_opacity.sh:55-63`:

```bash
CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT" \
--port "$PORT" \
-s="$DATASET_DIR" \
-m="$OUTPUT_DIR" \
--eval \
--prune_ratio1 "0.50"\
--prune_ratio2 "$SPA_RATIO"\
--imp_metric "indoor"\
--iterations "40000"
```

with `SPA_RATIO=$(echo "0.80" | bc)` at `train_opacity.sh:50` and
`OUTPUT_DIR="./output/"$DATASET_NAME"/opacity"` at `train_opacity.sh:46`.

`train_op.sh:55-61` is the same minus `--prune_ratio1` and `--imp_metric`, with
`SPA_RATIO=$(echo "0.85" | bc)` (`train_op.sh:50`) and `--iterations "30000"` (`train_op.sh:61`).

`BASE_DATASET_DIR="../Dataset"` at line 5 of all three scripts, iterated over `run_scenes` at the
bottom (`train_opacity.sh:72-75`).

No image-downscale flag is passed. `-i`/`--images` defaults to `"images"`
(`arguments/__init__.py:52`) and `-r`/`--resolution` to `-1` (`arguments/__init__.py:53`).

### d. Count and budget flags

**There is no absolute target count flag** (**not found**). The target is expressed as two survival
ratios. All are `OptimizationParams` fields, so argparse registers them as `--<name>` through
`ParamGroup.__init__` (`arguments/__init__.py:20-38`):

| flag | type | default | file:line | README |
|---|---|---|---|---|
| `--prune_ratio1` | `float` | `0.50` | `arguments/__init__.py:90` | `README.md:86-87` |
| `--prune_ratio2` | `float` | `0.80` | `arguments/__init__.py:91` | `README.md:88-89` |
| `--optimizing_spa_interval` | `int` | `50` | `arguments/__init__.py:92` | `README.md:90-91` |
| `--optimizing_spa_start_iter` | `int` | `15_200` | `arguments/__init__.py:93` | not documented |
| `--optimizing_spa_stop_iter` | `int` | `25_200` | `arguments/__init__.py:94` | not documented |
| `--optimizing_spa` | `bool`, `action="store_true"` | `True` | `arguments/__init__.py:95` | `README.md:84-85` |
| `--rho_lr` | `float` | `0.0005` | `arguments/__init__.py:96` | not documented |
| `--iterations` | `int` | `40_000` | `arguments/__init__.py:73` | not documented |
| `--simp_iteration1` | `int` | `15_000` | `train_opacity.py:359`, `train_imp_score.py:346` | not documented |
| `--num_depth` | `int` | `3_500_000` | `train_opacity.py:360`, `train_imp_score.py:347` | not documented |
| `--num_max` | `int` | `4_500_000` | `train_opacity.py:361`, `train_imp_score.py:348` | not documented |
| `--imp_metric` | `str` | `"indoor"` in `train_opacity.py:363`, `required=True` in `train_imp_score.py:350` | | not documented |

**Caveat on `--optimizing_spa`**: because it is a bool with `default=True` and
`action="store_true"` (`arguments/__init__.py:35-36`), passing it is a no-op and it cannot be
turned off from the CLI. Disabling the method means editing `arguments/__init__.py:95`.

`--prune_ratio1` and `--prune_ratio2` are **removal fractions**, not survival fractions. Two places
in the code make that explicit.

The first cut, at `simp_iteration1`, `train_opacity.py:197-214`:

```python
factor = 1 - opt.prune_ratio1
N_xyz = gaussians._xyz.shape[0]
num_sampled=int(N_xyz*factor)

indices = np.random.choice(N_xyz, size=num_sampled,
                           p=prob, replace=False)

mask = np.zeros(N_xyz, dtype=bool)
mask[indices] = True
gaussians.prune_points(mask==False)
```

It keeps `1 - prune_ratio1` of the Gaussians, sampled without replacement with probability
proportional to importance (`train_opacity.py:199-202`).

The second cut, at `optimizing_spa_stop_iter`, `train_opacity.py:226-235`:

```python
opacity = gaussians.get_opacity
k_threshold = int((1-opt.prune_ratio2) * opacity.shape[0])
_, indices = torch.topk(opacity[:, 0], k=k_threshold, largest=True)
mask = torch.ones(opacity.shape[0], dtype=bool)
mask[indices] = False
gaussians.prune_points(mask)
```

It keeps the top `1 - prune_ratio2` by opacity. In `train_op.py` the same top-k cut appears twice,
once at `optimizing_spa_start_iter` (`train_op.py:140-141`) and once at
`optimizing_spa_stop_iter` (`train_op.py:157-158`).

**Alternation schedule.** The "sparsifying" step runs every `optimizing_spa_interval` iterations
inside the window `(optimizing_spa_start_iter, optimizing_spa_stop_iter]`. The guard, verbatim
from `train_opacity.py:194`:

```python
elif iteration % opt.optimizing_spa_interval == 0 and opt.optimizing_spa == True and (iteration > opt.optimizing_spa_start_iter and iteration <= opt.optimizing_spa_stop_iter):
    optimizingSpa.update(imp_score)
```

The `OptimizingSpa` object is constructed once at `iteration == opt.optimizing_spa_start_iter`
(`train_opacity.py:191-193`), with a first `update(imp_score, update_u=False)`. The same window
gates the penalty term added to the loss, `train_opacity.py:128` and `train_opacity.py:273`.
Defaults put the window at iterations 15 200 to 25 200 with a period of 50, so 200 sparsifying
steps. `train_opacity.sh:24` sets `SPA_INTERVAL="50"` but never passes it on the command line.

The sparsifying operator itself, `optimizing_spa.py`:

- `class OptimizingSpa` at `optimizing_spa.py:12`, taking `prune_ratio = opt.prune_ratio2`
  (`:18`) and `init_rho = opt.rho_lr` (`:17`).
- `update` at `:25-34` recomputes the auxiliary variable `z` and the dual `u`.
- `prune_z` at `:36-43` zeroes the lowest `prune_ratio` fraction of `z` by opacity.
- `prune_z_metrics_imp_score` at `:53-60` does the same but ranks by importance score.
- `append_spa_loss` at `:45-47` adds
  `0.5 * init_rho * ||opacity - z + u||₂²` to the loss.
- `adjust_rho` at `:49-51` multiplies rho by 5 after 85 % of the iterations. Note this sets
  `self.rho`, while `append_spa_loss` reads `self.init_rho`, so the adjustment has no effect on the
  loss as written.

**Which checkpoint it starts from: none by default.** `--start_checkpoint` exists
(`train_op.py:259`, `train_opacity.py:358`, `train_imp_score.py:345`, all `str`, default `None`)
and is loaded at `train_opacity.py:66-68`:

```python
if checkpoint:
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)
```

All three shipped scripts **comment it out**: `train_opacity.sh:64`
(`#--start_checkpoint "$ckpt"`) and `train_op.sh:62`. The scripts do define
`ckpt="$OUTPUT_DIR"/chkpnt"$chkpnt_iter".pth` with `chkpnt_iter=14999`
(`train_opacity.sh:49` and `:6`), so resuming from iteration 14 999 is the intended optional path.
As shipped, GaussianSpa **trains from the SfM point cloud in one run**, unlike PUP.

Recommended per-scene ratios are tabulated in the README at `README.md:99-114` (percent values, so
`75` means `--prune_ratio1 0.75`), alongside which criterion each scene uses.

### e. Checkpoint format

- **ply**, at `{model_path}/point_cloud/iteration_{iteration}/point_cloud.ply`, written by
  `Scene.save` at `scene/__init__.py:85-87`, called from `train_opacity.py:147`. The model path is
  `./output/<scene>/opacity` in the shipped scripts (`train_opacity.sh:46`).
- Torch checkpoint `{model_path}/chkpnt{iteration}.pth` at `train_opacity.py:244`
  (`torch.save((gaussians.capture(), iteration), ...)`), on `--checkpoint_iterations`, which
  defaults to `[]` (`train_opacity.py:357`, `train_op.py:258`, `train_imp_score.py:344`), so
  nothing is written unless asked.
- `--save_iterations` defaults to `[op.iterations]` (`train_opacity.py:355`,
  `train_imp_score.py:342`, `train_op.py:256`), so one ply at the final iteration.

### f. Compression and size

**Not found.** No compression module and no byte-size reporting.

### g. Evaluation

Same two-step Inria pipeline. The README does not spell out the render and metrics commands
(**not found in README**), but both scripts are present and unmodified in shape:

- `render.py`, `render_set` at `render.py:25-29` writing
  `{model_path}/{name}/ours_{iteration}/renders/{idx:05d}.png` and `.../gt/{idx:05d}.png`
  (`render.py:34-35`).
- `metrics.py`, computing `ssim` (`metrics.py:72`), `psnr` (`:73`) and
  `lpips(..., net_type='vgg')` (`:74`), writing `{scene_dir}/results.json`
  (`metrics.py:88-89`) and `{scene_dir}/per_view.json` (`metrics.py:90-91`).
- `README.md:12` states "Results evaluated by metrics.py across all scenes are available now",
  and the committed numbers live under `results/MipNeRF360`, `results/TanksAndTemples`,
  `results/DeepBlending`.

**Test split rule, every 8th image, confirmed**: `scene/dataset_readers.py:132` (`llffhold=8`)
and `:149-150`, identical to the other two Inria forks. Gated on `--eval`, which all three shipped
scripts pass (`train_opacity.sh:59`, `train_op.sh:59`).

**Resize rule**: `utils/camera_utils.py:22-23` for `--resolution` in `{1, 2, 4, 8}`, and the
1600 px width cap for `--resolution -1` at `utils/camera_utils.py:26`, `:32`, `:34`, `:38`.

### h. Rendering speed

**Not found.** `render.py` writes no timing file and prints no FPS. There is no `--benchmark_dir`
equivalent.

### i. Determinism

- Seed set inside `safe_state(args.quiet)`. Hardcoded at `utils/general_utils.py:130-132`:

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

- No seed CLI flag (**not found**).
- One deliberately stochastic step remains inside the seeded stream: the importance-weighted
  `np.random.choice` at `train_opacity.py:209-210`.
- README note on non-determinism: **not found**.

### j. Datasets and directory layout

The only one of the five READMEs that draws the layout. `README.md:31` names the two downloads
(Mip-360 and Tanks&Temples plus Deep Blending) and `README.md:33-53` gives the tree:

```txt
GaussianSpa
├──train_opacity.sh
├──train_imp_score.sh
└── ...
Dataset
├── drjohnson
├── playroom
├── bicycle
├── bonsai
├── counter
├── flowers
├── garden
├── kitchen
├── room
├── stump
├── treehill
├── train
└── truck
```

The `Dataset` directory sits beside the repository, matching `BASE_DATASET_DIR="../Dataset"` at
line 5 of all three shell scripts. Inside each scene it is the stock Inria COLMAP layout
(`scene/dataset_readers.py:132`, image subfolder from `-i`/`--images`,
`arguments/__init__.py:52`).

---

## Cross-repository summary of the count knobs

| repo | flag | type | default | file:line | semantics |
|---|---|---|---|---|---|
| gsplat | `--strategy.cap-max` | `int` | `1_000_000` | `gsplat/strategy/mcmc.py:82` | hard ceiling, MCMC only |
| gsplat | `--init_num_pts` | `int` | `100_000` | `examples/simple_trainer.py:156` | initial count, ignored when `init_type=sfm` |
| ffsplat | not found | | | | no count control |
| taming-3dgs | `--budget` | `float` | `20` | `train.py:318` | exact final count or SfM multiplier |
| taming-3dgs | `--mode` | `str` | `"multiplier"` | `train.py:319` | selects which of the two `--budget` means |
| pup | `--prune_percent` | `float`, `nargs="+"` | `[0.8, 0.5]` | `prune_finetune.py:302` | removal fraction per round |
| GaussianSpa | `--prune_ratio1` | `float` | `0.50` | `arguments/__init__.py:90` | removal fraction at `simp_iteration1` |
| GaussianSpa | `--prune_ratio2` | `float` | `0.80` | `arguments/__init__.py:91` | removal fraction at the sparsifying stop |

Only taming-3dgs `--mode final_count` and gsplat `--strategy.cap-max` express an absolute
primitive count. The other two express fractions of whatever the run happens to produce.
