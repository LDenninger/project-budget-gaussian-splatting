# Byte-side facts: HAC++, SizeGS, MesonGS++

Gathered from shallow clones under
`<session scratchpad>/repos/`.
Every claim carries a `file:line`. Nothing here is inferred from the papers.

Clone commands used:

```
git clone --depth 1 https://github.com/YihangChen-ee/HAC-plus
git clone --depth 1 https://github.com/mmlab-sigs/SizeGS
git clone --depth 1 https://github.com/mmlab-sigs/mesongs_plus
```

Cloned commits (`git log -1`):

| repo | commit | date | subject |
|---|---|---|---|
| HAC-plus | `46e3c4f4e2b99f98fc9d3da3f235b63aea8504f0` | 2025-11-13 | `Update README.md` |
| SizeGS | `2929778582a78727f13af4c0a619451333b8133e` | 2025-10-29 | `update readme` |
| mesongs_plus | `0004153458b73387e0044b6f553f62116117388d` | 2026-05-07 | `add result.xlsx` |

---

# 1. HAC-plus (HAC++, TPAMI'25)

Repo root: `repos/HAC-plus/`. Paths below are relative to that root.

## a. Licence and vendored rasterizer

Licence file: `LICENSE.md`. First 5 lines verbatim (`LICENSE.md:1-5`):

```
Gaussian-Splatting License  
===========================  

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.  
The *Software* is in the process of being registered with the Agence pour la Protection des  
```

`README.md:165`: "Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting)."
Source headers repeat the Inria banner, for example `train.py:1-10`, `arguments/__init__.py:1-10`,
`scene/gaussian_model.py:1-10`.

Vendored Inria rasterizer: yes, as a zip archive, not an unpacked directory.
`submodules/diff-gaussian-rasterization.zip` (5.8 MB, 1576 entries). Inside it,
`diff-gaussian-rasterization/LICENSE.md` line 1 is `Gaussian-Splatting License` and
`diff-gaussian-rasterization/setup.py` carries `# Copyright (C) 2023, Inria`. It is a modified
copy: it adds `cuda_rasterizer/forward_reweighted.cu` and `cuda_rasterizer/backward_reweighted.cu`
next to the stock `forward.cu` / `backward.cu`.

All zip archives in the repo (four, all under `submodules/`):

| archive | contents |
|---|---|
| `submodules/diff-gaussian-rasterization.zip` | Inria rasterizer, `setup.py` builds `diff_gaussian_rasterization._C` |
| `submodules/gridencoder.zip` | `gridencoder/` hash-grid CUDA extension, `setup.py`, `src/gridencoder.cu` |
| `submodules/simple-knn.zip` | Inria `simple-knn`, `setup.py`, `simple_knn.cu` |
| `submodules/arithmetic.zip` | the arithmetic coder, `arithmetic/arithmetic.cpp`, `arithmetic/arithmetic_kernel.cu`, `arithmetic/setup.py` |

## b. Installation

`README.md:48-56`, step 1 verbatim:

```
cd submodules
unzip diff-gaussian-rasterization.zip
unzip gridencoder.zip
unzip simple-knn.zip
unzip arithmetic.zip
cd ..
```

`README.md:57-61`, step 2 verbatim:

```
conda env create --file environment.yml
conda activate HAC_env
```

`README.md:63-67`, step 3: install `tmc3` (GPCC) from
https://github.com/MPEGGroup/mpeg-pcc-tmc13 and put it on PATH, otherwise its location must be
edited in `utils/gpcc_utils.py`. `README.md:67` says it is commonly at
`/PATH/TO/mpeg-pcc-tmc13/build/tmc3`. The default in code is the bare name `tmc3`
(`utils/gpcc_utils.py:245` and `utils/gpcc_utils.py:260`, parameter
`gpcc_codec_path: str='tmc3'`), invoked through `os.system` at `utils/gpcc_utils.py:31` and
`utils/gpcc_utils.py:43`.

`README.md:46`: "We tested our code on a server with Ubuntu 20.04.1, cuda 11.8, gcc 9.4.0."

Versions from `environment.yml`:

- `python=3.7.13` (`environment.yml:10`)
- `pytorch=1.12.1` (`environment.yml:12`), `torchvision=0.13.1` (`:14`), `torchaudio=0.12.1` (`:13`)
- `cudatoolkit=11.6` (`environment.yml:8`), note it differs from the CUDA 11.8 in `README.md:46`
- `plyfile=0.8.1` (`:9`), `pip=22.3.1` (`:11`), `pytorch-scatter` (`:15`), `tqdm` (`:16`)
- pip section (`environment.yml:17-24`): `einops`, `wandb`, `lpips`,
  `submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, `submodules/gridencoder`,
  `submodules/arithmetic`

So the four extensions are built by `conda env create` itself, from the unpacked directories.
The env name is `HAC_env` (`environment.yml:1`).

Entropy coder: a **custom CUDA arithmetic coder named `arithmetic`**, shipped as
`submodules/arithmetic.zip` and imported at `utils/encodings_cuda.py:4` as `import arithmetic`.
Its entry points are `arithmetic.arithmetic_encode` (`utils/encodings_cuda.py:100`, `:234`, `:363`,
`:452`), `arithmetic.arithmetic_decode` (`:163`, `:306`, `:426`, `:488`) and
`arithmetic.calculate_cdf` (`:215`, `:293`, `:352`, `:418`).
**Not** torchac and **not** constriction. Neither string appears anywhere in the repo.
The anchor coordinates go through GPCC / `tmc3` instead (`utils/gpcc_utils.py`).

## c. Training entry point and command

Entry point: `train.py`, main guard at `train.py:564`, dispatch to `training(...)` at `train.py:636`.

The README does not print a bare `train.py` line. It tells you to run per-dataset drivers
(`README.md:117-127`):

```
 python run_shell_xxx.py
```

with `run_shell_tnt.py`, `run_shell_mip360.py`, `run_shell_bungee.py`, `run_shell_db.py`,
`run_shell_blender.py` (`README.md:118-122`).

The real command line for one COLMAP scene, verbatim from `run_shell_mip360.py:7`
(f-string with `{scene}`, `{lmbda}`, `{mask_lr_final}` substituted by the driver):

```
CUDA_VISIBLE_DEVICES={0} python train.py -s data/mipnerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}
```

Tanks&Temples, `run_shell_tnt.py:6`:

```
CUDA_VISIBLE_DEVICES={0} python train.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}
```

Deep Blending, `run_shell_db.py:6`:

```
CUDA_VISIBLE_DEVICES={0} python train.py -s data/blending/{scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 -m outputs/blending/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}
```

BungeeNeRF, `run_shell_bungee.py:6` (uses `--lod 30 --voxel_size 0 --update_init_factor 128`).
Blender / NeRF-synthetic, `run_shell_blender.py:6` (uses `--voxel_size 0.001 --update_init_factor 4`).

Flag mapping:

| purpose | flag | where parsed | default |
|---|---|---|---|
| data path | `-s` / `--source_path` | `arguments/__init__.py:58` (`self._source_path = ""`), shorthand built at `arguments/__init__.py:29-33` | `""` |
| output dir | `-m` / `--model_path` | `arguments/__init__.py:59` (`self._model_path = ""`) | `""` |
| iterations | `--iterations` | `arguments/__init__.py:82` | `30_000` |
| rate multiplier | `--lmbda` | `train.py:585` | `0.001` |
| mask lr floor | `--mask_lr_final` | `arguments/__init__.py:94` | `0.0001` |
| test/eval split on | `--eval` | `arguments/__init__.py:64` (`self.eval = True`) | `True` |
| hash table size | `--log2` | `train.py:582` | `13` |
| 2D hash size | `--log2_2D` | `train.py:583` | `15` |
| features per level | `--n_features` | `train.py:584` | `4` |
| save iterations | `--save_iterations` | `train.py:577` | `[30_000]` |
| test iterations | `--test_iterations` | `train.py:576` | `[30_000]` |
| checkpoint iterations | `--checkpoint_iterations` | `train.py:579` | `[]` |
| resume checkpoint | `--start_checkpoint` | `train.py:580` | `None` |

Data layout expected: `data/<dataset_name>/<scene>/images/` plus `data/<dataset_name>/<scene>/sparse/0/`
(`README.md:78-96`), COLMAP binaries read at `scene/dataset_readers.py:144-152`.

### The λ flag in detail

- Name `--lmbda`, type `float`, **default `0.001`**, parsed at `train.py:585`.
- The drivers override it: `run_shell_mip360.py:3`, `run_shell_tnt.py:3`, `run_shell_db.py:3`,
  `run_shell_bungee.py:3` all read `for lmbda in [0.004]:` with the comment
  `# Optionally, you can try: 0.003, 0.002, 0.001, 0.0005`.
  `run_shell_blender.py:3` reads `for lmbda in [0.001]:` with the same comment.
- It reaches training through `args_param` (`train.py:636` passes `args` as the first positional).

**The one place λ multiplies the entropy loss** is `train.py:186`, inside the
`if bit_per_param is not None:` guard opened at `train.py:183`:

```python
        if bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
            loss = loss + args_param.lmbda * (bit_per_param + bit_hash_grid / denom)
```

(`train.py:183-186`). There is exactly one such multiplication. `grep -n "lmbda" train.py` returns
`train.py:186` and `train.py:585` only. The base loss it is added to is
`train.py:181`:

```python
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
```

### Iteration at which the entropy loss starts

`bit_per_param` is `None` until step 10000. It is initialised to `None` at
`gaussian_renderer/__init__.py:39` and only assigned inside `if step > 10000:` opened at
`gaussian_renderer/__init__.py:56`, assignment at `gaussian_renderer/__init__.py:118`.
The gate on the training side is `train.py:183` (`if bit_per_param is not None:`).
So the rate term is inactive for iterations 1..10000 and active from iteration 10001 on.

Two earlier stages exist and do **not** add a rate term:
`gaussian_renderer/__init__.py:47` `if step > 3000 and step <= 10000:` adds uniform quantisation
noise only, and `gaussian_renderer/__init__.py:53` `if step == 10000:` calls
`pc.update_anchor_bound()`.
Densification is also suspended over iterations 3000..3999 by `train.py:216`
(`if iteration not in range(3000, 4000):  # let the model get fit to quantization`).

### Mask weight

There is **no separate mask loss weight** in the loss. `grep -n "mask" train.py` shows the mask
only in logging (`train.py:172-174`) and in the densification statistics.
The mask is controlled through its **learning rate** instead:

- `--mask_lr_init` default `0.01` (`arguments/__init__.py:93`), `--mask_lr_final` default `0.0001`
  (`arguments/__init__.py:94`), `--mask_lr_delay_mult` `0.01` (`:95`), `--mask_lr_max_steps`
  `30_000` (`:96`).
- Scheduler built at `scene/gaussian_model.py:650-653`, applied per step at
  `scene/gaussian_model.py:699-701`. Param group registered at `scene/gaussian_model.py:607`
  and `:626` under the name `"mask"`.
- The drivers scale `mask_lr_final` with λ:
  - mip360 `run_shell_mip360.py:5-6`: `mask_lr_final = 0.0005 * lmbda / 0.001` then
    `mask_lr_final = min(mask_lr_final, 0.0015)`
  - tnt `run_shell_tnt.py:5`: `mask_lr_final = 0.0001 * lmbda / 0.001`
  - db `run_shell_db.py:5`: `mask_lr_final = 0.00008 * lmbda / 0.001`
  - bungee `run_shell_bungee.py:5`: `mask_lr_final = 0.0001 * lmbda / 0.001`
  - blender `run_shell_blender.py:5`: `mask_lr_final = 0.00008 * lmbda / 0.001`

## d. The rate estimate used during training

Function: `generate_neural_gaussians` in `gaussian_renderer/__init__.py:25`, rate block at
`gaussian_renderer/__init__.py:104-119`. The per-attribute bit tensors come from
`pc.EG_mix_prob_2.forward(...)` (`gaussian_renderer/__init__.py:104`, class
`Entropy_gaussian_mix_prob_2` at `utils/entropy_models.py:52`, `bits = -torch.log2(likelihood)` at
`utils/entropy_models.py:85`) and `pc.entropy_gaussian.forward(...)`
(`gaussian_renderer/__init__.py:110` and `:112`, class `Entropy_gaussian` at
`utils/entropy_models.py:30`, `bits = -torch.log2(likelihood)` at `utils/entropy_models.py:49`).

What it returns: **bits per parameter (a scalar mean over quantised parameters), not total bits and
not bits per anchor**. Exact expressions, `gaussian_renderer/__init__.py:115-119`:

```python
            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())
```

It is estimated on a random 5 % subsample of anchors, drawn per iteration at
`gaussian_renderer/__init__.py:72`:

```python
            choose_idx = torch.rand_like(pc.get_anchor[:, 0]) <= 0.05
```

The mask multiplies the bits before the mean, `gaussian_renderer/__init__.py:109-113`:

```python
            bit_feat = bit_feat * mask_anchor_chosen
            ...
            bit_scaling = bit_scaling * mask_anchor_chosen
            ...
            bit_offsets = bit_offsets * mask_anchor_chosen * binary_grid_masks_chosen
```

**The exact normalisation of the hash-grid term** in the loss, `train.py:185`:

```python
            denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
```

with `feat_dim` default `50` (`arguments/__init__.py:49`) and `n_offsets` default `10`
(`arguments/__init__.py:50`), so `denom = N_anchor * (50 + 6 + 30) = N_anchor * 86`.
`bit_hash_grid` is the second return value of `get_binary_vxl_size`
(`utils/encodings.py:16`, returns `Pg, ttl_bit, ttl_bit.item()/8.0/1024/1024, ttl_num` at
`utils/encodings.py:33`), so `bit_hash_grid / denom` converts total hash bits to the same
bits-per-parameter unit as `bit_per_param`. Both terms in `train.py:186` are therefore
bits per parameter.

Conversion to MB for logging, every 1000 iterations, `train.py:159-170`. Scale constant
`bit2MB_scale = 8 * 1024 * 1024` at `train.py:45` and again at `scene/gaussian_model.py:40`.
Expressions at `train.py:161-164`:

```python
            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB
```

That total omits the hash grid, the masks and the MLPs, so it is not the final size.

### The full estimate (no bitstream written)

`GaussianModel.estimate_final_bits`, `scene/gaussian_model.py:1130`, decorated `@torch.no_grad()`
at `scene/gaussian_model.py:1129`. Runs on all masked-in anchors
(`scene/gaussian_model.py:1136`, `mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]`), not a
5 % subsample. Component totals at `scene/gaussian_model.py:1171-1179`:

```python
        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32
        bit_masks = get_binary_vxl_size(_mask)[1].item()
```

`anchor_round_digits = 16` at `utils/encodings.py:11`.
It **returns a formatted string**, not numbers (`scene/gaussian_model.py:1183-1193`), with
`Total` computed as
`(bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0]) / bit2MB_scale`
(`scene/gaussian_model.py:1191`). `get_mlp_size` at `scene/gaussian_model.py:398`, counting every
parameter whose name contains `'mlp'` at 32 bits (`scene/gaussian_model.py:400-403`).
Called once, at the last test iteration, `train.py:276`.

## e. The real encoder and decoder

### Encoder

`GaussianModel.conduct_encoding(self, pre_path_name)`, `scene/gaussian_model.py:1196`, decorated
`@torch.no_grad()` at `scene/gaussian_model.py:1195`.
Called at `train.py:283`, with the directory made at `train.py:280-281`:

```python
                        bit_stream_path = os.path.join(pre_path_name, 'bitstreams')
                        os.makedirs(bit_stream_path, exist_ok=True)
                        # conduct encoding
                        log_info = scene.gaussians.conduct_encoding(pre_path_name=bit_stream_path)
```

so the bitstream directory is `<model_path>/bitstreams/`, matching `README.md:131`
("Encoded bitstreams will be stored in `./bitstreams` of the output directory").

Files written into `pre_path_name`:

| file | written at | content |
|---|---|---|
| `xyz_gpcc.npz` | `scene/gaussian_model.py:1225-1227` | GPCC-coded anchor integers plus `voxel_size` |
| `x_bound_min.pkl` | `scene/gaussian_model.py:1237` | `torch.save(self.x_bound_min, ...)` |
| `x_bound_max.pkl` | `scene/gaussian_model.py:1238` | `torch.save(self.x_bound_max, ...)` |
| `hash.b` | name at `:1246`, written at `:1339` | binarised hash-grid parameters |
| `masks.b` | name at `:1247`, written at `:1345` | the offset masks |
| `feat_<s>_<cc>.b` | name at `:1253`, `cc` suffix at `:1302` | anchor features, 5 channel chunks per batch |
| `scaling_<s>.b` | name at `:1254`, written at `:1312` | scalings |
| `offsets_<s>.b` | name at `:1255`, written at `:1324` | offsets, masked-in entries only |

`s` runs over batches of `MAX_batch_size = 3000` anchors (`scene/gaussian_model.py:41`, loop bounds
at `scene/gaussian_model.py:1240` and `:1249`). Inner chunking adds a further `_<c>` suffix inside
`encoder_gaussian_chunk` (`utils/encodings_cuda.py:336`), chunk sizes `50_0000` for features
(`scene/gaussian_model.py:1302`) and `10_0000` for scaling and offsets (`:1312`, `:1324`).

**How the total size in MB is computed and printed**, `scene/gaussian_model.py:1355-1364`:

```python
        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale + 32*3*2/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
```

The trailing `32*3*2/bit2MB_scale` is explained by the comment at `scene/gaussian_model.py:1354`:
"32*3*2/bit2MB_scale is for xyz_bound_min and xyz_bound_max". `bit2MB_scale = 8 * 1024 * 1024`
(`scene/gaussian_model.py:40`), so the printed unit is MiB.

Where each bit count comes from:

- `bit_anchor = bits_xyz` (`scene/gaussian_model.py:1331`), and
  `bits_xyz = os.path.getsize(npz_path) * 8` (`scene/gaussian_model.py:1228`). This is the only
  `os.path.getsize` in the training path.
- `bit_feat`, `bit_scaling`, `bit_offsets` are sums over batches
  (`scene/gaussian_model.py:1332-1334`) of the return values of `encoder_gaussian_mixed_chunk`,
  `encoder_gaussian_chunk`. Those return real stream lengths, for example
  `utils/encodings_cuda.py:467`: `bit_len = (len(byte_stream_bytes) + len(cnt_bytes)) * 8 + 32 * 2`.
- `bit_hash` from `encoder(...)` at `scene/gaussian_model.py:1339`, `bit_masks` from `encoder(...)`
  at `scene/gaussian_model.py:1345`, both `utils/encodings_cuda.py:439`.

The log line is written to the logger at `train.py:284`, which appends to
`<model_path>/outputs.log` (handler at `train.py:551`). Note `README.md:130` calls it
`output.log`, the code writes `outputs.log`.

Timing: printed at `scene/gaussian_model.py:1351-1352` (`print('encoding time:', t2 - t1)` and
`print('codec time:', t_codec)`), and folded into the returned string as `EncTime`
(`scene/gaussian_model.py:1364`) plus a per-component breakdown
(`scene/gaussian_model.py:1365-1372`). **No wall-clock figure is given in the README or in a code
comment**, so the actual duration is "not found" as a documented number. The repo ships an empty
`contextlib_duration.txt` (0 bytes) that nothing reads.

### Can the encoder be called mid-training

Yes, with caveats. Facts:

- Input: only `pre_path_name`, a directory path (`scene/gaussian_model.py:1196`). Everything else
  is read off `self`.
- It does **not** modify parameters. It only reads `self.get_mask_anchor`, `self.get_anchor`,
  `self._anchor_feat`, `self._offset`, `self.get_scaling`, `self.get_mask`
  (`scene/gaussian_model.py:1211-1217`), plus `self.calc_interp_feat` / `self.get_grid_mlp` /
  `self.get_deform_mlp` forwards. It writes files and calls `torch.cuda.empty_cache()`
  (`scene/gaussian_model.py:1329`).
- It is `@torch.no_grad()` (`scene/gaussian_model.py:1195`), so it builds no graph.
- **Precondition**: `self.x_bound_min` / `self.x_bound_max` must be set, since they are saved at
  `scene/gaussian_model.py:1237-1238` and used by `contract_to_unisphere`
  (`scene/gaussian_model.py:1102`) via `calc_interp_feat` (`scene/gaussian_model.py:525`).
  They are set by `update_anchor_bound` (`scene/gaussian_model.py:513`), called once before the
  loop at `train.py:99` and again at step 10000 via `gaussian_renderer/__init__.py:53-54`.
- **Precondition**: `tmc3` must be on PATH, since `compress_gpcc` shells out
  (`utils/gpcc_utils.py:245`, `os.system` at `utils/gpcc_utils.py:31`).
- Eval mode is **not required functionally**. `train.py:271` does call `scene.gaussians.eval()`
  before encoding and `scene.gaussians.train()` after (`train.py:360`), but
  `GaussianModel.eval` / `.train` (`scene/gaussian_model.py:405` and `:416`) only toggle
  `nn.Module` training flags, and there is no `Dropout`, `BatchNorm` or `LayerNorm` anywhere in
  `scene/`, `utils/` or `gaussian_renderer/` (verified by grep). `mlp_grid` is
  `nn.Sequential(nn.Linear, nn.ReLU(True), nn.Linear)` (`scene/gaussian_model.py:369-373`).
- **Side effect on the file system**: re-encoding into the same directory overwrites the previous
  `.b` files, and stale files from a previous encode with more batches are not removed.

### Decoder

`GaussianModel.conduct_decoding(self, pre_path_name)`, `scene/gaussian_model.py:1377`, decorated
`@torch.no_grad()` at `scene/gaussian_model.py:1376`. Called at `train.py:286`.

Reads back `x_bound_min.pkl` / `x_bound_max.pkl` (`scene/gaussian_model.py:1392-1393`),
`xyz_gpcc.npz` (`:1404`), `masks.b` (`:1417`), `hash.b` (`:1424`), then per batch
`feat_<s>_<cc>.b` (`:1473`), `scaling_<s>.b` (`:1486`), `offsets_<s>.b` (`:1493`).

**The decoder is destructive to the training state.** `scene/gaussian_model.py:1528-1533`:

```python
        self._anchor_feat = nn.Parameter(_anchor_feat)
        self._offset = nn.Parameter(_offset)
        self.decoded_version = True
        self._anchor = nn.Parameter(_anchor)
        self._scaling = nn.Parameter(_scaling)
        self._mask = nn.Parameter(_mask)
```

and `scene/gaussian_model.py:1535-1544` replaces the hash-grid parameters.
Setting `self.decoded_version = True` (`scene/gaussian_model.py:1530`) permanently changes the
semantics of `get_scaling` (`scene/gaussian_model.py:459-460`), `get_mask`
(`scene/gaussian_model.py:465-466`) and `get_anchor` (`scene/gaussian_model.py:507-508`), and the
new `nn.Parameter` objects are not registered in `self.optimizer`, whose param groups still point
at the old tensors (`scene/gaussian_model.py:594` onwards). **Do not call `conduct_decoding` on a
model that will keep training.** It prints no size, only times (`scene/gaussian_model.py:1548-1557`,
`DecTime` at `:1548`, `print('decoding time:', t2 - t1)` at `:1511`).

In the stock flow the decode is followed by a fresh `GaussianModel` built with
`decoded_version=run_codec` (`train.py:432`, `run_codec = True` at `train.py:46`) for rendering.

## f. Densification and mask

Anchors are grown and pruned in `GaussianModel.adjust_anchor`, `scene/gaussian_model.py:1011`.

- Growth: `self.anchor_growing(grads_norm, grad_threshold, offset_mask)` at
  `scene/gaussian_model.py:1018`, implementation `scene/gaussian_model.py:920`.
  New anchors get `new_masks = torch.ones_like(...)` (`scene/gaussian_model.py:980`) and enter the
  optimizer at `scene/gaussian_model.py:988` / `:1008`.
- Pruning: `prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)`
  at `scene/gaussian_model.py:1034`, ANDed with
  `anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1)`
  at `scene/gaussian_model.py:1035-1036`, applied by `self.prune_anchor(prune_mask)` at
  `scene/gaussian_model.py:1063`, implementation `scene/gaussian_model.py:906`.

Call site and iteration window, `train.py:213-224`:

```python
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                if iteration not in range(3000, 4000):  # let the model get fit to quantization
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
```

Window defaults, `arguments/__init__.py:147-154`:

- `start_stat = 500` (`:147`), statistics accumulate from iteration 501
- `update_from = 1500` (`:148`), first growth at iteration 1600
- `update_interval = 100` (`:149`)
- `update_until = 15_000` (`:150`), last densification at iteration 14900
- `min_opacity = 0.005` (`:152`), `success_threshold = 0.8` (`:153`),
  `densify_grad_threshold = 0.0002` (`:154`)
- iterations 3000..3999 are skipped (`train.py:216`)
- at exactly `update_until` the accumulators are freed (`train.py:220-224`)

So densification runs over 1600..14900 in steps of 100, minus 3000..3999, and the rate loss only
switches on at 10001. The two windows overlap over 10001..14900.

**How the mask interacts with the rate.** The mask is a learned per-offset gate:

- `get_mask` (`scene/gaussian_model.py:464-468`) is a straight-through binarisation of
  `sigmoid(self._mask[:, :10, :])` at threshold `0.01`:
  `((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig` (`scene/gaussian_model.py:468`).
- `get_mask_anchor` (`scene/gaussian_model.py:471-475`) reduces it to one gate per anchor:
  `mask_rate = torch.mean(mask, dim=1)`, then
  `((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate` (`scene/gaussian_model.py:474`).
- In the training rate estimate the gates multiply the bits at
  `gaussian_renderer/__init__.py:109`, `:111`, `:113` (quoted in section d). Because the gate is
  straight-through, gradient from `args_param.lmbda * bit_per_param` at `train.py:186` flows back
  into `self._mask` and pushes anchors and offsets off.
- The mask also multiplies opacity and scaling for rendering,
  `gaussian_renderer/__init__.py:195-196`, and drops fully-masked Gaussians at
  `gaussian_renderer/__init__.py:198-204`.
- At encode time the anchor mask selects which anchors are coded at all
  (`scene/gaussian_model.py:1211-1217`), and the offset mask selects which offsets are coded
  (`scene/gaussian_model.py:1318-1324`). The mask itself is coded into `masks.b`
  (`scene/gaussian_model.py:1345`).
- Mask size is logged every 1000 iterations at `train.py:171-174` via
  `get_binary_vxl_size(binary_grid_masks_anchor + 0.0)`.

## g. Checkpoint format and path pattern

Two different things are saved.

**1. The model, saved every entry in `--save_iterations`** (`train.py:205-207`), through
`Scene.save` (`scene/__init__.py:98-101`):

```python
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))
```

Path pattern: `<model_path>/point_cloud/iteration_<N>/point_cloud.ply` and
`<model_path>/point_cloud/iteration_<N>/checkpoint.pth`.

- `save_ply` at `scene/gaussian_model.py:742`, attribute list at
  `scene/gaussian_model.py:727-740` (`x,y,z,nx,ny,nz`, `f_offset_*`, `f_mask_*`,
  `f_anchor_feat_*`, `opacity`, `scale_*`, `rot_*`), written with `plyfile` at
  `scene/gaussian_model.py:762-763`.
- `save_mlp_checkpoints` at `scene/gaussian_model.py:1067`, a `torch.save` of a dict with keys
  `opacity_mlp`, `cov_mlp`, `color_mlp`, `encoding_xyz`, `grid_mlp`, `deform_mlp`
  (`scene/gaussian_model.py:1081-1088`), plus `mlp_feature_bank` when `use_feat_bank`
  (`scene/gaussian_model.py:1070-1079`).
- Loading: `Scene.__init__` at `scene/__init__.py:86-94` calls
  `load_ply_sparse_gaussian` (`scene/gaussian_model.py:765`) and `load_mlp_checkpoints`
  (`scene/gaussian_model.py:1091`). `searchForMaxIteration` resolves `load_iteration == -1`
  (`scene/__init__.py:35`), and `render_sets` passes `-1` (`train.py:647`).
- `README.md:134` warns that after training the `point_cloud.ply` is superseded by `./bitstreams`
  and may be deleted.

**2. A resumable training checkpoint**, only when `--checkpoint_iterations` is given
(default `[]`, `train.py:579`). Written at `train.py:229-231`:

```python
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
```

Path pattern `<model_path>/chkpnt<N>.pth`. Loaded at `train.py:102-104` when
`--start_checkpoint` is passed:

```python
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
```

**Warning for the plan.** `capture` (`scene/gaussian_model.py:427-439`) returns a 10-tuple
(`_anchor`, `_offset`, `_mask`, `_scaling`, `_rotation`, `_opacity`, `max_radii2D`, `denom`,
`optimizer.state_dict()`, `spatial_lr_scale`) while `restore`
(`scene/gaussian_model.py:441-452`) unpacks an 11-tuple that starts with `active_sh_degree`.
`capture` also reads `self.denom`, which is never assigned in `GaussianModel.__init__`
(`grep -n "self.denom" scene/gaussian_model.py` gives only `:436` in `capture` and `:454` in
`restore`). Neither the MLPs nor the hash grid are in `capture`. **This checkpoint path is broken
as shipped**, and the default `--checkpoint_iterations []` means it is never exercised.
A closed-loop controller that needs to resume should use the ply + `checkpoint.pth` pair instead.

## h. Evaluation

All evaluation lives inside `train.py`. There is no separate `render.py` or `metrics.py`.

- Per-view rendering and dumping: `render_set`, `train.py:363`. Output directories
  `<model_path>/<name>/ours_<iteration>/renders`, `.../errors`, `.../gt`
  (`train.py:364-366`), images named `{idx:05d}.png` (`train.py:404-407`), plus
  `<model_path>/<name>/ours_<iteration>/per_view_count.json` (`train.py:410-411`).
  With `iteration = -1` resolved by `searchForMaxIteration`, the test directory is
  `<model_path>/test/ours_30000/renders`, matching `README.md:132`.
- Driver: `render_sets`, `train.py:418`, called at `train.py:647`, `skip_train=True` and
  `skip_test=False` by default (`train.py:418`), so only the test split is rendered
  (`train.py:451-452`).
- **The function that computes PSNR, SSIM and LPIPS on the test split is `evaluate`,
  `train.py:476`**, called at `train.py:652`. The metric loop is `train.py:508-511`:

```python
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx], normalize=False).detach().mean().double())
```

  `ssim` from `utils/loss_utils.py` (imported `train.py:28`), `psnr` from `utils/image_utils.py`
  (imported `train.py:35`), and `lpips_fn = lpips.LPIPS(net='vgg').to('cuda')` at `train.py:41`
  (the pip `lpips` package, imported `train.py:26`, with the repo's own `lpipsPyTorch` commented
  out at `train.py:25` and `train.py:334`).
  `psnr` is defined at `utils/image_utils.py:17-19` as `20 * torch.log10(1.0 / torch.sqrt(mse))`,
  `ssim` at `utils/loss_utils.py:33` with `window_size=11` (Inria's stock implementation).
  `searchForMaxIteration` is `utils/system_utils.py:26-28`.
  It reads back the PNGs from disk through `readImages` (`train.py:463`), so the metrics are
  computed on 8-bit images, not on the float renders.
- A second, in-loop evaluation runs at every `--test_iterations` inside `training_report`
  (`train.py:261`), metrics at `train.py:330-333`, logged at `train.py:340`.
- **Output file names**, `train.py:541-544`:

```python
    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
```

  so `<model_path>/results.json` (keys `SSIM`, `PSNR`, `LPIPS`, `train.py:533-535`) and
  `<model_path>/per_view.json` (adds `VISIBLE_COUNT`, `train.py:536-539`).
  The training log with the sizes is `<model_path>/outputs.log` (`train.py:551`).
- **Every-8th-image rule**: `scene/dataset_readers.py:142`
  (`def readColmapSceneInfo(path, images, eval, lod, llffhold=8)`) and
  `scene/dataset_readers.py:170-171`:

```python
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
```

  It applies only when `eval` is true (`scene/dataset_readers.py:158`) and `lod <= 0`
  (`scene/dataset_readers.py:169`). For `lod > 0` (BungeeNeRF) the split is a prefix cut,
  `scene/dataset_readers.py:161-167`. Cameras are sorted by `image_name` first
  (`scene/dataset_readers.py:156`).
- **Resize rule**: `loadCam`, `utils/camera_utils.py:19-39`. With `--resolution` in `{1,2,4,8}`
  the image is divided by that factor (`utils/camera_utils.py:22-23`). The default is
  `--resolution -1` (`arguments/__init__.py:61`), which caps width at 1600 px
  (`utils/camera_utils.py:25-32`):

```python
        if args.resolution == -1:
            if orig_w > 1600:
                ...
                global_down = orig_w / 1600
            else:
                global_down = 1
```

- **README command to evaluate a trained model: not found.** The README gives no standalone
  evaluation command. `train.py:645-652` runs rendering and evaluation automatically at the end of
  training, and there is no `render.py` or `metrics.py` in the repo.

## i. Compression to a byte budget

Not applicable, HAC++ has no target-size flag. It is rate-controlled only through `--lmbda`.

## j. Seed and non-determinism

- **No `--seed` flag.** `grep -n "seed" train.py` returns nothing.
- Seeds are hard-coded in `safe_state`, `utils/general_utils.py:112`, at
  `utils/general_utils.py:130-132`:

```python
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
```

  called at `train.py:628`.
- `torch.cuda.manual_seed_all` is not called, and no `torch.use_deterministic_algorithms` or
  `cudnn.deterministic` setting exists anywhere (grep finds none).
- Remaining stochasticity that the fixed seeds do not remove: the 5 % anchor subsample for the rate
  estimate (`gaussian_renderer/__init__.py:72`), the quantisation noise
  (`gaussian_renderer/__init__.py:49-51`, `:67-69`, `:94`, `:98-99`), the random growth mask
  (`scene/gaussian_model.py:928-930`), the camera shuffle (`scene/__init__.py:72-74`) and the
  random camera pick (`train.py:143`). One line after seeding is itself random:
  `train.py:631`, `args.port = np.random.randint(10000, 20000)`.
- **No README note on non-determinism.** Nothing in `README.md` mentions seeds, variance or
  reproducibility.

## k. Per-scene λ for the paper's rate points

- **The README says nothing about per-scene λ.** The only mention is `README.md:133`:
  "Optionally, you can change `lmbda` in these `run_shell_xxx.py` scripts to try variable bitrate."
- There are **no config files**. The λ values live in the driver scripts, one list each:
  `run_shell_mip360.py:3`, `run_shell_tnt.py:3`, `run_shell_db.py:3`, `run_shell_bungee.py:3` all
  hold `for lmbda in [0.004]:` with the comment
  `# Optionally, you can try: 0.003, 0.002, 0.001, 0.0005`, and `run_shell_blender.py:3` holds
  `for lmbda in [0.001]:` with the same comment. The value is the same for every scene inside a
  dataset, so there is no per-scene λ table anywhere in the repo.
- The paper's two rate points are reported in `results/`, formatted per the 3DGS.zip survey
  (`results/README.md:1`). Each CSV holds exactly two rows, for example
  `results/MipNeRF360/bicycle.csv:1-3`:

```
Submethod,PSNR,SSIM,LPIPS,Size [Bytes],#Gaussians
HAC++-highrate,25.0481567,0.7381741,0.2675207,30671057.7152,3272052
HAC++-lowrate,25.0757256,0.7317612,0.2894176,13466651.8528,1093342
```

  The CSVs give sizes in bytes but **do not record which λ produced each row**. Mapping
  `HAC++-highrate` / `HAC++-lowrate` to λ values is **not found** in the repository.
  Files present: `results/MipNeRF360/{bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill}.csv`,
  `results/DeepBlending/{drjohnson,playroom}.csv`,
  `results/TanksAndTemples/{train,truck}.csv`,
  `results/SyntheticNeRF/{chair,drums,ficus,hotdog,lego,materials,mic,ship}.csv`.

## l. Notes for a closed-loop rate controller

Facts a plan needs, all sourced above:

1. λ enters at exactly one line, `train.py:186`, and is read from `args_param.lmbda`. Changing
   `args_param.lmbda` between iterations changes the loss immediately, with no other state to
   update.
2. The rate term is inert before iteration 10001 (`gaussian_renderer/__init__.py:56`), so a
   controller has nothing to act on before then.
3. `estimate_final_bits` (`scene/gaussian_model.py:1130`) is safe to call mid-training. It is
   `@torch.no_grad()`, touches no parameters, writes no files, and needs `x_bound_min` /
   `x_bound_max` (set at `train.py:99` and again at step 10000). It returns a **string**
   (`scene/gaussian_model.py:1193`), so the numbers must be recomputed or the function changed to
   return them.
4. `conduct_encoding` (`scene/gaussian_model.py:1196`) is also safe on parameters, but it writes
   into `pre_path_name` and needs `tmc3` on PATH. Its `Total` MB expression is
   `scene/gaussian_model.py:1363`.
5. `conduct_decoding` (`scene/gaussian_model.py:1377`) **must not** be called on a model that
   keeps training, per `scene/gaussian_model.py:1528-1544`.
6. Mask lr, not a mask loss weight, is the second knob, and the drivers tie it to λ linearly
   (`run_shell_*.py:5`).

---

# 2. SizeGS

Repo root: `repos/SizeGS/`. Paths below are relative to that root.
Title, `readme.md:1`: "SizeGS: Size-aware Compression of 3D Gaussian Splatting via Mixed Integer
Programming". The README file is lower-case `readme.md`.

## a. Licence and vendored rasterizer

**Root licence file: not found.** There is no `LICENSE`, `LICENSE.md` or `COPYING` at the repo root,
even though the Python headers say "under the terms of the LICENSE.md file", for example
`meson.py:6-7`.

Two licence files exist further down. `submodules/diff-gaussian-rasterization/LICENSE.md:1-5`
verbatim:

```
Gaussian-Splatting License  
===========================  

**Inria** and **the Max Planck Institut for Informatik (MPII)** hold all the ownership rights on the *Software* named **gaussian-splatting**.  
The *Software* is in the process of being registered with the Agence pour la Protection des  
```

`SIBR_viewers/LICENSE.md:1-5` verbatim:

```
SIBR License  
============  

**Inria** and **UCA** hold all the ownership rights on the *Software* named **sibr-core**.  
The *Software* has been registered with the Agence pour la Protection des  
```

Vendored Inria rasterizer: **yes, as ordinary tracked source, not a zip and not a git submodule**,
at `submodules/diff-gaussian-rasterization/`. It is the Scaffold-GS variant, carrying
`cuda_rasterizer/forward_reweighted.cu` and `cuda_rasterizer/backward_reweighted.cu` which
`submodules/diff-gaussian-rasterization/setup.py:42-48` does not compile.

**Zip archives: none.** `find . -name '*.zip' -not -path './.git/*'` returns nothing. All five CUDA
extensions are plain source trees with a `setup.py`: `submodules/diff-gaussian-rasterization`,
`submodules/simple-knn`, `submodules/octree`, `submodules/quant`, `submodules/seg-quant`.

The one declared git submodule is `submodules/BDD` (`.gitmodules:1-3`,
`https://github.com/LPMP/BDD.git`), which is **empty on disk** after a shallow clone and whose
imports are commented out at `meson.py:51-53`. It is not used.

## b. Installation

`readme.md:5-7`, clone:

```shell
git clone https://github.com/mmlab-sigs/sizegs --recursive
```

`readme.md:25-37`, install, verbatim:

```shell
conda create -n sizegs python=3.10
conda activate sizegs
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install tqdm plyfile einops wandb lpips laspy colorama jaxtyping opencv-python tensorboard loguru pulp Ninja open3d torchac
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/octree
pip install submodules/quant
pip install submodules/seg-quant
pip install numpy==1.26.4
```

Versions named: Python 3.10 (`readme.md:26`), torch 2.2.2+cu121, torchvision 0.17.2+cu121,
torchaudio 2.2.2+cu121 (`readme.md:28`), numpy 1.26.4 (`readme.md:36`), CUDA 12.1 or 12.4
(`readme.md:23`). Prerequisites at `readme.md:13-18`: Compute Capability 7.0+, Ubuntu 18.04 or
newer, CUDA 11.6 or newer.
There is **no zip to unpack**, and no `requirements.txt`, `environment.yml` or `pyproject.toml`.

Entropy coder: **`torchac`**.

- Installed on the pip line `readme.md:30`.
- Imported at `utils/compression.py:9` and `scene/gaussian_model.py:13`.
- Encode: `scene/gaussian_model.py:808`,
  `byte_stream = torchac.encode_float_cdf(lower.cpu(), x_int_round_idx.cpu(), check_input_bounds=True)`
- Decode: `scene/gaussian_model.py:1100`,
  `qf[c, si:ei] = torchac.decode_float_cdf(lower.cpu(), byte_stream_d).int() + int(min_values[idx])`
- No constriction, no custom arithmetic extension.

**Two undeclared runtime dependencies.**

1. The MPEG G-PCC `tmc3` binary, never mentioned in the README, with a hard-coded default path
   `~/storage/mpeg-pcc-tmc13/build/tmc3/tmc3` at `utils/compression.py:287`, `:301`, `:339`, `:354`.
   Failure is a hard assert, `utils/compression.py:298`. It is on the mandatory path whenever
   `--use_pcc` is set, which every shell script does (`sizegs.sh:41`).
2. The Info-ZIP `zip` and `unzip` CLI, called through `os.system` at
   `scene/gaussian_model.py:915`, `:993`, `:1028`, `:1327`, `:2426`.

`simple_knn` is installed by `readme.md:32` but its only import is commented out at
`scene/gaussian_model.py:22`.

## c. Compression command and byte budget

Entry point: `meson.py`. There is no `compress.py`.

**Budget flag: `--target_size`, unit MB (mebibytes).**

- Declared at `arguments/__init__.py:82`, `self.target_size = 15`, inside `ModelParams.__init__`.
  It becomes `--target_size` with `type=int` through the generic loop at
  `arguments/__init__.py:36-38`. There is no explicit `add_argument("--target_size", ...)`.
- **Default 15.**
- Unit proof, `meson.py:613`: `model_size_limit = dataset.target_size*1024*1024*8` (bits).
  Second use, `meson.py:407`:
  `dataset.percent = gaussians.search_tau(save_dir, imp, pipe.n_block, dataset.target_size * 1024 * 1024)` (bytes).

**Input model: a Scaffold-GS checkpoint, not a vanilla Inria 3DGS ply.** `readme.md:69` and
`readme.md:76` say "Pretrained ScaffoldGSes ... serve at the input of our code".
`load_ply_sparse_gaussian` (`scene/gaussian_model.py:1831`) reads the anchor fields listed by
`construct_list_of_attributes` (`scene/gaussian_model.py:542-553`): `x,y,z,nx,ny,nz`,
`f_offset_{i}`, `f_anchor_feat_{i}`, `opacity`, `scale_{i}`, `rot_{i}`. Four MLPs load separately as
TorchScript through `load_mlp_checkpoints` (`scene/gaussian_model.py:2226`).

Input path layout, `scene/__init__.py:39-42` picks the root and `scene/__init__.py:109-115` loads:

```
<load_path>/point_cloud/iteration_<load_iter>/point_cloud.ply
<load_path>/point_cloud/iteration_<load_iter>/{opacity_mlp.pt, cov_mlp.pt, color_mlp.pt}
```

with `--load_iter` default `30_000` (`meson.py:1257`) and `--mesongs` selecting `--load_path`
(`sizegs.sh:37`, `:40`, `:52`). Scene source data must be COLMAP or Blender
(`scene/__init__.py:60-67`), README tree at `readme.md:49-67`.

The canonical command is the wrapper `sizegs.sh`, verbatim `sizegs.sh:20-53`:

```bash
SECONDS=0
CUDA_VISIBLE_DEVICES=$device python meson.py \
    --eval -s /mnt/storage/users/szxie_data/nerf_data/${log}/${name} \
    --lod 0 \
    --voxel_size ${voxel} \
    --target_size $size \
    --update_init_factor 4 \
    --appearance_dim 0 \
    --ratio 1 \
    --iterations $iter \
    --position_lr_max_steps $iter\
    --offset_lr_max_steps $iter\
    --mlp_opacity_lr_max_steps $iter\
    --mlp_cov_lr_max_steps $iter\
    --mlp_color_lr_max_steps $iter\
    --mlp_featurebank_lr_max_steps $iter\
    --appearance_lr_max_steps $iter\
    --load_iter 30000 \
    --port 8989 \
    --n_block $block \
    --mesongs \
    --use_pcc \
    --enable_drop \
    --percent $prune_per \
    --prune_iterations $prune_iter \
    --octree_iterations $octree_iter \
    --bit_upper_bound 16 \
    --bit_lower_bound 1 \
    --scene_name $name \
    --dis_type 3 \
    --csv_path $csv_path \
    --fluc_percent $fluc_percent \
    --load_path outputs/${log}/${name}/baseline/scaffold \
    -m outputs/${log}/${name}/${logdir}/${time}
```

Positional contract from `sizegs.sh:3-13`:
`sizegs.sh <name> <iter> <prune_iter> <octree_iter> <voxel> <log> <block> <size_MB> <device> <prune_per> <fluc_percent>`.

The source path is hard-coded at `sizegs.sh:22` to
`/mnt/storage/users/szxie_data/nerf_data/${log}/${name}` and the input model root at
`sizegs.sh:52`. `readme.md:106` tells the user to fix these paths "in meson.sh", but
**`meson.sh` does not exist**. The two lines to edit are `sizegs.sh:22` and `sizegs.sh:52`.

Variants:

- `sizegs_auto_tau.sh` is `sizegs.sh` plus `--use_search_prune` (`sizegs_auto_tau.sh:52`) and
  `logdir=rebuttal_auto_tau` (`:15`).
- `sizegs_torchac.sh` is `sizegs.sh` plus `--use_torchac` (`sizegs_torchac.sh:42`) and
  `logdir=torchac_p1` (`:15`).
- Dataset drivers: `size_mip_360.sh`, `size_ac_mip_360.sh`, `size_tandt.sh`, `size_db.sh`.

**Argument-count bug to plan around.** `sizegs.sh` reads 11 positionals (`sizegs.sh:3-13`), but
`size_mip_360.sh`, `size_tandt.sh` and `size_db.sh` pass only 8 or 9. `$prune_per` and
`$fluc_percent` then expand to empty, so `--percent` (`sizegs.sh:43`) and `--fluc_percent`
(`sizegs.sh:51`) are emitted with no value. Only `size_ac_mip_360.sh` passes all 11, adding
`0.5 0.05`. `size_db.sh:3` and `:9` also omit `$device`, leaving `CUDA_VISIBLE_DEVICES=` empty.

Explicit `add_argument` calls in `meson.py:1248-1260` cover only `--ip`, `--port`, `--debug_from`,
`--detect_anomaly`, `--warmup`, `--use_wandb`, `--test_iterations`, `--save_iterations`, `--quiet`,
`--load_iter`, `--checkpoint_iterations`, `--start_checkpoint`, `--gpu`. Everything else is
generated from the attribute loop at `arguments/__init__.py:20-38`.

Relevant defaults:

| flag | default | line |
|---|---|---|
| `--target_size` | `15` MB | `arguments/__init__.py:82` |
| `--percent` | `0.6` | `arguments/__init__.py:81` |
| `--fluc_percent` | `0.05` | `arguments/__init__.py:100` |
| `--bit_upper_bound` | `16` | `arguments/__init__.py:88` |
| `--bit_lower_bound` | `1` | `arguments/__init__.py:89` |
| `--dis_type` | `3` | `arguments/__init__.py:98` |
| `--use_search_prune` | `False` | `arguments/__init__.py:101` |
| `--use_torchac` | `False` | `arguments/__init__.py:102` |
| `--n_block` | `20` | `arguments/__init__.py:115` |
| `--iterations` | `30_000` | `arguments/__init__.py:121` |
| `--prune_iterations` | `1000` | `arguments/__init__.py:122` |
| `--octree_iterations` | `1000` | `arguments/__init__.py:123` |
| `--resolution` | `-1` | `arguments/__init__.py:61` |

## d. The solver

**PuLP with the CBC backend.**

- Import, `meson.py:36`:
  `from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, LpStatus, PULP_CBC_CMD`
  (identical line at `render.py:32`).
- Solve calls: `meson.py:287` and `meson.py:329`,
  `prob.solve(PULP_CBC_CMD(timeLimit=time_limit, msg=False))`. A third, dead call is at
  `render.py:89`.
- `LpStatus` is imported and never used, so infeasibility is not checked.

Two nested one-hot bit-allocation ILPs, both `LpMinimize` on a distortion proxy under a size
constraint.

- Coarse, `solve_coarse_ilp_pulp` at `meson.py:259-301`. Binaries `x{i}` over
  (channel, bit-width) pairs (`meson.py:267-270`). Size constraint `meson.py:273`, one-hot per
  channel `meson.py:275-277`, objective `meson.py:279`
  (`prob += sum([(variable[f"x{i}"]) * L[i] for i in range(num_variable)])`), warm start
  `meson.py:281-284`, time limit 30 s (`meson.py:639`).
- Fine, `solve_channel_ilp_pulp` at `meson.py:304-343`, one problem per channel in a
  `multiprocessing.Pool` (`meson.py:717-718`). Binaries over (block, bit-width), size constraint
  `meson.py:311`, one-hot per block `meson.py:313-315`, objective `meson.py:317`, time limit 50 s
  (`meson.py:692`).

**The analytic size model is corrected against the real encode in an outer loop of at most four
rounds**: coarse `meson.py:625-676`, fine `meson.py:695-757`. The update, `meson.py:667-670` and
`:747-750`:

```python
            delta_qbit = delta_qbit_limit / delta_actual_size * (model_size_limit - actual_size)
            if delta_qbit > actual_size:
                delta_qbit = model_size_limit - actual_size
            qbit_size_limit = qbit_size_limit + delta_qbit
```

Exit test, `meson.py:672` and `:753`:

```python
            if abs(actual_size - model_size_limit) / model_size_limit < dataset.fluc_percent:
                break
```

So `--fluc_percent` is the **relative size tolerance**, default 0.05, that is 5 %.
This outer loop is itself a closed-loop rate controller driven by real encodes, worth reading
before designing the HAC++ one.

## e. What it writes to disk

With `M` the `-m` / `--model_path` directory:

| path | written at |
|---|---|
| `M/cfg_args` | `meson.py:910-911` |
| `M/outputs.log` | `meson.py:1229` |
| `M/ingredients/A.npy` | `meson.py:596-597` |
| `M/ingredients/final_qbits.npy` | `meson.py:765` |
| `M/zip/iteration_<n>/bins/` | `scene/gaussian_model.py:719-721` |
| `M/zip/iteration_<n>/bins/fe.npz` | `scene/gaussian_model.py:818-825`, `:894-899`, `:902-910` |
| `M/zip/iteration_<n>/bins/xyz_gpcc.npz` | `scene/gaussian_model.py:814-816`, `:890-892` |
| `M/zip/iteration_<n>/bins/c{c}_b{b}.b` (torchac path) | `scene/gaussian_model.py:811-812` |
| `M/zip/iteration_<n>/bins/*.pt` (the MLPs) | `scene/gaussian_model.py:912` |
| **`M/zip/iteration_<n>/bins.zip`** (the deliverable) | `scene/gaussian_model.py:914-915`, `os.system(f'zip -jq {bin_zip_path} {bin_dir}/*')` |
| `M/test/ours_<iter>/renders/00000.png`, `.../gt/`, `.../errors/` | `meson.py:999-1001`, `:1037-1039` |
| `M/test/ours_<iter>/per_view_count.json` | `meson.py:1050-1051` |
| `M/results.json` | `meson.py:1208-1209` |
| `M/per_view.json` | `meson.py:1210-1211` |
| `outputs/0_exp_data/<log>/<logdir>.csv` (`--csv_path`) | header `meson.py:363-367`, row `meson.py:1219-1222` |

CSV header, `meson.py:366`: `['name', 'iteration', 'percent', 'fluc', 'psnr', 'ssim', 'lpips', 'size']`.

## f. How the achieved size is printed

Ground truth is `os.path.getsize` on `bins.zip`.

Inside the search loop, `scene/gaussian_model.py:916` and `:921`:

```python
        zip_file_size = os.path.getsize(bin_zip_path)
        ...
        self.logger.info("saving time: {}, size: {} MB".format(time.time() - st_time, zip_file_size / 1024 / 1024))
```

Fed back as bits by `scene.save_compressed(0) * 8` at `meson.py:609-610`, `:649`, `:732`, and logged
at `meson.py:671` and `meson.py:752`.

Final reported size in `evaluate`, `meson.py:1170`:

```python
        bytes_size = os.path.getsize(os.path.join(scene_dir, 'zip', f'iteration_{cur_iter}', 'bins.zip'))
```

printed at `meson.py:1175-1176`:

```python
        logger.info("  SIZE: \033[1;35m{:>12.7f}\033[0m".format(bytes_size / 1024 / 1024))
        logger.info("  SIZE (bytes): \033[1;35m{:>12.7f}\033[0m".format(bytes_size))
```

and stored in `results.json` at `meson.py:1201-1202` as `"SIZE"` (MiB) and `"SIZE (bytes)"`.
The target for comparison is logged at `meson.py:617`. Other size probes:
`scene/gaussian_model.py:994-996` (τ search) and `render.py:291-292`.

## g. Optional fine-tuning

**Not a separate command.** Fine-tuning is a phase of the same `meson.py` run, after the ILP
allocation is fixed by `gaussians.init_bqa(final_qbits)` at `meson.py:772-774`.

| phase | loop | flag | default | declared |
|---|---|---|---|---|
| prune warm-up | `meson.py:438` | `--prune_iterations` | `1000` | `arguments/__init__.py:122` |
| octree warm-up | `meson.py:527` | `--octree_iterations` | `1000` | `arguments/__init__.py:123` |
| post-compression fine-tune | `meson.py:812` | `--iterations` | `30_000` | `arguments/__init__.py:121` |

The fine-tune loop re-saves the compressed model at every test and save iteration
(`meson.py:872-874`), and the best-PSNR iteration is tracked at `meson.py:974-978` and returned at
`meson.py:897`. The shell scripts pass 4000 or 6000 for `--iterations` and bind the same value to
all seven `*_lr_max_steps` flags (`sizegs.sh:29-36`).

## h. Own 3DGS training code

**No.** `find . -name "train*.py"` returns nothing. The root Python files are `convert.py`,
`meson.py`, `metrics.py`, `raht_torch.py`, `render.py`, `render_utils.py`.
`meson.py` is a fine-tuner that requires an existing Scaffold-GS checkpoint, and only reaches
`create_from_pcd` (`scene/__init__.py:118`) when `load_iteration is None`, which no script does.
The README directs the user to download a pretrained Scaffold-GS (`readme.md:76`).
The only data-preparation entry point is the stock Inria COLMAP converter `convert.py`
(flags at `convert.py:19-25`), for which the README gives no command line.
**Training entry point and README training command: not found.**

## i. Evaluation

- **The function is `evaluate`, `meson.py:1123`**, called at `meson.py:1335`. Metric loop,
  `meson.py:1160-1163`:

```python
    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
```

  Images are re-read from disk by `readImages` (`meson.py:1110`). A second in-line computation is
  in `render_set` (`meson.py:998`, metrics `meson.py:1040-1043`), and a third in `render.py:212`.
- `psnr` at `utils/image_utils.py:17-19`, `ssim` at `utils/loss_utils.py:33` with `window_size=11`,
  `lpips_fn = lpips.LPIPS(net='vgg').to('cuda')` at `meson.py:58`. **LPIPS-VGG.** The bundled
  `lpipsPyTorch/` (default AlexNet, `lpipsPyTorch/__init__.py:8`) is not used, its import is
  commented out at `metrics.py:18`.
- **Every-8th-image rule**: `scene/dataset_readers.py:149`
  (`def readColmapSceneInfo(path, images, eval, lod, llffhold=8):`) and
  `scene/dataset_readers.py:177-178`:

```python
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
```

  guarded by `if eval:` at `:165` and `if lod>0:` at `:166`. Every script passes `--eval` and
  `--lod 0`, so this branch is taken. Cameras are sorted by image name at
  `scene/dataset_readers.py:163`.
- **Resize rule**: `utils/camera_utils.py:19-41`, identical to HAC++. `--resolution` default `-1`
  (`arguments/__init__.py:61`) caps width at 1600 px.
- **Output file names**: `results.json` (`meson.py:1208`), `per_view.json` (`meson.py:1210`),
  `per_view_count.json` (`meson.py:1050`), plus the CSV row (`meson.py:1219-1222`). The standalone
  `metrics.py` writes the same two names at `metrics.py:89` and `metrics.py:91`.
- **README evaluation command: not found.** `readme.md:109-119` only points at the four dataset
  shell scripts. Evaluation happens implicitly at `meson.py:1329-1336`. The standalone evaluator
  would be `python metrics.py -m <model_path>` from `metrics.py:103`, but no README line says so.
  `render.sh` is undocumented, its usage example is a comment at `render.sh:28`.

## j. Seed and non-determinism

- **No `--seed` flag.** Seeds are hard-coded in `safe_state`, `utils/general_utils.py:130-133`:

```python
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
```

  called at `meson.py:1300` and `render.py:364`.
- No `torch.cuda.manual_seed`, no cuDNN determinism flags.
- Extra non-determinism specific to this repo: the CBC `timeLimit` (30 s at `meson.py:639`, 50 s at
  `meson.py:692`) makes the ILP solution wall-clock dependent, and the fine stage fans out over a
  `multiprocessing.Pool()` (`meson.py:717`).
- The `--gpu` flag (`meson.py:1260`) is parsed but its handler is commented out
  (`meson.py:1275-1278`), so device choice goes through `CUDA_VISIBLE_DEVICES` (`sizegs.sh:21`)
  together with the pinned `cuda:0` at `utils/general_utils.py:133`.
- **README note on non-determinism: not found.**

## k. Per-scene hyperparameters

**No config files exist.** No `configs/` directory, no yaml, json, toml, cfg or ini outside
`SIBR_viewers` and the GLM CI files. Per-scene values are positional arguments hard-coded in the
launcher shell scripts, in the order
`<name> <iter> <prune_iter> <octree_iter> <voxel> <log> <block> <target_size_MB> <device> [<prune_per> <fluc_percent>]`.

Target sizes in MB, from `size_mip_360.sh:25-88` and `size_ac_mip_360.sh:25-89`:
bicycle 33, garden 28, stump 32, bonsai 16, counter 11, kitchen 14, room 8, flowers 21,
treehill 26, train 7, truck 9, drjohnson 4, playroom 4.
`size_tandt.sh:3` and `:9` instead use train 10 and truck 12 with `n_block` 100, and `size_db.sh:3`
and `:9` use drjohnson 8 and playroom 8. The budgets are therefore **not consistent between the
scripts**.

A commented block at `size_mip_360.sh:6-23` lists an alternative per-scene voxel table
(bicycle 0.0001, bonsai 0.00008, counter 0.00003, garden 0.00003, kitchen 0.0005, room 0.00003,
stump 0.00003, flowers 0.0001, treehill 0.0001) that the active lines do not use.

τ, the prune ratio, is `--percent` (default 0.6, `arguments/__init__.py:81`), passed as `0.5` in
`size_ac_mip_360.sh`. With `--use_search_prune` (only `sizegs_auto_tau.sh:52`) it is binary-searched
by `search_tau` (`scene/gaussian_model.py:927-938`) against `target_size × 1024 × 1024` bytes with a
±20 % acceptance band (`:933`) and a 0.05 bisection tolerance (`:930`).

**README note on per-scene hyperparameters: not found.** `readme.md:106` only mentions fixing paths.

## l. Checkpoint format and path pattern

Three formats.

1. **Compressed deliverable.** `<model_path>/zip/iteration_<n>/bins.zip`.
   Save: `GaussianModel.save_compressed`, `scene/gaussian_model.py:716`, wrapper
   `Scene.save_compressed`, `scene/__init__.py:128-138` (which `shutil.rmtree`s the target first,
   `:131-132`). Load: `GaussianModel.load_compressed`, `scene/gaussian_model.py:1018`, entered from
   `scene/__init__.py:107`.
2. **Uncompressed Scaffold-GS checkpoint, the input format.**
   `<model_path>/point_cloud/iteration_<n>/point_cloud.ply` plus the MLP `.pt` files.
   Save: `Scene.save`, `scene/__init__.py:123-126`, using `GaussianModel.save_ply`
   (`scene/gaussian_model.py:555`) and `save_mlp_checkpoints` (`scene/gaussian_model.py:2170`).
   **`Scene.save` is never called** from `meson.py` or `render.py`.
   Load: `load_ply_sparse_gaussian` (`scene/gaussian_model.py:1831`) and `load_mlp_checkpoints`
   (`scene/gaussian_model.py:2226`, `torch.jit.load`), from `scene/__init__.py:109-115`.
   `searchForMaxIteration` at `scene/__init__.py:52` scans `zip/`, not `point_cloud/`.
3. **Optimizer checkpoint.** `<model_path>/chkpnt<iteration>.pth`, `meson.py:889`, only when
   `--checkpoint_iterations` is given (default `[]`, `meson.py:1258`).
   `capture` at `scene/gaussian_model.py:199-211` returns 10 elements while `restore` at
   `scene/gaussian_model.py:213-227` unpacks 11, the same mismatch as HAC++. `restore` is never
   called: `--start_checkpoint` (`meson.py:1259`) is passed into `training(...)` at `meson.py:1307`
   and the parameter is unused in the body.

---

# 3. MesonGS++ (`mesongs_plus`)

Repo root: `repos/mesongs_plus/`. Paths below are relative to that root.
The package is named `splatwizard` and MesonGS++ is one model in its zoo.

## a. Licence and vendored rasterizer

Licence file: `LICENSE.md`. First 5 lines verbatim (`LICENSE.md:1-5`):

```
MIT License

Copyright (c) 2017 

Permission is hereby granted, free of charge, to any person obtaining a copy
```

The copyright holder field is empty (`LICENSE.md:3`), and the year 2017 predates 3DGS.

Vendored Inria rasterizer: **yes, as source, not a zip**, at
`splatwizard/_cmod/rasterizer/diff_gaussian_rasterization/`, whose
`LICENSE.md:1-4` begins `Gaussian-Splatting License`. Two further nested copies live at
`splatwizard/_cmod/rasterizer/indexed_gs/diff_gaussian_rasterization/` and
`splatwizard/_cmod/rasterizer/meson_gs/diff_gaussian_rasterization/`. A further nine rasterizer
variants ship under `splatwizard/_cmod/rasterizer/`: `accel_gs`, `compress`, `flashgs`, `gs_dr_aa`,
`pup_fisher`, `speedy_splat`, `speedy_tcgs`, `surfel_gs`, `trim3dgs`. Inria-derived Python carries
the Inria header, for example `splatwizard/metrics/loss_utils.py:1-10`.

**Zip archives: none.** `find . -name '*.zip' -not -path './.git/*'` returns nothing. Nothing needs
unpacking. There is also **no `.gitmodules`**, so no git submodules. Third-party C++ dependencies
are vendored under `splatwizard/_cmod/third_party/` (`cutlass`, `fmt`, `glm`, `json`, `pcg32`,
`pybind11_json`, `stbi`).

## b. Installation

`README.md:26-32`, verbatim:

```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r requirements.txt
pip install -e .
```

`README.md:40-44`, external dependency, verbatim:

```bash
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13 && mkdir build && cd build && cmake .. && make -j
export TMC3_PATH=$(pwd)/tmc3/tmc3
```

Versions named: torch 2.4.0+cu121, torchvision 0.19.0, CUDA cu121 (`README.md:27-29`).
Python: **no version in the README**, only `python_requires=">=3.7.13"` at `setup.py:133`.
`pyproject.toml:1-3` sets the build backend only, with no `[project]` table.
`requirements.txt` has exactly one pin, `vector-quantize-pytorch==1.22.0` (`requirements.txt:13`).
There is **no conda env file**, but all four shell scripts hard-code `conda activate compressgs`
(`scripts/compress_single_scene.sh:38-39`, `scripts/eval_mesongs_plus_360.sh:16-17`,
`scripts/eval_mesongs_plus_tandt.sh:8-9`, `scripts/eval_mesongs_plus_db.sh:8-9`).

`pip install -e .` builds 18 CUDA extensions through `setup.py:151-170`, driven by
`export_extension()` (`setup.py:31-64`), with `cmdclass={'build_ext': BuildExtension}`
(`setup.py:171-173`). The rasterizer is entry `setup.py:158`.

Entropy coders, several, in different code paths:

| dependency | role | file:line |
|---|---|---|
| `constriction` | the MesonGS++ range coder (RAHT coefficients, codebook, scaling) | declared `requirements.txt:16`, imported `splatwizard/model_zoo/mesongs_plus/laplace_codec.py:14`, encoder at `:119`, `:129`, `:151` |
| `constriction` | VQ index coding | `splatwizard/model_zoo/mesongs_plus/ntk_codec.py:20` (guarded by try/except), encoder `:94` |
| `torchac` | ContextGS baseline only | declared `requirements.txt:15`, `setup.py:115`, imported `splatwizard/model_zoo/contextgs/utils.py:10` |
| `dahuffman` | Compact3DGS baseline only | `requirements.txt:14`, `setup.py:114`, imported `splatwizard/model_zoo/compact3dgs/model.py:12-13` |
| custom CUDA arithmetic coder | built as `_arithmetic` | `setup.py:152`, source `splatwizard/_cmod/arithmetic/`, imported `splatwizard/compression/entropy_codec.py:5` |
| GPCC / TMC13 `tmc3` binary | octree geometry | `splatwizard/model_zoo/mesongs_plus/gpcc_codec.py:149-158` encode, `:226-235` decode, resolution `_find_tmc3()` at `:39-60` |

**So the MesonGS++ path uses `constriction`, not torchac.** No draco: the `oct_compressed.drc`
name at `gpcc_codec.py:145` and `:220` is a misleading temp filename for a TMC3 stream.

`tmc3` resolution order, `splatwizard/model_zoo/mesongs_plus/gpcc_codec.py:39-60`: `$TMC3_PATH`
(the literal `disabled` turns GPCC off, `:44-45`), then
`<repo>/../mpeg-pcc-tmc13/build/tmc3/tmc3` (`_DEFAULT_TMC3_PATH`, `:31-33`), then
`shutil.which('tmc3')`. GPCC configs are `cfgs/lossless_encoder.cfg` (`gpcc_codec.py:130`) and
`cfgs/decoder.cfg` (`gpcc_codec.py:207`). If tmc3 is absent the octree **silently** falls back to
`np.savez_compressed` (`model.py:2637-2638`, availability check `gpcc_codec.py:274-276`), which
changes the achieved size.

**Two undeclared dependencies that will break a clean install.**

1. **`pulp` is imported but not declared.** Import at
   `splatwizard/model_zoo/mesongs_plus/qbit_search_tool.py:25`, and `grep -n 'pulp'` over
   `requirements.txt`, `setup.py` and `pyproject.toml` finds nothing. Add `pip install pulp`.
2. The `zip` CLI binary, used at encode time, `splatwizard/model_zoo/mesongs_plus/model.py:2943`,
   `:2224`, and `splatwizard/model_zoo/mesongs_plus/qbit_search_tool.py:358`. Decode uses Python
   `zipfile` (`model.py:2970`).

## c. Compression command and byte budget

**Budget flag: `--size_limit_mb`, unit MB (mebibytes). Default `100`.**

- Declared as a `simple-parsing` dataclass field, not `argparse.add_argument`, at
  `splatwizard/model_zoo/mesongs_plus/config.py:50`: `    size_limit_mb: float = 100`
- Parser built at `splatwizard/scripts/eval.py:76-80`.
- Consumed at `splatwizard/model_zoo/mesongs_plus/model.py:350`.
- Converted to bits, `splatwizard/model_zoo/mesongs_plus/qbit_search_tool.py:496`:
  `model_size_limit = size_limit_mb * 1024 * 1024 * 8`
- Converted to bytes, `splatwizard/model_zoo/mesongs_plus/model.py:1798`:
  `size_limit_bytes = self.size_limit_mb * 1024 * 1024`
- **A `size_limit_mb` key in the scene YAML silently overrides the CLI value**,
  `splatwizard/model_zoo/mesongs_plus/model.py:399-400`, with no CLI-wins guard (unlike `percent`,
  `n_block` and `codebook_size` at `model.py:373-378`). The shipped YAMLs do not set it.
- A related list flag for RD sweeps is `rd_curve_size_limits: List[float]` in MB,
  `splatwizard/model_zoo/mesongs_plus/config.py:61`, read at
  `splatwizard/scripts/eval_rd_curve.py:43-45`.

**Input model: a vanilla Inria 3DGS `point_cloud.ply`, not Scaffold-GS.**
`README.md:115` states it: "`INIT_CHECKPOINT` — a pretrained 3DGS `point_cloud.ply` (from the
official 3DGS training pipeline)". Default in the script,
`scripts/compress_single_scene.sh:43`, and the dataset-script layout,
`scripts/eval_mesongs_plus_360.sh:49`:

```
INIT_CHECKPOINT="$INIT_CKPT_ROOT/$DS/$SCENE/baseline/3dgs/point_cloud/iteration_30000/point_cloud.ply"
```

Loading goes through `splatwizard/scripts/train_multi_prune.py:193-195` to `GaussianModel.load`
(`splatwizard/modules/gaussian_model.py:466-484`), ply branch `:476-478`, `load_ply` at `:382-431`.
It asserts full SH degree 3 at `splatwizard/modules/gaussian_model.py:400`:
`assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3`.
Scene data must be COLMAP (`<source_path>/sparse/0/` plus `<source_path>/<images>/`,
`splatwizard/data_loader/dataset_readers.py:170-178`, `:190-196`) or Blender
(`splatwizard/scene/__init__.py:166-168`). README at `README.md:114`.

**Compression is two stages.** Stage 1 prunes, builds the octree, does the VQ and saves a
checkpoint. Stage 2 runs the size-targeted encode, decode and evaluation.

Stage 1, verbatim `scripts/compress_single_scene.sh:108-131`:

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) CUDA_VISIBLE_DEVICES=0 python splatwizard/scripts/train_multi_prune.py \
    --source_path "$SOURCE_PATH" \
    --yaml_path "$MIN_YAML" \
    --model mesongs_plus \
    --optim mesongs_plus \
    --force_setup \
    --iterations 1 \
    --checkpoint_iterations 199 399 599 1999 2999 3999 4999 5999 6999 7999 \
    --checkpoint_type pth \
    --final_checkpoint pth \
    --scene_imp "$TAG" \
    --images "$IMAGES" \
    --use_quat \
    --eval_freq 1 \
    --n_block 80 \
    --codebook_size 4096 \
    --num_bits 16 \
    --raht True \
    --use_indexed True \
    --sh_keep_threshold -1 \
    --sh_keep_topk 1000000 \
    --init_checkpoint "$INIT_CHECKPOINT" \
    --pruning_rates_list 0.2 0.4 \
    --output_dir_template "$TRAIN_DIR_TMPL"
```

Stage 2, the actual byte-budget encode, verbatim `scripts/compress_single_scene.sh:153-176`:

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) CUDA_VISIBLE_DEVICES=0 python splatwizard/scripts/eval.py \
    --source_path "$SOURCE_PATH" \
    --output_dir "$EVAL_DIR" \
    --yaml_path "$MIN_YAML" \
    --model mesongs_plus \
    --optim mesongs_plus \
    --eval_mode ENCODE_DECODE \
    --checkpoint "$CHECKPOINT" \
    --force_setup \
    --scene_imp "$TAG" \
    --images "$IMAGES" \
    --use_quat False \
    --size_limit_mb 20 \
    --n_block 80 \
    --num_bits 16 \
    --percent 0.2 \
    --codebook_size 4096 \
    --raht True \
    --use_indexed True \
    --sh_keep_threshold -1 \
    --sh_keep_topk 1000000 \
    --pruning_rate -1.0 \
    --save_bitstream \
    --save_rendered_image
```

`--percent` in stage 2 selects which trained checkpoint to load, and `--pruning_rate -1.0` disables
eval-time re-pruning (`PRUNING_RATE_PLACEHOLDER=-1.0` at
`scripts/compress_single_scene.sh:66`, semantics at
`splatwizard/model_zoo/mesongs_plus/config.py:62-65` and `model.py:2587-2590`).
The checkpoint used is `$TRAIN_DIR/checkpoints/ckpt1.pth`
(`scripts/compress_single_scene.sh:139`).

README-level commands, `README.md:117-119`, verbatim:

```bash
# minimal: use built-in defaults (counter scene, rates=[0.2, 0.4], size=20 MB)
bash scripts/compress_single_scene.sh
```

and `README.md:86-90`, verbatim:

```bash
bash scripts/eval_mesongs_plus_360.sh     # Mip-NeRF 360
bash scripts/eval_mesongs_plus_tandt.sh   # Tanks and Temples
bash scripts/eval_mesongs_plus_db.sh      # Deep Blending
```

Environment overrides for the single-scene script, `scripts/compress_single_scene.sh:42-69`:
`SOURCE_PATH`, `INIT_CHECKPOINT`, `PRUNING_RATES` (default `"0.2 0.4"`), `SIZE_LIMIT_MB`
(**default `20`**, `:50`), `CUDA_DEVICE` (`0`), `IMAGES` (`images`), `OUTPUT_ROOT`
(`outputs_single`), `TAG`, `N_BLOCK` (`80`), `CODEBOOK_SIZE` (`4096`), `NUM_BITS` (`16`),
`RAHT` (`True`), `USE_INDEXED` (`True`), `SH_KEEP_THRESHOLD` (`-1`), `SH_KEEP_TOPK` (`1000000`),
`OCTREE_DEPTH` (`19`), `TMC3_PATH`.
The script auto-generates a minimal YAML at `scripts/compress_single_scene.sh:79-87`.

Dataset scripts hold author-specific absolute defaults, for example
`scripts/eval_mesongs_plus_360.sh:36-38` (`DATA_ROOT`, `INIT_CKPT_ROOT`, `TMC3_PATH`), scene list
at `:31`, hyper-parameters at `:19-29`.

Configs under `cfgs/`: `cfgs/decoder.cfg` and `cfgs/lossless_encoder.cfg` are GPCC configs, and
`cfgs/mesongs/c1/<scene>.yaml` holds 13 scene files. Every one has the same five leading keys on
lines 1-5 (`n_block`, `cb`, `depth`, `prune`, `finetune_lr_scale`), then `pruning_rates` and
`rd_curve_size_limits`. Table in section k.

**Three script bugs to plan around.**

1. `scripts/compress_single_scene.sh` builds `$SUMMARY_FILE` (`:134-135`, `:144`, `:186`, `:188`)
   and never prints it, despite `README.md:129-130`. Read `$EVAL_DIR/results.json` instead.
2. `scripts/eval_mesongs_plus_db.sh:22-23` assigns `SCENES` twice, so `playroom` is dropped:
   `SCENES=('drjohnson' 'playroom')` then `SCENES=('drjohnson')`.
3. `--use_quat` is passed in `scripts/compress_single_scene.sh:120` and
   `scripts/eval_mesongs_plus_360.sh:68` but omitted in the tandt and db stage-1 commands. It
   changes the RAHT channel count from 8 to 7
   (`splatwizard/model_zoo/mesongs_plus/qbit_search_tool.py:399-407`).

## d. The solver

**PuLP with the CBC backend.**

- Import, `splatwizard/model_zoo/mesongs_plus/qbit_search_tool.py:25`:
  `from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, LpStatus, PULP_CBC_CMD`
- Solve, `splatwizard/model_zoo/mesongs_plus/qbit_search_tool.py:200`:
  `prob.solve(PULP_CBC_CMD(timeLimit=50, msg=True, warmStart=use_warm_start))`
- `LpStatus` is imported and never used.

Problem construction, `search_one_round()` at `qbit_search_tool.py:132-225`. Variables are
`C × B × n_bit` binaries (`:149-150`), one per (channel, block, bit-width), where `C` is the channel
count (7 or 8 RAHT channels plus 3 scaling) and `B` is `n_block`.
Problem `LpProblem("Model_Size", LpMinimize)` at `:176`. Size constraint at `:177-189` through
`qbit_channel_size_estimator` (`:101-130`). One-hot per (channel, block) at `:191-196`. Objective at
`:198`, `prob += sum(x[i] * A_flat[i])`, minimising total quantisation distortion, with `A` the
rate-distortion cost tensor from `get_ddrf()` (`:66-99`), squared error by default
(`dis_type=2`, `:78-79`). Solution decoded to a `qbits[C, B]` array at `:206-224`.

The bit budget subtracts measured fixed overhead, `qbit_search_tool.py:542-543`:

```python
    delta_bits = model_size_limit - save_size_bytes * 8 
    qbit_size_limit = 8 * rf.shape[0] * (n_rf_channels + 3) + delta_bits
```

and is refined over `n_round` outer rounds by a secant-style update at `:626-629`, stopping when
the achieved size is within `fluc_percent` of the target (default `0.01`, that is 1 %, `:381`,
test at `:632-633`). Negative-budget guard at `:546-551`.
This is a second worked example of a closed loop driven by real encodes.

## e. What it writes to disk

Per eval run, with `output_dir` the `--output_dir`:

| file | file:line |
|---|---|
| `encoded.bin` (the bitstream, needs `--save_bitstream`) | `splatwizard/pipeline/eval_model.py:45`, reopened for decode `:59` |
| `results.json` | `splatwizard/pipeline/eval_model.py:107-108` |
| `per_image_metrics.json` | `splatwizard/pipeline/evaluation.py:129-131` |
| `eval.log` | `splatwizard/scripts/eval.py:44` |
| `qbits.npz` (the ILP allocation) | `splatwizard/model_zoo/mesongs_plus/model.py:2930-2938` |
| `render_results/00000.png`, … | dir `splatwizard/scripts/eval.py:37-38`, name `splatwizard/pipeline/evaluation.py:96` |

`encoded.bin` is itself a zip. It is assembled in a `TemporaryDirectory`
(`splatwizard/model_zoo/mesongs_plus/model.py:2615-2618`), zipped (`:2941-2943`) and streamed into
the caller's handle in 8 KB chunks (`:2952-2959`). Members under `bins/`: `oct.gpcc` or `oct.npz`
(`:2631-2632`, `:2636`, `:2638`), `ntk.bin` (`:2658-2659`), `cb_q.bin` (`:2684-2685`),
`cb_meta.npz` (`:2688-2696`), `um.npz` (`:2705`), `orgb.bin` (`:2883-2884`), `orgb_dc.npz`
(`:2885`), `ct.bin` (`:2920-2921`), `t.npz` (`:2926`). Decode extracts the same set
(`:2966-2971`).

RD-curve runs add `rd_curve/rd_results.json` (`splatwizard/pipeline/rd_curve.py:390-392`),
`rd_curve/rd_all_candidates.json` (`:397-399`), `rd_curve/rd_curve.png` (`:428-429`), and one
sub-directory per operating point (`:286-294`, `:367`, `:369`).

Stage-1 runs write `<rate_output_dir>/checkpoints/ckpt<iteration>.pth`
(`splatwizard/scripts/train_multi_prune.py:48-49`, `splatwizard/modules/gaussian_model.py:455`) and
`<rate_output_dir>/output.log` (`train_multi_prune.py:51`).

## f. How the achieved size is printed

The number that lands in `results.json` as `total_bytes` is the writer offset after the bitstream
is streamed out:

- `splatwizard/pipeline/eval_model.py:54`, `total_bytes = f.tell()` (with `--save_bitstream`)
- `splatwizard/pipeline/eval_model.py:75`, `total_bytes = tmp_file.tell()` (without)
- `splatwizard/pipeline/eval_model.py:89`, `total_bytes = os.path.getsize(ppl.bitstream)`
  (`EvalMode.DECODE`)

Logged at `splatwizard/modules/gaussian_model.py:131-135`:

```python
        if eval_pack.total_bytes != 0:
            logger.info(wrap_str(
                f'total {eval_pack.total_bytes} bytes ({eval_pack.total_bytes / 1024 / 1024} MB)',
                f'encode {eval_pack.encode_time} decode {eval_pack.decode_time}'
            ))
```

Stored into the result dict at `splatwizard/modules/gaussian_model.py:137-141`.
Printed to stdout inside the encoder at
`splatwizard/model_zoo/mesongs_plus/model.py:2946-2950`:

```python
                zip_file_size = os.path.getsize(bin_zip_path)

                print('final sum:', zip_file_size , 'B')
                print('final sum:', zip_file_size / 1024, 'KB')
                print('final sum:', zip_file_size / 1024 / 1024, 'MB')
```

Derived MB in the RD pipeline at `splatwizard/pipeline/rd_curve.py:59-63`, and in the shell script
at `scripts/compress_single_scene.sh:184` (`d.get('total_bytes', 0)/1048576`).
Intermediate probes during the search: `qbit_search_tool.py:359` and
`model.py:2226-2227`.

## g. Optional fine-tuning

**No separate fine-tune command.** Fine-tuning is the training loop inside
`splatwizard/scripts/train_multi_prune.py`, run per pruning rate after prune, octree and VQ and
before the checkpoint is saved. Loop `_run_finetune_loop` at
`splatwizard/scripts/train_multi_prune.py:62-119`, invoked at `:275-276`.

**Iteration flag: `--iterations`, default `30_000`** (`splatwizard/config.py:179`).
**All four shipped scripts pass `--iterations 1`**, so fine-tuning is disabled in the released
configuration (`scripts/compress_single_scene.sh:114`, `scripts/eval_mesongs_plus_360.sh:62`,
`scripts/eval_mesongs_plus_tandt.sh:49`, `scripts/eval_mesongs_plus_db.sh:50`). With
`iterations == 1` no backward pass runs (`train_multi_prune.py:93-94`, `:104-105`).

The per-scene `finetune_lr_scale` (YAML line 5, field
`splatwizard/model_zoo/mesongs_plus/config.py:48`, read at `model.py:398`) is applied by
`MesonGSPlus.finetuning_setup` (`model.py:1707-1732`), but **that setup is never registered**: the
scheduler line is commented out at `model.py:1754`, and `train_multi_prune.py:262` calls
`rate_model.training_setup(op)` instead. So `finetune_lr_scale` has no effect on the shipped path.

## h. Own 3DGS training code

**Yes.** `splatwizard/scripts/train.py` (136 lines, main at `:73-133`, dispatch at `:118` and
`:132`), loop at `splatwizard/pipeline/train_model.py:17-135` with densify, prune and checkpointing
(`:122-133`). A vanilla 3DGS model is registered as `"3dgs"` at
`splatwizard/model_zoo/registry.py:39`, parameters at `splatwizard/model_zoo/gs/config.py:5-28`
(SH degree 3, `position_lr_max_steps = 30_000`, `densify_until_iter = 15_000`). Console script
alias `sw-train=splatwizard.scripts.train:main` at `setup.py:146`.

**The README gives no training command line.** Its only mention is `README.md:76-79`, which says
that only the MesonGS++ shell scripts are officially released and that other baselines run
"directly via `splatwizard/scripts/train.py` / `eval.py`". There is no `train_3dgs.sh` in
`scripts/`. The flags exist as `simple-parsing` fields: `--source_path`
(`splatwizard/config.py:62`), `--images` (`:78`), `--resolution` (`:123`), `--output_dir` (`:92`),
`--iterations` (`splatwizard/config.py:179`), `--checkpoint_iterations` (`:95`),
`--checkpoint_type` (`:99`), `--final_checkpoint` (`:105`), `--seed` (`:157`), `--force_setup`
(`:197`), plus the `--model 3dgs --optim 3dgs` subgroup selectors
(`splatwizard/model_zoo/__init__.py:24-32`). `train.py` asserts `pp.eval_mode is None`
(`splatwizard/scripts/train.py:70`).

## i. Evaluation

- **The function is `evaluate`, `splatwizard/pipeline/evaluation.py:19-162`.** Metrics per test
  view at `splatwizard/pipeline/evaluation.py:98-101`:

```python
        l1_loss = l1_func(rendered_image, gt_image)
        ssim_loss = ssim_func(rendered_image, gt_image)
        psnr_val = psnr_func(rendered_image, gt_image)
        lpips_val = lpips_fn(rendered_image, gt_image, normalize=True)
```

- `psnr_func` at `splatwizard/metrics/loss_utils.py:75-77`, `ssim_func` at
  `splatwizard/metrics/loss_utils.py:35-43` with an 11x11 Gaussian window and `_ssim` at `:45-65`.
  The fused SSIM (`union_ssim_func`, `:67-73`) is **not** used in evaluation.
  `lpips_fn = lpips.LPIPS(net='vgg').to('cuda')` at `splatwizard/metrics/loss_utils.py:79`,
  **LPIPS-VGG**, instantiated at module import, which allocates CUDA and downloads VGG weights on
  import. The in-repo `splatwizard/metrics/lpipsPyTorch/` (AlexNet default,
  `__init__.py:8`) is unused here.
- Test cameras from `scene.getTestCameras()` at `splatwizard/pipeline/evaluation.py:64`.
- **The render is quantised to 8 bits before scoring**, `splatwizard/pipeline/evaluation.py:92-93`:

```python
        # Quantize rendered image to 8-bit precision (1/255) for fair comparison with FCGS
        rendered_image = torch.round(render_result.rendered_image.mul(255).clamp(0, 255)) / 255.0
```

  Averaging at `splatwizard/pipeline/evaluation.py:134-140`.
- **Every-8th-image rule**, `splatwizard/data_loader/dataset_readers.py:212-214`:

```python
    if data_mode == DataMode.SPLIT:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
```

  `llffhold` comes from `test_sample_freq`, default 8, `splatwizard/config.py:82`, wired at
  `splatwizard/scene/__init__.py:165` and `:189`. Signature default 8 at
  `splatwizard/data_loader/dataset_readers.py:168`. Cameras sorted by `image_name` at `:197`.
  Split mode default `DataMode.SPLIT` at `splatwizard/config.py:86`.
- **Resize rule**, `splatwizard/scene/dataset.py:32-56`, the same 1600 px width cap as the other two
  repos, with `resolution: int = -1` at `splatwizard/config.py:123`. Warning at
  `splatwizard/scene/__init__.py:122-124`. A duplicate copy of the rule is at
  `splatwizard/scene/camera_utils.py:27-49`. Only `resolution_scales=(1.0,)` is supported
  (`splatwizard/scene/__init__.py:144-145`).
- **Output file names**: `results.json` (`splatwizard/pipeline/eval_model.py:107-108`) with keys
  `L1`, `PSNR`, `SSIM`, `LPIPS`, `Visible_Gaus`, `total_gaus`, `frame_time`, `FPS`
  (`splatwizard/modules/gaussian_model.py:119-128`), plus `total_bytes`, `encode_time`,
  `decode_time` (`:137-141`), plus `peak_memory_allocated_bytes`, `peak_memory_reserved_bytes`
  (`:148-151`), plus `stage_timings` (`splatwizard/pipeline/eval_model.py:105-106`).
  Per-view metrics are in **`per_image_metrics.json`**, not `per_view.json`
  (`splatwizard/pipeline/evaluation.py:129-131`), with keys `index`, `filename`, `psnr`, `ssim`,
  `lpips`, `l1` (`:108-115`).
  RD sweeps add `rd_results.json`, `rd_all_candidates.json`, `rd_curve.png`
  (`splatwizard/pipeline/rd_curve.py:390`, `:397`, `:428`).
- **README evaluation command**: `README.md:86-90`, the three dataset scripts quoted in section c.
  There is no bare `python splatwizard/scripts/eval.py ...` line in the README.

## j. Seed and non-determinism

- **`--seed` exists, default `None`.** `splatwizard/config.py:156-157`:

```python
    # Random seed for reproducibility (None for random initialization)
    seed: int = None
```

- Applied at `splatwizard/scripts/eval.py:88-89`, `splatwizard/scripts/train.py:86-87`,
  `splatwizard/scripts/train_multi_prune.py:148-149`, and twice at
  `splatwizard/scripts/eval_rd_curve.py:23-27`.
- Implementation `safe_state`, `splatwizard/utils/misc.py:11-31`, seeding `random.seed(seed)`,
  `np.random.seed(seed)`, `torch.manual_seed(seed)` at `:29-31`. No
  `torch.cuda.manual_seed_all`, no `torch.use_deterministic_algorithms`, no cuDNN flags on this
  path (other, unused baselines do set them, for example
  `splatwizard/model_zoo/cat_3dgs/model.py:2414`).
- **No shipped script passes `--seed`**, so all released runs are unseeded.
- Extra non-determinism: the CBC `timeLimit=50` (`qbit_search_tool.py:200`) means a timeout returns
  whatever incumbent CBC holds, the k-means VQ (`model.py:926-937`) is unseeded, and each RD point
  runs in a spawned subprocess (`splatwizard/pipeline/rd_curve.py:270`, `:97-102`) that does not
  inherit the parent RNG state.
- **README note on non-determinism: not found.** `README.md:7` and `README.md:76` say "reproducible"
  and "reproduce" without discussing run-to-run variance.

## k. Per-scene hyperparameters

**Config files: `cfgs/mesongs/c1/<scene>.yaml`, 13 files.** README explains them at
`README.md:92-106` and gives the format at `README.md:150-169`.

| file | n_block (:1) | cb (:2) | depth (:3) | prune (:4) | finetune_lr_scale (:5) | rd_curve_size_limits, MB |
|---|---|---|---|---|---|---|
| `bicycle.yaml` | 80 | 2048 | 20 | 0.4 | 0.2 | `[109.2, 95.7, 83.8, 71.5, 62.2]` (:19) |
| `bonsai.yaml` | 80 | 2048 | 19 | 0.4 | 0.1 | `[24.3, 21.3, 18.6, 15.7, 13.5]` (:14) |
| `counter.yaml` | 80 | 2048 | 19 | 0.4 | 0.1 | `[24.3, 21.3, 18.6, 15.9, 13.6]` (:14) |
| `drjohnson.yaml` | 80 | 2048 | 20 | 0.4 | 0.4 | `[61.7, 54.1, 47.4, 40.1, 34.8]` (:14) |
| `flowers.yaml` | 80 | 2048 | 20 | 0.4 | 0.1 | `[72.9, 64.0, 55.8, 47.6, 40.6]` (:14) |
| `garden.yaml` | 80 | 2048 | 20 | 0.4 | 0.1 | `[113.2, 99.0, 85.5, 71.3, 61.0]` (:14) |
| `kitchen.yaml` | 80 | 8192 | 19 | 0.4 | 0.2 | `[39.2, 34.4, 29.7, 24.9, 20.7]` (:14) |
| `playroom.yaml` | 80 | 2048 | 20 | 0.4 | 0.4 | `[49.1, 42.6, 37.0, 31.6, 27.5]` (:14) |
| `room.yaml` | 80 | 2048 | 19 | 0.4 | 0.4 | `[26.5, 23.3, 20.4, 17.3, 15.2]` (:14) |
| `stump.yaml` | 80 | 2048 | 20 | 0.4 | 0.04 | `[101.7, 88.8, 77.5, 65.8, 56.4]` (:14) |
| `train.yaml` | 80 | 4096 | 20 | 0.4 | 0.2 | `[20.1, 17.5, 15.1, 12.8, 10.9]` (:16) |
| `treehill.yaml` | 80 | 2048 | 20 | 0.4 | 0.1 | `[77.6, 67.4, 58.5, 49.6, 42.2]` (:14) |
| `truck.yaml` | 80 | 4096 | 20 | 0.4 | 0.8 | `[42.1, 37.0, 32.5, 28.0, 24.7]` (:14) |

`pruning_rates` is `[0.2, 0.4]` in every file (line 8, or line 10 for `train.yaml`).
The shell scripts override `n_block=80`, `cb=4096` and `num_bits=16` on the CLI, so the YAML `cb`
only takes effect when the CLI value equals the default 2048 (`model.py:373-378`). `depth` and
`finetune_lr_scale` have no CLI override and always come from YAML (`model.py:362`, `:398`).

**Per-scene target sizes: yes**, the five-point `rd_curve_size_limits` lists above, derived from
FCGS compressed sizes per the comments at, for example, `cfgs/mesongs/c1/bicycle.yaml:15-18`.
The single-scene default is `SIZE_LIMIT_MB=20` (`scripts/compress_single_scene.sh:50`,
`README.md:137`).

**Per-scene λ or τ: not found.** No `lambda` or `tau` key exists in any config. The only
occurrences are comments naming FCGS's own lambda sweep
(`cfgs/mesongs/c1/{train,playroom,truck,drjohnson}.yaml:12` or `:14`).

Two README-versus-code mismatches worth noting: `README.md:161` shows `depth: 19` for bicycle while
`cfgs/mesongs/c1/bicycle.yaml:3` says `depth: 20`, and `README.md:143` gives `OCTREE_DEPTH` default
19 while nine of the 13 scene YAMLs use 20.

## l. Checkpoint format and path pattern

Format: a pickled tuple through `torch.save`, extension `.pth`.

Save, `splatwizard/modules/gaussian_model.py:452-464`:

```python
    def save(self, checkpoint_dir, iteration, type_='pth'):
        if type_ == 'pth':
            logger.info("[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((self.capture(), iteration), checkpoint_dir / f"ckpt{iteration}.pth")
        elif type_ == 'ply':
            logger.info("[ITER {}] Saving PLY file".format(iteration))
            # checkpoint_dir.parent point to output_dir,
            point_cloud_path = checkpoint_dir.parent / "point_cloud/iteration_{}".format(iteration)
            point_cloud_path.mkdir(parents=True, exist_ok=True)
            # mkdir_p(os.path.dirname(point_cloud_path))
            self.save_ply(point_cloud_path / "point_cloud.ply")
        else:
            raise NotImplementedError(f'Unsupported checkpoint type: {type_}')
```

Load, `splatwizard/modules/gaussian_model.py:466-484`. It accepts three forms: a directory (picks
the max `point_cloud/iteration_N`, `:470-476`), a `.ply` (`:477-479`), or anything else through
`torch.load(path, weights_only=False)` followed by `self.restore(model_params, opt)` (`:481-482`).

Payload: `MesonGSPlus.capture()` at `splatwizard/model_zoo/mesongs_plus/model.py:1018-1054`, a
30-element tuple carrying `active_sh_degree`, `_xyz`, `_features_dc`, `_features_rest`, `_scaling`,
`_rotation`, `_opacity`, `_cov`, `_euler`, `_feature_indices`, `max_radii2D`, `xyz_gradient_accum`,
`denom`, `optimizer.state_dict()`, `spatial_lr_scale`, `qas.state_dict()`, `reorder`, `res`, `oct`,
`oct_param`, `_keep_q_indices`, `_keep_scales`, `_keep_zero_points`, `_keep_split`, `_num_keep`,
`_keep_mask`, `_vq_indices_for_all`, `_max_num_keep`, `_original_codebook`,
`supports_dynamic_adjustment`, `imp`. `restore` at `model.py:1056` onwards is version tolerant on
tuple length (`:1058-1063`).

Path patterns:

- pth: `<output_dir>/checkpoints/ckpt{iteration}.pth`. With `--iterations 1` this is always
  `ckpt1.pth` (`scripts/compress_single_scene.sh:139`, `splatwizard/pipeline/rd_curve.py:131`,
  `README.md:100`, `splatwizard/model_zoo/mesongs_plus/config.py:90`).
- ply: `<output_dir>/point_cloud/iteration_{iteration}/point_cloud.ply`
  (`splatwizard/modules/gaussian_model.py:459-462`). Writer `save_ply` at `:433-450`, reader
  `load_ply` at `:382-431`, attribute layout `construct_list_of_attributes` at `:365-380`.
- RD checkpoint discovery, `splatwizard/pipeline/rd_curve.py:117-165`, template at `:128-132`:

```
mesongs_plus_{scene}_{config}_quat_train_nb{n_block}_bits{num_bits}_prune{percent}_cb{codebook_size}_topk{sh_keep_topk}_raht{raht}_use_indexed{use_indexed}/checkpoints/ckpt1.pth
```

  searched under three hard-coded bases (`splatwizard/pipeline/rd_curve.py:150-154`):
  `outputs_autotune`, `outputs_jcge`, and an author absolute path
  `/home/gejunchen/Work/2026-1/Projects/compressgs/outputs_jcge`. Overridable by
  `checkpoint_template` in the scene YAML (`splatwizard/scripts/eval_rd_curve.py:52-54`,
  `splatwizard/model_zoo/mesongs_plus/config.py:87-92`).

---

# 4. Cross-repo summary for the plan

| | HAC++ | SizeGS | MesonGS++ |
|---|---|---|---|
| licence | `LICENSE.md`, Inria Gaussian-Splatting | root licence **not found**, Inria only inside `submodules/diff-gaussian-rasterization/LICENSE.md` | `LICENSE.md`, MIT with an empty holder |
| Inria rasterizer | `submodules/diff-gaussian-rasterization.zip` | `submodules/diff-gaussian-rasterization/` (source) | `splatwizard/_cmod/rasterizer/diff_gaussian_rasterization/` (source) |
| zip archives to unpack | 4 (`arithmetic`, `diff-gaussian-rasterization`, `gridencoder`, `simple-knn`) | none | none |
| entropy coder | custom CUDA `arithmetic` extension + GPCC `tmc3` | `torchac` + GPCC `tmc3` | `constriction` + GPCC `tmc3` |
| rate knob | `--lmbda`, default `0.001` (`train.py:585`) | `--target_size` MB, default `15` (`arguments/__init__.py:82`) | `--size_limit_mb` MB, default `100` (`config.py:50`) |
| input model | trains from COLMAP SfM | Scaffold-GS checkpoint | vanilla 3DGS `point_cloud.ply` |
| own 3DGS training code | yes, `train.py` | **no** | yes, `splatwizard/scripts/train.py` |
| solver | none | PuLP + CBC (`meson.py:36`, `:287`, `:329`) | PuLP + CBC (`qbit_search_tool.py:25`, `:200`) |
| deliverable | `<m>/bitstreams/*` | `<m>/zip/iteration_<n>/bins.zip` | `<output_dir>/encoded.bin` |
| size ground truth | summed coder returns + `os.path.getsize` on `xyz_gpcc.npz` (`scene/gaussian_model.py:1228`, `:1363`) | `os.path.getsize(bins.zip)` (`meson.py:1170`) | writer offset `f.tell()` (`eval_model.py:54`) |
| per-view metrics file | `per_view.json` | `per_view.json` | `per_image_metrics.json` |
| test split | every 8th image, `dataset_readers.py:170-171` | every 8th image, `dataset_readers.py:177-178` | every 8th image, `dataset_readers.py:212-214` |
| resize | width capped at 1600 px, `camera_utils.py:25-32` | same, `camera_utils.py:19-41` | same, `scene/dataset.py:32-56` |
| seed | hard-coded 0, no flag (`general_utils.py:130-132`) | hard-coded 0, no flag (`general_utils.py:130-133`) | `--seed`, default `None`, never passed (`config.py:157`) |

Three shared traps for the plan:

1. All three cap image width at 1600 px and split every 8th image, so PSNR is comparable across the
   three only if the same `--resolution` and `--images` are used.
2. All three need the MPEG `tmc3` binary built from
   https://github.com/MPEGGroup/mpeg-pcc-tmc13, but only HAC++ (`README.md:63-67`) and MesonGS++
   (`README.md:40-44`) document it. SizeGS hard-codes it at `utils/compression.py:287`.
3. Both SizeGS and MesonGS++ already implement an outer loop that corrects an analytic size model
   against a real encode (`meson.py:625-676` and `:695-757`,
   `qbit_search_tool.py:626-633`). Read those before designing the HAC++ controller.
