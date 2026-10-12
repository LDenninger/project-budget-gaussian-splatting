# BudgetGS 第一周落地计划

**日期:** 2026-10-08 到 2026-10-14（周四到周三，10-15 留作缓冲）
**作者:** Yuefan Zhou
**状态:** 待执行
**输入:** `docs/02-research-program.md`（研究纲领，K1 到 K5、D1 到 D6）、`references/` 67 篇文献、
导师要求（代码作为 submodule 放进 `workspaces/`，并给他访问权）

---

## 0. 一周后的验收标准

七天后导师能打开一个仓库，clone 下来，跑两条命令，看到三张表。具体是：

| # | 交付物 | 验收标准 |
|---|---|---|
| 1 | `workspaces/budgetgs` 仓库，作为 submodule 注册进项目仓库 | 导师有写权限，`git clone --recurse-submodules` 后目录非空，7 天内至少 7 次 push |
| 2 | `budgetgs` CLI 两条命令 | `budgetgs train` 在集群上训 gsplat 3DGS 并每 1 000 步存 checkpoint；`budgetgs fit` 接受一个字节预算 S★，用真实写出的字节做反馈，落点在 max(1 %, 256 KB) 内 |
| 3 | 字节账本（本周核心） | 对坐标、缩放、旋转、不透明度、SH、量化参数、容器头逐项记账，预测码长与实际序列化文件之差 ≤ 1 %（文件 ≥ 1 MB） |
| 4 | K1 首组数据 | 3 个场景上 encoded bytes 对 N 的曲线、线性拟合 R² 与最坏残差 |
| 5 | 落点表 | 3 场景 × 10 MB，PSNR / SSIM / LPIPS / 实际字节 / 命中误差 ε |
| 6 | 可复现材料 | README 一条命令可复现，devlog 一篇，图三张，导师 viewer 能打开的 checkpoint 一个 |

**本周不做，也不要对外说做了:** augmented-Lagrangian 控制器、budget-aware densification、与 HAC
的对比、0.9 到 1.3 dB 的结论、Tanks and Temples、显存预算与帧时间预算。这些在第 2 到 5 周。

---

## 1. 现状盘点

**已有的:** 研究纲领（含 K1 到 K5 的实验设计）、67 篇文献与逐篇页引笔记、结果转录 CSV、导师的
run 日志器和 viewer（`workspaces/ldtrain`、`workspaces/open-splatting-viewer`）。

**没有的:** 任何代码、conda 环境、数据集、GPU 作业记录、GitHub 凭据（`ssh -T git@github.com` 目前是
Permission denied，机器上没有 `gh`、没有 `.git-credentials`）。

**算力与规矩（CVAI/I9 集群手册 2026-08-21 版）:**

- 账号等级 `stud`：每人 9 GPU、15 作业、优先级 5。`DEADLINE` 与 `DEADLINEBIG` 分区是 `wiss` 专属，
  我们只能提交 `NORMAL`。
- 当前机器 `devcube2` 是工作站：RTX 4090 24 GB、64 线程、251 GB 内存。手册明确工作站 GPU 用于调试
  与测试，重活交给计算节点（node14 是 8× RTX A6000 48 GB，node15 到 18 是 10× A40 48 GB）。
- 存储分区：代码放 home（SSD，只放小的），环境与大数据放 `~/storage/user`（15 TB 可用），作业工作集放
  `/storage/slurm/zyuef`（Ceph，跑完就删），`~/storage/scratch` 每周日 08:00 清理，`~/storage/local`
  是当前主机的本地盘，计算节点上看不到这台机器的它。手册里写的 `~/storage/personal` 在本部署里叫
  `~/storage/user`，这是手册与实际的一处差异。
- 计算节点不下载数据。数据集在工作站下好，放到共享存储。
- 日志目录必须事先存在，`--output` 必须是绝对路径且带 `%j`，脚本要有 `set -euo pipefail`，
  提交前 `sbatch --test-only`。项目仓库自带 `.claude/skills/slurm-tumcluster/`，其中
  `validate_sbatch.py` 是提交前的强制关卡。

---

## 2. 三个设计决策

**决策 A：以账本为中心，而不是以编码器为中心。** 真实算术编码 300 万个高斯在 Python 里是分钟级的，
放不进训练循环，也放不进 90 个 checkpoint 的批量实验。所以本周实现一个"与编码器等价的码长预测器"：
对量化后的属性索引用同一套概率模型算 −∑ log₂ p，得到每个属性、每个块的字节数，加上容器头、
尺度表、块索引等固定开销。编码器只在抽样点上跑到，用来证明账本 ≤ 1 %。账本快，才可能在第二步进
训练循环，这正是研究纲领 D1 里"primal 步用估计、dual 步用真实编码"的雏形。

**决策 B：本周的控制器只有一个执行器，全局块精度。** `fit` 用二分搜索全局量化步长，每一步都真实
写出文件并读回字节数，直到落进 max(1 %, 256 KB)。这是研究纲领里"fast actuator"的骨架，不含 λ、
不含 SH 复杂度、不含 densification。先把"真实字节在环内"这条通路打通，其余执行器每周加一个。

**决策 C：先 3 个场景，先 1 个预算。** Mip-NeRF 360 的 bicycle、garden、room（K1 指定的三个），
预算只做 10 MB。协议要求的 4 数据集 × 3 预算 × 3 种子放第 4 周。

---

## 3. 逐日计划

### Day 1（10-08 周四）环境与身份

目标：七天内随时能被导师看到东西，所以仓库和权限放在第一天，不放在最后。

- [ ] 装 micromamba 到 `~/storage/user/zyuef/micromamba`（不进 home），建环境 `main`：
      python 3.12、torch（cu124）、numpy、scipy、matplotlib、pytest。
- [ ] GitHub 凭据：`ssh-keygen -t ed25519`，公钥加到 GitHub 账号，测试 `ssh -T git@github.com`。
- [ ] 在 GitHub 建空仓库 `budgetgs`，Settings 里把 `LDenninger` 加为 collaborator。
- [ ] 仓库骨架：`pyproject.toml`（hatchling）、`budgetgs/{data,modules,evaluation,utils}/`、
      `slurm/`、`tests/`、按 `readme-conventions.md` 写的 README、`.gitignore`。
- [ ] 在项目仓库注册 submodule：`git submodule add <url> workspaces/budgetgs`，在 `.gitignore` 里
      加 `!/workspaces/budgetgs`，提交并 push（项目仓库约定：直接提交到 master，无 review）。

**验收:** 导师能在 GitHub 上看到仓库并收到 collaborator 邀请，项目仓库里 submodule 条目已提交。

### Day 2（10-09 周五）submodules、数据、第一次集群作业

- [ ] 项目仓库初始化 submodules：至少 `submodules/gsplat`，K1 需要第二编码器时再拉 `submodules/ffsplat`。
- [ ] 在工作站下载 Mip-NeRF 360 到 `~/storage/user/zyuef/datasets/`（先校验 bicycle、garden、room 三个）。
- [ ] 写第一个 `.sbatch`（`slurm/smoke.sbatch`）：`--gres=gpu:1,VRAM:24G`、`--time=00:20:00`、
      `--cpus-per-task=4`、`--mem=32G`、`--constraint=NOTBIG`、
      `--output=/storage/slurm/zyuef/logs/%x-%j.out`，脚本内
      `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate main`，
      跑 `python -c "import torch; print(torch.cuda.get_device_name())"` 与 gsplat 的百步冒烟。
- [ ] 用项目仓库的 skill 走一遍：`validate_sbatch.py` 通过，`sbatch --test-only` 通过，正式提交，
      用 `gather_job.py` 收结果。

**验收:** 集群上一个真实作业 COMPLETED，日志里有 GPU 型号与 gsplat 前向反向都跑通的输出。

### Day 3（10-10 周六）gsplat 基线训练

- [ ] 把 gsplat 训练包成 `budgetgs train`（`parse_args()` + 同名主函数 + `__main__` 两行）。
- [ ] 3 场景 × 30 000 步，每 1 000 步存 checkpoint，种子、配置、commit hash 写进 `runs.jsonl`。
- [ ] 提交作业（3 个独立 job 比 1 个串行 job 更容易被 backfill 挤进空档，但注意 15 作业上限）。
- [ ] 记录 30k 的 PSNR 作为本仓库的基线数字（与研究纲领里 leaderboard 的数字对照，差 0.5 dB 以上
      先查协议而不是先怀疑方法）。

**验收:** 三个场景各有 30 个 checkpoint 与一条 PSNR 曲线，`runs.jsonl` 三行。

### Day 4（10-11 周日）字节账本

- [ ] `budgetgs ledger`：属性分组（坐标、缩放、旋转、不透明度、SH0、SHN）、每组一个块自适应
      类别分布、−∑ log₂ p 逐项累加，再加容器头、每块的 min/max 尺度表、属性表与对齐填充。
- [ ] `budgetgs serialize`：真正写出文件（同一套量化与概率模型的算术编码，或先用 zstd 兜底并把
      差距归因到头部开销）。
- [ ] 测试：小规模（1 万高斯）手算对照，中期（3 个 30k checkpoint）一致性 ≤ 1 %。

**验收:** 三个 checkpoint 上 `|predicted − actual| / actual ≤ 1 %`，差异拆到属性级别可见。

### Day 5（10-12 周一）K1 与落点闭环

- [ ] K1：用账本过一遍 3 场景 × 30 个 checkpoint，算 bytes 对 N 的线性拟合 R² 与最坏残差，
      再抽样 6 个点跑真编码核对账本。按 K1 的判据报告（R² > 0.98 且残差 < 5 % 则字节近似仿射）。
- [ ] `budgetgs fit`：二分全局块精度，每一步真实写出并读回字节，直到 ε ≤ max(1 %, 256 KB)。
- [ ] 落点表：3 场景 × 10 MB，报告 ε、PSNR、SSIM、LPIPS、写盘时间。

**验收:** 三个场景都落进容差，且全程不需要重训、不需要 per-scene λ 搜索。

### Day 6（10-13 周二）结果、展示、文档

- [ ] 三张图：bytes 对 N（含拟合）、落点表、PSNR 对大小。
- [ ] 用 `open-splatting-viewer` 在 devcube2 上打开一个 checkpoint（SSH 隧道到 8080），截图进仓库。
- [ ] 写 devlog（项目仓库 `docs/devlog/`，用 `write-devlog` skill），README 更新到可复现状态。

**验收:** 导师从 README 一条命令能复现落点表；仓库里有一篇 devlog 与三张图。

### Day 7（10-14 周三）汇报与下周计划

- [ ] 给导师发消息：仓库链接、7 天产出清单、已知缺口、第 2 周计划、请求 review。
- [ ] 写第 2 周计划（AL 控制器、SH 复杂度、与 prune+quant 的基线对齐）。
- [ ] 当天不做新实验，只补漏与清理 `/storage/slurm/zyuef`。

---

## 4. 简历承诺 → 兑现时间表

简历上的 5 条是目标状态，不是已完成状态。诚实的时间表如下，第 2 列是本周结束时的真实状态：

| 简历表述 | 本周结束时 | 预计兑现 |
|---|---|---|
| 训练框架接受存储/显存/帧时间预算，无后处理修正 | 只有存储预算，CLI 骨架，落点靠真实编码闭环 | 存储第 2 到 3 周；显存第 5 周；帧时间第 6 周 |
| 精确字节账本，估计与序列化 ≤ 1 % | 完成，并且是本周的核心交付物 | 本周 |
| 比 Gaussian-count 代理的分配效率高 15 % | 未做，K1 的 bytes-N 曲线是它的前置证据 | 第 2 到 3 周（D2 实验） |
| AL 控制器 + 可微 gate + 自适应 SH 复杂度 + 块精度，max(1 %, 256 KB) 容差 | 只有块精度二分落点，容差达标；其余三个执行器未做 | AL 与 gate 第 2 到 3 周；联合版第 3 到 4 周 |
| 10 MB 上比 prune+quant 高 0.9 到 1.3 dB PSNR，与 HAC 相当或更好 | 未做，需要公平的基线实现与联合优化 | 第 4 到 5 周 |
| 可复现 CLI，直接输入预算 | 交付 | 本周 |

对外沟通时按第 2 列说话。CVPR 2027 的截稿日是 2026-11-16，距今 6 周，第 4 到 5 周的结果决定要不要投。

---

## 5. 风险与应对

| 风险 | 触发信号 | 应对 |
|---|---|---|
| 排队太久 | 作业 PENDING 超过 3 小时 | 缩短 walltime、只写 `--gres=gpu:1,VRAM:24G` 不加 `--constraint`，把长作业拆成短作业 |
| conda 或 gsplat 编译失败 | import 报 CUDA 版本不符 | 用 pip 预编译 wheel，torch 与 CUDA 版本对齐后重试，不在工作站上死磕 |
| 编码器太慢 | 单个 checkpoint 编码超过 10 分钟 | 账本为主，真编码只做抽样校验；必要时先降采样点数跑通流程 |
| 账本差 1 % 打不住 | 一致性测试反复超差 | 差异拆项，先修头部与尺度表这些确定性部分，浮动部分用 zstd 兜底并如实标注 |
| 数据下载慢 | 一小时没下完两个场景 | 先只下 bicycle 一个场景，先把通路跑通 |
| GitHub 权限卡住 | 导师几天不回邀请 | 先用 HTTPS + PAT 自己推，权限后续补 |

---

## 6. 第 2 到 4 周做什么（预告，不在本周范围）

- 第 2 周：augmented-Lagrangian 控制器（λ 作为对偶变量，真实字节驱动），可微 gate，SH 复杂度。
- 第 3 周：budget-aware densification（按边际字节定价的 allowance），D2 的分配效率实验。
- 第 4 周：基线（byte-matched prune+quant、SizeGS 或 MesonGS++ 后验、HAC++ 迁移表），
  4 数据集 × 3 预算 × 3 种子的完整协议。
