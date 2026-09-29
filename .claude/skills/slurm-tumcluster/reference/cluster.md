# CVAI/I9 Cluster Reference

Distilled from <https://cvai.cit.tum.de/intern/system/slurm> (scraped 2026-07-25).
Live state: <https://adm9.in.tum.de/slurm> · SLURM docs: <https://slurm.schedmd.com/sbatch.html>

The node table below **goes stale** — the sysadmins add and retire nodes. Always prefer:

```bash
sinfo -N -o "%10N %f" | sort -u -V     # current features per node
sinfo -s                               # partitions and their node counts
```

The validator does this automatically and only falls back to `node_features.json` when SLURM is
unreachable.

---

## Nodes

| Node | Class | GPU | VRAM | GPU gen | CC | CPU |
|---|---|---|---|---|---|---|
| node1  | NOTBIG | p6000 | 24G | PASCAL | 6.1 | E5-2687W @ 3000 |
| node2  | NOTBIG | titan | 12G | PASCAL | 6.1 | E5-2697 @ 2300 |
| node4  | BIG | rtx_6000_ada | 48G | ADA | 8.9 | E5-2697 @ 2300 |
| node5  | BIG | rtx_6000_ada | 48G | ADA | 8.9 | E5-2697 @ 2300 |
| node6  | NOTBIG | titan | 12G | PASCAL | 6.1 | E5-2697 @ 2300 |
| node7  | NOTBIG | p6000 | 24G | PASCAL | 6.1 | E5-2697 @ 2300 |
| node8  | NOTBIG | rtx_5000 | 16G | TURING | 7.5 | GO-6148 @ 2400 |
| node9  | NOTBIG | rtx_6000 | 24G | TURING | 7.5 | GO-6248 @ 2500 |
| node10 | NOTBIG | rtx_5000 | 16G | TURING | 7.5 | GO-6248 @ 2500 |
| node11–13 | BIG | rtx_8000 | 48G | TURING | 7.5 | GO-6254 @ 3100 |
| node14 | BIG | rtx_a6000 | 48G | AMPERE | 8.6 | GO-6254 @ 3100 |
| node15–18 | BIG | a40 | 48G | AMPERE | 8.6 | GO-6248R @ 3000 |
| node19–20 | BIG | nvidia_l40s | 48G | AMPERE¹ | 8.9 | GO-6530 @ 2100 |
| node21 | BIG | nvidia_h100 | 96G | HOPPER | 9.0 | GO-6554 @ 2200 |
| node22 | BIG | pro_6000 | 96G | BLACKWELL | 12.0 | TU-9555 @ 3200 |

¹ The L40S nodes are tagged `GPU_GEN:AMPERE` in SLURM despite being Ada silicon (CC 8.9). Select
them by `GPU_CC:8.9` or `GPU_MODEL:nvidia_l40s`, not by generation.

There is no node3.

---

## Choosing where the job runs

**The governing tradeoff:** the more nodes a job can land on, the sooner it starts. Constrain to
what the job genuinely requires and no further.

### By VRAM floor — the usual choice

```
#SBATCH --gres=gpu:1,VRAM:24G
```

Selects any GPU with at least 24G. Matching is **per GPU, not per node**, which makes it more
expensive for the scheduler than a feature constraint — but it expresses the real requirement and
reaches the widest node set. Prefer it unless you need specific hardware.

### By feature constraint

```
#SBATCH --constraint="GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER"
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9"
#SBATCH --constraint="GPU_MODEL:titan|GPU_MODEL:rtx_5000"
#SBATCH --constraint=BIG
```

`|` is OR, `&` is AND. Every feature must exist on some node or the job pends forever — a typo is
**not** a submit-time error. Available feature namespaces: `BIG`/`NOTBIG`, `GPU_GEN:`, `GPU_MEM:`,
`GPU_CC:`, `GPU_MODEL:`, `CPU_GEN:`, `CPU_MODEL:`, `CPU_MHZ:`, plus the bare model name (`a40`,
`titan`, …). Use `--constraint` for compute-capability requirements, since CC is what a CUDA build
actually cares about.

### By GPU type

```
#SBATCH --gres=gpu:titan:1
#SBATCH --gpus=titan:1
```

Unlike `--constraint`, this takes exactly one type — no alternatives. Narrow; use sparingly.

### By node list / exclusion

```
#SBATCH --exclude=node14
#SBATCH --nodelist=node[1,5,9]
```

Last resort. `--nodelist` pins the job to specific machines and usually means a long wait.
`--exclude` is the better tool when one node is misbehaving.

---

## Partitions and preemption

| Partition | Nodes | Preempts | Preempted by | Needs `--comment` |
|---|---|---|---|---|
| `NORMAL` (default) | all | — | DEADLINE, DEADLINEBIG | no |
| `DEADLINE` | all | NORMAL | DEADLINEBIG | **yes** |
| `DEADLINEBIG` | 48G VRAM only | DEADLINE, NORMAL | — | **yes** |

**A DEADLINE job kills someone else's running work.** Never select it on your own initiative —
only when the user explicitly asks, with a real deadline named in `--comment`.

```
#SBATCH --partition=DEADLINE
#SBATCH --comment="CVPR 2027 rebuttal"
#SBATCH --constraint=NOTBIG     # so DEADLINEBIG cannot preempt you in turn
```

A preempted job is **requeued by default** and restarts from the beginning. Handle it:

```bash
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "resuming after preemption"
    RESUME_FLAG="--resume"
fi
```

`--no-requeue` disables the automatic restart — only useful when a restart would corrupt state.

---

## Account limits

`MaxTRESPA` = per account · `MaxTRESPU` = per user · `MaxJobsPU` = concurrent jobs per user.

| Tier | Priority | GPU/account | CPU/account | Mem/account | GPU/user | Jobs/user |
|---|---|---|---|---|---|---|
| external | — | 10 | 20 | 128G | 2 | 5 |
| stud | 5 | 95 | 100 | 512G | 9 | 15 |
| wiss | 10 | 135 | 200 | — | 18 | 30 |

**These change constantly.** The table is a snapshot; check live limits with
`sacctmgr show assoc user=$USER format=Account,GrpTRES,MaxJobs` when a job pends on `AssocMax…`
or `QOSMax…`.

---

## Directive traps

Each of these produces a job that **submits successfully and then misbehaves**. This is the entire
reason the validator exists.

| Trap | Consequence |
|---|---|
| Whitespace in a value: `--gres=gpu:1, VRAM:12G` | SLURM stops parsing at the space. The VRAM floor is silently dropped. |
| `#SBATCH` after the first command | Ignored without warning. |
| `$HOME` or `~` in `--output`/`--error`/`--chdir` | No shell runs over directives; the literal text becomes part of the path. |
| Log directory does not exist | SLURM will not create it. Job dies with no log explaining why. |
| Typo in `--constraint` | Not a submit error. Job pends forever on `Resources`. |
| CRLF line endings | `bad interpreter` — the `\r` becomes part of the shebang path. |
| No `%j` in `--output` | Concurrent jobs overwrite each other's logs. |
| Missing `set -euo pipefail` | A failed `conda activate` is ignored; the job reports `COMPLETED` having done nothing. |
| `srun` without `--unbuffered` | glibc pipe-buffers stdout; logs arrive in bunches and a crash loses the last lines. |

Quoting is the fix when a value legitimately contains a space: `--job-name="My Training"`.

---

## Sizing guidance

- **CPUs:** match the dataloader worker count (`num_workers`). ~3 per GPU is a sane default.
- **RAM:** ~32G for a normal DL job. Too little and the cgroup kills the job mid-run; too much and
  it queues longer. This is system RAM, *not* VRAM.
- **Walltime:** an honest upper bound. Short walltimes schedule sooner (backfill), but a job killed
  at the limit loses everything not checkpointed.
- **VRAM:** the floor that decides node eligibility. Setting it higher than needed is the most
  common self-inflicted queue delay.

---

## Command reference

```bash
sbatch job.sbatch                                  # submit
sbatch --test-only job.sbatch                      # when could it start? does not queue
squeue -u $USER                                    # my queue
squeue -j <id> -o "%i %T %R %M %L %N"              # one job, with pending reason
sacct -j <id> --format=State,Elapsed,MaxRSS,ReqTRES,NodeList   # after it finishes
sstat <id>                                         # live stats (needs srun in the script)
scontrol show job <id>                             # everything, incl. StdOut and Restarts
scancel <id>                                       # cancel
sinfo -s                                           # partition summary
sinfo -N -o "%10N %f" | sort -u -V                 # node features
```

`sacct` is the source of truth once a job leaves the queue; `squeue` knows nothing about finished
jobs and `sacct` lags live state by a few seconds. `gather_job.py` queries both.

---

## Connecting from outside

SSH to chair hosts uses **port 58022**; FQDNs are `<host>.cvai.in.tum.de`.

```bash
ssh -p 58022 -l <user> atcremers1.cvai.in.tum.de
```
