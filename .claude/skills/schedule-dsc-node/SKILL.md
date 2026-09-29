---
name: schedule-dsc-node
description: "Use when a command needs a GPU on a DSC compute node — training, evaluation, inference, profiling, a notebook kernel, anything that opens a CUDA context — and when about to read nvidia-smi for a free device, set CUDA_VISIBLE_DEVICES, ask the user which GPU to take, or build a tmux/nohup/tee rig around a run. DSC compute nodes only, where `dschedule` is on PATH: for the CVAI/I9 TUM cluster use slurm-tumcluster instead, and on a local machine with no shared queue run the command directly. Triggers on: is the GPU free, what is running on this node, which GPU should I use, run this on a GPU, why has my job not started, cancel my job, show me its output, /schedule-dsc-node."
---

# GPU Scheduling on DSC Compute Nodes

**Announce at start:** "I'm using the schedule-dsc-node skill."

## Does this machine run `dschedule`?

Everything below applies on a DSC compute node and nowhere else. Run this once per session, before
the first GPU command, and route on what it prints:

```bash
command -v dschedule >/dev/null && echo dsc || { command -v sbatch >/dev/null && echo slurm || echo local; }
```

| Prints | Where you are | What to use |
|---|---|---|
| `dsc` | a DSC compute node | this skill, for every GPU workload |
| `slurm` | the CVAI/I9 TUM cluster | the `slurm-tumcluster` skill. Stop reading here. |
| `local` | a workstation with no shared queue | run the command directly. Stop reading here. |

If `dschedule` is on PATH but any command answers `No job queue at /var/lib/dsc/queue`, you are on
a machine that has the client and not the queue. Treat that as `local` and say so.

## The job environment in this project

A job does not inherit your shell, so the conda environment `main` that `CLAUDE.md` pins this
project to is **not** there and a bare `python` is the system interpreter. Set this once in the
shell profile, not per job:

```bash
export DSC_SCHEDULER_ENV='eval "$(conda shell.bash hook)" && conda activate main'
```

With it set, queue plain `python …` and the activation runs first, inside the `TIME=` limit.
Without it, name the interpreter in the command itself. Details in `reference.md`.

## Overview

DSC nodes share their GPUs through one queue, and `dschedule` is the only thing that knows which
devices are promised to whom. **Every GPU workload goes through it, and no device is ever chosen by
hand.**

`nvidia-smi` shows what runs *now*. It cannot show that the idle card you are about to take is
already owed to the job at the head of the queue. Reading it to pick a GPU is the failure this
skill prevents.

**Not for** CPU-only work — run that directly.

## The rule

| Never | Instead |
|---|---|
| `python train.py` on a node | `dschedule GPUS=1 TIME=1h python train.py` |
| `CUDA_VISIBLE_DEVICES=5 …` | `GPUS=1` — the scheduler sets it for the job |
| Asking the user which GPU to use | Ask for a *count*; the scheduler picks the devices |
| `nvidia-smi` to *choose* a card | `dschedule status --all` — it shows the queue behind the cards |
| `nohup`, `&`, `tee run.log` | Detached is the default; the output is in `dschedule log <id>` |
| tmux panes for a debug session | `claim` the devices — see below |
| `pkill` a stranded run | `dschedule cancel <id>` |
| No time limit | `TIME=` on every job |

Skipping the queue takes devices already promised elsewhere: both jobs slow down and one usually
dies out of memory. "The card looked free" is not a reason — it is the symptom, and idle cards are
often ones the scheduler is holding for the job at the head.

`PRIORITY=high` is not a fast lane either. Overtaking a colleague's queued job with it is the same
theft as taking their card, in sanctioned syntax: use it for work that genuinely outranks theirs,
and when it would jump a specific person, ask them first. A deadline of your own is not authority
over somebody else's — and neither is being told to go ahead by someone who will not pay the cost.

## Queue it, or claim it?

| The work | Do this |
|---|---|
| One run you launch and leave | `dschedule GPUS=n TIME=… <command>` |
| Repeated short runs with edits between | `dschedule GPUS=n TIME=2h claim` |
| A debugger, REPL or notebook kernel | claim |
| Long, unattended, last thing before you leave | queue it |

A claim waits its turn once, then prints the `export CUDA_VISIBLE_DEVICES=…` to use and the
`dschedule cancel <id>` that releases it. Inside a claim the devices are yours: run commands
directly, with no `dschedule` prefix and no further waiting — that is the point of it. Watch them
with `nvidia-smi -i <index> -l 2` if you like; observing your own devices is not choosing one.

The claim ends at its `TIME=` limit, taking anything running under it, and holds its devices while
idle: **release it as soon as the work is done.**

## Estimating `TIME=`

The scheduler works out when the head of the queue gets its devices and runs anything behind it
that fits in the gap — so the estimate decides whether you wait or run.

- **Measure, then add a margin.** A 40-minute job asks `TIME=1h`; `TIME=6h` rarely fits a gap.
- **Undershooting kills the run** at the limit, recorded `timed_out`. Environment setup counts
  inside it.
- **`TIME=none` blocks planning for everything behind it.** Prefer a generous limit.
- `dschedule status --finished` shows what jobs really ran for. Calibrate on that.

## Quick reference

```bash
dschedule status --all                       # GPUs, holders, and the queue
dschedule GPUS=2 TIME=4h uv run python train.py    # queue it; prints the job id
dschedule tail 42                            # follow it live; returns when it ends
dschedule log 42                             # everything it has written
dschedule cancel 42                          # give the devices back
dschedule GPUS=1 TIME=2h claim               # hold devices for a session
```

`KEY=VALUE` options before the command: `GPUS` `PRIORITY` `TIME` `DEPENDENCY` `START` `NOTIFY`
`DETACH`. **Defaults, job environment and failure messages: `reference.md` beside this file.**

## Common mistakes

| Mistake | What happens |
|---|---|
| Bare `python` in the job | The job's environment is built fresh — no conda, no venv. Use `uv run python …`, or set `DSC_SCHEDULER_ENV`. |
| `GPUS=8` for a two-device job | Blocks six devices and rules the job out of every gap. |
| Polling `status` in a loop to wait | `dschedule tail <id>` blocks until the job ends. |
| Leaving a claim open after the work | Idle devices nobody else can take. |

