# dschedule reference

Loaded by the schedule-dsc-node skill when the exact syntax is needed.

## Queueing a job

```
dschedule [KEY=VALUE ...] COMMAND ...
```

Everything after the last `KEY=VALUE` is the command. A leading `submit` is optional, and is only
needed when the command itself starts with a dash or reads like a dschedule command.

| Option | Values | Default | What it does |
|---|---|---|---|
| `GPUS=N` | integer ≥ 1 | `1` | Devices to hold. |
| `PRIORITY=` | `high`, `medium`, `low` | `medium` | Queue to wait in. Each is exhausted before the next. |
| `TIME=` | `30s`, `45m`, `6h`, `2d`, `none` | `24h` | Wall-clock limit, measured from launch. At the limit the job is stopped and recorded `timed_out`. `none` means no limit. |
| `DEPENDENCY=ID` | job id | none | Wait until that job has finished **successfully**. If it fails, is cancelled or times out, this job is cancelled too. |
| `START=` | `6h`, or `DD-MM-HH:mm` | none | Hold the job back until a wait has passed or a local wall-clock time has come. `24-12-22:30` is the 24th of December at 22:30. |
| `NOTIFY=` | `off`, `fail`, `all` | `all` with a webhook set, else `off` | Post to your Teams webhook. `fail` reports failures and time limits; `all` also reports the job's start and its clean finish. |
| `DETACH=` | `True`, `False` | `True` | `True` leaves the command to the daemon and returns at once. `False` waits in the queue and then runs the command on this terminal. |

```bash
dschedule GPUS=2 TIME=6h python train.py --epochs 10        # queued, returns immediately
dschedule GPUS=1 TIME=45m DETACH=False python eval.py       # runs here once its turn comes
dschedule GPUS=4 PRIORITY=high TIME=2h python sweep.py
dschedule DEPENDENCY=41 GPUS=1 TIME=30m python eval.py --checkpoint last.pt
```

The confirmation names the job id, which every other command takes:

```
Queued job 42 on 2 GPU(s), medium priority, limit 6h, notifying on all
```


## Choosing detached or attached

- **Detached (the default)** is right for anything long. The daemon runs it, captures its output,
  and it survives your shell closing. Follow it with `dschedule tail`.
- **`DETACH=False`** is right when you need the command's exit code inline or the run is short: it
  blocks until the job gets its devices, streams to the terminal, and exits with the command's own
  status. It writes **no log files** — the output only ever went to the terminal.


## Following a job

```bash
dschedule tail 42              # last 10 lines, then live until the job ends
dschedule tail 42 -n 100       # more history first
dschedule log 42               # everything written so far, then exit
```

`tail` waits if the job is still pending, prints as the job writes, and returns on its own when the
job finishes — so it is a safe way to block until a job is done. Ctrl-C stops following; the job
keeps running.

Both streams land in one log in the order the job printed them, so `dschedule log 42` is the whole
story. `--stderr` only means something on a node configured to keep the two apart.


## The queue

```bash
dschedule status               # what is running and waiting
dschedule status --all         # plus the GPU table
dschedule status --me          # only your jobs
dschedule status --finished    # what has ended
dschedule cancel 42            # a pending job goes at once; a running one is asked to stop
```

The `#` column is the place in the queue, `-` for a job already running. `WAITS` shows what still
holds a job back: `dep:41`, `at:24-12-22:30`, or both.

A job at the head that wants more GPUs than are free does not idle them: a job further down that
carries a `TIME=` limit short enough to be gone before the head is due runs in the gap. This is why
a realistic `TIME=` starts your job sooner.


## Holding GPUs instead of queueing a command

When the work cannot be a single command, take the devices themselves:

```bash
dschedule GPUS=2 TIME=2h claim
```

A claim waits in the queue like any job, then prints the `export CUDA_VISIBLE_DEVICES=...` to use
and the `dschedule cancel <id>` that gives the devices back. Nothing else is scheduled on them
until it ends, and it ends by itself at its `TIME=` limit. When to prefer this to queueing is in
SKILL.md.


## What the job's environment is

The daemon does not inherit your shell. A job runs:

- as the user who submitted it, in the directory it was submitted from,
- with `CUDA_VISIBLE_DEVICES` set to the devices it was given,
- with a fresh environment: `HOME`, `USER`, `PATH` (`~/.local/bin` plus the system directories).

So an activated virtualenv, conda or micromamba environment is **not** there, and a bare `python`
will not be the interpreter you meant. Two ways to fix that:

```bash
dschedule GPUS=1 TIME=1h uv run python train.py     # uv finds the project's own interpreter
export DSC_SCHEDULER_ENV='eval "$(micromamba shell hook -s bash)" && micromamba activate main'
export DSC_SCHEDULER_ENV='file:~/bin/train-env.sh'  # or a script to source
```

`DSC_SCHEDULER_ENV` is read when the job is queued and run before its command, so it belongs in the
shell profile once. It runs under `set -e`, so a job whose environment fails to come up is never
started with the wrong one — and its startup counts against `TIME=`, which matters when the limit
is measured in seconds.


## Notifications

`dschedule webhook <URL>` saves a Teams workflow webhook in `~/.config/dsc/scheduler-webhook`, mode
600, and posts a card to the channel at once so a wrong URL is caught there and then. Each user has
their own; a job only ever reports to its own owner's.

Once a webhook is set, jobs report everything unless they say otherwise — `NOTIFY=fail` for
failures and time limits only, `NOTIFY=off` for silence. A cancelled job is never reported.


## Gotchas

| What you see | What it means |
|---|---|
| `'stauts' is not a dschedule command; did you mean 'status'?` | A mistyped command. `dschedule submit stauts ...` queues it anyway if it really was the job. |
| `'trainn.py' is neither a dschedule command nor a program on this node` | The first word names nothing runnable. Check the path; queue it with `submit` if it only exists inside the job's environment. |
| `NOTIFY applies to queueing a job, not to status` | A `KEY=VALUE` given to a command that queues nothing. |
| `job 42 belongs to another user and cannot be stopped` | Only the owner cancels a job. |
| `no scheduler daemon is running on this node` | The queue is there but nothing will start. An administrator runs `sudo systemctl start dsc-scheduler`. |
| `No job queue at /var/lib/dsc/queue` | Not a scheduler node, or you are not on the node at all. |
| Job ends `timed_out` sooner than expected | The environment preamble runs inside the limit. |
| Job ends `failed` with no output | Read `dschedule log <id>`; an attached job has none by design. |


## A worked run

```bash
dschedule status --all                                   # 3 of 8 GPUs free
dschedule GPUS=2 TIME=4h uv run python train.py --epochs 50
# Queued job 43 on 2 GPU(s), medium priority, limit 4h, notifying on all
dschedule tail 43                                        # follow until it ends
dschedule status --finished | tail -1                    # how it ended, and what it ran for
```