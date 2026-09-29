---
name: slurm-tumcluster
description: "Use when running, submitting, monitoring or debugging jobs on the CVAI/I9 TUM cluster with SLURM — writing an .sbatch file, choosing GPUs/VRAM/constraints/partitions, submitting with sbatch, checking squeue/sacct, reading job logs, or diagnosing a job that failed, was preempted, or is stuck pending. Triggers on: run this on the cluster, submit to slurm, write an sbatch script, make a slurm job, why is my job pending, did my job finish, check job <id>, gather the results, cancel my job, /slurm-tumcluster."
---

# SLURM on the CVAI/I9 TUM Cluster

## Overview

Produce a **validated, runnable** `.sbatch` file, submit it under a recorded manifest, and gather
the job back deterministically — state, accounting, logs and diagnosis — without guesswork.

**Announce at start:** "I'm using the slurm-tumcluster skill."

**Why this exists:** SLURM fails *silently* in ways that look like success. A space inside a
directive value makes it drop every option after it. A directive below the first command is
ignored. `$HOME` in `--output` is taken literally. A missing log directory kills the job with no
log to explain why. A typo in `--constraint` leaves the job pending forever. In every case the job
is accepted, appears normal in `squeue`, and does the wrong thing. The validator in this skill
turns all of these into blocking errors *before* submission.

**Cluster:** `https://cvai.cit.tum.de/intern/system/slurm` · live state: `https://adm9.in.tum.de/slurm`
**Account tier:** `wiss` (18 GPU / 30 jobs per user) — pass `--tier` to change.

---

## Hard rules

These are not stylistic. Each one corresponds to a failure that is invisible at submit time.

1. **Never submit a script that has not passed `validate_sbatch.py`.** No exceptions, including
   "it's just a quick test".
2. **Never pass `--force` on your own initiative.** It exists so the *user* can override a check
   they have judged to be wrong. If validation blocks, report the errors and fix them.
3. **Never choose `--partition=DEADLINE` or `DEADLINEBIG` yourself.** These preempt other people's
   running jobs — someone else loses work. Use them only when the user explicitly asks, and only
   with a real deadline name in `--comment`.
4. **Always confirm with the user before submitting.** Submission consumes a shared, contended
   resource. Show the resolved resource request and wait for a yes.
5. **Never `scancel` a job the user did not name.** Never `scancel -u $USER`.
6. **Never invent node names, feature strings or GPU models.** They come from the live cluster;
   the validator checks every one against `sinfo`.
7. **Request honestly.** Over-requesting lengthens your own queue wait and starves others;
   under-requesting gets the job killed mid-run. Estimate from the actual workload.

---

## Workflow

### 1. Establish what the job needs

Before writing anything, you need these. Ask for whatever the request does not already answer —
don't guess at resources, and don't silently pick a default that costs an hour of queue time.

| Needed | Notes |
|---|---|
| Command to run | The exact entry point and its arguments. |
| Working directory | Must be on **shared storage** reachable from compute nodes, not node-local `/tmp`. |
| Environment | Conda env name, or module loads. |
| GPU count + VRAM floor | The VRAM floor decides which nodes are eligible — the single biggest factor in queue time. |
| CPUs | Match the dataloader worker count. ~3 per GPU is the usual default. |
| System RAM | ~32G for a normal DL job. Distinct from VRAM. |
| Walltime | An honest upper bound. Over-long walltimes queue longer; too-short ones get killed. |
| Log directory | Absolute path, must already exist. |

If the user says "just run it", propose a concrete resource set, state your reasoning in one line,
and confirm — don't silently pick.

### 2. Write the sbatch file

Start from `templates/job.sbatch`. Read `reference/cluster.md` when choosing GPUs, constraints or
partitions.

Write the file into the project (e.g. `slurm/<job-name>.sbatch`), not into a temp directory — the
script is a reproducibility artifact and belongs next to the code it runs.

### 3. Validate — this gate is not optional

```bash
python .claude/skills/slurm-tumcluster/scripts/validate_sbatch.py slurm/<job-name>.sbatch
```

Exit `0` = safe to submit, `1` = blocking errors, `2` = file unreadable. Fix every error and
re-run until clean. Address warnings too, or say why you are accepting each one.

Add `--offline` when SLURM is not reachable (uses the bundled node snapshot instead of `sinfo`),
`--strict` to treat warnings as blocking, `--json-output` when parsing the result.

### 4. Preflight and submit

```bash
# Validate + `sbatch --test-only`, without queueing anything:
python .claude/skills/slurm-tumcluster/scripts/submit_job.py slurm/<job-name>.sbatch --dry-run

# Then, after the user confirms:
python .claude/skills/slurm-tumcluster/scripts/submit_job.py slurm/<job-name>.sbatch
```

`submit_job.py` re-runs validation itself and refuses to submit on errors, so it is safe to call
directly — but run `--dry-run` first and show the user what SLURM says about schedulability.

On success it writes `slurm-runs/<job_id>.json` recording the job id, resolved log paths, the
script's SHA-256, and every honoured directive. **This manifest is what makes gathering
deterministic** — it is why you never have to guess where a job's output went.

Confirm `slurm-runs/` is ignored (it already is) unless the user wants manifests committed.

### 5. Gather

```bash
python .claude/skills/slurm-tumcluster/scripts/gather_job.py <job_id>
python .claude/skills/slurm-tumcluster/scripts/gather_job.py --all-runs   # every manifest
```

Queries `squeue` (live) then `sacct` (finished), resolves logs via the manifest, reports the
restart count, and matches known failure signatures to diagnostic hints. Exit `0` only when every
job reached `COMPLETED`.

**Do not poll in a tight loop.** A training job runs for hours. Check once, report the state and
the ETA, and let the user come back — or agree an interval with them first.

---

## Diagnosing a job

| Symptom | Read this |
|---|---|
| Pending, reason `Resources` | Request is valid, nothing free yet. Widen `--constraint` or lower the VRAM floor to reach more nodes. |
| Pending, reason `QOSMax…` / `AssocMax…` | A tier limit is saturated. `squeue -u $USER` — you or the account already hold too many GPUs/jobs. |
| Pending forever, reason never changes | The request can never be satisfied: a constraint matching no node, or more GPUs than the tier allows. Re-run the validator. |
| `COMPLETED` but no work done | A setup step failed and was ignored. The script needs `set -euo pipefail`. |
| No log file at all | The `--output` directory did not exist, or the path contained an unexpanded `$VAR`. |
| Job restarted from scratch | Preempted and requeued. Check `SLURM_RESTART_COUNT` and resume from a checkpoint. |
| `bad interpreter` | CRLF line endings — `dos2unix` the script. |
| Logs arrive in bunches / lost on crash | Missing `srun --unbuffered`. |
| `CUDA error: no kernel image` | GPU arch newer than the installed torch build. Pin `GPU_CC:` or reinstall. |

Raw commands: `squeue -u $USER`, `sacct -j <id> --format=State,Elapsed,MaxRSS,ReqTRES`,
`scontrol show job <id>`, `sinfo -s`, `scancel <id>`.

---

## Files

| Path | Purpose |
|---|---|
| `templates/job.sbatch` | Canonical template with the traps documented inline. |
| `reference/cluster.md` | Nodes, GPUs, features, partitions, limits, and how to choose them. |
| `reference/node_features.json` | Machine-readable snapshot + tier limits. Fallback for `sinfo`. |
| `scripts/validate_sbatch.py` | Static validation. The submission gate. |
| `scripts/submit_job.py` | Validate → preflight → submit → write manifest. |
| `scripts/gather_job.py` | squeue/sacct + logs + diagnosis for one or all jobs. |

The scripts are stdlib-only and need no environment beyond Python 3.12.
