---
name: write-devlog
description: "Use at the end of a work session, after an experiment, sweep or evaluation run (including one that failed), or mid-task to checkpoint before context is lost. Triggers on: write a devlog, /write-devlog, log what we did, checkpoint this session, record the session, log this experiment. Callable any number of times."
---

# Write Devlog

## Overview

Write the record of what happened this session — the goal, the numbers, the dead ends, and the cost
that no diff can reconstruct. Callable at any point, any number of times: an interim entry is a
checkpoint; a `final` entry closes out a task.

This skill is tuned for **research work with small, tangible goals** — one hypothesis, one sweep,
one measurable bar — where the value of the entry is that six months later the numbers are still
trustworthy and the run is still reproducible.

**Announce at start:** "I'm using the write-devlog skill to record this session."

**Why this matters:** the code history records *what* changed, and nothing about what was tried,
what was rejected, what it measured, or what it cost. Experiment logs get overwritten, checkpoints
get deleted, and a result nobody can reproduce is not a result. The devlog is the durable memory —
write it honestly.

## When to use

- After an experiment, sweep, or evaluation run — even a failed one.
- At the end of a work session, before handing off.
- Mid-task, to checkpoint before context is lost.
- Any time a session produced a number, a decision, or a ruled-out approach.

## Steps

### Step 1: Locate yourself

Find the branch and the project root. In this layout, worktrees live under `workspaces/`, and
`docs/` sits at the **project root**, one or more levels above the worktree.

```bash
BRANCH=$(git branch --show-current)
COMMIT=$(git rev-parse --short HEAD)
git diff --quiet && git diff --cached --quiet || COMMIT="$COMMIT-dirty"
```

Resolve the project root with the loop in `.claude/rules/docs-rules.md`, which is the one
source for that procedure, and use the absolute path it prints in every later tool call. Write into `<project-root>/docs/devlog/`, **never** into the worktree's own tree
— every worktree's entries share the one project-level `docs/devlog/`. If no such root is found (a
plain repo), fall back to `<repo>/docs/devlog/` and say so.

Record the **commit the results were produced at**, and mark it `-dirty` if the tree was not clean.
A number attributed to a dirty tree is a number nobody can reproduce — say so rather than hide it.

Map the branch to the artifact it came from, if one exists: a spec in `<project-root>/docs/specs/`,
a plan in `<project-root>/docs/plans/`, or a task file in `<project-root>/docs/tasks/` — the same
project root, never the worktree's own tree. If the branch maps to none of them, ask what the
goal of this session was — a session with no source artifact still has a goal, and the entry is
worthless without it.

### Step 2: State the goal and the bar

Fill `## Goal` and its **success criterion** first, before writing anything else. The criterion must
be falsifiable: a number, a threshold, a behaviour, a passing test. If no bar was agreed up front,
write "none set up front" and state the bar you are judging against now — retroactive is honest,
absent is not.

Then write `## Outcome` — 3–5 bullets, readable alone, headline number included, and an explicit
verdict against the criterion. A reader who stops after this section should not be misled.

### Step 3: Report *every* run, not the best one

This is the section that makes or breaks the entry. Fill one `### E<n>` block per run, and obey:

- **Every run that produced a number gets a block** — including runs that got worse, crashed, or
  were superseded. A sweep reported as its best row is a lie of omission; the next session will
  re-run the arms you silently dropped.
- **Every metric computed gets a row** — not only the ones that improved. If you computed five
  metrics and report two, report five.
- **Name the baseline and where its numbers came from.** A Δ against an unnamed baseline is
  uninterpretable. If the baseline was not re-run under identical conditions, say so.
- **Seeds:** multiple seeds → mean ± std and the seed count. One seed → say "single seed, n=1".
  Never present one seed as if it were a stable estimate.
- **The command must be the exact one that ran** — copy-pasteable, with the config path or an
  inline diff against the previous run. Not a reconstruction, not a cleaned-up version.
- **Verdict per run:** supports / refutes / inconclusive. "Inconclusive" is a legitimate and common
  answer; reaching for "supports" on noisy evidence is the failure mode to avoid.

Then `## Evaluation` consolidates: one table across all runs, a line for every run *not* detailed
above, and the two mandatory honesty sections — **Not evaluated** and **Threats to validity**. Both
are required. An unstated gap reads to the next reader as a covered one.

### Step 4: Verification, artifacts, reproduction

- **`## Verification`** — the commands you *actually ran* and what they *actually printed*. Not "all
  tests pass" — the command and its output. If tests were not run, say so. An unearned "all green"
  is worse than an empty section, because it will be believed.
- **`## Artifacts`** — checkpoint paths, run IDs, log dirs, result files, plot paths, so any number
  above can be re-checked. Flag anything local-only that will not survive the machine.
- **`## Reproduction`** — the shortest path from a clean checkout to the headline number. If you
  cannot write this, the result is not yet a result; say that explicitly in `## Findings`.

### Step 5: Findings, including the negative ones

`## Findings` records what is now known and with what confidence. **Ruled-out approaches belong
here explicitly** — a negative result that saves the next session a day is worth as much as a win,
and it is the first thing lost when a devlog is written as a highlight reel.

`## Open questions & next steps` must be pick-up-able cold: enough context that a fresh session can
act without re-deriving the state.

### Step 6: Scale the entry to the session

Sections that genuinely do not apply collapse to one line or `none` — brevity is fine. But these
are **never** omitted, however small the session:

`Goal` · `Success criterion` · `Outcome` · `Evaluation → Not evaluated` · `Verification` ·
`Reproduction` · negative results in `Findings`.

If the session produced no runs at all, keep `## Evaluation` and write "no runs this session" —
so the reader can tell that from an oversight.

### Step 7: Pull in the task

- No source artifact → write `Task: none` and state the goal you agreed in Step 1.
- Kind `interim` → **link** to the spec/plan/task file.
- Kind `final` → **inline the full source artifact**, so the entry stays self-contained after that
  file moves on. A `final` entry doubles as a PR description — write it for a reviewer about to read
  the diff.

### Step 8: Cross-link prior entries

Glob `<project-root>/docs/devlog/` for earlier entries on the same branch or task and list them under
`**Previous entries:**`. When an earlier entry's numbers are superseded by this one, say so there —
a stale result left unmarked will be cited later.

### Step 9: Write it

Copy `templates/devlog.md` and fill it in — it is the entry skeleton, and it carries sections
(`## Visualization`, `## Take-home message`, `## Changes → API`) that no step above names.

```
<project-root>/docs/devlog/<YYYY-MM-DD-HHMMSS>-<slug>.md
```

Date-first matches the `docs/specs` and `docs/plans` naming; the time and slug keep every entry
(from every worktree) unique in the shared directory.

### Step 10: Offer to persist it beyond this disk

If the project root's `docs/` is tracked (`git -C <project-root> ls-files --error-unmatch docs
>/dev/null 2>&1`, with the resolved path substituted), commit the entry there with
the `commit` skill — that is the persistence path. If it is untracked, a devlog that lives on exactly
one disk is one `rm -rf` from gone: offer to file it into a knowledge vault, brain server or Obsidian
vault, and **default to yes**. Either way the user must be asked before you finish.

## Red Flags — never

- Reporting the best run of a sweep and omitting the rest.
- A metric computed but not reported because it got worse.
- A Δ against a baseline that is never named or located.
- One seed presented as a stable result without saying "n=1".
- "Tests pass" with no command and no output in `## Verification`.
- An empty `Changes → API` or `Not evaluated` section instead of an explicit "none".
- A headline number attributed to a dirty tree without the `-dirty` marker.
- Writing into the worktree's `docs/` instead of the project-root `docs/devlog/`.
- A summary that reads like a diff. The diff already exists; write what it cannot say.

## Integration

**Uses:**
- `templates/devlog.md` — the entry skeleton.

**Pairs with:**
- Any handoff or finish-development flow — the `final` entry is the record those hand over.
- Any ingestion skill available in the project (Step 10).
