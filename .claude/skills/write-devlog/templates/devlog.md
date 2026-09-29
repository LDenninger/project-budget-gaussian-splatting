# Devlog — <slug>

- **Date:** <YYYY-MM-DD HH:MM:SS>
- **Branch:** <branch>
- **Commit:** <short SHA the results below were produced at, + `-dirty` if the tree was not clean>
- **Kind:** interim | final
- **Status:** goal met | partially met | not met | inconclusive | abandoned
- **Task:** <<project-root>/docs/tasks/YYYY-MM-DD-slug.md — link for interim, inlined below for final>

**Previous entries:** <list earlier <project-root>/docs/devlog/ entries on this branch/task, or "none">

## Goal

<One sentence: the tangible thing this session set out to achieve.>

**Success criterion:** <The falsifiable bar agreed up front — a number, a behaviour, a passing test.
"Improve X" is not a criterion; "mIoU ≥ 0.72 on val, up from 0.68" is. If none was set, write
"none set up front" and state the bar you judged against retroactively.>

## Outcome

<3–5 bullets, written first, readable alone. State the headline number and whether the criterion
was met. If the result was negative or inconclusive, say that here — not buried below.>

## What was done

<Narrative for someone who was not there. What was built or changed, in what order, and — just as
important — what was tried and abandoned, and why. This is the part no diff can reconstruct.>

## Changes

- **Code:** <files/modules touched and what they now do. Link paths.>
- **API:** <before → after for every signature, CLI flag, config key, or schema that moved.
  If none, write "none".>
- **Config / data:** <config files, dataset versions, preprocessing, splits that changed.
  If none, write "none".>

## Experiments

<One block per run. Repeat verbatim. Every run that produced a number goes here — including the
ones that got worse, crashed, or were superseded. Delete this whole section only if the session
produced no runs, and say so under "Evaluation".>

### E<n> — <short name>

- **Hypothesis:** <what this run was meant to show, stated so it can fail>
- **Setup:** commit `<sha>` · config `<path or inline diff vs. previous run>` · data
  `<dataset, split, N samples>` · seed(s) `<...>` · hardware `<GPU/CPU, count>` · runtime `<h:mm>`
- **Command:**
  ```bash
  <exact, copy-pasteable command that produced the numbers below>
  ```
- **Results:**

  | Metric | Split | Baseline | This run | Δ |
  |---|---|---|---|---|
  | <every metric computed, not only the good ones> | | | | |

  <Report the baseline you are comparing against by name and where its numbers come from. If
  multiple seeds were run, give mean ± std and the number of seeds — never a single cherry-picked
  seed. If only one seed was run, say so.>

- **Observations:** <what the numbers and the logs/plots actually showed, incl. failure modes,
  instabilities, suspicious wins>
- **Verdict:** supports | refutes | inconclusive — <one line why>

## Evaluation

- **Consolidated results:** <one table across all experiments above, so the reader sees the whole
  sweep in one place, best row marked. Skip only if there was a single experiment.>
- **Runs not detailed above:** <every crashed, aborted, or superseded run, one line each with why.
  If all runs are detailed above, write "none".>
- **Not evaluated:** <metrics, splits, or conditions deliberately or accidentally left unmeasured.
  This section is mandatory — an unstated gap reads as a covered one.>
- **Threats to validity:** <single seed, test-set leakage, tuned on the eval split, tiny N,
  non-deterministic ops, baseline not re-run under identical conditions, ... If genuinely none,
  write "none identified".>

## Verification

<The commands actually run and what they actually printed — tests, linters, type checks, sanity
scripts. Paste the real output, not "tests pass". If tests were not run, say so.>

```
$ <command>
<output>
```

## Artifacts

<Where the evidence lives, so any number above can be re-checked: checkpoint paths, run IDs
(W&B/MLflow/...), log dirs, result JSON/CSV, generated plots. Note anything that is local-only and
will not survive the machine.>

## Visualization

<Plots produced alongside the quantitative metrics — path plus one line on what each shows and
what to look at. Omit if none.>

## Findings

<What is now known that was not known before, and with what confidence. Include negative results
explicitly — a ruled-out approach is a result and saves the next session from repeating it.>

## Open questions & next steps

- <question or next action, with enough context to be picked up cold>

## Reproduction

<The shortest path from a clean checkout to the headline number: branch/commit, environment,
data location, command. Someone should be able to follow this without asking.>

## Take-home message

<One or two sentences. The thing worth remembering in six months.>

## Task

<Kind `interim`: a link to the task file.
Kind `final`: the full task markdown inlined here, so the entry stays self-contained.>
