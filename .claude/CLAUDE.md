# Development Project
This is a development project that is co-edited by multiple developers and Claude.

## Project Structure
- `./submodules/` - uneditable submodules for the project
- `./workspaces/` - directory holding the development repositories and its worktrees
- `./scratchpad/` - manual or Claude intermediate scratchpad implementations land here
- `./references/` - external references for additional context, may be md, pdf, docx etc.
- `./docs/` - project-level documentation; `./docs/devlog` holds session devlogs (shared by all worktrees)

## Python Environment
If not asked otherwise, use the `conda` environment `main`, activatable with `conda activate main`

## Rules
Rules that **always** have to be adhered. 

### Development rules
- **Code Minimalism** -> @.claude/rules/code-minimalism.md

### Output rules
- **Math notation** -> @.claude/rules/math-unicode.md
- **Writing style** -> @.claude/rules/writing-style.md

### Documentation rules
- **Documentation** -> read `.claude/rules/docs-rules.md` before writing or searching for a spec,
  plan, devlog or any other project document
- **README conventions** -> @.claude/rules/readme-conventions.md

### Python rules
- **Code Style** -> @.claude/rules/python-code-style.md
- **Docstring Style** -> @.claude/rules/python-docstring-style.md
- **Repository Structure** -> @.claude/rules/python-repo-structure.md

### C++ rules
- **Code Style** -> @.claude/rules/cpp-code-style.md
- **Docstring Style** -> @.claude/rules/cpp-docstring-style.md


## Skills

| Skill | When to invoke |
|-------|---------------|
| **brainstorm** (`.claude/skills/brainstorm/SKILL.md`) | Before any creative work — a new feature, component, or behaviour change. Classifies the request (spike / bounded / architectural), explores intent and approaches through one-question-at-a-time dialogue, and presents a design. Ends at optional handoffs: write a spec to `docs/specs/`, go straight to `write-plan`, or stop with the design in context. |
| **write-devlog** (`.claude/skills/write-devlog/SKILL.md`) | To record a development or research session as a durable devlog in `docs/devlog/` — after an experiment or evaluation run, at the end of a session, or mid-task as a checkpoint. Captures the goal and its success criterion, the full evaluation (every run, every metric, negative results included), artifacts, and reproduction steps. |
| **commit** (`.claude/skills/commit/SKILL.md`) | When writing git commit messages in this repository — call this skill to enforce the project's conventional commit format. |
| **slurm-tumcluster** (`.claude/skills/slurm-tumcluster/SKILL.md`) | When running, submitting, monitoring or debugging jobs on the CVAI/I9 TUM cluster — writing an `.sbatch` file, choosing GPUs/VRAM/partitions, submitting, reading job logs, or diagnosing a job that failed, was preempted, or is stuck pending. |
| **schedule-dsc-node** (`.claude/skills/schedule-dsc-node/SKILL.md`) | When a command needs a GPU on a **DSC compute node** — training, evaluation, inference, profiling, a notebook kernel. Every GPU workload is queued through `dschedule` and no device is ever picked by hand. Gated on `dschedule` being on PATH: the CVAI/I9 TUM cluster uses `slurm-tumcluster`, and a machine with no shared queue runs the command directly. |
| **systematic-debugging** (`.claude/skills/systematic-debugging/SKILL.md`) | On any bug, test failure, or unexpected behaviour — invoke before proposing a fix, not after. Drives root-cause investigation over symptom patching; companion references cover root-cause tracing, defense in depth, condition-based waiting, and finding test polluters. |
| **receiving-code-review** (`.claude/skills/receiving-code-review/SKILL.md`) | When acting on code-review feedback, before implementing any suggestion. Requires verifying each item against this codebase first, clarifying every unclear item before starting, and pushing back with technical reasoning where a suggestion is wrong. |
| **write-skill** (`.claude/skills/write-skill/SKILL.md`) | When creating a new skill under `.claude/skills/`, editing an existing one, or verifying a skill works before it ships. Treats skill authoring as TDD: write pressure scenarios, watch a subagent fail without the skill, then write it. |
| **write-plan** (`.claude/skills/write-plan/SKILL.md`) | When a spec or set of requirements exists for a multi-step task, before touching code. Produces a bite-sized, fully-specified implementation plan in `docs/plans/`, written for an engineer with zero context on this codebase. |
| **implement-plan-using-subagents** (`.claude/skills/implement-plan-using-subagents/SKILL.md`) | To execute a plan from `docs/plans/` with a fresh subagent per task, reviewed between tasks. Keeps this session's context free; best for long plans with independent tasks. Scratch artifacts live in `.sdd/<plan-name>/` (self-ignoring). |
| **implement-planless** (`.claude/skills/implement-planless/SKILL.md`) | To implement without a written plan — a change too small to need one, or work whose context came from a brainstorm earlier in the session or a spec in `docs/specs/`. Gathers context, states goal / done criterion / scope for confirmation, sets the branch and test command, then builds. Escalates to `write-plan` or `brainstorm` when the work outgrows it. |
| **implement-plan-inline** (`.claude/skills/implement-plan-inline/SKILL.md`) | To execute a plan from `docs/plans/` inline in the current session, reviewing the whole change at the end. Best for short plans, or plans whose tasks share a lot of context. |
| **requesting-code-review** (`.claude/skills/requesting-code-review/SKILL.md`) | To dispatch a code-reviewer subagent against finished work — after a task, a major feature, or before merging. The reviewer gets purpose-built context, never this session's history. |
| **github-pr** (`.claude/skills/github-pr/SKILL.md`) | To read a GitHub pull request or its review comments for the repository being worked in — listing the open PRs, or fetching one PR's description and threaded comments. Ships `fetch_github_pr.py`, which wraps the `gh` CLI and merges the conversation, review-summary and inline review-comment streams. Read-only: it never posts, approves or merges. |
| **bitbucket-pr** (`.claude/skills/bitbucket-pr/SKILL.md`) | To read a Bitbucket pull request or its review comments for the repository being worked in — listing the open PRs, or fetching one PR's description and threaded comments. Ships `fetch_bitbucket_pr.py`, which calls the Bitbucket REST API with the standard library only. Read-only: it never posts, approves or merges. |
| **fix-pr** (`.claude/skills/fix-pr/SKILL.md`) | To work through the review feedback on the current branch's pull request, on GitHub or Bitbucket. Detects the host from the `origin` remote and routes to `github-pr` or `bitbucket-pr`, parses each requested change into a record, applies trivial and moderate fixes directly, and stops for approval on anything complex or design-level. Requires `receiving-code-review`. |
| **using-git-worktrees** (`.claude/skills/using-git-worktrees/SKILL.md`) | Before starting feature work that should not touch the current branch, or before executing a plan. Creates the isolated workspace under `./workspaces/`, preferring a native worktree tool when one exists. |

## Hooks (`.claude/settings.json`)
- **startup** (SessionStart) — `.claude/hooks/startup.py` resolves the project root (nearest
  ancestor holding both `docs/` and `workspaces/`) and opens the session with `PROJECT_ROOT=<path>`
  plus a live snapshot: each workspace with its branch and dirty count, the newest specs and
  plans with their age, and the contents of `references/`. It also sets `env.PROJECT_ROOT` in
  `.claude/settings.local.json` (gitignored) when the session was launched at that root, which is
  what makes `$PROJECT_ROOT` resolve in Bash. Documentation rules: `.claude/rules/docs-rules.md`.
  Stdlib only, fails open.
- Hook commands are anchored at `${PROJECT_ROOT:-$PWD}`. The fallback covers the first session in a
  fresh clone, before the startup hook has written the variable.

## Important Do's
- When doing scratchpad implementations, save them into a folder named `./scratchpad/DD_MM_YY_HH_MM_<description>/`

## Important Dont's
- **NEVER** assume coordinate frames, if ambiguous or unclear, ask.
- **NEVER** edit external code to make local code work.
- **NEVER** pre-emptively drop a feature/idea due to worse results without building a hypothesis on why its not working