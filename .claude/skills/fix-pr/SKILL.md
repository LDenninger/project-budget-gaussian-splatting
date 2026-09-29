---
name: fix-pr
description: Use when the user wants to address, fix, or resolve reviewer feedback on the current branch's GitHub or Bitbucket pull request, for example "fix the PR comments", "address the review feedback", "apply the requested changes from the PR", "work through the review on this branch".
---

# fix-pr

**Announce at start:** "I'm using the fix-pr skill."

Work through the review feedback on the **current branch's** PR, on GitHub or Bitbucket: fetch the
comments, parse each requested change into a structured record, apply the fixes (asking first for
anything complex or design-level), and end with a summary mapping every comment to what was done.

This skill **reads** the PR. It never posts, resolves, approves, or merges. It reuses the fetch
script of whichever companion skill matches the host, `github-pr` or `bitbucket-pr`, so do **not**
reimplement either host's API.

**REQUIRED BACKGROUND:** Use `receiving-code-review`. Verify each comment is technically correct
before implementing, and push back on questionable feedback instead of complying performatively.

## Workflow

### 1. Find the PR for the current branch

If the user named a PR id (or handed you a comments dump), skip the lookup and use it directly.
Otherwise establish the branch and the host in one step, from inside the repository under review:

```bash
git rev-parse --abbrev-ref HEAD   # current branch
git remote get-url origin         # host lives in this URL
```

Match the remote URL against the two supported hosts. This covers both the HTTPS and the SCP-style
SSH remote forms:

| `origin` URL contains | Companion skill | Fetch script, under `.claude/skills/` | Auth precondition |
|---|---|---|---|
| `github.com` | `github-pr` | `github-pr/fetch_github_pr.py` | the `gh` CLI installed and authenticated via `gh auth login` |
| `bitbucket.org` | `bitbucket-pr` | `bitbucket-pr/fetch_bitbucket_pr.py` | `BITBUCKET_API_TOKEN` set |

Neither host matched? Tell the user that this skill supports GitHub and Bitbucket only, name the
host found in the remote URL, and stop. Do not guess a host, and do not fall back to either one.

Then list the open PRs with the matching script. Each line shows the `source → dest` branch:

```bash
# GitHub
python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/github-pr/fetch_github_pr.py"
# Bitbucket
python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/bitbucket-pr/fetch_bitbucket_pr.py"
```

The auth precondition for the detected host must be met. If it is not, the script says so and exits
non-zero. Relay that to the user rather than working around it.

The SessionStart hook exports `PROJECT_ROOT` (see the "Hooks" section of `.claude/CLAUDE.md`), and
`.claude/` lives at that project root. The repository under review is usually a worktree under
`workspaces/` with no `.claude/` of its own, so the script is always addressed through
`PROJECT_ROOT`. The repository it queries comes from the `origin` remote of the current working
directory, so run the command from inside that repository. Inline the full path each time, since
shell variables do not persist between commands.

The **list** command (no `--pr`) is the only output carrying branch names. Match the PR whose
**source branch** equals the current branch.
- No match → tell the user there is no open PR for this branch and stop.
- More than one → ask which PR.

Then fetch that PR's comments once with the same script and reuse the output (do not re-fetch):

```bash
# GitHub
python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/github-pr/fetch_github_pr.py" --pr <id>
# Bitbucket
python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/bitbucket-pr/fetch_bitbucket_pr.py" --pr <id>
```

Both scripts print the same Markdown shape, so every later step reads either one the same way.

### 2. Parse requested changes into records

From the fetched Markdown, keep only **actionable** comments. Drop approvals and praise ("looks
good"), the author's own non-actionable notes, and anything already resolved (resolved threads are
excluded by default, `--include-resolved` brings them back). **Group comments that target the same
problem** (same file/line cluster, or the same theme raised in several places) into one record:

```
[C1] <one-line problem statement>
  source:     <author>, copying the location the comment carries: <file:line> for an inline
              comment with a line, <file> when it has a path but no line, else "general PR
              comment" (list every grouped comment)
  goal:       <the desired end state, not the symptom>
  complexity: trivial | moderate | complex
  approach:   <1-2 line plan for the fix>
```

Present the full list of records to the user as a checkpoint, then proceed in the same turn. This is
an FYI, not an approval gate for trivial and moderate items. The only mandatory stop is for
`complex` records (step 3). The user may interject after seeing the list.

### 3. Apply the fixes

Classify by `complexity` and act accordingly:

| Complexity | Examples | Action |
|------------|----------|--------|
| **trivial** | typo, docstring, dead-code removal, mechanical lint fix, single-line edit, renaming a **local** variable or symbol | Apply directly. |
| **moderate** | localized logic change in one function or file, added guard, small refactor, renaming a module or public name whose imports touch **a few** call sites | Apply directly, then note the change. |
| **complex** | cross-cutting refactor, signature or API change touching **many** call sites, new abstraction, architecture or design decision, ambiguous intent | **Stop and ask the user to approve the approach before implementing.** |

A "rename" is only trivial when it is a local symbol. Renaming a module or public name cascades to
its import sites, so count them first, then classify as moderate (few) or complex (many).

**Always ask, never assume** when a comment touches a coordinate frame, units, or transform
direction, or when the requested change is genuinely ambiguous, regardless of size. This is a
standing rule of the project: `.claude/CLAUDE.md` states "**NEVER** assume coordinate frames, if
ambiguous or unclear, ask." A reviewer's wording is not evidence of the frame they meant.

Follow the repo conventions in `.claude/CLAUDE.md` and `.claude/rules/` for every edit. Read only the
files the comments reference, and do not explore the whole repo.

After editing code, run the pre-submission gate once over the affected package. `<package>` is the
top-level source directory containing the edited files, meaning the import root:

```bash
conda activate main
ruff check --fix <package>/ && ruff format <package>/ && ty check <package>/
```

Run `pytest <package>/tests/` in the same environment when the changes affect tested behavior.

### 4. Summary

Print a table mapping every requested change to its outcome:

| # | Comment (author · file:line) | Requested change | Action taken | Status |
|---|------------------------------|------------------|--------------|--------|
| C1 | … | … | … | fixed / awaiting approval / skipped (reason) |

State which packages passed the lint, type and format gate. Do not push or post anything unless
asked.

## Token discipline

- Fetch the PR exactly once and reuse the Markdown.
- Open only files referenced by comments, never sweep the repo.
- Run the lint and type gate once at the end, or once per touched package, not after each edit.
- Keep records terse, one line per field.

## Common mistakes

- **Re-fetching or re-listing PRs repeatedly.** List once to find the PR, fetch once for comments.
- **Implementing complex or design changes without approval.** Those require a go-ahead first.
- **Treating every comment separately.** Group related ones so a single fix resolves a cluster.
- **Blindly applying questionable feedback.** Verify correctness first (see required background).
- **Posting back to GitHub or Bitbucket.** This skill is read-only, and the summary stays local.
