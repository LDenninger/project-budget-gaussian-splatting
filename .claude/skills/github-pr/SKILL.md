---
name: github-pr
description: "Use when the user wants to read or act on a GitHub pull request or its review comments for the repository they are working in, for example \"what are the comments on PR 42\", \"show the open PRs\", \"summarize the review feedback\". Triggers on: show the open PRs, list pull requests, what did the reviewer say, read PR <id>, fetch the PR comments, is there a PR for this branch, /github-pr."
---

# GitHub Pull Requests

Read pull requests and their threaded review comments from GitHub. The skill ships
`fetch_github_pr.py`, which shells out to the `gh` CLI and renders the result as Markdown on stdout.
The output shape is the same as the `bitbucket-pr` sibling's, so `fix-pr` parses either one.

**Announce at start:** "I'm using the github-pr skill."

## Prerequisites

- Run from anywhere inside the target git repo. The `owner/repo` slug comes from its `origin`
  remote by way of `gh`, so subdirectories and git worktrees work too.
- The `gh` CLI must be installed. If it is missing, point the user at https://cli.github.com.
- `gh` must be authenticated. If `gh auth status` fails, tell the user to run `gh auth login`. No
  token is read from the environment. Authentication is entirely `gh`'s job.

The script reports a clear error and exits non-zero when a precondition is unmet. Relay that to the
user rather than working around it.

## How the script is located

The skill lives at the project root, under `.claude/skills/`, while the repository under review is
usually a worktree in `workspaces/` that has no `.claude/` of its own. The SessionStart hook exports
`PROJECT_ROOT`, so every command below resolves the script through that variable and falls back to
the current repo root when it is unset:

```text
"${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/github-pr/fetch_github_pr.py"
```

The *repository* the script talks to is a separate question. It comes from the `origin` remote of
the working directory, so running the command from inside a worktree targets that worktree's repo.
Inline the whole path each time. Shell variables you set do not persist between commands.

## Workflow

1. **The user named a PR** by number, or context makes one unambiguous. Fetch it directly:
   ```bash
   python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/github-pr/fetch_github_pr.py" --pr <id>
   ```
2. **No PR named. Do not guess.** List the open PRs and ask the user which one:
   ```bash
   python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/github-pr/fetch_github_pr.py"
   ```
   Present the list, wait for the user to choose, then fetch that PR as in step 1.
3. The fetch prints Markdown to stdout: header, description, threaded comments. Read it directly,
   then summarize or act on the comments as the user asked.

## Quick reference

| Goal | Command suffix |
|---|---|
| List every open PR | no flag |
| Fetch one PR with unresolved comments | `--pr <id>` |
| Fetch one PR with resolved threads too | `--pr <id> --include-resolved` |

## Notes

- Do not reach for `gh pr view --comments` instead of the script. That view shows the conversation
  tab only, so it silently drops every inline review comment anchored to a file and a line, which is
  the feedback a reviewer actually wants addressed. The script merges three separate streams:
  issue comments, review summary bodies, and inline review comments.
- Resolved review threads are hidden by default. Add `--include-resolved` when the user wants the
  full history. Resolution is not in GitHub's REST comment endpoints, so the script reads it from
  the GraphQL `reviewThreads` connection. A thread is filtered out only when GitHub reports it
  resolved, never by guessing from the reply text.
- A review that carries no body of its own is dropped. GitHub creates one as the container around a
  batch of inline comments, and those inline comments are already rendered on their own.
- Output goes to stdout and no file is written. To keep it, the user can redirect it.
- The skill only reads. It does not post comments, resolve threads, approve, or merge.
