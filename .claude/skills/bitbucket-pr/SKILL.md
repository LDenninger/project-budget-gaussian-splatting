---
name: bitbucket-pr
description: "Use when the user wants to read or act on a Bitbucket pull request or its review comments for the repository they are working in, for example \"what are the comments on PR 13\", \"show the open PRs\", \"summarize the review feedback\". Triggers on: show the open PRs, list pull requests, what did the reviewer say, read PR <id>, fetch the PR comments, is there a PR for this branch, /bitbucket-pr."
---

# Bitbucket Pull Requests

Read pull requests and their threaded review comments from Bitbucket Cloud. The skill ships
`fetch_bitbucket_pr.py`, which calls the Bitbucket REST API with the Python standard library only.

**Announce at start:** "I'm using the bitbucket-pr skill."

## Prerequisites

- Run from anywhere inside the target git repo. The workspace and the repository slug both come
  from its `origin` remote, so subdirectories and git worktrees work too.
- `BITBUCKET_API_TOKEN` must be set. If it is missing, tell the user to generate a scoped API token
  at https://id.atlassian.com/manage-profile/security/api-tokens and export it. Do not proceed.
- `git config user.email` must be set. It is the Basic-auth username.

The script reports a clear error and exits non-zero when a precondition is unmet. Relay that to the
user rather than working around it.

## How the script is located

The skill lives at the project root, under `.claude/skills/`, while the repository under review is
usually a worktree in `workspaces/` that has no `.claude/` of its own. The SessionStart hook exports
`PROJECT_ROOT`, so every command below resolves the script through that variable and falls back to
the current repo root when it is unset:

```text
"${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/bitbucket-pr/fetch_bitbucket_pr.py"
```

The *repository* the script talks to is a separate question. It comes from the `origin` remote of
the working directory, so running the command from inside a worktree targets that worktree's repo.
Inline the whole path each time. Shell variables you set do not persist between commands.

## Workflow

1. **The user named a PR** by number, or context makes one unambiguous. Fetch it directly:
   ```bash
   python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/bitbucket-pr/fetch_bitbucket_pr.py" --pr <id>
   ```
2. **No PR named. Do not guess.** List the open PRs and ask the user which one:
   ```bash
   python "${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}/.claude/skills/bitbucket-pr/fetch_bitbucket_pr.py"
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

- Resolved comment threads are hidden by default. Add `--include-resolved` when the user wants
  the full history.
- Output goes to stdout and no file is written. To keep it, the user can redirect it.
- The skill only reads. It does not post comments, approve, or merge.
