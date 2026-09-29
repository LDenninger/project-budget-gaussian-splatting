# Documentation Rules

Governs every document written during development: which kind serves which purpose, where it is
written, read and committed. Every worktree under `workspaces/` carries its own tree, so a relative
`docs/` path resolves to a different directory in each one. A project has one set of these documents
and they live at the project root, shared by every worktree.

Where a skill writes a bare `docs/specs/...`, `docs/plans/...` or `docs/devlog/...` path, it means
the resolved project-root path defined here.

---

## Document Kinds

| kind | lives at | written by | holds |
|---|---|---|---|
| spec | `<project-root>/docs/specs/` | `brainstorm` | a design agreed before implementation, named `YYYY-MM-DD-<topic>-design.md` |
| plan | `<project-root>/docs/plans/` | `write-plan` | a task-by-task implementation plan for an engineer with no context |
| devlog | `<project-root>/docs/devlog/` | `write-devlog` | what a session did, every run and metric, negative results included |
| anything else | `<project-root>/docs/` | any | architecture notes, guides, references |
| README | the repository root it describes | any | see `.claude/rules/readme-conventions.md` |

- Write every project document under the top-level `docs/`, never beside the code it describes. The
  README of a repository is the one exception, and it stays at that repository's root.
- Name a spec, plan or devlog with a date prefix, so the directory sorts chronologically.
- Never write a document into `.claude/`, `scratchpad/` or a worktree's own tree when one of the
  directories above fits.
- Scratch and intermediate artifacts are not documentation. They belong in
  `scratchpad/DD_MM_YY_HH_MM_<description>/`, and a plan's working files belong in `.sdd/`, which
  stays in the worktree.

---

## Take the Root from the Session, Resolve Only if it is Missing

`.claude/hooks/startup.py` runs at every session start and publishes the root two ways:

- A first context line, `PROJECT_ROOT=<absolute path>`, above the snapshot of `workspaces/`,
  `docs/specs/`, `docs/plans/` and `references/`. Use that literal path in every Write, Edit, Read
  and Glob call, which perform no shell expansion.
- `env.PROJECT_ROOT` in `<root>/.claude/settings.local.json`, so `$PROJECT_ROOT` resolves inside
  Bash calls. The hook writes it only when the session was launched at the root itself, so a
  session started inside `workspaces/` has the context line and no variable.

Run this when neither is present, in one Bash call, before the first read or write of any of the
three directories, and read the path it prints:

```bash
WORKTREE=$(git rev-parse --show-toplevel)
DIR="$WORKTREE"
while [ "$DIR" != "/" ] && ! { [ -d "$DIR/docs" ] && [ -d "$DIR/workspaces" ]; }; do
  DIR=$(dirname "$DIR")
done
PROJECT_ROOT="$DIR"
[ "$PROJECT_ROOT" = "/" ] && PROJECT_ROOT="$WORKTREE"   # plain repo: fall back to the worktree
echo "$PROJECT_ROOT"
```

- Substitute the absolute path into every later tool call. `<project-root>` in a skill means that
  literal path, never a variable: no file tool expands one, so a Write to
  `$PROJECT_ROOT/docs/specs/x.md` creates a directory named `$PROJECT_ROOT` in the worktree.
- Write every spec to `<project-root>/docs/specs/`, every plan to `<project-root>/docs/plans/` and
  every devlog to `<project-root>/docs/devlog/`, passing the absolute path to the write tool.
- Search those same three absolute directories when looking for an existing spec, plan or devlog.
  A worktree holding no `docs/` of its own is not evidence that no spec exists.
- Create a missing directory with `mkdir -p <project-root>/docs/<specs|plans|devlog>`. Never create
  `docs/specs/`, `docs/plans/` or `docs/devlog/` inside a worktree under `workspaces/`.
- Report the absolute path the document landed at, not the bare `docs/plans/<file>.md` form. Two
  worktrees produce the same relative path and the reader cannot tell which tree you meant.
- Commit the document in the repository that owns the project root, with `git -C <project-root>`.
  When that root is not a git repository, leave the file uncommitted and say so in the same message
  that reports the path.
- Fall back to the worktree root only when no ancestor holds both `docs/` and `workspaces/`, and
  name the fallback root in the report.
- Exception: a location the user names in the conversation wins over this file. State the absolute
  path you used.

---

## Quick Checklist Before Writing or Reading a Document

- [ ] The kind is one of the table's rows, and the target directory is that row's
- [ ] The root came from the session-start context line, `$PROJECT_ROOT`, or the fallback loop
- [ ] Every read, glob and write used the printed absolute path, with no `$PROJECT_ROOT` string in it
- [ ] No `docs/specs/`, `docs/plans/` or `docs/devlog/` directory was created under `workspaces/`
- [ ] The reported path is absolute, and names the fallback root if the worktree was used
- [ ] The commit ran against the project root, or the message says the file is uncommitted
