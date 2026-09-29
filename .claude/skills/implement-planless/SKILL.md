---
name: implement-planless
description: Use when implementing something with no written plan in docs/plans/ — a change small enough not to need one, or work whose context came from a brainstorm earlier in this conversation or a design spec in docs/specs/. For a written plan, use implement-plan-inline or implement-plan-using-subagents instead.
---

# Implement Planless

## Overview

Set up the context and environment for implementation, state the task in one
paragraph, then build. This is a frame, not a workflow — no task ledger, no
plan document, no per-task ceremony.

**Announce at start:** "I'm using the implement-planless skill."

**Note:** The no-plan sibling of `implement-plan-inline` and
`implement-plan-using-subagents`. If a plan exists in the project-root
`docs/plans/`, use one of those instead.

## Step 1 — Gather context, cheaply

In order of cost:

1. **Already in this conversation?** A brainstorm from earlier in this session
   is the context. Do not re-derive what you already agreed.
2. **A spec on disk?** Check `<project-root>/docs/specs/` for a design covering this work. It
   holds the decisions and their reasons. Resolve the root per
   `.claude/rules/docs-rules.md`. The worktree you are in has no `docs/specs/` of its own, so
   a miss inside it says nothing about whether a spec exists.
3. **Otherwise, read the code.** Read the surrounding code, not just the line
   you plan to change.

Stop when you can name the files you will edit. Context past that point is
procrastination.

## Step 2 — Define the task, out loud

State it before editing anything:

- **Goal** — what will be true when this is done
- **Done criterion** — the command, test, or observation that proves it
- **In scope** — the files you expect to touch
- **Out of scope** — the adjacent thing you are deliberately not doing

Confirm it. A wrong shared understanding is cheap to fix here and expensive
once the diff exists. If the user already said all four, say them back in one
line and proceed — do not interrogate.

## Step 3 — Set the environment

- **Never start on `main`/`master` without explicit consent.** Branch, or use
  `using-git-worktrees` for isolation.
- Know how to run the tests before you need them.
- Note which project rules apply to the files in scope (see CLAUDE.md).

## Step 4 — Implement

Work in whatever order the change wants. Keep the diff inside the confirmed
scope; anything else is a new task, offered separately.

When done: run the done criterion, report the result, and ask how to integrate
— merge, PR, or leave the branch. Do not pick that yourself.

## Escalate Instead of Grinding

Planless bets the work fits in one head. Say so and switch when the bet fails:

| Signal | Switch to |
|---|---|
| Approach unclear, or real alternatives to weigh | `brainstorm` |
| Scope spans several subsystems, or the task list stops fitting in your head | `write-plan` |
| It grew past the confirmed task definition | Re-define with the user, then continue or escalate |

Escalating is not failure. Discovering mid-change that this needed a plan is
the normal way to find out.

## Remember

- Context first, task definition second, code third
- Confirm the task definition before the first edit
- The confirmed scope is the boundary — extra work is a separate offer
- Escalate when the bet fails; don't grind
