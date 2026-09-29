---
name: implement-plan-inline
description: Use when implementing a written plan from docs/plans/ directly in this session. Best for short plans, or plans whose tasks share a lot of context. For a long plan with independent tasks, use implement-plan-using-subagents instead.
---

# Implement Plan Inline

## Overview

Load plan, review critically, execute all tasks, report when complete.

**Announce at start:** "I'm using the implement-plan-inline skill to implement this plan."

**Note:** This skill executes the plan inline, in the current session. When the tasks are independent enough to hand off one at a time, `implement-plan-using-subagents` keeps this session's context free and is usually the better choice.

## The Process

### Step 1: Load and Review Plan
1. Ensure an isolated workspace: use `using-git-worktrees` to create one, or verify the existing one
2. Read the plan file from `<project-root>/docs/plans/`, and the spec it names from
   `<project-root>/docs/specs/`. A worktree under `workspaces/` carries no `docs/plans/` of its
   own, so resolve the root per `.claude/rules/docs-rules.md` before globbing for the plan
3. Review critically - identify any questions or concerns about the plan
4. If concerns: Raise them with your human partner before starting
5. If no concerns: Create todos for the plan items and proceed

### Step 2: Execute Tasks

For each task:
1. Mark as in_progress
2. Follow each step exactly (plan has bite-sized steps)
3. Run verifications as specified
4. Mark as completed

### Step 3: Review

After the last task, dispatch a reviewer with the `requesting-code-review` skill
over the whole change. Act on the findings with `receiving-code-review` — verify
each against this codebase before implementing it.

### Step 4: Complete Development

After all tasks complete, reviewed and verified:
- Verify the full test suite passes, then ask your human partner how they want to integrate the work — merge, open a PR, or leave the branch for review.
- Do not pick the integration route yourself.

## When to Stop and Ask for Help

**STOP executing immediately when:**
- Hit a blocker (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- You don't understand an instruction
- Verification fails repeatedly

**Ask for clarification rather than guessing.**

## When to Revisit Earlier Steps

**Return to Review (Step 1) when:**
- Partner updates the plan based on your feedback
- Fundamental approach needs rethinking

**Don't force through blockers** - stop and ask.

## Remember
- Review plan critically first
- Follow plan steps exactly
- Don't skip verifications
- Reference skills when plan says to
- Stop when blocked, don't guess
- Never start implementation on main/master branch without explicit user consent
