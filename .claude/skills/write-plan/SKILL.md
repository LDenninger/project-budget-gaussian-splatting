---
name: write-plan
description: Use when a spec or set of requirements exists for a multi-step task, before touching code
---

# Write Plan

## Overview

Write comprehensive implementation plans assuming the engineer has zero context for our codebase and questionable taste. Document everything they need to know: which files to touch for each task, code, testing, docs they might need to check, how to test it. Give them the whole plan as bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Assume they are a skilled developer, but know almost nothing about our toolset or problem domain. Assume they don't know good test design very well.

**Announce at start:** "I'm using the write-plan skill to create the implementation plan."

**Context:** If working in an isolated worktree, it should have been created via the `using-git-worktrees` skill at execution time.

**Save plans to:** `<project-root>/docs/plans/YYYY-MM-DD-<branch-name>-<feature-name>.md`
- The project-root `docs/plans/` is shared by every worktree under `workspaces/`. Resolve the root
  and never create `docs/plans/` inside a worktree. Read `.claude/rules/docs-rules.md` before
  writing the file.
- Read the source spec from `<project-root>/docs/specs/` for the same reason.
- (A plan location the user names in the conversation overrides this default)

## Scope Check

If the spec covers multiple independent subsystems, it should have been broken into sub-project specs by the brainstorm skill. If it wasn't, suggest breaking this into separate plans — one per subsystem. Each plan should produce working, testable software on its own.

## File Structure

Before defining tasks, map out which files will be created or modified and what each one is responsible for. This is where decomposition decisions get locked in.

- Design units with clear boundaries and well-defined interfaces. Each file should have one clear responsibility.
- You reason best about code you can hold in context at once, and your edits are more reliable when files are focused. Prefer smaller, focused files over large ones that do too much.
- Files that change together should live together. Split by responsibility, not by technical layer.
- In existing codebases, follow established patterns. If the codebase uses large files, don't unilaterally restructure - but if a file you're modifying has grown unwieldy, including a split in the plan is reasonable.

This structure informs the task decomposition. Each task should produce self-contained changes that make sense independently.

## Task Right-Sizing

A task is the smallest unit that carries its own test cycle and is worth a
fresh reviewer's gate. When drawing task boundaries: fold setup,
configuration, scaffolding, and documentation steps into the task whose
deliverable needs them; split only where a reviewer could meaningfully
reject one task while approving its neighbor. Each task ends with an
independently testable deliverable.

## Bite-Sized Task Granularity

**Each step is one action (2-5 minutes):**
- "Write the failing test" - step
- "Run it to make sure it fails" - step
- "Implement the minimal code to make the test pass" - step
- "Run the tests and make sure they pass" - step
- "Commit" - step

## Plan Document Header

**Every plan MUST start with this header.** If no spec document exists — `brainstorm` can hand
off a design with no spec written — put `**Spec:** none — requirements captured inline below` and
write the constraints straight into Global Constraints.

```markdown
# [Feature Name] Implementation Plan

> **Executing this plan:** work it task-by-task, either with the `implement-plan-using-subagents` skill (a fresh subagent per task) or the `implement-plan-inline` skill (one session). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

**Spec:** [path to the spec/design doc this plan implements — the plan
argues from the spec, so the spec travels with it; executors read both]

## Global Constraints

[The spec's project-wide requirements — version floors, dependency limits,
naming and copy rules, platform requirements — one line each, with exact
values copied verbatim from the spec. Every task's requirements implicitly
include this section.]

---
```

## Task Structure

````markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Modify: `exact/path/to/existing.py:123-145`
- Test: `tests/exact/path/to/test.py`

**Interfaces:**
- Consumes: [what this task uses from earlier tasks — exact signatures]
- Produces: [what later tasks rely on — exact function names, parameter
  and return types. A task's implementer sees only their own task; this
  block is how they learn the names and types neighboring tasks use.]

- [ ] **Step 1: Write the failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/path/test.py::test_name -v`
Expected: FAIL with "function not defined"

- [ ] **Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/path/test.py::test_name -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## No Placeholders

Every step must contain the actual content an engineer needs. These are **plan failures** — never write them:
- "TBD", "TODO", "implement later", "fill in details"
- "Add appropriate error handling" / "add validation" / "handle edge cases"
- "Write tests for the above" (without actual test code)
- "Similar to Task N" (repeat the code — the engineer may be reading tasks out of order)
- Steps that describe what to do without showing how (code blocks required for code steps)
- References to types, functions, or methods not defined in any task

## Self-Review

After writing the complete plan, look at the spec with fresh eyes and check the plan against it. This is a checklist you run yourself — not a subagent dispatch.

**1. Spec coverage:** Skim each section/requirement in the spec, or the requirements as stated in this conversation when there is no spec file. Can you point to a task that implements it? List any gaps.

**2. Placeholder scan:** Search your plan for red flags — any of the patterns from the "No Placeholders" section above. Fix them.

**3. Type consistency:** Do the types, method signatures, and property names you used in later tasks match what you defined in earlier tasks? A function called `clearLayers()` in Task 3 but `clearFullLayers()` in Task 7 is a bug.

If you find issues, fix them inline. No need to re-review — just fix and move on. If you find a spec requirement with no task, add the task.

**For a large or high-stakes plan,** follow the self-review with an independent pass: dispatch a `general-purpose` subagent using the template at [plan-document-reviewer-prompt.md](plan-document-reviewer-prompt.md). A fresh reader catches spec gaps and naming drift that you will read past in your own draft.

## Execution Handoff

Writing the plan is the deliverable. Execution is a separate decision, and it
is the user's — do not roll straight into implementing.

After saving the plan, report where it landed and ask which of three the user
wants:

**"Plan saved to `<absolute path>`. How do you want to proceed?**

**1. Subagent-driven** — a fresh subagent per task, reviewed between tasks. Keeps this session's context free; best for a long plan with independent tasks.

**2. Inline** — work the tasks in this session, with checkpoints. Simpler; best for a short plan or one where the tasks share a lot of context.

**3. Stop here** — the plan stands on its own. Pick it up whenever, in whatever session."**

Then follow the answer:

| Answer | Do this |
|---|---|
| Subagent-driven | Use the `implement-plan-using-subagents` skill |
| Inline | Use the `implement-plan-inline` skill |
| Stop here | Stop. Do not start implementing. |

Stopping is a legitimate outcome, not a failure — a written plan is a complete
deliverable on its own. If the user does not answer, treat it as *stop here*
rather than picking a default and running.
