# Rule File Conventions

Rules for writing the files an agent reads as standing constraints: `CLAUDE.md` and everything under
`.claude/rules/`. An always-loaded rule spends context on every turn of every session in the repo,
whether or not the turn touches its subject, and it spends attention that the task itself needs.
Write every line to earn that.

Where this file collides with the rule it governs, this file wins on *form*; the other file wins on
*content*.

---

## Rule or Skill

A **rule** is a standing constraint: it holds on every turn, and the agent must already know it to
avoid doing the wrong thing. A **skill** is a procedure invoked when a named task comes up.

| | Rule (`.claude/rules/*.md`) | Skill (`.claude/skills/*/SKILL.md`) |
|---|---|---|
| Loaded | Always, or when a matching file is opened | Name and description always; body on invocation |
| Answers | "How must code look here?" | "How do I do X?" |
| Triggered by | Nothing — it is simply true | A task the description matches |
| Cost | Every turn | Only the turns that need it |
| Wrong shape | A workflow nobody runs most days | A constraint that must hold before the agent thinks to ask |

Decide with these three tests, in order:

1. **Would a violation be silently wrong?** A constraint the agent breaks without noticing —
   transform naming, IO routing, a docstring format — must be a rule. It has no trigger phrase.
2. **Does it have a trigger?** If you can write "use when the user asks to …", it is a skill. Ship
   it as one and let the description do the retrieval. A section of a rule file that has grown into
   a multi-step procedure has become a skill and should move.
3. **Is it reference material?** Tables, glyph inventories, API catalogues and command listings are
   looked up, not obeyed. They belong in a file the agent opens on demand, with one line in the
   rule saying when to open it.

Never state in a rule file what the harness already injects. Skill names and descriptions reach the
agent from skill frontmatter on every session; a table of them in `CLAUDE.md` is duplicated context
that goes stale the day a skill is added.

Never write a rule to enforce what a hook or a CI gate can block. An instruction is a request, not a
guarantee. Anything whose violation is unacceptable — a destructive command, a secret, a `terraform
apply` — goes in a `PreToolUse` hook or the pre-push gate, and the rule file merely says so.

**Prefer the additive form to the stop.** A rule to do something extra leaves evidence in the diff;
a rule to refuse or defer fails silently and cannot be checked against one. Write the additive form
wherever you can, and back every genuine stop with a hook, a permission deny rule or a CI gate.

---

## Write One Rule

Every rule is one imperative sentence naming an action and its object. If a reader cannot tell from
the sentence alone whether a given diff violates it, it is not a rule yet.

Test each one by handing it to a colleague with no context and asking them to apply it. If they
would have to ask you a question, the agent will guess instead.

- **Lead with the verb.** `Route every write through dsc_lib.io.` Not `IO should generally go
  through the library where possible.`
- **Say what to do, then what not to do.** State the required behaviour first and the prohibition
  as its boundary: `Name every transform tf_A_from_B; never write a bare pose in composition code.`
  A rule that is only a prohibition leaves the correct action unnamed. A bare `never X` is the weak
  form — not because the negation is wrong, but because it names no replacement. Pair it with the
  positive behaviour, or with the reason the prohibition exists, and it is as strong as any rule.
- **Name the observable.** A rule must point at something present in the diff: a symbol, a path, a
  file name, a decorator, a command, a number. `Keep functions small` names nothing. `A test body
  contains no if, no for and no computed expectation` names three.
- **Quantify every threshold.** No `short`, `large`, `reasonable`, `where appropriate`, `if
  possible`, `try to`. Pick the number and commit; a wrong number gets corrected, a missing one
  gets ignored.
- **State the exception where the rule is.** A rule with an unwritten exception gets broken quietly
  by everyone who has met the exception. Write it as a bounded carve-out with its condition, not as
  hedging in the main sentence.
- **One rule per bullet.** Two constraints joined by *and* are followed half the time.
- **Give the reason only when it changes behaviour.** `Patch the name the code under test looks up
  — patching the definition site silently does nothing and the test still passes` earns its clause:
  it tells the reader how the failure presents. Rationale that only justifies the rule to a human
  belongs in the commit message.

### Emphasis is a scarce resource

Bold, `IMPORTANT`, `ALWAYS`, `NEVER` and `MUST` work by contrast. A file where every line shouts has
no emphasis at all, and neither does a file where the same word opens forty bullets. Set the default
register in the file's opening line — *these rules are binding* — and reserve markup for the two or
three rules whose violation is unrecoverable. If a rule is repeatedly ignored, escalate that one
line rather than the whole file.

### Prefer the check to the prose

If a linter, formatter, type checker or CI gate already enforces it, do not restate the rule — name
the command that enforces it and move on. A rule restating a `pyproject.toml` setting has two
sources of truth and drifts from the real one without failing anything.

Restating the linter is the defect this section exists to stop. Write a rule only for what tooling
cannot see: naming semantics, domain conventions, what belongs in a file at all, and the judgement
calls in between.

**Generate or lint anything that can drift.** A command list, a hook id, a config value or a file
path stated in prose will eventually be wrong. Either generate the block from its source between
markers, or add a check that resolves every path, command and flag the file mentions. A rule file
that cannot go stale beats one that is merely up to date today.

### Never write these

The disqualifying shapes are listed once, in **The Accountability Pass** — a rule you would delete
in an audit is a rule not worth writing. Check a draft against that table before submitting it.

---

## Structure

Every rule file follows this shape:

```
# <Domain> Conventions          ← title names the domain, not the file

<Two to four lines: what this governs, and the principle the rules discharge.>

<Precedence line, if this file yields to another.>

---

## <Section>                    ← one concern per section, ordered most-binding first

---

## Quick Checklist Before <action>

- [ ] <one line per binding rule, in the order the file states them>
```

- **Title, opening, precedence, sections, checklist.** In that order. Every rule file created or
  substantially revised from now takes this shape; see **Conformance** for where the set stands.
- **`---` between every top-level `##` section**, never inside one.
- **Order sections by how often the rule binds**, not by the lifecycle of the task. What the agent
  needs on most turns goes first; the rare carve-out goes last.
- **Depth stops at `###`.** A fourth level means the file is two files.
- **Bullets for rules, tables for taxonomies, fenced code for examples, prose only to set up a
  section.** A rule buried in a paragraph is not found.
- **Wrap prose at 100 characters.** Tables and code fences run as long as they need to.
- **British spelling** in prose; API names keep their real spelling (`parametrize`, `visualizer`).
- **No meta-narration.** Nothing about when the file was written, what it replaced, or why it
  changed. A rule file addresses a reader who has never seen its history.
- **Write in the register the rule demands.** A file demanding terse imperative code is written in
  terse imperative prose.

### The checklist is the contract

The closing checklist is not a summary — it is the file's testable surface and the agent's last pass
before submitting. Every binding rule appears in it as one line phrased as something
a reader can confirm or deny against a diff. A rule that cannot be written as a checklist line is
too vague to keep; a checklist line with no rule above it is an orphan. Both are audit findings.

---

## Length and Loading

**Count rules, not lines.** What binds an agent is the number of constraints in play, not the page
count. Line budgets are a forcing function on rule count and an aid to review and findability;
treat them as discipline, and measure the effect of any rule against the task it governs rather
than assuming it.

- **`CLAUDE.md` stays under 200 lines.** It carries the repo's identity, its commands, its hard
  constraints and pointers — nothing that belongs in a rule file.
- **One rule file stays under 500 lines.** Past that, split or cut.
- **Cut the rule before you cut the words.** Removing one rule buys more than compressing ten.
- **The always-loaded closure stays under 2 000 lines.** Measure it — `wc -l` over `CLAUDE.md` plus
  every file it `@`-imports, transitively — and record the number in the audit. A file that merely
  sits in `.claude/rules/` is not loaded; only the `@` import loads it. Counting by directory
  overstates the closure and hides which files actually cost anything.
- **`@path` imports do not save budget.** An imported file loads at launch exactly as if pasted in.
  Splitting a long file into imports improves navigation and changes nothing about cost.
- **Drop the `@` to drop the cost.** A file referenced by a backticked path is read only when the
  agent needs it; a file behind `@` is loaded every turn. Moving a conditional file from `@` to a
  backticked pointer is the only change that reduces the always-loaded closure.
- **Register a repo-specific file in `_REPO_SUFFIX_RULES`** in `dsc_cli/apps/claude.py`, so
  `dsc claude init` links it only into repos whose name carries a matching suffix. This controls
  which repos receive the file, not what loads — a suffix-scoped file still costs its full length in
  any repo that both receives it and `@`-imports it.
- **Reference material leaves the rule set entirely.** A lookup table longer than the rules that use
  it moves to its own file, referenced by a backticked path — never `@`-imported — with one line
  saying when to open it.

### Conformance

`testing.md` and `code-style.md` carry a closing checklist; `code-minimalism.md` declares its own
precedence. No other file in the set does either. `math-unicode.md` is 479 lines, roughly 28% of
everything loaded and majority glyph tables; it is `@`-imported today, in breach of the
reference-material rule above, until its tables are split out and the remainder de-imported.
Bring each file to this shape when you next edit it,
and record the disposition in the audit — a rule file is not exempt because it predates this one.

When a file crosses its cap, cut before you split. Restated tool config, duplicated constraints,
derivable facts and unfalsifiable adjectives usually account for the whole overage.

---

## Cross-References and Precedence

- **Every rule lives in exactly one file.** When a second file needs it, link — `see
  code-style.md` — and never restate it. Two copies become two rules.
- **Reference by file name and section**, both exact. A pointer to a file, flag or symbol that no
  longer exists discredits every other line in the file.
- **Distinguish the two reference forms.** `@path` means *always loaded*; a backticked path means
  *read when relevant*. Use the difference deliberately, and never `@`-import a file the agent needs
  on a minority of turns.
- **Declare precedence in the subordinate file, not the dominant one.** A file that yields says so
  in its opening lines: *Where this collides with `x.md`, `x.md` wins.* The dominant file stays
  silent, so precedence is stated once and cannot drift.
- **Resolve contradictions rather than ranking them.** Precedence covers a rule deliberately
  narrower in one domain. It does not cover two rules that simply disagree — faced with those the
  agent picks one arbitrarily, so the fix is always to delete one.
- **Order of authority**, highest first: an explicit instruction in the current turn, a repo-local
  `CLAUDE.md`, then the shared rule files on content and this file on form. Nothing in a rule file
  overrides the user.

---

## Examples

Examples carry a rule further than its statement does, and they cost more budget than any other
content. Spend them where the prose is genuinely underdetermined.

- **Show a rule that has a shape, not a rule that has a value.** A naming convention, a file layout,
  a call pattern earns an example. `Line length is 120` does not.
- **Pair Good with Bad, and label both.** The bad case makes the boundary legible; a lone good
  example reads as one acceptable option among many.
- **Make the pair minimal and differ in one axis.** Everything the two examples share is noise the
  reader must subtract.
- **Examples are real code.** Copy from the codebase, keep the real names, keep them syntactically
  valid — the agent pattern-matches them straight into its output.
- **One pair per rule.** A third example adds budget, not clarity.
- **Never let an example carry a rule the prose omits.** Anything shown only in code is not a rule
  and will be followed inconsistently.

```python
# Good — fixture composes other fixtures
@pytest.fixture
def sfo_annotations(sfo_postprocessed_annotations_path: Path) -> Annotations:
    return Annotations.load(str(sfo_postprocessed_annotations_path))

# Bad — path reconstructed inline
@pytest.fixture
def sfo_annotations() -> Annotations:
    return Annotations.load(str(DATABATCHES_ROOT / '...' / '...'))
```

---

## The Accountability Pass

Audit a rule file whenever it is edited, the whole set once a quarter, and every file again after a
major model release — a rule written around an older model's weakness becomes pure overhead once the
weakness is gone. Every file has one named owner; an unowned rule file is deleted, not adopted. Work
through the file rule by rule and record a disposition for each: **keep**, **rewrite**, **move**,
**delete**.

The test for every line, applied one line at a time: *would removing this cause the agent to make a
mistake?* If not, cut it.

### Delete on sight

| Name | How to recognise it |
|---|---|
| **Ghost rule** | References a file, flag, command or symbol that no longer exists |
| **Echo rule** | The same constraint already stated in another rule file, or twice in this one |
| **Config mirror** | Restates a value that `pyproject.toml`, pre-commit or CI already enforces |
| **Vibe rule** | Contains no observable — no symbol, path, number or command |
| **Default rule** | The agent would do this without being told |
| **Aspiration rule** | The codebase it governs does not follow it, and no one intends to fix that |
| **Narration** | States a fact about the project with no action attached |
| **Orphan checklist line** | Appears in the checklist with no rule above it, or a rule with no line |
| **Fossilised rule** | Written once by `/init` or a first draft and never revised since |
| **Unenforceable stop** | Tells the agent to refuse or defer with no hook or gate behind it |
| **Hedged obligation** | `should generally`, `where possible`, `try to prefer` — reads optional, is treated as optional |
| **Derivable fact** | Directory layouts, dependency lists, file-by-file descriptions — the agent can read them |

### Evidence to gather

An audit that only reads the file finds ghosts and echoes and nothing else. Check every rule against
the repository:

- **Ghosts** — resolve every path, file name, flag, command and symbol the file mentions. Do this
  first; it is mechanical, and it is where rot shows up.
- **Contradiction** — for every rule naming a config value, open the config. A rule saying markers
  are registered in `pyproject.toml` is a defect in a repo with no `[tool.pytest.ini_options]`.
- **Adherence** — sample the code the rule governs and count violations. A rule the codebase
  violates everywhere is unenforced, unenforceable, or wrong; decide which and act.
- **Echoes** — grep the distinguishing phrase of each rule across the whole rule directory. More
  than one hit is a finding.
- **Budget** — total the lines of the always-loaded closure and report the number against the caps.
- **Behaviour** — run the task the rule governs twice, with the file and without it, and compare.
  The comparison is the measurement: a run with the file proves nothing alone, because most of what
  passes would pass anyway. A rule never observed to change behaviour is a candidate for deletion,
  not for rewording at greater length.
- **Grade the output, not the route.** Check what landed in the diff, never which tools the agent
  called on the way. Asserting on the path produces brittle checks that fail on harmless variation.

### Fixing

Delete first, then rewrite, then split. A rule you cannot make checkable does not become checkable
by being restated longer. Removing a rule the codebase already ignores costs nothing and buys
attention for the rules that remain.

No backward compatibility: when a rule changes, update every file that referenced it in the same
edit, and never leave a superseded rule standing beside its replacement.

---

## Quick Checklist Before Submitting a Rule File

- [ ] Every entry is a standing constraint, not a procedure with a trigger — procedures are skills
- [ ] Nothing restates what the harness injects (skill names and descriptions), what a linter,
      formatter, type checker or CI enforces, or what a hook can block outright
- [ ] Every unacceptable-violation rule is backed by a hook or a CI gate, not left as prose
- [ ] Nothing states a fact the agent can derive by reading the code
- [ ] Every rule is one imperative sentence, leading with a verb, one constraint per bullet
- [ ] Every rule names an observable — a symbol, path, file name, decorator, command or number
- [ ] Every threshold is a number; no `short`, `large`, `reasonable`, `where appropriate`, `try to`
- [ ] Required behaviour stated before its prohibition; no rule is prohibition-only
- [ ] Exceptions written as bounded carve-outs at the rule, not as hedging in the sentence
- [ ] Emphasis markup reserved for the few unrecoverable rules, not applied line after line
- [ ] No unfalsifiable adjectives, no narration, no aspiration the codebase does not follow
- [ ] Frontmatter, title, opening, precedence, sections, checklist — in that order
- [ ] `---` between every `##` section; sections ordered most-binding first; depth stops at `###`
- [ ] Prose wrapped at 100 characters; British spelling outside API names
- [ ] Every binding rule appears once in the closing checklist; no orphan lines either way
- [ ] `CLAUDE.md` under 200 lines; this file under 500; always-loaded closure under 2 000 and
      measured, not estimated
- [ ] Conditional content is referenced by a backticked path, never `@`-imported; repo-specific
      files are also registered in `_REPO_SUFFIX_RULES`
- [ ] Reference material moved out of the rule set, with one line saying when to open it
- [ ] Each rule stated in exactly one file; second users link rather than restate
- [ ] Precedence declared in the subordinate file; no two rules left contradicting each other
- [ ] Examples paired Good/Bad, minimal, differing in one axis, copied from real code
- [ ] No rule exists only inside an example
- [ ] Every path, flag, command and symbol mentioned in the file resolves today, ideally by a check
      rather than by reading
- [ ] Every rule naming a config value checked against that config
- [ ] Drift-prone blocks generated from their source, not retyped
- [ ] Changed rules compared with and without the file on the task they govern
- [ ] The file has a named owner and has been revised since it was first written
- [ ] No meta-narration anywhere: no history, no rationale for the edit, no reference to the prompt