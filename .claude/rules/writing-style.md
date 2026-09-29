# Writing Style Conventions

Governs the prose of every response, commit message, docstring paragraph and document written in
this repository. Prose carries information or it is cut. Each rule below names a specific word or
mark, so a draft can be checked against it mechanically rather than judged.

These rules bind prose only. Mathematical notation follows `math-unicode.md`, and quoted code,
commands, error messages, URLs and paths stay verbatim per `code-minimalism.md`.

---

## Punctuation

- Split the sentence in two, or join it with a comma or a colon, wherever an em dash would go.
- Write two sentences wherever a semicolon would join two independent clauses.
- Use a dash only as a delimiter inside a definition list, a symbol binding or a table cell.

```text
Good  The build failed on the second pass. The cache was stale.
Bad   The build failed on the second pass, the cache was stale — as expected.
```

---

## Register

- State a magnitude with its measurement beside it, and cut `significantly`, `dramatically`,
  `substantially`, `vastly`, `massively` and `orders of magnitude` wherever no number follows.
- Cut these openers: `It is worth noting`, `Importantly`, `Essentially`, `Fundamentally`,
  `At its core`, `In essence`, `Let us`, `Let's`.
- Cut these adjectives and verbs: `powerful`, `seamless`, `robust`, `elegant`, `comprehensive`,
  `cutting-edge`, `game-changing`, `delve`, and `leverage` used as a verb.
- Name the thing plainly wherever a sentence reaches for `not just X, but Y` or `it is not about X,
  it is about Y`.
- Write a list of three only when there are three items, never to close a paragraph on cadence.

```text
Good  The second pass runs 4.2× faster than the first.
Bad   The second pass is dramatically faster, delivering a powerful speedup.
```

---

## Response structure

- Open with the answer, never with a restatement of the question.
- Close at the last piece of new information, and never restate what the response already said. A
  note on what was deliberately not built is new information, and `code-minimalism.md` requires it.
- Reserve bold for a list label or for the one claim a section turns on.
- Use no emoji in prose, commit messages, code comments or documentation. The one exception is the
  `##` section headings of a README, which follow `readme-conventions.md`.

---

## Quick Checklist Before Sending Prose

- [ ] No em dash and no semicolon outside a definition list, symbol binding or table cell
- [ ] Every magnitude word carries a number beside it
- [ ] None of the listed openers, adjectives or verbs appears
- [ ] No `not just X, but Y` construction, and no three-item list built for cadence
- [ ] The first sentence answers the question rather than restating it
- [ ] The last sentence carries new information rather than a summary
- [ ] Bold appears only on a list label or a section's single load-bearing claim, and no emoji outside a README `##` heading
