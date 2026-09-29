# Docstring Rules — Google Format (Strict)

All Python code MUST follow these docstring conventions. No exceptions.

## Format

- Triple double-quotes `"""` only. Never single-quotes.
- Summary on same line as opening `"""`.
- Closing `"""` on its own line for multi-line docstrings.
- Line wrap at 88 characters.
- Present tense, imperative mood: `"""Return the..."""` not `"""Returns the..."""`.

## Required Coverage

- Every public module, class, method, and function MUST have a docstring.
- Private members (`_`-prefixed) MUST have a docstring unless name + signature are fully self-explanatory.
- `@overload` stubs: no docstring. Document only the implementation.
- `@property`: docstring on the getter describes what the attribute represents.

## Section Order (omit sections that don't apply)

```
"""Summary line.

Extended description.

Args:
    param: Meaning and constraints. Continuation lines indented
        8 spaces. Document defaults for optional params.
        Do NOT repeat type annotations.

Attributes:
    attr: Description. (classes only)

Returns:
    Meaning of return value, not just the type.
    For tuples, one indented line per element.

Yields:
    Used instead of Returns for generators.

Raises:
    ValueError: Condition. Only caller-facing exceptions.

Note:
    Caveats, thread-safety, deprecation. 1-3 sentences max.

Example:
    >>> obj = MyClass()
    >>> obj.do_thing(42)
    'result'
"""
```

## Section Rules

**Summary:** One line. Describe *what*, not *how*. No period unless the sentence is complex.

**Extended description:** Separated by blank line. Non-obvious behavior, preconditions, caveats. Do NOT restate the signature.

**Args:** Header is `Args:` — never `Arguments:` or `Parameters:`. 4-space indent per param. Format: `name: Description.` — no type in parens, types live in annotations. Document meaning/constraints/expected values, not type names. Always state default behavior for optional params. For `*args`/`**kwargs`: document the expected contents and meaning.

**Returns:** Header is `Returns:` — never `Return:`. Omit entirely for `-> None`. Describe meaning, not type.

**Raises:** Only exceptions the caller should anticipate. Never document `AssertionError` or internal bugs.

**Class docstrings:** Docstring goes on the class, NOT on `__init__`. Summary describes what the class *represents*. Constructor params under `Args:`. Public instance attrs under `Attributes:`. Do not redocument inherited attrs.

**Module docstrings:** Top of file, before imports. 3-6 sentences covering purpose and main public API.

## Tensor and Array Shapes

Every `torch.Tensor` and `np.ndarray` arg and return MUST include exact shape notation in its description. Use `shape (dim0, dim1, ...)` inline, with symbolic names like `B` (batch), `T`/`S` (sequence), `C` (channels), `H`/`W` (height/width), `D` (feature dim), `N` (count). Use descriptive names when clearer: `num_points`, `vocab_size`.

```python
# ❌
x: Input tensor.

# ✅
x: Input tokens, shape (B, T).
# ✅ return
Token embeddings, shape (B, T, D).
```

## Anti-patterns — NEVER do these

```python
# ❌ Restating the name
def get_count(self) -> int:
    """Get the count."""

# ❌ Type in docstring when annotation exists
def fit(self, points: np.ndarray) -> None:
    """Args:
        points (np.ndarray): The points.
    """

# ❌ Describing implementation instead of behavior
"""Loops over all points and calls np.linalg.norm."""

# ❌ Useless description
"""Args:
    max_iterations: The max iterations.
"""
```

Params must describe **meaning, shape, units, constraints, or valid values** — not just echo the parameter name.