---
name: cpp-docstring-style
description: Use when writing or editing C++ docstrings and comments — strict Doxygen format.
copy_to_claudemd: false
enabled: true
order: 0
tags: [cpp, docstrings]
---
# Docstring Rules — Doxygen Format (Strict)

All C++ code MUST follow these docstring conventions. No exceptions.

## Format

- `/** ... */` Javadoc-style blocks only. Never `///` line comments, never `//!`.
- `@` command prefix only. Never the `\` prefix (`@brief`, not `\brief`).
- Summary starts with `@brief` on the first content line after `/**`.
- Closing `*/` on its own line for multi-line docstrings.
- Line wrap at 88 characters.
- Present tense, imperative mood, ends with a period: `@brief Return the...` not `@brief Returns the...`.
- Use `` `identifier` `` (backticks) for inline code references — never `@c`, `@p`, or `<code>`.

## Required Coverage

- Every public class, struct, method, free function, and enum MUST have a docstring.
- Private/protected members MUST have a docstring unless name + signature are fully self-explanatory.
- Declaration (header) carries the full `@brief`/`@param`/`@return` block. The definition (source file), if separated, carries at most a one-line non-Doxygen `//` comment for implementation notes — never a duplicate Doxygen block.
- `@property`-equivalent getters/setters: docstring on the getter describes what the attribute represents; the setter only documents constraints on the assigned value if any exist.

## Section Order (omit sections that don't apply)

```cpp
/**
 * @brief Summary line.
 *
 * Extended description.
 *
 * @tparam T Meaning and constraints on the template parameter.
 *
 * @param[in] param Meaning and constraints. Continuation lines indented
 *     to align with the description. Document defaults for optional params.
 *     Do NOT repeat the type — it lives in the signature.
 * @param[out] result Meaning of the value written back through this parameter.
 *
 * @return Meaning of the return value, not just the type.
 *     For structs/tuples, one indented line per member.
 *
 * @throws std::invalid_argument Condition. Only caller-facing exceptions.
 *
 * @note Caveats, thread-safety, deprecation. 1-3 sentences max.
 *
 * @code
 * MyClass obj;
 * obj.do_thing(42);
 * @endcode
 */
```

## Section Rules

**Summary:** One line after `@brief`. Describe *what*, not *how*. Ends with a period.

**Extended description:** Separated by a blank `*` line. Non-obvious behavior, preconditions, invariants, ownership semantics. Do NOT restate the signature.

**Template parameters:** Header is `@tparam` — one per template parameter, in declaration order, before `@param`.

**Parameters:** Header is `@param`, with `[in]`, `[out]`, or `[in,out]` qualifying every parameter that is not a plain by-value/const-ref input. Format: `@param[in] name Description.` — no type in parens, types live in the signature. Document meaning/constraints/valid ranges, not type names. Always state default behavior for optional/defaulted params.

**Returns:** Header is `@return` (or `@returns`, pick one and use it consistently within a project — do not mix). Omit entirely for `void`. Describe meaning, not type. For `std::optional`/`std::expected` returns, document both the success and empty/error case.

**Throws:** Header is `@throws`, one entry per exception type the caller should anticipate. Never document exceptions that indicate an internal bug (`std::logic_error` from a violated invariant, assertion failures).

**Class/struct docstrings:** Docstring goes immediately above the `class`/`struct` keyword, NOT on the constructor. Summary describes what the type *represents*. Constructor params are documented on the constructor's own `@param` block. Public data members get an inline `///<` trailing comment or their own `/** @brief ... */` block immediately above the declaration; do not redocument inherited members.

**File docstrings:** Top of file, before includes. `@file <name>` followed by 3-6 sentences covering purpose and the main public API declared in the file.

## Array / Tensor / Matrix Shapes

Every array-like, `std::vector`, `std::span`, `std::array`, or matrix arg and return MUST include exact shape notation in its description. Use `shape (dim0, dim1, ...)` inline, with symbolic names like `B` (batch), `T`/`S` (sequence), `C` (channels), `H`/`W` (height/width), `D` (feature dim), `N` (count). Use descriptive names when clearer: `num_points`, `vocab_size`.

```cpp
// ❌
/**
 * @param[in] x Input tensor.
 */

// ✅
/**
 * @param[in] x Input tokens, shape (B, T).
 * @return Token embeddings, shape (B, T, D).
 */
```

## Anti-patterns — NEVER do these

```cpp
// ❌ Restating the name
/**
 * @brief Get the count.
 */
int get_count() const;

// ❌ Type in docstring when the signature already has it
/**
 * @param[in] points The points (std::vector<Point3f>).
 */
void fit(const std::vector<Point3f>& points);

// ❌ Describing implementation instead of behavior
/**
 * @brief Loops over all points and calls norm() on each.
 */

// ❌ Useless description
/**
 * @param[in] max_iterations The max iterations.
 */

// ❌ Duplicate Doxygen block on both declaration and definition
// header.hpp
/** @brief Compute the norm. @param[in] v The vector. @return The norm. */
double compute_norm(const Vec3& v);
// header.cpp
/** @brief Compute the norm. @param[in] v The vector. @return The norm. */
double compute_norm(const Vec3& v) { ... }
```

Params must describe **meaning, shape, units, constraints, or valid values** — not just echo the parameter name.
