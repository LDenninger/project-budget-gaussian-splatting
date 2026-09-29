---
name: cpp-code-style
description: Use when writing or reviewing C++ code — formatting, naming, and layout conventions.
copy_to_claudemd: false
enabled: true
order: 0
tags: [cpp, style]
---
# Coding Conventions

## Core Principle

> Code is written once but read many times. Always prefer **readability over compactness**.

---

## C++: Standard & Formatting

- **Target C++20.** Do not use C++23-only features unless the project's toolchain and CI are confirmed to support them.
- **Formatter**: **clang-format** (enforced in CI). Line length **160**, indentation **4 spaces**, braces on their own line for functions/classes, same-line for control statements.
- **Strings**: double quotes only (`"..."`), per the language — there is no single-quote string literal in C++.
- **No trailing whitespace**; do not indent empty lines.
- **Headers**: use `#pragma once`, never manual include guards.
- **Includes**, ordered in blocks separated by a blank line, each block alphabetized:
    1. the file's own matching header (for `.cpp` files)
    2. C system headers (`<cstdio>`, ...)
    3. C++ standard library headers (`<vector>`, ...)
    4. third-party library headers
    5. project headers (`"..."`)
- Functions and methods in a class or file should always be semantically sorted with a heading comment of style:
    ```cpp
    //---------------------------------------------------------------------
    // <semantic section heading>
    //---------------------------------------------------------------------
    ```
- In long code passages use comments of style to guide the reader's eyes:
    ```cpp
    //--- <short passage description> ---
    ```
- When working with arrays, vectors, matrices, or tensors and the shape changes, document it as an inline comment like:
    ```cpp
    <code line>  // (<dimensions>)
    ```

## C++: Linting & Static Analysis

- **Linter**: **clang-tidy** (enforced in CI), with checks for `bugprone-*`, `modernize-*`, `performance-*`, `readability-*`.
- **Warnings as errors**: build with `-Wall -Wextra -Wpedantic -Werror` (or `/W4 /WX` on MSVC).
- Prefer `auto` only where the deduced type is obvious from the right-hand side (e.g. `auto it = container.begin();`); do not use `auto` where it hides the type of a value the reader needs to reason about.
- Mark every function that does not modify state `const`; mark every function that could be `constexpr` (or `consteval`) `constexpr`/`consteval`.
- Mark single-argument constructors `explicit` unless implicit conversion is intended.

## C++: Memory & Resource Management

- **Never** use raw `new`/`delete`. Ownership is expressed through smart pointers and containers, not manual lifetime management.
- `std::unique_ptr` for single ownership (the default choice). `std::shared_ptr` only when shared ownership is genuinely required. Pass a non-owning pointer/reference to a function that does not participate in ownership — never a smart pointer just to "be safe."
- Follow **RAII**: every resource (memory, file handle, lock, socket) is owned by an object whose destructor releases it. No manual `open`/`close` or `lock`/`unlock` pairs.
- Prefer standard containers (`std::vector`, `std::array`, `std::string`, ...) over C-style arrays and raw buffers.
- Prefer `std::span` (or a project-provided view type) over a raw `(pointer, length)` pair for non-owning array views.

## C++: Library Preferences

- **Always** use `std::filesystem::path` for path variables.
- Prefer the C++ Standard Library and `std::ranges`/`std::views` (C++20) over hand-rolled loops and algorithms.
- Whenever possible, use `opencv2` for image-related tasks.
- Whenever possible, use `open3d` for 3D-related tasks.

---

## Naming Conventions

### General

- `snake_case` — functions, variables, namespaces, files
- `PascalCase` — classes, structs, enums, enum values, type aliases, template parameters
- `SCREAMING_SNAKE_CASE` — constants (`constexpr`/`const` at namespace or class scope) and macros
- **No single-letter variable names.**
- **Function names start with a verb.**
- Iteration variables are `ii`, `jj`, `kk` for nested loops.
- Private/protected member variables carry a trailing underscore: `member_`.
- Macros are used only when a language feature cannot achieve the same result (prefer `constexpr`, templates, and inline functions).

### Path variables (strict)

| Suffix       | Meaning                                              |
|--------------|------------------------------------------------------|
| `*_dir`      | Path to a **directory**                              |
| `*_root`     | Path to a directory **containing sub-directories**   |
| `*_path`     | Path to a **local** or **remote** file               |
| `*_uri`      | Path to a **remote** file or directory               |
| `file_name`  | A bare file name, **no slashes**                     |

### Class method naming

- Save: `save_to_{format}`, `save_{part}`
- Load: `load_from_{format}`, `load_{part}`
- Convert: `to_{type}`, `from_{type}`
- Private helpers: same verb-first `snake_case`, no leading underscore (reserved by the standard for implementation names) — rely on the `private:`/`protected:` section instead.

### Array / tensor shapes

- `shape` = `(height, width)` — matches the numpy convention used elsewhere in this codebase, for consistency across Python and C++ sides of the same project.
- `size`  = `(width, height)` — Pillow convention, same rationale.

---

## Coordinate Frames & Transforms

### Transform variable naming (strict)

- Use the form `tf_A_to_B` / `tf_A2B` for a transform that maps a point in frame `A` into frame `B`.
  - `tf_w2c` — world → camera
  - `tf_c2w` — the inverse
- For composition, **always prefer the explicit `tf_A_to_B` / `tf_A2B` form** over ambiguous names.

### Naming Concept

- `camera_pose`, `*pose*`, ... — by definition, a pose is the transform from **camera → world**.
- `*extrinsic*`, ... — by definition, a pose is the transform from **world → camera**.
- Everything else, is a `transform` / `tf`.

---

## Classes & Object Design

- One class per header/source pair, named after the file: `my_class.hpp` / `my_class.cpp` define `MyClass`.
- Declare members in this order within each access section: type aliases, constructors/destructor, public methods, then data members. Order access sections `public:`, `protected:`, `private:`.
- Rule of five: if a class defines any of destructor, copy constructor, copy assignment, move constructor, or move assignment, it must consider all five explicitly (`= default` / `= delete` is an explicit decision).
- Prefer composition over inheritance. When inheriting for polymorphism, mark the base destructor `virtual` and derived overrides `override` (never re-state `virtual` on an override).
- Avoid multiple inheritance except for interfaces made entirely of pure virtual functions.

## Error Handling

- Use exceptions for exceptional, non-local error conditions (I/O failure, invalid construction). Use `std::expected<T, E>` (C++23) or a project result type for expected, local failure paths when the toolchain supports it; otherwise return an error/status alongside the value.
- Never use error codes and exceptions interchangeably within the same API — pick one discipline per interface.
- Never let a destructor throw.

## Runnable Executables

Every runnable C++ program **must** follow this exact structure:

```cpp
int run_<program_name>(const Args& args);

struct Args {
    // parsed command-line options
};

Args parse_args(int argc, char** argv);

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    return run_<program_name>(args);
}
```

Rules:

- The main function is named `run_<program_name>` after the executable target (the file's stem), and returns the process exit code.
- Argument parsing lives in a dedicated `parse_args()` function that returns a plain `Args` struct.
- `main()` does **only** two things: call `parse_args()` and dispatch to `run_<program_name>()`.
- No top-level logic outside functions; nothing else in `main()`.

---

## Project Layout

- Each project uses a flat layout: `include/{package_name}/` for public headers, `src/{package_name}/` for implementation, `tests/` for tests, `tests/assets/` for fixtures.
- External dependencies are pulled to `submodules/` (or fetched via `FetchContent`/a package manager, per project).
- Build system: **CMake**, with one `CMakeLists.txt` per directory that owns a target.

## Workflow Guardrails

- Don't introduce abstractions for one-time operations; three similar lines beats a premature helper.
- **No backward compatibility.** When improving an API or convention, update all call sites rather than adding shims or deprecated aliases.

---

## Important Instructions
- **ALWAYS** initialize every variable at declaration; no default-then-assign.
- **ALWAYS** use `const`/`constexpr` for values that do not change after initialization.
- **ALWAYS** pass non-trivial types by `const&` (read-only) or by value with `std::move` (transfer of ownership); avoid passing large objects by value without moving.
- **ALWAYS** use `enum class` (scoped enums), never unscoped `enum`.
- **ALWAYS** use brace initialization (`Type value{...}`) over parentheses where it does not change meaning.
