# Coding Conventions

## Core Principle

> Code is written once but read many times. Always prefer **readability over compactness**.

---

## Python: Formatting

- **PEP8 compliant**, line length **160**, indentation **4 spaces**.
- **Strings**: single quotes by default.
- **No trailing whitespace**; do not indent empty lines.
- **Imports** at the top of the file. Use `from __future__ import annotations` when strict typing is needed.
- Functions in scripts and classes should always be semantically sorted with heading comment of style:
    ```python
    #---------------------------------------------------------------------
    # <semantic section heading>
    #---------------------------------------------------------------------
    ```
- **Always** adhere to the docstring style in @.claude/rules/docstring-style.md.
- In long code passages use comments of style to guide the user's eyes:
    ```python
    #--- <short passage description> ---
    ```
- When working with arrays or tensors and the shape changes document it as an inline-comment like:
    ```python
    <code line> # (<dimensions>)
    ```

## Python: Linting & Typing

- **Linter**: **ruff** (enforced in CI). Selects `E4, E7, E9, F`. Target `py312`. Line length 160.
- **Type checker**: **ty** (astral-sh). Type annotations are required on all function definitions.

## Python: Library Preferences
- **Always** use `pathlib.Path` for path variables.
- Whenever possible, use `opencv2` for image-related tasks.
- Whenever possible, use `open3d` for 3D-related tasks. 

---

## Naming Conventions

### General

- `snake_case` — functions, variables, modules
- `PascalCase` — classes
- `SCREAMING_SNAKE_CASE` — constants
- **No single-letter variable names.**
- **Function names start with a verb.**
- Iteration variables are `ii`, `jj`, `kk` for nested loops

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
- Private helpers: `__{verb}_{part}`

### Array shapes

- `shape` = `(height, width)` — numpy convention
- `size`  = `(width, height)` — Pillow convention

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
- Everything else, is a `transform` / `tf`

---

## Runnable Scripts

Every runnable Python script **must** follow this exact structure:

```python
def <script_name>(...) -> None:
    ...

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ...
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    <script_name>(**vars(args))
```

Rules:

- The main function is named after the script (the file's stem).
- Argument parsing lives in a dedicated `parse_args()` function that returns `argparse.Namespace`.
- The `if __name__ == '__main__':` block does **only** two things: call `parse_args()` and dispatch to the main function via `**vars(args)`.
- Argparse argument names must therefore match the main function's parameter names exactly.
- No top-level logic outside functions; nothing else under the `__main__` guard.

---

## Project Layout

- Each project uses a flat layout: `{package_name}/` for source, `{package_name}/tests/` for tests, `{package_name}/tests/assets/` for fixtures.
- External dependencies are pulled to `submodules/`
- Build system: hatchling (most projects) or setuptools.


## Workflow Guardrails

- Don't introduce abstractions for one-time operations; three similar lines beats a premature helper.
- **No backward compatibility.** When improving an API or convention, update all call sites rather than adding shims or deprecated aliases.

---

## Important Instructions
- **ALWAYS** use f-strings for string interpolation
- **ALWAYS** use type hints and annotations
- **ALWAYS** use `dataclasses` for structured data, not plain dicts
