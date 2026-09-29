---
name: commit
description: Use when writing a git commit message in this repository, or when splitting a set of changes into commits.
---

# Conventional Commits

When creating any git commit in this repository this skill applies automatically following the commit conventions in `./.github/commit_convention.md`

## Format

Single-line, when a GitHub issue applies:

    <type>: (#<issue_number>) <issue_name> - <description>.

Single-line, when no GitHub issue applies:

    <type>: <description>.

Multiline, when the change spans multiple concerns:

    <type>: (#<issue_number>) <issue_name>:
    - <description_line_1>.
    - <description_line_2>.

## Rules

- `<type>` is exactly one of: `New feature`, `Fix issue`, `Other`.
- Single-line messages end with `.`; the first line of a multiline message ends with `:`.
- Every bullet starts with `- ` and ends with `.`.
- Use present tense, imperative mood ("add feature", not "added feature").
- Keep commits atomic — one logical concern per commit. Split unrelated changes into
  separate commits.
- NEVER add trailers such as "Generated with Claude Code", author info, or co-author lines.

## References

- Full template with examples: `.gitmessage`
- Commit-splitting guidance and workflow integration: `.github/commit_convention.md`