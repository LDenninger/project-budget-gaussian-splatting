# README Conventions

Standing guidance for authoring a project `README`. Follow it whenever you create or substantially
edit a repository's README.

## Core Principles

- The README is the front door: it must let a reader install and use the project without reading
  the source.
- The file is named `README.md`. Translations use BCP 47 language tags (`README.de.md`).
- The top of the page shows what the project is before any heading: title, badges, teaser and a
  one-line description.
- No table of contents. GitHub renders an outline from the headings, and the emoji headings make
  each section findable while scrolling.
- No broken links. Code examples follow the project's linting standards.

## Content Discipline

The README describes the project as it is: what it does, how to install it, how to use it. Every
sentence either states a fact about the project or tells the reader what to type.

- Cut every sentence that justifies a design choice, compares rejected alternatives or explains how
  a feature is implemented. Rationale and architecture go to `docs/`, linked with one line from the
  README when a reader needs them.
- Cut meta comments: sentences about the README itself ("This section describes"), `Note:` and
  `TODO` markers, and any reference to how or by whom the text was written.
- Cut history: "previously", "now supports", "we changed", "new in". Changes go to the release notes
  or a `CHANGELOG.md`.
- Cut hedging and asides: "should", "hopefully", "for now", "in most cases", parenthetical remarks.
- Keep a caveat only when it changes what the reader types or installs, and state it in one
  sentence.
- Keep each prose paragraph to at most 4 sentences. A longer explanation belongs in `docs/`.

```text
Good  Omit `[viewer]` in environments that only run training.
Bad   We decided to make the viewer optional because its web server pulls in several dependencies
      that most training clusters don't need, which keeps the install lean.
```

## Section Order

Sections appear in this order. **Required** sections must be present, optional ones only when they
add value. Every `##` section carries the emoji listed beside it.

| # | section | emoji | status |
|---|---|---|---|
| 1 | Title | none | required |
| 2 | Badges | none | optional |
| 3 | Teaser | none | optional |
| 4 | Short description | none | required |
| 5 | Long description | none | optional |
| 6 | Features | ✨ | optional |
| 7 | Security | 🔒 | optional |
| 8 | Background | 📖 | optional |
| 9 | Install | 📦 | required¹ |
| 10 | Usage | 🚀 | required¹ |
| 11 | Extra sections | see below | optional |
| 12 | API | 🧩 | optional |
| 13 | Maintainers | 👥 | optional |
| 14 | Contributing | 🤝 | required |
| 15 | Citation | 📝 | optional |
| 16 | Credits | 🙏 | optional |
| 17 | License | 📄 | required, always last |

¹ Documentation-only repositories may omit Install and Usage.

Items 1 to 5 carry no heading of their own. Extra sections sit between Usage and API and take an
emoji from this list, or one no other section of the README uses:

| extra section | emoji |
|---|---|
| Repository or output layout | 📁 |
| Results, benchmarks | 📊 |
| Configuration | ⚙️ |
| Examples, gallery | 🖼️ |
| Architecture, design | 🏗️ |
| Roadmap | 🗺️ |

## Emoji Rules

- Write a `##` heading as `## <emoji> <Section name>`, one emoji followed by one space.
- Use the emoji from the tables above for a listed section, never a substitute.
- Leave the `#` title and every `###` subsection without an emoji.
- Keep emoji out of body prose, bullets, tables and code blocks. Headings are the only place they
  appear.

## Section Rules

- **Title:** a level-1 heading matching the installable name. A logo may replace the text, as a
  `<picture>` with a dark and a light source and the project name as `alt`:

  ```html
  <h1>
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
      <img src="assets/logo-light.svg" alt="my-project" height="60">
    </picture>
  </h1>
  ```

- **Badges:** one shields.io or CI badge per line, each linking to its source. Show build status,
  release, supported Python versions and license, in that order.
- **Teaser:** one image, GIF or WebP at `width="100%"`, stored under `assets/`. The `alt` text
  describes what the image shows in a full sentence.
- **Short description:** a single factual line under 120 characters, matching the GitHub and
  package-manager descriptions.
- **Long description:** one paragraph saying how the project is used and what it produces.
- **✨ Features:** bullets of the form `- **Label:** one or two sentences.`, one capability each.
- **📖 Background:** at most 3 sentences naming the prior work the project builds on, with links. No
  argument for the approach.
- **📦 Install:** a copy-pasteable block that works on a clean machine, release install first, then
  the development install. A `### Dependencies` subsection lists prerequisites and what each
  optional extra adds.
- **🚀 Usage:** the most common case first, as a runnable example. A CLI or viewer gets its own
  `###` subsection.
- **🧩 API:** a `| Name | Purpose |` table of the exported names, followed by one sentence on how
  the subpackages divide the implementation.
- **🤝 Contributing:** where questions and bug reports go, whether pull requests are welcome, and
  what CI runs. A `### Testing` subsection holds the test commands. Link a `CONTRIBUTING.md` when
  one exists.
- **📝 Citation:** a `bibtex` block for a repository that accompanies a paper.
- **📄 License:** the license name linked to `LICENSE`, its SPDX identifier and the copyright owner,
  e.g. ``[MIT](LICENSE) (`MIT`) © 2026 Jane Doe``.

## Describing Folder Structure

When documenting the repository or an output layout, use a `tree`-style block with aligned inline
comments:

```
.
├── build                   # Compiled files (alternatively `dist`)
├── docs                    # Documentation files (alternatively `doc`)
├── src                     # Source files (alternatively `lib` or `app`)
├── test                    # Automated tests (alternatively `spec` or `tests`)
├── tools                   # Tools and utilities
├── LICENSE
└── README.md
```

- Use box-drawing characters (`├──`, `└──`, `│`). The last entry in each level uses `└──`.
- Align the `#` comments into a single column.
- Comment only the entries that need explanation, and list bare files (`LICENSE`, `README.md`)
  without a comment.

## Quick Checklist Before Saving a README

- [ ] No table of contents, and every `##` heading carries its emoji from the tables above
- [ ] No sentence justifies a design choice or explains an implementation
- [ ] No meta comment, `Note:`, `TODO` or change history
- [ ] No prose paragraph runs past 4 sentences
- [ ] License is the last section
