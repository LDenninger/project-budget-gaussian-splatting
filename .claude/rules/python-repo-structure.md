# Python repository structure

Python repositories should follow this structure:

```text
.
├── <package name>              # Package source code
│   ├── data                    # data processing, loaders, and transformers
│   ├── diagnostics             # diagnostic tools and utilities
│   ├── modules                 # Holds the main modules implementing logic
│   ├── evaluation              # fixed evaluation scripts and protocols
│   ├── utils                   # global utility functions and helpers for the package
│   ├── __init__.py             # package initialization file
│   └── 00_<script name>.py     # runnable script with order prefix for execution order
├── LICENSE                     # License file for the project (MIT by default)
└── README.md                   # Project README file

```

## Important Dos
- The `README.md` should be written according to the conventions in `.claude/rules/readme-conventions.md`.