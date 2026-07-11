# ROBERT manuscript

This directory contains the working MNRAS manuscript for ROBERT. Scientific
claims should describe released or reproducible behaviour in the repository;
planned work belongs in explicitly labelled roadmap text.

Overleaf project: <https://www.overleaf.com/project/6a52bcdc49adad6c1f9f2b86>

## Build

The manuscript uses the official MNRAS class and BibTeX style. Overleaf's
official MNRAS template already contains `mnras.cls` and `mnras.bst`. For a
local build, place those two files in this directory (they are intentionally
not vendored here) and run:

```bash
latexmk -pdf main.tex
```

## GitHub and Overleaf

1. Push this repository (including `paper/`) to GitHub.
2. In Overleaf, create a project by importing the GitHub repository, or create
   the official MNRAS template and replace its manuscript sources with this
   directory.
3. Set `paper/main.tex` as the Overleaf main document if the whole repository
   is imported.
4. Use Overleaf's GitHub synchronisation controls to pull from or push to the
   linked branch. Synchronisation is manual, not continuous; agree which side
   is authoritative before each editing session.

Do not commit generated files (`*.aux`, `*.bbl`, `*.blg`, `*.fls`, `*.fdb_latexmk`,
`*.log`, `*.out`, or PDFs). Keep figures in `paper/figures/` and regenerate them
from version-controlled scripts wherever possible.

## Drafting conventions

- `\roberttodo{...}` marks unresolved scientific or editorial work.
- Numbers in the benchmark section must be traceable to a script and a
  versioned machine-readable output before submission.
- The WASP-80b Results section is a prospective analysis plan until the full
  retrieval and robustness suite has run.
- Replace author, affiliation, software-version, repository DOI, and data
  availability placeholders before circulating the manuscript externally.
