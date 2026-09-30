# Paper draft (TMLR format)

`main.pdf` is the built draft. To rebuild everything from the frozen results:

```bash
python paper/compute_numbers.py          # every number -> paper/numbers.tex (LaTeX macros) + paper/NUMBERS.md (sources)
python paper/figures/make_figures.py     # figures -> paper/figures/*.pdf (+ .png previews)
cd paper && latexmk -pdf main.tex        # needs TeX Live (latex-extra, fonts-recommended, science) + lmodern
```

- `main.tex` contains no hand-typed result numbers: each one is a macro from `numbers.tex`; `NUMBERS.md` lists
  every macro with its value, meaning and source file (runs/ or results/).
- `refs.bib`: entries checked against the PDFs in `relevant_literature/` are marked `[local PDF]`; the others are
  standard references cited from memory and marked `TODO verify` in a comment above the entry.
- Visible `TODO`s in the PDF: author affiliation/email, repository URL, the AI assistance disclosure, and the
  venue of Yu et al. (Self-Consolidation).
- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`: official TMLR style files (github.com/JmlrOrg/tmlr-style-file, Apache 2.0,
  see `TMLR_STYLE_LICENSE`). `main.tex` uses the `[preprint]` option; switch to `\usepackage{tmlr}` for an anonymous
  submission.
