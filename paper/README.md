# Paper (TMLR format)

Two PDFs are built from the one source, `main.tex`:

- `main.pdf`: preprint with the author's name and the repository URL (for the final review).
- `main_anonymous.pdf`: TMLR double-blind submission format ("Anonymous authors", no name, no repository URL).

To rebuild everything from the frozen results:

```bash
python paper/compute_numbers.py          # every number -> paper/numbers.tex (LaTeX macros) + paper/NUMBERS.md (sources)
python paper/figures/make_figures.py     # figures -> paper/figures/*.pdf (vector) + .png previews
cd paper
latexmk -pdf main.tex                    # preprint -> main.pdf
latexmk -pdf -jobname=main_anonymous \
  -pdflatex='pdflatex %O "\def\anonymousbuild{}\input{%S}"' main.tex   # anonymous -> main_anonymous.pdf
```

Needs TeX Live (latex-recommended, latex-extra, fonts-recommended, science), lmodern and cm-super.
The figures use the CMU Serif fonts (Debian/Ubuntu package `fonts-cmu`), which carry a proper Unicode map, so
minus signs (U+2212) and λ extract correctly from the embedded TrueType fonts.

- `main.tex` contains no hand-typed result numbers: each one is a macro from `numbers.tex`, and `NUMBERS.md` lists
  every macro with its value, meaning and source file in `runs/` or `results/`.
- The anonymous build is selected by defining `\anonymousbuild`: it loads `tmlr` without `[preprint]` and replaces the
  repository URL with "an anonymised repository provided as supplementary material".
- Figures are drawn at their printed width (at most 6.5 in, the TMLR text width) in Computer Modern (CMU Serif,
  embedded as TrueType), with the same method colours as the dashboard (`demo/index.html`). Sleep is blue, replay orange, isolation green, EWC yellow,
  naive pink and replay-matched sleep violet; non-method series use grey and violet.
- `refs.bib`: entries checked against the author's local copies of the cited PDFs (not distributed with the repository)
  are marked `[local PDF]`. The others are
  standard references from well-known metadata, marked `TODO verify` in a comment above the entry.
- Visible `TODO`s in the PDF: author affiliation and email (preprint only), the AI-assistance disclosure, and the
  venue of Yu et al. (Self-Consolidation).
- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`: official TMLR style files (github.com/JmlrOrg/tmlr-style-file, Apache 2.0,
  see `TMLR_STYLE_LICENSE`).
