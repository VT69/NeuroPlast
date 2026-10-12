# Capstone report (DSN4091)

`NeuroPlast_Capstone_Report.docx` and `.pdf` are the capstone report in the university's format. They are generated
by `build_report.py` with the content in `content.py`; edit those two files, not the `.docx`.

    python report/build_report.py

What the build needs:
- the university template at `docs/report_template/DSN4091_sample.doc`. It is a sample report for another project,
  so it is not committed (see `.gitignore`); put the copy you were given there.
- LibreOffice Writer (`soffice`), used to convert the template and to render the PDF for page numbers;
- Times New Roman installed for LibreOffice (otherwise a metric-compatible substitute changes the pagination);
- `python-docx`, plus the paper's numbers pipeline dependencies (`paper/compute_numbers.py`).

How it works: the template is converted to `.docx`, its cover, bonafide certificate and acknowledgement are edited in
place, and everything after them is replaced by our chapters, written with the template's styles. The build renders
the document twice: the first PDF gives the page of every heading, figure and table, and the second build writes those
page numbers into the Table of Contents, List of Figures and List of Tables and checks that they no longer change.
The page numbers are PAGEREF fields, so Word refreshes them when the file is opened (the front-matter rows of the
Table of Contents are plain text, because LibreOffice prints PAGEREF in Arabic numerals even for roman pages).

Checks written by the build:
- `NUMBERS_CHECK.md`: every result number in the report, with its macro in `paper/numbers.tex`, its value and its
  source file; plus the non-result numbers (hardware, software versions, protocol settings) and where they come from.
- `REFERENCES_TO_VERIFY.md`: the cited entries still marked `TODO verify` in `paper/refs.bib`, with their number in
  the report.

`figures/` holds the dashboard screenshots, architecture crops and GIF frame grids used by the report (taken from
`demo/index.html` in the light theme and from `demo/assets/`).
