# Capstone report (DSN4091 Capstone Project Phase-I)

`NeuroPlast_Capstone_Report.docx` and `.pdf` are the Phase-I report in the university's format. They are generated;
edit `content.py` (text) or `build_report.py` (layout and front-matter details), not the `.docx`.

    python report/make_figures.py      # dashboard screenshots, architecture diagrams, GIF frames -> report/figures/
    python report/build_report.py      # the .docx, the .pdf and NUMBERS_CHECK.md

What the build needs:
- the university's sample report at `docs/report_template/DSN4091_sample.doc`. It belongs to another project and names
  its authors in its metadata, so it is not committed (see `.gitignore`); put your copy there.
- LibreOffice Writer (`soffice`), to convert the template and to render the PDF used for page numbers;
- Times New Roman installed for LibreOffice (a substitute font changes the pagination);
- `python-docx`, Playwright with Chromium (screenshots only), and the paper's numbers pipeline
  (`paper/compute_numbers.py`).

How it works: the template is converted to `.docx` and switched from US Letter to A4 (same margins; the footer frame
and cover logo are moved to match). Its cover, bonafide certificate and acknowledgement are edited in place, and
everything after them is replaced by our chapters, written with the template's heading and body styles. The build
renders the document twice: the first PDF gives the page of every heading, figure and table, and the second build
writes those pages into the Table of Contents, List of Figures and List of Tables and checks that they no longer
change. Page numbers are PAGEREF fields, so Word refreshes them on opening; the three front-matter rows of the Table of
Contents are plain text, because LibreOffice prints PAGEREF in Arabic numerals even on roman-numbered pages.

`NUMBERS_CHECK.md` lists every result number in the report with its macro in `paper/numbers.tex`, its value and its
source file, plus the non-result numbers (hardware, software versions, demo accuracy matrices) and where they come from.
