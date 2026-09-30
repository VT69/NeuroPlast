# Final-review presentation

- `NeuroPlast_final_review.pptx`: 16:9 deck, 15 main slides plus 4 backup slides (full results table, the
  pre-registered contrasts, protocol constants, references). Every slide has speaker notes.
- `NeuroPlast_final_review.pdf`: the same deck converted with LibreOffice.
- `build_deck.py`: regenerates the deck (`python presentation/build_deck.py`, needs `python-pptx`). Every number comes
  from the paper's numbers pipeline, so the deck, the paper and the dashboard always agree. The figures are the
  paper's (`paper/figures/*.png`, 300 dpi), the chart on slide 2 is a native PowerPoint chart, and the colours are the
  dashboard's.
- `assets/`: dashboard screenshots used on the demo slide.

To fill in on slide 1: roll number, guide, department, institution and date. Fonts are Cambria (titles) and Calibri
(body), which ship with Office.
