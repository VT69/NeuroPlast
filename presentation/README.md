# Final-review presentation

- `NeuroPlast_final_review.pptx`: 16:9 deck, 18 main slides plus 4 backup slides (full results table, the
  prospectively specified contrasts, protocol constants, references). Every slide has speaker notes. Slide 10 is the
  labelled post-freeze follow-up (isolation width); slides 16-17 are the proposed next phase (plans, not results);
  slide 18 groups the conclusion into survived / conditional / did not survive / unresolved.
- `NeuroPlast_final_review.pdf`: the same deck converted with LibreOffice.
- `build_deck.py`: regenerates the deck (`python presentation/build_deck.py`, needs `python-pptx`). Every number comes
  from the paper's numbers pipeline (including the post-freeze follow-up in `explore/`), so the deck, the paper and the dashboard always agree. The figures are the
  paper's (`paper/figures/*.png`, 300 dpi), the chart on slide 2 is a native PowerPoint chart, and the colours are the
  dashboard's.
- `assets/`: dashboard screenshots used on the demo slide.

To fill in on slide 1: roll number, guide, department, institution and date. Fonts are Cambria (titles) and Calibri
(body), which ship with Office.
