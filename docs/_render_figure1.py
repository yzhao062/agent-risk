"""Render the README hero image docs/figure1.png from the paper's Figure 1.

Usage: python docs/_render_figure1.py   (needs PyMuPDF: pip install pymupdf)
"""
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
page = pymupdf.open(ROOT / "experiments" / "figures" / "fig1_overview.pdf")[0]
page.get_pixmap(dpi=220, alpha=False).save(ROOT / "docs" / "figure1.png")
