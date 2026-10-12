"""Build the DSN4091 Capstone Project Phase-I report from the university template.

    python report/make_figures.py      # dashboard screenshots, diagrams, GIF frames -> report/figures/
    python report/build_report.py      # -> report/NeuroPlast_Capstone_Report.docx and .pdf, report/NUMBERS_CHECK.md

The template is docs/report_template/DSN4091_sample.doc, a sample report for another project; it is not committed (see
.gitignore). It is converted to .docx with LibreOffice at build time, so the cover page, university logo, margins,
styles and footer come from it; the page size is changed from US Letter to A4. The cover, bonafide certificate and
acknowledgement are edited in place; everything after them is replaced by our content (report/content.py), written
with the template's styles.

Every result number comes from the paper's numbers pipeline (paper/compute_numbers.py, via demo/build_dashboard.py);
report/NUMBERS_CHECK.md lists each number used with its macro, value and source file. The Table of Contents, List of
Figures and List of Tables use PAGEREF fields to bookmarks; their cached page numbers are taken from the rendered PDF,
and a second build checks that they no longer change.
"""
from __future__ import annotations

import copy
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, os.path.join(ROOT, "report"))
import build_dashboard as bd  # noqa: E402  (chdirs to ROOT; the paper's numbers, formatted like the dashboard)
import compute_numbers as cn  # noqa: E402  (already imported by build_dashboard; gives each number's source)

from docx import Document  # noqa: E402
from docx.oxml import parse_xml  # noqa: E402
from docx.oxml.ns import nsdecls, qn  # noqa: E402
from docx.shared import Emu, Inches, Pt  # noqa: E402

TEMPLATE = "docs/report_template/DSN4091_sample.doc"
OUT_DOCX = "report/NeuroPlast_Capstone_Report.docx"
OUT_PDF = "report/NeuroPlast_Capstone_Report.pdf"

# ---------------------------------------------------------------- front-matter details (as given by the team)
TITLE = "NeuroPlast - A Self-Consolidating RL Agent"
TEAM = [("Vaibhav Tiwari", "23BAI10108"), ("Kavish Bishnoi", "23BAI10789"), ("Kushal Upadhyay", "23BAI10259"),
        ("Yash Yadav", "23BAI10484"), ("Adamya Upadhyay", "23BAI11221"), ("Siddharth Ubbott", "23BAI10997")]
CHAIR = "Dr. Pradeep Kumar Mishra"                       # no designation, as instructed
GUIDE = "Dr. Trapti Sharma"
GUIDE_DESIGNATION = "[TBD: guide designation]"
SCHOOL = "School of Computing Science Engineering and Artificial Intelligence"
MONTH = "October 2026"
VIVA_DATE = "October 12, 2026"

# ---------------------------------------------------------------- numbers (logged for NUMBERS_CHECK.md)
N = bd.numbers()
USED: dict[str, str] = {}


def n(key):
    USED[key] = N[key]
    return N[key]


# ---------------------------------------------------------------- page geometry: A4 with the template's margins
LETTER_W, LETTER_H = 12240, 15840
PAGE_W, PAGE_H = 11906, 16838
MARGIN = dict(right="540", bottom="960", left="880", header="0", footer="764")
TW = 835, 873            # body text indents (start, end) in twips, as the template's Body Text paragraphs
TEXT_W = PAGE_W - 880 - 540 - TW[0] - TW[1]   # usable width inside the indents, twips (= 8778, about 6.1 in)
MONO = "Courier New"
EMU_PER_TWIP = 635


def esc(t):
    return html.escape(t, quote=False)


def rpr(b=False, i=False, sz=None, font=None, color=None):
    x = ""
    if font:
        x += f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}" w:eastAsia="{font}"/>'
    if b:
        x += "<w:b/><w:bCs/>"
    if i:
        x += "<w:i/><w:iCs/>"
    if color:
        x += f'<w:color w:val="{color}"/>'
    if sz:
        x += f'<w:sz w:val="{int(sz * 2)}"/><w:szCs w:val="{int(sz * 2)}"/>'
    return f"<w:rPr>{x}</w:rPr>" if x else ""


def run(t, **k):
    return f'<w:r>{rpr(**k)}<w:t xml:space="preserve">{esc(t)}</w:t></w:r>'


def ppr(style=None, keep_next=False, page_break=False, numpr=None, tabs=None, before=0, after=120, line=360,
        ind=None, hanging=None, jc=None, sect=None):
    x = f'<w:pStyle w:val="{style}"/>' if style else ""
    if keep_next:
        x += "<w:keepNext/>"
    if page_break:
        x += "<w:pageBreakBefore/>"
    if numpr is not None:
        x += f'<w:numPr><w:ilvl w:val="{numpr[1]}"/><w:numId w:val="{numpr[0]}"/></w:numPr>'
    if tabs:
        x += "<w:tabs>" + "".join(f'<w:tab w:val="{v}" w:pos="{p}"/>' for v, p in tabs) + "</w:tabs>"
    x += f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
    if ind is not None:
        x += f'<w:ind w:left="{ind[0]}" w:right="{ind[1]}"' + (f' w:hanging="{hanging}"' if hanging else "") + "/>"
    if jc:
        x += f'<w:jc w:val="{jc}"/>'
    if sect is not None:
        x += sect
    return f"<w:pPr>{x}</w:pPr>"


def P(inner, pp):
    return parse_xml(f"<w:p {nsdecls('w', 'r')}>{pp}{inner}</w:p>")


def scale(widths, total=None):
    """Column widths in inches -> twips, scaled to fill the text width."""
    total = total or TEXT_W
    s = total / sum(widths)
    tw = [int(w * s) for w in widths]
    tw[-1] += total - sum(tw)
    return tw


# ---------------------------------------------------------------- inline markup: **bold**, *italic*, `code`, [[cite]], <<fig:x>>
class Refs:
    def __init__(self):
        self.order: list[str] = []

    def num(self, key):
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1


REFS = Refs()
LABELS: dict[str, str] = {}     # fig/tab label -> "Fig 4.1" / "Table 3.2"
TOKEN = re.compile(r"(\*\*.+?\*\*|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*])|`[^`]+`|\[\[[^\]]+\]\]|<<(?:fig|tab):[\w-]+>>)")


def cite(keys):
    nums = sorted({REFS.num(k.strip()) for k in keys.split(",")})
    return "[" + "], [".join(str(x) for x in nums) + "]"


def inline(text, sz=None, base_b=False, base_i=False):
    out = ""
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out += run(part[2:-2], b=True, i=base_i, sz=sz)
        elif part.startswith("`"):
            out += run(part[1:-1], font=MONO, sz=(sz or 12) - 1.5, b=base_b)
        elif part.startswith("[["):
            out += run(cite(part[2:-2]), sz=sz, b=base_b, i=base_i)
        elif part.startswith("<<"):
            out += run(LABELS[part[2:-2]], sz=sz, b=base_b, i=base_i)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out += run(part[1:-1], i=True, b=base_b, sz=sz)
        else:
            out += run(part, sz=sz, b=base_b, i=base_i)
    return out


# ---------------------------------------------------------------- the builder
class Builder:
    def __init__(self, doc):
        self.doc = doc
        self.anchor = doc.element.body[-1]   # the body-level sectPr; new content goes before it
        self.bm_id = 1000
        self.toc: list[tuple] = []           # (level, label, title, bookmark)
        self.lof: list[tuple] = []           # (label, caption, bookmark)
        self.lot: list[tuple] = []
        self.chap = ""
        self.numbering_setup()

    def add(self, el):
        self.anchor.addprevious(el)
        return el

    def bookmark(self, name, inner):
        self.bm_id += 1
        return (f'<w:bookmarkStart w:id="{self.bm_id}" w:name="{name}"/>{inner}'
                f'<w:bookmarkEnd w:id="{self.bm_id}"/>')

    def numbering_setup(self):
        numbering = self.doc.part.numbering_part.element
        lvl = ('<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/><w:lvlText w:val="{txt}"/>'
               '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="1195" w:hanging="360"/></w:pPr>{r}</w:lvl>')
        tnr = '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr>'
        first_num = numbering.find(qn("w:num"))
        for aid, fmt, txt, r in ((901, "bullet", "•", tnr), (902, "decimal", "%1.", "")):
            el = parse_xml(f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{aid}"><w:multiLevelType '
                           f'w:val="singleLevel"/>{lvl.format(fmt=fmt, txt=txt, r=r)}</w:abstractNum>')
            first_num.addprevious(el) if first_num is not None else numbering.append(el)
        self.numbering = numbering
        self.next_num = 900
        self.bullet_num = self.new_num(901)

    def new_num(self, abstract):
        self.next_num += 1
        self.numbering.append(parse_xml(
            f'<w:num {nsdecls("w")} w:numId="{self.next_num}"><w:abstractNumId w:val="{abstract}"/>'
            '<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>'))
        return self.next_num

    # -- blocks
    def para(self, text, jc="both", after=120, keep_next=False, line=360):
        return self.add(P(inline(text), ppr("BodyText", keep_next=keep_next, after=after, line=line, ind=TW, jc=jc)))

    def list(self, items, ordered=False):
        num = self.new_num(902) if ordered else self.bullet_num
        for k, t in enumerate(items):
            self.add(P(inline(t), ppr("BodyText", numpr=(num, 0), after=60 if k < len(items) - 1 else 140,
                                      ind=(1195, TW[1]), hanging=360, jc="both")))

    def front_heading(self, text, page_break=True, bm=None):
        inner = run(text, b=True, sz=16)
        return self.add(P(self.bookmark(bm, inner) if bm else inner,
                          ppr("Normal", page_break=page_break, before=60, after=360, line=240, jc="center")))

    def heading1(self, text, bm, page_break=True):
        """Unnumbered Heading 1 on a new page (appendices, contributions, references)."""
        self.add(P(self.bookmark(bm, run(text, b=True, sz=16)),
                   ppr("Heading1", page_break=page_break, keep_next=True, numpr=(0, 0), before=60, after=360,
                       ind=(0, 0), jc="center")))

    def chapter(self, num, title, first=False):
        bm = f"_TocCh{num}"
        # Chapter 1 follows the front matter's section break (a new page already), so it needs no page break
        self.add(P(run(f"CHAPTER {num}", b=True, sz=16), ppr("Normal", page_break=not first, keep_next=True,
                                                              before=60, after=0, jc="center")))
        self.add(P(self.bookmark(bm, run(title.upper(), b=True, sz=16)),
                   ppr("Heading1", keep_next=True, numpr=(0, 0), before=0, after=360, ind=(0, 0), jc="center")))
        self.toc.append((0, f"CHAPTER {num}", title.upper(), bm))
        self.chap = str(num)

    def appendix(self, letter, title):
        bm = f"_TocApp{letter}"
        self.heading1(f"APPENDIX {letter}: {title.upper()}", bm)
        self.toc.append((0, f"APPENDIX {letter}", title.upper(), bm))
        self.chap = letter

    def unnumbered(self, title, bm):
        self.heading1(title.upper(), bm)
        self.toc.append((0, "", title.upper(), bm))

    def h2(self, num, title):
        bm = "_TocS" + num.replace(".", "_")
        self.add(P(self.bookmark(bm, run(num, b=True, sz=14) + "<w:r><w:tab/></w:r>" + run(title.upper(), b=True, sz=14)),
                   ppr("Heading2", keep_next=True, numpr=(0, 0), tabs=[("left", 1560)], before=240, after=120,
                       ind=(775, 0), jc="left")))
        self.toc.append((1, num, title.upper(), bm))

    def h3(self, num, title):
        bm = "_TocS" + num.replace(".", "_")
        self.add(P(self.bookmark(bm, run(num, b=True, sz=12) + "<w:r><w:tab/></w:r>" + run(title.upper(), b=True, sz=12)),
                   ppr("Heading3", keep_next=True, numpr=(0, 0), tabs=[("left", 1700)], before=180, after=100,
                       ind=(835, 0), jc="left")))
        self.toc.append((2, num, title.upper(), bm))

    def caption(self, kind, lab, text):
        name = LABELS[f"{kind}:{lab}"]
        bm = "_Cap" + re.sub(r"\W", "_", name)
        inner = self.bookmark(bm, run(name + ": ", b=True, sz=11) + inline(text, sz=11))
        (self.lof if kind == "fig" else self.lot).append((name, text, bm))
        return P(inner, ppr("Normal", keep_next=(kind == "tab"), before=60 if kind == "tab" else 80,
                            after=200 if kind == "fig" else 80, line=276, ind=TW, jc="center"))

    def figure(self, lab, path, caption, width_in):
        from PIL import Image
        w, h = Image.open(path).size
        width = Inches(min(width_in, TEXT_W / 1440))
        inline_el = self.doc.part.new_pic_inline(path, width, Emu(int(width * h / w)))
        p = P("", ppr("Normal", keep_next=True, before=120, after=0, line=240, ind=TW, jc="center"))
        r = parse_xml(f"<w:r {nsdecls('w')}><w:drawing/></w:r>")
        r[0].append(inline_el)
        p.append(r)
        self.add(p)
        self.add(self.caption("fig", lab, caption))

    def table(self, lab, caption, header, rows, widths, sz=10, align=None):
        if lab:
            self.add(self.caption("tab", lab, caption))
        self.add(self.tbl(header, rows, widths, sz, align))
        self.add(P("", ppr("Normal", after=120, line=240)))

    def tbl(self, header, rows, widths, sz=10, align=None):
        tw = scale(widths)
        b = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        bdr = "".join(b.format(s) for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        xml = (f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{sum(tw)}" w:type="dxa"/>'
               f'<w:jc w:val="center"/><w:tblBorders>{bdr}</w:tblBorders><w:tblLayout w:type="fixed"/>'
               '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
               '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
               + "".join(f'<w:gridCol w:w="{w}"/>' for w in tw) + "</w:tblGrid>")

        together = len(rows) <= 15   # short tables stay on one page (keep-with-next on every row but the last)

        def cell(text, w, head=False, jc="left", keep=False):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E0"/>' if head else ""
            ps = "".join(f'<w:p>{ppr("Normal", keep_next=keep, after=0, line=240, jc=jc)}'
                         f'{inline(t, sz=sz, base_b=head)}</w:p>' for t in (text if isinstance(text, list) else [text]))
            return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}</w:tcPr>{ps}</w:tc>'
        if header:
            xml += '<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + "".join(
                cell(h, w, head=True, jc="center", keep=together) for h, w in zip(header, tw)) + "</w:tr>"
        for i, r in enumerate(rows):
            keep = together and i < len(rows) - 1
            xml += "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(
                cell(c, w, jc=(align[j] if align else "left"), keep=keep) for j, (c, w) in enumerate(zip(r, tw))) + "</w:tr>"
        return parse_xml(xml + "</w:tbl>")

    def code(self, caption, text, sz=8.5):
        self.add(P(inline(caption, sz=10.5, base_i=True),
                   ppr("Normal", keep_next=True, before=120, after=60, line=240, ind=TW, jc="left")))
        cw = round(sz * 12)  # monospace advance (0.6 em) in twips
        ps = ""
        for ln in text.rstrip("\n").split("\n"):  # leading spaces become an indent; wrapped lines go 4 columns further
            k = len(ln) - len(ln.lstrip(" "))
            ps += (f'<w:p>{ppr("Normal", after=0, line=240, ind=((k + 4) * cw, 0), hanging=4 * cw, jc="left")}'
                   f'<w:r>{rpr(font=MONO, sz=sz)}<w:t xml:space="preserve">{esc(ln.lstrip(" ")) or " "}</w:t></w:r></w:p>')
        w = TEXT_W
        self.add(parse_xml(
            f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{w}" w:type="dxa"/><w:jc w:val="center"/>'
            '<w:tblBorders><w:left w:val="single" w:sz="18" w:space="0" w:color="9CA3AF"/></w:tblBorders>'
            '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="140" w:type="dxa"/>'
            '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{w}"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>'
            f'<w:shd w:val="clear" w:color="auto" w:fill="F4F4F1"/></w:tcPr>{ps}</w:tc></w:tr></w:tbl>'))
        self.add(P("", ppr("Normal", after=80, line=240)))

    def refs(self, entries):
        self.unnumbered("References", "_TocRefs")
        for k, txt in enumerate(entries, 1):
            self.add(P(run(f"[{k}]", sz=11) + "<w:r><w:tab/></w:r>" + inline(txt, sz=11),
                       ppr("Normal", tabs=[("left", TW[0] + 560)], after=100, line=276,
                           ind=(TW[0] + 560, TW[1]), hanging=560, jc="both")))


# ---------------------------------------------------------------- page-number fields for the lists
def pageref(bm, cached, sz=11, b=False):
    r = rpr(sz=sz, b=b)
    if bm.startswith("_Front"):
        # LibreOffice prints PAGEREF in Arabic even when the target section is lower-roman (and ignores the
        # \* roman switch), so the front-matter rows carry the roman label read from the rendered PDF.
        return f'<w:r>{r}<w:t>{esc(str(cached))}</w:t></w:r>'
    return (f'<w:r>{r}<w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r>{r}<w:instrText xml:space="preserve"> PAGEREF {bm} \\h </w:instrText></w:r>'
            f'<w:r>{r}<w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r>{r}<w:t>{esc(str(cached))}</w:t></w:r>'
            f'<w:r>{r}<w:fldChar w:fldCharType="end"/></w:r>')


def list_table(header, rows, widths, sz=11):
    """Bordered list table (TOC, LoF, LoT); rows are (col0, title, bookmark, cached page, level, group_start)."""
    tw = scale(widths, TEXT_W + TW[0] + TW[1] - 400)
    b = '<w:{0} w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
    xml = (f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{sum(tw)}" w:type="dxa"/><w:jc w:val="center"/>'
           f'<w:tblBorders>{"".join(b.format(s) for s in ("top", "left", "bottom", "right", "insideV"))}</w:tblBorders>'
           '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="30" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
           '<w:bottom w:w="30" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
           + "".join(f'<w:gridCol w:w="{w}"/>' for w in tw) + "</w:tblGrid>")
    hb = '<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>'
    xml += '<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + "".join(
        f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{hb}</w:tcPr><w:p>{ppr("Normal", after=0, line=240, jc="center")}'
        f'{run(h, b=True, sz=sz)}</w:p></w:tc>' for h, w in zip(header, tw)) + "</w:tr>"
    for c0, title, bm, cached, level, group in rows:
        top = '<w:tcBorders><w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>' if group else ""
        bold, before = level == 0, (100 if group else 20)
        ind = {0: 0, 1: 300, 2: 600}[level]
        cells = [
            f'<w:p>{ppr("Normal", before=before, after=20, line=240, jc="center")}{run(c0, b=bold, sz=sz)}</w:p>',
            f'<w:p>{ppr("Normal", before=before, after=20, line=240, ind=(ind, 0), jc="left")}'
            f'{inline(title, sz=sz, base_b=bold)}</w:p>',
            f'<w:p>{ppr("Normal", before=before, after=20, line=240, jc="center")}{pageref(bm, cached, sz=sz, b=bold)}</w:p>']
        xml += "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(
            f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{top}</w:tcPr>{c}</w:tc>' for c, w in zip(cells, tw)) + "</w:tr>"
    return parse_xml(xml + "</w:tbl>")


# ---------------------------------------------------------------- template front matter
def set_text(p, text, keep=0):
    """Replace a paragraph's text with `text`, keeping the formatting of run `keep`."""
    runs = p.runs
    for r in runs[:keep] + runs[keep + 1:]:
        r._r.getparent().remove(r._r)
    runs[keep].text = text


def is_red(r):
    return r.font.color is not None and r.font.color.type is not None and str(r.font.color.rgb) == "FF0000"


def demote(p, doc):
    """Front-matter paragraphs that use heading styles in the template become Normal, with the heading style's
    formatting copied as direct formatting: the look is unchanged and Word's TOC ignores them."""
    sp = p.style.element.find(qn("w:pPr"))
    sr = p.style.element.find(qn("w:rPr"))
    pPr = p._p.get_or_add_pPr()
    for tag in ("w:ind", "w:jc", "w:spacing"):
        if pPr.find(qn(tag)) is None and sp is not None and sp.find(qn(tag)) is not None:
            pPr.append(copy.deepcopy(sp.find(qn(tag))))
    p.style = doc.styles["Normal"]
    for r in p.runs:
        rp = r._r.get_or_add_rPr()
        for tag in ("w:b", "w:bCs", "w:sz", "w:szCs"):
            if sr is not None and rp.find(qn(tag)) is None and sr.find(qn(tag)) is not None:
                rp.append(copy.deepcopy(sr.find(qn(tag))))


def signature_table(sz):
    """The template aligns the Program Chair and Project Guide blocks with runs of spaces; rebuilt here as a
    borderless two-column table with the same text and order, so both blocks stay aligned."""
    left = [("PROGRAM CHAIR", True), (CHAIR, True), ("", False), (SCHOOL, False), ("VIT BHOPAL UNIVERSITY", False)]
    right = [("PROJECT GUIDE", True), (GUIDE, True), (GUIDE_DESIGNATION, False), (SCHOOL, False),
             ("VIT BHOPAL UNIVERSITY", False)]
    w = (PAGE_W - 880 - 540) // 2
    rows = ""
    for (lt, lb), (rt, rb) in zip(left, right):
        rows += "<w:tr>" + "".join(
            f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr><w:p>{ppr("Normal", after=0, line=240, jc="center")}'
            f'{run(t, b=bb, sz=sz) if t else ""}</w:p></w:tc>' for t, bb in ((lt, lb), (rt, rb))) + "</w:tr>"
    return parse_xml(f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{2 * w}" w:type="dxa"/><w:jc w:val="center"/>'
                     '<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>'
                     f'<w:gridCol w:w="{w}"/><w:gridCol w:w="{w}"/></w:tblGrid>{rows}</w:tbl>')


def edit_front(doc):
    P_ = doc.paragraphs
    # the template paragraphs edited here are where we expect them
    assert P_[0].text.startswith("A PROPOSED DESIGN") and P_[2].text == "DSN4091-CAPSTONE PROJECT PHASE-I"
    assert P_[3].text == "Phase – I Report" and P_[6].text.startswith("STUDENT NAME")
    assert P_[26].text.replace(" ", "") == "BONAFIDECERTIFICATE" and P_[45].text == "ACKNOWLEDGEMENT"
    # cover title; "DSN4091-CAPSTONE PROJECT PHASE-I" and "Phase – I Report" already match (Phase-I)
    set_text(P_[0], TITLE.upper())
    # team: the template's single "STUDENT NAME (21BAI10xxx)" line becomes one line per member, same formatting
    tmpl = P_[6]._p
    prev = tmpl
    for _ in TEAM:
        el = copy.deepcopy(tmpl)
        prev.addnext(el)
        prev = el
    tmpl.getparent().remove(tmpl)
    P_ = doc.paragraphs
    for k, (name, reg) in enumerate(TEAM):
        p = P_[6 + k]
        p.runs[0].text = name + " "
        p.runs[1].text = f"({reg})"
        for r in p.runs[2:]:
            r._r.getparent().remove(r._r)
        pf = p.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(3), 1.0
    P_[6 + len(TEAM) - 1].paragraph_format.space_after = Pt(10)
    # the template narrows these cover lines with start/end indents sized for its own text; on A4 the longer title,
    # names and degree line would wrap, so they are centred across the full width instead
    degree = next(p for p in P_[:30] if "(ARTIFICIAL INTELLIGENCE" in p.text)
    degree.runs[0].text = degree.runs[0].text.lstrip()
    for p in [P_[0], degree, next(p for p in P_[:30] if p.text == "COMPUTER SCIENCE AND ENGINEERING")] + \
            [P_[6 + k] for k in range(len(TEAM))]:
        pPr = p._p.get_or_add_pPr()
        for el in pPr.findall(qn("w:ind")) + pPr.findall(qn("w:jc")):
            pPr.remove(el)
        pPr.append(parse_xml(f'<w:ind {nsdecls("w")} w:left="0" w:right="0" w:firstLine="0"/>'))
        pPr.append(parse_xml(f'<w:jc {nsdecls("w")} w:val="center"/>'))
    month = next(p for p in P_[:35] if p.text.strip() == "September 2026")
    set_text(month, MONTH)
    # bonafide certificate: the template's two paragraphs (words separated by tabs) become one justified paragraph
    cert0 = next(i for i, p in enumerate(P_) if p.text.startswith("Certified"))
    p0, p1 = P_[cert0], P_[cert0 + 1]
    base = copy.deepcopy(p1.runs[16]._r.find(qn("w:rPr")))   # "who carried out the project work ..." (plain)
    assert p1.runs[16].text.startswith("who carried out")

    def r_(text, bold=False):
        rp = copy.deepcopy(base)
        if bold:
            rp.insert(0, parse_xml(f'<w:b {nsdecls("w")}/>'))
        el = parse_xml(f'<w:r {nsdecls("w")}><w:t xml:space="preserve">{esc(text)}</w:t></w:r>')
        el.insert(0, rp)
        return el
    names = ", ".join(f"{a} ({b})" for a, b in TEAM)
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    for text, bold in (("Certified that this project report titled ", False), (f"“{TITLE.upper()}”", True),
                       (" is the bonafide work of ", False), (f"“{names}”", True),
                       (" who carried out the project work (DSN4091- Capstone Project Phase-I) under my supervision. "
                        "Certified further that to the best of my knowledge the work reported at this time does not "
                        "form part of any other project/research work based on which a degree or award was conferred "
                        "on an earlier occasion on this or any other candidate.", False)):
        p0._p.append(r_(text, bold))
    p0.paragraph_format.line_spacing = 1.5
    p0.alignment = 3  # justify
    p1._p.getparent().remove(p1._p)
    P_ = doc.paragraphs
    # Program Chair and Project Guide blocks
    gi = next(i for i, p in enumerate(P_) if "PROGRAM" in p.text and "GUIDE" in p.text)
    sz = P_[gi + 2].runs[1].font.size.pt if P_[gi + 2].runs[1].font.size else 11
    block = [P_[gi + k]._p for k in range(5)]
    assert "VIT BHOPAL UNIVERSITY" in P_[gi + 4].text
    block[0].addprevious(signature_table(sz))
    for el in block:
        el.getparent().remove(el)
    P_ = doc.paragraphs
    viva = next(p for p in P_ if "Viva Voce" in p.text)
    assert viva.runs[-1].text.strip("_ ") == ""
    viva.runs[-1].text = f" {VIVA_DATE}"
    # acknowledgement: only the guide's name (the template's red placeholder) changes
    ack = next(p for p in P_ if "internal guide" in p.text)
    for r in ack.runs:
        if is_red(r):
            r.text = f"our internal guide {GUIDE}"
            r.font.color.rgb = None
    for p in doc.paragraphs[:60]:
        if p.style is not None and p.style.name.startswith("Heading"):
            demote(p, doc)
    leftovers = [p.text for p in doc.paragraphs[:60] if any(is_red(r) and r.text.strip() for r in p.runs)]
    assert not leftovers, leftovers


# ---------------------------------------------------------------- sections, footer, page size
def sect_xml(top, fmt, footer_rid):
    return (f'<w:sectPr {nsdecls("w", "r")}><w:footerReference w:type="default" r:id="{footer_rid}"/>'
            f'<w:type w:val="nextPage"/><w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
            f'<w:pgMar w:top="{top}" w:right="540" w:bottom="960" w:left="880" w:header="0" w:footer="764" w:gutter="0"/>'
            f'<w:pgNumType w:fmt="{fmt}"/><w:docGrid w:type="default" w:linePitch="360" w:charSpace="0"/></w:sectPr>')


def fix_footer(doc, footer_rid):
    """The footer frame ("Page | n") is anchored to the page. Widen it (the template's 0.52 in fits one digit only)
    and move it down by the extra height of A4, so it keeps its distance from the bottom edge."""
    ftr = doc.part.related_parts[footer_rid].element
    for el in ftr.iter():
        if (el.tag.endswith("}extent") or el.tag.endswith("}ext")) and el.get("cx") == "502920":
            el.set("cx", "1280160")
    for pos in ftr.iter(qn("wp:positionV")):
        off = pos.find(qn("wp:posOffset"))
        off.text = str(int(off.text) + (PAGE_H - LETTER_H) * EMU_PER_TWIP)


def fix_logo(doc):
    """The cover logo is positioned from the page's left edge; keep it centred in the narrower A4 text area."""
    for pos in doc.element.body.iter(qn("wp:positionH")):
        if pos.get("relativeFrom") == "page":
            off = pos.find(qn("wp:posOffset"))
            off.text = str(int(off.text) - (LETTER_W - PAGE_W) // 2 * EMU_PER_TWIP)


def fix_sections(doc, footer_rid):
    """Every section: A4, the template's footer frame. Front matter in lower-case roman from i; chapters onwards
    in Arabic from 1 (the body-level sectPr)."""
    sects = doc.element.body.xpath(".//w:sectPr")
    for k, s in enumerate(sects):
        for tag in ("w:headerReference", "w:footerReference", "w:titlePg"):
            for e in s.findall(qn(tag)):
                s.remove(e)
        s.insert(0, parse_xml(f'<w:footerReference {nsdecls("w", "r")} w:type="default" r:id="{footer_rid}"/>'))
        sz = s.find(qn("w:pgSz"))
        sz.set(qn("w:w"), str(PAGE_W))
        sz.set(qn("w:h"), str(PAGE_H))
        pg = s.find(qn("w:pgNumType"))
        if pg is None:
            pg = parse_xml(f'<w:pgNumType {nsdecls("w")}/>')
            s.append(pg)
        last = s.getparent() is doc.element.body
        pg.set(qn("w:fmt"), "decimal" if last else "lowerRoman")
        if k == 0 or last:
            pg.set(qn("w:start"), "1")
        elif qn("w:start") in pg.attrib:
            del pg.attrib[qn("w:start")]
    mar = doc.element.body[-1].find(qn("w:pgMar"))
    for k, v in dict(top="1120", **MARGIN).items():
        mar.set(qn(f"w:{k}"), v)


def set_doc_properties(doc):
    """Replace the template's document properties (they name the sample's author and title). The converted file
    links core.xml with a lower-case relationship type that python-docx does not recognise, so the existing parts are
    rewritten in place (doc.core_properties would add a second core.xml)."""
    def put(part, blob):  # core.xml loads as an XmlPart, which serialises its element rather than its blob
        if hasattr(part, "_element"):
            part._element = parse_xml(blob)
        else:
            part._blob = blob
    authors = ", ".join(a for a, _ in TEAM)
    for part in doc.part.package.iter_parts():
        name = str(part.partname)
        if name == "/docProps/core.xml":
            put(part, ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<cp:coreProperties '
                       'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                       'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                       'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                       f'<dc:title>{esc(TITLE)}</dc:title><dc:subject>DSN4091 Capstone Project Phase-I Report</dc:subject>'
                       f'<dc:creator>{esc(authors)}</dc:creator><cp:lastModifiedBy>{esc(TEAM[0][0])}</cp:lastModifiedBy>'
                       '<dc:language>en-US</dc:language></cp:coreProperties>').encode())
        elif name == "/docProps/custom.xml":
            put(part, ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties '
                       'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
                       'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"/>').encode())
        elif name == "/docProps/app.xml":
            put(part, ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties '
                       'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
                       'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
                       '<Template>Normal</Template></Properties>').encode())


# OOXML child order (the converted template and our section breaks do not always follow it; Word can reject that)
PPR_ORDER = ("pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs "
             "suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi "
             "adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection "
             "textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange").split()
RPR_ORDER = ("rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid "
             "vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl "
             "cs em lang eastAsianLayout specVanish oMath").split()
LVL_ORDER = "start numFmt lvlRestart pStyle isLgl suff lvlText lvlPicBulletId legacy lvlJc pPr rPr".split()
SETTINGS_AFTER = "hdrShapeDefaults footnotePr endnotePr compat docVars rsids".split()  # updateFields precedes these


def _sort_children(el, order):
    kids = list(el)
    key = {t: i for i, t in enumerate(order)}
    ordered = sorted(kids, key=lambda k: key.get(k.tag.split("}")[1], len(order)))
    if ordered != kids:
        for k in kids:
            el.remove(k)
        for k in ordered:
            el.append(k)


def schema_order(doc):
    for ppr_ in doc.element.body.iter(qn("w:pPr")):
        if ppr_.getparent().tag == qn("w:p"):
            _sort_children(ppr_, PPR_ORDER)
    for rpr_ in doc.element.body.iter(qn("w:rPr")):
        if rpr_.getparent().tag == qn("w:r"):
            _sort_children(rpr_, RPR_ORDER)
    for lvl in doc.part.numbering_part.element.iter(qn("w:lvl")):
        _sort_children(lvl, LVL_ORDER)
    settings = doc.settings.element
    uf = settings.find(qn("w:updateFields"))
    later = [k for k in settings if k.tag.split("}")[1] in SETTINGS_AFTER]
    if uf is not None and later:
        later[0].addprevious(uf)


def cleanup_rels(doc):
    """Drop relationships (sample images, unused footers, hyperlinks) no longer referenced by the document."""
    xml = doc.element.xml
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype.split("/")[-1] in ("image", "footer", "header", "hyperlink") and f'"{rid}"' not in xml:
            doc.part.drop_rel(rid)


# ---------------------------------------------------------------- PDF rendering and page lookup
def soffice(src, fmt, outdir):
    subprocess.run(["soffice", "--headless", "--convert-to", fmt, "--outdir", outdir, src], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)


def pdf_pages(pdf):
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    pages = []
    for i in range(1, int(re.search(r"Pages:\s+(\d+)", info).group(1)) + 1):
        t = subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), pdf, "-"], capture_output=True,
                           text=True).stdout
        m = re.findall(r"Page\s*\|\s*([ivxlcdm]+|\d+)\s*$", t, re.M)
        pages.append((m[-1] if m else None, t))
    return pages


def norm(s):
    return re.sub(r"\s+", " ", s).strip().upper()


def locate(pages, needle, start):
    nd = norm(needle)
    return next((i for i in range(start, len(pages)) if nd in norm(pages[i][1])), None)


# ---------------------------------------------------------------- source excerpts (exact lines from the repository)
def snippet(path, start, end=None):
    """Lines of `path` from the first line matching regex `start` to the first later line matching `end`
    (inclusive), or to the end of the file. Returned verbatim: code listings in the report are not edited."""
    lines = open(path).read().split("\n")
    i = next(k for k, ln in enumerate(lines) if re.search(start, ln))
    j = next(k for k in range(i + 1, len(lines)) if re.search(end, lines[k])) if end else len(lines) - 1
    return "\n".join(lines[i:j + 1]).rstrip() + "\n"


# ---------------------------------------------------------------- build
def build(cached):
    import content
    LABELS.clear()
    REFS.order.clear()
    USED.clear()
    content.OTHER.clear()
    tmp = tempfile.mkdtemp(prefix="np_report_")
    shutil.copy(TEMPLATE, os.path.join(tmp, "template.doc"))
    soffice(os.path.join(tmp, "template.doc"), "docx", tmp)
    doc = Document(os.path.join(tmp, "template.docx"))
    footer_rid = next(rid for rid, rel in doc.part.rels.items() if rel.reltype.endswith("/footer")
                      and rel.target_ref.endswith("footer1.xml"))
    fix_footer(doc, footer_rid)
    fix_logo(doc)
    edit_front(doc)
    # delete the sample's content after the acknowledgement (keep the body-level sectPr)
    body = doc.element.body
    ack_end = next(p._p for p in doc.paragraphs if p.text.startswith("Last, but not least"))
    assert ack_end.xpath("./w:pPr/w:sectPr"), "acknowledgement section break not found"
    for el in list(ack_end.itersiblings()):
        if el.tag != qn("w:sectPr"):
            body.remove(el)
    b = Builder(doc)
    blocks = content.blocks(n, snippet)
    # pass 1: number figures and tables per chapter, so the text can refer to them before they appear
    chap, fig, tab = None, 0, 0
    for blk in blocks:
        if blk[0] in ("chapter", "appendix"):
            chap, fig, tab = str(blk[1]), 0, 0
        elif blk[0] == "fig":
            fig += 1
            LABELS[f"fig:{blk[1]}"] = f"Fig {chap}.{fig}"
        elif blk[0] == "tab" and blk[1]:
            tab += 1
            LABELS[f"tab:{blk[1]}"] = f"Table {chap}.{tab}"
    # content first (so the lists know their entries); the front matter is inserted before it afterwards
    marker = P("", ppr("Normal"))
    b.add(marker)
    first = True
    for blk in blocks:
        k = blk[0]
        if k == "chapter":
            b.chapter(blk[1], blk[2], first=first)
            first = False
        elif k == "appendix":
            b.appendix(blk[1], blk[2])
        elif k == "unnumbered":
            b.unnumbered(blk[1], blk[2])
        elif k == "h2":
            b.h2(blk[1], blk[2])
        elif k == "h3":
            b.h3(blk[1], blk[2])
        elif k == "p":
            b.para(blk[1], **(blk[2] if len(blk) > 2 else {}))
        elif k in ("ul", "ol"):
            b.list(blk[1], ordered=(k == "ol"))
        elif k == "fig":
            b.figure(blk[1], blk[2], blk[3], blk[4])
        elif k == "tab":
            b.table(blk[1], blk[2], blk[3], blk[4], blk[5], **(blk[6] if len(blk) > 6 else {}))
        elif k == "code":
            b.code(blk[1], blk[2], **(blk[3] if len(blk) > 3 else {}))
        elif k == "refs":
            b.refs([content.REFERENCES[key] for key in REFS.order])
        else:
            raise ValueError(k)
    unused = set(content.REFERENCES) - set(REFS.order)
    assert not unused, f"references never cited: {unused}"
    # front matter, in front of the marker: List of Figures, List of Tables, Abstract, Table of Contents
    b.anchor = marker
    c = cached.get
    b.front_heading("LIST OF FIGURES", page_break=False, bm="_FrontLoF")   # the acknowledgement ends with a break
    b.add(list_table(["FIGURE NO.", "TITLE", "PAGE NO."],
                     [(lab.replace("Fig ", ""), cap, bm, c(bm, "?"), 1, False) for lab, cap, bm in b.lof],
                     [1.0, 4.6, 0.95]))
    b.front_heading("LIST OF TABLES", bm="_FrontLoT")
    b.add(list_table(["TABLE NO.", "TITLE", "PAGE NO."],
                     [(lab.replace("Table ", ""), cap, bm, c(bm, "?"), 1, False) for lab, cap, bm in b.lot],
                     [1.0, 4.6, 0.95]))
    b.front_heading("ABSTRACT", bm="_FrontAbs")
    b.para(content.ABSTRACT(n))
    b.front_heading("TABLE OF CONTENTS")
    rows = [("", "List of Figures", "_FrontLoF", c("_FrontLoF", "?"), 1, True),
            ("", "List of Tables", "_FrontLoT", c("_FrontLoT", "?"), 1, False),
            ("", "Abstract", "_FrontAbs", c("_FrontAbs", "?"), 1, False)]
    for level, lab, title, bm in b.toc:
        if level == 0:
            no = lab.replace("CHAPTER ", "") if lab.startswith("CHAPTER") else ""
            rows.append((no, f"{lab}: {title}" if lab else title, bm, c(bm, "?"), 0, True))
        else:
            rows.append(("", f"{lab} {title}", bm, c(bm, "?"), level, False))
    b.add(list_table(["CHAPTER NO.", "TITLE", "PAGE NO."], rows, [1.0, 4.6, 0.95]))
    # front matter ends with a section break (it keeps roman numbering); chapters restart at 1
    b.add(P("", ppr("Normal", after=0, line=240, sect=sect_xml(1120, "lowerRoman", footer_rid))))
    marker.getparent().remove(marker)
    fix_sections(doc, footer_rid)
    cleanup_rels(doc)
    set_doc_properties(doc)
    settings = doc.settings.element
    if settings.find(qn("w:updateFields")) is None:
        settings.append(parse_xml(f'<w:updateFields {nsdecls("w")} w:val="true"/>'))
    schema_order(doc)
    doc.save(OUT_DOCX)
    shutil.rmtree(tmp, ignore_errors=True)
    return b


def page_map(b, pages):
    """Bookmark -> page label, found by searching the rendered PDF (front matter from the start, content after the
    table of contents)."""
    found = {}
    toc_end = max(i for i, (_, t) in enumerate(pages[:20]) if "TABLE OF CONTENTS" in t or "CHAPTER NO." in t)
    for bm, needle in (("_FrontLoF", "LIST OF FIGURES"), ("_FrontLoT", "LIST OF TABLES"), ("_FrontAbs", "ABSTRACT")):
        found[bm] = pages[locate(pages, needle, 0)][0]
    pos = toc_end + 1
    for level, lab, title, bm in b.toc:
        needle = f"{lab} {title}" if level or lab.startswith("CHAPTER") else (f"{lab}: {title}" if lab else title)
        i = locate(pages, needle, pos)
        assert i is not None, needle
        found[bm] = pages[i][0]
        pos = i
    for lab, cap, bm in b.lof + b.lot:
        i = locate(pages, f"{lab}:", toc_end + 1)
        assert i is not None, lab
        found[bm] = pages[i][0]
    missing = [k for k, (lbl, _) in enumerate(pages, 1) if lbl is None]
    assert not missing, f"pages without a footer number: {missing}"
    return found


def write_numbers_check():
    import content
    with open("report/NUMBERS_CHECK.md", "w") as f:
        f.write("# Numbers in the capstone report and where they come from\n\n"
                "Generated by `python report/build_report.py`. Every result number in the report is read from the "
                "paper's numbers pipeline (`paper/compute_numbers.py`, formatted as on the dashboard), never typed by "
                "hand. This table lists each one used: the macro name (as in `paper/numbers.tex`), the value printed in "
                "the report, what it is, and its source file.\n\n| macro | value in the report | what it is | source |\n"
                "|---|---|---|---|\n")
        for k, v in USED.items():
            desc, src = cn.N[k][2].replace("$\\pm$", "±"), cn.N[k][3]
            f.write(f"| `{k}` | {v.replace('|', '/')} | {desc} | `{src}` |\n")
        f.write("\n## Other numbers in the report (not results)\n\n")
        for what, val, src in content.OTHER:
            f.write(f"- {what}: **{val}** ({src})\n")


def main():
    b = build({})
    with tempfile.TemporaryDirectory() as tmp:
        soffice(OUT_DOCX, "pdf", tmp)
        cached = page_map(b, pdf_pages(os.path.join(tmp, os.path.basename(OUT_PDF))))
    b = build(cached)
    soffice(OUT_DOCX, "pdf", "report")
    pages = pdf_pages(OUT_PDF)
    again = page_map(b, pages)
    assert again == cached, {k: (cached.get(k), v) for k, v in again.items() if cached.get(k) != v}
    write_numbers_check()
    print(f"wrote {OUT_DOCX} and {OUT_PDF}: {len(pages)} pages; {len(b.lof)} figures, {len(b.lot)} tables, "
          f"{len(REFS.order)} references, {len(USED)} pipeline numbers")


if __name__ == "__main__":
    main()
