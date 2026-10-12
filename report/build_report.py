"""Build the DSN4091 capstone report from the university template.

    python report/build_report.py      # -> report/NeuroPlast_Capstone_Report.docx and .pdf, report/NUMBERS_CHECK.md

The template is docs/report_template/DSN4091_sample.doc (a sample report for another project; not committed). It is
converted to .docx with LibreOffice at build time, so the cover page, university logo, page size, margins, styles and
footer come from it. The cover, bonafide certificate and acknowledgement are edited in place; everything after them is
replaced by our content (report/content.py), written with the template's styles.

Every result number comes from the paper's numbers pipeline (paper/compute_numbers.py, via demo/build_dashboard.py);
report/NUMBERS_CHECK.md lists each number used with its macro, value and source file. The Table of Contents, List of
Figures and List of Tables use PAGEREF fields to bookmarks; their cached page numbers are taken from the rendered PDF
and checked against it.
"""
from __future__ import annotations

import copy
import html
import json
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

# ---------------------------------------------------------------- fill-ins (as given by the team; TBD stays visible)
TITLE = ("Testing Brain-Inspired Mechanisms for Task-Incremental Continual Reinforcement Learning: "
         "A Controlled Ablation Study")
REPORT_TYPE = "[Phase-I / Phase-II]"
TEAM = [("Vaibhav Tiwari", "TBD"), ("Kavish Bishnoi", "TBD"), ("Kushal Upadhyay", "TBD"),
        ("Adamya Upadhyay", "TBD"), ("Yash Yadav", "TBD"), ("Siddharth Ubbot", "TBD")]
GUIDE, GUIDE_DESIGNATION = "TBD", "TBD"
MONTH = "October 2026"

# ---------------------------------------------------------------- numbers (logged for NUMBERS_CHECK.md)
N = bd.numbers()
USED: dict[str, str] = {}


def n(key):
    USED[key] = N[key]
    return N[key]


# ---------------------------------------------------------------- page geometry (from the template's chapter pages)
TW = 835, 873            # body text indents (start, end) in twips, as the template's Body Text paragraphs
TEXT_W = 12240 - 880 - 540 - TW[0] - TW[1]   # usable width inside the indents, twips (= 9112, about 6.33 in)
MONO = "Courier New"


def esc(t):
    return html.escape(t, quote=False)


def rpr(b=False, i=False, sz=None, font=None, color=None, caps=False, shade=None):
    x = ""
    if font:
        x += f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}" w:eastAsia="{font}"/>'
    if b:
        x += "<w:b/><w:bCs/>"
    if i:
        x += "<w:i/><w:iCs/>"
    if caps:
        x += "<w:caps/>"
    if color:
        x += f'<w:color w:val="{color}"/>'
    if sz:
        x += f'<w:sz w:val="{int(sz * 2)}"/><w:szCs w:val="{int(sz * 2)}"/>'
    if shade:
        x += f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>'
    return f"<w:rPr>{x}</w:rPr>" if x else ""


def run(t, **k):
    return f'<w:r>{rpr(**k)}<w:t xml:space="preserve">{esc(t)}</w:t></w:r>'


def ppr(style=None, keep_next=False, page_break=False, numpr=None, tabs=None, before=0, after=120, line=360,
        rule="auto", ind=None, hanging=None, first=None, jc=None, outline=None, keep_lines=False, sect=None):
    x = f'<w:pStyle w:val="{style}"/>' if style else ""
    if keep_next:
        x += "<w:keepNext/>"
    if keep_lines:
        x += "<w:keepLines/>"
    if page_break:
        x += "<w:pageBreakBefore/>"
    if numpr is not None:
        x += f'<w:numPr><w:ilvl w:val="{numpr[1]}"/><w:numId w:val="{numpr[0]}"/></w:numPr>'
    if tabs:
        x += "<w:tabs>" + "".join(f'<w:tab w:val="{v}" w:pos="{p}"{" w:leader=%s" % chr(34) + ld + chr(34) if ld else ""}/>'
                                  for v, p, ld in tabs) + "</w:tabs>"
    x += f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="{rule}"/>'
    if ind is not None:
        a = f'<w:ind w:left="{ind[0]}" w:right="{ind[1]}"'
        if hanging:
            a += f' w:hanging="{hanging}"'
        if first:
            a += f' w:firstLine="{first}"'
        x += a + "/>"
    if jc:
        x += f'<w:jc w:val="{jc}"/>'
    if outline is not None:
        x += f'<w:outlineLvl w:val="{outline}"/>'
    if sect is not None:
        x += sect
    return f"<w:pPr>{x}</w:pPr>"


def P(inner, pp):
    return parse_xml(f"<w:p {nsdecls('w', 'r')}>{pp}{inner}</w:p>")


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
        elif part.startswith("<<fig:") or part.startswith("<<tab:"):
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
        self.body = doc.element.body
        self.anchor = self.body[-1]          # the body-level sectPr; new content goes before it
        self.bm_id = 1000
        self.toc: list[tuple] = []           # (level, label, title, bookmark)
        self.lof: list[tuple] = []           # (label, caption, bookmark)
        self.lot: list[tuple] = []
        self.numbering_setup()

    # -- low level
    def add(self, el):
        self.anchor.addprevious(el)
        return el

    def bookmark(self, name, inner):
        self.bm_id += 1
        return (f'<w:bookmarkStart w:id="{self.bm_id}" w:name="{name}"/>{inner}'
                f'<w:bookmarkEnd w:id="{self.bm_id}"/>')

    def numbering_setup(self):
        numbering = self.doc.part.numbering_part.element
        bullet = (f'<w:abstractNum {nsdecls("w")} w:abstractNumId="901"><w:multiLevelType w:val="singleLevel"/>'
                  '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/>'
                  '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="1195" w:hanging="360"/></w:pPr>'
                  '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:lvl></w:abstractNum>')
        dec = (f'<w:abstractNum {nsdecls("w")} w:abstractNumId="902"><w:multiLevelType w:val="singleLevel"/>'
               '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
               '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="1195" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>')
        first_num = numbering.find(qn("w:num"))
        for xml in (bullet, dec):
            el = parse_xml(xml)
            if first_num is not None:
                first_num.addprevious(el)
            else:
                numbering.append(el)
        self.numbering = numbering
        self.next_num = 900
        self.bullet_num = self.new_num(901)

    def new_num(self, abstract):
        self.next_num += 1
        nid = self.next_num
        self.numbering.append(parse_xml(
            f'<w:num {nsdecls("w")} w:numId="{nid}"><w:abstractNumId w:val="{abstract}"/>'
            '<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>'))
        return nid

    # -- blocks
    def para(self, text, jc="both", after=120, keep_next=False, sz=None, ind=TW, line=360, b=False, i=False):
        return self.add(P(inline(text, sz=sz, base_b=b, base_i=i),
                          ppr("BodyText", keep_next=keep_next, after=after, line=line, ind=ind, jc=jc)))

    def list(self, items, ordered=False):
        num = self.new_num(902) if ordered else self.bullet_num
        for k, t in enumerate(items):
            self.add(P(inline(t), ppr("BodyText", numpr=(num, 0), after=60 if k < len(items) - 1 else 140, line=360,
                                      ind=(1195, TW[1]), hanging=360, jc="both")))

    def front_heading(self, text, page_break=True, bm=None):
        inner = run(text, b=True, sz=16)
        if bm:
            inner = self.bookmark(bm, inner)
        return self.add(P(inner, ppr("Normal", page_break=page_break, before=60, after=360, line=240, jc="center")))

    def chapter(self, num, title):
        bm = f"_TocCh{num}"
        self.add(P(run(f"CHAPTER {num}", b=True, sz=16), ppr("Normal", page_break=True, keep_next=True, before=60,
                                                              after=0, line=360, jc="center")))
        self.add(P(self.bookmark(bm, run(title.upper(), b=True, sz=16)),
                   ppr("Heading1", keep_next=True, numpr=(0, 0), before=0, after=360, line=360, ind=(0, 0), jc="center")))
        self.toc.append((0, f"CHAPTER {num}", title.upper(), bm))
        self.chap = str(num)
        self.fig_n = self.tab_n = 0

    def appendix(self, letter, title):
        bm = f"_TocApp{letter}"
        self.add(P(self.bookmark(bm, run(f"APPENDIX {letter}: {title.upper()}", b=True, sz=16)),
                   ppr("Heading1", page_break=True, keep_next=True, numpr=(0, 0), before=60, after=360, line=360,
                       ind=(0, 0), jc="center")))
        self.toc.append((0, f"APPENDIX {letter}", title.upper(), bm))
        self.chap = letter
        self.fig_n = self.tab_n = 0

    def h2(self, num, title):
        bm = "_TocS" + num.replace(".", "_")
        self.add(P(self.bookmark(bm, run(num, b=True, sz=14) + "<w:r><w:tab/></w:r>" + run(title.upper(), b=True, sz=14)),
                   ppr("Heading2", keep_next=True, numpr=(0, 0), tabs=[("left", 1560, None)], before=240, after=120,
                       line=360, ind=(775, 0), jc="left")))
        self.toc.append((1, num, title.upper(), bm))

    def h3(self, num, title):
        bm = "_TocS" + num.replace(".", "_")
        self.add(P(self.bookmark(bm, run(num, b=True, sz=12) + "<w:r><w:tab/></w:r>" + run(title.upper(), b=True, sz=12)),
                   ppr("Heading3", keep_next=True, numpr=(0, 0), tabs=[("left", 1700, None)], before=180, after=100,
                       line=360, ind=(835, 0), hanging=0, jc="left")))
        self.toc.append((2, num, title.upper(), bm))

    def label(self, kind, lab):
        if kind == "fig":
            self.fig_n += 1
            LABELS[f"fig:{lab}"] = f"Fig {self.chap}.{self.fig_n}"
        else:
            self.tab_n += 1
            LABELS[f"tab:{lab}"] = f"Table {self.chap}.{self.tab_n}"

    def caption(self, kind, lab, text):
        name = LABELS[f"{kind}:{lab}"]
        bm = "_Cap" + re.sub(r"\W", "_", name)
        inner = self.bookmark(bm, run(name + ": ", b=True, sz=11) + inline(text, sz=11))
        (self.lof if kind == "fig" else self.lot).append((name, text, bm))
        return P(inner, ppr("Normal", keep_next=(kind == "tab"), before=60 if kind == "tab" else 80,
                            after=200 if kind == "fig" else 80, line=276, ind=(TW[0], TW[1]), jc="center"))

    def figure(self, lab, path, caption, width_in):
        from PIL import Image
        w, h = Image.open(path).size
        width = Inches(min(width_in, TEXT_W / 1440))
        height = Emu(int(width * h / w))
        inline_el = self.doc.part.new_pic_inline(path, width, height)
        p = P("", ppr("Normal", keep_next=True, before=120, after=0, line=240, ind=(TW[0], TW[1]), jc="center"))
        r = parse_xml(f"<w:r {nsdecls('w')}><w:drawing/></w:r>")
        r[0].append(inline_el)
        p.append(r)
        self.add(p)
        self.add(self.caption("fig", lab, caption))

    def table(self, lab, caption, header, rows, widths, sz=10, bold_first_col=False, align=None):
        self.add(self.caption("tab", lab, caption))
        self.add(self.tbl(header, rows, widths, sz, bold_first_col, align))
        self.add(P("", ppr("Normal", after=120, line=240)))

    def tbl(self, header, rows, widths, sz=10, bold_first_col=False, align=None, borders="all", head_shade="E7E6E0"):
        tw = [int(w * 1440) for w in widths]
        total = sum(tw)
        b = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        bdr = "".join(b.format(s) for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        xml = (f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>'
               f'<w:jc w:val="center"/><w:tblBorders>{bdr}</w:tblBorders><w:tblLayout w:type="fixed"/>'
               '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
               '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
               + "".join(f'<w:gridCol w:w="{w}"/>' for w in tw) + "</w:tblGrid>")

        def cell(text, w, head=False, bold=False, jc="left"):
            shade = f'<w:shd w:val="clear" w:color="auto" w:fill="{head_shade}"/>' if head else ""
            paras = text if isinstance(text, list) else [text]
            ps = "".join(f'<w:p>{ppr("Normal", after=0, line=240, jc=jc)}{inline(t, sz=sz, base_b=head or bold)}</w:p>'
                         for t in paras)
            return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}</w:tcPr>{ps}</w:tc>'
        if header:
            xml += '<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + "".join(
                cell(h, w, head=True, jc="center") for h, w in zip(header, tw)) + "</w:tr>"
        for r in rows:
            xml += "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(
                cell(c, w, bold=(bold_first_col and j == 0), jc=(align[j] if align else "left"))
                for j, (c, w) in enumerate(zip(r, tw))) + "</w:tr>"
        return parse_xml(xml + "</w:tbl>")

    def code(self, caption, text, sz=8.5):
        if caption:
            self.add(P(inline(caption, sz=10.5, base_i=True),
                       ppr("Normal", keep_next=True, before=120, after=60, line=240, ind=(TW[0], TW[1]), jc="left")))
        lines = text.rstrip("\n").split("\n")
        cw = round(sz * 12)  # monospace advance (0.6 em) in twips
        ps = ""
        for ln in lines:  # leading spaces become an indent, so wrapped lines continue 4 columns further in
            k = len(ln) - len(ln.lstrip(" "))
            ps += (f'<w:p>{ppr("Normal", after=0, line=240, ind=((k + 4) * cw, 0), hanging=4 * cw, jc="left")}'
                   f'<w:r>{rpr(font=MONO, sz=sz)}<w:t xml:space="preserve">{esc(ln.lstrip(" ")) or " "}</w:t></w:r></w:p>')
        w = TEXT_W
        xml = (f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{w}" w:type="dxa"/><w:jc w:val="center"/>'
               '<w:tblBorders><w:left w:val="single" w:sz="18" w:space="0" w:color="9CA3AF"/></w:tblBorders>'
               '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="140" w:type="dxa"/>'
               '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr>'
               f'<w:tblGrid><w:gridCol w:w="{w}"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>'
               f'<w:shd w:val="clear" w:color="auto" w:fill="F4F4F1"/></w:tcPr>{ps}</w:tc></w:tr></w:tbl>')
        self.add(parse_xml(xml))
        self.add(P("", ppr("Normal", after=80, line=240)))

    def pagebreak(self):
        self.add(P('<w:r><w:br w:type="page"/></w:r>', ppr("Normal", after=0, line=240)))

    def refs(self, entries):
        bm = "_TocRefs"
        self.add(P(self.bookmark(bm, run("REFERENCES", b=True, sz=16)),
                   ppr("Heading1", page_break=True, keep_next=True, numpr=(0, 0), before=60, after=360, line=360,
                       ind=(0, 0), jc="center")))
        self.toc.append((0, "", "REFERENCES", bm))
        for k, txt in enumerate(entries, 1):
            self.add(P(run(f"[{k}]", sz=11) + "<w:r><w:tab/></w:r>" + inline(txt, sz=11),
                       ppr("Normal", tabs=[("left", TW[0] + 560, None)], after=100, line=276,
                           ind=(TW[0] + 560, TW[1]), hanging=560, jc="both")))


# ---------------------------------------------------------------- page-number fields for the lists
def pageref(bm, cached, sz=11, b=False):
    r = rpr(sz=sz, b=b)
    if bm.startswith("_Front"):
        # LibreOffice renders PAGEREF in Arabic even when the target section is lower-roman (and ignores
        # the \\* roman switch), so front-matter rows carry the roman label read from the rendered PDF.
        return f'<w:r>{r}<w:t>{esc(str(cached))}</w:t></w:r>'
    return (f'<w:r>{r}<w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r>{r}<w:instrText xml:space="preserve"> PAGEREF {bm} \\h </w:instrText></w:r>'
            f'<w:r>{r}<w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r>{r}<w:t>{esc(str(cached))}</w:t></w:r>'
            f'<w:r>{r}<w:fldChar w:fldCharType="end"/></w:r>')


def list_table(header, rows, widths, sz=11):
    """Bordered list table in the template's style; rows are (col0, title, bookmark, cached, level, group_start)."""
    tw = [int(w * 1440) for w in widths]
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
        bold = level == 0
        ind = {0: 0, 1: 300, 2: 600}[level]
        before = 100 if group else 20
        cells = [
            f'<w:p>{ppr("Normal", before=before, after=20, line=240, jc="center")}{run(c0, b=bold, sz=sz)}</w:p>',
            f'<w:p>{ppr("Normal", before=before, after=20, line=240, ind=(ind, 0), jc="left")}{inline(title, sz=sz, base_b=bold)}</w:p>',
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


def demote(p, doc):
    """Front-matter paragraphs that use heading styles in the template: give them Normal style with the heading
    style's formatting copied as direct formatting, so the look is unchanged and Word's TOC ignores them."""
    style = p.style
    sp = style.element.find(qn("w:pPr"))
    sr = style.element.find(qn("w:rPr"))
    pPr = p._p.get_or_add_pPr()
    for tag in ("w:ind", "w:jc", "w:spacing"):
        if pPr.find(qn(tag)) is None and sp is not None and sp.find(qn(tag)) is not None:
            pPr.append(copy.deepcopy(sp.find(qn(tag))))
    p.style = doc.styles["Normal"]
    for r in p.runs:
        rp = r._r.get_or_add_rPr()
        if sr is not None:
            for tag in ("w:b", "w:bCs", "w:sz", "w:szCs"):
                if rp.find(qn(tag)) is None and sr.find(qn(tag)) is not None:
                    rp.append(copy.deepcopy(sr.find(qn(tag))))


def edit_front(doc):
    P_ = doc.paragraphs
    # sanity: the template paragraphs we edit are where we expect them
    assert P_[2].text.startswith("DSN4091") and P_[6].text.startswith("STUDENT NAME"), "unexpected template layout"
    assert P_[26].text.replace(" ", "") == "BONAFIDECERTIFICATE" and P_[45].text == "ACKNOWLEDGEMENT"
    # title (template: one Heading 1 line) -> our title in capitals, centred
    title_lines = ["TESTING BRAIN-INSPIRED MECHANISMS FOR", "TASK-INCREMENTAL CONTINUAL", "REINFORCEMENT LEARNING:",
                   "A CONTROLLED ABLATION STUDY"]
    assert " ".join(title_lines) == TITLE.upper()
    set_text(P_[0], title_lines[0])
    r0 = P_[0].runs[0]
    r0.font.size = Pt(16)
    for line in title_lines[1:]:
        el = copy.deepcopy(r0._r)
        for t in el.findall(qn("w:t")):
            el.remove(t)
        el.append(parse_xml(f'<w:br {nsdecls("w")}/>'))
        el.append(parse_xml(f'<w:t {nsdecls("w")} xml:space="preserve">{esc(line)}</w:t>'))
        P_[0]._p.append(el)
    set_text(P_[2], f"DSN4091-CAPSTONE PROJECT {REPORT_TYPE.upper()}")
    set_text(P_[3], f"{REPORT_TYPE} Report")
    # team: the template's single "STUDENT NAME (21BAI10xxx)" line becomes one line per member, same formatting
    tmpl = P_[6]._p
    prev = tmpl
    for k, (name, reg) in enumerate(TEAM):
        el = copy.deepcopy(tmpl)
        prev.addnext(el)
        prev = el
    tmpl.getparent().remove(tmpl)
    P_ = doc.paragraphs
    for k, (name, reg) in enumerate(TEAM):
        p = P_[6 + k]
        p.runs[0].text = name + " "
        p.runs[1].text = f"({reg})"
        for r in p.runs:
            r.font.size = Pt(13)
        pf = p.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(2), 1.0
    tp = P_[0].paragraph_format
    tp.alignment, tp.left_indent, tp.first_line_indent = 1, Pt(0), Pt(0)
    sub = next(p for p in P_[:12] if p.text.strip() == "Submitted by")
    sub.paragraph_format.space_before, sub.paragraph_format.space_after = Pt(8), Pt(4)
    P_[6 + len(TEAM) - 1].paragraph_format.space_after = Pt(10)
    # drop some of the cover's empty spacer lines so six names still fit on the cover page
    P_ = doc.paragraphs
    for p in list(P_[:30]):
        if p.text == "" and not p._p.xpath(".//w:drawing|.//w:pict|.//w:sectPr") and p.style is not None \
                and p.style.name in ("Body Text", "Normal") and p.runs and p.runs[0].font.size \
                and p.runs[0].font.size.pt >= 13:
            p._p.getparent().remove(p._p)
    P_ = doc.paragraphs
    month = next(p for p in P_[:30] if p.text.strip().startswith("September"))
    set_text(month, MONTH)
    # bonafide certificate
    cert0 = next(i for i, p in enumerate(P_) if p.text.startswith("Certified"))
    p0, p1 = P_[cert0], P_[cert0 + 1]
    names = ", ".join(f"{a} ({b})" for a, b in TEAM)
    for r in list(p0.runs[1:]):
        r._r.getparent().remove(r._r)
    p0.runs[0].text = ""
    x = (run("Certified that this project report titled ", sz=14) + run(f"“{TITLE.upper()}”", b=True, sz=14)
         + run(" is the bonafide work of ", sz=14) + run(f"“{names}”", b=True, sz=14)
         + run(f" who carried out the project work (DSN4091 Capstone Project {REPORT_TYPE}) under my supervision. "
               "Certified further that to the best of my knowledge the work reported at this time does not form part "
               "of any other project/research work based on which a degree or award was conferred on an earlier "
               "occasion on this or any other candidate.", sz=14))
    for el in parse_xml(f"<w:p {nsdecls('w')}>{x}</w:p>"):
        p0._p.append(el)
    p0.paragraph_format.line_spacing = 1.5
    p1._p.getparent().remove(p1._p)
    P_ = doc.paragraphs
    # guide block: replace only the template's red placeholder runs
    gi = next(i for i, p in enumerate(P_) if "PROGRAM" in p.text and "GUIDE" in p.text)
    for r in P_[gi + 1].runs:
        if r.font.color is not None and r.font.color.type is not None and str(r.font.color.rgb) == "FF0000":
            r.text = ""
    red = [r for r in P_[gi + 1].runs if r.font.color is not None and r.font.color.type is not None
           and str(r.font.color.rgb) == "FF0000"]
    red[0].text = GUIDE
    for r in P_[gi + 2].runs:
        if r.font.color is not None and r.font.color.type is not None and str(r.font.color.rgb) == "FF0000":
            r.text = GUIDE_DESIGNATION
    for p in (P_[gi + 1], P_[gi + 2]):
        for r in p.runs:
            r.font.color.rgb = None
    viva = next(p for p in P_ if "Viva Voce" in p.text)
    for r in viva.runs:
        if "Phase-I" in r.text:
            r.text = f"DSN4091 Capstone Project {REPORT_TYPE}"
    # acknowledgement: only the guide's name changes
    ack = next(p for p in P_ if p.text.startswith("We would like to thank our internal guide") or
               "internal guide" in p.text)
    for r in ack.runs:
        if "internal guide" in r.text:
            r.text = f"our project guide, {GUIDE}"
            r.font.color.rgb = None
    # front-matter paragraphs that use heading styles -> Normal with the same look (keeps Word's TOC clean)
    for p in doc.paragraphs[:60]:
        if p.style is not None and p.style.name.startswith("Heading"):
            demote(p, doc)


# ---------------------------------------------------------------- sections, footer, numbering
def sect_xml(top, fmt, start=None, footer_rid=None):
    st = f' w:start="{start}"' if start is not None else ""
    return (f'<w:sectPr {nsdecls("w", "r")}><w:footerReference w:type="default" r:id="{footer_rid}"/>'
            '<w:type w:val="nextPage"/><w:pgSz w:w="12240" w:h="15840"/>'
            f'<w:pgMar w:top="{top}" w:right="540" w:bottom="960" w:left="880" w:header="0" w:footer="764" w:gutter="0"/>'
            f'<w:pgNumType w:fmt="{fmt}"{st}/><w:docGrid w:type="default" w:linePitch="360" w:charSpace="0"/></w:sectPr>')


def widen_footer(doc, footer_rid):
    """The template's footer frame ("Page | n") is 0.52 in wide, enough for one digit only; in the template's
    rendering two-digit page numbers wrap out of view. Widen the frame so "Page | 10" and "Page | xviii" fit."""
    ftr = doc.part.related_parts[footer_rid].element
    for el in ftr.iter():
        if el.tag.endswith("}extent") or el.tag.endswith("}ext"):
            if el.get("cx") == "502920":
                el.set("cx", "1280160")


def fix_sections(doc, footer_rid):
    """Every section uses the template's footer frame ("Page | n"). Front matter: lower-case roman from i.
    Chapters onwards: Arabic from 1 (set on the body-level sectPr)."""
    sects = doc.element.body.xpath(".//w:sectPr")
    for k, s in enumerate(sects):
        for tag in ("w:headerReference", "w:footerReference", "w:titlePg"):
            for e in s.findall(qn(tag)):
                s.remove(e)
        s.insert(0, parse_xml(f'<w:footerReference {nsdecls("w", "r")} w:type="default" r:id="{footer_rid}"/>'))
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
    # body-level (last) section: the template's chapter-page geometry
    last = doc.element.body[-1]
    mar = last.find(qn("w:pgMar"))
    for k, v in dict(top="1120", right="540", bottom="960", left="880", header="0", footer="764").items():
        mar.set(qn(f"w:{k}"), v)


def set_doc_properties(doc):
    """Replace the template's document properties (they name the sample's author and title). The converted file
    links core.xml with a lower-case relationship type that python-docx does not recognise, so the existing parts are
    rewritten in place instead of using doc.core_properties (which would add a second core.xml)."""
    def put(part, blob):  # core.xml loads as an XmlPart, which serialises its element rather than its blob
        if hasattr(part, "_element"):
            part._element = parse_xml(blob)
        else:
            part._blob = blob

    for part in doc.part.package.iter_parts():
        name = str(part.partname)
        if name == "/docProps/core.xml":
            put(part, (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<cp:coreProperties '
                'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                f'<dc:title>{esc(TITLE)}</dc:title><dc:subject>DSN4091 Capstone Project Report</dc:subject>'
                '<dc:creator>NeuroPlast team</dc:creator><cp:lastModifiedBy>NeuroPlast team</cp:lastModifiedBy>'
                '<dc:language>en-US</dc:language></cp:coreProperties>').encode())
        elif name == "/docProps/custom.xml":
            put(part, ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties '
                          'xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
                          'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"/>').encode())


# OOXML child order (the converted template and our section breaks do not always follow it; Word can reject that)
PPR_ORDER = ("pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs "
             "suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi "
             "adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection "
             "textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange").split()
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
    for ppr in doc.element.body.iter(qn("w:pPr")):
        if ppr.getparent().tag == qn("w:p"):
            _sort_children(ppr, PPR_ORDER)
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


# ---------------------------------------------------------------- PDF checks
def soffice(src, fmt, outdir):
    subprocess.run(["soffice", "--headless", "--convert-to", fmt, "--outdir", outdir, src], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)


def pdf_pages(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    pages = []
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), pdf, "-"], capture_output=True,
                           text=True).stdout
        m = re.findall(r"Page\s*\|\s*([ivxlcdm]+|\d+)\s*$", t, re.M)
        pages.append((m[-1] if m else None, t))
    return pages


def norm(s):
    return re.sub(r"\s+", " ", s).strip().upper()


def locate(pages, needle, start):
    nd = norm(needle)
    for i in range(start, len(pages)):
        if nd in norm(pages[i][1]):
            return i
    return None


# ---------------------------------------------------------------- bibliography (paper/refs.bib -> numbered list)
LATEX = {r"{\'s}": "ś", "``": "\u201c", "''": "\u201d", r"{\l}": "ł", r"{\L}": "Ł", r"{\k{a}}": "ą", r"{\'n}": "ń", r"{\'e}": "é", r'{\"o}': "ö", r'{\"u}': "ü",
         r"{\o}": "ø", r"{\'a}": "á", r"{\'o}": "ó", r"{\'i}": "í", r"{\v{s}}": "š", r"{\c{c}}": "ç", "--": "–"}


def bib():
    s = open("paper/refs.bib").read()
    out = {}
    for m in re.finditer(r"((?:%[^\n]*\n)*)@(\w+)\{([^,]+),(.*?)\n\}", s, re.S):
        com, typ, key, body = m.groups()
        f = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body)}
        f.update(type=typ, key=key, todo=("TODO" in com) or ("TODO" in f.get("note", "")))
        out[key] = f
    return out


def tex(t):
    for a, b in LATEX.items():
        t = t.replace(a, b)
    return t.replace("{", "").replace("}", "").replace("\\&", "&")


def authors(a):
    names = []
    for nm in tex(a).split(" and "):
        last, _, first = nm.partition(",")
        initials = " ".join(x[0] + "." for x in re.split(r"[\s]+", first.strip()) if x and x[0].isalpha())
        initials = initials.replace(". -", ".-")
        names.append(f"{initials} {last.strip()}".strip())
    if len(names) > 6:
        return names[0] + " et al."
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + (", and " if len(names) > 2 else " and ") + names[-1]


def fmt_ref(e):
    a, t, y = authors(e["author"]), tex(e["title"]), e.get("year", "")
    if e["type"] == "article":
        x = f"{a}, “{t},” *{tex(e['journal'])}*"
        if e.get("volume"):
            x += f", vol. {e['volume']}"
        if e.get("number"):
            x += f", no. {e['number']}"
        if e.get("pages"):
            x += f", pp. {tex(e['pages'])}"
        return x + f", {y}."
    if e["type"] == "inproceedings":
        x = f"{a}, “{t},” in *{tex(e['booktitle'])}*"
        if e.get("pages"):
            x += f", pp. {tex(e['pages'])}"
        return x + f", {y}."
    x = f"{a}, “{t},”"
    if e.get("howpublished"):
        x += f" {tex(e['howpublished'])},"
    return x + f" {y}."


# ---------------------------------------------------------------- source excerpts (exact lines from the repository)
def snippet(path, start, end=None, count=None):
    """Lines of `path` from the first line matching regex `start` to the first later line matching `end`
    (inclusive), or `count` lines. Returned verbatim (code listings in the report are not edited)."""
    lines = open(path).read().split("\n")
    i = next(k for k, ln in enumerate(lines) if re.search(start, ln))
    if count:
        j = i + count - 1
    elif end:
        j = next(k for k in range(i + 1, len(lines)) if re.search(end, lines[k]))
    else:
        j = len(lines) - 1
    return "\n".join(lines[i:j + 1]).rstrip() + "\n"


# ---------------------------------------------------------------- build
def build(cached, refs_out=None):
    import content
    LABELS.clear()
    REFS.order.clear()
    USED.clear()
    tmp = tempfile.mkdtemp(prefix="np_report_")
    shutil.copy(TEMPLATE, os.path.join(tmp, "template.doc"))
    soffice(os.path.join(tmp, "template.doc"), "docx", tmp)
    doc = Document(os.path.join(tmp, "template.docx"))
    footer_rid = next(rid for rid, rel in doc.part.rels.items() if rel.reltype.endswith("/footer")
                      and rel.target_ref.endswith("footer1.xml"))
    widen_footer(doc, footer_rid)
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
    # pass 1: number the figures and tables so the text can refer to them before they appear
    chap = None
    fig = tab = 0
    for blk in blocks:
        if blk[0] in ("chapter", "appendix"):
            chap, fig, tab = str(blk[1]), 0, 0
        elif blk[0] == "fig":
            fig += 1
            LABELS[f"fig:{blk[1]}"] = f"Fig {chap}.{fig}"
        elif blk[0] == "tab":
            tab += 1
            LABELS[f"tab:{blk[1]}"] = f"Table {chap}.{tab}"
    # front matter is inserted later, in front of chapter 1; content goes first so the lists know their entries
    marker = P("", ppr("Normal"))
    b.add(marker)
    first_chapter = True
    for blk in blocks:
        k = blk[0]
        if k == "chapter":
            b.chapter(blk[1], blk[2]) if not first_chapter else _first_chapter(b, blk[1], blk[2])
            first_chapter = False
        elif k == "appendix":
            b.appendix(blk[1], blk[2])
        elif k == "h2":
            b.h2(blk[1], blk[2])
        elif k == "h3":
            b.h3(blk[1], blk[2])
        elif k == "p":
            b.para(blk[1], **(blk[2] if len(blk) > 2 else {}))
        elif k == "ul":
            b.list(blk[1])
        elif k == "ol":
            b.list(blk[1], ordered=True)
        elif k == "fig":
            b.fig_n = int(LABELS[f"fig:{blk[1]}"].split(".")[-1]) - 1
            b.figure(blk[1], blk[2], blk[3], blk[4])
        elif k == "tab":
            b.table(blk[1], blk[2], blk[3], blk[4], blk[5], **(blk[6] if len(blk) > 6 else {}))
        elif k == "code":
            b.code(blk[1], blk[2], **(blk[3] if len(blk) > 3 else {}))
        elif k == "pagebreak":
            b.pagebreak()
        elif k == "refs":
            refs = bib()
            b.refs([fmt_ref(refs[key]) for key in REFS.order])
            if refs_out is not None:
                refs_out.extend((i + 1, key, refs[key]["todo"], fmt_ref(refs[key])) for i, key in enumerate(REFS.order))
        else:
            raise ValueError(k)
    # front matter, in front of the marker
    b.anchor = marker
    ai = b.front_heading("DECLARATION ON THE USE OF AI TOOLS", page_break=False)
    b.para("[TODO: written by the team]", jc="left")
    c = cached.get
    b.front_heading("LIST OF FIGURES", bm="_FrontLoF")
    b.add(list_table(["FIGURE NO.", "TITLE", "PAGE NO."],
                     [(lab.replace("Fig ", ""), cap, bm, c(bm, "?"), 1, False) for lab, cap, bm in b.lof],
                     [1.05, 4.45, 0.95]))
    b.front_heading("LIST OF TABLES", bm="_FrontLoT")
    b.add(list_table(["TABLE NO.", "TITLE", "PAGE NO."],
                     [(lab.replace("Table ", ""), cap, bm, c(bm, "?"), 1, False) for lab, cap, bm in b.lot],
                     [1.05, 4.45, 0.95]))
    b.front_heading("ABSTRACT", bm="_FrontAbs")
    b.para(content.ABSTRACT(n), line=360)
    b.front_heading("TABLE OF CONTENTS")
    rows = [("", "List of Figures", "_FrontLoF", c("_FrontLoF", "?"), 1, True),
            ("", "List of Tables", "_FrontLoT", c("_FrontLoT", "?"), 1, False),
            ("", "Abstract", "_FrontAbs", c("_FrontAbs", "?"), 1, False)]
    for level, lab, title, bm in b.toc:
        if level == 0:
            no = lab.replace("CHAPTER ", "") if lab.startswith("CHAPTER") else ""
            t = f"{lab}: {title}" if lab else title
            rows.append((no, t, bm, c(bm, "?"), 0, True))
        else:
            rows.append(("", f"{lab} {title}", bm, c(bm, "?"), level, False))
    b.add(list_table(["CHAPTER NO.", "TITLE", "PAGE NO."], rows, [1.05, 4.45, 0.95]))
    # front matter ends here: section break (roman numbering continues in this section)
    b.add(P("", ppr("Normal", after=0, line=240, sect=sect_xml(1120, "lowerRoman", footer_rid=footer_rid))))
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


def _first_chapter(b, num, title):
    """Chapter 1 follows a section break (new page already), so no page break before it."""
    bm = f"_TocCh{num}"
    b.add(P(run(f"CHAPTER {num}", b=True, sz=16), ppr("Normal", keep_next=True, before=60, after=0, line=360, jc="center")))
    b.add(P(b.bookmark(bm, run(title.upper(), b=True, sz=16)),
            ppr("Heading1", keep_next=True, numpr=(0, 0), before=0, after=360, line=360, ind=(0, 0), jc="center")))
    b.toc.append((0, f"CHAPTER {num}", title.upper(), bm))
    b.chap = str(num)
    b.fig_n = b.tab_n = 0


def page_map(b, pages):
    """Bookmark -> page label, found by searching the rendered PDF (front matter from the start, content after the
    table of contents)."""
    found = {}
    toc_end = max(i for i, (_, t) in enumerate(pages[:20]) if "TABLE OF CONTENTS" in t or "CHAPTER NO." in t)
    for bm, needle in (("_FrontLoF", "LIST OF FIGURES"), ("_FrontLoT", "LIST OF TABLES"), ("_FrontAbs", "ABSTRACT")):
        found[bm] = pages[locate(pages, needle, 0)][0]
    pos = toc_end + 1
    for level, lab, title, bm in b.toc:
        needle = (f"{lab} {title}" if level else (f"{lab}: {title}" if lab.startswith("APPENDIX") else
                                                  (f"{lab} {title}" if lab else title)))
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
    return found, toc_end


def main():
    refs_out = []
    b = build({})
    with tempfile.TemporaryDirectory() as tmp:
        soffice(OUT_DOCX, "pdf", tmp)
        pages = pdf_pages(os.path.join(tmp, os.path.basename(OUT_PDF)))
    cached, _ = page_map(b, pages)
    b = build(cached, refs_out)
    soffice(OUT_DOCX, "pdf", "report")
    pages = pdf_pages(OUT_PDF)
    again, toc_end = page_map(b, pages)
    assert again == cached, {k: (cached.get(k), v) for k, v in again.items() if cached.get(k) != v}
    write_checks(b, refs_out, pages, again)
    print(f"wrote {OUT_DOCX} and {OUT_PDF}: {len(pages)} pages; {len(b.lof)} figures, {len(b.lot)} tables, "
          f"{len(REFS.order)} references, {len(USED)} pipeline numbers")


def write_checks(b, refs_out, pages, pmap):
    with open("report/NUMBERS_CHECK.md", "w") as f:
        f.write("# Numbers in the capstone report and where they come from\n\n"
                "Generated by `python report/build_report.py`. Every result number in the report is read from the "
                "paper's numbers pipeline (`paper/compute_numbers.py`, formatted as on the dashboard), never typed by "
                "hand. This table lists each one used: the macro name (as in `paper/numbers.tex`), the value printed in "
                "the report, what it is, and its source file.\n\n| macro | value in the report | what it is | source |\n"
                "|---|---|---|---|\n")
        for k, v in USED.items():
            desc, src = cn.N[k][2].replace("$\\pm$", "±"), cn.N[k][3]
            val = v.replace("|", "/")
            f.write(f"| `{k}` | {val} | {desc} | `{src}` |\n")
        f.write("\n## Other numbers in the report (not results)\n\n")
        import content
        for what, val, src in content.OTHER:
            f.write(f"- {what}: **{val}** ({src})\n")
    with open("report/REFERENCES_TO_VERIFY.md", "w") as f:
        f.write("# References in the report that still need checking\n\n"
                "These entries are marked `TODO verify` in `paper/refs.bib` (not checked against a local copy of the "
                "PDF, or with metadata the local copy does not print). The report cites them with their standard "
                "metadata; the marks are kept here, not in the report.\n\n| no. in report | key | entry as printed |\n"
                "|---|---|---|\n")
        for k, key, todo, txt in refs_out:
            if todo:
                f.write(f"| [{k}] | `{key}` | {txt.replace('|', '/')} |\n")


if __name__ == "__main__":
    main()
