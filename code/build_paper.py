"""Build the .docx paper from paper/paper.md. Supported syntax:
  # / ## / ### headings; paragraphs; - bullets; 1. numbered; **bold**, *italic*; Markdown tables (| a | b |);
  {{csv:path/to/file.csv|col1,col2,...|caption}}  -> table from CSV (selected columns, optional rounding)
  {{fig:figures/name.png|caption|width_in}}       -> figure with caption
Usage: python build_paper.py paper/paper.md paper/paper.docx
"""
import os, re, sys
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.path.join(os.path.dirname(__file__), "..")

def add_runs(par, text):
    pos = 0
    for m in re.finditer(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if m.start() > pos: par.add_run(text[pos:m.start()])
        t = m.group(0)
        if t.startswith("**"): r = par.add_run(t[2:-2]); r.bold = True
        elif t.startswith("`"): r = par.add_run(t[1:-1]); r.font.name = "Consolas"
        else: r = par.add_run(t[1:-1]); r.italic = True
        pos = m.end()
    if pos < len(text): par.add_run(text[pos:])

def add_table(doc, rows, caption=None, font=8):
    if caption:
        p = doc.add_paragraph(); r = p.add_run(caption); r.bold = True; r.font.size = Pt(9)
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol); t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j); cell.text = ""
            p = cell.paragraphs[0]; add_runs(p, str(row[j]) if j < len(row) else "")
            for run in p.runs:
                run.font.size = Pt(font)
                if i == 0: run.bold = True
    doc.add_paragraph("")

def fmt(x):
    """Numbers from numeric CSV columns: at most three decimals, trailing zeros removed; strings are kept as written."""
    if isinstance(x, float):
        if x != x: return "n/a"
        s = f"{x:.3f}".rstrip("0").rstrip("."); return s if s not in ("", "-0") else "0"
    s = str(x)
    return "n/a" if s == "" else s

def main(src, dst):
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
    lines = open(src, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\{\{csv:([^|]+)\|([^|]*)\|([^}]*)\}\}", line.strip())
        if m:
            path, cols, cap = m.group(1), m.group(2), m.group(3)
            if not os.path.exists(os.path.join(ROOT, path)):
                p = doc.add_paragraph(); r = p.add_run(f"[table pending: {path}]"); r.italic = True; i += 1; continue
            df = pd.read_csv(os.path.join(ROOT, path), keep_default_na=False, dtype=str)  # 'NA' is a corpus name; numbers kept as written
            if cols.strip():
                df = df[[c.strip() for c in cols.split(",")]]
            rows = [list(df.columns)] + [[fmt(v) for v in r] for r in df.itertuples(index=False)]
            add_table(doc, rows, cap); i += 1; continue
        m = re.match(r"\{\{fig:([^|]+)\|([^|]*)\|?([^}]*)\}\}", line.strip())
        if m:
            path, cap, w = m.group(1), m.group(2), m.group(3)
            p = os.path.join(ROOT, path)
            if os.path.exists(p):
                doc.add_picture(p, width=Inches(float(w) if w else 6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                c = doc.add_paragraph(); r = c.add_run(cap); r.italic = True; r.font.size = Pt(9)
            i += 1; continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells): rows.append(cells)
                i += 1
            add_table(doc, rows); continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#")); doc.add_heading(line.lstrip("#").strip(), level=min(level, 4))
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet"); add_runs(p, line[2:])
        elif re.match(r"^\d+\. ", line):
            p = doc.add_paragraph(style="List Number"); add_runs(p, re.sub(r"^\d+\. ", "", line))
        elif line.strip() == "":
            pass
        else:
            p = doc.add_paragraph(); add_runs(p, line)
        i += 1
    doc.save(dst); print("wrote", dst)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
