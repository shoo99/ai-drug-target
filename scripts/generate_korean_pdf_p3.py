#!/usr/bin/env python3
"""Generate Korean manuscript PDF for Paper 3 using WeasyPrint."""
import markdown
from weasyprint import HTML
from pathlib import Path

PAPER_DIR = Path(__file__).parent.parent / "paper3"

md_text = (PAPER_DIR / "manuscript_ko.md").read_text(encoding="utf-8")
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    body {{ font-family: 'Noto Sans KR', sans-serif; font-size: 11pt; line-height: 1.8; color: #212121; max-width: 180mm; margin: 20mm auto; }}
    h1 {{ font-size: 16pt; color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 8px; }}
    h2 {{ font-size: 13pt; color: #1565c0; margin-top: 25px; border-bottom: 1px solid #e0e0e0; }}
    h3 {{ font-size: 11pt; color: #283593; }}
    p {{ text-align: justify; margin-bottom: 8px; }}
    strong {{ color: #c62828; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 9.5pt; }}
    th {{ background: #0d47a1; color: white; padding: 8px 10px; text-align: left; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }}
    tr:nth-child(even) {{ background: #f5f7fa; }}
    @page {{ size: A4; margin: 25mm 20mm; @bottom-center {{ content: counter(page); font-size: 9pt; color: #757575; }} }}
</style>
</head>
<body>{html_body}</body>
</html>"""

pdf_path = PAPER_DIR / "manuscript_ko.pdf"
HTML(string=html, base_url=str(PAPER_DIR)).write_pdf(str(pdf_path))
print(f"Korean PDF: {pdf_path} ({pdf_path.stat().st_size//1024}KB)")
