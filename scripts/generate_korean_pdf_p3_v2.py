#!/usr/bin/env python3
"""Generate Korean manuscript PDF for Paper 3 v2.2 using WeasyPrint."""
import markdown
from weasyprint import HTML
from pathlib import Path

PAPER_DIR = Path(__file__).parent.parent / "paper3"

md_text = (PAPER_DIR / "manuscript_v2_2_ko.md").read_text(encoding="utf-8")
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    body {{ font-family: 'Noto Sans KR', sans-serif; font-size: 10.5pt; line-height: 1.75; color: #212121; max-width: 180mm; margin: 18mm auto; }}
    h1 {{ font-size: 15pt; color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 8px; line-height: 1.4; }}
    h2 {{ font-size: 12.5pt; color: #1565c0; margin-top: 24px; border-bottom: 1px solid #e0e0e0; }}
    h3 {{ font-size: 11pt; color: #283593; }}
    h4 {{ font-size: 10.5pt; color: #303f9f; }}
    p {{ text-align: justify; margin-bottom: 8px; }}
    strong {{ color: #c62828; }}
    blockquote {{ border-left: 3px solid #90a4ae; margin: 10px 0; padding: 4px 12px; color: #455a64; font-size: 9.5pt; background: #f5f7fa; }}
    img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #e0e0e0; }}
    code, pre {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }}
    pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 9pt; }}
    th {{ background: #0d47a1; color: white; padding: 6px 8px; text-align: left; }}
    td {{ padding: 5px 8px; border-bottom: 1px solid #e0e0e0; }}
    tr:nth-child(even) {{ background: #f5f7fa; }}
    @page {{ size: A4; margin: 22mm 18mm; @bottom-center {{ content: counter(page); font-size: 9pt; color: #757575; }} }}
</style>
</head>
<body>{html_body}</body>
</html>"""

pdf_path = PAPER_DIR / "manuscript_v2_2_ko.pdf"
HTML(string=html, base_url=str(PAPER_DIR)).write_pdf(str(pdf_path))
print(f"Korean PDF: {pdf_path} ({pdf_path.stat().st_size//1024}KB)")
