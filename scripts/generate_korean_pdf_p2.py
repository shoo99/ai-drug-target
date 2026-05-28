#!/usr/bin/env python3
"""Generate Korean manuscript PDF for Paper 2 using WeasyPrint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import markdown
from weasyprint import HTML

PAPER_DIR = Path(__file__).parent.parent / "paper2"

def main():
    md_path = PAPER_DIR / "manuscript_ko.md"
    if not md_path.exists():
        print("manuscript_ko.md not found")
        return

    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    body {{
        font-family: 'Noto Sans KR', -apple-system, sans-serif;
        font-size: 11pt;
        line-height: 1.8;
        color: #212121;
        max-width: 180mm;
        margin: 20mm auto;
    }}
    h1 {{
        font-size: 16pt;
        color: #0d47a1;
        border-bottom: 2px solid #0d47a1;
        padding-bottom: 8px;
        margin-top: 30px;
    }}
    h2 {{
        font-size: 13pt;
        color: #1565c0;
        margin-top: 25px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 4px;
    }}
    h3 {{
        font-size: 11pt;
        color: #283593;
        margin-top: 20px;
    }}
    p {{
        text-align: justify;
        margin-bottom: 8px;
    }}
    strong {{
        color: #c62828;
    }}
    code {{
        background: #f5f5f5;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10pt;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        font-size: 9.5pt;
    }}
    th {{
        background: #0d47a1;
        color: white;
        padding: 8px 10px;
        text-align: left;
        font-weight: 700;
    }}
    td {{
        padding: 6px 10px;
        border-bottom: 1px solid #e0e0e0;
    }}
    tr:nth-child(even) {{
        background: #f5f7fa;
    }}
    blockquote {{
        border-left: 4px solid #0d47a1;
        margin: 10px 0;
        padding: 8px 15px;
        background: #e3f2fd;
        font-size: 9.5pt;
    }}
    @page {{
        size: A4;
        margin: 25mm 20mm;
        @bottom-center {{
            content: counter(page);
            font-size: 9pt;
            color: #757575;
        }}
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    html_path = PAPER_DIR / "manuscript_ko.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path = PAPER_DIR / "manuscript_ko.pdf"
    HTML(string=html, base_url=str(PAPER_DIR)).write_pdf(str(pdf_path))

    print(f"Korean PDF generated: {pdf_path}")
    print(f"Size: {pdf_path.stat().st_size // 1024}KB")

if __name__ == "__main__":
    main()
