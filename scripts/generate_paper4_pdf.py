#!/usr/bin/env python3
"""Generate Paper 4 (MDD GWAS) submission PDFs (EN + KO) with embedded figures."""
import sys
import markdown
from weasyprint import HTML
from pathlib import Path

PAPER_DIR = Path("/home/sysoft/ai-drug-target/paper4")
FIG_DIR = PAPER_DIR / "figures"

# Fig N -> file + caption (EN/KO)
FIGURES = [
    ("fig1_manhattan.png",        "Figure 1. Manhattan plot of PGC MDD2025 (EUR no-23andMe).",
                                   "그림 1. PGC MDD2025(EUR no-23andMe) 맨해튼 플롯."),
    ("fig2_qq.png",               "Figure 2. QQ plot (λGC = 1.85).",
                                   "그림 2. QQ 플롯 (λGC = 1.85)."),
    ("fig3_magma_top.png",        "Figure 3. Top MAGMA gene-level associations (DCC strongest).",
                                   "그림 3. MAGMA 유전자 수준 상위 연관 (DCC 최강)."),
    ("fig4_twas_forest.png",      "Figure 4. Brain-tissue TWAS Z-score forest (discovery), colored by colocalization status. DRD2 is the top TWAS signal but is coloc-rejected (grey); SLC12A5 is colocalized (green).",
                                   "그림 4. 뇌 조직 TWAS Z-score 포레스트(발견 단계), colocalization 상태로 색칠. DRD2는 최강 TWAS 신호이나 coloc 기각(회색); SLC12A5는 공존(녹색)."),
    ("fig5_tissue_dot.png",       "Figure 5. Cross-tissue TWAS effect heatmap. Cell intensity reflects TWAS-level significance only; the brightest cell (DRD2) is NOT colocalized (see Fig. 4 / Table 2).",
                                   "그림 5. 조직 간 TWAS 효과 히트맵. 셀 강도는 TWAS 수준 유의성만 반영하며, 가장 밝은 셀(DRD2)은 colocalize되지 않음(그림 4 / 표 2 참조)."),
    ("fig6_directional_drugs.png","Figure 6. Direction-aware drug repurposing: (A) raw approved candidates (pre-confirmation); (B) colocalization-stratified prioritization, with the DRD2 dopaminergic cluster demoted to the exploratory tier.",
                                   "그림 6. 방향성 인지 약물 재배치: (A) 원시 승인 후보(확증 이전); (B) colocalization 층화 우선순위 — DRD2 도파민 클러스터는 탐색적 등급으로 강등."),
    ("figS1_drug_network.png",    "Supplementary Figure S1. DGIdb gene-drug interactions for TWAS-prioritized genes (gene color = colocalization status; DRD2 = coloc-rejected, its drugs exploratory).",
                                   "보충 그림 S1. TWAS 우선순위 유전자의 DGIdb 유전자-약물 상호작용(유전자 색 = colocalization 상태; DRD2 = coloc 기각, 해당 약물은 탐색적)."),
]

CSS = """
@page { size: A4; margin: 22mm 18mm; @bottom-center { content: counter(page); font-size: 9pt; color: #777; } }
body { font-family: 'Noto Sans CJK KR','Noto Sans',sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1a1a1a; }
h1 { font-size: 16pt; color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 6px; }
h2 { font-size: 12.5pt; color: #1565c0; margin-top: 18px; border-bottom: 1px solid #ddd; }
h3 { font-size: 11pt; color: #283593; }
p { text-align: justify; margin: 6px 0; }
strong { color: #b71c1c; }
em { color: #00695c; font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt; }
th { background: #0d47a1; color: #fff; padding: 5px 7px; text-align: left; }
td { padding: 4px 7px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) { background: #f5f7fa; }
figure { margin: 14px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid #eee; }
figcaption { font-size: 9pt; color: #444; margin-top: 4px; font-style: italic; }
"""

def build(lang):
    md_path = PAPER_DIR / f"paper_draft_v19_{lang}.md"
    md_text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    cap_idx = 2 if lang == "ko" else 1
    figs_title = "그림" if lang == "ko" else "Figures"
    fig_html = [f"<h2>{figs_title}</h2>"]
    for fname, cap_en, cap_ko in FIGURES:
        fpath = FIG_DIR / fname
        if not fpath.exists():
            continue
        cap = cap_ko if lang == "ko" else cap_en
        fig_html.append(
            f'<figure><img src="file://{fpath}"/>'
            f'<figcaption>{cap}</figcaption></figure>'
        )
    full = (f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="UTF-8">'
            f"<style>{CSS}</style></head><body>{body}{''.join(fig_html)}</body></html>")

    out = PAPER_DIR / f"paper4_submission_v19_{lang}.pdf"
    HTML(string=full, base_url=str(PAPER_DIR)).write_pdf(str(out))
    print(f"{lang.upper()} PDF: {out} ({out.stat().st_size//1024} KB)")

if __name__ == "__main__":
    for lang in ("en", "ko"):
        build(lang)
