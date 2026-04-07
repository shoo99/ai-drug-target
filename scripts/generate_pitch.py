#!/usr/bin/env python3
"""Generate pitch deck PDF + cold email templates for biotech sales."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from config.settings import REPORTS_DIR


# Colors
PRIMARY = HexColor("#0d47a1")
ACCENT = HexColor("#00897b")
WHITE = HexColor("#ffffff")
DARK = HexColor("#212121")
GRAY = HexColor("#616161")
LIGHT = HexColor("#e3f2fd")
BG = HexColor("#fafafa")


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("SlideTitle", fontSize=24, textColor=PRIMARY,
                               spaceAfter=8*mm, alignment=TA_CENTER, leading=30))
    styles.add(ParagraphStyle("SlideSubtitle", fontSize=14, textColor=GRAY,
                               spaceAfter=6*mm, alignment=TA_CENTER))
    styles.add(ParagraphStyle("BulletLarge", fontSize=13, textColor=DARK,
                               spaceAfter=4*mm, leading=18, leftIndent=10*mm))
    styles.add(ParagraphStyle("BulletSmall", fontSize=11, textColor=GRAY,
                               spaceAfter=3*mm, leading=15, leftIndent=15*mm))
    styles.add(ParagraphStyle("BigNumber", fontSize=36, textColor=PRIMARY,
                               alignment=TA_CENTER, spaceAfter=2*mm))
    styles.add(ParagraphStyle("CenterText", fontSize=12, textColor=DARK,
                               alignment=TA_CENTER, spaceAfter=4*mm))
    styles.add(ParagraphStyle("Footer", fontSize=8, textColor=GRAY,
                               alignment=TA_CENTER))
    return styles


def generate_pitch_deck():
    filepath = REPORTS_DIR / "pitch_deck.pdf"
    doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = get_styles()
    story = []

    # --- Slide 1: Title ---
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("AI Drug Target<br/>Discovery Platform", styles["SlideTitle"]))
    story.append(Paragraph("AI 기반 신약 타겟 발굴 플랫폼", styles["SlideSubtitle"]))
    story.append(Spacer(1, 20*mm))
    story.append(Paragraph(
        "Dual Track: Antimicrobial Resistance + Chronic Pruritus",
        styles["CenterText"]))
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(f"Confidential | {datetime.now().strftime('%B %Y')}", styles["Footer"]))
    story.append(PageBreak())

    # --- Slide 2: Problem ---
    story.append(Paragraph("The Problem", styles["SlideTitle"]))
    story.append(Paragraph(
        "신약 개발의 90%가 실패합니다. 가장 큰 원인은 <b>잘못된 타겟 선택</b>입니다.",
        styles["BulletLarge"]))
    story.append(Spacer(1, 5*mm))
    problems = [
        "• 신약 개발 평균 비용: <b>$2.6B (약 3.5조원)</b>",
        "• 평균 개발 기간: <b>10~15년</b>",
        "• 임상 1상→승인 성공률: <b>7.9%</b>",
        "• 실패 원인 1위: <b>타겟 선택 오류 (40%+)</b>",
        "",
        "• AMR: WHO 긴급 위협, 신규 항생제 파이프라인 고갈",
        "• Pruritus: 전 세계 수억명, 치료 옵션 극히 제한적",
    ]
    for p in problems:
        if p:
            story.append(Paragraph(p, styles["BulletLarge"]))
        else:
            story.append(Spacer(1, 3*mm))
    story.append(PageBreak())

    # --- Slide 3: Solution ---
    story.append(Paragraph("Our Solution", styles["SlideTitle"]))
    story.append(Paragraph(
        "AI가 수백만 데이터 포인트를 분석하여<br/>최적의 신약 타겟을 발굴합니다",
        styles["SlideSubtitle"]))
    story.append(Spacer(1, 5*mm))

    solution_data = [
        ["Component", "Technology", "Output"],
        ["Knowledge Graph", "Neo4j (3,200+ nodes)", "Gene-Disease-Drug relationships"],
        ["Literature Mining", "NLP on 1,000+ papers", "Gene mentions & trends"],
        ["AI Prediction", "Node2Vec + GBM\n(AUC: 0.973)", "Novel target candidates"],
        ["Scoring Engine", "6-dimension scoring", "Ranked target list"],
        ["Structure Analysis", "AlphaFold integration", "Druggability assessment"],
    ]
    table = Table(solution_data, colWidths=[40*mm, 45*mm, 55*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(PageBreak())

    # --- Slide 4: Validation ---
    story.append(Paragraph("Validation Results", styles["SlideTitle"]))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("<b>AMR Track</b>", styles["BulletLarge"]))
    story.append(Paragraph("• AI 모델 AUC-ROC: <b>0.973</b>", styles["BulletSmall"]))
    story.append(Paragraph("• 상위 20개 타겟 중 <b>18개 (90%)</b>가 최신 논문에서 검증됨", styles["BulletSmall"]))
    story.append(Paragraph("• GYRA, LPXC, PARE 등 검증된 핵심 타겟 발굴", styles["BulletSmall"]))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("<b>Pruritus Track</b>", styles["BulletLarge"]))
    story.append(Paragraph("• 상위 20개 중 <b>17개 (85%)</b>가 아직 미탐색 영역", styles["BulletSmall"]))
    story.append(Paragraph("• IL7R, PRKCQ, MMP12 등 새로운 타겟 후보 발굴", styles["BulletSmall"]))
    story.append(Paragraph("• <b>→ 특허 출원 + 공동연구 기회</b>", styles["BulletSmall"]))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph("<b>AlphaFold 구조 분석</b>", styles["BulletLarge"]))
    story.append(Paragraph("• PRKCQ (kinase): 드러거빌리티 0.65 — Small molecule 가능", styles["BulletSmall"]))
    story.append(Paragraph("• MMP12 (protease): 드러거빌리티 0.60 — Small molecule 가능", styles["BulletSmall"]))
    story.append(Paragraph("• IL7R/IL6R (receptor): 항체 또는 소분자 접근 가능", styles["BulletSmall"]))
    story.append(PageBreak())

    # --- Slide 5: Business Model ---
    story.append(Paragraph("Business Model", styles["SlideTitle"]))
    story.append(Spacer(1, 5*mm))

    biz_data = [
        ["Service", "Price", "Delivery"],
        ["Pilot Analysis Report\n(1 disease, top 10 targets)", "Free ~ ₩2M", "2 weeks"],
        ["Full Target Discovery\n(complete pipeline + report)", "₩10M ~ 50M", "4-6 weeks"],
        ["Platform Subscription\n(dashboard + ongoing updates)", "₩1M ~ 3M/month", "Continuous"],
        ["Co-development\n(joint IP, milestone payments)", "Negotiable", "6-12 months"],
    ]
    table = Table(biz_data, colWidths=[50*mm, 35*mm, 30*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(PageBreak())

    # --- Slide 6: Competitive Advantage ---
    story.append(Paragraph("Why Us", styles["SlideTitle"]))
    story.append(Spacer(1, 5*mm))
    advantages = [
        "• <b>도메인 전문성:</b> 바이오인포매틱스 + 시스템 엔지니어링 결합",
        "• <b>자체 인프라:</b> 프라이빗 서버에서 운영 → 데이터 보안 보장",
        "• <b>빠른 턴어라운드:</b> AI 자동화로 수개월 → 수주 단축",
        "• <b>다중 소스 분석:</b> 유전체 + 문헌 + 구조 + 임상 통합 분석",
        "• <b>듀얼 트랙:</b> 감염병(AMR) + 피부과(Pruritus) 전문성",
        "• <b>검증된 모델:</b> AUC 0.973, 90% 타겟 검증률",
        "",
        "• vs Insilico Medicine: 특화 니치 집중 (vs 범용)",
        "• vs CRO 분석: AI 자동화로 10x 빠르고 저렴",
        "• vs 내부 인포매틱스: 즉시 사용 가능, 채용 불필요",
    ]
    for a in advantages:
        if a:
            story.append(Paragraph(a, styles["BulletLarge"]))
        else:
            story.append(Spacer(1, 3*mm))
    story.append(PageBreak())

    # --- Slide 7: Roadmap ---
    story.append(Paragraph("Roadmap", styles["SlideTitle"]))
    story.append(Spacer(1, 5*mm))
    roadmap = [
        ["Phase", "Timeline", "Milestone"],
        ["Phase 1\nPilot", "Month 1-2", "• 2-3 biotech pilot projects\n• Validate pipeline end-to-end"],
        ["Phase 2\nFirst Revenue", "Month 3-4", "• First paid contracts\n• Paper/preprint submission"],
        ["Phase 3\nScale", "Month 5-8", "• Expand to 3+ disease areas\n• Patent filing"],
        ["Phase 4\nGrowth", "Month 9-12", "• Monthly revenue ₩10M+\n• Series seed investment"],
    ]
    table = Table(roadmap, colWidths=[30*mm, 25*mm, 80*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(PageBreak())

    # --- Slide 8: Contact ---
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("Let's Discover<br/>Your Next Drug Target", styles["SlideTitle"]))
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(
        "무료 파일럿 분석을 제안드립니다.<br/>"
        "관심 질환을 알려주시면, AI 타겟 분석 리포트를 무료로 제공합니다.",
        styles["CenterText"]))
    story.append(Spacer(1, 20*mm))
    story.append(Paragraph("AI Drug Target Discovery Platform", styles["Footer"]))
    story.append(Paragraph(f"Confidential | {datetime.now().strftime('%B %Y')}", styles["Footer"]))

    doc.build(story)
    print(f"📄 Pitch deck: {filepath}")
    return filepath


def generate_cold_email():
    """Generate cold email templates."""
    filepath = REPORTS_DIR / "cold_email_templates.txt"

    templates = """
================================================================================
COLD EMAIL TEMPLATE 1 — AMR 바이오텍 대상
================================================================================

Subject: AI가 발굴한 [회사명] 파이프라인 관련 신규 항생제 타겟 — 무료 리포트 제안

[담당자명]님 안녕하세요,

[회사명]의 항균제 파이프라인에 관심을 갖고 연락드립니다.

저희는 AI 기반 신약 타겟 발굴 플랫폼을 운영하고 있으며,
항생제 내성(AMR) 분야에서 ESKAPE 병원체 대상 신규 타겟을 발굴하고 있습니다.

최근 분석 결과:
- 3,200+ 노드 지식그래프 구축 (ChEMBL, OpenTargets, PubMed 1,000+ 논문)
- AI 모델 AUC-ROC: 0.973
- 기존에 알려지지 않은 신규 타겟 후보 다수 발굴

[회사명]의 관심 병원체/질환에 맞춤화된 AI 타겟 분석 리포트를
무료로 제공해드리고 싶습니다.

10분 정도 시간을 내주실 수 있으신가요?

감사합니다,
[이름]
AI Drug Target Discovery Platform


================================================================================
COLD EMAIL TEMPLATE 2 — Pruritus 제약사 대상
================================================================================

Subject: 만성 가려움증 신규 타겟 — AI 분석 결과 공유 제안

[담당자명]님 안녕하세요,

[회사명]의 피부과/면역 파이프라인에 관해 연락드립니다.

저희 AI 플랫폼이 만성 가려움증(Chronic Pruritus) 분야에서
기존에 탐색되지 않은 신규 약물 타겟을 발굴했습니다.

주요 발견:
- IL7R: 가려움-면역 경로의 새로운 연결고리 (드러거빌리티 0.45)
- PRKCQ: T세포 활성화 kinase — small molecule 접근 가능 (드러거빌리티 0.65)
- MMP12: 피부 염증 관련 protease — 억제제 개발 가능 (드러거빌리티 0.60)

이 타겟들은 최근 문헌에서 아직 pruritus 맥락으로 연구되지 않아
선점 기회가 있습니다.

[회사명]의 파이프라인과 관련하여 맞춤 분석 리포트를
무료로 제공해드리겠습니다.

간단한 미팅이 가능하실까요?

감사합니다,
[이름]
AI Drug Target Discovery Platform


================================================================================
COLD EMAIL TEMPLATE 3 — 대학/연구소 공동연구 제안
================================================================================

Subject: [연구실명] 연구 관련 AI 타겟 분석 협력 제안

[교수님/연구원님] 안녕하세요,

[연구실]에서 진행하시는 [연구주제] 연구에 깊은 관심을 갖고 연락드립니다.

저희는 바이오인포매틱스 기반 AI 신약 타겟 발굴 플랫폼을 운영하고 있으며,
현재 AMR(항생제 내성)과 만성 가려움증 분야에서 연구를 진행 중입니다.

공동연구 제안:
1. 저희 AI 플랫폼으로 [연구주제] 관련 신규 타겟 후보를 발굴
2. 선생님 연구실에서 실험적 검증 진행
3. 공동 논문 발표 + 특허 공동 출원

현재 AI가 발굴한 후보 중 실험 검증이 필요한 타겟이 다수 있으며,
이를 공유드리고 협력 가능성을 논의하고 싶습니다.

짧은 온라인 미팅이 가능하실까요?

감사합니다,
[이름]


================================================================================
고객 리스트 — 우선 접촉 대상
================================================================================

[AMR — 한국 바이오텍]
1. 레고켐바이오사이언스 — ADC + 항균 파이프라인
2. 큐리언트 — 항감염 특화
3. 인트론바이오 — 파지 치료 + 항균
4. 프로젠 — 미생물 치료제
5. 이뮨메드 — 면역항암 (AMR 확장 가능)

[AMR — 정부/공공]
6. 질병관리청 — AMR 대응 연구사업
7. 한국생명공학연구원 — 감염병 연구
8. 한국화학연구원 — 신약 합성

[Pruritus — 제약/화장품]
9. 아모레퍼시픽 R&D — 피부과학 연구
10. LG화학 생명과학 — 면역/피부
11. 대웅제약 — 피부과 파이프라인
12. 한미약품 — 바이오 신약

[해외]
13. Galderma — 피부과 글로벌 리더
14. Arcutis Biotherapeutics — 피부 염증 특화
15. Entasis Therapeutics — AMR 특화 바이오텍

[대학/연구소]
16. 서울대 약학대학 — 항균제 연구
17. KAIST 바이오공학과 — AI 신약
18. 연세대 의대 피부과학교실 — 가려움 연구
"""

    with open(filepath, "w") as f:
        f.write(templates)

    print(f"📧 Cold email templates: {filepath}")
    return filepath


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("📊 Generating pitch deck + sales materials...\n")
    generate_pitch_deck()
    generate_cold_email()
    print("\n✅ All sales materials generated!")


if __name__ == "__main__":
    main()
