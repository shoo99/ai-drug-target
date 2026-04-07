#!/usr/bin/env python3
"""Generate professional pilot project proposal PDF."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER
from config.settings import REPORTS_DIR

P = HexColor("#0d47a1")
A = HexColor("#00897b")
W = HexColor("#ffffff")
D = HexColor("#212121")
G = HexColor("#616161")
L = HexColor("#e3f2fd")


def build():
    fp = REPORTS_DIR / "pilot_proposal.pdf"
    doc = SimpleDocTemplate(str(fp), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("T", fontSize=22, textColor=P, spaceAfter=6*mm, alignment=TA_CENTER))
    ss.add(ParagraphStyle("ST", fontSize=12, textColor=G, spaceAfter=4*mm, alignment=TA_CENTER))
    ss.add(ParagraphStyle("H", fontSize=15, textColor=P, spaceBefore=8*mm, spaceAfter=4*mm))
    ss.add(ParagraphStyle("B", fontSize=10.5, textColor=D, leading=15, spaceAfter=3*mm))
    ss.add(ParagraphStyle("Sm", fontSize=8, textColor=G, alignment=TA_CENTER))

    s = []

    # Title
    s.append(Spacer(1, 20*mm))
    s.append(Paragraph("AI Drug Target Discovery<br/>Pilot Project Proposal", ss["T"]))
    s.append(Paragraph("무료 파일럿 프로젝트 제안서", ss["ST"]))
    s.append(Spacer(1, 5*mm))
    s.append(HRFlowable(width="100%", thickness=2, color=P))
    s.append(Spacer(1, 5*mm))
    s.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')} | Confidential", ss["ST"]))
    s.append(Spacer(1, 10*mm))

    # Executive Summary
    s.append(Paragraph("1. Executive Summary", ss["H"]))
    s.append(Paragraph(
        "본 제안서는 AI 기반 신약 타겟 발굴 파일럿 프로젝트에 대한 내용입니다. "
        "귀사가 관심 있는 질환/병원체에 대해, 저희 AI 플랫폼이 신규 약물 타겟 후보를 "
        "발굴하고 상세 분석 리포트를 제공합니다. <b>파일럿은 무료</b>로 진행되며, "
        "결과에 만족하실 경우 정식 프로젝트로 전환합니다.", ss["B"]))

    # What We Deliver
    s.append(Paragraph("2. Deliverables (제공물)", ss["H"]))
    dd = [
        ["#", "Deliverable", "Detail"],
        ["1", "타겟 후보 리스트\n(Top 20~50)", "AI 스코어링 기반 랭킹\n6차원 분석 (유전적 근거, 드러거빌리티, 경쟁 등)"],
        ["2", "개별 타겟 분석 리포트\n(Top 5 상세)", "유전자/단백질 정보, 질환 연관 근거,\n구조 분석(AlphaFold), 경쟁 현황, 추천 사항"],
        ["3", "지식그래프 시각화", "타겟-질환-약물 네트워크 인터랙티브 뷰\n(웹 대시보드 접근권 제공)"],
        ["4", "문헌 트렌드 분석", "최근 5년간 관련 연구 동향\n트렌딩 유전자, 핵심 논문 리스트"],
        ["5", "종합 요약 리포트\n(PDF)", "전체 분석 결과 + 전략적 추천사항\n특허/IP 기회 분석 포함"],
    ]
    t = Table(dd, colWidths=[10*mm, 40*mm, 90*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), P), ("TEXTCOLOR", (0, 0), (-1, 0), W),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.5, G),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [W, L]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    s.append(t)

    # Timeline
    s.append(Paragraph("3. Timeline (일정)", ss["H"]))
    td = [
        ["Week", "Activity", "Output"],
        ["Week 1", "킥오프 미팅 + 요구사항 정의\n데이터 수집 시작", "분석 범위 확정"],
        ["Week 2", "지식그래프 구축 + AI 분석\nNLP 문헌 마이닝", "초기 타겟 리스트"],
        ["Week 3", "타겟 스코어링 + AlphaFold 분석\n리포트 작성", "상세 분석 리포트"],
        ["Week 4", "최종 리포트 전달 + 리뷰 미팅\n정식 프로젝트 논의", "최종 납품물"],
    ]
    t2 = Table(td, colWidths=[25*mm, 55*mm, 55*mm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), P), ("TEXTCOLOR", (0, 0), (-1, 0), W),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.5, G),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [W, L]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    s.append(t2)

    # What We Need
    s.append(Paragraph("4. What We Need From You (요청 사항)", ss["H"]))
    needs = [
        "• 관심 질환/병원체 정보 (예: 특정 암종, 특정 세균, 가려움 아형 등)",
        "• 기존에 검토한 타겟이 있다면 공유 (선택, 비공개 유지)",
        "• 선호하는 약물 형태 (소분자, 항체, RNA 등) — 선택사항",
        "• 킥오프 미팅 (30분) + 중간점검 (15분) + 최종 리뷰 (30분)",
    ]
    for n in needs:
        s.append(Paragraph(n, ss["B"]))

    # Pricing
    s.append(Paragraph("5. Pricing (가격)", ss["H"]))
    pd_data = [
        ["Service", "Pilot (본 제안)", "Standard", "Premium"],
        ["Price", "무료", "₩10M ~ 30M", "₩30M ~ 50M"],
        ["질환 수", "1개", "1~2개", "3개+"],
        ["타겟 분석", "Top 5 상세", "Top 20 상세", "Top 50 상세 + 분자도킹"],
        ["AlphaFold 분석", "Top 5", "Top 20", "전체 + 결합포켓"],
        ["대시보드 접근", "2주", "3개월", "12개월"],
        ["특허/IP 분석", "기본", "상세", "상세 + 자유실시 분석"],
        ["공동연구 옵션", "—", "가능", "우선 협의"],
    ]
    t3 = Table(pd_data, colWidths=[35*mm, 30*mm, 35*mm, 40*mm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), P), ("TEXTCOLOR", (0, 0), (-1, 0), W),
        ("BACKGROUND", (1, 0), (1, -1), HexColor("#e8f5e9")),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.5, G),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    s.append(t3)

    # Platform Stats
    s.append(Paragraph("6. Platform Credentials", ss["H"]))
    stats = [
        "• 지식그래프: <b>3,200+ 노드</b>, 2,500+ 관계",
        "• AI 예측 모델: <b>AUC-ROC 0.973</b>",
        "• 데이터 소스: ChEMBL, OpenTargets, UniProt, PubMed, AlphaFold, GWAS Catalog",
        "• 논문 분석: <b>1,099편</b> NLP 처리",
        "• AMR 검증률: 상위 20 타겟 중 <b>90%</b> 최신 문헌 확인",
        "• Pruritus: 상위 20 중 <b>85%가 미탐색 영역</b> (특허 기회)",
        "• 프라이빗 서버 운영: <b>데이터 보안 보장</b>",
    ]
    for stat in stats:
        s.append(Paragraph(stat, ss["B"]))

    # Footer
    s.append(Spacer(1, 15*mm))
    s.append(HRFlowable(width="100%", thickness=1, color=G))
    s.append(Paragraph(
        "AI Drug Target Discovery Platform | Confidential | "
        f"{datetime.now().strftime('%Y-%m-%d')}", ss["Sm"]))

    doc.build(s)
    print(f"📄 Pilot proposal: {fp}")
    return fp


if __name__ == "__main__":
    build()
