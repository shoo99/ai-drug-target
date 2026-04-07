#!/usr/bin/env python3
"""Generate customized cold emails for each target customer."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import REPORTS_DIR

CUSTOMERS = [
    # AMR - Korean Biotech
    {
        "company": "레고켐바이오사이언스",
        "track": "amr",
        "contact_title": "연구소장",
        "focus": "ADC (항체약물접합체) 기반 항균 파이프라인",
        "hook": "ESKAPE 병원체 대상 필수유전자 분석에서 기존 ADC 타겟 외 신규 표면단백질 후보를 발굴했습니다. 귀사의 ADC 기술과 결합 시 새로운 항균 ADC 개발이 가능합니다.",
        "relevant_targets": "LPS 생합성 경로 타겟, 외막 단백질",
    },
    {
        "company": "큐리언트",
        "track": "amr",
        "contact_title": "CSO",
        "focus": "항감염 신약 전문 (결핵, 그람음성균)",
        "hook": "그람음성균 필수유전자 분석에서 기존 항생제 타겟(GyrA, ParE)을 넘어서는 신규 타겟 후보를 확인했습니다. 귀사의 항감염 전문성과 시너지가 기대됩니다.",
        "relevant_targets": "LPXC, 필수 대사효소, 세포벽 합성",
    },
    {
        "company": "인트론바이오",
        "track": "amr",
        "contact_title": "CTO",
        "focus": "박테리오파지 치료제 + 항균 엔도라이신",
        "hook": "AI 분석으로 파지 치료와 병용 가능한 세균 타겟을 발굴했습니다. 파지 저항성 발생 시 대안 경로로 활용 가능한 필수유전자 목록이 있습니다.",
        "relevant_targets": "세균 필수유전자 중 파지 수용체 관련",
    },
    {
        "company": "프로젠",
        "track": "amr",
        "contact_title": "R&D 디렉터",
        "focus": "미생물 치료제, 마이크로바이옴",
        "hook": "AMR 관련 미생물 상호작용 네트워크 분석에서 내성균 억제 가능한 타겟 경로를 확인했습니다.",
        "relevant_targets": "Quorum sensing, 생물막 형성 관련 타겟",
    },
    # AMR - Government
    {
        "company": "질병관리청 감염병연구센터",
        "track": "amr",
        "contact_title": "연구관",
        "focus": "국가 AMR 대응 R&D 사업",
        "hook": "ESKAPE 병원체 전체를 커버하는 AI 타겟 발굴 파이프라인을 구축했습니다. 국가 AMR 대응 전략에 기여할 수 있는 데이터와 분석 역량을 보유하고 있습니다.",
        "relevant_targets": "ESKAPE 6종 전체 필수유전자 매핑",
    },
    {
        "company": "한국생명공학연구원",
        "track": "amr",
        "contact_title": "감염병연구팀장",
        "focus": "감염병 기초/응용 연구",
        "hook": "공동연구 제안: AI가 발굴한 신규 항균 타겟의 실험적 검증을 함께 진행하고, 공동 논문 발표 + 특허 출원이 가능합니다.",
        "relevant_targets": "실험 검증이 필요한 신규 필수유전자 후보",
    },
    # Pruritus - Pharma
    {
        "company": "아모레퍼시픽 기술연구원",
        "track": "pruritus",
        "contact_title": "피부과학연구팀장",
        "focus": "피부 과학 기초연구 + 기능성 화장품",
        "hook": "만성 가려움증 분야에서 기존에 탐색되지 않은 신규 타겟 17개를 AI로 발굴했습니다. 특히 피부 면역 경로의 PRKCQ, MMP12는 기능성 화장품 소재 개발에도 활용 가능합니다.",
        "relevant_targets": "PRKCQ (kinase), MMP12 (protease), IL7R",
    },
    {
        "company": "대웅제약",
        "track": "pruritus",
        "contact_title": "신약연구센터장",
        "focus": "피부과 파이프라인 (아토피, 건선)",
        "hook": "AI가 발굴한 pruritus 타겟 중 PRKCQ는 kinase로 소분자 억제제 개발이 가능하며 (드러거빌리티 0.65), 아직 pruritus 맥락에서 연구되지 않은 블루오션입니다.",
        "relevant_targets": "PRKCQ, IL7R, DOK2",
    },
    {
        "company": "한미약품",
        "track": "pruritus",
        "contact_title": "바이오신약연구소장",
        "focus": "바이오 신약 (항체, 이중항체)",
        "hook": "IL7R, IL6R 등 receptor 타겟은 귀사의 항체 기술 플랫폼과 직접 연결됩니다. pruritus 적응증으로의 확장 가능성을 AI 데이터로 뒷받침할 수 있습니다.",
        "relevant_targets": "IL7R, IL6R, CSF2RB, IL18R1",
    },
    {
        "company": "LG화학 생명과학",
        "track": "pruritus",
        "contact_title": "면역/피부 R&D 리드",
        "focus": "면역질환, 피부질환 파이프라인",
        "hook": "만성 가려움증에서 면역 경로의 신규 타겟을 발굴했습니다. 귀사의 면역질환 전문성에 pruritus 적응증을 추가할 수 있는 데이터를 보유하고 있습니다.",
        "relevant_targets": "IL7R, PRKCQ, NLRP10, BACH2",
    },
    # International
    {
        "company": "Galderma",
        "track": "pruritus",
        "contact_title": "Head of External Innovation",
        "focus": "Global dermatology leader",
        "hook": "Our AI platform identified 17 novel pruritus targets not yet explored in recent literature. PRKCQ (druggability 0.65) and MMP12 (0.60) are ready for small molecule screening.",
        "relevant_targets": "PRKCQ, MMP12, IL7R, DOK2",
        "lang": "en",
    },
    {
        "company": "Arcutis Biotherapeutics",
        "track": "pruritus",
        "contact_title": "VP of Research",
        "focus": "Dermatology-focused biotech",
        "hook": "We identified novel itch targets including PRKCQ (kinase, druggability 0.65) and MMP12 (protease, 0.60) — both amenable to small molecule approaches and uncharted in pruritus context.",
        "relevant_targets": "PRKCQ, MMP12, IL7R",
        "lang": "en",
    },
    # Universities
    {
        "company": "서울대 약학대학 이교수 연구실",
        "track": "amr",
        "contact_title": "교수님",
        "focus": "항균제 합성 및 작용기전 연구",
        "hook": "공동연구 제안: AI가 발굴한 ESKAPE 필수유전자 타겟의 in vitro 검증을 함께 진행하고, 공동 논문(Nature Microbiology급) 투고를 목표로 합니다.",
        "relevant_targets": "신규 필수유전자 후보 리스트",
    },
    {
        "company": "연세대 의대 피부과학교실",
        "track": "pruritus",
        "contact_title": "교수님",
        "focus": "가려움 기전 연구, 아토피 임상",
        "hook": "AI가 발굴한 가려움 관련 신규 타겟(IL7R, DOK2, TESPA1)의 피부 조직 발현 검증을 함께 진행하고 싶습니다. 공동 논문 + 특허 공동 출원이 가능합니다.",
        "relevant_targets": "IL7R, DOK2, TESPA1, PRKCQ",
    },
]


def generate_emails():
    output = []
    for i, customer in enumerate(CUSTOMERS, 1):
        lang = customer.get("lang", "ko")
        company = customer["company"]
        track = customer["track"]
        title = customer["contact_title"]
        focus = customer["focus"]
        hook = customer["hook"]
        targets = customer["relevant_targets"]

        if lang == "en":
            email = f"""
{'='*70}
[{i:02d}] {company} ({track.upper()})
{'='*70}

To: {title}, {company}
Subject: AI-Discovered Novel {track.upper()} Drug Targets — Free Pilot Offer

Dear {title},

I'm reaching out regarding {company}'s work in {focus}.

We've built an AI-powered drug target discovery platform specializing in
{'antimicrobial resistance (AMR)' if track == 'amr' else 'chronic pruritus'}.

{hook}

Our platform:
• Knowledge graph: 3,200+ nodes from ChEMBL, OpenTargets, PubMed
• AI model: AUC-ROC 0.973
• Top targets: {targets}

We'd like to offer a FREE customized pilot analysis for {company}.
Just tell us your disease/pathogen of interest, and we'll deliver
an AI-generated target report within 2 weeks.

Would you have 15 minutes for a brief call?

Best regards,
[Name]
AI Drug Target Discovery Platform
"""
        else:
            email = f"""
{'='*70}
[{i:02d}] {company} ({track.upper()})
{'='*70}

수신: {company} {title}
제목: AI 신약 타겟 발굴 — {company} 맞춤 무료 파일럿 제안

{title}님 안녕하세요,

{company}의 {focus}에 깊은 관심을 갖고 연락드립니다.

저희는 AI 기반 신약 타겟 발굴 플랫폼을 운영하고 있으며,
{'항생제 내성(AMR)' if track == 'amr' else '만성 가려움증(Chronic Pruritus)'} 분야를 전문으로 합니다.

{hook}

플랫폼 현황:
• 지식그래프: 3,200+ 노드 (ChEMBL, OpenTargets, PubMed 1,000+ 논문)
• AI 예측 모델: AUC-ROC 0.973
• 관련 타겟: {targets}

{company}의 관심 분야에 맞춤화된 AI 타겟 분석 리포트를
무료로 제공해드리고 싶습니다.

15분 정도 짧은 미팅이 가능하실까요?

감사합니다,
[이름]
AI Drug Target Discovery Platform
"""
        output.append(email)

    filepath = REPORTS_DIR / "customized_outreach_emails.txt"
    with open(filepath, "w") as f:
        f.write("\n".join(output))
    print(f"📧 Generated {len(output)} customized emails: {filepath}")
    return filepath


if __name__ == "__main__":
    generate_emails()
