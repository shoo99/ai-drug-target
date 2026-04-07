"""CRM / Sales Pipeline Tracking — Streamlit page"""
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

CRM_FILE = Path(__file__).parent.parent.parent / "data" / "common" / "crm_pipeline.json"

DEFAULT_PIPELINE = [
    {"id": 1, "company": "레고켐바이오사이언스", "track": "AMR", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "ADC 기반 항균", "updated": ""},
    {"id": 2, "company": "큐리언트", "track": "AMR", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "항감염 전문", "updated": ""},
    {"id": 3, "company": "인트론바이오", "track": "AMR", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "파지 치료제", "updated": ""},
    {"id": 4, "company": "프로젠", "track": "AMR", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "마이크로바이옴", "updated": ""},
    {"id": 5, "company": "질병관리청", "track": "AMR", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "정부 R&D", "updated": ""},
    {"id": 6, "company": "한국생명공학연구원", "track": "AMR", "stage": "Prospect", "contact": "", "next_action": "공동연구 제안", "notes": "감염병 연구", "updated": ""},
    {"id": 7, "company": "아모레퍼시픽", "track": "Pruritus", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "피부과학", "updated": ""},
    {"id": 8, "company": "대웅제약", "track": "Pruritus", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "피부과 파이프라인", "updated": ""},
    {"id": 9, "company": "한미약품", "track": "Pruritus", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "바이오 항체", "updated": ""},
    {"id": 10, "company": "LG화학 생명과학", "track": "Pruritus", "stage": "Prospect", "contact": "", "next_action": "콜드 이메일 발송", "notes": "면역/피부", "updated": ""},
    {"id": 11, "company": "Galderma", "track": "Pruritus", "stage": "Prospect", "contact": "", "next_action": "영문 이메일 발송", "notes": "글로벌 피부과", "updated": ""},
    {"id": 12, "company": "Arcutis Bio", "track": "Pruritus", "stage": "Prospect", "contact": "", "next_action": "영문 이메일 발송", "notes": "피부 염증 특화", "updated": ""},
    {"id": 13, "company": "서울대 약학대학", "track": "AMR", "stage": "Prospect", "contact": "", "next_action": "공동연구 제안", "notes": "항균제 합성", "updated": ""},
    {"id": 14, "company": "연세대 피부과학교실", "track": "Pruritus", "stage": "Prospect", "contact": "", "next_action": "공동연구 제안", "notes": "가려움 기전", "updated": ""},
]

STAGES = ["Prospect", "Email Sent", "Responded", "Meeting Set", "Pilot Active", "Pilot Done", "Negotiating", "Closed Won", "Closed Lost"]
STAGE_COLORS = {
    "Prospect": "🔵", "Email Sent": "📧", "Responded": "💬",
    "Meeting Set": "📅", "Pilot Active": "🔬", "Pilot Done": "📊",
    "Negotiating": "🤝", "Closed Won": "✅", "Closed Lost": "❌",
}


def load_pipeline():
    if CRM_FILE.exists():
        with open(CRM_FILE) as f:
            return json.load(f)
    return DEFAULT_PIPELINE


def save_pipeline(data):
    CRM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CRM_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def render_crm():
    st.title("📋 Sales Pipeline / CRM")
    pipeline = load_pipeline()
    df = pd.DataFrame(pipeline)

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Leads", len(df))
    c2.metric("Emails Sent", len(df[df["stage"] != "Prospect"]))
    c3.metric("Active Pilots", len(df[df["stage"] == "Pilot Active"]))
    c4.metric("Won", len(df[df["stage"] == "Closed Won"]))
    c5.metric("Pipeline Value", "₩0")

    st.markdown("---")

    # Pipeline view
    st.subheader("Pipeline Board")
    cols = st.columns(len(STAGES))
    for i, stage in enumerate(STAGES):
        with cols[i]:
            stage_items = [p for p in pipeline if p["stage"] == stage]
            st.markdown(f"**{STAGE_COLORS.get(stage, '')} {stage}**")
            st.markdown(f"({len(stage_items)})")
            for item in stage_items:
                st.markdown(
                    f"<div style='background:#f5f5f5;padding:4px 8px;margin:2px 0;"
                    f"border-radius:4px;font-size:11px;'>"
                    f"<b>{item['company']}</b><br/>"
                    f"<span style='color:#666;'>{item['track']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Edit table
    st.subheader("Lead Management")
    edited = st.data_editor(
        df,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "company": st.column_config.TextColumn("Company", width="medium"),
            "track": st.column_config.SelectboxColumn("Track", options=["AMR", "Pruritus"]),
            "stage": st.column_config.SelectboxColumn("Stage", options=STAGES),
            "contact": st.column_config.TextColumn("Contact"),
            "next_action": st.column_config.TextColumn("Next Action"),
            "notes": st.column_config.TextColumn("Notes"),
            "updated": st.column_config.TextColumn("Last Updated", width="small"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
    )

    if st.button("💾 Save Changes"):
        records = edited.to_dict("records")
        for r in records:
            r["updated"] = datetime.now().strftime("%Y-%m-%d")
        save_pipeline(records)
        st.success("Pipeline saved!")
        st.rerun()

    # Quick Actions
    st.markdown("---")
    st.subheader("Quick Actions")
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button("📧 Mark All Prospects as 'Email Sent'"):
            for p in pipeline:
                if p["stage"] == "Prospect":
                    p["stage"] = "Email Sent"
                    p["updated"] = datetime.now().strftime("%Y-%m-%d")
            save_pipeline(pipeline)
            st.success("Updated!")
            st.rerun()
    with qc2:
        st.download_button(
            "📥 Export Pipeline CSV",
            df.to_csv(index=False),
            "pipeline_export.csv",
            "text/csv"
        )
