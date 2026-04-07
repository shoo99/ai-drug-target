"""
Target Analysis Report Generator — Auto-generate PDF reports for drug target candidates
"""
import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import pandas as pd
from config.settings import REPORTS_DIR, AMR_CONFIG, PRURITUS_CONFIG


# Colors
PRIMARY = HexColor("#1a237e")
SECONDARY = HexColor("#283593")
ACCENT = HexColor("#00897b")
LIGHT_BG = HexColor("#e8eaf6")
WHITE = HexColor("#ffffff")
GRAY = HexColor("#757575")
DARK = HexColor("#212121")
RED = HexColor("#c62828")
GREEN = HexColor("#2e7d32")
AMBER = HexColor("#f57f17")


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=22, textColor=PRIMARY, spaceAfter=6*mm,
    ))
    styles.add(ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=12, textColor=GRAY, spaceAfter=4*mm,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=14, textColor=SECONDARY, spaceBefore=6*mm, spaceAfter=3*mm,
        borderWidth=1, borderColor=SECONDARY, borderPadding=2,
    ))
    styles.add(ParagraphStyle(
        "BodyText2", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=DARK,
    ))
    styles.add(ParagraphStyle(
        "SmallGray", parent=styles["Normal"],
        fontSize=8, textColor=GRAY,
    ))
    return styles


class ReportGenerator:
    def __init__(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.styles = get_styles()

    def _score_badge(self, score: float) -> str:
        if score >= 0.7:
            return f'<font color="#2e7d32"><b>★ HIGH ({score:.3f})</b></font>'
        elif score >= 0.5:
            return f'<font color="#f57f17"><b>● MEDIUM ({score:.3f})</b></font>'
        else:
            return f'<font color="#c62828"><b>○ LOW ({score:.3f})</b></font>'

    def generate_target_report(self, target: dict, track: str = "amr") -> Path:
        """Generate a detailed PDF report for a single target candidate."""
        gene_name = target.get("gene_name", "Unknown")
        filename = f"{track}_{gene_name}_report.pdf"
        filepath = REPORTS_DIR / filename

        doc = SimpleDocTemplate(
            str(filepath), pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        story = []

        # Header
        story.append(Paragraph(
            f"Drug Target Analysis Report", self.styles["Title2"]
        ))
        story.append(Paragraph(
            f"Target: <b>{gene_name}</b> | Track: {track.upper()} | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles["Subtitle"]
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
        story.append(Spacer(1, 4*mm))

        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles["SectionHead"]))
        composite = target.get("composite_score", 0)
        tier = target.get("tier", "N/A")
        story.append(Paragraph(
            f"<b>Composite Score:</b> {self._score_badge(composite)}<br/>"
            f"<b>Classification:</b> {tier}<br/>"
            f"<b>Known Target:</b> {'Yes' if target.get('is_known_target') else 'No — Novel Candidate'}<br/>"
            f"<b>Essential Gene:</b> {'Yes' if target.get('is_essential') else 'No/Unknown'}",
            self.styles["BodyText2"]
        ))
        story.append(Spacer(1, 4*mm))

        # Score Breakdown
        story.append(Paragraph("Score Breakdown", self.styles["SectionHead"]))
        score_data = [
            ["Dimension", "Score", "Weight", "Assessment"],
            ["Genetic Evidence", f"{target.get('genetic_evidence', 0):.3f}", "25%",
             self._assess(target.get('genetic_evidence', 0))],
            ["Expression Specificity", f"{target.get('expression_specificity', 0):.3f}", "20%",
             self._assess(target.get('expression_specificity', 0))],
            ["Druggability", f"{target.get('druggability', 0):.3f}", "20%",
             self._assess(target.get('druggability', 0))],
            ["Novelty", f"{target.get('novelty', 0):.3f}", "15%",
             self._assess(target.get('novelty', 0))],
            ["Competition (low=good)", f"{target.get('competition', 0):.3f}", "10%",
             self._assess(target.get('competition', 0))],
            ["Literature Trend", f"{target.get('literature_trend', 0):.3f}", "10%",
             self._assess(target.get('literature_trend', 0))],
        ]
        table = Table(score_data, colWidths=[45*mm, 25*mm, 20*mm, 50*mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (1, 0), (2, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        story.append(Spacer(1, 4*mm))

        # Recommendations
        story.append(Paragraph("Recommendations", self.styles["SectionHead"]))
        recs = self._generate_recommendations(target, track)
        for rec in recs:
            story.append(Paragraph(f"• {rec}", self.styles["BodyText2"]))
        story.append(Spacer(1, 4*mm))

        # Next Steps
        story.append(Paragraph("Suggested Next Steps", self.styles["SectionHead"]))
        steps = self._generate_next_steps(target, track)
        for i, step in enumerate(steps, 1):
            story.append(Paragraph(f"{i}. {step}", self.styles["BodyText2"]))

        # Footer
        story.append(Spacer(1, 10*mm))
        story.append(HRFlowable(width="100%", thickness=1, color=GRAY))
        story.append(Paragraph(
            "AI Drug Target Discovery Platform | Confidential",
            self.styles["SmallGray"]
        ))

        doc.build(story)
        return filepath

    def generate_summary_report(self, track: str = "amr") -> Path:
        """Generate a summary report of all top targets for a track."""
        if track == "amr":
            csv_path = AMR_CONFIG["data_dir"] / "top_targets_scored.csv"
        else:
            csv_path = PRURITUS_CONFIG["data_dir"] / "top_targets_scored.csv"

        if not csv_path.exists():
            print(f"[Report] No scored targets found for {track}")
            return None

        df = pd.read_csv(csv_path)
        filename = f"{track}_summary_report.pdf"
        filepath = REPORTS_DIR / filename

        doc = SimpleDocTemplate(
            str(filepath), pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        story = []

        # Title
        track_name = "Antimicrobial Resistance (AMR)" if track == "amr" else "Chronic Pruritus"
        story.append(Paragraph(
            f"Target Discovery Summary Report", self.styles["Title2"]
        ))
        story.append(Paragraph(
            f"Track: <b>{track_name}</b> | "
            f"Date: {datetime.now().strftime('%Y-%m-%d')} | "
            f"Candidates: {len(df)}",
            self.styles["Subtitle"]
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
        story.append(Spacer(1, 6*mm))

        # Pipeline Summary
        story.append(Paragraph("Analysis Pipeline Summary", self.styles["SectionHead"]))
        story.append(Paragraph(
            "This report summarizes the AI-driven drug target discovery analysis. "
            "Targets were identified using a multi-source approach combining: "
            "knowledge graph analysis (Neo4j), NLP-based literature mining (PubMed), "
            "genetic association data (OpenTargets, GWAS), and graph neural network "
            "link prediction (Node2Vec + GBM classifier, AUC-ROC: 0.948).",
            self.styles["BodyText2"]
        ))
        story.append(Spacer(1, 4*mm))

        # Top Targets Table
        story.append(Paragraph("Top Target Candidates", self.styles["SectionHead"]))
        header = ["Rank", "Gene", "Score", "Tier", "Known", "Novel"]
        table_data = [header]

        for i, (_, row) in enumerate(df.head(20).iterrows(), 1):
            table_data.append([
                str(i),
                str(row.get("gene_name", "")),
                f"{row.get('composite_score', 0):.3f}",
                str(row.get("tier", "")).replace("Tier ", "T"),
                "Y" if row.get("is_known_target") else "N",
                "N" if row.get("is_known_target") else "Y",
            ])

        table = Table(table_data, colWidths=[12*mm, 30*mm, 20*mm, 40*mm, 15*mm, 15*mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)
        story.append(Spacer(1, 6*mm))

        # Key Insights
        story.append(Paragraph("Key Insights", self.styles["SectionHead"]))

        novel_count = 0
        if "is_known_target" in df.columns:
            try:
                novel_count = len(df[~df["is_known_target"].fillna(False).astype(bool)])
            except Exception:
                pass
        elif "is_novel_target" in df.columns:
            try:
                novel_count = len(df[df["is_novel_target"].fillna(False).astype(bool)])
            except Exception:
                pass
        tier1 = len(df[df["tier"].str.contains("Tier 1", na=False)])
        tier2 = len(df[df["tier"].str.contains("Tier 2", na=False)])

        insights = [
            f"Total candidates analyzed: <b>{len(df)}</b>",
            f"Novel (previously unknown) targets: <b>{novel_count}</b>",
            f"Tier 1 (High Priority): <b>{tier1}</b>",
            f"Tier 2 (Promising): <b>{tier2}</b>",
            f"Average composite score: <b>{df['composite_score'].mean():.3f}</b>",
        ]
        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.styles["BodyText2"]))

        # Footer
        story.append(Spacer(1, 10*mm))
        story.append(HRFlowable(width="100%", thickness=1, color=GRAY))
        story.append(Paragraph(
            "AI Drug Target Discovery Platform | Confidential | "
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles["SmallGray"]
        ))

        doc.build(story)
        print(f"[Report] Generated: {filepath}")
        return filepath

    def generate_all_reports(self):
        """Generate summary reports for both tracks + individual top target reports."""
        reports = []

        for track in ["amr", "pruritus"]:
            # Summary
            path = self.generate_summary_report(track)
            if path:
                reports.append(path)

            # Individual top targets
            if track == "amr":
                csv_path = AMR_CONFIG["data_dir"] / "top_targets_scored.csv"
            else:
                csv_path = PRURITUS_CONFIG["data_dir"] / "top_targets_scored.csv"

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.head(5).iterrows():
                    target = row.to_dict()
                    path = self.generate_target_report(target, track)
                    reports.append(path)
                    print(f"[Report] Generated: {path.name}")

        print(f"\n[Report] Total reports generated: {len(reports)}")
        return reports

    def _assess(self, score: float) -> str:
        if score >= 0.7:
            return "Strong"
        elif score >= 0.5:
            return "Moderate"
        elif score >= 0.3:
            return "Weak"
        else:
            return "Insufficient"

    def _generate_recommendations(self, target: dict, track: str) -> list[str]:
        recs = []
        score = target.get("composite_score", 0)
        is_known = target.get("is_known_target", False)
        drug = target.get("druggability", 0)

        if score >= 0.7:
            recs.append("HIGH PRIORITY — Proceed to experimental validation immediately")
        elif score >= 0.5:
            recs.append("PROMISING — Worth deeper investigation, gather more evidence")
        else:
            recs.append("EXPLORATORY — Monitor literature, low priority for wet-lab work")

        if not is_known:
            recs.append("NOVEL TARGET — Potential for IP protection (patent filing recommended)")

        if drug >= 0.6:
            recs.append("Good druggability — Suitable for small molecule or antibody approach")
        elif drug >= 0.3:
            recs.append("Moderate druggability — Consider alternative modalities (PROTACs, RNA)")
        else:
            recs.append("Low druggability — Explore indirect targeting strategies")

        return recs

    def _generate_next_steps(self, target: dict, track: str) -> list[str]:
        gene = target.get("gene_name", "target")
        steps = [
            f"Validate {gene} expression in disease-relevant tissue (qPCR/Western)",
            f"Check AlphaFold structure for {gene} — assess binding pocket availability",
            f"Run molecular docking simulation against known compound libraries",
            f"Design CRISPR knockout/knockdown experiments for functional validation",
            f"File provisional patent if novel mechanism confirmed",
            f"Prepare grant application for experimental validation funding",
        ]
        if track == "amr":
            steps.insert(2, f"Test {gene} essentiality in ESKAPE organisms via CRISPRi")
        return steps
