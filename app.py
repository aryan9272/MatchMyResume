"""
app.py - MATCH MY RESUME: Glassmorphic 3D ATS Analyzer.

Tech Stack: Python, Streamlit, scikit-learn (TF-IDF + Cosine Similarity),
            PyMuPDF, Pandas, Plotly.

Aesthetic:
- 3D Glassmorphism UI with animated floating background geometry.
- Strict 3-4 color palette: Midnight Navy (#0a0e1a), Electric Indigo (#6366f1),
  Cyber Cyan (#06b6d4), and Frosted White (rgba(255, 255, 255, 0.12)).
- Large glowing headline with cinematic 3D opening animation.
- No external APIs or paid services required.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluate import evaluate_candidates, load_job_descriptions, load_resumes
from matcher import MatchResult, match_resume_to_jd
from pdf_parser import extract_text_from_pdf, parse_uploaded_pdf
from skills import TECH_SKILLS, categorize_skills, get_skill_category

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MATCH MY RESUME | 3D ATS Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Glassmorphism & 3D Animated Background CSS
# Limited 3-4 Color Palette: #0a0e1a (Void), #06b6d4 (Cyan), #6366f1 (Indigo), #ffffff (Frost)
# ---------------------------------------------------------------------------
GLASS_THEME_CSS = """
<style>
    /* =======================================================================
       1. GLOBAL 3D BACKGROUND & ANIMATIONS
       ======================================================================= */
    .stApp {
        background-color: #0a0e1a !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.18) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.18) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(99, 102, 241, 0.15) 0px, transparent 60%);
        background-attachment: fixed;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        overflow-x: hidden;
    }

    /* Ambient Floating 3D Orbs in Background */
    @keyframes floatOrb1 {
        0% { transform: translate3d(0, 0, 0) scale(1) rotate(0deg); }
        50% { transform: translate3d(60px, 40px, 80px) scale(1.15) rotate(180deg); }
        100% { transform: translate3d(0, 0, 0) scale(1) rotate(360deg); }
    }
    @keyframes floatOrb2 {
        0% { transform: translate3d(0, 0, 0) scale(1.1) rotate(0deg); }
        50% { transform: translate3d(-50px, -60px, 60px) scale(0.9) rotate(-180deg); }
        100% { transform: translate3d(0, 0, 0) scale(1.1) rotate(-360deg); }
    }

    .bg-orb-1 {
        position: fixed;
        top: 10%;
        left: -5%;
        width: 450px;
        height: 450px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.22) 0%, rgba(6, 182, 212, 0) 70%);
        filter: blur(60px);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: floatOrb1 18s ease-in-out infinite;
    }
    .bg-orb-2 {
        position: fixed;
        bottom: 15%;
        right: -5%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.22) 0%, rgba(99, 102, 241, 0) 70%);
        filter: blur(70px);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: floatOrb2 22s ease-in-out infinite;
    }

    /* =======================================================================
       2. CINEMATIC 3D OPENING ENTRANCE ANIMATION
       ======================================================================= */
    @keyframes cinematicEntrance {
        0% {
            opacity: 0;
            transform: perspective(1200px) rotateX(18deg) translateY(40px) scale(0.92);
            filter: blur(12px);
        }
        70% {
            transform: perspective(1200px) rotateX(-2deg) translateY(-4px) scale(1.01);
            filter: blur(0px);
        }
        100% {
            opacity: 1;
            transform: perspective(1200px) rotateX(0deg) translateY(0) scale(1);
            filter: blur(0px);
        }
    }

    @keyframes glowPulse {
        0%, 100% { text-shadow: 0 0 25px rgba(6, 182, 212, 0.4), 0 0 50px rgba(99, 102, 241, 0.2); }
        50% { text-shadow: 0 0 45px rgba(6, 182, 212, 0.7), 0 0 75px rgba(99, 102, 241, 0.4); }
    }

    .hero-container {
        animation: cinematicEntrance 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transform-style: preserve-3d;
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 36px 32px 30px 32px;
        margin-bottom: 28px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        position: relative;
        z-index: 1;
        overflow: hidden;
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #06b6d4, #6366f1, transparent);
    }

    .large-website-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.05 !important;
        margin: 0 0 8px 0 !important;
        background: linear-gradient(135deg, #ffffff 25%, #22d3ee 65%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glowPulse 4s ease-in-out infinite;
        display: inline-block;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(6, 182, 212, 0.12);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* =======================================================================
       3. FROSTED GLASS CARDS & 3D HOVER TILT
       ======================================================================= */
    .glass-card {
        background: rgba(17, 24, 39, 0.58) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        transform-style: preserve-3d;
    }

    .glass-card:hover {
        transform: translateY(-5px) perspective(1000px) rotateX(2deg) scale(1.01);
        border-color: rgba(6, 182, 212, 0.4) !important;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6), 0 0 25px rgba(6, 182, 212, 0.25);
    }

    .glass-metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    .glass-metric-value {
        font-size: 2.3rem;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.1;
        background: linear-gradient(135deg, #ffffff 40%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .glass-metric-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Badges & Tags */
    .badge-tier-high {
        display: inline-block;
        background: rgba(6, 182, 212, 0.18);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.45);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.25);
    }
    .badge-tier-good {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.45);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-tier-mod {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-tier-low {
        display: inline-block;
        background: rgba(239, 68, 68, 0.16);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
    }

    /* Skill Tags (Strict 3-4 Colors) */
    .skill-tag-matched {
        display: inline-block;
        background: rgba(6, 182, 212, 0.14);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 2px;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.15);
    }
    .skill-tag-missing {
        display: inline-block;
        background: rgba(99, 102, 241, 0.14);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 2px;
    }
    .skill-tag-additional {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 2px;
    }

    /* =======================================================================
       4. SIDEBAR & INPUT ELEMENTS IN GLASS
       ======================================================================= */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 14, 26, 0.85) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        font-family: ui-monospace, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.86rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.3) !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 12px 35px rgba(6, 182, 212, 0.5) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #22d3ee !important;
        border-bottom: 2px solid #06b6d4 !important;
        background: rgba(6, 182, 212, 0.08) !important;
    }
</style>

<!-- Floating Ambient 3D Glow Orbs -->
<div class="bg-orb-1"></div>
<div class="bg-orb-2"></div>
"""
st.markdown(GLASS_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Caching
# ---------------------------------------------------------------------------
@st.cache_data
def get_sample_resumes() -> pd.DataFrame:
    """Load sample resumes from CSV with caching."""
    return load_resumes()


@st.cache_data
def get_sample_jds() -> pd.DataFrame:
    """Load sample job descriptions from CSV with caching."""
    return load_job_descriptions()


# ---------------------------------------------------------------------------
# Plotly Charts Styled in Glassmorphic Theme (Strict 3-4 Colors)
# ---------------------------------------------------------------------------
def create_score_gauge(score_pct: float) -> go.Figure:
    """Generate a clean Plotly gauge chart styled for the glass theme."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            number={"suffix": "%", "font": {"size": 40, "color": "#ffffff", "family": "Inter, sans-serif"}},
            title={"text": "Composite ATS Score", "font": {"size": 14, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": "#06b6d4", "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.15)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [35, 60], "color": "rgba(99, 102, 241, 0.15)"},
                    {"range": [60, 100], "color": "rgba(6, 182, 212, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.75,
                    "value": score_pct,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=25, r=25, t=35, b=20),
    )
    return fig


def create_score_breakdown_chart(tfidf_pct: float, skill_pct: float, composite_pct: float) -> go.Figure:
    """Bar chart comparing sub-scores with composite score in glass theme."""
    categories = ["TF-IDF Cosine", "Skill Coverage", "Composite ATS"]
    values = [tfidf_pct, skill_pct, composite_pct]
    colors = ["#6366f1", "#06b6d4", "#38bdf8"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
                textfont=dict(color="#ffffff", size=12),
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                ),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, 100],
            title="Score (%)",
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8"),
            titlefont=dict(color="#94a3b8"),
            ticksuffix="%",
        ),
        xaxis=dict(tickfont=dict(color="#f8fafc", size=12)),
        height=220,
        margin=dict(l=20, r=20, t=25, b=20),
    )
    return fig


def create_category_breakdown_chart(
    matched_by_cat: Dict[str, List[str]],
    missing_by_cat: Dict[str, List[str]],
) -> Optional[go.Figure]:
    """Horizontal stacked bar chart showing matched vs missing skills per domain."""
    all_categories = sorted(list(set(matched_by_cat.keys()) | set(missing_by_cat.keys())))
    if not all_categories:
        return None

    matched_counts = [len(matched_by_cat.get(cat, [])) for cat in all_categories]
    missing_counts = [len(missing_by_cat.get(cat, [])) for cat in all_categories]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=all_categories,
            x=matched_counts,
            name="Matched Skills",
            orientation="h",
            marker=dict(color="#06b6d4", line=dict(color="rgba(255,255,255,0.2)", width=1)),
        )
    )
    fig.add_trace(
        go.Bar(
            y=all_categories,
            x=missing_counts,
            name="Missing Skills",
            orientation="h",
            marker=dict(color="#6366f1", line=dict(color="rgba(255,255,255,0.2)", width=1)),
        )
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Skill Count", gridcolor="rgba(255, 255, 255, 0.08)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#f8fafc", size=11)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f8fafc"),
        ),
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def create_candidate_comparison_chart(benchmark_df: pd.DataFrame) -> go.Figure:
    """Plotly horizontal bar chart ranking fictional candidate resumes."""
    df_sorted = benchmark_df.sort_values("Composite ATS Score (%)", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_sorted["Candidate"],
            x=df_sorted["Composite ATS Score (%)"],
            name="Composite ATS Score (%)",
            orientation="h",
            text=[f"{v:.1f}%" for v in df_sorted["Composite ATS Score (%)"]],
            textposition="auto",
            textfont=dict(color="#ffffff"),
            marker=dict(color="#06b6d4"),
        )
    )
    fig.add_trace(
        go.Bar(
            y=df_sorted["Candidate"],
            x=df_sorted["Skill Coverage (%)"],
            name="Skill Coverage (%)",
            orientation="h",
            text=[f"{v:.1f}%" for v in df_sorted["Skill Coverage (%)"]],
            textposition="auto",
            textfont=dict(color="#ffffff"),
            marker=dict(color="#6366f1"),
        )
    )
    fig.add_trace(
        go.Bar(
            y=df_sorted["Candidate"],
            x=df_sorted["TF-IDF Similarity (%)"],
            name="TF-IDF Similarity (%)",
            orientation="h",
            text=[f"{v:.1f}%" for v in df_sorted["TF-IDF Similarity (%)"]],
            textposition="auto",
            textfont=dict(color="#ffffff"),
            marker=dict(color="rgba(255, 255, 255, 0.4)"),
        )
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 100],
            title="Percentage (%)",
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8"),
            ticksuffix="%",
        ),
        yaxis=dict(tickfont=dict(color="#f8fafc", size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f8fafc"),
        ),
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar Controls & Navigation (Glassmorphic)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ Navigation")
view_mode = st.sidebar.radio(
    "Select View:",
    options=[
        "🎯 Resume Matcher & Gap Analysis",
        "📊 Candidate Benchmark & Ranking",
        "🔬 Model Transparency & Viva Guide",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Scoring Weights")
st.sidebar.caption("Adjust composite linear weights (normalized to 1.0):")

tfidf_weight = st.sidebar.slider(
    "TF-IDF Cosine Weight (Context)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Measures overall vocabulary and n-gram overlap.",
)
skill_weight = st.sidebar.slider(
    "Skill Coverage Weight (Keywords)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Measures the fraction of required technical skills present in resume.",
)

# Normalize weights
sum_w = tfidf_weight + skill_weight
norm_tfidf_w = round(tfidf_weight / sum_w, 2)
norm_skill_w = round(skill_weight / sum_w, 2)
st.sidebar.caption(f"Formula: `{norm_tfidf_w} × TFIDF + {norm_skill_w} × SkillCoverage`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Quick Candidate Pre-Loader")
resumes_df = get_sample_resumes()
candidate_names = ["(Upload PDF or Paste Custom)"] + resumes_df["name"].tolist()
selected_candidate = st.sidebar.selectbox("Choose Sample Profile:", options=candidate_names, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Glassmorphic 3D Architecture:**
    - Engine: *Unsupervised Vector Space Model*
    - Tokenizer: *Punctuation-Safe Lookaround*
    - Parser: *PyMuPDF Digital Text Layer*
    - Palette: *Midnight Navy / Cyan / Indigo*
    """
)


# ---------------------------------------------------------------------------
# VIEW 1: Resume Matcher & Gap Analysis
# ---------------------------------------------------------------------------
if view_mode == "🎯 Resume Matcher & Gap Analysis":

    # Hero Banner with Large Heading and 3D Opening Animation
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-subtitle">Intelligent ATS Resume Analyzer & NLP Matcher</div>
            <h1 class="large-website-title">MATCH MY RESUME</h1>
            <br>
            <span class="hero-tag">⚡ Vector Space Model • Unsupervised TF-IDF • Punctuation-Safe</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preload Job Descriptions
    jds_df = get_sample_jds()
    role_options = jds_df["role"].tolist()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Candidate Resume")
        resume_source_mode = st.radio(
            "Resume Source:",
            options=["Upload PDF Document", "Paste or Edit Resume Text"],
            horizontal=True,
        )

        extracted_resume_text = ""
        pdf_warning_list: List[str] = []

        if selected_candidate != "(Upload PDF or Paste Custom)":
            row_match = resumes_df[resumes_df["name"] == selected_candidate]
            if not row_match.empty:
                extracted_resume_text = row_match.iloc[0]["text"]
                st.info(f"Loaded profile: **{selected_candidate}**")

        if resume_source_mode == "Upload PDF Document":
            uploaded_pdf = st.file_uploader(
                "Upload Resume (PDF format)",
                type=["pdf"],
                help="PyMuPDF parses digital text layers directly.",
            )
            if uploaded_pdf is not None:
                parse_out = parse_uploaded_pdf(uploaded_pdf)
                if not parse_out.success:
                    st.error(f"❌ PDF Extraction Failed: {parse_out.error}")
                else:
                    extracted_resume_text = parse_out.text
                    pdf_warning_list = parse_out.warnings
                    st.success(
                        f"✅ Extracted {parse_out.word_count} words across {parse_out.page_count} page(s)."
                    )
                    if pdf_warning_list:
                        for w in pdf_warning_list:
                            st.warning(f"⚠️ {w}")

            resume_input = st.text_area(
                "Extracted Resume Content (Editable):",
                value=extracted_resume_text,
                height=260,
                placeholder="Upload a PDF above or select a sample candidate from the sidebar...",
            )
        else:
            resume_input = st.text_area(
                "Paste Resume Text:",
                value=extracted_resume_text,
                height=320,
                placeholder="Paste candidate work history, education, skills, and projects here...",
            )

    with col_right:
        st.subheader("2. Target Job Description")
        selected_role = st.selectbox(
            "Select Benchmark Job Role:",
            options=role_options,
            index=0,
            help="Selecting a role auto-populates standard requirements.",
        )

        default_jd_text = ""
        matched_role_row = jds_df[jds_df["role"] == selected_role]
        if not matched_role_row.empty:
            default_jd_text = matched_role_row.iloc[0]["description"]

        jd_input = st.text_area(
            "Job Description Text (Editable):",
            value=default_jd_text,
            height=320,
            placeholder="Paste employer requirements, responsibilities, and required tech skills...",
        )

    # Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    run_match = st.button("🚀 Analyze Resume & Calculate 3D Match", use_container_width=True)

    if run_match or (resume_input and jd_input):
        if not resume_input.strip():
            st.error("Please provide resume text (upload a PDF or paste text) to run analysis.")
        elif not jd_input.strip():
            st.error("Please provide a job description to run analysis.")
        else:
            with st.spinner("Executing NLP vectorization, cosine similarity, and skill extraction..."):
                match_result: MatchResult = match_resume_to_jd(
                    resume_input,
                    jd_input,
                    tfidf_weight=norm_tfidf_w,
                    skill_weight=norm_skill_w,
                )

            if match_result.error:
                st.error(f"Analysis Error: {match_result.error}")
            else:
                # -----------------------------------------------------------
                # Metrics Row (Glassmorphic 3D Cards)
                # -----------------------------------------------------------
                st.markdown("---")
                st.subheader("📊 Match Performance Summary")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                with mcol1:
                    tier_class = "badge-tier-mod"
                    if match_result.composite_pct >= 80:
                        tier_class = "badge-tier-high"
                    elif match_result.composite_pct >= 60:
                        tier_class = "badge-tier-good"
                    elif match_result.composite_pct < 40:
                        tier_class = "badge-tier-low"

                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <div class="glass-metric-title">Composite ATS Score</div>
                            <div class="glass-metric-value">{match_result.composite_pct}%</div>
                            <div class="glass-metric-sub">
                                <span class="{tier_class}">{match_result.ats_label}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol2:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <div class="glass-metric-title">TF-IDF Similarity</div>
                            <div class="glass-metric-value">{match_result.tfidf_pct}%</div>
                            <div class="glass-metric-sub">N-gram (1,2) Cosine Overlap</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol3:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <div class="glass-metric-title">Skill Coverage</div>
                            <div class="glass-metric-value">{match_result.skill_coverage_pct}%</div>
                            <div class="glass-metric-sub">
                                {len(match_result.matched_skills)} of {len(match_result.jd_skills)} Required Skills
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol4:
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <div class="glass-metric-title">Vocabulary & Speed</div>
                            <div class="glass-metric-value">{match_result.tfidf_vocab_size}</div>
                            <div class="glass-metric-sub">
                                Joint Vocab Size | {match_result.processing_time_ms} ms
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # -----------------------------------------------------------
                # Visual Analytics Row (Glass Charts)
                # -----------------------------------------------------------
                chart_col1, chart_col2 = st.columns([1, 1])

                with chart_col1:
                    st.markdown("**ATS Score Gauge**")
                    st.plotly_chart(create_score_gauge(match_result.composite_pct), use_container_width=True)

                with chart_col2:
                    st.markdown("**Metric Contribution Breakdown**")
                    st.plotly_chart(
                        create_score_breakdown_chart(
                            match_result.tfidf_pct,
                            match_result.skill_coverage_pct,
                            match_result.composite_pct,
                        ),
                        use_container_width=True,
                    )

                # -----------------------------------------------------------
                # Detailed Analysis Tabs
                # -----------------------------------------------------------
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "🏷️ Skill Gap Breakdown",
                        "🔍 Top Shared Keywords (XAI)",
                        "💡 Actionable Recommendations",
                        "📈 Domain Category Coverage",
                        "📄 Raw Extracted Text",
                    ]
                )

                with tab1:
                    st.markdown("#### Punctuation-Aware Skill Taxonomy Matching")
                    st.caption(
                        "Extracted via boundary matching preserving C++, C#, .NET, Node.js, and CI/CD."
                    )

                    scol1, scol2, scol3 = st.columns(3)

                    with scol1:
                        st.markdown(
                            f"**✅ Matched Skills ({len(match_result.matched_skills)})**"
                        )
                        if match_result.matched_skills:
                            tags_html = " ".join(
                                f'<span class="skill-tag-matched">{s}</span>'
                                for s in sorted(match_result.matched_skills)
                            )
                            st.markdown(tags_html, unsafe_allow_html=True)
                        else:
                            st.info("No matching skills detected between resume and JD.")

                    with scol2:
                        st.markdown(
                            f"**❌ Missing Skills ({len(match_result.missing_skills)})**"
                        )
                        if match_result.missing_skills:
                            tags_html = " ".join(
                                f'<span class="skill-tag-missing">{s}</span>'
                                for s in sorted(match_result.missing_skills)
                            )
                            st.markdown(tags_html, unsafe_allow_html=True)
                        else:
                            st.success("All listed job skills are represented in your resume!")

                    with scol3:
                        st.markdown(
                            f"**ℹ️ Additional Skills ({len(match_result.additional_skills)})**"
                        )
                        if match_result.additional_skills:
                            tags_html = " ".join(
                                f'<span class="skill-tag-additional">{s}</span>'
                                for s in sorted(match_result.additional_skills)
                            )
                            st.markdown(tags_html, unsafe_allow_html=True)
                        else:
                            st.caption("No extra skills outside the JD scope detected.")

                with tab2:
                    st.markdown("#### Explainable AI: Top Shared N-Gram Features")
                    st.caption(
                        "Terms with highest vector dot product weights ($u_i \\times v_i$). "
                        "These terms directly explain the TF-IDF similarity score."
                    )

                    if match_result.top_shared_terms:
                        xai_df = pd.DataFrame(
                            match_result.top_shared_terms,
                            columns=["Shared Keyword / N-Gram", "Vector Dot Product Weight"],
                        )
                        xai_col1, xai_col2 = st.columns([1, 1])
                        with xai_col1:
                            st.dataframe(xai_df, use_container_width=True, hide_index=True)
                        with xai_col2:
                            xai_bar = px.bar(
                                xai_df.head(8).sort_values("Vector Dot Product Weight", ascending=True),
                                x="Vector Dot Product Weight",
                                y="Shared Keyword / N-Gram",
                                orientation="h",
                                title="Top Vector Space Influencers",
                                color="Vector Dot Product Weight",
                                color_continuous_scale=["#6366f1", "#06b6d4"],
                            )
                            xai_bar.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#f8fafc"),
                                height=260,
                                margin=dict(l=10, r=10, t=30, b=10),
                            )
                            st.plotly_chart(xai_bar, use_container_width=True)
                    else:
                        st.info("No significant overlapping n-grams found outside stop words.")

                with tab3:
                    st.markdown("#### Actionable Resume Optimization Steps")
                    st.caption(
                        "Suggestions derived automatically from identified skill gaps, "
                        "document length, and vocabulary alignment."
                    )
                    for rec in match_result.recommendations:
                        st.markdown(f"- {rec}")

                with tab4:
                    st.markdown("#### Technical Domain Skill Distribution")
                    st.caption("Visual breakdown of matched vs missing skills grouped by engineering category.")
                    cat_chart = create_category_breakdown_chart(
                        match_result.matched_by_category,
                        match_result.missing_by_category,
                    )
                    if cat_chart:
                        st.plotly_chart(cat_chart, use_container_width=True)
                    else:
                        st.info("Not enough categorical skills detected for breakdown.")

                with tab5:
                    st.markdown("#### Extracted Text Inspector")
                    raw_col1, raw_col2 = st.columns(2)
                    with raw_col1:
                        st.caption(f"**Resume Text ({match_result.resume_word_count} words)**")
                        st.text_area("Resume Dump", resume_input, height=220, disabled=True)
                    with raw_col2:
                        st.caption(f"**Job Description ({match_result.jd_word_count} words)**")
                        st.text_area("JD Dump", jd_input, height=220, disabled=True)


# ---------------------------------------------------------------------------
# VIEW 2: Candidate Benchmark & Ranking
# ---------------------------------------------------------------------------
elif view_mode == "📊 Candidate Benchmark & Ranking":

    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-subtitle">Multi-Resume Benchmark Evaluation</div>
            <h1 class="large-website-title">CANDIDATE RANKING</h1>
            <br>
            <span class="hero-tag">⚡ Batch Evaluation • Real Discrimination • Fictional Candidate Pool</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jds_df = get_sample_jds()
    role_options = jds_df["role"].tolist()

    bench_col1, bench_col2 = st.columns([2, 1])
    with bench_col1:
        target_role = st.selectbox(
            "Select Job Role for Candidate Benchmark:",
            options=role_options,
            index=0,
        )
    with bench_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        recompute_btn = st.button("🔄 Re-evaluate Candidates", use_container_width=True)

    with st.spinner(f"Evaluating candidate pool against '{target_role}'..."):
        benchmark_results = evaluate_candidates(
            target_role,
            tfidf_weight=norm_tfidf_w,
            skill_weight=norm_skill_w,
        )

    st.subheader(f"Rankings for '{target_role}'")
    st.plotly_chart(create_candidate_comparison_chart(benchmark_results), use_container_width=True)

    st.markdown("#### Detailed Candidate Comparison Table")
    display_cols = [
        "Candidate",
        "Composite ATS Score (%)",
        "TF-IDF Similarity (%)",
        "Skill Coverage (%)",
        "ATS Tier",
        "Matched Skills Count",
        "Missing Skills Count",
        "Top Matched Skills",
    ]
    st.dataframe(benchmark_results[display_cols], use_container_width=True)

    # Download Results CSV
    csv_data = benchmark_results.to_csv(index=True)
    st.download_button(
        label="📥 Export Benchmark Rankings as CSV",
        data=csv_data,
        file_name=f"candidate_rankings_{target_role.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# VIEW 3: Model Transparency & Viva Guide
# ---------------------------------------------------------------------------
elif view_mode == "🔬 Model Transparency & Viva Guide":

    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-subtitle">Academic Documentation & Viva Voce Guide</div>
            <h1 class="large-website-title">MODEL TRANSPARENCY</h1>
            <br>
            <span class="hero-tag">⚡ Mathematical Derivations • Zero Black-Box • 16 Viva Questions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_tab1, t_tab2, t_tab3 = st.tabs(
        [
            "📐 Mathematical Formulation",
            "🛡️ Punctuation-Aware Architecture",
            "🎓 15+ Viva Questions & Answers",
        ]
    )

    with t_tab1:
        st.markdown("### 1. Vector Space Model (VSM) & TF-IDF Formulation")
        st.markdown(
            r"""
            In an unsupervised Vector Space Model, each document (Resume $d_1$ and Job Description $d_2$)
            is mapped to a high-dimensional sparse vector $\mathbf{v} \in \mathbb{R}^{|V|}$, where $|V|$
            is the shared vocabulary size of unigrams and bigrams.

            #### Term Frequency with Sublinear Scaling:
            $$\text{tf}_{\text{scaled}}(t, d) = \begin{cases} 1 + \ln(\text{tf}(t, d)) & \text{if } \text{tf}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

            #### Inverse Document Frequency (Smooth IDF):
            $$\text{idf}(t) = \ln\left(\frac{1 + n}{1 + \text{df}(t)}\right) + 1$$

            #### Composite TF-IDF Weight:
            $$\text{tfidf}(t, d) = \text{tf}_{\text{scaled}}(t, d) \times \text{idf}(t)$$

            #### Cosine Similarity Metric:
            $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \sum_{i=1}^{|V|} u_i v_i$$
            """
        )

    with t_tab2:
        st.markdown("### Punctuation-Aware Skill Extraction Architecture")
        st.markdown(
            """
            Standard regex tokenizers (`\\w+`) strip punctuation, reducing `C++` and `C#` into `c`,
            causing false matches on the single letter 'c'.

            #### Solution Implemented in `skills.py`:
            - Negative Lookaround boundaries:
              `(?<![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])`
            - Preserves `.NET`, `Node.js`, `CI/CD`, and `scikit-learn` cleanly.
            """
        )

    with t_tab3:
        st.markdown("### 🎓 Comprehensive Viva Examination Questions & Model Answers")

        viva_qa = [
            (
                "Q1: Why is an unsupervised TF-IDF Vector Space Model used instead of a supervised classifier?",
                "Supervised learning requires thousands of labeled (Resume, Job Description, Hired/Rejected) pairs. "
                "Such datasets are rarely publicly available, suffer from severe historical hiring bias, and don't generalize "
                "to new, unseen job descriptions. An unsupervised Vector Space Model directly measures semantic and textual "
                "congruence between any arbitrary resume and job posting without requiring training labels or hallucinated parameters.",
            ),
            (
                "Q2: What is the mathematical definition and role of TF-IDF?",
                "TF-IDF (Term Frequency - Inverse Document Frequency) reflects how important a word is to a document relative to a corpus. "
                "TF measures local term frequency with sublinear scaling (1 + log(tf)). IDF penalizes pervasive words that appear across all "
                "documents (e.g., 'experience', 'work', 'skills') while rewarding distinctive domain terms (e.g., 'PyTorch', 'Kubernetes').",
            ),
            (
                "Q3: Why is Cosine Similarity preferred over Euclidean Distance for text comparison?",
                "Euclidean distance is sensitive to document length. If a resume is 3 pages long and another is 1 page long, their Euclidean "
                "distance will be large even if they discuss identical topics. Cosine similarity evaluates the angle between vectors rather than "
                "their magnitudes, making it completely length-invariant.",
            ),
            (
                "Q4: What is sublinear TF scaling and why is it important?",
                "Sublinear TF scaling replaces raw term frequency tf with 1 + ln(tf). In resume screening, a candidate who writes 'Python' "
                "20 times in a keyword-stuffing attempt should not receive 20 times the weight of a candidate who writes it once. "
                "Log scaling diminishes the marginal return of repeated words.",
            ),
            (
                "Q5: How does the system handle punctuation in technical terms like C++, C#, and .NET?",
                "Standard tokenizers strip punctuation, reducing C++ and C# to 'c'. Our engine uses a specialized regex pattern with negative "
                "lookaheads and lookbehinds: '(?<![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])'. This ensures 'C' only matches when isolated, "
                "and never matches inside 'C++' or 'C#'.",
            ),
            (
                "Q6: Why include bigrams (1, 2) in the TF-IDF vectorizer?",
                "Unigrams alone cannot distinguish crucial phrases. For example, 'machine' and 'learning' separately have very different meanings "
                "than the compound bigram 'machine learning'. Similarly, bigrams capture 'random forest', 'data science', and 'deep learning'.",
            ),
            (
                "Q7: Why does the system maintain both TF-IDF similarity and a separate Skill Coverage score?",
                "TF-IDF measures overall textual and stylistic alignment across all vocabulary. However, hiring managers have explicit non-negotiable "
                "skills (e.g., 'Must have Docker and SQL'). The Skill Coverage metric explicitly checks for the presence of required technical skills "
                "regardless of where or how frequently they appear.",
            ),
            (
                "Q8: What library is used for PDF text extraction and why?",
                "We use PyMuPDF (fitz), which is one of the fastest and most accurate C-based PDF parsing engines available for Python. It directly "
                "accesses digital text streams, character bounding boxes, and metadata without spawning external sub-processes.",
            ),
            (
                "Q9: How does the system handle corrupt, password-protected, or scanned PDFs?",
                "pdf_parser.py inspects doc.needs_pass for encryption, catches stream corruption via fitz.open exceptions, and verifies that the "
                "extracted character count is non-trivial. If a PDF contains only scanned images (zero text layer), it returns a clear diagnostic "
                "warning recommending an OCR preprocessor.",
            ),
            (
                "Q10: What are stop words and why are they removed?",
                "Stop words are high-frequency grammatical words like 'the', 'is', 'at', 'which'. Because they carry virtually no domain-specific "
                "discriminative information, removing them reduces vocabulary dimensionality and prevents common words from skewing cosine similarity.",
            ),
            (
                "Q11: What are the main limitations of pure TF-IDF matching?",
                "TF-IDF relies on exact lexical matching. It cannot recognize synonyms (e.g., 'software engineer' vs 'developer') or polysemy (e.g., "
                "'Apple' the fruit vs 'Apple' the company). Dense semantic embeddings (like Sentence-BERT) resolve this by capturing latent semantics.",
            ),
            (
                "Q12: How could this project be extended to Sentence-BERT or Word2Vec?",
                "By replacing or complementing the sparse TF-IDF vectors with dense embeddings from pre-trained models (e.g., all-MiniLM-L6-v2). "
                "Sentences from the resume and JD would be passed through transformer encoders to produce 384-dimensional dense vectors, "
                "followed by cosine similarity in semantic space.",
            ),
            (
                "Q13: What is the computational complexity of TF-IDF vectorization and cosine similarity?",
                "For two documents with average word count L and joint vocabulary V, building the vocabulary takes O(L) time. Fitting the matrix takes "
                "O(L) time. Computing the cosine similarity takes O(|V|) time, where |V| <= 2L. Overall time complexity is linear O(L), executing in under 20ms.",
            ),
            (
                "Q14: How does Explainable AI (XAI) work in this dashboard?",
                "Because cosine similarity is the dot product of L2-normalized vectors (sum u_i * v_i), each term's contribution is simply the product "
                "u_i * v_i. We sort these products descendingly to display the exact top keywords that drove the match score.",
            ),
            (
                "Q15: How do real corporate ATS platforms differ from this academic project?",
                "Enterprise ATS tools (e.g., Taleo, Workday, Greenhouse) integrate with relational databases, applicant workflow pipelines, OCR engines, "
                "compliance tracking (EEOC), email automation, and sometimes proprietary machine learning models. This project focuses on the core "
                "information retrieval and NLP matching component.",
            ),
            (
                "Q16: How can algorithmic bias occur in automated resume screening and how can it be mitigated?",
                "Bias enters if models are trained on historical hiring data that favored specific genders, ethnicities, or elite colleges. Our unsupervised "
                "baseline avoids this because it computes objective lexical overlap between the job requirements and candidate text, without any demographic "
                "features or historical training weights.",
            ),
        ]

        for q, a in viva_qa:
            with st.expander(f"📌 {q}"):
                st.write(a)
