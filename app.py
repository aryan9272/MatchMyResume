"""
app.py - ATS Resume Matcher using Natural Language Processing and Machine Learning.

College-Level Introduction to Machine Learning (IML) Project.
Tech Stack: Python, Streamlit, scikit-learn (TF-IDF + Cosine Similarity),
            PyMuPDF, Pandas, Plotly.

Design: Clean, high-contrast, professional LIGHT THEME (no dark mode, no glassmorphism).
No external APIs, cloud databases, or paid services required.
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
    page_title="ATS Resume Matcher | NLP & ML",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional Light Theme CSS
# Strict Light Theme: Crisp surfaces, high contrast, no glassmorphism
# ---------------------------------------------------------------------------
LIGHT_THEME_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #334155;
    }

    /* Cards and Containers */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-subtitle {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Status Tier Badges */
    .badge-tier-high {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-tier-good {
        display: inline-block;
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-tier-mod {
        display: inline-block;
        background-color: #fef9c3;
        color: #a16207;
        border: 1px solid #fde047;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-tier-low {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Skill Tags */
    .skill-tag-matched {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 3px 9px;
        border-radius: 5px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 2px;
    }
    .skill-tag-missing {
        display: inline-block;
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 3px 9px;
        border-radius: 5px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 2px;
    }
    .skill-tag-additional {
        display: inline-block;
        background-color: #f0f9ff;
        color: #075985;
        border: 1px solid #bae6fd;
        padding: 3px 9px;
        border-radius: 5px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 2px;
    }

    /* Information and Header Banners */
    .header-banner {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .header-desc {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 0px;
    }
    .badge-transparency {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 8px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent;
        border: none;
        color: #475569;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Inputs and Buttons */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.85rem !important;
    }
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }
</style>
"""
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)


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
# Visualization Helpers (Plotly in Light Theme)
# ---------------------------------------------------------------------------
def create_score_gauge(score_pct: float) -> go.Figure:
    """Generate a clean Plotly gauge chart for the Composite ATS Score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            number={"suffix": "%", "font": {"size": 36, "color": "#0f172a", "weight": "bold"}},
            title={"text": "Composite ATS Score", "font": {"size": 14, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#2563eb", "thickness": 0.28},
                "bgcolor": "#f1f5f9",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 30], "color": "#fee2e2"},
                    {"range": [30, 50], "color": "#fef3c7"},
                    {"range": [50, 70], "color": "#ecfdf5"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.75,
                    "value": score_pct,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=220,
        margin=dict(l=25, r=25, t=35, b=20),
    )
    return fig


def create_score_breakdown_chart(tfidf_pct: float, skill_pct: float, composite_pct: float) -> go.Figure:
    """Bar chart comparing the sub-scores with composite score."""
    categories = ["TF-IDF Cosine Similarity", "Skill Coverage Ratio", "Composite ATS Score"]
    values = [tfidf_pct, skill_pct, composite_pct]
    colors = ["#0284c7", "#10b981", "#2563eb"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
                marker=dict(color=colors, line=dict(color="#e2e8f0", width=1)),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis=dict(range=[0, 100], title="Score (%)", gridcolor="#f1f5f9", ticksuffix="%"),
        xaxis=dict(tickfont=dict(color="#0f172a", size=12)),
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
            marker=dict(color="#10b981"),
        )
    )
    fig.add_trace(
        go.Bar(
            y=all_categories,
            x=missing_counts,
            name="Missing Skills",
            orientation="h",
            marker=dict(color="#f87171"),
        )
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(title="Number of Skills", gridcolor="#f1f5f9"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def create_candidate_comparison_chart(benchmark_df: pd.DataFrame) -> go.Figure:
    """Plotly horizontal bar chart ranking fictional candidate resumes on the selected role."""
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
            marker=dict(color="#2563eb"),
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
            marker=dict(color="#10b981"),
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
            marker=dict(color="#0284c7"),
        )
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(range=[0, 100], title="Percentage (%)", gridcolor="#f1f5f9", ticksuffix="%"),
        yaxis=dict(tickfont=dict(color="#0f172a", size=12, family="sans-serif")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 📄 ATS Matcher Navigation")
view_mode = st.sidebar.radio(
    "Select Project View:",
    options=[
        "🎯 Resume Matcher & Gap Analysis",
        "📊 Candidate Benchmark & Ranking",
        "🔬 Model Transparency & Viva Guide",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Scoring Weights")
st.sidebar.caption("Adjust composite linear weights (sum is normalized to 1.0):")

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
st.sidebar.caption(f"Normalized formula: `{norm_tfidf_w} × TFIDF + {norm_skill_w} × SkillCoverage`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Quick Candidate Pre-Loader")
st.sidebar.caption("Pre-fill resume with sample fictional candidates:")
resumes_df = get_sample_resumes()
candidate_names = ["(Upload PDF or Paste Custom)"] + resumes_df["name"].tolist()
selected_candidate = st.sidebar.selectbox("Choose Sample Candidate:", options=candidate_names, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Specs:**
    - Model: *Unsupervised Vector Space Model*
    - Feature Extraction: *TF-IDF (1, 2 n-grams)*
    - Similarity: *Cosine Similarity*
    - Parser: *PyMuPDF (Fitz)*
    - Platform: *100% Local / Zero API Keys*
    """
)


# ---------------------------------------------------------------------------
# VIEW 1: Resume Matcher & Gap Analysis
# ---------------------------------------------------------------------------
if view_mode == "🎯 Resume Matcher & Gap Analysis":

    st.markdown(
        """
        <div class="header-banner">
            <div class="header-title">ATS Resume Matcher & NLP Keyword Analyzer</div>
            <div class="header-desc">
                Evaluate resume relevance against target job descriptions using Natural Language Processing,
                unigram/bigram TF-IDF cosine similarity, and punctuation-aware skill dictionary taxonomy.
            </div>
            <span class="badge-transparency">Zero-Fabrication Baseline: TF-IDF fitted on compared document vocabulary</span>
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
                st.info(f"Loaded sample profile for **{selected_candidate}**.")

        if resume_source_mode == "Upload PDF Document":
            uploaded_pdf = st.file_uploader(
                "Upload Resume in PDF Format",
                type=["pdf"],
                help="PyMuPDF will parse text layer directly without external servers.",
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

            # Show text area to let user review or edit the extracted text
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
            help="Selecting a role auto-populates standard industry requirements.",
        )

        # Retrieve default text for selected role
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
    run_match = st.button("🚀 Calculate ATS Match & Skill Coverage", use_container_width=True)

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
                # Metrics Row
                # -----------------------------------------------------------
                st.markdown("---")
                st.subheader("📊 Match Performance Summary")

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                with mcol1:
                    tier_class = "badge-tier-mod"
                    if match_result.composite_pct >= 80:
                        tier_class = "badge-tier-high"
                    elif match_result.composite_pct >= 65:
                        tier_class = "badge-tier-good"
                    elif match_result.composite_pct < 40:
                        tier_class = "badge-tier-low"

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Composite ATS Score</div>
                            <div class="metric-value">{match_result.composite_pct}%</div>
                            <div class="metric-subtitle">
                                <span class="{tier_class}">{match_result.ats_label}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">TF-IDF Similarity</div>
                            <div class="metric-value">{match_result.tfidf_pct}%</div>
                            <div class="metric-subtitle">N-gram (1,2) Cosine Overlap</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Skill Coverage</div>
                            <div class="metric-value">{match_result.skill_coverage_pct}%</div>
                            <div class="metric-subtitle">
                                {len(match_result.matched_skills)} of {len(match_result.jd_skills)} Required Skills
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Vocabulary & Speed</div>
                            <div class="metric-value">{match_result.tfidf_vocab_size}</div>
                            <div class="metric-subtitle">
                                Joint Vocab Size | {match_result.processing_time_ms} ms
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # -----------------------------------------------------------
                # Visual Analytics Row
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
                        "Skills extracted using exact boundary matching preserving C++, C#, .NET, "
                        "Node.js, CI/CD, and hyphenated machine learning algorithms."
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
                        "Shows the specific terms with highest vector dot product weights ($u_i \\times v_i$). "
                        "These terms directly explain why the TF-IDF similarity score was assigned."
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
                                color_continuous_scale="Blues",
                            )
                            xai_bar.update_layout(
                                paper_bgcolor="#ffffff",
                                plot_bgcolor="#ffffff",
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
                    st.caption("Visual breakdown of matched versus missing skills grouped by engineering category.")
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
        <div class="header-banner">
            <div class="header-title">Candidate Benchmark & Multi-Resume Ranking</div>
            <div class="header-desc">
                Evaluate all 5 fictional sample candidates against any target job role simultaneously.
                Demonstrates the vector space model's discriminatory power across diverse engineering backgrounds.
            </div>
            <span class="badge-transparency">Unsupervised Batch Evaluation: No synthetic scores or hardcoded rankings</span>
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

    st.markdown("---")
    st.markdown("#### 💡 Candidate Profile Insights")
    st.markdown(
        """
        - **Eva Chen** (AI/ML Engineer): Top match for AI/ML and Data Science roles due to deep transformer & MLOps coverage.
        - **Carol Smith** (M.Sc. Data Science): High match for Data Science and Analyst roles with strong PyTorch and Spark experience.
        - **Alice Johnson** (Fresh Graduate): Strong college-level Data Science profile with solid baseline NLP and scikit-learn skills.
        - **Bob Martinez** (Full Stack Developer): Dominates for Web / Full Stack roles with React, Node.js, and Docker; low fit for Data Science.
        - **David Lee** (Fresher / IT Support): Demonstrates accurate low scoring for advanced roles, preventing false positive hires.
        """
    )


# ---------------------------------------------------------------------------
# VIEW 3: Model Transparency & Viva Guide
# ---------------------------------------------------------------------------
elif view_mode == "🔬 Model Transparency & Viva Guide":

    st.markdown(
        """
        <div class="header-banner">
            <div class="header-title">Model Transparency, Mathematical Foundation & Viva Guide</div>
            <div class="header-desc">
                Comprehensive academic documentation for Introduction to Machine Learning (IML) viva examinations,
                detailing TF-IDF formulations, Vector Space Models, and practical engineering trade-offs.
            </div>
            <span class="badge-transparency">Zero Black-Box: Fully explainable NLP pipeline</span>
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
            Raw term frequency $\text{tf}(t, d)$ represents the number of occurrences of term $t$ in document $d$.
            To prevent documents with repetitive keyword stuffing from gaining unfair dominance, we apply sublinear log-scaling:
            $$\text{tf}_{\text{scaled}}(t, d) = \begin{cases} 1 + \ln(\text{tf}(t, d)) & \text{if } \text{tf}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

            #### Inverse Document Frequency (Smooth IDF):
            With $n=2$ documents in the pairwise comparison (or $N$ in a corpus):
            $$\text{idf}(t) = \ln\left(\frac{1 + n}{1 + \text{df}(t)}\right) + 1$$
            Where $\text{df}(t)$ is the document frequency (number of documents containing term $t$).

            #### Composite TF-IDF Weight:
            $$\text{tfidf}(t, d) = \text{tf}_{\text{scaled}}(t, d) \times \text{idf}(t)$$

            #### L2 Normalization:
            Each document vector is normalized to unit Euclidean length:
            $$\mathbf{u} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{|V|} v_i^2}}$$

            #### Cosine Similarity Metric:
            Cosine similarity measures the cosine of the angle $\theta$ between the unit vectors:
            $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \sum_{i=1}^{|V|} u_i v_i$$
            Because all TF-IDF weights are non-negative, $\text{Cosine Similarity} \in [0.0, 1.0]$.
            """
        )

        st.markdown("### 2. Composite ATS Score Formulation")
        st.markdown(
            r"""
            The final ATS Score balances contextual text congruence with explicit skill coverage:
            $$\text{ATS Score} = w_{\text{tfidf}} \cdot \text{CosineSimilarity} + w_{\text{skill}} \cdot \left(\frac{|\text{ResumeSkills} \cap \text{JDSkills}|}{|\text{JDSkills}|}\right)$$
            Where $w_{\text{tfidf}} + w_{\text{skill}} = 1.0$.
            """
        )

    with t_tab2:
        st.markdown("### Punctuation-Aware Skill Extraction Architecture")
        st.markdown(
            """
            Standard regex tokenizers (`\\w+`) strip all non-alphanumeric punctuation. This creates catastrophic failures for technical skills:
            - **`C++`** collapses into `c`, falsely matching any isolated letter 'c'.
            - **`C#`** collapses into `c`.
            - **`.NET`** loses its leading dot or is discarded as punctuation.
            - **`Node.js`** separates into `node` and `js`.
            - **`CI/CD`** separates into `ci` and `cd`.

            #### Solution Implemented in `skills.py`:
            1. **Lookahead / Lookbehind Boundaries**: For single-letter languages like `C`, we use:
               `(?<![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])`
               This guarantees `C` will **never** trigger when the resume contains `C++` or `C#`.
            2. **Alias Canonicalization**: Terms like `sklearn` are mapped to `scikit-learn`, `k8s` to `Kubernetes`.
            3. **Custom TF-IDF Token Pattern**:
               `token_pattern=r"(?u)\\b[a-zA-Z0-9][a-zA-Z0-9\\+\\#\\.\\-_/]*\\b"`
               Preserves technical punctuation during scikit-learn vectorization.
            """
        )

    with t_tab3:
        st.markdown("### 🎓 Comprehensive Viva Examination Questions & Model Answers")
        st.caption("Review these 16 questions to prepare for project defense and viva evaluation:")

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
