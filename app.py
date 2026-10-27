"""
app.py - MATCH MY RESUME: Light Theme with Floating 3D Cards & Ambient Animation.

Tech Stack: Python, Streamlit, scikit-learn (TF-IDF + Cosine Similarity),
            PyMuPDF, Pandas, Plotly.

Design:
- Crisp Light Theme with Frosted Floating Cards.
- Subtle 3D hover elevation and ambient floating background elements.
- Strict 3-4 color palette: Clean Canvas/White (#f8fafc / #ffffff),
  Royal Blue (#2563eb), Cyber Cyan (#06b6d4), and Slate (#0f172a / #475569).
- Large headline with cinematic 3D opening animation.
- No external APIs, cloud databases, or paid services required.
"""

from __future__ import annotations

import io
from pathlib import Path
import base64
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluate import (
    evaluate_candidates,
    evaluate_kaggle_candidates,
    get_kaggle_categories,
    load_job_descriptions,
    load_kaggle_resumes,
    load_resumes,
)
from matcher import MatchResult, match_resume_to_jd
from pdf_parser import extract_text_from_pdf, parse_uploaded_pdf
from skills import TECH_SKILLS, categorize_skills, get_skill_category

# ---------------------------------------------------------------------------
# Brand Assets & Base64 Loading
# ---------------------------------------------------------------------------
LOGO_PATH = Path("assets/logo.png")
FAVICON_PATH = Path("assets/favicon.png")

if LOGO_PATH.exists():
    with open(LOGO_PATH, "rb") as _f:
        LOGO_B64 = base64.b64encode(_f.read()).decode("utf-8")
else:
    LOGO_B64 = ""

# ---------------------------------------------------------------------------
# Streamlit Page Configuration with Custom Favicon
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MatchMyResume | ATS Resume Matcher & NLP Analyzer",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light Theme with Floating Cards & 3D Interactive Hover Effects CSS
# Strict 3-4 Color Palette: #f8fafc (Canvas), #ffffff (Card), #2563eb (Royal Blue), #06b6d4 (Cyan), #0f172a (Slate)
# ---------------------------------------------------------------------------
LIGHT_FLOATING_THEME_CSS = """
<style>
    /* =======================================================================
       1. GLOBAL LIGHT BACKGROUND & 3D AMBIENT ORBS
       ======================================================================= */
    .stApp {
        background-color: #f8fafc !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.07) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.07) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(37, 99, 235, 0.05) 0px, transparent 60%);
        background-attachment: fixed;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        overflow-x: hidden;
    }

    /* Ambient Soft Floating 3D Orbs in Background */
    @keyframes floatOrbLight1 {
        0% { transform: translate3d(0, 0, 0) scale(1) rotate(0deg); }
        50% { transform: translate3d(50px, 35px, 60px) scale(1.12) rotate(180deg); }
        100% { transform: translate3d(0, 0, 0) scale(1) rotate(360deg); }
    }
    @keyframes floatOrbLight2 {
        0% { transform: translate3d(0, 0, 0) scale(1.08) rotate(0deg); }
        50% { transform: translate3d(-40px, -50px, 40px) scale(0.92) rotate(-180deg); }
        100% { transform: translate3d(0, 0, 0) scale(1.08) rotate(-360deg); }
    }

    .bg-orb-light-1 {
        position: fixed;
        top: 8%;
        left: -4%;
        width: 480px;
        height: 480px;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.08) 0%, rgba(37, 99, 235, 0) 70%);
        filter: blur(55px);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: floatOrbLight1 20s ease-in-out infinite;
    }
    .bg-orb-light-2 {
        position: fixed;
        bottom: 12%;
        right: -4%;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.09) 0%, rgba(6, 182, 212, 0) 70%);
        filter: blur(65px);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: floatOrbLight2 24s ease-in-out infinite;
    }

    /* =======================================================================
       2. CINEMATIC 3D OPENING ENTRANCE ANIMATION & LARGE HEADING
       ======================================================================= */
    @keyframes openingReveal3D {
        0% {
            opacity: 0;
            transform: perspective(1100px) rotateX(16deg) translateY(40px) scale(0.93);
            filter: blur(10px);
        }
        70% {
            transform: perspective(1100px) rotateX(-2deg) translateY(-4px) scale(1.01);
            filter: blur(0px);
        }
        100% {
            opacity: 1;
            transform: perspective(1100px) rotateX(0deg) translateY(0) scale(1);
            filter: blur(0px);
        }
    }

    .hero-floating-container {
        animation: openingReveal3D 1.1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transform-style: preserve-3d;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 20px;
        padding: 34px 32px 28px 32px;
        margin-bottom: 26px;
        box-shadow: 
            0 15px 35px -5px rgba(15, 23, 42, 0.06), 
            0 6px 12px -4px rgba(15, 23, 42, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        position: relative;
        z-index: 1;
        overflow: hidden;
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.35s ease,
                    border-color 0.35s ease;
    }

    .hero-floating-container:hover {
        transform: translateY(-3px) scale(1.004);
        border-color: rgba(37, 99, 235, 0.35);
        box-shadow: 
            0 24px 48px -10px rgba(37, 99, 235, 0.12),
            0 10px 20px -4px rgba(15, 23, 42, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 1);
    }

    .hero-floating-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2563eb, #06b6d4, #2563eb);
        transition: height 0.3s ease;
    }

    .hero-floating-container:hover::before {
        height: 4px;
    }

    .large-website-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.05 !important;
        margin: 0 0 8px 0 !important;
        background: linear-gradient(135deg, #0f172a 20%, #2563eb 65%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        transition: transform 0.3s ease;
    }

    .large-website-title:hover {
        transform: scale(1.01);
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #475569;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(37, 99, 235, 0.08);
        color: #2563eb;
        border: 1px solid rgba(37, 99, 235, 0.25);
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        transition: all 0.25s ease;
    }

    .hero-tag:hover {
        background: rgba(37, 99, 235, 0.14);
        border-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }

    /* =======================================================================
       3. FLOATING CARDS & 3D HOVER ELEVATION WITH LIGHT SWEEP
       ======================================================================= */
    .floating-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(226, 232, 240, 0.95) !important;
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 
            0 10px 25px -5px rgba(15, 23, 42, 0.05),
            0 8px 10px -6px rgba(15, 23, 42, 0.02),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), 
                    box-shadow 0.35s cubic-bezier(0.16, 1, 0.3, 1), 
                    border-color 0.35s ease !important;
        transform-style: preserve-3d;
        position: relative;
        overflow: hidden;
    }

    .floating-card:hover {
        transform: translateY(-6px) perspective(1000px) rotateX(1.5deg) scale(1.012) !important;
        border-color: rgba(37, 99, 235, 0.45) !important;
        box-shadow: 
            0 22px 42px -10px rgba(37, 99, 235, 0.18),
            0 0 0 1px rgba(37, 99, 235, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 1) !important;
    }

    /* Subtle Light Glint Sweep on Card Hover */
    .floating-card::after {
        content: "";
        position: absolute;
        top: -60%;
        left: -80%;
        width: 220%;
        height: 220%;
        background: linear-gradient(
            60deg,
            transparent 40%,
            rgba(255, 255, 255, 0.7) 50%,
            transparent 60%
        );
        transform: rotate(25deg);
        transition: all 0.75s ease;
        opacity: 0;
        pointer-events: none;
    }

    .floating-card:hover::after {
        left: 40%;
        opacity: 1;
    }

    .floating-metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        transition: color 0.25s ease;
    }

    .floating-card:hover .floating-metric-title {
        color: #2563eb;
    }

    .floating-metric-value {
        font-size: 2.3rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.1;
        background: linear-gradient(135deg, #0f172a 40%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: transform 0.25s ease;
    }

    .floating-card:hover .floating-metric-value {
        transform: scale(1.02);
    }

    .floating-metric-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Badges & Tags (Strict 3-4 Color Palette) */
    .badge-tier-high {
        display: inline-block;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .badge-tier-good {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .badge-tier-mod {
        display: inline-block;
        background: #fefce8;
        color: #a16207;
        border: 1px solid #fde047;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .badge-tier-low {
        display: inline-block;
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .badge-tier-high:hover, .badge-tier-good:hover, .badge-tier-mod:hover, .badge-tier-low:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
    }

    /* Skill Tags with Spring Hover Feedback */
    .skill-tag-matched {
        display: inline-block;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 5px 11px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 3px;
        box-shadow: 0 1px 2px rgba(16, 185, 129, 0.08);
        transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.22s ease,
                    background-color 0.2s ease,
                    border-color 0.2s ease !important;
        cursor: pointer;
        user-select: none;
    }
    .skill-tag-matched:hover {
        transform: translateY(-3px) scale(1.08) !important;
        background: #d1fae5 !important;
        border-color: #34d399 !important;
        box-shadow: 0 6px 14px rgba(16, 185, 129, 0.28) !important;
    }

    .skill-tag-missing {
        display: inline-block;
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 5px 11px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 3px;
        box-shadow: 0 1px 2px rgba(239, 68, 68, 0.08);
        transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.22s ease,
                    background-color 0.2s ease,
                    border-color 0.2s ease !important;
        cursor: pointer;
        user-select: none;
    }
    .skill-tag-missing:hover {
        transform: translateY(-3px) scale(1.08) !important;
        background: #fee2e2 !important;
        border-color: #f87171 !important;
        box-shadow: 0 6px 14px rgba(239, 68, 68, 0.28) !important;
    }

    .skill-tag-additional {
        display: inline-block;
        background: #f0f9ff;
        color: #075985;
        border: 1px solid #bae6fd;
        padding: 5px 11px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 3px;
        box-shadow: 0 1px 2px rgba(6, 182, 212, 0.08);
        transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.22s ease,
                    background-color 0.2s ease,
                    border-color 0.2s ease !important;
        cursor: pointer;
        user-select: none;
    }
    .skill-tag-additional:hover {
        transform: translateY(-3px) scale(1.08) !important;
        background: #e0f2fe !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 6px 14px rgba(6, 182, 212, 0.28) !important;
    }

    /* Actionable Recommendation Cards */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        font-size: 0.92rem;
        color: #1e293b;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.25s ease,
                    border-left-color 0.25s ease !important;
    }
    .rec-card:hover {
        transform: translateX(6px) translateY(-1px) !important;
        border-left-color: #06b6d4 !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.12) !important;
        background: #fcfdfe !important;
    }

    /* =======================================================================
       4. SIDEBAR & INPUT ELEMENTS WITH INTERACTIVE HOVER
       ======================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* Sidebar Navigation Menu Radio Items styled as Modern Interactive Cards */
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 11px 16px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #eff6ff !important;
        border-color: #93c5fd !important;
        transform: translateX(5px) translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.12) !important;
    }

    /* Text Area & Input Hover Feedback */
    .stTextArea textarea {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-family: ui-monospace, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.86rem !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    .stTextArea textarea:hover {
        border-color: #94a3b8 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    }
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
    }

    /* File Uploader Hover Feedback */
    div[data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 14px !important;
        padding: 18px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #2563eb !important;
        background: rgba(37, 99, 235, 0.03) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 24px -4px rgba(37, 99, 235, 0.12) !important;
    }

    /* Button Styling & Glow Hover */
    .stButton>button, div[data-testid="stDownloadButton"]>button {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 13px 28px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        box-shadow: 0 8px 22px -4px rgba(37, 99, 235, 0.35) !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1),
                    box-shadow 0.25s ease,
                    filter 0.25s ease !important;
        cursor: pointer !important;
    }
    .stButton>button:hover, div[data-testid="stDownloadButton"]>button:hover {
        transform: translateY(-3px) scale(1.015) !important;
        box-shadow: 0 14px 30px -4px rgba(37, 99, 235, 0.45), 0 0 0 2px rgba(6, 182, 212, 0.25) !important;
        filter: brightness(1.06) !important;
    }
    .stButton>button:active, div[data-testid="stDownloadButton"]>button:active {
        transform: translateY(1px) scale(0.99) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }

    /* Tabs Styling & Hover */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 18px;
        border-radius: 10px 10px 0 0 !important;
        transition: all 0.22s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563eb !important;
        background: rgba(37, 99, 235, 0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
        background: rgba(37, 99, 235, 0.06) !important;
    }

    /* Expander Details Hover */
    div[data-testid="stExpander"] details summary {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        border-color: #2563eb !important;
        background: #f8fafc !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08) !important;
        transform: translateY(-1px) !important;
    }
    /* Top Brand Navigation Bar */
    .top-brand-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 16px;
        padding: 12px 22px;
        margin-bottom: 22px;
        box-shadow: 
            0 8px 20px -4px rgba(15, 23, 42, 0.04),
            0 4px 6px -2px rgba(15, 23, 42, 0.02);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .top-brand-navbar:hover {
        transform: translateY(-2px);
        border-color: rgba(37, 99, 235, 0.35);
        box-shadow: 
            0 14px 28px -6px rgba(37, 99, 235, 0.12),
            0 4px 10px -2px rgba(15, 23, 42, 0.04);
    }
    .brand-logo-group {
        display: flex;
        align-items: center;
        gap: 14px;
        text-decoration: none;
    }
    .brand-logo-img {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .top-brand-navbar:hover .brand-logo-img {
        transform: rotate(6deg) scale(1.08);
    }
    .brand-title-wrap {
        display: flex;
        flex-direction: column;
    }
    .brand-name-main {
        font-size: 1.5rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #0f172a;
        line-height: 1.1;
    }
    .brand-name-accent {
        color: #2563eb;
    }
    .brand-tagline {
        font-size: 0.76rem;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .brand-status-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #065f46;
        transition: all 0.25s ease;
    }
    .brand-status-badge:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.2);
    }
    .status-pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseDot 2s infinite;
    }
    @keyframes pulseDot {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Sidebar Brand Header */
    .sidebar-brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        margin-bottom: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        transition: all 0.25s ease;
    }
    .sidebar-brand-box:hover {
        background: #eff6ff;
        border-color: #93c5fd;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
    }
    .sidebar-logo-img {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    .sidebar-brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hero Brand Layout with Logo */
    .hero-brand-header {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 8px;
    }
    .hero-logo-img {
        width: 72px;
        height: 72px;
        border-radius: 18px;
        box-shadow: 0 10px 25px -4px rgba(37, 99, 235, 0.35);
        transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .hero-floating-container:hover .hero-logo-img {
        transform: rotate(6deg) scale(1.08);
    }
</style>

<!-- Floating Ambient Subtle Glow Orbs -->
<div class="bg-orb-light-1"></div>
<div class="bg-orb-light-2"></div>
"""
st.markdown(LIGHT_FLOATING_THEME_CSS, unsafe_allow_html=True)


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
# Plotly Charts Styled in Light Theme with Fixed Modern Syntax
# (No deprecated 'titlefont'; uses title=dict(text=..., font=dict(...)))
# ---------------------------------------------------------------------------
def create_score_gauge(score_pct: float) -> go.Figure:
    """Generate a clean Plotly gauge chart styled for the light theme."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            number={"suffix": "%", "font": {"size": 40, "color": "#0f172a", "family": "Inter, sans-serif"}},
            title={"text": "Composite ATS Score", "font": {"size": 14, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#2563eb", "thickness": 0.28},
                "bgcolor": "#f1f5f9",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 35], "color": "#fee2e2"},
                    {"range": [35, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#dcfce7"},
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=25, r=25, t=35, b=20),
    )
    return fig


def create_score_breakdown_chart(tfidf_pct: float, skill_pct: float, composite_pct: float) -> go.Figure:
    """
    Bar chart comparing sub-scores with composite score in light theme.
    Fixed: Uses modern Plotly title dict syntax (no deprecated 'titlefont').
    """
    categories = ["TF-IDF Cosine", "Skill Coverage", "Composite ATS"]
    values = [tfidf_pct, skill_pct, composite_pct]
    colors = ["#06b6d4", "#10b981", "#2563eb"]

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
                    line=dict(color="#e2e8f0", width=1),
                ),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, 100],
            title=dict(text="Score (%)", font=dict(color="#64748b")),
            gridcolor="#f1f5f9",
            tickfont=dict(color="#64748b"),
            ticksuffix="%",
        ),
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
            marker=dict(color="#10b981", line=dict(color="#cbd5e1", width=1)),
        )
    )
    fig.add_trace(
        go.Bar(
            y=all_categories,
            x=missing_counts,
            name="Missing Skills",
            orientation="h",
            marker=dict(color="#f87171", line=dict(color="#cbd5e1", width=1)),
        )
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Skill Count", font=dict(color="#64748b")),
            gridcolor="#f1f5f9",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#0f172a", size=11)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#0f172a"),
        ),
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def create_candidate_comparison_chart(benchmark_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Plotly horizontal bar chart ranking candidate resumes."""
    cand_col = "Candidate" if "Candidate" in benchmark_df.columns else "Candidate ID"
    # Take top N and sort ascending for horizontal display
    df_top = benchmark_df.head(top_n)
    df_sorted = df_top.sort_values("Composite ATS Score (%)", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_sorted[cand_col],
            x=df_sorted["Composite ATS Score (%)"],
            name="Composite ATS Score (%)",
            orientation="h",
            text=[f"{v:.1f}%" for v in df_sorted["Composite ATS Score (%)"]],
            textposition="auto",
            textfont=dict(color="#ffffff"),
            marker=dict(color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Bar(
            y=df_sorted[cand_col],
            x=df_sorted["Skill Coverage (%)"],
            name="Skill Coverage (%)",
            orientation="h",
            text=[f"{v:.1f}%" for v in df_sorted["Skill Coverage (%)"]],
            textposition="auto",
            textfont=dict(color="#ffffff"),
            marker=dict(color="#10b981"),
        )
    )
    fig.add_trace(
        go.Bar(
            y=df_sorted[cand_col],
            x=df_sorted["TF-IDF Similarity (%)"],
            name="TF-IDF Similarity (%)",
            orientation="h",
            text=[f"{v:.1f}%" for v in df_sorted["TF-IDF Similarity (%)"]],
            textposition="auto",
            textfont=dict(color="#ffffff"),
            marker=dict(color="#06b6d4"),
        )
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 100],
            title=dict(text="Percentage (%)", font=dict(color="#64748b")),
            gridcolor="#f1f5f9",
            tickfont=dict(color="#64748b"),
            ticksuffix="%",
        ),
        yaxis=dict(tickfont=dict(color="#0f172a", size=11)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#0f172a"),
        ),
        height=max(320, len(df_sorted) * 28),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def create_score_distribution_histogram(benchmark_df: pd.DataFrame) -> go.Figure:
    """Plotly histogram showing score distribution across candidate population."""
    fig = px.histogram(
        benchmark_df,
        x="Composite ATS Score (%)",
        nbins=14,
        title="Score Distribution Across Population",
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Composite ATS Score (%)", font=dict(color="#64748b")),
            gridcolor="#f1f5f9",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(
            title=dict(text="Candidate Count", font=dict(color="#64748b")),
            gridcolor="#f1f5f9",
            tickfont=dict(color="#64748b"),
        ),
        height=280,
        margin=dict(l=20, r=20, t=35, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar Controls & Navigation (Clean Light Theme)
# ---------------------------------------------------------------------------
# Sidebar Top Brand Header with Logo & Name
st.sidebar.markdown(
    f"""
    <div class="sidebar-brand-box">
        <img src="data:image/png;base64,{LOGO_B64}" class="sidebar-logo-img" alt="MatchMyResume Logo" />
        <div>
            <div class="sidebar-brand-name">Match<span style="color:#2563eb;">My</span>Resume</div>
            <div class="sidebar-brand-sub">ATS Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### ⚡ Navigation")
view_mode = st.sidebar.radio(
    "Select View:",
    options=[
        "🎯 Resume Matcher & Gap Analysis",
        "📊 Candidate Benchmark & Ranking",
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


# ---------------------------------------------------------------------------
# Top Brand Navigation Header Bar with Logo and Name MatchMyResume
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="top-brand-navbar">
        <div class="brand-logo-group">
            <img src="data:image/png;base64,{LOGO_B64}" class="brand-logo-img" alt="MatchMyResume Logo" />
            <div class="brand-title-wrap">
                <span class="brand-name-main">Match<span class="brand-name-accent">My</span>Resume</span>
                <span class="brand-tagline">AI-Powered ATS Matching Engine</span>
            </div>
        </div>
        <div class="brand-status-badge">
            <span class="status-pulse-dot"></span>
            <span class="status-text">NLP &amp; ML Verified • 100% Local</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# VIEW 1: Resume Matcher & Gap Analysis
# ---------------------------------------------------------------------------
if view_mode == "🎯 Resume Matcher & Gap Analysis":

    # Hero Banner with Large Heading, Brand Logo and 3D Opening Animation
    st.markdown(
        f"""
        <div class="hero-floating-container">
            <div class="hero-brand-header">
                <img src="data:image/png;base64,{LOGO_B64}" class="hero-logo-img" alt="MatchMyResume Logo" />
                <div>
                    <div class="hero-subtitle">Intelligent ATS Resume Analyzer &amp; NLP Matcher</div>
                    <h1 class="large-website-title">MatchMyResume</h1>
                </div>
            </div>
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
    run_match = st.button("🚀 Analyze Resume & Calculate ATS Match", use_container_width=True)

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
                # Metrics Row (Floating 3D Cards)
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
                        <div class="floating-card">
                            <div class="floating-metric-title">Composite ATS Score</div>
                            <div class="floating-metric-value">{match_result.composite_pct}%</div>
                            <div class="floating-metric-sub">
                                <span class="{tier_class}">{match_result.ats_label}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol2:
                    st.markdown(
                        f"""
                        <div class="floating-card">
                            <div class="floating-metric-title">TF-IDF Similarity</div>
                            <div class="floating-metric-value">{match_result.tfidf_pct}%</div>
                            <div class="floating-metric-sub">N-gram (1,2) Cosine Overlap</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol3:
                    st.markdown(
                        f"""
                        <div class="floating-card">
                            <div class="floating-metric-title">Skill Coverage</div>
                            <div class="floating-metric-value">{match_result.skill_coverage_pct}%</div>
                            <div class="floating-metric-sub">
                                {len(match_result.matched_skills)} of {len(match_result.jd_skills)} Required Skills
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with mcol4:
                    st.markdown(
                        f"""
                        <div class="floating-card">
                            <div class="floating-metric-title">Vocabulary & Speed</div>
                            <div class="floating-metric-value">{match_result.tfidf_vocab_size}</div>
                            <div class="floating-metric-sub">
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
                                color_continuous_scale=["#2563eb", "#06b6d4"],
                            )
                            xai_bar.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#0f172a"),
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
                        st.markdown(f'<div class="rec-card">💡 {rec}</div>', unsafe_allow_html=True)

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
        f"""
        <div class="hero-floating-container">
            <div class="hero-brand-header">
                <img src="data:image/png;base64,{LOGO_B64}" class="hero-logo-img" alt="MatchMyResume Logo" />
                <div>
                    <div class="hero-subtitle">Multi-Resume Benchmark &amp; Large Dataset Evaluation</div>
                    <h1 class="large-website-title">Candidate Benchmark</h1>
                </div>
            </div>
            <span class="hero-tag">⚡ Genuine Kaggle Dataset (962 Resumes) • 25 Job Categories • Multi-Tier Benchmarking</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jds_df = get_sample_jds()
    role_options = jds_df["role"].tolist()

    bench_tab1, bench_tab2 = st.tabs(
        [
            "🏆 Controlled 5-Tier Benchmark Pool",
            "🌐 Genuine Kaggle Dataset (962 Resumes / 25 Domains)",
        ]
    )

    with bench_tab1:
        st.markdown("#### Controlled Discrimination Benchmark (5 Qualification Tiers)")
        st.caption(
            "Quickly demonstrates model discrimination across 5 standardized profiles: "
            "Senior AI/ML, M.Sc. Data Science, Entry-Level Graduate, Cross-Domain Web Dev, and IT Support."
        )

        bcol1, bcol2 = st.columns([2, 1])
        with bcol1:
            target_role = st.selectbox(
                "Target Role for 5-Tier Benchmark:",
                options=role_options,
                index=0,
                key="bench_5_role",
            )
        with bcol2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            recompute_5_btn = st.button("🔄 Re-evaluate 5 Candidates", use_container_width=True, key="btn_5_recomp")

        with st.spinner(f"Evaluating 5 candidates against '{target_role}'..."):
            benchmark_results = evaluate_candidates(
                target_role,
                tfidf_weight=norm_tfidf_w,
                skill_weight=norm_skill_w,
            )

        st.plotly_chart(create_candidate_comparison_chart(benchmark_results, top_n=5), use_container_width=True)

        st.markdown("#### Benchmark Comparison Table")
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

        csv_data_5 = benchmark_results.to_csv(index=True)
        st.download_button(
            label="📥 Export 5-Tier Rankings as CSV",
            data=csv_data_5,
            file_name=f"benchmark_5_{target_role.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="dl_5_csv",
        )

    with bench_tab2:
        st.markdown("#### Large-Scale Evaluation: Kaggle Updated Resume Dataset")
        st.caption(
            "Evaluates genuine real-world resumes from the **Kaggle Updated Resume Dataset (962 Resumes across 25 Categories)**. "
            "Filter by category to evaluate domain-specific candidates, or run across the entire population."
        )

        kaggle_categories = ["All Categories (962 Resumes)"] + get_kaggle_categories()

        kcol1, kcol2, kcol3 = st.columns([2, 2, 1])
        with kcol1:
            kaggle_role = st.selectbox(
                "Target Job Description:",
                options=role_options,
                index=0,
                key="kaggle_role_sel",
            )
        with kcol2:
            selected_category = st.selectbox(
                "Filter by Kaggle Category (25 Domains):",
                options=kaggle_categories,
                index=1,  # Default to 'Data Science' or first domain
                key="kaggle_cat_sel",
            )
        with kcol3:
            sample_limit = st.select_slider(
                "Batch Limit:",
                options=[10, 20, 40, 80, 962],
                value=20,
                key="kaggle_limit_sel",
                help="Limits number of resumes evaluated for instant live presentation response.",
            )

        run_kaggle_btn = st.button("🚀 Run Large-Scale Evaluation on Kaggle Dataset", use_container_width=True, key="btn_run_kaggle")

        # Automatically run initial view
        with st.spinner(f"Evaluating {sample_limit} Kaggle resumes from '{selected_category}' against '{kaggle_role}'..."):
            kaggle_results = evaluate_kaggle_candidates(
                kaggle_role,
                category_filter=selected_category,
                limit=sample_limit,
                tfidf_weight=norm_tfidf_w,
                skill_weight=norm_skill_w,
            )

        # Summary Metrics Row
        km1, km2, km3, km4 = st.columns(4)
        with km1:
            st.markdown(
                f"""
                <div class="floating-card">
                    <div class="floating-metric-title">Candidates Evaluated</div>
                    <div class="floating-metric-value">{len(kaggle_results)}</div>
                    <div class="floating-metric-sub">Kaggle Resume Batch</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with km2:
            peak_score = kaggle_results["Composite ATS Score (%)"].max() if not kaggle_results.empty else 0.0
            st.markdown(
                f"""
                <div class="floating-card">
                    <div class="floating-metric-title">Highest Score</div>
                    <div class="floating-metric-value">{peak_score:.1f}%</div>
                    <div class="floating-metric-sub">Top Ranked Candidate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with km3:
            avg_score = kaggle_results["Composite ATS Score (%)"].mean() if not kaggle_results.empty else 0.0
            st.markdown(
                f"""
                <div class="floating-card">
                    <div class="floating-metric-title">Average Match</div>
                    <div class="floating-metric-value">{avg_score:.1f}%</div>
                    <div class="floating-metric-sub">Population Mean</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with km4:
            st.markdown(
                f"""
                <div class="floating-card">
                    <div class="floating-metric-title">Source Dataset</div>
                    <div class="floating-metric-value">962</div>
                    <div class="floating-metric-sub">UpdatedResumeDataSet.csv</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visual Analytics
        kchart1, kchart2 = st.columns([1, 1])
        with kchart1:
            st.markdown("**Top Ranked Candidates Leaderboard**")
            st.plotly_chart(create_candidate_comparison_chart(kaggle_results, top_n=10), use_container_width=True)

        with kchart2:
            st.markdown("**Score Distribution Histogram**")
            st.plotly_chart(create_score_distribution_histogram(kaggle_results), use_container_width=True)

        st.markdown("#### Full Ranked Candidate Leaderboard")
        k_display_cols = [
            "Candidate ID",
            "Domain Category",
            "Composite ATS Score (%)",
            "TF-IDF Similarity (%)",
            "Skill Coverage (%)",
            "ATS Tier",
            "Matched Skills Count",
            "Missing Skills Count",
            "Top Matched Skills",
            "Word Count",
        ]
        st.dataframe(kaggle_results[k_display_cols], use_container_width=True)

        csv_kaggle_data = kaggle_results.to_csv(index=True)
        st.download_button(
            label=f"📥 Export All {len(kaggle_results)} Kaggle Candidate Rankings as CSV",
            data=csv_kaggle_data,
            file_name=f"kaggle_rankings_{selected_category.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="dl_kaggle_csv",
        )

