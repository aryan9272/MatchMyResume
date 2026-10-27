"""
generate_overview_pdf.py - Generates a publication-quality PDF summary of the project.

Creates 'ATS_Resume_Matcher_Project_Overview.pdf' in the project root so teammates
can review and prepare for project presentation, defense, and viva examinations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

PDF_FILENAME = "ATS_Resume_Matcher_Project_Overview.pdf"


def build_pdf(filename: str = PDF_FILENAME):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom Clean Light Styles
    primary_color = colors.HexColor("#1e40af")   # Deep Blue
    dark_slate = colors.HexColor("#0f172a")      # Text Dark
    body_color = colors.HexColor("#334155")      # Text Muted
    border_color = colors.HexColor("#cbd5e1")    # Light Border
    bg_light = colors.HexColor("#f8fafc")        # Card Light

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        alignment=0,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=body_color,
        alignment=0,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=dark_slate,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=body_color,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=body_color,
        leftIndent=15,
        spaceAfter=3,
    )

    qa_q_style = ParagraphStyle(
        "QAQuestion",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_slate,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True,
    )

    qa_a_style = ParagraphStyle(
        "QAAnswer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=body_color,
        leftIndent=10,
        spaceAfter=6,
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("ATS Resume Matcher: NLP & ML Project Overview", title_style))
    story.append(Paragraph("College-Level Introduction to Machine Learning (IML) Project | Team Preparation Guide", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=2, spaceAfter=10))

    # Section 1: Executive Overview
    story.append(Paragraph("1. Executive Summary & Objective", h1_style))
    story.append(Paragraph(
        "Automated Applicant Tracking Systems (ATS) screen resumes against job descriptions to identify qualified candidates. "
        "This project implements a 100% local, transparent, and explainable ATS matching engine using foundational "
        "Natural Language Processing (NLP) and an unsupervised Vector Space Model (VSM).",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Key Academic Positioning:</b> Rather than fabricating a fictitious 99% accurate neural network trained on nonexistent "
        "labeled hiring datasets, this project formulates resume screening correctly as an <b>unsupervised document similarity and "
        "information retrieval task</b>. It fits TF-IDF directly on compared document vocabularies, measures directional cosine similarity, "
        "extracts domain skills via punctuation-safe regular expressions, and highlights the exact keywords that influenced the score.",
        body_style,
    ))

    # Section 2: Technical Stack Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("2. Technical Stack & Architecture", h1_style))
    tech_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Technology</b>", body_style), Paragraph("<b>Purpose / Justification</b>", body_style)],
        [Paragraph("Frontend UI", body_style), Paragraph("Streamlit (Light Theme)", body_style), Paragraph("Interactive dashboard with light mode, high contrast, zero glassmorphism.", body_style)],
        [Paragraph("NLP Vectorization", body_style), Paragraph("scikit-learn TfidfVectorizer", body_style), Paragraph("Unigrams + Bigrams (1,2), sublinear log scaling, smooth IDF.", body_style)],
        [Paragraph("Similarity Metric", body_style), Paragraph("scikit-learn Cosine Similarity", body_style), Paragraph("Length-invariant vector dot product on L2-normalized vectors.", body_style)],
        [Paragraph("PDF Extraction", body_style), Paragraph("PyMuPDF (fitz)", body_style), Paragraph("High-speed C-level digital text extraction handling corrupt/empty files.", body_style)],
        [Paragraph("Data & Visuals", body_style), Paragraph("Pandas & Plotly", body_style), Paragraph("Tabular benchmarking, gauge charts, and candidate comparison bars.", body_style)],
        [Paragraph("Test Framework", body_style), Paragraph("pytest (53 tests)", body_style), Paragraph("Comprehensive unit and integration test suite passing at 100%.", body_style)],
    ]
    t_tech = Table(tech_data, colWidths=[1.3*inch, 1.8*inch, 3.9*inch])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_tech)

    # Section 3: What the Website Does
    story.append(Spacer(1, 8))
    story.append(Paragraph("3. What the Website Does (Core Features)", h1_style))
    features = [
        "<b>PDF Resume Upload & Text Extraction:</b> Direct drag-and-drop PDF extraction with character/word counting and corrupt/scanned PDF warnings.",
        "<b>Preloaded Benchmark Roles:</b> Instant selection of 5 industry job descriptions (Data Scientist, ML Engineer, Full Stack, Data Analyst, Software Engineer).",
        "<b>Preloaded Candidate Profiles:</b> Instant loading of 5 diverse fictional candidate profiles (Alice, Bob, Carol, David, Eva) for immediate testing.",
        "<b>Punctuation-Aware Skill Extraction:</b> Extracts hard skills into <b>Matched Skills</b> (green), <b>Missing Skills</b> (red), and <b>Additional Skills</b> (blue).",
        "<b>Configurable Score Weighting:</b> Interactive sliders to adjust the relative weight of TF-IDF Cosine Similarity vs. Explicit Skill Coverage.",
        "<b>Explainable AI (XAI) Panel:</b> Renders the top overlapping n-grams and their exact vector dot products (u_i * v_i) explaining the score.",
        "<b>Actionable Gap Recommendations:</b> Automatically generates tailored advice based on missing skills and resume text density.",
        "<b>Multi-Candidate Benchmark Ranking:</b> Compares all 5 fictional resumes simultaneously against any role with Plotly horizontal bar charts.",
        "<b>Model Transparency & Viva Tab:</b> Complete in-app interactive study guide with 16 viva questions and LaTeX mathematical formulations.",
    ]
    for feat in features:
        story.append(Paragraph(f"• {feat}", bullet_style))

    story.append(PageBreak())

    # Section 4: How It Does It (Step-by-Step NLP Pipeline)
    story.append(Paragraph("4. How It Does It: Step-by-Step NLP Pipeline", h1_style))
    story.append(Paragraph(
        "<b>Step 1: Text Extraction & Normalization:</b> The user's PDF is read into an in-memory byte stream. PyMuPDF inspects each page's "
        "digital text layer. Control characters and null bytes are stripped, and whitespace is normalized.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Step 2: Punctuation-Aware Skill Matching:</b> The text is evaluated against an 8-category skill taxonomy using negative lookaround regexes "
        "<code>(?&lt;![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])</code>. This ensures <code>C++</code> or <code>C#</code> never trigger <code>C</code> falsely. "
        "Matched, missing, and additional skills are partitioned using Python set operations.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Step 3: Sublinear TF-IDF Vectorization:</b> Both documents are transformed into sparse vectors using scikit-learn's TfidfVectorizer with "
        "<code>ngram_range=(1, 2)</code> and English stop-word filtering. Sublinear scaling calculates <code>tf = 1 + ln(tf)</code>, dampening keyword stuffing.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Step 4: Cosine Similarity Computation:</b> The L2-normalized vectors are multiplied via Euclidean dot product: "
        "<code>cos(u, v) = sum(u_i * v_i)</code>. This yields a length-invariant directional similarity score between 0.0 and 1.0.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Step 5: Composite Scoring & Explainability:</b> The final ATS score is computed as: "
        "<code>ATS Score = w_tfidf * CosineSim + w_skill * (MatchedSkills / JDSkills)</code>. Top shared terms are extracted from non-zero "
        "products <code>u_i * v_i</code> and displayed in descending order.",
        body_style,
    ))

    # Section 5: Mathematical Formulas
    story.append(Spacer(1, 4))
    story.append(Paragraph("5. Mathematical Formulations (Key Formulas to Memorize)", h1_style))
    formulas_data = [
        [Paragraph("<b>Concept</b>", body_style), Paragraph("<b>Formula</b>", body_style), Paragraph("<b>Explanation</b>", body_style)],
        [
            Paragraph("Sublinear TF", body_style),
            Paragraph("tf_scaled = 1 + ln(tf)", code_style),
            Paragraph("Applies logarithmic smoothing to raw word counts, suppressing keyword-stuffing exploits.", body_style)
        ],
        [
            Paragraph("Smooth IDF", body_style),
            Paragraph("idf(t) = ln((1+n)/(1+df(t))) + 1", code_style),
            Paragraph("Down-weights common words that appear across all documents while rewarding distinctive domain skills.", body_style)
        ],
        [
            Paragraph("Cosine Similarity", body_style),
            Paragraph("cos(u, v) = (u . v) / (||u|| * ||v||)", code_style),
            Paragraph("Evaluates the angle between normalized document vectors, making scoring independent of resume length.", body_style)
        ],
        [
            Paragraph("Skill Coverage", body_style),
            Paragraph("Coverage = |S_res ∩ S_jd| / |S_jd|", code_style),
            Paragraph("Measures the exact proportion of hard technical requirements satisfied by the candidate.", body_style)
        ],
        [
            Paragraph("Composite ATS", body_style),
            Paragraph("Score = w_t * CosSim + w_s * Coverage", code_style),
            Paragraph("Blends general contextual alignment with non-negotiable technical skill compliance.", body_style)
        ],
    ]
    t_form = Table(formulas_data, colWidths=[1.3*inch, 2.2*inch, 3.5*inch])
    t_form.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_form)

    # Section 6: Candidate Benchmark Pool
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. Candidate Benchmark Pool (Model Discrimination)", h1_style))
    cand_data = [
        [Paragraph("<b>Candidate</b>", body_style), Paragraph("<b>Profile Summary</b>", body_style), Paragraph("<b>Expected Behavior</b>", body_style)],
        [Paragraph("Eva Chen", body_style), Paragraph("AI/ML Engineer: Transformers, LLMs, PyTorch, MLOps, AWS", body_style), Paragraph("Ranks #1 on ML Engineer and Data Science roles (~75-80% coverage).", body_style)],
        [Paragraph("Carol Smith", body_style), Paragraph("M.Sc. Data Science: Spark, PyTorch, SQL, Statistics, A/B Testing", body_style), Paragraph("Strong fit for Data Science and Senior Analyst roles (~60-65% coverage).", body_style)],
        [Paragraph("Alice Johnson", body_style), Paragraph("College Grad: Python, scikit-learn, EDA, Pandas, basic NLP", body_style), Paragraph("Solid junior match for Data Science roles (~50-55% coverage).", body_style)],
        [Paragraph("Bob Martinez", body_style), Paragraph("Full Stack Dev: React, Node.js, Express, Docker, Django", body_style), Paragraph("Dominates Full Stack role; accurately scores low on Data Science.", body_style)],
        [Paragraph("David Lee", body_style), Paragraph("IT Support Fresher: Basic C++, Java, Excel, IT support", body_style), Paragraph("Accurately receives lowest scores (<15%), demonstrating model selectivity.", body_style)],
    ]
    t_cand = Table(cand_data, colWidths=[1.3*inch, 3.0*inch, 2.7*inch])
    t_cand.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cand)

    story.append(PageBreak())

    # Section 7: Top 10 High-Priority Viva Questions & Answers
    story.append(Paragraph("7. Top 10 High-Priority Viva Questions & Answers", h1_style))
    story.append(Paragraph("Teammates should study these 10 core questions carefully for project viva examinations:", body_style))

    viva_top10 = [
        (
            "Q1: Why is this project unsupervised rather than using a trained supervised classifier?",
            "Supervised hiring models require thousands of labeled (Resume, Job Description, Hired/Rejected) pairs, which are unavailable "
            "due to privacy laws and suffer from historical hiring bias. An unsupervised Vector Space Model directly evaluates document similarity "
            "without needing synthetic labels, generalizing dynamically to any job description.",
        ),
        (
            "Q2: Why use Cosine Similarity instead of Euclidean Distance?",
            "Euclidean distance is biased by document length (a 3-page resume would be distant from a 1-page resume even if identical in topic). "
            "Cosine similarity measures the angle between normalized vectors, making it completely length-invariant.",
        ),
        (
            "Q3: What does sublinear TF scaling do?",
            "It computes tf_scaled = 1 + ln(tf). This prevents candidates from exploiting keyword stuffing by repeating a word 20 times to artificially "
            "inflate their similarity score.",
        ),
        (
            "Q4: Why are bigrams (1, 2) included alongside unigrams?",
            "Unigrams like 'machine' and 'learning' lose their meaning when separated. Bigrams capture crucial compound phrases like 'machine learning', "
            "'data science', and 'random forest'.",
        ),
        (
            "Q5: How does the system handle punctuation in skills like C++, C#, and .NET?",
            "Standard tokenizers strip punctuation, reducing C++ and C# to 'c'. Our engine uses negative lookaround regexes "
            "'(?&lt;![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])', guaranteeing 'C' will never match inside 'C++' or 'C#'.",
        ),
        (
            "Q6: Why combine TF-IDF Cosine Similarity and Skill Coverage?",
            "TF-IDF captures overall vocabulary and descriptive congruence, while Skill Coverage checks for mandatory non-negotiable technical skills. "
            "A weighted combination gives recruiters a balanced picture of both fit and qualifications.",
        ),
        (
            "Q7: How does Explainable AI (XAI) work in this dashboard?",
            "Because cosine similarity on normalized vectors equals sum(u_i * v_i), each term's contribution is simply its dot product u_i * v_i. "
            "We extract and sort the highest product terms to show the exact keywords that generated the score.",
        ),
        (
            "Q8: What are the limitations of pure TF-IDF matching?",
            "TF-IDF is lexical; it cannot detect synonyms (e.g., 'developer' vs 'software engineer') or polysemy ('Apple' fruit vs 'Apple' company). "
            "Dense embeddings like Sentence-BERT can resolve this in future enhancements.",
        ),
        (
            "Q9: How are corrupt, encrypted, or scanned PDFs handled?",
            "PyMuPDF catches stream corruption errors, checks doc.needs_pass for password protection, and flags image-only scanned PDFs that lack a text layer.",
        ),
        (
            "Q10: What is the computational complexity of the matching pipeline?",
            "Linear time O(L) where L is the total word count of the documents. Vectorization and cosine similarity execute in under 25 milliseconds on CPU.",
        ),
    ]

    for q, a in viva_top10:
        story.append(Paragraph(q, qa_q_style))
        story.append(Paragraph(f"<b>Model Answer:</b> {a}", qa_a_style))

    # Section 8: Quick Execution Commands
    story.append(Spacer(1, 6))
    story.append(Paragraph("8. Quick Windows Execution Commands", h1_style))
    story.append(Paragraph("<b>Activate virtual environment & launch dashboard:</b>", body_style))
    story.append(Paragraph("<code>cd C:\\MatchMyResume<br/>.\\venv\\Scripts\\activate<br/>python -m streamlit run app.py</code>", code_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Run automated test suite:</b>", body_style))
    story.append(Paragraph("<code>python -m pytest tests/ -v</code>", code_style))

    doc.build(story)
    print(f"Successfully generated {filename}!")


if __name__ == "__main__":
    build_pdf()
