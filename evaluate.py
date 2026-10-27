"""
evaluate.py - Benchmark evaluation engine for candidate resume comparisons.

Features:
- Loads sample resumes and job descriptions from local CSV files.
- Evaluates all candidates against any target job role.
- Computes TF-IDF cosine similarity, Skill Coverage, and Composite ATS scores.
- Ranks candidates with full metric breakdowns.

No external APIs, cloud services, or paid subscriptions required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from matcher import match_resume_to_jd

DATA_DIR = Path(__file__).parent / "sample_data"
RESUMES_CSV = DATA_DIR / "resumes.csv"
JD_CSV = DATA_DIR / "job_descriptions.csv"


def load_resumes(path: Path = RESUMES_CSV) -> pd.DataFrame:
    """Load sample resumes from CSV with validation."""
    if not path.exists():
        raise FileNotFoundError(f"Resumes CSV file not found at: {path}")
    df = pd.read_csv(path)
    required = {"name", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"resumes.csv must contain columns: {required}, got {list(df.columns)}")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_job_descriptions(path: Path = JD_CSV) -> pd.DataFrame:
    """Load sample job descriptions from CSV with validation."""
    if not path.exists():
        raise FileNotFoundError(f"Job descriptions CSV file not found at: {path}")
    df = pd.read_csv(path)
    required = {"role", "description"}
    if not required.issubset(df.columns):
        raise ValueError(f"job_descriptions.csv must contain columns: {required}, got {list(df.columns)}")
    df["description"] = df["description"].fillna("").astype(str)
    return df


def get_available_roles(path: Path = JD_CSV) -> List[str]:
    """Retrieve list of available predefined roles from job descriptions CSV."""
    df = load_job_descriptions(path)
    return df["role"].tolist()


def evaluate_candidates(
    jd_role_or_text: str,
    *,
    is_custom_text: bool = False,
    tfidf_weight: float = 0.5,
    skill_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate all candidates in sample_data/resumes.csv against a job description.

    Parameters
    ----------
    jd_role_or_text : str
        Either a role name from job_descriptions.csv or custom JD text.
    is_custom_text : bool, default=False
        If True, jd_role_or_text is treated directly as the job description body.
    tfidf_weight : float, default=0.5
        Weight for TF-IDF cosine similarity.
    skill_weight : float, default=0.5
        Weight for skill coverage score.

    Returns
    -------
    pd.DataFrame
        Ranked candidate evaluation results.
    """
    resumes_df = load_resumes()

    if is_custom_text:
        jd_text = jd_role_or_text
        role_label = "Custom Job Description"
    else:
        jd_df = load_job_descriptions()
        matches = jd_df[jd_df["role"].str.lower() == jd_role_or_text.lower()]
        if matches.empty:
            # Fallback: if role not found, use first row
            jd_text = jd_df.iloc[0]["description"]
            role_label = jd_df.iloc[0]["role"]
        else:
            jd_text = matches.iloc[0]["description"]
            role_label = matches.iloc[0]["role"]

    rows: List[dict] = []
    for _, row in resumes_df.iterrows():
        candidate_name = row["name"]
        resume_text = row["text"]

        match_res = match_resume_to_jd(
            resume_text,
            jd_text,
            tfidf_weight=tfidf_weight,
            skill_weight=skill_weight,
        )

        rows.append({
            "Candidate": candidate_name,
            "Target Role": role_label,
            "Composite ATS Score (%)": match_res.composite_pct,
            "TF-IDF Similarity (%)": match_res.tfidf_pct,
            "Skill Coverage (%)": match_res.skill_coverage_pct,
            "ATS Tier": match_res.ats_label,
            "Matched Skills Count": len(match_res.matched_skills),
            "Missing Skills Count": len(match_res.missing_skills),
            "Top Matched Skills": ", ".join(sorted(match_res.matched_skills)[:5]) if match_res.matched_skills else "None",
            "Word Count": match_res.resume_word_count,
        })

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("Composite ATS Score (%)", ascending=False).reset_index(drop=True)
    results_df.index += 1  # 1-based rank
    results_df.index.name = "Rank"
    return results_df


if __name__ == "__main__":
    df = evaluate_candidates("Data Scientist")
    print("=== Candidate Benchmark Ranking for 'Data Scientist' ===")
    print(df[["Candidate", "Composite ATS Score (%)", "TF-IDF Similarity (%)", "Skill Coverage (%)", "ATS Tier"]].to_string())
