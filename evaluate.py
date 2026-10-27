"""
evaluate.py - Benchmark evaluation engine for candidate resume comparisons.

Features:
- Loads the standard 5-candidate benchmark (sample_data/resumes.csv).
- Loads the genuine Kaggle Updated Resume Dataset (sample_data/UpdatedResumeDataSet.csv - 962 resumes, 25 categories).
- Evaluates candidates against any target job role.
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
KAGGLE_CSV = DATA_DIR / "UpdatedResumeDataSet.csv"


# ---------------------------------------------------------------------------
# Core Benchmark Loaders (5 Standard Tiers)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Kaggle 962-Resume Dataset Loaders
# ---------------------------------------------------------------------------

def load_kaggle_resumes(path: Path = KAGGLE_CSV) -> pd.DataFrame:
    """
    Load genuine Kaggle Updated Resume Dataset (962 resumes, 25 categories).
    Columns: Category, Resume.
    """
    if not path.exists():
        raise FileNotFoundError(f"Kaggle UpdatedResumeDataSet.csv not found at: {path}")
    df = pd.read_csv(path)
    if "Category" not in df.columns or "Resume" not in df.columns:
        raise ValueError(f"UpdatedResumeDataSet.csv must contain 'Category' and 'Resume' columns, got {list(df.columns)}")
    df["Resume"] = df["Resume"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("Unknown").astype(str)
    # Assign readable candidate IDs
    df["Candidate_ID"] = [f"Candidate #{i+1} ({cat})" for i, cat in enumerate(df["Category"])]
    return df


def get_kaggle_categories(path: Path = KAGGLE_CSV) -> List[str]:
    """Return sorted unique list of the 25 Kaggle job categories."""
    df = load_kaggle_resumes(path)
    return sorted(df["Category"].unique().tolist())


# ---------------------------------------------------------------------------
# Candidate Evaluation Engine (5 Benchmark Candidates)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Large Dataset Evaluation Engine (Kaggle 962 Resumes)
# ---------------------------------------------------------------------------

def evaluate_kaggle_candidates(
    jd_role_or_text: str,
    *,
    category_filter: Optional[str] = None,
    limit: Optional[int] = None,
    is_custom_text: bool = False,
    tfidf_weight: float = 0.5,
    skill_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate resumes from the genuine Kaggle Updated Resume Dataset.

    Parameters
    ----------
    jd_role_or_text : str
        Target job role or description.
    category_filter : str | None
        Specific category to filter by (e.g. 'Data Science', 'Java Developer').
        If None or 'All Categories', evaluates across all domains.
    limit : int | None
        Upper bound on candidates evaluated (for fast live demonstration).
    tfidf_weight : float, default=0.5
        Weight for TF-IDF cosine similarity.
    skill_weight : float, default=0.5
        Weight for skill coverage.

    Returns
    -------
    pd.DataFrame
        Ranked leaderboard of Kaggle candidates.
    """
    kaggle_df = load_kaggle_resumes()

    if category_filter and category_filter != "All Categories (962 Resumes)":
        filtered_df = kaggle_df[kaggle_df["Category"].str.lower() == category_filter.lower()]
        if filtered_df.empty:
            filtered_df = kaggle_df
    else:
        filtered_df = kaggle_df

    if limit is not None and limit > 0:
        filtered_df = filtered_df.head(limit)

    if is_custom_text:
        jd_text = jd_role_or_text
        role_label = "Custom Job Description"
    else:
        jd_df = load_job_descriptions()
        matches = jd_df[jd_df["role"].str.lower() == jd_role_or_text.lower()]
        if matches.empty:
            jd_text = jd_df.iloc[0]["description"]
            role_label = jd_df.iloc[0]["role"]
        else:
            jd_text = matches.iloc[0]["description"]
            role_label = matches.iloc[0]["role"]

    rows: List[dict] = []
    for _, row in filtered_df.iterrows():
        cand_id = row["Candidate_ID"]
        domain = row["Category"]
        resume_text = row["Resume"]

        match_res = match_resume_to_jd(
            resume_text,
            jd_text,
            tfidf_weight=tfidf_weight,
            skill_weight=skill_weight,
        )

        rows.append({
            "Candidate ID": cand_id,
            "Domain Category": domain,
            "Composite ATS Score (%)": match_res.composite_pct,
            "TF-IDF Similarity (%)": match_res.tfidf_pct,
            "Skill Coverage (%)": match_res.skill_coverage_pct,
            "ATS Tier": match_res.ats_label,
            "Matched Skills Count": len(match_res.matched_skills),
            "Missing Skills Count": len(match_res.missing_skills),
            "Top Matched Skills": ", ".join(sorted(match_res.matched_skills)[:4]) if match_res.matched_skills else "None",
            "Word Count": match_res.resume_word_count,
        })

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("Composite ATS Score (%)", ascending=False).reset_index(drop=True)
    results_df.index += 1  # 1-based rank
    results_df.index.name = "Rank"
    return results_df


if __name__ == "__main__":
    print("Testing Kaggle evaluation on 'Data Science' category...")
    df_ds = evaluate_kaggle_candidates("Data Scientist", category_filter="Data Science", limit=10)
    print(df_ds[["Candidate ID", "Domain Category", "Composite ATS Score (%)", "TF-IDF Similarity (%)", "Skill Coverage (%)"]].to_string())
