"""
evaluate.py - Benchmark evaluation engine for candidate resume comparisons.

Features:
- Single Source of Truth: Loads the genuine Kaggle Updated Resume Dataset
  (sample_data/UpdatedResumeDataSet.csv - 962 resumes across 25 job domains).
- Evaluates candidates against target job descriptions.
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
JD_CSV = DATA_DIR / "job_descriptions.csv"
KAGGLE_CSV = DATA_DIR / "UpdatedResumeDataSet.csv"


# ---------------------------------------------------------------------------
# Core Dataset Loaders (Kaggle Updated Resume Dataset - 962 Resumes)
# ---------------------------------------------------------------------------

def load_resumes(path: Path = KAGGLE_CSV) -> pd.DataFrame:
    """
    Load genuine Kaggle Updated Resume Dataset (962 resumes, 25 categories).
    Provides 'name' and 'text' columns for standard compatibility, as well
    as 'Category' and 'Candidate_ID'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Resume dataset not found at: {path}")
    df = pd.read_csv(path)
    if "Category" not in df.columns or "Resume" not in df.columns:
        raise ValueError(
            f"UpdatedResumeDataSet.csv must contain 'Category' and 'Resume' columns, got {list(df.columns)}"
        )
    df["Resume"] = df["Resume"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("Unknown").astype(str)

    # Standard compatibility columns
    df["text"] = df["Resume"]
    df["name"] = [f"Candidate #{i+1} ({cat})" for i, cat in enumerate(df["Category"])]
    df["Candidate_ID"] = df["name"]
    return df


def load_kaggle_resumes(path: Path = KAGGLE_CSV) -> pd.DataFrame:
    """Alias for load_resumes returning the Kaggle dataset."""
    return load_resumes(path)


def get_kaggle_categories(path: Path = KAGGLE_CSV) -> List[str]:
    """Return sorted unique list of the 25 Kaggle job categories."""
    df = load_resumes(path)
    return sorted(df["Category"].unique().tolist())


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
# Candidate Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_candidates(
    jd_role_or_text: str,
    *,
    category_filter: Optional[str] = None,
    limit: Optional[int] = 25,
    is_custom_text: bool = False,
    tfidf_weight: float = 0.5,
    skill_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate resumes from the genuine Kaggle Updated Resume Dataset against a job description.

    Parameters
    ----------
    jd_role_or_text : str
        Target job role or description.
    category_filter : str | None
        Specific domain category to filter (e.g. 'Data Science', 'Java Developer').
    limit : int | None, default=25
        Number of candidates to evaluate (for fast response).
    is_custom_text : bool, default=False
        If True, jd_role_or_text is treated directly as the job description body.
    tfidf_weight : float, default=0.5
        Weight for TF-IDF cosine similarity.
    skill_weight : float, default=0.5
        Weight for skill coverage.

    Returns
    -------
    pd.DataFrame
        Ranked leaderboard of evaluated candidates.
    """
    resumes_df = load_resumes()

    if category_filter and category_filter not in ("All Categories", "All Categories (962 Resumes)"):
        filtered_df = resumes_df[resumes_df["Category"].str.lower() == category_filter.lower()]
        if filtered_df.empty:
            filtered_df = resumes_df
    else:
        filtered_df = resumes_df

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
        resume_text = row["text"]

        match_res = match_resume_to_jd(
            resume_text,
            jd_text,
            tfidf_weight=tfidf_weight,
            skill_weight=skill_weight,
        )

        rows.append({
            "Candidate": cand_id,
            "Candidate ID": cand_id,
            "Domain Category": domain,
            "Target Role": role_label,
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


def evaluate_kaggle_candidates(
    jd_role_or_text: str,
    *,
    category_filter: Optional[str] = None,
    limit: Optional[int] = None,
    is_custom_text: bool = False,
    tfidf_weight: float = 0.5,
    skill_weight: float = 0.5,
) -> pd.DataFrame:
    """Alias for evaluate_candidates on Kaggle dataset."""
    return evaluate_candidates(
        jd_role_or_text,
        category_filter=category_filter,
        limit=limit,
        is_custom_text=is_custom_text,
        tfidf_weight=tfidf_weight,
        skill_weight=skill_weight,
    )


if __name__ == "__main__":
    print("Testing evaluation on genuine Kaggle dataset ('Data Science')...")
    df_ds = evaluate_candidates("Data Scientist", category_filter="Data Science", limit=5)
    print(df_ds[["Candidate ID", "Domain Category", "Composite ATS Score (%)", "TF-IDF Similarity (%)", "Skill Coverage (%)"]].to_string())
