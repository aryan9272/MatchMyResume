"""
tests/test_matcher.py - Pytest test suite for ATS Resume Matcher.

Covers:
- skills.py: Skill extraction, punctuation awareness (C vs C++, C#, .NET, Node.js, CI/CD), categorization
- matcher.py: TF-IDF cosine similarity, vocabulary metrics, explainable AI terms, score bounds, edge cases
- pdf_parser.py: Empty streams, corrupt files, nonexistent paths, valid PDF extraction
- evaluate.py: CSV loading, benchmark ranking, candidate evaluation

Execute via:
    python -m pytest tests/ -v
"""

import io
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matcher import (
    compute_tfidf_similarity,
    match_resume_to_jd,
    MatchResult,
)
from pdf_parser import extract_text_from_pdf, ParseResult
from skills import extract_skills, categorize_skills, get_skill_category
from evaluate import (
    load_resumes,
    load_job_descriptions,
    evaluate_candidates,
    load_kaggle_resumes,
    get_kaggle_categories,
    evaluate_kaggle_candidates,
)

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ===========================================================================
# Fixtures & Samples
# ===========================================================================

SAMPLE_RESUME = """
Alice Johnson - Data Scientist
Skills: Python, Machine Learning, scikit-learn, TF-IDF, Pandas, NumPy,
TensorFlow, NLP, BERT, XGBoost, SQL, Git, Matplotlib, Seaborn, Jupyter Notebook,
Statistical Analysis, Feature Engineering, Data Cleaning, Communication, Teamwork.
Experience: Built sentiment analysis model using BERT and NLP achieving 91% accuracy.
Performed EDA on 500K rows using Pandas. Used XGBoost for churn prediction.
"""

SAMPLE_JD = """
Data Scientist role requirements:
Python, R, Machine Learning, scikit-learn, TensorFlow, PyTorch, XGBoost,
NLP, BERT, TF-IDF, Pandas, NumPy, SQL, PostgreSQL, Feature Engineering,
Data Cleaning, Statistical Analysis, Matplotlib, Seaborn, Plotly,
Git, GitHub, Docker, Communication, Problem Solving, Analytical Skills.
"""

UNRELATED_TEXT_A = "I love cooking pasta and baking artisan sourdough bread every weekend."
UNRELATED_TEXT_B = "The history of ancient Roman architecture and civil engineering aqueducts."


# ===========================================================================
# skills.py Tests
# ===========================================================================

class TestSkillExtraction:
    """Tests for skills.extract_skills()"""

    def test_basic_skill_found(self):
        skills = extract_skills("Experience with Python and Machine Learning")
        assert "Python" in skills
        assert "Machine Learning" in skills

    def test_multiple_skills_found(self):
        text = "Proficient in Python, Pandas, NumPy, scikit-learn, and SQL"
        skills = extract_skills(text)
        assert len(skills) >= 4
        assert "Python" in skills
        assert "Pandas" in skills
        assert "SQL" in skills

    def test_case_insensitive(self):
        assert "Python" in extract_skills("PYTHON programming")
        assert "Python" in extract_skills("python programming")
        assert "Python" in extract_skills("PyThOn developer")

    def test_empty_text_returns_empty_set(self):
        assert extract_skills("") == set()

    def test_none_text_returns_empty_set(self):
        assert extract_skills(None) == set()

    def test_skill_not_in_garbage_text(self):
        skills = extract_skills("Sunny morning walk in the botanical garden park.")
        assert "Python" not in skills
        assert "Machine Learning" not in skills

    def test_tfidf_as_skill(self):
        skills = extract_skills("Used TF-IDF and N-grams for feature extraction.")
        assert "TF-IDF" in skills

    def test_sql_found(self):
        assert "SQL" in extract_skills("Skilled in complex SQL queries and relational schemas")

    def test_agile_soft_skill(self):
        assert "Agile" in extract_skills("Collaborated in bi-weekly Agile sprints.")

    def test_c_plus_plus_does_not_falsely_match_c(self):
        """Crucial punctuation check: C++ must NOT trigger C falsely."""
        skills = extract_skills("Developed high-performance game engine using C++.")
        assert "C++" in skills
        assert "C" not in skills, "C was falsely matched when text only contained C++"

    def test_c_sharp_does_not_falsely_match_c(self):
        """Crucial punctuation check: C# must NOT trigger C falsely."""
        skills = extract_skills("Built enterprise desktop application using C#.")
        assert "C#" in skills
        assert "C" not in skills, "C was falsely matched when text only contained C#"

    def test_isolated_c_language_matches(self):
        """C alone should match when genuinely present."""
        skills = extract_skills("Embedded firmware development using C programming language.")
        assert "C" in skills

    def test_both_c_and_c_plus_plus_match(self):
        """Both C and C++ match when both are present."""
        skills = extract_skills("Experienced in both C and C++ systems programming.")
        assert "C" in skills
        assert "C++" in skills

    def test_dotnet_extraction(self):
        """Extracts .NET correctly without being broken by leading dot."""
        skills_dot = extract_skills("Developed backend services in .NET framework.")
        skills_alias = extract_skills("Developed backend services in dotnet.")
        assert ".NET" in skills_dot
        assert ".NET" in skills_alias

    def test_node_js_and_ci_cd(self):
        """Extracts Node.js and CI/CD cleanly."""
        skills = extract_skills("Fullstack app using Node.js with CI/CD deployment pipelines.")
        assert "Node.js" in skills
        assert "CI/CD" in skills

    def test_alias_sklearn_maps_to_scikit_learn(self):
        """Extracts 'scikit-learn' when 'sklearn' is written."""
        skills = extract_skills("Model trained using sklearn and XGBoost.")
        assert "scikit-learn" in skills


class TestSkillCategorization:
    """Tests for skills.categorize_skills() and get_skill_category()"""

    def test_python_category(self):
        cat = get_skill_category("Python")
        assert cat == "Programming Languages"

    def test_unknown_skill_category(self):
        cat = get_skill_category("Quantum Teleportation Engine")
        assert cat == "Unknown"

    def test_categorize_returns_dict(self):
        result = categorize_skills({"Python", "Pandas", "Agile"})
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "Programming Languages" in result
        assert "Data Science & Analytics" in result
        assert "Soft Skills" in result

    def test_empty_set_categorize(self):
        result = categorize_skills(set())
        assert result == {}


# ===========================================================================
# matcher.py Tests
# ===========================================================================

class TestTfidfSimilarity:
    """Tests for matcher.compute_tfidf_similarity()"""

    def test_identical_docs_high_similarity(self):
        text = "Python Machine Learning Data Science scikit-learn"
        score, vocab, terms = compute_tfidf_similarity(text, text)
        assert score > 0.95, f"Expected > 0.95, got {score}"
        assert vocab > 0
        assert len(terms) > 0

    def test_unrelated_docs_low_similarity(self):
        score, vocab, terms = compute_tfidf_similarity(UNRELATED_TEXT_A, UNRELATED_TEXT_B)
        assert score < 0.25, f"Expected < 0.25, got {score}"

    def test_similar_docs_mid_range(self):
        score, vocab, terms = compute_tfidf_similarity(SAMPLE_RESUME, SAMPLE_JD)
        assert 0.0 < score <= 1.0

    def test_empty_inputs_returns_zero(self):
        score, vocab, terms = compute_tfidf_similarity("", "")
        assert score == 0.0
        assert vocab == 0
        assert terms == []

    def test_score_range_zero_to_one(self):
        score, _, _ = compute_tfidf_similarity(SAMPLE_RESUME, SAMPLE_JD)
        assert 0.0 <= score <= 1.0

    def test_returns_vocab_size(self):
        _, vocab, _ = compute_tfidf_similarity(SAMPLE_RESUME, SAMPLE_JD)
        assert isinstance(vocab, int) and vocab >= 0

    def test_shared_terms_contain_overlapping_keywords(self):
        _, _, terms = compute_tfidf_similarity(SAMPLE_RESUME, SAMPLE_JD)
        term_strings = [t[0] for t in terms]
        # Should contain common tech terms like python, pandas, or scikit
        assert any("python" in t or "learn" in t or "pandas" in t or "sql" in t for t in term_strings)


class TestMatchResult:
    """Tests for matcher.match_resume_to_jd()"""

    def test_full_match_result_structure(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert isinstance(result, MatchResult)
        assert result.resume_word_count > 0
        assert result.jd_word_count > 0

    def test_scores_in_valid_range(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert 0.0 <= result.tfidf_score <= 1.0
        assert 0.0 <= result.skill_coverage_score <= 1.0
        assert 0.0 <= result.composite_score <= 1.0

    def test_matched_skills_subset_of_both(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert result.matched_skills.issubset(result.resume_skills)
        assert result.matched_skills.issubset(result.jd_skills)

    def test_missing_skills_not_in_resume(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert result.missing_skills.isdisjoint(result.resume_skills)

    def test_additional_skills_not_in_jd(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert result.additional_skills.isdisjoint(result.jd_skills)

    def test_skill_union_correct(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert result.matched_skills | result.missing_skills == result.jd_skills

    def test_empty_resume_returns_error(self):
        result = match_resume_to_jd("", SAMPLE_JD)
        assert result.error != ""

    def test_empty_jd_returns_error(self):
        result = match_resume_to_jd(SAMPLE_RESUME, "")
        assert result.error != ""

    def test_recommendations_list(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0

    def test_ats_label_string(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert isinstance(result.ats_label, str)
        assert len(result.ats_label) > 0

    def test_processing_time_positive(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert result.processing_time_ms >= 0.0

    def test_composite_is_weighted_blend(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD, tfidf_weight=0.6, skill_weight=0.4)
        expected = 0.6 * result.tfidf_score + 0.4 * result.skill_coverage_score
        assert abs(result.composite_score - expected) < 1e-5

    def test_high_similarity_resume(self):
        result = match_resume_to_jd(SAMPLE_JD, SAMPLE_JD)
        assert result.tfidf_score > 0.9
        assert result.skill_coverage_score == 1.0

    def test_low_similarity_unrelated(self):
        result = match_resume_to_jd(UNRELATED_TEXT_A, SAMPLE_JD)
        assert result.composite_pct < 35.0


# ===========================================================================
# pdf_parser.py Tests
# ===========================================================================

class TestPdfParser:
    """Tests for pdf_parser.extract_text_from_pdf()"""

    def test_empty_bytes_fails_gracefully(self):
        result = extract_text_from_pdf(io.BytesIO(b""))
        assert result.success is False
        assert result.error != ""

    def test_corrupt_bytes_fails_gracefully(self):
        result = extract_text_from_pdf(io.BytesIO(b"this is corrupt binary data not a valid pdf 12345"))
        assert result.success is False
        assert result.error != ""

    def test_nonexistent_file_fails(self):
        result = extract_text_from_pdf(Path("/nonexistent/file/path/resume.pdf"))
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_parse_result_fields(self):
        result = ParseResult(text="Alice Johnson Python Developer", page_count=1)
        assert result.word_count == 4
        assert result.char_count == 30
        assert result.success is True

    def test_parse_result_empty_text_word_count(self):
        result = ParseResult()
        assert result.word_count == 0
        assert result.char_count == 0

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF is required for creating synthetic PDF test")
    def test_valid_pdf_generation_and_extraction(self):
        """Create a real minimal PDF in-memory and verify extraction."""
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Senior Machine Learning Engineer with Python and PyTorch experience.")
        pdf_bytes = doc.write()
        doc.close()

        result = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        assert result.success is True
        assert "Machine Learning Engineer" in result.text
        assert "Python" in result.text


# ===========================================================================
# evaluate.py Tests
# ===========================================================================

class TestEvaluationModule:
    """Tests for evaluate.py loading and benchmark evaluation."""

    def test_load_resumes(self):
        df = load_resumes()
        assert len(df) >= 5
        assert "name" in df.columns
        assert "text" in df.columns

    def test_load_job_descriptions(self):
        df = load_job_descriptions()
        assert len(df) >= 5
        assert "role" in df.columns
        assert "description" in df.columns

    def test_evaluate_candidates_ranking(self):
        df = evaluate_candidates("Data Scientist")
        assert len(df) >= 5
        assert "Composite ATS Score (%)" in df.columns
        assert "Candidate" in df.columns
        # Verify ranking is sorted descending
        scores = df["Composite ATS Score (%)"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_load_kaggle_dataset(self):
        df = load_kaggle_resumes()
        assert len(df) == 962
        assert "Category" in df.columns
        assert "Resume" in df.columns
        categories = get_kaggle_categories()
        assert len(categories) == 25
        assert "Data Science" in categories

    def test_evaluate_kaggle_candidates_filter(self):
        df = evaluate_kaggle_candidates("Data Scientist", category_filter="Data Science", limit=5)
        assert len(df) == 5
        assert "Composite ATS Score (%)" in df.columns
        assert "Domain Category" in df.columns
        assert all(df["Domain Category"] == "Data Science")
        scores = df["Composite ATS Score (%)"].tolist()
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_no_errors(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert result.error == ""
        assert result.composite_score >= 0.0

    def test_matched_skills_found_in_both(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        assert len(result.matched_skills) > 0

    def test_categorized_skills_dict(self):
        result = match_resume_to_jd(SAMPLE_RESUME, SAMPLE_JD)
        for cat, skills in result.matched_by_category.items():
            assert isinstance(cat, str)
            assert isinstance(skills, list)
