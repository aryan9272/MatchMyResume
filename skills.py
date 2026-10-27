"""
skills.py - Punctuation-aware skill extraction engine and taxonomy.

Features:
- Domain-categorized skill dictionary (8 major engineering categories).
- Punctuation-safe regex matching that preserves C++, C#, .NET, Node.js,
  CI/CD, scikit-learn, etc.
- Prevents false-positive matches (e.g. 'C' will NOT trigger on 'C++' or 'C#').
- Aliases support (e.g. 'sklearn' -> 'scikit-learn', 'k8s' -> 'Kubernetes').

No external APIs, cloud services, or paid subscriptions required.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Structured Skill Taxonomy
# ---------------------------------------------------------------------------

TECH_SKILLS: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
        "R", "Go", "Rust", "Swift", "Kotlin", "PHP", "Ruby", "Scala",
        "MATLAB", "Perl", "Bash", "Shell Scripting", "SQL", "PL/SQL",
        "HTML", "CSS", "SASS", "SCSS", ".NET", "ASP.NET", "Dart",
    ],
    "Machine Learning & AI": [
        "Machine Learning", "Deep Learning", "Neural Networks",
        "Natural Language Processing", "NLP", "Computer Vision",
        "Reinforcement Learning", "Transfer Learning",
        "scikit-learn", "TensorFlow", "PyTorch", "Keras",
        "XGBoost", "LightGBM", "CatBoost", "Random Forest",
        "Decision Trees", "Support Vector Machine", "SVM",
        "K-Nearest Neighbors", "KNN", "Naive Bayes",
        "Logistic Regression", "Linear Regression",
        "Gradient Boosting", "AdaBoost",
        "Convolutional Neural Network", "CNN",
        "Recurrent Neural Network", "RNN", "LSTM",
        "Transformer", "BERT", "GPT",
        "Generative AI", "GenAI", "Large Language Models", "LLMs",
        "Hugging Face", "spaCy", "NLTK",
        "TF-IDF", "Word2Vec", "GloVe", "FastText",
        "Principal Component Analysis", "PCA",
        "Dimensionality Reduction", "Feature Engineering",
        "Hyperparameter Tuning", "Cross-Validation",
        "Model Evaluation", "Confusion Matrix", "ROC Curve", "AUC",
        "Precision", "Recall", "F1 Score",
        "Overfitting", "Underfitting", "Regularization",
        "Clustering", "K-Means", "DBSCAN", "Hierarchical Clustering",
        "Anomaly Detection", "Time Series Analysis",
        "Recommender Systems", "Collaborative Filtering",
        "Object Detection", "Image Segmentation",
        "Speech Recognition", "Text Classification",
        "Sentiment Analysis", "Named Entity Recognition", "NER",
        "Text Summarization", "Machine Translation",
        "Prompt Engineering", "RAG", "Retrieval Augmented Generation",
        "MLflow", "Weights & Biases", "W&B",
        "ONNX", "OpenCV",
    ],
    "Data Science & Analytics": [
        "Data Science", "Data Analysis", "Data Engineering",
        "Data Visualization", "Statistical Analysis",
        "Exploratory Data Analysis", "EDA",
        "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn",
        "Plotly", "Bokeh", "Tableau", "Power BI",
        "Excel", "Google Sheets",
        "Statistics", "Probability", "Hypothesis Testing",
        "A/B Testing", "Regression Analysis",
        "Data Cleaning", "Data Preprocessing", "Data Wrangling",
        "Feature Selection", "Data Pipelines",
        "ETL", "ELT", "Data Warehousing",
        "Business Intelligence", "BI",
        "Jupyter Notebook", "Google Colab",
    ],
    "Databases": [
        "MySQL", "PostgreSQL", "SQLite", "Microsoft SQL Server",
        "Oracle Database", "MongoDB", "Redis", "Cassandra",
        "DynamoDB", "Firebase", "Elasticsearch",
        "Neo4j", "GraphQL", "NoSQL",
        "Database Design", "Database Administration",
        "Query Optimization", "Indexing",
        "BigQuery", "Snowflake", "Redshift", "Apache Hive",
    ],
    "Cloud & DevOps": [
        "AWS", "Amazon Web Services", "Azure", "Google Cloud Platform", "GCP",
        "Docker", "Kubernetes", "Terraform", "Ansible",
        "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI",
        "Helm", "ArgoCD", "Linux", "Unix",
        "Nginx", "Apache", "Load Balancing",
        "Microservices", "Serverless", "Lambda",
        "Cloud Storage", "S3", "IAM",
        "Monitoring", "Logging", "Prometheus", "Grafana",
    ],
    "Web & App Development": [
        "React", "Angular", "Vue.js", "Next.js", "Nuxt.js",
        "Node.js", "Express.js", "Django", "Flask", "FastAPI",
        "Spring Boot", "Laravel", "Ruby on Rails",
        "REST API", "RESTful API",
        "Microservices Architecture", "WebSocket",
        "Streamlit", "Gradio",
        "Android Development", "iOS Development",
        "React Native", "Flutter",
        "HTML5", "CSS3", "Bootstrap", "Tailwind CSS",
        "jQuery", "Redux",
    ],
    "Tools & Platforms": [
        "Git", "GitHub", "GitLab", "Bitbucket",
        "JIRA", "Confluence", "Trello", "Asana",
        "VS Code", "PyCharm", "IntelliJ IDEA", "Eclipse",
        "Postman", "Swagger", "Insomnia",
        "Ubuntu", "Windows Server",
        "Apache Spark", "Hadoop", "Kafka", "Airflow",
        "Databricks", "dbt",
        "Figma", "Adobe XD", "Sketch",
        "MS Office", "LaTeX",
    ],
    "Soft Skills": [
        "Communication", "Teamwork", "Leadership",
        "Problem Solving", "Critical Thinking",
        "Time Management", "Project Management",
        "Agile", "Scrum", "Kanban",
        "Analytical Skills", "Attention to Detail",
        "Creativity", "Adaptability", "Collaboration",
        "Presentation Skills", "Public Speaking",
        "Research", "Documentation", "Technical Writing",
        "Mentoring", "Coaching",
    ],
}

# ---------------------------------------------------------------------------
# Skill Aliases & Variants mapping
# ---------------------------------------------------------------------------
# Maps common shorthand / alternate spellings to canonical taxonomy names
SKILL_ALIASES: Dict[str, str] = {
    "sklearn": "scikit-learn",
    "k8s": "Kubernetes",
    "postgres": "PostgreSQL",
    "mongo": "MongoDB",
    "tf": "TensorFlow",
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "gcp": "Google Cloud Platform",
    "aws": "AWS",
    "bi": "Business Intelligence",
    "ci cd": "CI/CD",
    "ci/cd": "CI/CD",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "reactjs": "React",
    "react.js": "React",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "dotnet": ".NET",
    ".net": ".NET",
    "asp.net": "ASP.NET",
    "aspdotnet": "ASP.NET",
    "restful api": "REST API",
    "rest api": "REST API",
    "rest apis": "REST API",
}

# ---------------------------------------------------------------------------
# Precompiled Regex Matchers
# ---------------------------------------------------------------------------

def _build_skill_patterns() -> List[Tuple[str, re.Pattern]]:
    """
    Build compiled regular expressions for every skill and alias.
    Handles punctuation carefully:
    - C++: matches c++ but not C
    - C#: matches c# but not C
    - C: matches standalone 'c', NOT followed/preceded by +, #, or word char
    - R: matches standalone 'r', NOT followed/preceded by +, #, or word char
    - .NET: matches .net, dotnet
    - Node.js, React.js, etc.
    """
    patterns: List[Tuple[str, re.Pattern]] = []

    # Process all canonical skills
    all_canonical: Set[str] = set()
    for cat_skills in TECH_SKILLS.values():
        all_canonical.update(cat_skills)

    for skill in sorted(all_canonical, key=lambda s: len(s), reverse=True):
        skill_lower = skill.lower()

        if skill_lower == "c":
            # Match 'c' ONLY if not preceded/followed by word char, +, or #
            # e.g., 'c programming', 'c, c++', 'c/c++' -> matches 'c'
            pat = re.compile(r"(?<![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])", re.IGNORECASE)
        elif skill_lower == "r":
            # Standalone 'r' or 'r language' or 'r programming'
            pat = re.compile(r"(?<![a-zA-Z0-9+#])r(?![a-zA-Z0-9+#])", re.IGNORECASE)
        elif skill_lower == "c++":
            pat = re.compile(r"(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "c#":
            pat = re.compile(r"(?<![a-zA-Z0-9])c\#(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == ".net":
            pat = re.compile(r"(?<![a-zA-Z0-9])(?:\.net|dotnet)(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "asp.net":
            pat = re.compile(r"(?<![a-zA-Z0-9])asp\.net(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "ci/cd":
            pat = re.compile(r"(?<![a-zA-Z0-9])ci\s*[/\\-]?\s*cd(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "scikit-learn":
            pat = re.compile(r"(?<![a-zA-Z0-9])(?:scikit[\s\-_]*learn|sklearn)(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "node.js":
            pat = re.compile(r"(?<![a-zA-Z0-9])node(?:\.js|js)?(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "vue.js":
            pat = re.compile(r"(?<![a-zA-Z0-9])vue(?:\.js|js)?(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "next.js":
            pat = re.compile(r"(?<![a-zA-Z0-9])next(?:\.js|js)?(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "react":
            pat = re.compile(r"(?<![a-zA-Z0-9])react(?:\.js|js)?(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "tf-idf":
            pat = re.compile(r"(?<![a-zA-Z0-9])tf[\s\-_]*idf(?![a-zA-Z0-9])", re.IGNORECASE)
        elif skill_lower == "rest api":
            pat = re.compile(r"(?<![a-zA-Z0-9])rest(?:ful)?\s*apis?(?![a-zA-Z0-9])", re.IGNORECASE)
        else:
            # Standard boundary match with regex escape
            escaped = re.escape(skill_lower)
            pat = re.compile(r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])", re.IGNORECASE)

        patterns.append((skill, pat))

    # Add aliases
    for alias_raw, canonical in SKILL_ALIASES.items():
        if alias_raw in ["c", "r", "c++", "c#", ".net", "ci/cd", "scikit-learn", "node.js", "react", "vue.js", "next.js"]:
            continue
        escaped = re.escape(alias_raw)
        if alias_raw in ["js", "ts", "py"]:
            # Do not match after a period (e.g. node.js, script.py)
            pat = re.compile(r"(?<![a-zA-Z0-9\.])" + escaped + r"(?![a-zA-Z0-9])", re.IGNORECASE)
        else:
            pat = re.compile(r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])", re.IGNORECASE)
        patterns.append((canonical, pat))

    return patterns


COMPILED_PATTERNS = _build_skill_patterns()

# Quick reverse-lookup mapping: lowercase skill -> category
_SKILL_TO_CATEGORY: Dict[str, str] = {}
for _cat, _skills in TECH_SKILLS.items():
    for _s in _skills:
        _SKILL_TO_CATEGORY[_s.lower()] = _cat


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> Set[str]:
    """
    Extract skills from arbitrary text using punctuation-aware regex rules.

    Parameters
    ----------
    text : str
        Resume or job description text.

    Returns
    -------
    Set[str]
        Set of matched canonical skill names.
    """
    if not text or not isinstance(text, str):
        return set()

    found: Set[str] = set()
    for canonical_name, pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            found.add(canonical_name)

    return found


def get_skill_category(skill: str) -> str:
    """Return the category of a skill or 'Unknown'."""
    return _SKILL_TO_CATEGORY.get(skill.lower(), "Unknown")


def categorize_skills(skill_set: Set[str]) -> Dict[str, List[str]]:
    """Group a set of skills into their respective taxonomy categories."""
    result: Dict[str, List[str]] = {}
    for skill in sorted(skill_set):
        cat = get_skill_category(skill)
        result.setdefault(cat, []).append(skill)
    return result
