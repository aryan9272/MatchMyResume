# ATS Resume Matcher using Natural Language Processing and Machine Learning

A transparent, college-level Introduction to Machine Learning (IML) project that evaluates and ranks resumes against job descriptions using an unsupervised Vector Space Model (TF-IDF + Cosine Similarity) and a punctuation-aware technical skill extraction engine.

---

## 🌟 Key Features

- **100% Local & Self-Contained:** No external APIs, cloud databases, paid subscriptions, or API keys required.
- **Unsupervised Vector Space Model:** TF-IDF unigram + bigram vectorization with sublinear scaling and smooth IDF. Zero fabricated models, fake accuracy claims, or black-box classifiers.
- **Explainable Cosine Similarity:** Computes normalized vector dot products and identifies the exact top n-grams driving the match.
- **Punctuation-Aware Skill Taxonomy:** Extracts technical skills using negative lookaround regex patterns, preventing false positives (e.g., `C++` or `C#` will **never** falsely trigger `C`). Handles `.NET`, `Node.js`, `CI/CD`, `scikit-learn`, `TF-IDF`, etc.
- **Three-Way Skill Categorization:** Identifies **Matched Skills** (green), **Missing Skills** (red), and **Additional Skills** (blue) grouped across 8 engineering domains.
- **Actionable Gap Recommendations:** Delivers automated, role-specific advice based on missing skills, word counts, and vocabulary alignment.
- **Candidate Benchmark Ranking:** Evaluates and ranks 5 fictional candidate profiles simultaneously against any selected job role using interactive Plotly charts.
- **Robust PDF Parsing:** Built with PyMuPDF (fitz) to extract clean text from digital PDFs with graceful handling for corrupt, empty, password-protected, or image-only files.
- **Glassmorphic 3D UI & Animated Background:** Sleek, high-contrast 3D frosted glass design with dynamic floating background geometry, cinematic opening animations, and a strict 3-4 color palette (Midnight Navy, Cyber Cyan, Electric Indigo, and Frost White).
- **Large Hero Branding:** Prominent glowing header with smooth 3D entrance and metallic gradient typography.
- **Thorough Test Suite:** 53 unit and integration tests covering all parsing, vectorization, scoring, and edge cases.

---

## 🏗️ Project Architecture

```
c:\MatchMyResume/
├── app.py                      # Main Streamlit web dashboard (Light Theme)
├── matcher.py                  # NLP matching engine (TF-IDF, Cosine Similarity, XAI)
├── pdf_parser.py               # Robust PDF text extraction via PyMuPDF
├── skills.py                   # Punctuation-aware skill taxonomy and regex engine
├── evaluate.py                 # Multi-candidate benchmark evaluation & ranking
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── PRESENTATION_GUIDE.md       # Defense guide with 16 viva questions & answers
├── generate_overview_pdf.py    # ReportLab script generating teammate overview PDF
├── ATS_Resume_Matcher_Project_Overview.pdf # Generated printable PDF overview
├── .streamlit/
│   └── config.toml             # Streamlit Light Theme enforcement
├── sample_data/
│   ├── resumes.csv             # 5 fictional candidate profiles (Alice, Bob, Carol, David, Eva)
│   ├── job_descriptions.csv    # 5 industry benchmark JDs (Data Scientist, ML Engineer, etc.)
│   └── Alice_Johnson_Resume.pdf# Sample PDF resume for immediate testing
└── tests/
    └── test_matcher.py         # 53 automated pytest unit & integration tests
```

---

## 📐 Mathematical Formulation

### 1. Sublinear Term Frequency ($\text{tf}$)
To penalize keyword stuffing, term frequency is logarithmically scaled:
$$\text{tf}_{\text{scaled}}(t, d) = \begin{cases} 1 + \ln(\text{tf}(t, d)) & \text{if } \text{tf}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2. Smooth Inverse Document Frequency ($\text{idf}$)
$$\text{idf}(t) = \ln\left(\frac{1 + n}{1 + \text{df}(t)}\right) + 1$$
Where $n=2$ in pairwise comparison and $\text{df}(t)$ is document frequency.

### 3. Cosine Similarity in Vector Space
Documents are L2-normalized ($\|\mathbf{u}\|_2 = 1, \|\mathbf{v}\|_2 = 1$), reducing cosine similarity to the Euclidean dot product:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{|V|} u_i \cdot v_i \quad \in [0.0, 1.0]$$

### 4. Skill Coverage Ratio
$$\text{Skill Coverage} = \frac{|\text{Resume Skills} \cap \text{JD Skills}|}{|\text{JD Skills}|} \quad \in [0.0, 1.0]$$

### 5. Composite Weighted ATS Score
$$\text{Composite Score} = w_{\text{tfidf}} \cdot \text{Cosine Similarity} + w_{\text{skill}} \cdot \text{Skill Coverage}$$
Where $w_{\text{tfidf}} + w_{\text{skill}} = 1.0$ (default $0.50$ / $0.50$, configurable via sidebar sliders).

---

## 💻 Windows Setup & Installation

Follow these steps in Windows PowerShell to set up and run the project locally.

### Step 1: Open PowerShell and Navigate to the Project Directory
```powershell
cd C:\MatchMyResume
```

### Step 2: Create a Virtual Environment (Optional if venv already exists)
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If you see an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Run Automated Tests
```powershell
python -m pytest tests/ -v
```
All 53 unit and integration tests will run and pass.

### Step 6: Launch the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Streamlit will automatically open your default browser to `http://localhost:8501`.

---

## 📊 Sample Datasets Included

### 1. `sample_data/job_descriptions.csv`
Contains 5 comprehensive, realistic tech job postings:
- **Data Scientist:** Strong focus on Python, R, scikit-learn, PyTorch, SQL, EDA, and NLP.
- **Full Stack Developer:** Focus on React, Node.js, TypeScript, Django, Docker, and REST APIs.
- **Machine Learning Engineer:** Focus on Deep Learning, Transformers, MLOps, MLflow, and Kubernetes.
- **Data Analyst:** Focus on SQL, Tableau, Power BI, Excel, Pandas, and Business Intelligence.
- **Software Engineer:** Focus on Data Structures, OOP, C++, Java, Git, and Linux.

### 2. `sample_data/resumes.csv`
Contains 5 fictional candidate resumes representing diverse qualification tiers:
- **Eva Chen:** Senior AI/ML Engineer with extensive deep learning and MLOps credentials.
- **Carol Smith:** M.Sc. Data Science graduate with Spark, PyTorch, and NLP research.
- **Alice Johnson:** B.Tech Computer Science graduate with strong entry-level ML and data analytics projects.
- **Bob Martinez:** Full Stack Developer with strong React/Node.js/Docker skills (low fit for Data Science).
- **David Lee:** IT Support intern with basic C++/Java (accurately receives low ATS scores for advanced roles).

### 3. `sample_data/Alice_Johnson_Resume.pdf`
A clean digital PDF resume generated with PyMuPDF for instant testing of the PDF upload feature.

---

## 🔬 Methodology & Academic Integrity

- **No Fabricated Machine Learning Claims:** This baseline does not claim to train a 99% accurate supervised neural network on nonexistent hiring data. Instead, it correctly frames resume screening as an **unsupervised Vector Space Model (VSM)** problem.
- **Reproducibility:** Every similarity score and skill count is 100% deterministic and explainable from the text.
- **Explainable AI (XAI):** The dashboard renders the exact overlapping n-grams and their mathematical weights ($u_i \times v_i$), satisfying modern explainability standards.

---

## 🎓 Viva Voce & Presentation Defense

Refer to [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md) for 16 detailed viva questions and model answers, complete with mathematical derivations, architecture diagrams, and examiner talking points.

To share a clean PDF summary with your teammates, refer to `ATS_Resume_Matcher_Project_Overview.pdf` in this directory.
