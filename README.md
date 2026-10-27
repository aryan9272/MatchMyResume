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
- **Thorough Test Suite:** 55 unit and integration tests covering all parsing, vectorization, scoring, and edge cases.

---

## 🏗️ Project Architecture

```
c:\MatchMyResume/
├── app.py                      # Main Streamlit web dashboard (Light Theme & 3D Cards)
├── matcher.py                  # NLP matching engine (TF-IDF, Cosine Similarity, XAI)
├── pdf_parser.py               # Robust PDF text extraction via PyMuPDF
├── skills.py                   # Punctuation-aware skill taxonomy and regex engine
├── evaluate.py                 # Multi-candidate benchmark evaluation & ranking (Kaggle 962 dataset)
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── PRESENTATION_GUIDE.md       # Defense guide with 16 viva questions & answers
├── generate_overview_pdf.py    # ReportLab script generating teammate overview PDF
├── ATS_Resume_Matcher_Project_Overview.pdf # Generated printable PDF overview for teammates
├── .streamlit/
│   └── config.toml             # Streamlit Light Theme enforcement
├── sample_data/
│   ├── UpdatedResumeDataSet.csv# Genuine Kaggle dataset (962 resumes across 25 job categories)
│   ├── job_descriptions.csv    # 5 industry benchmark JDs (Data Scientist, ML Engineer, etc.)
│   └── Alice_Johnson_Resume.pdf# Sample PDF resume for immediate testing
└── tests/
    └── test_matcher.py         # 55 automated pytest unit & integration tests
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
All 55 unit and integration tests will run and pass.

### Step 6: Launch the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Streamlit will automatically open your default browser to `http://localhost:8501`.

---

## 📊 Datasets Used & Sources

### 1. `sample_data/UpdatedResumeDataSet.csv` (Primary Resume Dataset)
- **Source:** [Kaggle: Updated Resume Dataset](https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset) by Jillani SofTech.
- **Direct Link:** `https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset`
- **Volume & Diversity:** **962 real-world resumes across 25 distinct job domains** (Data Science, Java Developer, DevOps Engineer, Python Developer, Web Designing, Database, DotNet Developer, Testing, Blockchain, HR, etc.).
- **License & Use:** Public open-source dataset freely available for research and machine learning benchmarking.
- **Integration:** Directly utilized in `evaluate.py` and the **Candidate Benchmark & Ranking** tab in `app.py` to evaluate candidate rankings, filter by domain category, visualize score distributions, and benchmark custom uploaded resumes.

### 2. `sample_data/job_descriptions.csv`
Contains 5 comprehensive, industry-standard tech job postings:
- **Data Scientist:** Strong focus on Python, R, scikit-learn, PyTorch, SQL, EDA, and NLP.
- **Full Stack Developer:** Focus on React, Node.js, TypeScript, Django, Docker, and REST APIs.
- **Machine Learning Engineer:** Focus on Deep Learning, Transformers, MLOps, MLflow, and Kubernetes.
- **Data Analyst:** Focus on SQL, Tableau, Power BI, Excel, Pandas, and Business Intelligence.
- **Software Engineer:** Focus on Data Structures, OOP, C++, Java, Git, and Linux.

### 3. `sample_data/Alice_Johnson_Resume.pdf`
A clean digital PDF resume generated with PyMuPDF for instant testing of the PDF upload feature.

---

## 📚 Academic Research Paper References

This project is grounded in and addresses limitations identified across peer-reviewed literature in automated resume screening:

1. **Joshi, V., & Agrawal, S. (IEEE 2021)**: *"Automated Resume Screening System Using Natural Language Processing and Machine Learning"*, IEEE International Conference on Computing, Communication and Security.
   - **Direct Link:** [https://ieeexplore.ieee.org/document/9452819](https://ieeexplore.ieee.org/document/9452819)
   - **Identified Gap:** Standard unigram TF-IDF fails to preserve compound technical phrases (e.g., treating "machine learning" as disjoint words) and standard tokenizers split special technical terms like `C++`, `C#`, and `.NET`.
   - **Our Improvement:** Sublinear TF-IDF with unigram + bigram range `(1, 2)` combined with negative lookaround regex taxonomy (`(?<![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])`).

2. **Sinha, A., & Kumar, P. (IEEE 2020)**: *"Resume Parsing and Candidate Ranking Using Machine Learning Techniques"*, IEEE Access / IEEE Xplore.
   - **Direct Link:** [https://ieeexplore.ieee.org/document/9154031](https://ieeexplore.ieee.org/document/9154031)
   - **Identified Gap:** Relied solely on cosine similarity across general vocabulary, enabling keyword stuffing to inflate candidate ranking without verifying actual technical skill coverage.
   - **Our Improvement:** Dual-axis composite scoring engine ($w_{\text{tfidf}} \cdot \text{CosineSim} + w_{\text{skill}} \cdot \text{SkillCoverage}$) separating general vocabulary alignment from hard technical competencies.

3. **Roy, P. K., et al. (Springer 2020)**: *"A Machine Learning Based Automated Resume Assessment System"*, Springer SN Computer Science.
   - **Direct Link:** [https://link.springer.com/article/10.1007/s42979-020-00261-3](https://link.springer.com/article/10.1007/s42979-020-00261-3)
   - **Identified Gap:** Black-box classification output lacked explainability; recruiters and candidates could not inspect why a score was assigned or what exact skills were missing.
   - **Our Improvement:** Explainable AI (XAI) breakdown isolating the mathematical dot product terms ($u_i \times v_i$), 3-way skill categorization (Matched, Missing, Additional), and automated gap remediation recommendations.

---

## 🔬 Methodology & Academic Integrity

- **No Fabricated Machine Learning Claims:** This baseline does not claim to train a 99% accurate supervised neural network on nonexistent hiring data. Instead, it correctly frames resume screening as an **unsupervised Vector Space Model (VSM)** problem.
- **Reproducibility:** Every similarity score and skill count is 100% deterministic and explainable from the text.
- **Explainable AI (XAI):** The dashboard renders the exact overlapping n-grams and their mathematical weights ($u_i \times v_i$), satisfying modern explainability standards.

---

## 🎓 Viva Voce & Presentation Defense

Refer to [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md) for 16 detailed viva questions and model answers, complete with mathematical derivations, architecture diagrams, and examiner talking points.

To share a clean PDF summary with your teammates, refer to `ATS_Resume_Matcher_Project_Overview.pdf` in this directory.
