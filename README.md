# ⚡ SmartSum - Multi-Source, Multilingual, Customizable Text Summarizer

SmartSum is a full-stack Natural Language Processing (NLP) application designed for multi-source text summarization, document analytics, quiz generation, domain adaptation, and automated quality evaluation.

Developed as a Final-Year B.E. (CSE AI/ML) Mini Project for Natural Language Processing.

---

## 🌟 Key Features

1. **Multi-Source Input Handling:**
   - Plain Text, PDF (`PyMuPDF`), Word Documents (`python-docx`), Web Article URLs (`trafilatura`), YouTube Transcripts (`youtube-transcript-api`), and Audio Files (`faster-whisper`).
2. **Hybrid Summarization Engine:**
   - **Extractive Algorithms:** TextRank (Graph PageRank), TF-IDF Sentence Scoring, Latent Semantic Analysis (LSA).
   - **Abstractive Models:** DistilBART (`sshleifer/distilbart-cnn-12-6`), T5-Small (`t5-small`), Google Gemini API (optional online upgrade).
   - **Query-Focused Summarization:** Embedding similarity (`all-MiniLM-L6-v2`) filtering prior to summary generation.
   - **Hierarchical Map-Reduce:** Automatic chunking for long documents exceeding model context limits.
3. **Multilingual Capabilities:**
   - Automatic language detection (English, Hindi, Marathi).
   - Bi-directional translation using `Helsinki-NLP/opus-mt`.
4. **Customizable Output Styles:**
   - Standard Paragraph, Bullet Points, ELI5 (Explain Like I'm 5), and Structured Exam Notes.
5. **Document Insights & Analytics:**
   - Keyword comparison (YAKE vs. KeyBERT), WordCloud visualization, spaCy Named Entity Recognition (NER), Sentiment shift tracking, Interactive PyVis Concept Mind Map, and Action Item Extractor.
6. **Study & Quiz Tools:**
   - Auto-generated 4-option MCQs with explanations, Flashcards, and interactive test mode with scoring.
7. **Domain Adaptation:**
   - Specialized section detection and summarization for Research Papers, Legal Case Briefs, and Medical Reports.
8. **Automated Quality & Faithfulness Evaluation:**
   - ROUGE-1/2/L, BERTScore, and NLI Cross-Encoder hallucination checking (`cross-encoder/nli-deberta-v3-small`).

---

## 📏 Evaluation Notes

- ROUGE-1, ROUGE-2, and ROUGE-L compare a candidate against a supplied reference summary; they are not source-only quality scores.
- BERTScore is reported only when its model and dependencies are available. An unavailable calculation is not represented as a zero score.
- Key Content Coverage in the summary panel is a lexical overlap measure: the percentage of unique source lemmas longer than three characters that also occur in the summary. It does not claim to detect semantic equivalence or measure clinical concept coverage.
- NLI faithfulness uses entailment probabilities from the configured cross-encoder. If that model cannot load or inference fails, the result is explicitly marked as a token-overlap fallback, not as an NLI score.
- Long abstractive input is split into overlapping sentence-aligned chunks and reduced hierarchically; the local model wrappers no longer silently discard text after a fixed word limit.

---

## 🛠️ Project Structure

```text
smartsum/
├── app.py                  # Streamlit multi-tab user interface entry point
├── config.py               # Model constants, hardware device detection, defaults
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment template for Gemini API key
├── README.md               # Documentation & Viva Q&A Guide
├── PROJECT_REPORT.md       # Full B.E. Project Report Draft
├── core/                   # Core NLP engines (Extractive, Abstractive, Preprocessing, Multilingual)
├── inputs/                 # Input loaders (PDF, DOCX, URL, YouTube, Audio)
├── extras/                 # Document insights, Keywords, NER, Sentiment, Mindmap, MCQs
├── evaluation/             # ROUGE, BERTScore, Faithfulness NLI check, Benchmark
├── utils/                  # Helper utilities and file export functions (TXT, DOCX, PDF)
├── tests/                  # Unit tests (pytest)
└── sample_data/            # Sample files for testing
```

---

## 🔒 Phase 11-15 Feature Additions

SmartSum now includes user-aware persistence and AI-assisted workflows while preserving the original summarization engine:

- **Authentication & SQLite Database:** Secure signup/login/logout with bcrypt password hashing, input validation, guest access, and persistence for users, summaries, feedback, and chat history in `database/smartsum.db`.
- **History & Favorites:** Auto-save generated summaries with metadata, searchable history cards, favorites, reload, delete actions, and download support in TXT/DOCX/PDF.
- **Dashboard & Analytics:** Metric cards and Plotly charts for summaries per day, method usage, input type usage, average compression, and feedback trends; admin overview for the first registered user.
- **Document Q&A Chat (RAG):** Retrieve relevant document chunks using sentence-transformer embeddings, answer with Gemini when available, and fall back to a local question-answering model.
- **AI Mode & Settings:** Sidebar AI toggle, per-user default method/language/style preferences, lite mode, and thumbs-up/down feedback tracking.

---

## 🚀 Quick Setup Instructions

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

### Streamlit Community Cloud
1. Push the project to a GitHub repository and deploy `app.py` from Streamlit Community Cloud.
2. Add `requirements.txt` as the dependency file and configure optional credentials in the app's **Secrets** settings (for example, `GEMINI_API_KEY = "..."`). Do not commit `.env` files or credentials.
3. SmartSum loads local NLP models on first use and caches them in process memory. The Transformer, translation, embedding, and audio dependencies/models can require substantial cold-start time, download bandwidth, and RAM; test the selected deployment tier with the features you intend to use.
4. The default SQLite database is `database/smartsum.db`. Streamlit Community Cloud's local filesystem is not durable across all restarts/redeployments, so do not rely on this default for persistent user accounts or history. Use a durable mounted path via `SMARTSUM_DB_PATH` where the hosting platform supports one, or an external persistent database before deploying for multi-user production use.

---

## 📸 Screenshots & Workflow

*(Placeholders for project report and presentation)*
- **Tab 1:** Multi-Source Input & Summarization Result with Export buttons.
- **Tab 2:** Interactive Mind Map & WordCloud.
- **Tab 3:** Interactive MCQ Quiz with instant score feedback.
- **Tab 5:** NLI Hallucination Verification highlighting entailed vs unsupported sentences.

---
