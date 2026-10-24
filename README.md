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

## 🎓 Viva Questions & Answers (NLP Engineering)

### 1. What is the difference between Extractive and Abstractive text summarization?
- **Extractive:** Identifies and extracts key sentences directly from the source text without altering words.
- **Abstractive:** Generates new sentences that paraphrase and condense original concepts using generative deep learning models.

### 2. How does the TextRank algorithm work?
- TextRank models a document as a graph where sentences are nodes and edge weights represent TF-IDF cosine similarity. PageRank is run over the graph to rank sentence importance.

### 3. What is TF-IDF and how is it calculated?
- $\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$, where TF is term frequency in document $d$, and IDF is inverse document frequency across corpus $D$.

### 4. What is ROUGE and what do ROUGE-1, ROUGE-2, and ROUGE-L measure?
- **ROUGE-1:** Overlap of unigrams (single words) between candidate and reference.
- **ROUGE-2:** Overlap of bigrams (two-word sequences).
- **ROUGE-L:** Longest Common Subsequence (LCS) matching.

### 5. What is BERTScore and why is it superior to ROUGE for abstractive summaries?
- BERTScore calculates token similarity using contextual embeddings from pre-trained Transformer models rather than exact n-gram matching, allowing evaluation of semantic equivalency.

### 6. How does SmartSum handle long documents that exceed Transformer token limits?
- It uses a **Hierarchical Map-Reduce Strategy**: long text is split into overlapping chunks, each chunk is summarized independently (Map), and partial summaries are concatenated and re-summarized (Reduce).

### 7. What is Natural Language Inference (NLI) and how is it used for hallucination checking?
- NLI evaluates whether a hypothesis (summary sentence) is logically **Entailed**, **Neutral**, or **Contradicted** by a premise (source document sentence) using a Cross-Encoder DeBERTa model.

### 8. What is the difference between Stemming and Lemmatization?
- **Stemming:** Heuristic suffix chopping (e.g., "running" -> "runn").
- **Lemmatization:** Vocabulary and morphological dictionary analysis returning true dictionary root form (e.g., "better" -> "good").

### 9. Why are Stopwords removed during extractive summarization?
- Stopwords (e.g., "is", "the", "and") carry low informational value and skew TF-IDF term frequencies.

### 10. How does Query-Focused Summarization select relevant content?
- Document sentences are converted into dense vector embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`). Sentences with highest cosine similarity to the user query are selected for summarization.

### 11. Explain the Self-Attention mechanism in Transformer architectures.
- Self-attention allows each token in a sequence to compute attention weights against all other tokens: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.

### 12. How does YAKE differ from KeyBERT for keyword extraction?
- **YAKE:** Unsupervised statistical keyword extractor using text features (case, position, frequency).
- **KeyBERT:** Uses BERT embeddings to calculate semantic similarity between candidate phrases and the document.

### 13. What is spaCy's role in Named Entity Recognition (NER)?
- spaCy identifies real-world entities (PERSON, ORG, GPE, DATE) using pre-trained convolutional neural networks and transition-based entity parsing.

### 14. What is Latent Semantic Analysis (LSA)?
- LSA applies Singular Value Decomposition (SVD) to the TF-IDF matrix to identify latent concept vectors and rank sentences based on concept representation.

### 15. How does bi-directional translation work in SmartSum?
- Non-English input (Hindi/Marathi) is translated to English via `Helsinki-NLP/opus-mt`, processed by the summarization pipeline, and back-translated to the target language.

### 16. What is Cloze-style Question Generation for local MCQs?
- Key entities/terms are extracted from sentences and replaced with blank lines (`_______`), while distractors are sampled from other document keywords.

### 17. Why is PyMuPDF (fitz) used instead of PyPDF2 for PDF loading?
- PyMuPDF is significantly faster and handles font encoding and complex multi-column PDF layouts more reliably.

### 18. How does faster-whisper transcribe audio files?
- It uses CTranslate2, a reimplemented C++ inference engine for Whisper models that runs up to 4x faster with lower memory footprint on CPU.

### 19. What is cosine similarity in vector space modeling?
- $\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$, measuring the cosine of the angle between two multi-dimensional vectors.

### 20. How is Sentiment Drift measured between original document and summary?
- VADER sentiment analysis calculates compound polarity scores for both text blocks; significant divergence indicates sentiment shift.

### 21. What is the role of Streamlit's `@st.cache_resource` decorator?
- It caches heavy ML model objects in memory across user sessions so models are loaded only once on startup.

### 22. What are modal verbs used in the Action Items extractor?
- Verbs indicating obligation or task assignment (e.g., "will", "must", "assigned to", "submit").

### 23. How does PyVis render interactive mind maps in web browsers?
- It converts NetworkX graph objects into standalone Vis.js HTML and JavaScript code rendered inside an inline iframe.

### 24. What is the difference between Encoder-only, Decoder-only, and Encoder-Decoder architectures?
- **Encoder-only (BERT):** Ideal for classification and embeddings.
- **Decoder-only (GPT):** Ideal for autoregressive text generation.
- **Encoder-Decoder (T5, BART):** Ideal for sequence-to-sequence transformation tasks like summarization and translation.

### 25. What limitations exist in offline local models compared to large cloud APIs like Gemini?
- Local models (DistilBART, T5-small) have smaller context windows and lower generative fluency, whereas cloud APIs require internet connectivity and API keys.

### 26. Why is bcrypt preferred over plain hashing or MD5 for password storage?
- bcrypt includes a built-in salt and adaptive work factor, making it computationally expensive to brute-force and resilient to rainbow table attacks.

### 27. What is the purpose of a SQLite database layer in SmartSum?
- It centralizes user management, summary persistence, feedback, and chat history in a single parameterized data access module, improving consistency and security.

### 28. How does a RAG-based chat system improve answering over a plain summarization model?
- Retrieval-Augmented Generation first finds the most relevant document chunks and then conditions the answer on that evidence, reducing hallucinations and improving grounding.

### 29. Why does SmartSum use sentence-transformers for document retrieval?
- Sentence-transformers convert text into dense semantic vectors so similarity search can rank passages by meaning rather than lexical overlap alone.

### 30. What is the difference between a local QA fallback and the Gemini-assisted answer path?
- Gemini is used when an API key is configured and AI mode is enabled, while the fallback uses a local question-answering pipeline such as DistilBERT SQuAD for offline operation.

### 31. Why is `parameterized SQL` essential for the database layer?
- It prevents SQL injection by separating query structure from user-controlled data and is a strong security best practice.

### 32. How does SmartSum store per-user preferences such as default method, language, and style?
- These values are stored in the `users` table and loaded into the Streamlit session when a user logs in, enabling personalized summarization defaults.

### 33. What is the value of the history and favorite system for academic workflows?
- It allows users to revisit and compare previously generated summaries, track document progress, and quickly reload saved results without repeating the full summarization pipeline.

### 34. Why is a guest mode useful in interactive applications like SmartSum?
- Guest mode allows quick exploration without registration, while protecting the privacy of users who do not want their documents or activity saved.

### 35. How are chat sessions linked to the user profile and summary context?
- Each chat message is stored with `user_id`, `role`, content, and optional `summary_id`, allowing history to persist per user and remain anchored to a specific document summary.
