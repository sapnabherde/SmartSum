"""
Central configuration for SmartSum application.
Holds model names, default settings, device configuration, Lite mode, and secrets.
"""
import os
import torch
import psutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# Hardware Device Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Memory Check & Lite Mode Logic
try:
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
except Exception:
    available_ram_gb = 4.0

# Auto-enable Lite mode if available RAM is under 2.5 GB
AUTO_LITE_MODE = available_ram_gb < 2.5
LITE_MODE = os.getenv("SMARTSUM_LITE_MODE", "false").lower() == "true" or AUTO_LITE_MODE

# Gemini API Key loading from st.secrets or environment
GEMINI_API_KEY = ""
try:
    import streamlit as st
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

HAS_GEMINI = bool(GEMINI_API_KEY)

# Local NLP Model Identifiers
MODELS = {
    "distilbart": "sshleifer/distilbart-cnn-12-6",
    "t5_small": "t5-small",
    "t5_qg": "mrm8488/t5-base-finetuned-question-generation-ap",
    "sentence_transformer": "all-MiniLM-L6-v2",
    "nli_evaluator": "cross-encoder/nli-deberta-v3-small",
    "spacy_model": "en_core_web_sm",
    "translation_hi_en": "Helsinki-NLP/opus-mt-hi-en",
    "translation_en_hi": "Helsinki-NLP/opus-mt-en-hi",
    "translation_mr_en": "Helsinki-NLP/opus-mt-mr-en",
    "translation_en_mr": "Helsinki-NLP/opus-mt-en-mr",
    "whisper_model": "base",  # base or small for fast CPU inference
}

# Summarization Defaults
DEFAULT_RATIO = 0.3
DEFAULT_MIN_SENTENCES = 2
DEFAULT_MAX_SENTENCES = 20

# Input Limits
MAX_CHAR_LIMIT = 500000  # ~100k words
MAX_AUDIO_SIZE_MB = 25
MAX_PDF_SIZE_MB = 15

# Supported Languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

# Style Choices
SUMMARY_STYLES = [
    "Bullet Points",
    "Paragraph",
    "Executive Summary",
    "Key Takeaways",
    "Structured Summary",
    "Study Notes",
    "Bullet points",
    "ELI5 (Explain Like I'm 5)",
    "Exam notes (Headings + Key Points)",
]

