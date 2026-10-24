"""
Text Preprocessing Module for SmartSum.
Provides text cleaning, sentence segmentation, tokenization, lemmatization, stopword removal,
and language detection.
"""
import re
import logging
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger("SmartSum.Preprocessing")

# Ensure required NLTK resources are available
def setup_nltk():
    """Ensure NLTK datasets (punkt, stopwords, wordnet) are downloaded."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet"]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK resource {res}: {e}")

setup_nltk()

# Initialize NLTK components
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw input text by stripping HTML tags, redundant spaces, and normalizing line breaks.
    """
    if not text:
        return ""
    # Strip HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    # Replace non-breaking spaces and weird whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Split text into a list of clean, non-empty sentences.
    Uses NLTK sent_tokenize with a regex fallback.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    try:
        sentences = sent_tokenize(cleaned)
    except Exception as e:
        logger.warning(f"NLTK sentence tokenizer failed ({e}), using regex fallback.")
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    
    # Filter out empty or whitespace-only sentences
    return [s.strip() for s in sentences if s and len(s.strip()) > 3]


def tokenize_and_lemmatize(sentence: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize a sentence into words, convert to lowercase, remove punctuation/stopwords,
    and apply lemmatization.
    """
    try:
        words = word_tokenize(sentence.lower())
    except Exception:
        words = re.findall(r'\b\w+\b', sentence.lower())

    tokens = []
    for w in words:
        if not w.isalnum():
            continue
        if remove_stopwords and w in STOP_WORDS:
            continue
        lemma = LEMMATIZER.lemmatize(w)
        tokens.append(lemma)
    return tokens


def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    Returns ISO 639-1 code ('en', 'hi', 'mr').
    Uses Unicode character inspection as a primary/fallback mechanism for Devanagari script.
    """
    if not text or not text.strip():
        return "en"

    # Check for Devanagari Unicode characters (Hindi/Marathi)
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_chars = len(re.findall(r'\w', text))

    if total_chars > 0 and (devanagari_chars / total_chars) > 0.2:
        # Check specific Marathi vocabulary / markers if needed, default to 'hi' / 'mr'
        # Marathi unique markers like 'आहे', 'नाही', 'आणि', 'हे' vs Hindi 'है', 'और', 'का', 'के'
        marathi_keywords = ["आहे", "नाही", "आणि", "म्हणून", "होता", "होती"]
        has_marathi_kw = any(kw in text for kw in marathi_keywords)
        return "mr" if has_marathi_kw else "hi"

    # Default to langdetect if available, else 'en'
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in ["en", "hi", "mr"]:
            return lang
    except Exception:
        pass

    return "en"
