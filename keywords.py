"""
Keyword Extraction & WordCloud Module for SmartSum.
Provides YAKE, KeyBERT, TF-IDF keyword extraction and WordCloud visualization.
"""
import logging
import io
from typing import List, Tuple
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

logger = logging.getLogger("SmartSum.Keywords")

def extract_yake_keywords(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract keywords using YAKE algorithm."""
    if not text or not text.strip():
        return []
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=top_n)
        keywords = kw_extractor.extract_keywords(text)
        # YAKE scores: lower score = higher importance. Re-score for readability.
        return [(kw, round(1.0 / (score + 1e-5), 3)) for kw, score in keywords]
    except Exception as e:
        logger.error(f"YAKE keyword extraction error: {e}")
        return []

@st.cache_resource(show_spinner="Loading KeyBERT Model...")
def load_keybert_model():
    """Lazy load KeyBERT model."""
    try:
        from keybert import KeyBERT
        return KeyBERT(model="all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Failed to load KeyBERT: {e}")
        return None

def extract_keybert_keywords(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract keywords using KeyBERT."""
    if not text or not text.strip():
        return []
    kw_model = load_keybert_model()
    if kw_model is None:
        return []
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=20,
            top_n=top_n
        )
        return [(kw, round(score, 3)) for kw, score in keywords]
    except Exception as e:
        logger.error(f"KeyBERT error: {e}")
        return []

def generate_wordcloud(text: str):
    """Generate WordCloud image buffer from text."""
    if not text or not text.strip():
        return None
    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Blues',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        logger.error(f"WordCloud generation error: {e}")
        return None
