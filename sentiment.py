"""
Sentiment Analysis Module for SmartSum.
Analyzes sentiment shifts between original text and summary.
"""
import logging
from typing import Dict, Any
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger("SmartSum.Sentiment")

def setup_vader():
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception as e:
        logger.warning(f"VADER download error: {e}")

setup_vader()

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze text sentiment using NLTK VADER.
    Returns compound score, label (Positive/Negative/Neutral), and breakdown.
    """
    if not text or not text.strip():
        return {"compound": 0.0, "label": "Neutral", "pos": 0.0, "neu": 1.0, "neg": 0.0}

    try:
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive 🟢"
        elif compound <= -0.05:
            label = "Negative 🔴"
        else:
            label = "Neutral ⚪"

        return {
            "compound": compound,
            "label": label,
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"]
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {"compound": 0.0, "label": "Neutral", "pos": 0.0, "neu": 1.0, "neg": 0.0}


def compare_sentiments(original_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Compare sentiment scores between original document and generated summary.
    """
    orig_sent = analyze_sentiment(original_text)
    summ_sent = analyze_sentiment(summary_text)

    score_diff = abs(orig_sent["compound"] - summ_sent["compound"])
    preserved = score_diff < 0.25

    return {
        "original": orig_sent,
        "summary": summ_sent,
        "sentiment_preserved": preserved,
        "drift_score": round(score_diff, 3)
    }
