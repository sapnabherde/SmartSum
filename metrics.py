"""
Evaluation Metrics Module for SmartSum.
Calculates ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore against a reference summary.
"""
import logging
from typing import Dict, Any
import streamlit as st
import config

logger = logging.getLogger("SmartSum.Metrics")

def compute_rouge_scores(candidate_summary: str, reference_summary: str) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores using rouge-score library.
    """
    if not candidate_summary or not reference_summary:
        return {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        }

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, candidate_summary)

        results = {}
        for key in ['rouge1', 'rouge2', 'rougeL']:
            results[key] = {
                "precision": round(scores[key].precision, 4),
                "recall": round(scores[key].recall, 4),
                "fmeasure": round(scores[key].fmeasure, 4)
            }
        return results
    except Exception as e:
        logger.exception("ROUGE computation failed.")
        raise RuntimeError("ROUGE scoring is unavailable; install the declared rouge-score dependency.") from e


@st.cache_resource(show_spinner="Loading BERTScore Model...")
def load_bertscore_evaluator():
    """Lazy load BERTScore evaluator."""
    try:
        import bert_score
        return bert_score
    except Exception as e:
        logger.error(f"Failed to import bert_score: {e}")
        return None


def compute_bertscore(candidate_summary: str, reference_summary: str) -> Dict[str, float] | None:
    """
    Compute BERTScore Precision, Recall, and F1 between candidate and reference summaries.
    """
    if not candidate_summary or not reference_summary:
        return None

    bs = load_bertscore_evaluator()
    if bs is None:
        logger.warning("BERTScore is unavailable; no score was produced.")
        return None

    try:
        P, R, F1 = bs.score([candidate_summary], [reference_summary], lang="en", verbose=False)
        return {
            "precision": round(float(P[0]), 4),
            "recall": round(float(R[0]), 4),
            "f1": round(float(F1[0]), 4)
        }
    except Exception as e:
        logger.warning("BERTScore computation failed; no score was produced: %s", e)
        return None
