"""
Faithfulness & Hallucination Checker Module for SmartSum using NLI Cross-Encoder.
Verifies whether generated summary sentences are factually entailed by the source document.
"""
import logging
from typing import List, Dict, Any
import numpy as np
import streamlit as st
import config
from core.preprocessing import split_sentences, tokenize_and_lemmatize

logger = logging.getLogger("SmartSum.Faithfulness")

@st.cache_resource(show_spinner="Loading NLI Faithfulness Model...")
def load_nli_cross_encoder():
    """Lazy load CrossEncoder model for Natural Language Inference (NLI)."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(config.MODELS["nli_evaluator"], device=config.DEVICE)
        return model
    except Exception as e:
        logger.error(f"Failed to load CrossEncoder NLI model: {e}")
        return None


def verify_summary_faithfulness(original_text: str, summary_text: str) -> List[Dict[str, Any]]:
    """
    Check each sentence of summary against original document sentences.
    Returns list of dicts with:
    - 'sentence': summary sentence
    - 'support_score': entailment confidence float [0.0 - 1.0]
    - 'status': 'Supported' (Green) or 'Potentially Hallucinated / Unsupported' (Red)
    """
    summary_sentences = split_sentences(summary_text)
    if not summary_sentences:
        return []

    original_sentences = split_sentences(original_text)
    if not original_sentences:
        return [{"sentence": s, "support_score": 0.0, "status": "Unsupported"} for s in summary_sentences]

    nli_model = load_nli_cross_encoder()

    results = []
    for summ_sent in summary_sentences:
        max_score = 0.0
        verification_method = "token_overlap_fallback"
        summary_tokens = set(tokenize_and_lemmatize(summ_sent))
        ranked_source_sentences = sorted(
            original_sentences,
            key=lambda source_sentence: len(
                summary_tokens.intersection(set(tokenize_and_lemmatize(source_sentence)))
            ),
            reverse=True,
        )
        evidence_sentences = ranked_source_sentences[:30]

        if nli_model is not None:
            try:
                pairs = [(source_sentence, summ_sent) for source_sentence in evidence_sentences]
                logits = np.asarray(nli_model.predict(pairs, batch_size=32, show_progress_bar=False))
                if logits.ndim == 2 and logits.shape[1] == 3:
                    labels = getattr(
                        getattr(getattr(nli_model, "model", None), "config", None),
                        "id2label",
                        {},
                    )
                    entailment_index = next(
                        (
                            int(index)
                            for index, label in labels.items()
                            if "entail" in str(label).casefold()
                        ),
                        2,
                    )
                    shifted = logits - np.max(logits, axis=1, keepdims=True)
                    probabilities = np.exp(shifted)
                    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
                    max_score = float(np.max(probabilities[:, entailment_index]))
                    verification_method = "nli_cross_encoder"
                else:
                    raise ValueError(f"Expected 3-class NLI logits, received shape {logits.shape}.")
            except Exception as e:
                logger.warning("NLI inference failed; using lexical support heuristic: %s", e)

        else:
            logger.warning("NLI model unavailable; faithfulness result uses lexical overlap only.")

        if verification_method == "token_overlap_fallback" and summary_tokens:
            max_score = max(
                (
                    len(summary_tokens.intersection(set(tokenize_and_lemmatize(source_sentence))))
                    / len(summary_tokens)
                    for source_sentence in original_sentences
                ),
                default=0.0,
            )

        max_score = min(1.0, max(0.0, max_score))
        status = "Supported" if max_score >= 0.45 else "Potentially Unsupported / Hallucinated"

        results.append({
            "sentence": summ_sent,
            "support_score": round(max_score, 3),
            "status": status,
            "verification_method": verification_method,
        })

    return results
