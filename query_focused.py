"""
Query-Focused Summarization Module for SmartSum.
Retrieves sentences most relevant to a user query using embedding similarity / TF-IDF,
then summarizes the relevant subset.
"""
import logging
from typing import List
import numpy as np
import streamlit as st
import config
from core.preprocessing import split_sentences, tokenize_and_lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("SmartSum.QueryFocused")

@st.cache_resource(show_spinner="Loading Sentence Embedding Model...")
def load_sentence_transformer():
    """Lazy-load SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.MODELS["sentence_transformer"], device=config.DEVICE)
        return model
    except Exception as e:
        logger.error(f"Failed to load sentence-transformers model: {e}")
        return None


def get_relevant_sentences(text: str, query: str, top_k: int = 5) -> List[str]:
    """
    Find top_k sentences from text that are most semantically similar to query.
    Uses sentence-transformers with TF-IDF similarity fallback.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= top_k:
        return sentences

    st_model = load_sentence_transformer()

    if st_model is not None:
        try:
            query_embedding = st_model.encode([query], convert_to_numpy=True)
            sentence_embeddings = st_model.encode(sentences, convert_to_numpy=True)

            similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_indices = sorted(top_indices)  # preserve reading order
            return [sentences[i] for i in top_indices]
        except Exception as e:
            logger.warning(f"SentenceTransformer embedding error ({e}). Using TF-IDF similarity fallback.")

    # TF-IDF Fallback
    try:
        corpus = [query] + [" ".join(tokenize_and_lemmatize(s)) for s in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_mat = vectorizer.fit_transform(corpus)

        query_vec = tfidf_mat[0:1]
        sent_vecs = tfidf_mat[1:]

        sims = cosine_similarity(query_vec, sent_vecs)[0]
        top_indices = sorted(np.argsort(sims)[::-1][:top_k])
        return [sentences[i] for i in top_indices]
    except Exception as e:
        logger.error(f"TF-IDF similarity fallback error: {e}")
        return sentences[:top_k]


def query_focused_summarize(text: str, query: str, num_sentences: int = 3) -> str:
    """
    Summarize document focusing specifically on content related to user's query.
    """
    if not query or not query.strip():
        from core.extractive import textrank_summarize
        return textrank_summarize(text, num_sentences=num_sentences)

    relevant = get_relevant_sentences(text, query, top_k=max(5, num_sentences * 2))
    if not relevant:
        return "No content matching your query was found in the input document."

    focused_context = " ".join(relevant)
    from core.extractive import textrank_summarize
    return textrank_summarize(focused_context, num_sentences=num_sentences)
