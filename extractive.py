"""
Extractive Summarization Module for SmartSum.
Includes:
- TextRank (NetworkX Graph PageRank on TF-IDF / Cosine similarity)
- TF-IDF Sentence Scoring
- LSA (Latent Semantic Analysis via TruncatedSVD / Sumy fallback)
"""
import math
import logging
import re
from functools import lru_cache
from typing import List, Tuple
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

from core.preprocessing import split_sentences, tokenize_and_lemmatize

logger = logging.getLogger("SmartSum.Extractive")


def _normalize_num_sentences(num_sentences: int) -> int:
    """Coerce sentence counts to a safe positive integer for summary generation."""
    if num_sentences is None:
        return 3
    try:
        normalized = int(num_sentences)
    except (TypeError, ValueError):
        return 3
    return max(1, normalized)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Scale sentence scores to [0, 1], preserving ties for flat score vectors."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        return scores
    score_range = float(np.ptp(scores))
    if score_range <= 1e-12:
        return np.full(len(scores), 0.5)
    return (scores - scores.min()) / score_range


def _select_diverse_sentences(
    sentences: List[str],
    tfidf_matrix,
    feature_names: np.ndarray,
    importance_scores: np.ndarray,
    num_sentences: int,
    semantic_scores: np.ndarray = None,
    semantic_similarity=None,
    term_weights=None,
    key_terms=None,
    seed_text: str = "",
    section_groups=None,
) -> List[int]:
    """Select salient sentences with coverage-aware MMR across source sections."""
    count = min(_normalize_num_sentences(num_sentences), len(sentences))
    if count >= len(sentences) and not seed_text.strip():
        return list(range(len(sentences)))
    lexical_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    if semantic_similarity is None:
        semantic_similarity = lexical_similarity
    semantic_relevance = (
        _normalize_scores(semantic_scores)
        if semantic_scores is not None
        else _normalize_scores(importance_scores)
    )
    term_to_column = {term: index for index, term in enumerate(feature_names)}
    key_term_columns = {
        term_to_column[term] for term in (key_terms or ()) if term in term_to_column
    }
    term_weights = (
        np.asarray(term_weights, dtype=float)
        if term_weights is not None
        else np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    )
    if len(key_term_columns):
        term_weights[list(key_term_columns)] *= 1.75

    sentence_tokens = [
        set(tfidf_matrix.getrow(i).indices)
        for i in range(len(sentences))
    ]
    relevance = _normalize_scores(importance_scores)
    selected: List[int] = []
    selected_tokens = {
        term_to_column[token]
        for token in tokenize_and_lemmatize(seed_text)
        if token in term_to_column
    }
    seed_texts = {
        " ".join(sentence.casefold().split())
        for sentence in split_sentences(seed_text)
    }
    selected_texts = set(seed_texts)
    remaining = set(range(len(sentences)))
    if seed_texts:
        seed_columns = list(selected_tokens)
        seed_vector = csr_matrix(
            (np.ones(len(seed_columns)), (np.zeros(len(seed_columns)), seed_columns)),
            shape=(1, tfidf_matrix.shape[1]),
        )
        seed_similarity = cosine_similarity(tfidf_matrix, seed_vector).ravel()
    else:
        seed_similarity = np.zeros(len(sentences))

    if not section_groups:
        section_groups = [
            [int(index) for index in group]
            for group in np.array_split(np.arange(len(sentences)), count)
            if len(group)
        ]
    section_ids = {}
    for section_id, group in enumerate(section_groups):
        for index in group:
            section_ids[index] = section_id
    selected_sections = set()
    position = np.arange(len(sentences), dtype=float) / max(1, len(sentences) - 1)
    position_scores = np.maximum(np.exp(-4.0 * position), np.exp(-4.0 * (1.0 - position)))
    position_scores = _normalize_scores(position_scores)

    def is_redundant(index: int) -> bool:
        normalized_text = " ".join(sentences[index].casefold().split())
        if normalized_text in selected_texts:
            return True
        if not selected and not selected_tokens:
            return False
        token_set = sentence_tokens[index]
        novelty = len(token_set - selected_tokens) / max(1, len(token_set))
        prior_similarity = max(
            [float(lexical_similarity[index, chosen]) for chosen in selected]
            + [float(seed_similarity[index])]
        )
        return prior_similarity >= 0.72 and novelty < 0.30

    def add_sentence(index: int) -> None:
        selected.append(index)
        selected_tokens.update(sentence_tokens[index])
        selected_sections.add(int(section_ids[index]))
        selected_texts.add(" ".join(sentences[index].casefold().split()))

    def marginal_score(index: int) -> float:
        token_set = sentence_tokens[index]
        new_term_weight = sum(
            term_weights[column] for column in token_set - selected_tokens
        )
        total_term_weight = sum(term_weights[column] for column in token_set)
        coverage_gain = new_term_weight / max(total_term_weight, 1e-12)
        key_token_set = token_set.intersection(key_term_columns)
        new_key_weight = sum(
            term_weights[column] for column in key_token_set - selected_tokens
        )
        total_key_weight = sum(term_weights[column] for column in key_token_set)
        key_coverage_gain = new_key_weight / max(total_key_weight, 1e-12)
        lexical_redundancy = max(
            [float(lexical_similarity[index, chosen]) for chosen in selected]
            + [float(seed_similarity[index])]
        )
        semantic_redundancy = max(
            (float(semantic_similarity[index, chosen]) for chosen in selected),
            default=float(seed_similarity[index]),
        )
        redundancy = max(lexical_redundancy, semantic_redundancy)
        section_novelty = 1.0 if section_ids[index] not in selected_sections else 0.0
        return (
            0.20 * relevance[index]
            + 0.25 * coverage_gain
            + 0.25 * key_coverage_gain
            + 0.10 * semantic_relevance[index]
            + 0.05 * position_scores[index]
            + 0.08 * section_novelty
            + 0.17 * (1.0 - redundancy)
        )

    for section in section_groups:
        candidates = [index for index in section if index in remaining]
        while candidates and len(selected) < count:
            best_index = max(candidates, key=marginal_score)
            candidates.remove(best_index)
            remaining.remove(best_index)
            if is_redundant(best_index):
                continue
            add_sentence(best_index)
            break

    while remaining and len(selected) < count:
        best_index = max(remaining, key=marginal_score)
        remaining.remove(best_index)
        if is_redundant(best_index):
            continue
        add_sentence(best_index)

    return sorted(selected)


@lru_cache(maxsize=1)
def _load_key_term_nlp():
    """Load the configured local English model once for noun/entity signals."""
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser"])


def _extract_key_terms(sentences: List[str], feature_names: np.ndarray) -> set:
    """Identify source nouns and named entities when the local spaCy model is available."""
    feature_set = set(feature_names)
    try:
        nlp = _load_key_term_nlp()
        key_terms = set()
        for doc in nlp.pipe(sentences, batch_size=64):
            key_terms.update(
                token.lemma_.lower()
                for token in doc
                if (token.pos_ in {"NOUN", "PROPN"} or token.ent_type_)
                and token.lemma_.lower() in feature_set
            )
        return key_terms
    except (ImportError, OSError) as error:
        logger.warning(
            "Noun/entity weighting unavailable (%s: %s); continuing with TF-IDF terms.",
            type(error).__name__,
            error,
        )
        return set()


def _source_section_groups(text: str, sentences: List[str], count: int) -> List[List[int]]:
    """Map sentences to paragraphs when boundaries are recoverable, else use source spans."""
    paragraphs = re.split(r"(?:\r?\n\s*\r?\n|\s{2,})", text.strip())
    merged_paragraphs = []
    pending = ""
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        pending = f"{pending} {paragraph.strip()}".strip()
        if re.search(r"[.!?][\"')\]]*$", pending):
            merged_paragraphs.append(pending)
            pending = ""
    if pending:
        merged_paragraphs.append(pending)

    paragraph_sentences = [split_sentences(paragraph) for paragraph in merged_paragraphs]
    flattened = [sentence for group in paragraph_sentences for sentence in group]
    normalize = lambda sentence: " ".join(sentence.casefold().split())
    if len(paragraph_sentences) > 1 and list(map(normalize, flattened)) == list(
        map(normalize, sentences)
    ):
        groups = []
        offset = 0
        for group in paragraph_sentences:
            groups.append(list(range(offset, offset + len(group))))
            offset += len(group)
        return groups

    return [
        [int(index) for index in group]
        for group in np.array_split(np.arange(len(sentences)), min(count, len(sentences)))
        if len(group)
    ]


def select_important_sentences(
    text: str,
    num_sentences: int,
    seed_text: str = "",
) -> List[str]:
    """Return source sentences ranked for salience, novelty, and concept coverage."""
    num_sentences = _normalize_num_sentences(num_sentences)
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= num_sentences and not seed_text.strip():
        return sentences

    corpus = [" ".join(tokenize_and_lemmatize(sentence)) for sentence in sentences]
    if not any(item.strip() for item in corpus):
        return sentences[:num_sentences]

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        np.fill_diagonal(similarity, 0.0)
        graph = nx.from_numpy_array(similarity)
        pagerank = nx.pagerank(graph, max_iter=200, tol=1e-4)
        pagerank_scores = np.asarray([pagerank[index] for index in range(len(sentences))])
        tfidf_scores = _tfidf_sentence_scores(tfidf_matrix)

        semantic_scores = tfidf_scores
        semantic_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        component_count = min(12, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if component_count >= 2:
            latent_vectors = TruncatedSVD(
                n_components=component_count,
                random_state=42,
            ).fit_transform(tfidf_matrix)
            document_centroid = latent_vectors.mean(axis=0, keepdims=True)
            semantic_scores = cosine_similarity(latent_vectors, document_centroid).ravel()
            semantic_similarity = np.clip(
                cosine_similarity(latent_vectors, latent_vectors),
                0.0,
                1.0,
            )

        importance = (
            0.50 * _normalize_scores(pagerank_scores)
            + 0.50 * _normalize_scores(tfidf_scores)
        )
        selected_indices = _select_diverse_sentences(
            sentences,
            tfidf_matrix,
            vectorizer.get_feature_names_out(),
            importance,
            num_sentences,
            semantic_scores=semantic_scores,
            semantic_similarity=semantic_similarity,
            term_weights=vectorizer.idf_,
            key_terms=_extract_key_terms(sentences, vectorizer.get_feature_names_out()),
            seed_text=seed_text,
            section_groups=_source_section_groups(text, sentences, num_sentences),
        )
        return [sentences[index] for index in selected_indices]
    except Exception:
        logger.exception("Diverse sentence selection failed; using source-order selection.")
        return sentences[:num_sentences]


def _tfidf_sentence_scores(tfidf_matrix) -> np.ndarray:
    """Score sentences by their mean non-zero term weight."""
    scores = []
    for index in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(index).data
        scores.append(float(np.mean(row)) if len(row) else 0.0)
    return np.asarray(scores, dtype=float)


def textrank_summarize(text: str, num_sentences: int = 3) -> str:
    """
    Summarize text using Graph-based TextRank algorithm.
    1. Segment text into sentences.
    2. Build TF-IDF sentence matrix.
    3. Compute cosine similarity matrix between sentences.
    4. Construct graph and run PageRank to rank sentences.
    5. Return top sentences ordered by original position.
    """
    return " ".join(select_important_sentences(text, num_sentences))


def tfidf_summarize(text: str, num_sentences: int = 3) -> str:
    """
    Summarize text by scoring sentences based on cumulative TF-IDF word weights.
    """
    return " ".join(select_important_sentences(text, num_sentences))


def lsa_summarize(text: str, num_sentences: int = 3) -> str:
    """
    Summarize text using Latent Semantic Analysis (LSA / SVD on TF-IDF matrix).
    """
    num_sentences = _normalize_num_sentences(num_sentences)
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    corpus = [" ".join(tokenize_and_lemmatize(s)) for s in sentences]
    non_empty = [c for c in corpus if c.strip()]
    if not non_empty:
        return " ".join(sentences[:num_sentences])

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        n_components = min(num_sentences, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if n_components < 1:
            return " ".join(sentences[:num_sentences])

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_matrix = svd.fit_transform(tfidf_matrix)

        # Combine latent-concept salience with term salience before diverse selection.
        lsa_scores = np.linalg.norm(svd_matrix, axis=1)
        importance = (
            0.75 * _normalize_scores(lsa_scores)
            + 0.25 * _normalize_scores(_tfidf_sentence_scores(tfidf_matrix))
        )
        ranked_indices = _select_diverse_sentences(
            sentences, tfidf_matrix, vectorizer.get_feature_names_out(), importance, num_sentences
        )
        selected = [sentences[i] for i in ranked_indices]
        return " ".join(selected)
    except Exception as e:
        logger.error(f"LSA summarizer error: {e}")
        return " ".join(sentences[:num_sentences])
