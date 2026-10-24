"""
Smart Summarization Pipeline for SmartSum.
Implements:
1. Content analysis (length, structure, language, complexity)
2. Smart Strategy selection (Smart Auto, Extractive, Abstractive, Hybrid)
3. Length mapping (Short, Balanced, Detailed)
4. Focus-aware synthesis
5. Long document Map-Reduce chunking
6. Rigorous, authentic quality metrics calculation (Flesch Readability, Keyword Coverage, Compression)
"""
import re
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import config

from core.preprocessing import split_sentences, detect_language, tokenize_and_lemmatize
from core.extractive import (
    textrank_summarize,
    tfidf_summarize,
    lsa_summarize,
    select_important_sentences,
)
from core.abstractive import distilbart_summarize, t5_summarize, gemini_summarize
from core.query_focused import query_focused_summarize, get_relevant_sentences
from core.long_doc import hierarchical_summarize
from core.multilingual import process_multilingual_summarization
from core.styles import apply_summary_style, apply_summary_style_html

logger = logging.getLogger("SmartSum.SmartPipeline")

SCANNED_PDF_WARNING = "[Warning: Scanned image PDF detected. No extractable text found.]"
MIN_SUMMARIZABLE_SENTENCES = 3


def count_syllables_word(word: str) -> int:
    """Estimate syllable count in an English word using phonetic vowel heuristics."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Count vowel groups
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    syllables = len(re.findall(r'[aeiouy]{1,2}', word))
    return max(1, syllables)


def calculate_flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score:
    206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    Higher score indicates easier to read. Standard scale: 0 - 100.
    """
    sentences = split_sentences(text)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words or not sentences:
        return 65.0
    total_words = len(words)
    total_sentences = max(1, len(sentences))
    total_syllables = sum(count_syllables_word(w) for w in words)
    
    score = 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))
    return round(max(0.0, min(100.0, score)), 1)


def analyze_content(text: str) -> Dict[str, Any]:
    """
    Analyze input text structure, length, complexity, and language.
    """
    text = (text or "").strip()
    words = text.split()
    word_count = len(words)
    sentences = split_sentences(text)
    sentence_count = len(sentences)
    avg_sentence_len = round(word_count / max(1, sentence_count), 1)
    
    # Lexical richness (Type-Token Ratio)
    clean_tokens = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    unique_tokens = set(clean_tokens)
    lexical_diversity = round((len(unique_tokens) / max(1, len(clean_tokens))) * 100, 1)

    lang = detect_language(text[:1000]) if text else "en"

    if word_count < 150:
        complexity = "Compact"
    elif word_count < 800:
        complexity = "Moderate"
    else:
        complexity = "Extensive"

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "lexical_diversity": lexical_diversity,
        "language": lang,
        "complexity": complexity,
    }


def map_length_to_sentences(total_sentences: int, length_preset: str) -> int:
    """Map user-friendly length presets (Short, Balanced, Detailed) to target sentence count."""
    total = max(1, total_sentences)
    preset = (length_preset or "Balanced").lower()
    short_target = min(total, max(1, min(3, int(math.ceil(total * 0.20)))))
    balanced_target = min(
        total,
        max(short_target + 1, min(20, int(math.ceil(total * 0.65)))),
    )
    detailed_target = min(
        total,
        max(balanced_target + 1, min(26, int(math.ceil(total * 0.90)))),
    )

    if "short" in preset:
        return short_target
    if "detail" in preset:
        return detailed_target
    return balanced_target


def map_length_to_abstractive_limits(length_preset: str) -> Tuple[int, int]:
    """Return (max_length, min_length) token parameters for abstractive models."""
    preset = (length_preset or "Balanced").lower()
    if "short" in preset:
        return (70, 20)
    elif "detail" in preset:
        return (260, 75)
    else:  # Balanced
        return (140, 35)


def hybrid_summarize(
    text: str,
    target_sentences: int,
    abstractive_model_name: str = "auto",
    style: str = "Paragraph",
    has_gemini: bool = False,
    max_length: int = 160,
    min_length: int = 35,
    on_fallback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Hybrid Summarization:
    1. Extractive stage: Extract key salient sentences using TextRank.
    2. Abstractive stage: Re-synthesize and smooth extracted content using neural model or Gemini.
    """
    # Give the model a broad, diverse candidate set, but keep inference within context limits.
    interim_target = max(
        target_sentences,
        min(18, int(math.ceil(target_sentences * 1.15))),
    )
    extracted_context = textrank_summarize(text, num_sentences=interim_target)

    if not extracted_context.strip():
        extracted_context = text

    fallback_reasons = []

    def track_fallback(reason: str) -> None:
        fallback_reasons.append(reason)
        if on_fallback is not None:
            on_fallback(reason)

    # Abstractive pass
    if (abstractive_model_name == "gemini" or (abstractive_model_name == "auto" and has_gemini)) and has_gemini:
        refined_summary = gemini_summarize(
            extracted_context,
            style=style,
            on_fallback=track_fallback,
        )
    elif "t5" in abstractive_model_name.lower():
        refined_summary = t5_summarize(
            extracted_context,
            max_length=max_length,
            min_length=min_length,
            on_fallback=track_fallback,
        )
    else:
        # DistilBART default or fallback
        refined_summary = distilbart_summarize(
            extracted_context,
            max_length=max_length,
            min_length=min_length,
            on_fallback=track_fallback,
        )

    if not refined_summary.strip():
        logger.warning(
            "Abstractive refinement returned an empty summary; preserving the extractive target."
        )
        if on_fallback is not None:
            on_fallback("textrank_empty_generation")
        return textrank_summarize(text, num_sentences=target_sentences)

    if fallback_reasons:
        return refined_summary

    preserved_points = select_important_sentences(
        text,
        num_sentences=target_sentences,
        seed_text=refined_summary,
    )
    return " ".join([refined_summary, *preserved_points]).strip()


def determine_smart_strategy(
    analysis: Dict[str, Any],
    focus: str = "",
    has_gemini: bool = False,
    lite_mode: bool = False,
    length_preset: str = "Balanced",
) -> Dict[str, Any]:
    """
    Intelligently select the most suitable summarization strategy based on content analysis.
    """
    words = analysis["word_count"]
    lang = analysis["language"]
    has_focus = bool(focus and focus.strip())
    short_requested = "short" in (length_preset or "Balanced").lower()

    if has_focus:
        strategy_name = "Focus-Guided Extractive & Semantic Filter"
        base_method = "query_focused"
        rationale = f"Focus query detected. Prioritizing sections directly addressing '{focus.strip()[:40]}'."
    elif words > 800:
        strategy_name = "Hierarchical Map-Reduce (Long Document)"
        base_method = "hierarchical"
        rationale = f"Document is extensive ({words:,} words). Using chunk-level extraction and multi-stage synthesis."
    elif not lite_mode and words >= 120 and lang == "en" and short_requested:
        strategy_name = "High-Priority Extractive (TextRank)"
        base_method = "textrank"
        rationale = "Short summaries prioritize faithful selection of the source's highest-salience sentences."
    elif has_gemini:
        strategy_name = "Neural Abstractive (Gemini AI)"
        base_method = "gemini"
        rationale = "High-performance Gemini AI selected for abstractive conceptual synthesis."
    elif not lite_mode and words >= 120 and lang == "en":
        strategy_name = "Hybrid Neural (TextRank + DistilBART)"
        base_method = "hybrid"
        rationale = "Combined key sentence extraction with neural abstractive refinement for maximum clarity."
    elif lite_mode:
        strategy_name = "High-Speed Extractive (TextRank)"
        base_method = "textrank"
        rationale = "Lite mode active. Optimized Graph PageRank selected for fast, zero-latency summarization."
    else:
        strategy_name = "Salience Graph Ranking (TextRank)"
        base_method = "textrank"
        rationale = "Graph-based sentence salience ranking chosen for optimal factual preservation."

    return {
        "strategy_name": strategy_name,
        "base_method": base_method,
        "rationale": rationale,
        "needs_chunking": words > 800,
    }


def execute_smart_pipeline(
    raw_input: str,
    summarization_mode: str = "Smart Auto",
    length_preset: str = "Balanced",
    output_format: str = "Bullet Points",
    focus: str = "",
    target_language: str = "en",
    manual_extractive_algo: str = "Automatic",
    manual_ai_model: str = "Automatic",
    manual_sentences: Optional[int] = None,
    has_gemini: bool = False,
    lite_mode: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Full end-to-end execution of the SmartSum summarization pipeline.
    """
    text = (raw_input or "").strip()
    if not text:
        raise ValueError("No text provided for summarization.")
    if text.casefold() == SCANNED_PDF_WARNING.casefold():
        raise ValueError(
            "Scanned image PDF detected. No extractable text found. "
            "OCR is required to extract text from this scanned PDF."
        )

    if progress_callback:
        progress_callback(0.15, "Analyzing document structure & complexity...")

    analysis = analyze_content(text)
    total_sentences = analysis["sentence_count"]
    if total_sentences < MIN_SUMMARIZABLE_SENTENCES:
        raise ValueError(
            "Source is too short to summarize reliably; provide at least three complete sentences."
        )

    # Determine target sentences
    if manual_sentences is not None and manual_sentences > 0:
        num_sentences = min(manual_sentences, max(1, total_sentences))
    else:
        num_sentences = map_length_to_sentences(total_sentences, length_preset)

    max_tokens, min_tokens = map_length_to_abstractive_limits(length_preset)

    # Resolve strategy
    mode_lower = (summarization_mode or "Smart Auto").lower()
    
    if "auto" in mode_lower:
        decision = determine_smart_strategy(
            analysis,
            focus=focus,
            has_gemini=has_gemini,
            lite_mode=lite_mode,
            length_preset=length_preset,
        )
        chosen_strategy_title = decision["strategy_name"]
        chosen_rationale = decision["rationale"]
        strategy_type = decision["base_method"]
    elif "extract" in mode_lower:
        algo = manual_extractive_algo.lower()
        if "tfidf" in algo or "tf-idf" in algo:
            strategy_type = "tfidf"
            chosen_strategy_title = "Extractive (TF-IDF Term Scoring)"
        elif "lsa" in algo:
            strategy_type = "lsa"
            chosen_strategy_title = "Extractive (Latent Semantic Analysis)"
        else:
            strategy_type = "textrank"
            chosen_strategy_title = "Extractive (TextRank PageRank)"
        chosen_rationale = "User-selected extractive salience extraction preserving exact original phrasing."
    elif "abstract" in mode_lower:
        ai_choice = manual_ai_model.lower()
        if "gemini" in ai_choice and has_gemini:
            strategy_type = "gemini"
            chosen_strategy_title = "Abstractive (Gemini AI)"
        elif "t5" in ai_choice:
            strategy_type = "t5"
            chosen_strategy_title = "Abstractive (T5-Small)"
        elif not lite_mode:
            strategy_type = "distilbart"
            chosen_strategy_title = "Abstractive (DistilBART-CNN)"
        else:
            strategy_type = "t5"
            chosen_strategy_title = "Abstractive (T5-Small)"
        chosen_rationale = "User-selected neural abstractive model generating new synthesized text."
    elif "hybrid" in mode_lower:
        strategy_type = "hybrid"
        if has_gemini:
            chosen_strategy_title = "Hybrid (TextRank + Gemini)"
        elif lite_mode:
            chosen_strategy_title = "Hybrid (TextRank + T5-Small)"
        else:
            chosen_strategy_title = "Hybrid Neural (TextRank + DistilBART)"
        chosen_rationale = "User-selected two-stage hybrid pipeline."
    else:
        strategy_type = "textrank"
        chosen_strategy_title = "Extractive (TextRank)"
        chosen_rationale = "Default fallback strategy."

    if progress_callback:
        progress_callback(0.40, f"Running {chosen_strategy_title}...")

    fallback_methods = []

    def record_fallback(method: str) -> None:
        fallback_methods.append(method)

    # Define core worker function
    def base_worker(chunk_text: str) -> str:
        if strategy_type == "query_focused":
            return query_focused_summarize(chunk_text, query=focus.strip(), num_sentences=num_sentences)

        # Apply focus as a retrieval filter while still honoring an explicitly
        # selected summarizer (including abstractive and hybrid strategies).
        if focus and focus.strip():
            focused_sentences = get_relevant_sentences(
                chunk_text,
                query=focus.strip(),
                top_k=max(5, num_sentences * 2),
            )
            if focused_sentences:
                chunk_text = " ".join(focused_sentences)

        if strategy_type in ["textrank", "hierarchical"]:
            return textrank_summarize(chunk_text, num_sentences=num_sentences)
        elif strategy_type == "tfidf":
            return tfidf_summarize(chunk_text, num_sentences=num_sentences)
        elif strategy_type == "lsa":
            return lsa_summarize(chunk_text, num_sentences=num_sentences)
        elif strategy_type == "gemini":
            return gemini_summarize(
                chunk_text,
                style=output_format,
                on_fallback=record_fallback,
            )
        elif strategy_type == "distilbart":
            return distilbart_summarize(
                chunk_text,
                max_length=max_tokens,
                min_length=min_tokens,
                on_fallback=record_fallback,
            )
        elif strategy_type == "t5":
            return t5_summarize(
                chunk_text,
                max_length=max_tokens,
                min_length=min_tokens,
                on_fallback=record_fallback,
            )
        elif strategy_type == "hybrid":
            return hybrid_summarize(
                chunk_text,
                target_sentences=num_sentences,
                abstractive_model_name="gemini" if has_gemini else ("distilbart" if not lite_mode else "t5"),
                style=output_format,
                has_gemini=has_gemini,
                max_length=max_tokens,
                min_length=min_tokens,
                on_fallback=record_fallback,
            )
        elif strategy_type == "query_focused":
            return query_focused_summarize(chunk_text, query=focus.strip(), num_sentences=num_sentences)
        else:
            return textrank_summarize(chunk_text, num_sentences=num_sentences)

    # Handle long documents (> 750 words)
    if len(text.split()) > 750 and strategy_type != "gemini":
        if progress_callback:
            progress_callback(0.50, "Executing Map-Reduce chunking for long document...")
        raw_summary = hierarchical_summarize(
            text,
            base_worker,
            max_words_per_chunk=600,
            progress_callback=progress_callback
        )
    else:
        # Multilingual pipeline handling
        if progress_callback:
            progress_callback(0.60, f"Synthesizing summary in {target_language.upper()}...")
        raw_summary = process_multilingual_summarization(text, target_language, base_worker)

    if progress_callback:
        progress_callback(0.85, f"Formatting output as {output_format} & evaluating...")

    # Apply output format styling
    styled_plain = apply_summary_style(raw_summary, output_format)
    styled_html = apply_summary_style_html(raw_summary, output_format)

    # Compute authentic metrics
    metrics = calculate_quality_metrics(text, raw_summary)

    if progress_callback:
        progress_callback(1.0, "Summary generation complete!")

    worker_strategy = "textrank" if strategy_type == "hierarchical" else strategy_type
    execution_path = (
        f"hierarchical_map_reduce+{worker_strategy}"
        if len(text.split()) > 750 and strategy_type != "gemini"
        else strategy_type
    )
    if fallback_methods:
        fallback_label = " + ".join(dict.fromkeys(fallback_methods))
        if strategy_type == "hybrid" and "textrank" in fallback_methods:
            chosen_strategy_title = "TextRank (runtime fallback; DistilBART failed)"
            chosen_rationale = (
                f"{chosen_rationale} DistilBART was unavailable or failed; "
                "the summary was produced with TextRank."
            )
            execution_path = "textrank_fallback"
        else:
            chosen_strategy_title = f"{chosen_strategy_title} (runtime fallback: {fallback_label})"
            chosen_rationale = f"{chosen_rationale} Runtime fallback used: {fallback_label}."
            execution_path = f"{execution_path}+fallback:{fallback_label.replace(' ', '_')}"

    return {
        "raw_summary": raw_summary,
        "styled_plain": styled_plain,
        "styled_html": styled_html,
        "strategy_title": chosen_strategy_title,
        "strategy_rationale": chosen_rationale,
        "execution_path": execution_path,
        "metrics": metrics,
        "analysis": analysis,
        "num_sentences": num_sentences,
    }


def calculate_quality_metrics(original_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Calculate genuine, mathematically verifiable summary quality metrics.
    NO fabricated or fake percentages.
    """
    orig_words = len(re.findall(r'\b\w+\b', original_text or ""))
    summ_words = len(re.findall(r'\b\w+\b', summary_text or ""))
    orig_sents = max(1, len(split_sentences(original_text or "")))
    summ_sents = max(1, len(split_sentences(summary_text or "")))

    # 1. Compression Ratio: % of text reduced
    compression = round((1 - (summ_words / max(1, orig_words))) * 100, 1) if orig_words > 0 else 0.0
    compression = max(0.0, min(100.0, compression))

    # 2. Reading time saved (based on 200 WPM)
    orig_time_min = round(orig_words / 200.0, 2)
    summ_time_min = round(summ_words / 200.0, 2)
    time_saved_min = round(max(0.0, orig_time_min - summ_time_min), 2)

    # 3. Readability Score: Flesch Reading Ease
    readability = calculate_flesch_reading_ease(summary_text)

    # 4. Lexical coverage: distinct normalized source terms present in the summary
    orig_tokens = set([t.lower() for t in tokenize_and_lemmatize(original_text) if len(t) > 3])
    summ_tokens = set([t.lower() for t in tokenize_and_lemmatize(summary_text) if len(t) > 3])
    
    if orig_tokens:
        overlap = orig_tokens.intersection(summ_tokens)
        keyword_coverage = round((len(overlap) / len(orig_tokens)) * 100, 1)
    else:
        keyword_coverage = 0.0

    return {
        "orig_words": orig_words,
        "summ_words": summ_words,
        "orig_sents": orig_sents,
        "summ_sents": summ_sents,
        "compression_ratio": compression,
        "time_saved_min": time_saved_min,
        "readability_score": readability,
        "keyword_coverage": min(100.0, keyword_coverage),
    }
