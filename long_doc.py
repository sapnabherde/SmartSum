"""
Long Document Hierarchical Summarization (Map-Reduce) for SmartSum.
Chunks input documents exceeding token limits and hierarchically combines summaries.
"""
import logging
import re
from typing import List, Callable
from core.preprocessing import split_sentences

logger = logging.getLogger("SmartSum.LongDoc")


def chunk_text_by_words(text: str, max_words_per_chunk: int = 600, overlap_words: int = 50) -> List[str]:
    """
    Split a long text document at sentence boundaries with bounded word overlap.
    """
    if not text or not text.strip():
        return []

    if max_words_per_chunk is None:
        max_words_per_chunk = 600
    try:
        max_words_per_chunk = int(max_words_per_chunk)
    except (TypeError, ValueError):
        max_words_per_chunk = 600

    if max_words_per_chunk <= 0:
        return [text]

    try:
        overlap_words = int(overlap_words)
    except (TypeError, ValueError):
        overlap_words = 0

    overlap_words = max(0, min(overlap_words, max_words_per_chunk - 1))
    if len(text.split()) <= max_words_per_chunk:
        return [text]

    units = []
    for sentence in split_sentences(text) or [text]:
        words = sentence.split()
        if len(words) <= max_words_per_chunk:
            units.append(sentence)
        else:
            units.extend(
                " ".join(words[i : i + max_words_per_chunk])
                for i in range(0, len(words), max_words_per_chunk)
            )

    chunks = []
    start = 0
    while start < len(units):
        end = start
        chunk_word_count = 0
        while end < len(units):
            unit_word_count = len(units[end].split())
            if end > start and chunk_word_count + unit_word_count > max_words_per_chunk:
                break
            chunk_word_count += unit_word_count
            end += 1

        chunks.append(" ".join(units[start:end]))
        if end >= len(units):
            break

        next_start = end
        overlap_count = 0
        while next_start > start:
            previous_count = len(units[next_start - 1].split())
            if overlap_count + previous_count > overlap_words:
                break
            overlap_count += previous_count
            next_start -= 1
        start = max(start + 1, next_start)

    return chunks


def _deduplicate_sentences(text: str) -> str:
    """Remove exact repeated sentences introduced by overlapping chunks."""
    seen = set()
    unique_sentences = []
    for sentence in split_sentences(text):
        key = re.sub(r"\W+", " ", sentence.casefold()).strip()
        if key and key not in seen:
            seen.add(key)
            unique_sentences.append(sentence)
    return " ".join(unique_sentences)


def _preserve_terminal_sentence(source_text: str, summary_text: str) -> str:
    """Retain the last source sentence if a map or reduce summary omitted it."""
    source_sentences = split_sentences(source_text)
    if not source_sentences:
        return summary_text

    terminal_sentence = source_sentences[-1]
    terminal_key = re.sub(r"\W+", " ", terminal_sentence.casefold()).strip()
    summary_keys = {
        re.sub(r"\W+", " ", sentence.casefold()).strip()
        for sentence in split_sentences(summary_text)
    }
    if terminal_key and terminal_key not in summary_keys:
        return f"{summary_text.strip()} {terminal_sentence}".strip()
    return summary_text


def hierarchical_summarize(
    text: str,
    summarize_func: Callable[[str], str],
    max_words_per_chunk: int = 600,
    progress_callback: Callable[[float, str], None] = None,
    overlap_words: int = 50,
) -> str:
    """
    Perform Map-Reduce hierarchical summarization:
    1. Chunk long input text into manageable pieces.
    2. Map Step: Summarize each chunk independently.
    3. Reduce Step: Combine partial summaries and perform final summary pass.
    """
    chunks = chunk_text_by_words(
        text,
        max_words_per_chunk=max_words_per_chunk,
        overlap_words=overlap_words,
    )
    if not chunks:
        return ""
    
    if len(chunks) == 1:
        if progress_callback:
            progress_callback(1.0, "Processing single chunk...")
        return summarize_func(chunks[0])

    logger.info(f"Hierarchical Map-Reduce triggered for {len(chunks)} chunks.")
    chunk_summaries = []

    # Map Step
    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress = (idx + 1) / (len(chunks) + 1)
            progress_callback(progress, f"Summarizing chunk {idx + 1} of {len(chunks)}...")
        
        part_summary = _preserve_terminal_sentence(chunk, summarize_func(chunk))
        if part_summary and part_summary.strip():
            chunk_summaries.append(part_summary.strip())

    if not chunk_summaries:
        return ""

    combined_summary_input = _deduplicate_sentences(" ".join(chunk_summaries))
    while len(combined_summary_input.split()) > max_words_per_chunk:
        if progress_callback:
            progress_callback(0.95, "Synthesizing final meta-summary...")
        reduce_chunks = chunk_text_by_words(
            combined_summary_input,
            max_words_per_chunk=max_words_per_chunk,
            overlap_words=0,
        )
        reduced = [
            _preserve_terminal_sentence(chunk, summarize_func(chunk)).strip()
            for chunk in reduce_chunks
            if chunk.strip()
        ]
        next_summary = _deduplicate_sentences(" ".join(summary for summary in reduced if summary))
        if not next_summary or len(next_summary.split()) >= len(combined_summary_input.split()):
            logger.warning("Map-Reduce stopped because the reduce step did not shorten its input.")
            return combined_summary_input
        combined_summary_input = next_summary

    if progress_callback:
        progress_callback(0.95, "Synthesizing final meta-summary...")
    final_summary = _preserve_terminal_sentence(
        combined_summary_input,
        summarize_func(combined_summary_input),
    )

    if progress_callback:
        progress_callback(1.0, "Summarization complete!")

    return final_summary
