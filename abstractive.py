"""
Abstractive Summarization Module for SmartSum.
Supports DistilBART, T5-Small, and Google Gemini with cached model loading and timing logs.
"""
import time
import logging
import math
import re
from typing import Callable, Optional
import streamlit as st
import torch
import config
from core.long_doc import hierarchical_summarize

logger = logging.getLogger("SmartSum.Abstractive")


def _fallback_sentence_count(max_length: int) -> int:
    """Scale the extractive fallback with the requested abstractive length."""
    return max(1, min(12, math.ceil(max_length / 40)))


def _notify_fallback(callback: Optional[Callable[[str], None]], method: str) -> None:
    if callback is not None:
        callback(method)


def _clean_generated_summary(text: str) -> str:
    """Normalize tokenizer spacing around punctuation in decoded generations."""
    return re.sub(r"\s+", " ", re.sub(r"\s+([,.;:!?])", r"\1", text)).strip()


@st.cache_resource(show_spinner=False)
def _load_seq2seq_pipeline(model_name: str):
    """Load a cached encoder-decoder model using Transformers APIs stable across major versions."""
    start_time = time.time()
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading encoder-decoder summarization model '%s'...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(config.DEVICE)
        model.eval()

        max_source_tokens = (
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "n_positions", None)
            or 512
        )
        if max_source_tokens > 100000:
            max_source_tokens = 512

        def summarize(text: str, max_length: int, min_length: int, do_sample: bool = False):
            inputs = tokenizer(text, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > max_source_tokens:
                raise ValueError(
                    f"Input is {inputs['input_ids'].shape[1]} tokens; model limit is {max_source_tokens}."
                )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min(min_length, max_length - 1),
                    do_sample=do_sample,
                )
            return [{"generated_text": tokenizer.decode(output[0], skip_special_tokens=True)}]

        elapsed = time.time() - start_time
        logger.info("Summarization model '%s' loaded in %.2fs", model_name, elapsed)
        print(f"[SmartSum] Summarization model loaded in {elapsed:.2f}s")
        return summarize
    except Exception:
        logger.exception("Failed to load summarization model '%s'", model_name)
        raise


def load_distilbart_pipeline():
    """Load sshleifer/distilbart-cnn-12-6 once."""
    return _load_seq2seq_pipeline(config.MODELS["distilbart"])


@st.cache_resource(show_spinner=False)
def load_t5_pipeline():
    """Load t5-small summarization pipeline once."""
    return _load_seq2seq_pipeline(config.MODELS["t5_small"])


def distilbart_summarize(
    text: str,
    max_length: int = 120,
    min_length: int = 25,
    on_fallback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Summarize text using DistilBART without dropping trailing source content.
    """
    if not text or not text.strip():
        return ""

    if len(text.split()) > 350:
        return hierarchical_summarize(
            text,
            lambda chunk: distilbart_summarize(chunk, max_length, min_length, on_fallback),
            max_words_per_chunk=350,
            overlap_words=40,
        )
    
    try:
        start_time = time.time()
        pipe = load_distilbart_pipeline()
        if pipe is None:
            raise RuntimeError("DistilBART loader returned no model pipeline.")

        result = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary = _clean_generated_summary(
            result[0].get("generated_text", result[0].get("summary_text", ""))
        )
        if not summary:
            raise ValueError("DistilBART generated an empty summary.")
        
        elapsed = time.time() - start_time
        logger.info(f"DistilBART summary generated in {elapsed:.2f}s")
        print(f"[SmartSum] DistilBART summary generated in {elapsed:.2f}s")
        return summary
    except Exception as e:
        logger.exception("DistilBART could not summarize the input; using TextRank fallback.")
        st.warning(
            f"DistilBART failed ({type(e).__name__}: {e}). "
            "Falling back to TextRank extractive summarization."
        )
        _notify_fallback(on_fallback, "textrank")
        from core.extractive import textrank_summarize
        return textrank_summarize(text, num_sentences=_fallback_sentence_count(max_length))


def t5_summarize(
    text: str,
    max_length: int = 100,
    min_length: int = 20,
    on_fallback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Summarize text using T5-small without dropping trailing source content.
    """
    if not text or not text.strip():
        return ""

    if len(text.split()) > 300:
        return hierarchical_summarize(
            text,
            lambda chunk: t5_summarize(chunk, max_length, min_length, on_fallback),
            max_words_per_chunk=300,
            overlap_words=35,
        )

    start_time = time.time()
    try:
        pipe = load_t5_pipeline()
        if pipe is None:
            raise RuntimeError("T5 loader returned no model pipeline.")

        t5_input = "summarize: " + text
        result = pipe(t5_input, max_length=max_length, min_length=min_length, do_sample=False)
        summary = _clean_generated_summary(
            result[0].get("generated_text", result[0].get("summary_text", ""))
        )
        if not summary:
            raise ValueError("T5 generated an empty summary.")

        elapsed = time.time() - start_time
        logger.info(f"T5-small summary generated in {elapsed:.2f}s")
        print(f"[SmartSum] T5-small summary generated in {elapsed:.2f}s")
        return summary
    except Exception as e:
        logger.exception("T5 could not summarize the input; using TextRank fallback.")
        st.warning(
            f"T5 failed ({type(e).__name__}: {e}). "
            "Falling back to TextRank extractive summarization."
        )
        _notify_fallback(on_fallback, "textrank")
        from core.extractive import textrank_summarize
        return textrank_summarize(text, num_sentences=_fallback_sentence_count(max_length))


def gemini_summarize(
    text: str,
    style: str = "Paragraph",
    on_fallback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Summarize text using Google Gemini API if configured.
    """
    if not config.HAS_GEMINI:
        st.warning("Gemini API key not found. Using local DistilBART model instead.")
        _notify_fallback(on_fallback, "distilbart")
        return distilbart_summarize(text, on_fallback=on_fallback)

    start_time = time.time()
    try:
        from google import genai
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        prompt = (
            f"Please summarize the following text using the style '{style}'. "
            f"Be concise, accurate, and preserve core facts:\n\n{text[:8000]}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        summary = response.text.strip()
        elapsed = time.time() - start_time
        logger.info(f"Gemini summary generated in {elapsed:.2f}s")
        print(f"[SmartSum] Gemini summary generated in {elapsed:.2f}s")
        return summary
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        st.warning(f"Gemini API call failed ({e}). Falling back to DistilBART.")
        _notify_fallback(on_fallback, "distilbart")
        return distilbart_summarize(text, on_fallback=on_fallback)
