"""
Multilingual Module for SmartSum.
Handles translation between Hindi/Marathi and English using Helsinki-NLP Opus-MT models.
Allows summarization of non-English inputs and generating summaries in Hindi or Marathi.
"""
import logging
import streamlit as st
import config
from core.preprocessing import detect_language

logger = logging.getLogger("SmartSum.Multilingual")

@st.cache_resource(show_spinner="Loading Translation Model...")
def load_translation_pipeline(src_lang: str, tgt_lang: str):
    """
    Lazy load translation pipeline based on source and target language pairs.
    """
    key = f"translation_{src_lang}_{tgt_lang}"
    model_name = config.MODELS.get(key)
    if not model_name:
        logger.warning(f"No translation model configured for {key}")
        return None

    try:
        from transformers import pipeline
        pipe = pipeline("translation", model=model_name, device=-1 if config.DEVICE == "cpu" else 0)
        return pipe
    except Exception as e:
        logger.error(f"Failed to load translation model {model_name}: {e}")
        return None


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text from src_lang ('en', 'hi', 'mr') to tgt_lang ('en', 'hi', 'mr').
    """
    if not text or not text.strip() or src_lang == tgt_lang:
        return text

    pipe = load_translation_pipeline(src_lang, tgt_lang)
    if pipe is None:
        st.warning(f"Translation pipeline ({src_lang}->{tgt_lang}) unavailable. Output kept in original language.")
        return text

    try:
        # Split text into chunks to avoid model length exceeded errors
        from core.preprocessing import split_sentences
        sentences = split_sentences(text)
        translated_sents = []

        for sent in sentences:
            if not sent.strip():
                continue
            res = pipe(sent, max_length=512)
            translated_sents.append(res[0]['translation_text'].strip())

        return " ".join(translated_sents)
    except Exception as e:
        logger.error(f"Translation error ({src_lang}->{tgt_lang}): {e}")
        st.warning(f"Translation failed ({e}). Returning untranslated text.")
        return text


def process_multilingual_summarization(
    input_text: str,
    target_lang: str,
    summarize_func
) -> str:
    """
    Multilingual pipeline:
    1. Detect input language.
    2. If input is Hindi/Marathi, translate to English first.
    3. Run summarizer in English for best performance.
    4. If target_lang is Hindi/Marathi, translate English summary back to target_lang.
    """
    detected_lang = detect_language(input_text)
    
    # Translate input to English if non-English
    if detected_lang != "en":
        st.info(f"Detected input language: **{config.SUPPORTED_LANGUAGES.get(detected_lang, detected_lang).upper()}**. Translating to English for summarization...")
        english_text = translate_text(input_text, src_lang=detected_lang, tgt_lang="en")
    else:
        english_text = input_text

    # Summarize in English
    english_summary = summarize_func(english_text)

    # Back-translate if target language is non-English
    if target_lang != "en":
        st.info(f"Translating final summary to **{config.SUPPORTED_LANGUAGES.get(target_lang, target_lang)}**...")
        final_summary = translate_text(english_summary, src_lang="en", tgt_lang=target_lang)
    else:
        final_summary = english_summary

    return final_summary
