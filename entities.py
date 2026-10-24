"""
Named Entity Recognition (NER) Module for SmartSum using spaCy.
Highlights entities inline in HTML and extracts entity lists.
"""
import logging
from typing import List, Dict, Tuple
import streamlit as st
import config

logger = logging.getLogger("SmartSum.Entities")

@st.cache_resource(show_spinner="Loading spaCy NER model...")
def load_spacy_nlp():
    """Lazy-load spaCy model (en_core_web_sm). Fallback to download if not installed."""
    import spacy
    try:
        nlp = spacy.load(config.MODELS["spacy_model"])
        return nlp
    except Exception:
        logger.warning("spaCy en_core_web_sm model not found. Downloading...")
        spacy.cli.download(config.MODELS["spacy_model"])
        return spacy.load(config.MODELS["spacy_model"])


def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract categorized named entities using spaCy."""
    if not text or not text.strip():
        return {}

    nlp = load_spacy_nlp()
    doc = nlp(text)
    
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        val = ent.text.strip()
        if label not in entities:
            entities[label] = []
        if val not in entities[label]:
            entities[label].append(val)

    return entities


def render_highlighted_entities_html(text: str) -> str:
    """
    Render text with inline HTML badges for named entities (PERSON, ORG, GPE, DATE, etc.).
    """
    if not text or not text.strip():
        return text

    nlp = load_spacy_nlp()
    doc = nlp(text)

    entity_colors = {
        "PERSON": "#E3F2FD",   # Blue tint
        "ORG": "#F3E5F5",      # Purple tint
        "GPE": "#E8F5E9",      # Green tint
        "DATE": "#FFF3E0",     # Orange tint
        "TIME": "#FFF8E1",
        "MONEY": "#E0F2F1",
        "PRODUCT": "#FFEBEE",
        "EVENT": "#FCE4EC"
    }

    # Render doc with displacy HTML or custom badge string
    try:
        from spacy import displacy
        html = displacy.render(doc, style="ent", page=False, options={"colors": entity_colors})
        return html
    except Exception as e:
        logger.error(f"displacy render error: {e}")
        return text
