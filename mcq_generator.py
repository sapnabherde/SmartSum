"""
MCQ & Flashcard Generator Module for SmartSum.
Generates multiple-choice questions with 4 options, explanations, and flashcards from text summaries.
Supports local rule-based/distractor generation with optional Gemini API upgrade.
"""
import random
import logging
from typing import List, Dict, Any
from core.preprocessing import split_sentences, tokenize_and_lemmatize
from extras.keywords import extract_yake_keywords
from extras.entities import extract_named_entities
import config

logger = logging.getLogger("SmartSum.MCQGenerator")


def generate_distractors(target_word: str, text: str, num_distractors: int = 3) -> List[str]:
    """
    Generate plausible multiple-choice distractors for target_word using other keywords/entities in text.
    """
    kws = [kw for kw, _ in extract_yake_keywords(text, top_n=20) if kw.lower() != target_word.lower()]
    ents_dict = extract_named_entities(text)
    
    candidates = list(kws)
    for ent_list in ents_dict.values():
        candidates.extend([e for e in ent_list if e.lower() != target_word.lower()])

    candidates = list(set([c.strip().title() for c in candidates if len(c.strip()) > 2]))
    
    if len(candidates) < num_distractors:
        # Fallback generic distractors if document keywords are sparse
        generic = ["Neural Network", "Data Analysis", "Feature Extraction", "Machine Learning", "Statistical Modeling", "Information Retrieval"]
        candidates.extend([g for g in generic if g.lower() != target_word.lower()])

    random.shuffle(candidates)
    return candidates[:num_distractors]


def generate_mcqs_local(text: str, num_questions: int = 4) -> List[Dict[str, Any]]:
    """
    Generate MCQs locally by converting key sentences into Cloze / question prompts.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    mcqs = []
    selected_sents = sentences[:min(len(sentences), num_questions * 2)]

    for sent in selected_sents:
        if len(mcqs) >= num_questions:
            break

        # Pick a key word/entity to blank out
        words = [w for w in tokenize_and_lemmatize(sent) if len(w) > 4]
        if not words:
            continue

        target = words[0].title()
        blank_sent = sent.replace(target, "________").replace(target.lower(), "________")
        
        if "________" not in blank_sent:
            continue

        distractors = generate_distractors(target, text, num_distractors=3)
        options = distractors + [target]
        random.shuffle(options)

        mcqs.append({
            "question": f"Fill in the blank: {blank_sent}",
            "options": options,
            "answer": target,
            "explanation": f"In the document context: '{sent}'"
        })

    return mcqs


def generate_mcqs(text: str, num_questions: int = 4) -> List[Dict[str, Any]]:
    """
    Generate MCQs using Gemini API if present, otherwise local generator.
    """
    if config.HAS_GEMINI:
        try:
            from google import genai
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            prompt = (
                f"Generate {num_questions} multiple-choice questions (MCQs) based on this text. "
                "Output JSON format as a list of objects with fields: 'question', 'options' (list of 4 strings), "
                f"'answer' (exact matching option string), and 'explanation'.\n\nText:\n{text[:5000]}"
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            import json, re
            # Extract JSON substring
            match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.warning(f"Gemini MCQ generation failed ({e}). Using local fallback.")

    return generate_mcqs_local(text, num_questions=num_questions)


def generate_flashcards(text: str, num_cards: int = 5) -> List[Dict[str, str]]:
    """
    Generate Question/Answer Flashcards from key sentences in text.
    """
    sentences = split_sentences(text)
    flashcards = []

    for i, sent in enumerate(sentences[:num_cards]):
        flashcards.append({
            "front": f"Flashcard #{i+1}: Key Concept",
            "back": sent
        })

    return flashcards
