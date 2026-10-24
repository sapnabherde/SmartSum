"""
Action Items Extractor Module for SmartSum.
Extracts tasks, assignees (who), actions (what), and deadlines (when) from meeting transcripts.
"""
import re
import logging
from typing import List, Dict, Any
from core.preprocessing import split_sentences
from extras.entities import load_spacy_nlp

logger = logging.getLogger("SmartSum.ActionItems")

# Modal verbs and imperative action verbs
ACTION_VERBS = [
    "will", "shall", "must", "should", "need to", "needs to", "assigned to",
    "submit", "complete", "prepare", "deliver", "review", "schedule",
    "send", "create", "update", "organize", "present", "draft", "finalize"
]

DEADLINE_KEYWORDS = ["by", "before", "due", "until", "deadline", "tomorrow", "next week", "monday", "tuesday", "wednesday", "thursday", "friday"]


def extract_action_items(text: str) -> List[Dict[str, Any]]:
    """
    Scan sentences for action triggers, assigned person, action verb, and potential deadline.
    """
    sentences = split_sentences(text)
    action_items = []
    nlp = load_spacy_nlp()

    for sent in sentences:
        sent_lower = sent.lower()

        # Check if sentence contains any action verb
        has_action = any(re.search(rf'\b{verb}\b', sent_lower) for verb in ACTION_VERBS)
        if not has_action:
            continue

        doc = nlp(sent)
        
        # Extract PERSON entities as assignees
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        # Extract DATE/TIME entities as deadlines
        dates = [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]

        assignee = ", ".join(persons) if persons else "Unassigned / Team"
        deadline = ", ".join(dates) if dates else "Not specified"

        action_items.append({
            "sentence": sent,
            "assignee": assignee,
            "action": sent.strip(),
            "deadline": deadline
        })

    return action_items
