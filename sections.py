"""
Domain Mode & Section-wise Summarizer for SmartSum.
Detects structured sections in Research Papers, Legal Briefs, and Medical/Technical Reports.
"""
import re
import logging
from typing import Dict, List
from core.extractive import textrank_summarize

logger = logging.getLogger("SmartSum.Sections")

DOMAIN_PATTERNS = {
    "research": {
        "Problem / Abstract": [r'abstract', r'introduction', r'background', r'problem statement'],
        "Methodology / Design": [r'method', r'methodology', r'approach', r'architecture', r'implementation'],
        "Results & Discussion": [r'result', r'discussion', r'evaluation', r'experiment', r'finding'],
        "Conclusion & Future Work": [r'conclusion', r'future work', r'summary']
    },
    "legal": {
        "Case Summary & Facts": [r'fact', r'background', r'statement of facts', r'parties'],
        "Legal Arguments & Statutes": [r'argument', r'statute', r'precedent', r'holding', r'reasoning'],
        "Final Ruling & Orders": [r'ruling', r'order', r'disposition', r'verdict', r'judgment']
    },
    "medical": {
        "Patient History & Symptoms": [r'history', r'symptom', r'chief complaint', r'presentation'],
        "Diagnosis & Lab Results": [r'diagnosis', r'test', r'lab', r'finding', r'examination'],
        "Treatment Plan & Prescription": [r'treatment', r'plan', r'prescription', r'medication', r'management']
    }
}


def parse_document_sections(text: str, domain: str = "research") -> Dict[str, str]:
    """
    Parse document into logical domain sections based on matching regex headers or paragraph blocks.
    """
    patterns = DOMAIN_PATTERNS.get(domain, DOMAIN_PATTERNS["research"])
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    sections = {sec_name: [] for sec_name in patterns.keys()}
    current_section = list(patterns.keys())[0]  # default to first section

    for para in paragraphs:
        para_lower = para.lower()
        matched = False
        for sec_name, regexes in patterns.items():
            for rgx in regexes:
                if re.search(rf'^\b{rgx}\b', para_lower) or re.search(rf'#{1,3}\s*{rgx}', para_lower):
                    current_section = sec_name
                    matched = True
                    break
            if matched:
                break
        sections[current_section].append(para)

    # Convert lists to text blocks
    return {k: "\n\n".join(v) for k, v in sections.items() if v}


def summarize_domain_sections(text: str, domain: str = "research") -> Dict[str, str]:
    """
    Produce section-by-section summaries for domain documents.
    """
    parsed_sections = parse_document_sections(text, domain=domain)
    section_summaries = {}

    for sec_title, sec_text in parsed_sections.items():
        if not sec_text.strip():
            continue
        summary = textrank_summarize(sec_text, num_sentences=2)
        section_summaries[sec_title] = summary

    return section_summaries
