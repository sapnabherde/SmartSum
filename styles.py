"""
Style Post-Processing Module for SmartSum.
Formats raw summaries into Paragraph, Bullet Points, Executive Summary,
Key Takeaways, Structured Summary, Study Notes, ELI5, or Exam Notes.
Provides both plain-text formatting (for TXT/DOCX/PDF exports) and HTML rendering (for on-screen Streamlit UI).
"""
import re
from typing import List
from core.preprocessing import split_sentences

STYLE_PARAGRAPH = "Paragraph"
STYLE_BULLETS = "Bullet Points"
STYLE_EXECUTIVE = "Executive Summary"
STYLE_TAKEAWAYS = "Key Takeaways"
STYLE_STRUCTURED = "Structured Summary"
STYLE_STUDY_NOTES = "Study Notes"
STYLE_ELI5 = "ELI5 (Explain Like I'm 5)"
STYLE_EXAM_NOTES = "Exam notes (Headings + Key Points)"

# Primary supported format choices as requested
FORMAT_CHOICES = [
    STYLE_BULLETS,
    STYLE_PARAGRAPH,
    STYLE_EXECUTIVE,
    STYLE_TAKEAWAYS,
    STYLE_STRUCTURED,
    STYLE_STUDY_NOTES,
]

# Backward-compatible list of all style identifiers
STYLE_CHOICES = [
    STYLE_PARAGRAPH,
    "Bullet points",
    STYLE_BULLETS,
    STYLE_EXECUTIVE,
    STYLE_TAKEAWAYS,
    STYLE_STRUCTURED,
    STYLE_STUDY_NOTES,
    STYLE_ELI5,
    STYLE_EXAM_NOTES,
]

STYLE_CSS = """
<style>
.summary-container {
    background: #111a2e;
    border: 1px solid #283652;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    margin-bottom: 20px;
    color: #e8eef8;
}
.summary-paragraph {
    font-size: 1.05rem;
    line-height: 1.75;
    color: #e8eef8;
}
.summary-bullet-list {
    margin-top: 10px;
    padding-left: 20px;
}
.summary-bullet-item {
    font-size: 1.02rem;
    line-height: 1.65;
    margin-bottom: 12px;
    color: #e8eef8;
}
.summary-section-header {
    font-family: 'Poppins', sans-serif;
    font-size: 1.12rem;
    font-weight: 700;
    color: #a99cff;
    margin-top: 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.summary-card-item {
    background: #17233b;
    border-left: 4px solid #8b7cff;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 0.98rem;
    line-height: 1.6;
    color: #e8eef8;
}
.eli5-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
    color: #111827;
    font-weight: 700;
    font-size: 0.88rem;
    padding: 6px 14px;
    border-radius: 20px;
    margin-bottom: 14px;
}
.exam-section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #c4b5fd;
    margin-top: 16px;
    margin-bottom: 8px;
}
.exam-item {
    background: #17233b;
    border-left: 4px solid #8b7cff;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.98rem;
    color: #e8eef8;
}
</style>
"""


def format_bullet_points(text: str) -> str:
    """Format summary as a clean list of bullet points."""
    sentences = split_sentences(text)
    if not sentences:
        return text
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
    return "\n\n".join(bullets)


def format_executive_summary(text: str) -> str:
    """
    Format into Executive Summary:
    1. Executive Overview
    2. Key Findings
    3. Strategic Implications / Action Points
    """
    sentences = split_sentences(text)
    if not sentences:
        return text

    if len(sentences) == 1:
        return f"### 📌 Executive Overview\n\n{sentences[0]}"

    overview = sentences[0]
    remainder = sentences[1:]
    mid = max(1, len(remainder) // 2)
    findings = remainder[:mid]
    implications = remainder[mid:] or findings

    out = []
    out.append("### 📌 Executive Overview")
    out.append(overview)
    out.append("\n### 📊 Key Findings")
    for s in findings:
        out.append(f"• {s}")
    out.append("\n### 💡 Strategic Implications")
    for s in implications:
        out.append(f"• {s}")

    return "\n\n".join(out)


def format_key_takeaways(text: str) -> str:
    """Format into numbered Key Takeaways with highlighted lead-ins."""
    sentences = split_sentences(text)
    if not sentences:
        return text
    out = []
    out.append("### ⚡ Key Takeaways")
    for idx, s in enumerate(sentences):
        words = s.split()
        if len(words) > 3:
            lead = " ".join(words[:3])
            rest = " ".join(words[3:])
            out.append(f"{idx+1}. **{lead}** {rest}")
        else:
            out.append(f"{idx+1}. {s}")
    return "\n\n".join(out)


def format_structured_summary(text: str) -> str:
    """Format into Structured Summary: Background, Core Analysis, Conclusion."""
    sentences = split_sentences(text)
    if not sentences:
        return text

    if len(sentences) <= 2:
        return f"### 🏛️ Context & Overview\n\n{sentences[0]}\n\n### 🏁 Conclusion\n\n" + (sentences[1] if len(sentences) > 1 else "")

    background = sentences[0]
    conclusion = sentences[-1]
    discussion = sentences[1:-1]

    out = [
        "### 🏛️ Background & Context",
        background,
        "\n### 🔍 Core Analysis",
        "\n".join([f"• {s}" for s in discussion]),
        "\n### 🏁 Conclusion & Outlook",
        conclusion
    ]
    return "\n\n".join(out)


def format_study_notes(text: str) -> str:
    """Format into Study Notes: Key Concepts, Definitions, and Review Checklist."""
    sentences = split_sentences(text)
    if not sentences:
        return text

    mid = max(1, len(sentences) // 2)
    concepts = sentences[:mid]
    details = sentences[mid:]

    out = [
        "### 📌 Core Concepts",
        "\n".join([f"• **Concept:** {s}" for s in concepts]),
        "\n### 📖 Important Terms & Details",
        "\n".join([f"• {s}" for s in details]),
        "\n### 💡 Quick Memory Checklist",
        "✔ Master the fundamental ideas listed above",
        "✔ Review definitions and interrelationships",
    ]
    return "\n\n".join(out)


def format_eli5(text: str) -> str:
    """Format summary in ELI5 (Explain Like I'm 5) style using simplified sentences."""
    sentences = split_sentences(text)
    if not sentences:
        return text

    eli5_header = "👶 **In simple terms (ELI5):**\n\n"
    bullets = []
    for s in sentences:
        simple = s.replace("utilize", "use").replace("implement", "do").replace("subsequently", "then").replace("optimal", "best")
        bullets.append(f"• {simple}")
    
    return eli5_header + "\n\n".join(bullets)


def format_exam_notes(text: str) -> str:
    """Format summary into Exam Notes style."""
    return format_study_notes(text)


def apply_summary_style(summary_text: str, style_choice: str) -> str:
    """
    Apply chosen style transformation to summary text as plain text.
    Used for TXT, DOCX, and PDF exports.
    """
    if not summary_text:
        return ""

    choice = (style_choice or STYLE_PARAGRAPH).strip().lower()

    if "bullet" in choice:
        return format_bullet_points(summary_text)
    elif "executive" in choice:
        return format_executive_summary(summary_text)
    elif "takeaway" in choice:
        return format_key_takeaways(summary_text)
    elif "structure" in choice:
        return format_structured_summary(summary_text)
    elif "study" in choice or "exam" in choice:
        return format_study_notes(summary_text)
    elif "eli5" in choice:
        return format_eli5(summary_text)
    else:
        # Paragraph (default)
        sentences = split_sentences(summary_text)
        return " ".join(sentences)


def apply_summary_style_html(summary_text: str, style_choice: str) -> str:
    """
    Apply chosen style transformation to summary text as HTML for Streamlit UI rendering.
    NOTE: Unindented string formatting so Streamlit markdown never renders HTML as a code block.
    """
    if not summary_text:
        return ""

    choice = (style_choice or STYLE_PARAGRAPH).strip().lower()
    sentences = split_sentences(summary_text)
    if not sentences:
        return f'<div class="summary-container summary-paragraph">{summary_text}</div>'

    if "bullet" in choice:
        items = "".join([f'<li class="summary-bullet-item">{s}</li>' for s in sentences if s.strip()])
        return f'<div class="summary-container"><ul class="summary-bullet-list">{items}</ul></div>'

    elif "executive" in choice:
        overview = sentences[0]
        remainder = sentences[1:]
        mid = max(1, len(remainder) // 2)
        findings = remainder[:mid] or [overview]
        implications = remainder[mid:] or findings

        f_html = "".join([f'<div class="summary-card-item">📊 {s}</div>' for s in findings])
        i_html = "".join([f'<div class="summary-card-item">💡 {s}</div>' for s in implications])

        return (
            f'<div class="summary-container">'
            f'<div class="summary-section-header">📌 Executive Overview</div>'
            f'<p class="summary-paragraph">{overview}</p>'
            f'<div class="summary-section-header">📊 Key Findings</div>{f_html}'
            f'<div class="summary-section-header">💡 Strategic Implications</div>{i_html}'
            f'</div>'
        )

    elif "takeaway" in choice:
        t_html = []
        for idx, s in enumerate(sentences):
            words = s.split()
            if len(words) > 3:
                lead = " ".join(words[:3])
                rest = " ".join(words[3:])
                t_html.append(f'<div class="summary-card-item"><b>{idx+1}. {lead}</b> {rest}</div>')
            else:
                t_html.append(f'<div class="summary-card-item"><b>{idx+1}.</b> {s}</div>')
        return f'<div class="summary-container"><div class="summary-section-header">⚡ Key Takeaways</div>{"".join(t_html)}</div>'

    elif "structure" in choice:
        background = sentences[0]
        conclusion = sentences[-1] if len(sentences) > 1 else background
        discussion = sentences[1:-1] if len(sentences) > 2 else [background]

        d_html = "".join([f'<div class="summary-card-item">🔍 {s}</div>' for s in discussion])
        return (
            f'<div class="summary-container">'
            f'<div class="summary-section-header">🏛️ Background & Context</div>'
            f'<p class="summary-paragraph">{background}</p>'
            f'<div class="summary-section-header">🔍 Core Analysis</div>{d_html}'
            f'<div class="summary-section-header">🏁 Conclusion & Outlook</div>'
            f'<p class="summary-paragraph">{conclusion}</p>'
            f'</div>'
        )

    elif "study" in choice or "exam" in choice:
        mid = max(1, len(sentences) // 2)
        concepts = sentences[:mid]
        details = sentences[mid:]

        c_html = "".join([f'<div class="summary-card-item">📌 <b>Core Concept:</b> {s}</div>' for s in concepts])
        d_html = "".join([f'<div class="summary-card-item">📖 {s}</div>' for s in details]) if details else ""

        return (
            f'<div class="summary-container">'
            f'<div class="summary-section-header">📌 Core Concepts</div>{c_html}'
            f'<div class="summary-section-header">📖 Important Terms & Details</div>{d_html}'
            f'<div class="summary-section-header">💡 Quick Memory Checklist</div>'
            f'<div class="summary-card-item">✔ Master the fundamental ideas listed above</div>'
            f'<div class="summary-card-item">✔ Review definitions and interrelationships</div>'
            f'</div>'
        )

    elif "eli5" in choice:
        items = "".join([f'<li class="summary-bullet-item">{s.replace("utilize", "use").replace("implement", "do").replace("subsequently", "then").replace("optimal", "best")}</li>' for s in sentences if s.strip()])
        return f'<div class="summary-container"><div class="eli5-badge">👶 In Simple Terms (ELI5)</div><ul class="summary-bullet-list">{items}</ul></div>'

    else:
        # Paragraph (default)
        p_text = " ".join(sentences)
        return f'<div class="summary-container summary-paragraph">{p_text}</div>'
