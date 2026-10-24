"""
SmartSum - Advanced AI Text & Document Summarizer
"Understand more. Read less."
Streamlit Main Application Entry Point.
"""
import os
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from database import db as db_module

# Custom Config & Helpers
import config
from utils.helpers import (
    count_words, count_sentences, calculate_stats,
    export_as_txt, export_as_docx, export_as_pdf
)
from utils.ui_helpers import load_css, render_footer
from utils.model_manager import preload_all_models

# Core NLP Modules
from core.preprocessing import split_sentences, detect_language
from core.extractive import textrank_summarize, tfidf_summarize, lsa_summarize
from core.abstractive import distilbart_summarize, t5_summarize, gemini_summarize
from core.styles import (
    STYLE_CSS,
    STYLE_CHOICES,
    FORMAT_CHOICES,
    STYLE_PARAGRAPH,
    STYLE_BULLETS,
    STYLE_EXECUTIVE,
    STYLE_TAKEAWAYS,
    STYLE_STRUCTURED,
    STYLE_STUDY_NOTES,
    apply_summary_style,
    apply_summary_style_html,
)
from core.query_focused import query_focused_summarize
from core.long_doc import hierarchical_summarize
from core.multilingual import process_multilingual_summarization
from core.smart_pipeline import (
    analyze_content,
    execute_smart_pipeline,
    calculate_quality_metrics,
    determine_smart_strategy
)

# Input Loaders
from inputs.pdf_loader import extract_text_from_pdf
from inputs.docx_loader import extract_text_from_docx
from inputs.url_loader import extract_text_from_url
from inputs.youtube_loader import extract_youtube_transcript
from inputs.audio_loader import transcribe_audio_file

# Extras & Insights
from extras.keywords import extract_yake_keywords, extract_keybert_keywords, generate_wordcloud
from extras.entities import extract_named_entities, render_highlighted_entities_html
from extras.sentiment import compare_sentiments
from extras.action_items import extract_action_items
from extras.mindmap import generate_mindmap_html
from extras.mcq_generator import generate_mcqs, generate_flashcards

# Evaluation & Metrics
from evaluation.metrics import compute_rouge_scores, compute_bertscore
from evaluation.faithfulness import verify_summary_faithfulness
from evaluation.benchmark import run_algorithm_benchmark, DEFAULT_BENCHMARK_TEXT, DEFAULT_REFERENCE_SUMMARY

# Configure Page
st.set_page_config(
    page_title="SmartSum - Advanced AI Text & Document Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
load_css()
st.markdown(STYLE_CSS, unsafe_allow_html=True)

# ----------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------
if "current_input_text" not in st.session_state:
    st.session_state.current_input_text = ""
if "current_doc_title" not in st.session_state:
    st.session_state.current_doc_title = "Untitled Document"
if "current_summary_text" not in st.session_state:
    st.session_state.current_summary_text = ""
if "current_summary_style" not in st.session_state:
    st.session_state.current_summary_style = STYLE_BULLETS
if "current_summary_length" not in st.session_state:
    st.session_state.current_summary_length = "Balanced"
if "current_summary_id" not in st.session_state:
    st.session_state.current_summary_id = None
if "current_strategy_info" not in st.session_state:
    st.session_state.current_strategy_info = {}
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = {}
if "selected_source" not in st.session_state:
    st.session_state.selected_source = "Text"
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "is_guest" not in st.session_state:
    st.session_state.is_guest = False
if "lite_mode" not in st.session_state:
    st.session_state.lite_mode = config.LITE_MODE
if "custom_gemini_key" not in st.session_state:
    st.session_state.custom_gemini_key = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "focus_query" not in st.session_state:
    st.session_state.focus_query = ""

# Dynamic Gemini Check
has_gemini_active = bool(config.HAS_GEMINI or st.session_state.get("custom_gemini_key", "").strip())
if st.session_state.get("custom_gemini_key", "").strip():
    config.GEMINI_API_KEY = st.session_state.custom_gemini_key.strip()
    config.HAS_GEMINI = True

# Hero Logo SVG
try:
    _hero_logo_svg = (config.ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")
except Exception:
    _hero_logo_svg = '<svg viewBox="0 0 48 48"><circle cx="24" cy="24" r="20" fill="#8B7CFF"/></svg>'


# ----------------------------------------------------
# AUTHENTICATION & PROFILE HELPERS
# ----------------------------------------------------
def _user_initials(name: str) -> str:
    parts = [part.strip() for part in (name or "").split() if part.strip()]
    if not parts:
        return "U"
    initials = "".join(part[0].upper() for part in parts[:2])
    return initials[:2] or "U"


def logout_user():
    for key in [
        "current_user", "user_id", "user_name",
        "chat_messages", "history_search", "history_method"
    ]:
        st.session_state.pop(key, None)
    st.session_state.is_guest = False


def set_authenticated_user(user):
    st.session_state.current_user = user
    st.session_state.user_id = user["id"]
    st.session_state.user_name = user["name"]
    st.session_state.current_summary_style = user.get("default_style") or STYLE_BULLETS
    st.session_state.lite_mode = bool(user.get("lite_mode", config.LITE_MODE))
    st.session_state.is_guest = False


def authenticate_user(auth_mode: str, name: str = "", email: str = "", password: str = "", confirm_password: str = ""):
    normalized_email = (email or "").strip().lower()
    if not normalized_email or not db_module.validate_email(normalized_email):
        raise ValueError("Please enter a valid email address.")
    if not password or not db_module.validate_password(password):
        raise ValueError("Password must be at least 8 characters long.")

    if auth_mode == "Sign Up":
        if not (name or "").strip():
            raise ValueError("Please enter your name.")
        if password != confirm_password:
            raise ValueError("Passwords do not match. Please re-enter your password.")
        user = db_module.create_user(name.strip(), normalized_email, password)
        return user

    user = db_module.login_user(normalized_email, password)
    if user is None:
        raise ValueError("Incorrect email or password. Please try again.")
    return user


def set_guest_mode():
    st.session_state.current_user = None
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.is_guest = True


# ----------------------------------------------------
# PROFILE CHIP (TOP-RIGHT)
# ----------------------------------------------------
def render_profile_chip():
    if st.session_state.current_user is None:
        if st.session_state.get("is_guest", False):
            with st.popover("👤 Guest ▾", use_container_width=False):
                st.markdown("### Guest Mode")
                st.caption("Summaries generated in guest mode are not saved to database history.")
                st.markdown("---")
                if st.button("🔐 Login / Sign Up", key="guest_login_action", type="primary", use_container_width=True):
                    st.session_state.is_guest = False
                    st.rerun()
        return

    user_name = st.session_state.user_name or st.session_state.current_user.get("name", "User")
    user_email = st.session_state.current_user.get("email", "")
    initials = _user_initials(user_name)
    with st.popover(f"👤 {initials} {user_name} ▾", use_container_width=False):
        st.markdown(f"**{user_name}**")
        st.caption(user_email)
        st.markdown("---")
        st.markdown("#### Preferences")
        with st.form("profile_preferences_form", clear_on_submit=False):
            user = st.session_state.current_user
            default_style = st.selectbox(
                "Default summary format",
                options=FORMAT_CHOICES,
                index=FORMAT_CHOICES.index(user.get("default_style", STYLE_BULLETS)) if user.get("default_style") in FORMAT_CHOICES else 0
            )
            default_lang = st.selectbox(
                "Default language",
                options=["en", "hi", "mr"],
                index=["en", "hi", "mr"].index(user.get("default_language", "en")),
                format_func=lambda x: config.SUPPORTED_LANGUAGES.get(x, x)
            )
            user_lite_mode = st.checkbox("Default Lite Mode", value=bool(user.get("lite_mode", 0)))
            save_pref_btn = st.form_submit_button("Save Preferences", use_container_width=True)
            if save_pref_btn:
                db_module.update_user_settings(
                    user["id"],
                    default_style=default_style,
                    default_language=default_lang,
                    lite_mode=user_lite_mode
                )
                st.session_state.current_summary_style = default_style
                st.session_state.lite_mode = user_lite_mode
                st.toast("Preferences successfully updated!", icon="✅")

        st.markdown("---")
        if st.button("🚪 Logout", key="header_logout", use_container_width=True):
            logout_user()
            st.rerun()


# ----------------------------------------------------
# AUTHENTICATION SCREEN (CENTERED CARD)
# ----------------------------------------------------
def render_auth_screen():
    st.markdown("<div class='auth-center-wrapper'>", unsafe_allow_html=True)

    col_l, col_center, col_r = st.columns([1, 1.8, 1])
    with col_center:
        st.markdown(
            f"""
            <div class="auth-center-card animate-slide-up">
                <div class="auth-brand-header">
                    <span class="auth-brand-logo">{_hero_logo_svg}</span>
                    <h2 class="auth-brand-title">SmartSum</h2>
                    <div class="auth-brand-tagline">Understand more. Read less.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["Login", "Create Account"])

        with tab_login:
            with st.form("auth_login_form", clear_on_submit=False):
                show_login_password = st.checkbox("Show password", key="auth_login_show_password")
                email = st.text_input("Email", placeholder="you@example.com", key="auth_login_email")
                password = st.text_input(
                    "Password",
                    type="text" if show_login_password else "password",
                    placeholder="••••••••",
                    key="auth_login_password"
                )
                login_btn = st.form_submit_button("Login to SmartSum", type="primary", use_container_width=True)
                if login_btn:
                    try:
                        user = authenticate_user("Login", email=email, password=password)
                        set_authenticated_user(user)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception:
                        st.error("Login failed. Please verify your details.")

        with tab_signup:
            with st.form("auth_signup_form", clear_on_submit=False):
                show_signup_password = st.checkbox("Show password", key="auth_signup_show_password")
                name = st.text_input("Full Name", placeholder="Sapna", key="auth_signup_name")
                email = st.text_input("Email", placeholder="you@example.com", key="auth_signup_email")
                password = st.text_input(
                    "Password (min 8 characters)",
                    type="text" if show_signup_password else "password",
                    placeholder="••••••••",
                    key="auth_signup_password"
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="text" if show_signup_password else "password",
                    placeholder="••••••••",
                    key="auth_signup_confirm_password"
                )
                signup_btn = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                if signup_btn:
                    try:
                        user = authenticate_user(
                            "Sign Up",
                            name=name,
                            email=email,
                            password=password,
                            confirm_password=confirm_password
                        )
                        set_authenticated_user(user)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception:
                        st.error("Account creation failed. Please check your information.")

        st.markdown("<div class='auth-divider'>or</div>", unsafe_allow_html=True)
        if st.button("Continue as Guest", key="auth_guest_btn", use_container_width=True):
            set_guest_mode()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    if st.session_state.current_user is None and not st.session_state.get("is_guest", False):
        st.markdown("### ⚡ SmartSum")
        st.caption("Advanced AI Text & Document Summarizer\n\n*Understand more. Read less.*")
        st.markdown("---")
        st.info("Sign in or continue as guest to access the full summarizer.")
    else:
        st.markdown("### ⚡ SmartSum")
        st.caption("Advanced AI Text & Document Summarizer\n\n*Understand more. Read less.*")
        st.markdown("---")

        # Engine status
        if has_gemini_active:
            st.markdown(
                '<div style="background: rgba(34, 197, 94, 0.12); border: 1px solid #22c55e; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">'
                '<span style="color: #22c55e; font-weight: 700; font-size: 0.9rem;">🟢 AI Engine: Gemini Active</span>'
                '<div style="color: var(--text-muted); font-size: 0.78rem; margin-top: 2px;">Abstractive & Hybrid neural models connected.</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background: rgba(148, 163, 184, 0.1); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">'
                '<span style="color: #cbd5e1; font-weight: 700; font-size: 0.9rem;">⚪ AI Engine: Local Models (Offline)</span>'
                '<div style="color: var(--text-muted); font-size: 0.78rem; margin-top: 2px;">Running local extractive & neural models. Add a Gemini API key in Settings to enable Gemini.</div>'
                '</div>',
                unsafe_allow_html=True
            )

        # Settings Expander
        with st.expander("⚙️ Settings & Performance", expanded=False):
            st.markdown("#### Performance")
            lite_active = st.checkbox(
                "Lite Mode",
                value=st.session_state.lite_mode,
                help="Use less memory for faster operation by prioritizing lightweight extractive algorithms."
            )
            st.session_state.lite_mode = lite_active

            col_pre, col_clr = st.columns(2)
            with col_pre:
                if st.button("Preload Models", use_container_width=True, help="Load selected models into memory for faster generation."):
                    preload_all_models()
                    st.success("Preloaded!")
            with col_clr:
                if st.button("Clear Cache", use_container_width=True, help="Clear locally stored model/cache data."):
                    st.cache_resource.clear()
                    st.toast("Model cache cleared!")

            st.markdown("---")
            st.markdown("#### Cloud AI (Gemini)")
            custom_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.custom_gemini_key or config.GEMINI_API_KEY,
                placeholder="AIzaSy...",
                help="Enter your Google Gemini API key to activate Gemini summarization without code modification."
            )
            if custom_key != st.session_state.custom_gemini_key:
                st.session_state.custom_gemini_key = custom_key.strip()
                if custom_key.strip():
                    config.GEMINI_API_KEY = custom_key.strip()
                    config.HAS_GEMINI = True
                    st.rerun()


# ----------------------------------------------------
# MAIN HEADER & AUTH GATE
# ----------------------------------------------------
header_col, profile_col = st.columns([5, 1.8], vertical_alignment="top")
with header_col:
    st.markdown(
        f"""
        <div class="hero-container animate-slide-up">
            <div class="hero-title">
                <span class="hero-logo">{_hero_logo_svg}</span>
                <span>SmartSum</span>
            </div>
            <div class="hero-tagline">
                Advanced AI Text &amp; Document Summarizer &nbsp;•&nbsp; <em>Understand more. Read less.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with profile_col:
    render_profile_chip()

# Gate unauthenticated users who haven't selected guest
if st.session_state.current_user is None and not st.session_state.get("is_guest", False):
    render_auth_screen()
    st.stop()


# ----------------------------------------------------
# PRIMARY NAVIGATION TABS
# ----------------------------------------------------
tab_summarize, tab_history, tab_insights, tab_study, tab_compare, tab_qa = st.tabs([
    "⚡ Summarize",
    "🕘 History",
    "🔍 Insights",
    "🎓 Study Tools",
    "⚖️ Compare Methods",
    "💬 Ask Document",
])


# ----------------------------------------------------
# TAB 1: SUMMARIZE (CORE WORKFLOW)
# ----------------------------------------------------
with tab_summarize:
    col_input, col_result = st.columns([1.05, 1.15], gap="large")

    with col_input:
        # 1. SELECT SOURCE
        st.markdown("### 1. Select Source")
        st.caption("What do you want to summarize?")

        sources = [
            ("Text", "Text"),
            ("PDF", "PDF"),
            ("Word", "DOCX"),
            ("Web URL", "Website"),
            ("YouTube", "YouTube"),
            ("Audio", "Audio"),
        ]

        source_cols = st.columns(6, gap="small")
        for idx, (label, src_key) in enumerate(sources):
            with source_cols[idx]:
                is_active = (st.session_state.selected_source == src_key)
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"src_btn_{src_key}", type=btn_type, use_container_width=True):
                    st.session_state.selected_source = src_key
                    st.rerun()

        # 2. CONTENT INPUT AREA (Shows only relevant input)
        st.markdown("### 2. Content")
        st.caption("Paste or type your text here, or choose a source above.")
        active_src = st.session_state.selected_source

        if active_src == "Text":
            text_val = st.text_area(
                "Text / Document Content",
                value=st.session_state.current_input_text if active_src == "Text" else "",
                height=150,
                placeholder="Paste or type your text here, or choose a source above.",
                help="Accepts raw text up to 500,000 characters."
            )
            if text_val:
                st.session_state.current_input_text = text_val.strip()
                st.session_state.current_doc_title = "Pasted Text"

        elif active_src == "PDF":
            uploaded_pdf = st.file_uploader("Upload PDF document (Max 15MB)", type=["pdf"])
            if uploaded_pdf is not None:
                if len(uploaded_pdf.getvalue()) > config.MAX_PDF_SIZE_MB * 1024 * 1024:
                    st.error(f"File exceeds maximum size of {config.MAX_PDF_SIZE_MB}MB.")
                else:
                    with st.status("Extracting PDF text with PyMuPDF...", expanded=False) as status:
                        try:
                            start_t = time.time()
                            extracted = extract_text_from_pdf(uploaded_pdf.getvalue())
                            elapsed = time.time() - start_t
                            st.session_state.current_doc_title = uploaded_pdf.name
                            if extracted.casefold() == (
                                "[Warning: Scanned image PDF detected. No extractable text found.]".casefold()
                            ):
                                st.session_state.current_input_text = ""
                                st.session_state.current_summary_text = ""
                                st.session_state.current_metrics = {}
                                st.session_state.current_strategy_info = {}
                                status.update(
                                    label="Scanned image PDF detected. No extractable text found.",
                                    state="error",
                                )
                                st.error(
                                    "OCR is required to extract text from this scanned PDF."
                                )
                            else:
                                st.session_state.current_input_text = extracted
                                w_count = len(extracted.split())
                                status.update(label=f"✅ Extracted {w_count:,} words in {elapsed:.2f}s", state="complete")
                        except Exception as e:
                            status.update(label=f"⚠️ Unable to extract PDF text. Please check the file and try again.", state="error")

        elif active_src == "DOCX":
            uploaded_docx = st.file_uploader("Upload Word document (.docx)", type=["docx"])
            if uploaded_docx is not None:
                with st.status("Extracting Word document...", expanded=False) as status:
                    try:
                        start_t = time.time()
                        extracted = extract_text_from_docx(uploaded_docx.getvalue())
                        elapsed = time.time() - start_t
                        st.session_state.current_input_text = extracted
                        st.session_state.current_doc_title = uploaded_docx.name
                        w_count = len(extracted.split())
                        status.update(label=f"✅ Extracted {w_count:,} words in {elapsed:.2f}s", state="complete")
                    except Exception:
                        status.update(label="⚠️ Unable to extract Word content. Please check the file and try again.", state="error")

        elif active_src == "Website":
            url_input = st.text_input("Web URL", placeholder="https://example.com/article")
            if url_input:
                if st.button("Fetch article content", type="secondary"):
                    with st.status("Fetching webpage content...", expanded=False) as status:
                        try:
                            start_t = time.time()
                            extracted = extract_text_from_url(url_input, timeout=15)
                            elapsed = time.time() - start_t
                            st.session_state.current_input_text = extracted
                            st.session_state.current_doc_title = url_input.split("//")[-1][:40]
                            w_count = len(extracted.split())
                            status.update(label=f"✅ Fetched {w_count:,} words in {elapsed:.2f}s", state="complete")
                        except Exception:
                            status.update(label="⚠️ Unable to fetch the webpage. Please verify the URL and try again.", state="error")

        elif active_src == "YouTube":
            yt_input = st.text_input("YouTube URL or video ID", placeholder="https://www.youtube.com/watch?v=...")
            if yt_input:
                if st.button("Fetch transcript", type="secondary"):
                    with st.status("Fetching video transcript...", expanded=False) as status:
                        try:
                            start_t = time.time()
                            extracted = extract_youtube_transcript(yt_input, timeout=20)
                            elapsed = time.time() - start_t
                            st.session_state.current_input_text = extracted
                            st.session_state.current_doc_title = f"YouTube: {yt_input[:30]}"
                            w_count = len(extracted.split())
                            status.update(label=f"✅ Fetched transcript ({w_count:,} words) in {elapsed:.2f}s", state="complete")
                        except Exception:
                            status.update(label="⚠️ Unable to fetch the YouTube transcript. Please check the video link and try again.", state="error")

        elif active_src == "Audio":
            uploaded_audio = st.file_uploader("Upload audio (MP3/WAV/M4A, Max 25MB)", type=["mp3", "wav", "m4a", "ogg"])
            if uploaded_audio is not None:
                if len(uploaded_audio.getvalue()) > config.MAX_AUDIO_SIZE_MB * 1024 * 1024:
                    st.error(f"Audio file exceeds {config.MAX_AUDIO_SIZE_MB}MB limit.")
                else:
                    if st.button("Transcribe audio", type="secondary"):
                        with st.status("Transcribing audio with Whisper...", expanded=False) as status:
                            try:
                                start_t = time.time()
                                ext = uploaded_audio.name.split(".")[-1]
                                extracted = transcribe_audio_file(uploaded_audio.getvalue(), file_extension=ext)
                                elapsed = time.time() - start_t
                                st.session_state.current_input_text = extracted
                                st.session_state.current_doc_title = uploaded_audio.name
                                w_count = len(extracted.split())
                                status.update(label=f"✅ Transcribed {w_count:,} words in {elapsed:.2f}s", state="complete")
                            except Exception:
                                status.update(label="⚠️ Unable to transcribe the audio file. Please check the file and try again.", state="error")

        # Document Stats chip
        current_words = len(st.session_state.current_input_text.split())
        if current_words > 0:
            st.caption(f"📊 **Content Ready:** `{current_words:,}` words | `{count_sentences(st.session_state.current_input_text)}` sentences")

        st.markdown("---")

        # 3. SUMMARY SETTINGS
        st.markdown("### 3. Summary Settings")
        
        # Summary Length
        length_col, format_col = st.columns([1, 1.2])
        with length_col:
            summary_length = st.radio(
                "Summary Length:",
                options=["Short", "Balanced", "Detailed"],
                index=["Short", "Balanced", "Detailed"].index(st.session_state.current_summary_length),
                horizontal=True,
                help="Short (~20% ratio), Balanced (~35% ratio), Detailed (~50% ratio)."
            )
            st.session_state.current_summary_length = summary_length

        with format_col:
            output_format = st.selectbox(
                "Output Format:",
                options=FORMAT_CHOICES,
                index=FORMAT_CHOICES.index(st.session_state.current_summary_style) if st.session_state.current_summary_style in FORMAT_CHOICES else 0,
                help="Choose structured output: Bullet Points, Paragraph, Executive Summary, Key Takeaways, Structured Summary, or Study Notes."
            )
            st.session_state.current_summary_style = output_format

        # Focus (Optional)
        focus_input = st.text_input(
            "🎯 Focus (Optional):",
            value=st.session_state.focus_query,
            placeholder="What should the summary emphasize? (e.g. methodology, financial metrics)",
            help="Guides the summarizer to prioritize sections addressing your specific query."
        )
        st.session_state.focus_query = focus_input

        # Advanced Controls (Collapsed by default)
        with st.expander("▾ Advanced Controls", expanded=False):
            st.markdown("##### Summarization Strategy")
            strat_options = ["Smart Auto (Recommended)", "Extractive", "Abstractive", "Hybrid"]
            chosen_strat_mode = st.radio(
                "Strategy Selection:",
                options=strat_options,
                index=0,
                horizontal=False,
                help="Smart Auto chooses the optimal algorithm based on document length and available models."
            )
            st.caption(
                "• **Smart Auto:** Automatically selects the best strategy.\n"
                "• **Extractive:** Selects salient sentences preserving original phrasing.\n"
                "• **Abstractive:** Generates a new, concise synthesis using AI.\n"
                "• **Hybrid:** Combines salient sentence extraction with neural refinement."
            )

            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                manual_extractive = st.selectbox(
                    "Extractive Algorithm:",
                    options=["Automatic", "TextRank", "TF-IDF", "LSA"],
                    index=0,
                    help="Graph-based TextRank, word-weighted TF-IDF, or Latent Semantic Analysis."
                )
            with col_adv2:
                ai_options = ["Automatic", "DistilBART", "T5-Small"]
                if has_gemini_active:
                    ai_options.insert(1, "Gemini AI")
                manual_ai = st.selectbox(
                    "AI Model:",
                    options=ai_options,
                    index=0,
                    help="Neural model for abstractive synthesis."
                )

            col_lang, col_sents = st.columns(2)
            with col_lang:
                target_lang = st.selectbox(
                    "Language:",
                    options=["en", "hi", "mr"],
                    index=0,
                    format_func=lambda x: config.SUPPORTED_LANGUAGES.get(x, x)
                )
            with col_sents:
                total_sents = max(1, count_sentences(st.session_state.current_input_text))
                manual_sentences = st.number_input(
                    "Target Sentences Override (Optional):",
                    min_value=0,
                    max_value=max(5, total_sents),
                    value=0,
                    help="Leave 0 for automatic length based on preset (Short/Balanced/Detailed)."
                )

        # Primary Action Button
        generate_clicked = st.button("✨ Generate Summary", type="primary", use_container_width=True)


    # ----------------------------------------------------
    # GENERATION & SUMMARY DISPLAY (RIGHT COLUMN)
    # ----------------------------------------------------
    with col_result:
        def run_generation(override_length=None, override_format=None, override_focus=None):
            raw_input = st.session_state.current_input_text.strip()
            if not raw_input:
                st.warning("Please provide or upload a document first.")
                return

            length_to_use = override_length or st.session_state.current_summary_length
            format_to_use = override_format or st.session_state.current_summary_style
            focus_to_use = override_focus if override_focus is not None else st.session_state.focus_query

            with st.spinner("Processing document with SmartSum pipeline..."):
                prog_bar = st.progress(0, text="Initializing SmartSum pipeline...")

                def update_prog(pct, msg):
                    prog_bar.progress(pct, text=msg)

                try:
                    start_time = time.time()
                    pipeline_result = execute_smart_pipeline(
                        raw_input=raw_input,
                        summarization_mode=chosen_strat_mode,
                        length_preset=length_to_use,
                        output_format=format_to_use,
                        focus=focus_to_use,
                        target_language=target_lang,
                        manual_extractive_algo=manual_extractive,
                        manual_ai_model=manual_ai,
                        manual_sentences=manual_sentences if manual_sentences > 0 else None,
                        has_gemini=has_gemini_active,
                        lite_mode=st.session_state.lite_mode,
                        progress_callback=update_prog
                    )
                    elapsed = time.time() - start_time

                    st.session_state.current_summary_text = pipeline_result["raw_summary"]
                    st.session_state.current_summary_style = format_to_use
                    st.session_state.current_summary_length = length_to_use
                    st.session_state.current_metrics = pipeline_result["metrics"]
                    st.session_state.current_strategy_info = {
                        "title": pipeline_result["strategy_title"],
                        "rationale": pipeline_result["strategy_rationale"],
                        "time_elapsed": round(elapsed, 2)
                    }

                    # Save to DB if authenticated
                    if st.session_state.get("user_id") and not st.session_state.get("is_guest", True):
                        doc_title = st.session_state.current_doc_title or "Document Summary"
                        m = pipeline_result["metrics"]
                        db_module.save_summary(
                            user_id=st.session_state.user_id,
                            title=doc_title[:80],
                            method=pipeline_result["strategy_title"][:50],
                            style=format_to_use,
                            language=target_lang,
                            input_type=active_src,
                            original_words=m["orig_words"],
                            summary_words=m["summ_words"],
                            compression_ratio=m["compression_ratio"],
                            time_saved_min=m["time_saved_min"],
                            summary_text=pipeline_result["raw_summary"]
                        )
                    st.rerun()

                except Exception as err:
                    st.error(f"Unable to complete summarization: {err}")

        if generate_clicked:
            run_generation()

        # Render Generated Summary if available
        if st.session_state.current_summary_text:
            metrics = st.session_state.current_metrics
            strat_info = st.session_state.current_strategy_info

            # SUMMARY READY BANNER
            st.markdown(
                f"""
                <div class="summary-ready-banner animate-slide-up">
                    <div class="summary-ready-tag">SUMMARY READY ✓</div>
                    <div style="font-size: 0.86rem; color: var(--text-muted); font-weight: 500;">
                        {strat_info.get('title', 'Smart Auto')} ({strat_info.get('time_elapsed', 0.5)}s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Metadata chips
            orig_w = metrics.get("orig_words")
            if orig_w is None:
                orig_w = count_words(st.session_state.current_input_text)
            summ_w = metrics.get("summ_words")
            if summ_w is None:
                summ_w = count_words(st.session_state.current_summary_text)
            comp_r = metrics.get("compression_ratio")
            if comp_r is None:
                comp_r = round((1 - summ_w / max(1, orig_w)) * 100, 1)
            t_saved = metrics.get("time_saved_min")
            if t_saved is None:
                t_saved = round(max(0.0, (orig_w - summ_w) / 200.0), 2)

            st.markdown(
                f"""
                <div class="summary-meta-chips">
                    <div class="summary-chip">📄 <strong>{st.session_state.current_doc_title}</strong></div>
                    <div class="summary-chip">Original: <strong>{orig_w:,} words</strong></div>
                    <div class="summary-chip">Summary: <strong>{summ_w:,} words</strong></div>
                    <div class="summary-chip">Compression: <strong>{comp_r:.1f}%</strong></div>
                    <div class="summary-chip">Time Saved: <strong>~{t_saved:.1f} min</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Summary Content Display
            formatted_html = apply_summary_style_html(
                st.session_state.current_summary_text,
                st.session_state.current_summary_style
            )
            st.markdown(formatted_html, unsafe_allow_html=True)

            # SUMMARY QUALITY (Real calculations)
            readability_val = metrics.get("readability_score")
            coverage_val = metrics.get("keyword_coverage")
            quality_rows = []
            for label, value, suffix in [
                ("Readability (Flesch)", readability_val, "/100"),
                ("Key Content Coverage (lexical overlap)", coverage_val, "%"),
                ("Compression Ratio", comp_r, "%"),
            ]:
                if value is None:
                    quality_rows.append(
                        f'<div class="quality-row"><span class="quality-label">{label}:</span>'
                        '<span class="quality-val">Unavailable</span></div>'
                    )
                else:
                    value = float(value)
                    quality_rows.append(
                        f'<div class="quality-row">'
                        f'<span class="quality-label">{label}:</span>'
                        f'<div class="quality-bar-bg"><div class="quality-bar-fill" '
                        f'style="width: {min(100.0, max(0.0, value))}%;"></div></div>'
                        f'<span class="quality-val">{value:.1f}{suffix}</span></div>'
                    )

            st.markdown(
                f"""
                <div class="quality-card">
                    <div class="quality-header">📊 Summary Quality &amp; Evaluation</div>
                    {''.join(quality_rows)}
                    <div class="quality-note">Coverage reports exact overlap between unique source content lemmas and summary lemmas; it is lexical, not semantic concept coverage.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # REFINEMENT ACTIONS
            st.markdown("##### Refine & Adapt Summary")
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                if st.button("Regenerate", use_container_width=True, help="Re-run the current summarization pipeline."):
                    run_generation()
            with r_col2:
                if st.button("Make Detailed", use_container_width=True, help="Generate a more comprehensive version."):
                    run_generation(override_length="Detailed")
            with r_col3:
                if st.button("Simplify", use_container_width=True, help="Reformat the summary into a simpler, easier-to-read variant."):
                    run_generation(override_format="ELI5 (Explain Like I'm 5)")
            with r_col4:
                if st.button("Key Points", use_container_width=True, help="Refocus the summary on the key findings and conclusions."):
                    run_generation(override_focus="key findings, main conclusions, and core takeaways")

            # EXPORT & DOWNLOADS
            st.markdown("---")
            st.markdown("##### 📥 Export Summary")
            plain_export = apply_summary_style(st.session_state.current_summary_text, st.session_state.current_summary_style)
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                txt_bytes = export_as_txt(plain_export)
                st.download_button(
                    "📄 Download TXT",
                    data=txt_bytes,
                    file_name="SmartSum_Summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with exp_col2:
                docx_bytes = export_as_docx(st.session_state.current_doc_title, plain_export)
                st.download_button(
                    "📝 Download DOCX",
                    data=docx_bytes,
                    file_name="SmartSum_Summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            with exp_col3:
                pdf_bytes = export_as_pdf(st.session_state.current_doc_title, plain_export)
                st.download_button(
                    "📑 Download PDF",
                    data=pdf_bytes,
                    file_name="SmartSum_Summary.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        else:
            # Modern Empty State
            st.markdown(
                """
                <div class="empty-state-card">
                    <svg class="empty-state-illustration" viewBox="0 0 220 145" role="img" aria-label="SmartSum waiting for content">
                        <rect x="54" y="12" width="104" height="120" rx="12" fill="var(--surface-raised)" stroke="var(--border)" stroke-width="3"/>
                        <path d="M77 48h57M77 66h57M77 84h38" fill="none" stroke="var(--accent)" stroke-width="6" stroke-linecap="round"/>
                        <circle cx="158" cy="105" r="27" fill="var(--surface)" stroke="var(--accent)" stroke-width="4"/>
                        <path d="M150 105h16M158 97v16" fill="none" stroke="var(--accent-strong)" stroke-width="4" stroke-linecap="round"/>
                    </svg>
                    <div class="empty-state-title">Your intelligent summary will appear here</div>
                    <p class="empty-state-hint">
                        1. Select your source on the left (Text, PDF, DOCX, URL, YouTube, Audio).<br>
                        2. Smart Auto will automatically select the best summarization strategy.<br>
                        3. Click <strong>Generate Summary</strong> to analyze and synthesize.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------
# TAB 2: HISTORY (AUTHENTICATION PROTECTED)
# ----------------------------------------------------
with tab_history:
    st.subheader("🕘 Saved Summaries & History")
    if st.session_state.get("current_user") is None or st.session_state.get("is_guest", True):
        st.markdown(
            """
            <div class="locked-state-card animate-slide-up" style="background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 32px; text-align: center; margin: 24px auto; max-width: 520px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🔒</div>
                <h3 style="margin: 0 0 8px; color: var(--text);">Login to unlock History</h3>
                <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 20px;">
                    Create a free account to automatically save your summaries, organize favorites, and re-export past analyses at any time.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_h1, col_h2, col_h3 = st.columns([1, 1.2, 1])
        with col_h2:
            if st.button("🔐 Login / Create Account", key="history_login_gate_btn", type="primary", use_container_width=True):
                st.session_state.is_guest = False
                st.rerun()
    else:
        user_id = st.session_state.user_id
        search_col, filter_col = st.columns([2, 1])
        with search_col:
            search_query = st.text_input("🔍 Search history by title or text:", key="hist_search_q")
        with filter_col:
            fav_only = st.checkbox("★ Favorites only", key="hist_fav_only")

        summaries = db_module.get_user_summaries(
            user_id,
            search=search_query or None,
            favorites_only=fav_only
        )

        if not summaries:
            st.info("No saved summaries found. Generate a summary in the Summarize tab to build your library!")
        else:
            for item in summaries:
                with st.expander(f"{item['title']}  •  {item['created_at'][:10]}  •  {item['method']}", expanded=False):
                    st.write(item["summary_text"])
                    st.caption(f"Format: {item['style']} | Words: {item['summary_words']} | Compression: {item['compression_ratio']:.1f}%")
                    
                    row1, row2, row3, row4 = st.columns([1.2, 1, 1, 1.2])
                    with row1:
                        if st.button("📂 Open in Summarize", key=f"open_hist_{item['id']}", use_container_width=True):
                            st.session_state.current_summary_text = item["summary_text"]
                            st.session_state.current_summary_style = item["style"]
                            st.session_state.current_doc_title = item["title"]
                            st.success("Summary loaded into Summarize tab!")
                    with row2:
                        fav_state = bool(item.get("favorite"))
                        fav_label = "★ Unfavorite" if fav_state else "☆ Favorite"
                        if st.button(fav_label, key=f"fav_hist_{item['id']}", use_container_width=True):
                            db_module.toggle_favorite(item["id"], not fav_state)
                            st.rerun()
                    with row3:
                        if st.button("🗑️ Delete", key=f"del_hist_{item['id']}", use_container_width=True):
                            db_module.delete_summary(item["id"])
                            st.rerun()
                    with row4:
                        plain_txt = apply_summary_style(item["summary_text"], item["style"])
                        st.download_button(
                            "📄 Download TXT",
                            data=export_as_txt(plain_txt),
                            file_name=f"{item['title']}.txt",
                            mime="text/plain",
                            key=f"dl_txt_hist_{item['id']}",
                            use_container_width=True
                        )


# ----------------------------------------------------
# TAB 3: INSIGHTS (CONTENT ANALYTICS)
# ----------------------------------------------------
with tab_insights:
    st.subheader("🔍 Document Content Insights")
    doc_text = st.session_state.current_input_text
    summ_text = st.session_state.current_summary_text

    if not doc_text:
        st.info("Please provide content or generate a summary in the Summarize tab to view document insights.")
    else:
        ins_tab1, ins_tab2, ins_tab3, ins_tab4, ins_tab5 = st.tabs([
            "🔑 Keywords & WordCloud",
            "🏷️ Named Entities",
            "🎭 Sentiment Tone",
            "🕸️ Concept Map",
            "📋 Action Items"
        ])

        with ins_tab1:
            st.markdown("##### Extracted Key Topics (YAKE & KeyBERT)")
            k1, k2 = st.columns(2)
            with k1:
                yake_kws = extract_yake_keywords(doc_text, top_n=8)
                if yake_kws:
                    st.dataframe(pd.DataFrame(yake_kws, columns=["Keyword", "Relevance Score"]), use_container_width=True)
            with k2:
                keybert_kws = extract_keybert_keywords(doc_text, top_n=8)
                if keybert_kws:
                    st.dataframe(pd.DataFrame(keybert_kws, columns=["Keyword", "Semantic Similarity"]), use_container_width=True)
                else:
                    st.caption("KeyBERT requires sentence-transformers embeddings.")

            st.markdown("##### WordCloud Visualization")
            fig = generate_wordcloud(doc_text)
            if fig:
                st.pyplot(fig)

        with ins_tab2:
            st.markdown("##### Named Entity Recognition (spaCy)")
            entities_html = render_highlighted_entities_html(doc_text[:3000])
            st.components.v1.html(f"<div style='font-family:sans-serif; line-height:1.6; color:#e8eef8;'>{entities_html}</div>", height=300, scrolling=True)

        with ins_tab3:
            st.markdown("##### Sentiment Tone Preservation (Original vs Summary)")
            if summ_text:
                sent_res = compare_sentiments(doc_text, summ_text)
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.metric("Original Document Tone", sent_res["original"]["label"], delta=f"Score: {sent_res['original']['compound']}")
                with sc2:
                    st.metric("Summary Tone", sent_res["summary"]["label"], delta=f"Score: {sent_res['summary']['compound']}")
                if sent_res["sentiment_preserved"]:
                    st.success("✅ Summary faithfully preserves original sentiment tone.")
                else:
                    st.warning("⚠️ Sentiment drift detected between original and summary.")
            else:
                st.info("Generate a summary to compare sentiment preservation.")

        with ins_tab4:
            st.markdown("##### Concept Mind Map")
            html_graph = generate_mindmap_html(doc_text)
            st.components.v1.html(html_graph, height=480, scrolling=False)

        with ins_tab5:
            st.markdown("##### Extracted Action Items & Tasks")
            items = extract_action_items(doc_text)
            if items:
                st.dataframe(pd.DataFrame(items), use_container_width=True)
            else:
                st.info("No explicit actionable directives detected in the document.")


# ----------------------------------------------------
# TAB 4: STUDY TOOLS
# ----------------------------------------------------
with tab_study:
    st.subheader("🎓 Study Tools & Knowledge Checks")
    text_for_study = st.session_state.current_summary_text or st.session_state.current_input_text

    if not text_for_study:
        st.info("Please generate a summary or provide text in the Summarize tab first to activate study tools.")
    else:
        study_tab1, study_tab2 = st.tabs(["📝 Interactive Quiz", "🎴 Flashcards"])

        with study_tab1:
            if "study_mcqs" not in st.session_state:
                st.session_state.study_mcqs = generate_mcqs(text_for_study, num_questions=4)

            if st.button("🔄 Regenerate Questions", key="regen_study_quiz"):
                st.session_state.study_mcqs = generate_mcqs(text_for_study, num_questions=4)

            user_answers = {}
            for idx, q in enumerate(st.session_state.study_mcqs):
                st.markdown(f"**Question {idx+1}: {q['question']}**")
                user_answers[idx] = st.radio(f"Answer for Q{idx+1}:", options=q['options'], key=f"quiz_ans_{idx}")
                st.markdown("---")

            if st.button("Submit Answers & Check Score", type="primary"):
                score = 0
                for idx, q in enumerate(st.session_state.study_mcqs):
                    if user_answers[idx] == q['answer']:
                        score += 1
                        st.success(f"Q{idx+1}: Correct! ({q['explanation']})")
                    else:
                        st.error(f"Q{idx+1}: Incorrect. Correct answer: **{q['answer']}** ({q['explanation']})")
                st.markdown(f"### Final Score: `{score} / {len(st.session_state.study_mcqs)}`")

        with study_tab2:
            flashcards = generate_flashcards(text_for_study)
            for card in flashcards:
                with st.expander(f"🎴 {card['front']}"):
                    st.write(card["back"])


# ----------------------------------------------------
# TAB 5: COMPARE METHODS
# ----------------------------------------------------
with tab_compare:
    st.subheader("⚖️ Compare Summarization Methods")
    st.caption("Compare extractive, neural, and hybrid algorithms side-by-side on your current document.")
    text_to_compare = st.session_state.current_input_text

    if not text_to_compare:
        st.info("Please provide document content in the Summarize tab to compare algorithms.")
    else:
        if st.button("🚀 Run Comparative Benchmark on Document", type="primary"):
            methods_to_run = [
                ("TextRank (Extractive)", lambda t: textrank_summarize(t, num_sentences=4)),
                ("TF-IDF (Extractive)", lambda t: tfidf_summarize(t, num_sentences=4)),
                ("LSA (Extractive)", lambda t: lsa_summarize(t, num_sentences=4)),
            ]
            if has_gemini_active:
                methods_to_run.append(("Gemini AI (Abstractive)", lambda t: gemini_summarize(t, style="Paragraph")))
            elif not st.session_state.lite_mode:
                methods_to_run.append(("DistilBART (Abstractive)", lambda t: distilbart_summarize(t)))

            results = []
            with st.spinner("Generating comparative summaries..."):
                for name, fn in methods_to_run:
                    t0 = time.time()
                    try:
                        summ = fn(text_to_compare)
                        dur = round(time.time() - t0, 2)
                        w = len(summ.split())
                        comp = round((1 - (w / max(1, len(text_to_compare.split())))) * 100, 1)
                        results.append({
                            "Method": name,
                            "Summary": summ,
                            "Words": w,
                            "Reduction": f"{comp}%",
                            "Time (s)": dur
                        })
                    except Exception as err:
                        results.append({
                            "Method": name,
                            "Summary": f"Error: {err}",
                            "Words": 0,
                            "Reduction": "0%",
                            "Time (s)": 0
                        })

            st.dataframe(pd.DataFrame([{"Method": r["Method"], "Words": r["Words"], "Reduction": r["Reduction"], "Time (s)": r["Time (s)"]} for r in results]), use_container_width=True)

            for r in results:
                with st.expander(f"📌 {r['Method']} — {r['Words']} words in {r['Time (s)']}s"):
                    st.write(r["Summary"])


# ----------------------------------------------------
# TAB 6: ASK DOCUMENT (DOCUMENT Q&A)
# ----------------------------------------------------
with tab_qa:
    st.subheader("💬 Ask Questions About Document")
    doc_for_qa = st.session_state.current_input_text

    if not doc_for_qa:
        st.info("Please load or paste a document in the Summarize tab first.")
    else:
        chat_user_id = st.session_state.user_id if st.session_state.get("user_id") and not st.session_state.get("is_guest", True) else None
        
        # Display history
        if chat_user_id:
            chat_msgs = db_module.get_chat_messages(chat_user_id)
        else:
            chat_msgs = st.session_state.get("chat_messages", [])

        for msg in chat_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        q_prompt = st.chat_input("Ask a question about your document...")
        if q_prompt:
            # Append user message
            if chat_user_id:
                db_module.save_chat_message(chat_user_id, "user", q_prompt)
            else:
                st.session_state.chat_messages.append({"role": "user", "content": q_prompt})

            # Retrieve context & answer
            from core.query_focused import get_relevant_sentences
            rel_sents = get_relevant_sentences(doc_for_qa, q_prompt, top_k=4)
            context = " ".join(rel_sents)

            if has_gemini_active:
                ans_prompt = f"Using ONLY the following context from the document, answer the question accurately:\n\nContext:\n{context}\n\nQuestion: {q_prompt}"
                try:
                    answer = gemini_summarize(ans_prompt, style="Paragraph")
                except Exception:
                    answer = f"Relevant information found: {context}"
            else:
                answer = f"Relevant information from document:\n\n{context}"

            if chat_user_id:
                db_module.save_chat_message(chat_user_id, "assistant", answer, source_snippets=rel_sents)
            else:
                st.session_state.chat_messages.append({"role": "assistant", "content": answer, "source_snippets": rel_sents})

            st.rerun()


# Render Footer
render_footer()
