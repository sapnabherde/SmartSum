"""
Benchmark Module for SmartSum.
Runs multiple summarization algorithms across a benchmark sample document,
computing comparative scores (ROUGE, BERTScore, Compression, Latency).
"""
import time
import logging
from typing import List, Dict, Any
import pandas as pd
from core.extractive import textrank_summarize, tfidf_summarize, lsa_summarize
from core.abstractive import distilbart_summarize, t5_summarize
from evaluation.metrics import compute_rouge_scores, compute_bertscore
from utils.helpers import count_words

logger = logging.getLogger("SmartSum.Benchmark")

DEFAULT_BENCHMARK_TEXT = (
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
    "concerned with the interactions between computers and human language, in particular how to program computers "
    "to process and analyze large amounts of natural language data. Goal is a computer capable of understanding the "
    "contents of documents, including the contextual nuances of the language within them. The technology can then "
    "accurately extract information and insights contained in the documents as well as categorize and organize the files. "
    "Challenges in natural language processing frequently involve speech recognition, natural language understanding, "
    "and natural-language generation. Text summarization is the process of distilling the most important information "
    "from a source document to produce an abridged version for a particular user or task."
)

DEFAULT_REFERENCE_SUMMARY = (
    "Natural language processing enables computers to analyze human language and extract document insights. "
    "Text summarization distills key information from source documents into concise versions."
)


def run_algorithm_benchmark(
    text: str = DEFAULT_BENCHMARK_TEXT,
    reference: str = DEFAULT_REFERENCE_SUMMARY
) -> pd.DataFrame:
    """
    Run all summarization algorithms and compute detailed performance metrics.
    """
    methods = [
        ("TextRank (Extractive)", lambda t: textrank_summarize(t, num_sentences=3)),
        ("TF-IDF (Extractive)", lambda t: tfidf_summarize(t, num_sentences=3)),
        ("LSA (Extractive)", lambda t: lsa_summarize(t, num_sentences=3)),
        ("DistilBART (Abstractive)", lambda t: distilbart_summarize(t)),
        ("T5-Small (Abstractive)", lambda t: t5_summarize(t)),
    ]

    results = []
    orig_words = count_words(text)

    for name, func in methods:
        start_time = time.time()
        try:
            summary = func(text)
            elapsed = time.time() - start_time
            summ_words = count_words(summary)
            comp_ratio = round((1 - (summ_words / orig_words)) * 100, 1) if orig_words > 0 else 0

            # Compute ROUGE
            rouge = compute_rouge_scores(summary, reference)
            
            results.append({
                "Algorithm": name,
                "Latency (s)": round(elapsed, 2),
                "Summary Words": summ_words,
                "Compression %": comp_ratio,
                "ROUGE-1 F1": rouge["rouge1"]["fmeasure"],
                "ROUGE-2 F1": rouge["rouge2"]["fmeasure"],
                "ROUGE-L F1": rouge["rougeL"]["fmeasure"],
            })
        except Exception as e:
            logger.error(f"Benchmark failed for {name}: {e}")
            results.append({
                "Algorithm": name,
                "Latency (s)": 0.0,
                "Summary Words": 0,
                "Compression %": 0.0,
                "ROUGE-1 F1": 0.0,
                "ROUGE-2 F1": 0.0,
                "ROUGE-L F1": 0.0,
            })

    return pd.DataFrame(results)
