"""
Mind Map / Concept Graph Visualizer for SmartSum using NetworkX and PyVis.
Builds concept nodes from top keywords, entities, and sentence co-occurrence.
"""
import os
import tempfile
import logging
from typing import List
import networkx as nx
from extras.keywords import extract_yake_keywords
from extras.entities import extract_named_entities

logger = logging.getLogger("SmartSum.MindMap")

def generate_mindmap_html(text: str, max_nodes: int = 15) -> str:
    """
    Construct a concept graph of keywords and entities, then export as interactive PyVis HTML string.
    """
    if not text or not text.strip():
        return "<div>No content available for mind map generation.</div>"

    try:
        from pyvis.network import Network
        
        # Extract keywords and entities
        yake_kws = extract_yake_keywords(text, top_n=max_nodes)
        entities_dict = extract_named_entities(text)

        G = nx.Graph()
        
        # Central Topic Node
        root_label = "Core Document Concept"
        G.add_node(root_label, label=root_label, color="#1E88E5", size=30, title="Document Root")

        # Add Keyword Nodes
        for kw, score in yake_kws[:10]:
            kw_name = kw.title()
            if kw_name != root_label:
                G.add_node(kw_name, label=kw_name, color="#42A5F5", size=20, title=f"Keyword (Relevance: {score})")
                G.add_edge(root_label, kw_name, weight=score)

        # Add Named Entity Nodes
        for label_type, ent_list in entities_dict.items():
            if label_type in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                for ent in ent_list[:3]:
                    if ent not in G:
                        G.add_node(ent, label=f"{ent} ({label_type})", color="#66BB6A", size=18, title=f"Entity: {label_type}")
                        G.add_edge(root_label, ent, weight=1.5)

        # Build PyVis Network
        net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False)
        net.from_nx(G)
        net.toggle_physics(True)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            net.write_html(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return html_content
    except Exception as e:
        logger.error(f"Mind map generation error: {e}")
        return f"<div style='color:red;'>Could not generate interactive mind map ({e}).</div>"
