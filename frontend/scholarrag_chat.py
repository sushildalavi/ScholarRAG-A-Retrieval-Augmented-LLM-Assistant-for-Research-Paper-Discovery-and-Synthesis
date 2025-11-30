import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from utils.config import get_backend_base_url


st.set_page_config(page_title="ScholarRAG Chat", layout="wide")

# Minimal ChatGPT-inspired theming
st.markdown(
    """
    <style>
    .chat-container {padding: 1rem 0;}
    .stChatMessage {border: 1px solid #e0e0e0; border-radius: 10px; padding: 0.9rem 1rem; background: #fafafa;}
    .stChatMessage[data-testid="stChatMessage-user"] {background: #f1f5fb; border-color: #d2def0;}
    .source-chip {display: inline-block; margin-right: 8px; margin-bottom: 6px; padding: 4px 8px; border-radius: 999px; background: #eef2f7; font-size: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

backend = get_backend_base_url()

if "history" not in st.session_state:
    st.session_state.history = []


def stream_markdown(text: str, placeholder, delay: float = 0.02) -> None:
    """Render markdown with a simple streamed effect."""
    acc = ""
    for chunk in text.split(" "):
        acc += (chunk + " ")
        placeholder.markdown(acc + "▌")
        time.sleep(delay)
    placeholder.markdown(acc)


with st.sidebar:
    st.header("Controls")
    k = st.number_input("Results", min_value=1, max_value=10, value=10)
    year_from = st.number_input("Year from", min_value=1900, max_value=2100, value=2000)
    year_to = st.number_input("Year to", min_value=1900, max_value=2100, value=2100)
    st.caption("Answers stream in like ChatGPT once the backend responds.")

st.title("🎓 ScholarRAG – Ask about research")
st.write("Chat-style answers with live streaming and source context.")

query = st.chat_input("Ask a question about research…")

if query:
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                f"{backend}/ask",
                json={"query": query, "k": int(k), "year_from": int(year_from), "year_to": int(year_to)},
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            data = None

    if data:
        st.session_state.history.append({"query": query, "response": data, "streamed": False})

for idx, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(turn["query"])

    with st.chat_message("assistant"):
        answer = turn["response"].get("answer", "")
        placeholder = st.empty()
        # Stream only the latest unstreamed turn; prior turns render statically
        if idx == len(st.session_state.history) - 1 and not turn.get("streamed", False):
            stream_markdown(answer, placeholder)
            turn["streamed"] = True
        else:
            placeholder.markdown(answer)

        metrics = turn["response"].get("metrics", {})
        stats = turn["response"].get("candidate_counts", {})
        fallback_flag = metrics.get("fallback_used", turn["response"].get("fallback_used"))
        hops = turn["response"].get("hops", [])

        if metrics:
            cols = st.columns(4)
            latency = metrics.get("latency_ms")
            cols[0].metric("Latency (ms)", f"{latency:.1f}" if latency is not None else "—")
            max_sim = metrics.get("max_similarity")
            cols[1].metric("Max similarity", f"{max_sim:.3f}" if max_sim is not None else "—")
            cols[2].metric("Fallback", "Yes" if fallback_flag else "No")
            cols[3].metric("Sources", stats.get("pool", "—"))

        if stats:
            detail = f"Pool: {stats.get('pool', 0)} · Ranked: {stats.get('scored', 0)} · OpenAlex: {stats.get('openalex', 0)}"
            detail += f" · arXiv: {stats.get('arxiv', 0)}"
            st.caption(f"Retrieval — {detail}")
        if fallback_flag:
            st.info("Expanded search used (OpenAlex fallback).")
        if hops:
            with st.expander("Multi-hop retrieval steps"):
                for h in hops:
                    st.write(f"- {h.get('query')}: pool {h.get('candidate_counts', {}).get('pool', 0)}, fallback {h.get('fallback_used')}")

        if turn["response"].get("sources"):
            st.markdown("**Sources**")
            src_df = pd.DataFrame(turn["response"]["sources"])
            if not src_df.empty and "similarity" in src_df.columns:
                chart_df = src_df[["title", "similarity"]]
                fig = px.bar(chart_df, x="similarity", y="title", orientation="h", title="Similarity per source")
                fig.update_layout(height=240, yaxis_title="Source", xaxis_title="Similarity")
                st.plotly_chart(fig, use_container_width=True)
            if not src_df.empty and "year" in src_df.columns:
                year_chart = px.histogram(src_df, x="year", nbins=20, title="Year distribution")
                year_chart.update_layout(height=180)
                st.plotly_chart(year_chart, use_container_width=True)

            for i, s in enumerate(turn["response"]["sources"], start=1):
                title = s.get("title") or "Untitled"
                year = s.get("year") or "—"
                trust = s.get("trust_score")
                sim = s.get("similarity", "?")
                header = f"[{i}] {title} ({year}) – sim {sim}"
                if trust is not None:
                    header += f" · trust {trust}"
                with st.expander(header):
                    link = s.get("doi")
                    if link:
                        st.write(f"DOI: https://doi.org/{link}")
                    st.write(s.get("why_relevant") or "")
                    st.write(s.get("snippet") or "")
                    concepts = s.get("concepts") or []
                    authors = s.get("authors") or []
                    chips = []
                    if concepts:
                        chips.append("Concepts: " + ", ".join(concepts[:6]))
                    if authors:
                        chips.append("Authors: " + ", ".join(a.get("display_name") for a in authors if a.get("display_name")))
                    if chips:
                        for c in chips:
                            st.caption(c)
