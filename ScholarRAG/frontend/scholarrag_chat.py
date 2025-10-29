import time
import requests
import streamlit as st

from utils.config import get_backend_base_url


st.set_page_config(page_title="ScholarRAG Chat", layout="wide")
st.title("🎓 ScholarRAG – Ask about research")

backend = get_backend_base_url()

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask_form"):
    query = st.text_input("Ask a question about research:")
    k = st.number_input("Results", min_value=1, max_value=10, value=10)
    col1, col2 = st.columns(2)
    with col1:
        year_from = st.number_input("Year from", min_value=1900, max_value=2100, value=2000)
    with col2:
        year_to = st.number_input("Year to", min_value=1900, max_value=2100, value=2100)
    submitted = st.form_submit_button("Ask")

if submitted and query:
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{backend}/ask", json={"query": query, "k": int(k), "year_from": int(year_from), "year_to": int(year_to)})
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            data = None

    if data:
        st.session_state.history.append({"query": query, "response": data})

for turn in reversed(st.session_state.history):
    st.markdown(f"**You:** {turn['query']}")
    st.markdown("---")
    st.markdown("**Answer**")
    st.markdown(turn["response"]["answer"])
    st.markdown("**Sources**")
    for i, s in enumerate(turn["response"]["sources"], start=1):
        with st.expander(f"[{i}] {s['title']} ({s['year']}) – sim {s['similarity']}"):
            link = s.get("doi")
            if link:
                st.write(f"DOI: https://doi.org/{link}")
            st.write(s.get("why_relevant") or "")
            st.write(s.get("snippet") or "")
    if turn["response"].get("fallback_used"):
        st.info("Expanded search used (OpenAlex fallback)")

