# streamlit_app.py
import streamlit as st
import requests

from utils.config import get_backend_base_url

st.title("📚 ScholarRAG Search")
query = st.text_input("Enter your research query:")
backend_base = get_backend_base_url()

if st.button("Search"):
    try:
        response = requests.get(f"{backend_base}/search", params={"query": query})
        response.raise_for_status()
    except requests.RequestException as err:
        st.error(f"Backend error: {err}")
    else:
        data = response.json()
        if data.get("error"):
            st.error(data["error"])
        elif not data.get("results"):
            st.warning("No results found.")
        else:
            for paper in data["results"]:
                st.subheader(paper["title"])
                st.write(f"Year: {paper['year']}")
                st.write(f"DOI: {paper['doi']}")
                st.write(f"Concepts: {paper['concepts']}")
                st.write(f"Similarity: {paper['similarity']:.3f}")
