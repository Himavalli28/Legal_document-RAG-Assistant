from io import BytesIO
from typing import Any

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Legal Document RAG", page_icon="⚖️", layout="wide")


def extract_chunks(file_bytes: bytes, filename: str, chunk_size: int = 900, overlap: int = 150) -> list[dict[str, Any]]:
    reader = PdfReader(BytesIO(file_bytes))
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\n", " ").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "source": filename, "page": page_number})
            if end == len(text):
                break
            start = end - overlap
    return chunks


def search(query: str, chunks: list[dict[str, Any]], vectorizer: TfidfVectorizer, matrix, top_k: int = 4):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[index], "score": float(scores[index])} for index in ranked]


def grounded_answer(query: str, results: list[dict[str, Any]], threshold: float = 0.12) -> str:
    useful = [item for item in results if item["score"] >= threshold]
    if not useful:
        return "Information not found in the uploaded documents. Try different keywords or upload the relevant act/rule."
    lines = ["Based only on the retrieved document passages:"]
    for item in useful[:3]:
        excerpt = item["text"].replace("  ", " ")
        if len(excerpt) > 420:
            excerpt = excerpt[:420].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {excerpt} [Source: {item['source']}, page {item['page']}]")
    return "\n".join(lines)


st.title("⚖️ Legal Document RAG Assistant")
st.caption("A small, transparent retrieval-augmented generation demonstration")
st.warning("Educational information-retrieval system only. This is not legal advice. Verify the original legislation and consult a qualified professional.")

with st.sidebar:
    st.header("1. Build the index")
    uploaded_files = st.file_uploader("Upload PDF acts, rules, or regulations", type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk size", 400, 1600, 900, 100)
    top_k = st.slider("Retrieved passages", 1, 6, 4)
    st.header("Pipeline")
    st.write("PDF → text chunks → TF-IDF embeddings → cosine search → cited answer")

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        all_chunks.extend(extract_chunks(uploaded_file.getvalue(), uploaded_file.name, chunk_size))
    if all_chunks:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([item["text"] for item in all_chunks])
        st.session_state["index"] = (all_chunks, vectorizer, matrix)
        st.sidebar.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} PDF(s).")
    else:
        st.error("No selectable text was found. This demo needs text-based PDFs, not scanned images.")

st.header("2. Ask a grounded question")
query = st.text_input("Question", placeholder="What is the penalty for non-compliance?")
ask = st.button("Search documents", type="primary", disabled="index" not in st.session_state)

if ask and query.strip():
    chunks, vectorizer, matrix = st.session_state["index"]
    results = search(query, chunks, vectorizer, matrix, top_k)
    st.subheader("Answer")
    st.write(grounded_answer(query, results))
    st.subheader("Retrieved evidence")
    for number, item in enumerate(results, start=1):
        with st.expander(f"{number}. {item['source']} · page {item['page']} · similarity {item['score']:.3f}"):
            st.write(item["text"])

with st.expander("Viva notes: grounding, hallucination, and limitations"):
    st.markdown("""
- **Grounding:** The response is built from the highest-scoring passages and every passage includes a filename and page citation.
- **Hallucination control:** If no result crosses the similarity threshold, the system says **Information not found** instead of guessing.
- **Retrieval accuracy:** TF-IDF is fast and explainable, but lexical matching can miss synonyms and legal concepts. Test with known questions and measure recall@k.
- **LLM limitations:** A production LLM may paraphrase incorrectly, rely on stale knowledge, or overstate certainty. Restrict its context, require citations, and show the retrieved text.
- **Scope:** Scanned PDFs, tables, OCR errors, conflicting amendments, and jurisdiction/version issues need extra handling.
""")
