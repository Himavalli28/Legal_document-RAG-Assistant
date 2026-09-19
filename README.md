# Legal Document RAG Assistant

A simple educational RAG demonstration for questions over uploaded legal PDFs.

> **Important:** This is an educational information-retrieval system, not legal advice. Always verify the original document and consult a qualified professional.

## Run in five minutes

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Upload one or more text-based PDFs, then ask a question. The app extracts page text, creates overlapping chunks, generates local TF-IDF embeddings, performs cosine-similarity vector search, and displays cited evidence. It returns **Information not found** when the best similarity is below the grounding threshold.

## Viva talking points

- Retrieval is separate from answer construction, so the evidence can be inspected.
- TF-IDF embeddings are a lightweight baseline; semantic embeddings can improve synonym matching.
- The threshold reduces unsupported answers but can also create false negatives.
- Production systems should handle OCR, document versions, amendments, jurisdiction, access control, evaluation datasets, and human review.
