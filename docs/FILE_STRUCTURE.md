# File Structure

```
NLP_PROJ/
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── FILE_STRUCTURE.md
│   └── STATUS_REPORT.md
├── data/
│   ├── metadata.json
│   └── scholar_index.faiss
├── backend/
│   ├── app.py
│   └── scholar_index_builder.py
├── frontend/
│   └── streamlit_app.py
├── utils/
│   └── config.py
├── .env
├── .env.example
├── .gitignore
└── README.md
```

Notes:
- `.env` is gitignored; `.env.example` stays tracked for reference.
- `data/` contains the FAISS index and metadata consumed by the backend.
- `utils/config.py` centralizes secret/config resolution.

