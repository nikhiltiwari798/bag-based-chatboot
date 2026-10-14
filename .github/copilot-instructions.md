# CareDocs project guidance

- Python Streamlit app for retrieval over user-uploaded medical documents.
- Uploaded document text must remain in memory by default; do not log, persist, or commit health information.
- Every grounded answer must cite retrieved document names and chunk references; if evidence is missing, say so.
- Keep the educational-only, non-diagnostic safety notices visible; urgent symptoms should be directed to emergency services.
- Optional external model use is controlled by environment configuration and must disclose when uploaded excerpts are sent to that provider.
- Run tests with `pytest`; launch using `streamlit run app.py`.
