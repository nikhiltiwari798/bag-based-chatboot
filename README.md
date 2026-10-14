# CareDocs — Personal Medical Document Q&A

CareDocs is a retrieval-augmented question-answering demo for a person's own uploaded medical documents. It searches the documents, returns relevant excerpts, and produces answers with source citations. PDF, TXT, and Markdown files are supported. There is intentionally no bundled medical corpus.

## Medical safety and privacy

**CareDocs is an educational document assistant, not a medical device. It does not diagnose, recommend treatment, interpret results as a clinician, or replace professional medical advice.** For severe, sudden, or potentially life-threatening symptoms, contact local emergency services immediately.

Only upload documents you are authorized to use. Avoid uploading third-party health data unless you have their informed consent. By default, uploads and extracted text remain in the current Streamlit session's memory and are not written to disk. Data can still be present in your browser/session and machine memory; this is not a certified health-record system.

With no model credentials, answers are assembled locally from retrieved passages using TF-IDF similarity. If you configure `OPENAI_API_KEY`, the retrieved excerpts and your question are sent to that configured external provider to generate a response. Review its data-handling terms and configure it only if that disclosure is acceptable. The app displays this distinction before chat use.

## Setup

Requires Python 3.10 or later.

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Optionally copy `.env.example` to `.env` and set an API key/model if you want provider-generated answers. Never commit `.env`.
4. Launch with `streamlit run app.py`.
5. Run tests with `pytest`.

For hosting on Streamlit Community Cloud, see [DEPLOYMENT.md](DEPLOYMENT.md). This workspace needs a GitHub repository before the hosted deployment can be created.

Without an API key, the app still works using local retrieval and extractive answers. No data is uploaded to an AI provider in that mode.

## Workflow

1. Upload one or more of your own PDF, TXT, or Markdown documents.
2. CareDocs extracts text in memory and splits it into overlapping chunks.
3. TF-IDF vector retrieval ranks chunks against your question.
4. Ask a question. The local mode displays relevant excerpts with citations; the optional model mode receives only the retrieved excerpts and question, with source labels, and is instructed not to answer beyond them.
5. Use “Clear session documents” to discard uploaded text from the current session.

## Limitations

- Retrieval quality depends on text extraction, scan quality, document specificity, and the wording of the question. Scanned PDFs without selectable text may not work.
- Citations indicate which uploaded document excerpts were retrieved; they do not independently verify medical accuracy.
- Do not use it for emergencies, diagnosis, medication changes, or treatment decisions. Confirm important information with a qualified healthcare professional.
