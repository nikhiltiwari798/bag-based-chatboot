# Deployment (Streamlit Community Cloud)

This app is configured as a Streamlit application with its entry point at `app.py` and dependencies in `requirements.txt`.

## Before you publish

CareDocs processes medical documents. A hosted deployment is not automatically a compliant medical-record service. Do not use real patient-identifiable records on a public or unreviewed deployment. Confirm your organization's privacy/security requirements, hosting terms, access controls, and any applicable legal obligations before using sensitive health data. The app does not persist uploads intentionally, but data is processed by the hosting service in memory while the session is active.

By default, CareDocs uses local TF-IDF retrieval and extractive answers. With `OPENAI_API_KEY` configured, each question sends the question and retrieved document passages to the configured external AI provider. Configure the key only after reviewing that provider's terms and disclosure; never place credentials in repository files.

## Publish the app

1. Put this project folder's contents in a GitHub repository. Ensure `app.py`, `requirements.txt`, `care_docs/`, and `.streamlit/config.toml` are at the repository root (or record the correct subdirectory for the deploy setup).
2. Push the project to a GitHub repository you control. Keep `.env`, `.streamlit/secrets.toml`, virtual environments, and user documents out of Git; `.gitignore` is configured for these.
3. In Streamlit Community Cloud, create a new app and select the repository, branch, and `app.py` as the entry point.
4. Deploy. Streamlit installs dependencies from `requirements.txt` and applies `.streamlit/config.toml`.
5. If you intentionally want optional provider-generated answers, add values to the app's Secrets settings as TOML, for example `OPENAI_API_KEY = "..."`, `OPENAI_BASE_URL = "https://api.openai.com/v1"`, and `OPENAI_MODEL = "gpt-4o-mini"`. Without the key, the app remains local-answer-only.
6. Verify the app's privacy notice, upload a harmless test document, check its citations, then use Clear session documents.

This workspace does not currently contain a dedicated GitHub remote for CareDocs, so publishing the hosted app requires choosing/creating its GitHub repository and deploying it through your Streamlit account.
