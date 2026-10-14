from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
import re
from typing import Iterable

import numpy as np
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAFETY_NOTICE = (
    "CareDocs provides educational information from documents you upload. It is not a doctor, "
    "cannot diagnose or recommend treatment, and does not replace a licensed clinician. "
    "For severe, sudden, or potentially life-threatening symptoms, contact local emergency services now."
)
NO_EVIDENCE = (
    "I couldn't find relevant information in the documents uploaded for this session. "
    "I can't answer this from the available sources. Please ask a qualified healthcare professional."
)
URGENT_NOTICE = (
    "Your message may describe an emergency. CareDocs cannot assess emergencies. "
    "Call your local emergency number or go to the nearest emergency department now. "
    "Do not wait for an answer from this app."
)
URGENT_PATTERNS = (
    r"chest pain", r"difficulty breathing", r"trouble breathing", r"can't breathe", r"cannot breathe",
    r"signs of stroke", r"face droop", r"sudden weakness", r"uncontrolled bleeding", r"severe bleeding",
    r"overdose", r"anaphylaxis", r"severe allergic reaction", r"passed out", r"loss of consciousness",
    r"suicidal", r"kill myself", r"want to die",
)


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    number: int


def _safe_filename(filename: str) -> str:
    # Keep only a display name, not any user-supplied directory path.
    return PurePosixPath(filename.replace("\\", "/")).name or "uploaded-document"


def extract_document(filename: str, content: bytes) -> str:
    """Extract text from a PDF, TXT, or Markdown upload without writing it to disk."""
    suffix = PurePosixPath(filename.lower()).suffix
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in {".txt", ".md", ".markdown"}:
        text = content.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("Unsupported document type. Upload a PDF, TXT, or Markdown file.")
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(cleaned) < 20:
        raise ValueError("No usable text was found. Scanned/image-only PDFs are not supported yet.")
    return cleaned


def chunk_documents(documents: Iterable[tuple[str, str]], chunk_size: int = 900, overlap: int = 140) -> list[Chunk]:
    """Split (source name, text) documents into labeled, overlapping chunks."""
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size >= 100 and overlap between 0 and chunk_size - 1.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=False,
    )
    result: list[Chunk] = []
    for source, text in documents:
        name = _safe_filename(source)
        pieces = splitter.split_text(text)
        result.extend(Chunk(text=piece, source=name, number=index + 1) for index, piece in enumerate(pieces) if piece.strip())
    return result


class TfidfRetriever:
    """In-memory sparse vector retrieval. Nothing is persisted to disk."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if chunks:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
            try:
                self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
            except ValueError:
                self.vectorizer = None
                self.matrix = None

    def search(self, query: str, top_k: int = 4, min_score: float = 0.025) -> list[tuple[Chunk, float]]:
        if not query.strip() or not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indexes = np.argsort(scores)[::-1][: max(1, top_k)]
        return [(self.chunks[int(index)], float(scores[index])) for index in indexes if scores[index] >= min_score]


def local_grounded_answer(query: str, retrieved: list[tuple[Chunk, float]]) -> str:
    """Return cited excerpts, never synthesize medical advice when not using an LLM."""
    if not retrieved:
        return NO_EVIDENCE
    lines = ["Here are the most relevant excerpts from your uploaded documents. I can't interpret them as a diagnosis:"]
    for chunk, _score in retrieved:
        lines.append(f"\n> {chunk.text.strip()}\n\n**Source:** {chunk.source} · chunk {chunk.number}")
    lines.append("\nIf this affects a medical decision, confirm the information with your healthcare professional.")
    return "\n".join(lines)


def urgent_safety_response(query: str) -> str | None:
    """Redirect common emergency phrases without retrieving documents or calling a model."""
    normalized = query.casefold()
    if any(re.search(pattern, normalized) for pattern in URGENT_PATTERNS):
        return URGENT_NOTICE
    return None


def _local_answer(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    """Compatibility alias used by the UI and tests."""
    del question
    return local_grounded_answer("", retrieved)


def generate_answer(
    question: str,
    retrieved: list[tuple[Chunk, float]],
    *,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    timeout: int = 20,
) -> tuple[str, bool]:
    """Generate an answer from retrieved passages, or use cited local excerpts.

    Returns (answer, external_provider_used). Only retrieved excerpts and the question are sent
    to the optional provider. No provider calls happen unless an API key is explicitly supplied.
    """
    urgent = urgent_safety_response(question)
    if urgent:
        return urgent, False
    if not retrieved:
        return NO_EVIDENCE, False
    if not api_key:
        return local_grounded_answer(question, retrieved), False

    sources = "\n\n".join(
        f"[Source: {chunk.source}, chunk {chunk.number}]\n{chunk.text}" for chunk, _score in retrieved
    )
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 500,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are CareDocs, an educational assistant that summarizes only the provided user-uploaded excerpts. "
                    "Do not diagnose, recommend starting/stopping/changing treatment or medication, or present unsupported claims. "
                    "If evidence is insufficient, say so. Cite sources exactly as [Source: filename, chunk N]; never invent citations. "
                    "Keep user privacy. Be clear that you are not a clinician."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nRetrieved excerpts:\n{sources}\n\nSafety note: {SAFETY_NOTICE}"},
        ],
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    answer = str(content).strip()
    if not answer:
        return local_grounded_answer(question, retrieved), False
    return answer, True
