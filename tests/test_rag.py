from care_docs.rag import (
    NO_EVIDENCE,
    Chunk,
    TfidfRetriever,
    chunk_documents,
    extract_document,
    generate_answer,
    local_grounded_answer,
    urgent_safety_response,
)


def test_text_document_extract_and_chunk_have_citations():
    text = ("Medication list and follow-up plan. Contact the clinic if symptoms change. " * 30).encode()
    extracted = extract_document("visit.txt", text)
    chunks = chunk_documents([("visit.txt", extracted)], chunk_size=150, overlap=20)
    assert chunks
    assert all(chunk.source == "visit.txt" for chunk in chunks)
    retrieved = TfidfRetriever(chunks).search("follow-up plan clinic", top_k=2)
    answer = local_grounded_answer("follow-up", retrieved)
    assert "visit.txt" in answer
    assert "chunk" in answer


def test_unsupported_and_empty_documents_rejected():
    try:
        extract_document("image.png", b"binary")
        assert False, "expected unsupported file to fail"
    except ValueError as exc:
        assert "Unsupported" in str(exc)

    try:
        extract_document("blank.txt", b"  \n")
        assert False, "expected empty text to fail"
    except ValueError as exc:
        assert "No usable text" in str(exc)


def test_no_evidence_declines_and_does_not_call_external_provider(monkeypatch):
    def fail_request(*args, **kwargs):
        raise AssertionError("provider must not be called without retrieved evidence")

    monkeypatch.setattr("care_docs.rag.requests.post", fail_request)
    answer, external = generate_answer("What should I take?", [], api_key="test")
    assert answer == NO_EVIDENCE
    assert external is False


def test_local_answer_is_cited_source_extract():
    retrieved = [(Chunk("The document says schedule a follow-up visit.", "notes.md", 3), 0.5)]
    answer, external = generate_answer("What follow-up?", retrieved)
    assert "schedule a follow-up visit" in answer
    assert "[Source:" not in answer or "notes.md" in answer
    assert "notes.md" in answer
    assert external is False


def test_filenames_are_path_stripped():
    chunks = chunk_documents([("C:\\private\\record.txt", "This is enough document text for a supported passage.")])
    assert chunks[0].source == "record.txt"


def test_emergency_message_short_circuits_external_model(monkeypatch):
    def fail_request(*args, **kwargs):
        raise AssertionError("emergency questions must not call an external model")

    monkeypatch.setattr("care_docs.rag.requests.post", fail_request)
    answer, external = generate_answer("I have severe chest pain and trouble breathing", [], api_key="test")
    assert "emergency" in answer.lower()
    assert "Do not wait" in answer
    assert external is False
    assert urgent_safety_response("I have a mild cough") is None
