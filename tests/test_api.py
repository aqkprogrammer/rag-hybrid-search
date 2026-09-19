import json

from fastapi.testclient import TestClient


def _sse(text: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    events = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_and_config(client: TestClient) -> None:
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok"
    assert body["documents"] == 9
    assert body["providers"]["llm"] == "mock:mock-extractive"
    assert body["bm25"]["chunks"] == body["chunks"]
    assert "x-request-id" in h.headers

    cfg = client.get("/api/config").json()
    assert "HR" in cfg["facets"]["department"]
    assert ".pdf" in cfg["supported_extensions"]


def test_index_page_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Hybrid RAG" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_document_lifecycle(client: TestClient) -> None:
    files = [
        (
            "files",
            (
                "notes/team.md",
                b"# Team\n\n## Lunch\n\nTeam lunch is every Friday at noon.",
                "text/markdown",
            ),
        ),
        ("files", ("bad.exe", b"MZ", "application/octet-stream")),
    ]
    r = client.post("/api/documents", files=files, data={"department": "Ops"})
    assert r.status_code == 201
    created, failed = r.json()
    assert created["status"] == "created" and created["source"] == "notes/team.md"
    assert failed["status"] == "failed"

    again = client.post("/api/documents", files=files[:1], data={"department": "Ops"}).json()
    assert again[0]["status"] == "unchanged"

    doc_id = created["doc_id"]
    detail = client.get(f"/api/documents/{doc_id}", params={"include_chunks": True}).json()
    assert detail["metadata"]["department"] == "Ops"
    assert detail["chunks"][0]["heading_path"] == ["Lunch"]
    assert any(d["doc_id"] == doc_id for d in client.get("/api/documents").json())

    hits = client.post(
        "/api/search", json={"query": "when is team lunch", "filters": {"department": "Ops"}}
    ).json()["hits"]
    assert hits[0]["doc_id"] == doc_id

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    assert client.get(f"/api/documents/{doc_id}").status_code == 404


def test_text_ingest(client: TestClient) -> None:
    r = client.post(
        "/api/documents/text",
        json={
            "title": "Parking",
            "text": "Parking spots are first come, first served.",
            "metadata": {"department": "Facilities"},
        },
    )
    assert r.status_code == 201
    assert r.json()["source"] == "Parking.md"


def test_search_debug(client: TestClient) -> None:
    r = client.post("/api/search", json={"query": "minimum password length", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hits"]) == 3
    assert body["hits"][0]["source"] == "security/information-security-policy.md"
    assert body["debug"]["mode"] == "hybrid"
    assert body["hits"][0]["stages"]["rerank_rank"] == 1


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/api/search", json={"query": ""}).status_code == 422
    r = client.post("/api/search", json={"query": "x", "filters": {"secret": "1"}})
    assert r.status_code == 422
    assert client.post("/api/query", json={"question": "x", "mode": "magic"}).status_code == 422


def test_query_grounded_answer(client: TestClient) -> None:
    r = client.post("/api/query", json={"question": "How long must a password be?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert "14 characters" in body["answer"]
    assert "[1]" in body["answer"]
    v = body["verification"]
    assert v["total_claims"] >= 1
    assert v["groundedness"] > 0.8
    assert v["claims"][0]["status"] == "supported"
    cited = [c for c in body["citations"] if c["cited"]]
    assert cited and cited[0]["source"] == "security/information-security-policy.md"
    assert body["retrieval"]["candidates"]


def test_query_refuses_out_of_scope(client: TestClient) -> None:
    body = client.post(
        "/api/query", json={"question": "What is the company's stock ticker?"}
    ).json()
    assert body["refused"] is True
    assert body["verification"] is None
    assert body["answer"].startswith("I don't know")


def test_query_with_filter_that_matches_nothing(client: TestClient) -> None:
    body = client.post(
        "/api/query", json={"question": "password length", "filters": {"department": "Nope"}}
    ).json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_query_stream(client: TestClient) -> None:
    with client.stream(
        "POST", "/api/query/stream", json={"question": "How long is paid parental leave?"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse(r.read().decode())
    names = [e for e, _ in events]
    assert names[0] == "retrieval"
    assert names[-2:] == ["verification", "done"]
    assert "token" in names
    answer = "".join(d["text"] for e, d in events if e == "token")
    verification = dict(events)["verification"]
    assert verification["answer"] == answer.strip()
    assert "20 weeks" in answer
    assert verification["verification"]["supported_claims"] >= 1
    assert dict(events)["done"]["refused"] is False


def test_verify_endpoint_flags_hallucinations(client: TestClient) -> None:
    hits = client.post("/api/search", json={"query": "minimum password length", "top_k": 1}).json()
    cid = hits["hits"][0]["chunk_id"]
    answer = (
        "Passwords must be at least 14 characters long [1]. "
        "Passwords must be at least 8 characters long [1]. "
        "Office plants are watered on Mondays [2]."
    )
    r = client.post("/api/verify", json={"answer": answer, "chunk_ids": [cid]})
    assert r.status_code == 200
    body = r.json()
    statuses = [cl["status"] for cl in body["verification"]["claims"]]
    assert statuses == ["supported", "unsupported", "unsupported"]
    assert body["verification"]["invalid_citations"] == [2]
    assert body["verified_answer"].startswith("Passwords must be at least 14 characters long [1].")
    assert "8 characters long." in body["verified_answer"]
    assert "[2]" not in body["verified_answer"]
    assert (
        client.post("/api/verify", json={"answer": "x [1]", "chunk_ids": ["nope"]}).status_code
        == 404
    )
