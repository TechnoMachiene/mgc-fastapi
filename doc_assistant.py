"""
Grounded document assistant.

Primary path: calls your published n8n Agent ("MGC Sales Document
Assistant") through a webhook you expose from n8n. See README.md,
step "Connect the n8n agent", for the one-time setup.

Fallback path: if N8N_WEBHOOK_URL is unset, or the call fails, this
falls back to local TF-IDF retrieval over documents/*.txt, so the
page still answers questions before n8n is wired up.
"""
import glob
import os

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

N8N_WEBHOOK_URL = "https://abdultauseef.app.n8n.cloud/webhook/ask-mgc"

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CONFIDENCE_THRESHOLD = 0.12

_chunks, _sources, _vectorizer, _matrix = [], [], None, None


def _load_documents():
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            _chunks.append(para)
            _sources.append(name)
    global _vectorizer, _matrix
    if _chunks:
        _vectorizer = TfidfVectorizer(stop_words="english")
        _matrix = _vectorizer.fit_transform(_chunks)


_load_documents()


def _local_answer(question: str) -> dict:
    if not _chunks:
        return {"answer": "No documents loaded and N8N_WEBHOOK_URL not set.", "source": None}
    q_vec = _vectorizer.transform([question])
    sims = cosine_similarity(q_vec, _matrix)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    if best_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "Not in the MGC documents. Please confirm with the marketing manager.",
            "source": None,
            "confidence": round(best_score, 3),
        }
    return {"answer": _chunks[best_idx], "source": _sources[best_idx], "confidence": round(best_score, 3)}


async def answer_question(question: str) -> dict:
    if N8N_WEBHOOK_URL:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(N8N_WEBHOOK_URL, json={"question": question})
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                return {
                    "answer": data.get("answer") or data.get("output") or data.get("text") or str(data),
                    "source": data.get("source"),
                }
            return {"answer": str(data), "source": None}
        except Exception as e:
            fallback = _local_answer(question)
            fallback["answer"] = f"[n8n unreachable: {e}] " + fallback["answer"]
            return fallback

    return _local_answer(question)
