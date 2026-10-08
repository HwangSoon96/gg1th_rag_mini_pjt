"""청크 임베딩 → Qdrant 저장·검색.

컬렉션 내용이 (임베딩 모델 + 청크 텍스트) 지문과 같으면 재임베딩하지 않는다 (MonoRouter 호출 절약).
"""
import hashlib

from qdrant_client import models

from common.ai_model import get_embedding_model
from common.config import EMBEDDING_MODEL
from common.qdrant import get_qdrant_client

STRUCT = "law_articles"   # 구조 청크 (조문/정의 호)
NAIVE = "law_naive500"    # 비교 기준선 (500자 고정 분할)

_client = None
_emb = None


def client():
    global _client
    if _client is None:
        _client = get_qdrant_client()
    return _client


def embedder():
    global _emb
    if _emb is None:
        _emb = get_embedding_model()
    return _emb


def _fingerprint(texts: list[str]) -> str:
    return hashlib.sha256("\x00".join([EMBEDDING_MODEL, *texts]).encode()).hexdigest()[:16]


def ensure(name: str, texts: list[str], payloads: list[dict]) -> bool:
    """컬렉션을 texts로 맞춘다. 새로 임베딩했으면 True."""
    c, fp = client(), _fingerprint(texts)
    if c.collection_exists(name) and c.count(name).count == len(texts):
        head = c.retrieve(name, ids=[0], with_payload=["fp"])
        if head and head[0].payload.get("fp") == fp:
            return False
    vecs = embedder().embed_documents(texts)
    if c.collection_exists(name):
        c.delete_collection(name)
    c.create_collection(name, vectors_config=models.VectorParams(size=len(vecs[0]), distance=models.Distance.COSINE))
    c.upsert(name, points=[models.PointStruct(id=i, vector=v, payload={**p, "idx": i, "text": t, "fp": fp})
                           for i, (t, v, p) in enumerate(zip(texts, vecs, payloads))])
    return True


def embed_queries(qs: list[str]) -> list[list[float]]:
    """질의 여러 개를 요청 1회로 임베딩."""
    return embedder().embed_documents(qs)


def dense_rank(name: str, vec: list[float], limit: int) -> list[int]:
    """코사인 유사도 순 청크 인덱스."""
    res = client().query_points(name, query=vec, limit=limit, with_payload=["idx"])
    return [p.payload["idx"] for p in res.points]
