"""Cross-encoder 리랭킹: Qwen3-Reranker-0.6B로 원 질문과 후보 청크 쌍을 직접 채점 (로컬 GPU/CPU)."""
import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 캐시된 모델만 사용 (없으면 HF_HUB_OFFLINE=0으로 1회 다운로드)

MODEL = "Qwen/Qwen3-Reranker-0.6B"
PROMPT = "Given a Korean question about Korea's AI Basic Act, judge whether this law article passage helps answer the question"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL)


def rerank(q: str, cand: list[int], texts: list[str]) -> list[tuple[int, float]]:
    """후보 청크 인덱스 → (인덱스, logit) 점수 내림차순."""
    if not cand:
        return []
    scores = _model().predict([(q, texts[i]) for i in cand], prompt=PROMPT)
    return sorted(zip(cand, map(float, scores)), key=lambda x: -x[1])
