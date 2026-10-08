"""검색: Dense(Qdrant) + BM25 → RRF, 조문번호 부스트, 멀티쿼리.

단계별(S0~S3) 순위 함수를 그대로 노출해 eval이 같은 코드로 개선 폭을 잰다.
"""
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from rag import vectorstore as vs
from rag.chunker import chunks, naive_chunks
from rag.loader import load

RRF_K = 60


def tokenize(s: str) -> list[str]:
    """형태소 분석기 없이: 공백 단어 + 공백 제거 글자 bigram (한국어 조사·복합어 대응)."""
    z = re.sub(r"\s+", "", s)
    return [z[i:i + 2] for i in range(len(z) - 1)] + s.split()


def rrf(rank_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    score: dict[int, float] = {}
    for rl in rank_lists:
        for r, i in enumerate(rl):
            score[i] = score.get(i, 0.0) + 1 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])


def dedupe(seq):
    return list(dict.fromkeys(seq))


def cited_articles(q: str) -> list[str]:
    """질문에 명시된 '제N조(의M)' → 조문 번호."""
    return [a + (f"의{b}" if b else "") for a, b in re.findall(r"제(\d+)조(?:의(\d+))?", q)]


class Retriever:
    def __init__(self):
        self.arts = load()
        self.by = {a["no"]: a for a in self.arts}
        self.chunks = chunks(self.arts)
        self.chunk_no = [c["no"] for c in self.chunks]
        vs.ensure(vs.STRUCT, [c["text"] for c in self.chunks], [{"no": c["no"], "label": c["label"]} for c in self.chunks])
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._naive = None

    # ----- 순위 원재료 -----
    def dense(self, vec) -> list[int]:
        return vs.dense_rank(vs.STRUCT, vec, len(self.chunks))

    def sparse(self, q: str) -> list[int]:
        return [int(i) for i in np.argsort(-self.bm25.get_scores(tokenize(q)))]

    def boost(self, q: str, ranked: list[int]) -> list[int]:
        """질문에 '제N조'가 있으면 그 조문 청크를 맨 앞으로."""
        want = cited_articles(q)
        return [i for n in want for i in ranked if self.chunk_no[i] == n] + [i for i in ranked if self.chunk_no[i] not in want]

    def hybrid(self, q: str, vec) -> list[int]:
        return self.boost(q, rrf([self.dense(vec), self.sparse(q)]))

    def multi(self, q: str, queries: list[str], vecs) -> list[int]:
        """queries[i]마다 하이브리드 검색 → RRF. 원 질문의 조문번호 부스트 유지."""
        return self.boost(q, rrf([self.hybrid(x, v) for x, v in zip(queries, vecs)]))

    def to_articles(self, ranked: list[int]) -> list[str]:
        return dedupe(self.chunk_no[i] for i in ranked)

    # ----- 비교 기준선 S0: 500자 고정 분할 + dense -----
    def naive(self, vec) -> list[str]:
        if self._naive is None:
            self._naive = naive_chunks(self.arts)
            vs.ensure(vs.NAIVE, [c["text"] for c in self._naive], [{} for _ in self._naive])
        ranked = vs.dense_rank(vs.NAIVE, vec, len(self._naive))
        return dedupe(no for i in ranked for no in self._naive[i]["nos"])

    def naive_texts(self, vec, k: int = 4) -> list[str]:
        self.naive(vec)
        return [self._naive[i]["text"] for i in vs.dense_rank(vs.NAIVE, vec, k)]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
