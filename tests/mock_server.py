"""UI 확인용: /ask/stream을 고정 이벤트로 바꿔 띄운다 (LLM·임베딩 호출 없음).
uv run python tests/mock_server.py → http://127.0.0.1:8001"""
import time

import uvicorn

import app.main as m
from rag.pipeline import parse_sources

ANSWER = ("국내대리인을 지정하지 않으면 3천만원 이하의 과태료 대상이에요[제43조①2]. 국내대리인은 안전성 확보 조치 이행 결과 제출, "
          "고영향 인공지능 해당 여부 확인 요청, 사업자 책무 이행 지원을 대리해요[제36조①]. 지정 기준(이용자 수·매출액 등)은 "
          "대통령령으로 정해요[제36조①].")


def fake_stream(q, rw=None, vecs=None):
    if "환불" in q:
        yield "rewrite", {"rewrite": q, "subqueries": []}
        yield "retrieve", {"candidates": [{"article": "제31조", "title": "인공지능 투명성 확보 의무"}]}
        yield "rerank", {"top": [{"article": "제31조", "title": "인공지능 투명성 확보 의무", "score": -6.2}]}
        yield "context", {"articles": ["제31조"], "definitions": [], "references": [], "reverse_references": [], "chars": 900}
        yield "token", {"text": "제공된 조문에서는 확인할 수 없습니다."}
        yield "done", {"answer": "제공된 조문에서는 확인할 수 없습니다.", "sources": [], "refused": True, "tokens": 700, "latency_ms": 3100}
        return
    yield "rewrite", {"rewrite": "국외 인공지능사업자의 국내대리인 지정 의무와 미지정 시 제재",
                      "subqueries": ["국내대리인 지정 의무", "국내대리인 미지정 과태료"]}
    time.sleep(0.4)
    yield "retrieve", {"candidates": [{"article": a, "title": t} for a, t in
                                      [("제36조", "국내대리인 지정"), ("제43조", "과태료"), ("제4조", "적용범위"), ("제32조", "인공지능 안전성 확보 의무")]]}
    time.sleep(0.4)
    yield "rerank", {"top": [{"article": "제36조", "title": "국내대리인 지정", "score": 7.1},
                             {"article": "제43조", "title": "과태료", "score": 4.3},
                             {"article": "제4조", "title": "적용범위", "score": 0.8}]}
    yield "context", {"articles": ["제36조", "제43조", "제4조"], "definitions": [], "references": ["제32조②", "제33조①"],
                      "reverse_references": [], "chars": 4592}
    for i in range(0, len(ANSWER), 6):
        time.sleep(0.02)
        yield "token", {"text": ANSWER[i:i + 6]}
    yield "done", {"answer": ANSWER, "sources": parse_sources(ANSWER), "refused": False, "tokens": 3120, "latency_ms": 8200}


m.run_stream = fake_stream

if __name__ == "__main__":
    uvicorn.run(m.app, host="127.0.0.1", port=8001)
