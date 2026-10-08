"""Query Rewriting → Multi-Query Hybrid Retrieval → Reranking → Context Engineering(정의·참조·역참조) → Grounded Answer → 인용 파싱.

stream()은 단계별 이벤트를 내보내는 제너레이터 (UI의 SSE), ask()는 그 최종 결과 (POST /ask).
"""
import json
import re
import time

from common.ai_model import get_llm_model
from rag.loader import PARA_MARKS, load, units
from rag.reranker import rerank
from rag.retriever import get_retriever

TOP_K = 5             # Context에 원문 그대로 넣는 조문 수
RERANK_POOL = 20      # Reranker에 넣는 후보 청크 수
MAX_REFS = 6          # 정방향 참조 첨부 상한
MAX_DEFS = 4          # 정의 첨부 상한
ENFORCEMENT = ("40", "42", "43")  # 사실조사·벌칙·과태료: 의무 조문을 '가리키는' 쪽 → 역참조 대상
REFUSAL = "제공된 조문에서는 확인할 수 없습니다"

REWRITE_P = """너는 한국 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」 조문 검색을 위한 질의 재작성기다.
사용자 질문을 법률 조문 검색에 적합한 법률 용어 질의로 바꾸고, 질문에 쟁점이 여러 개면 하위 질의(최대 3개)로 나눠라.
JSON만 출력: {{"rewrite": "...", "subqueries": ["...", "..."]}}
질문: {q}"""

ANSWER_P = """당신은 한국 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」 질의응답 도우미입니다. 규칙:
1. 아래 [컨텍스트]의 조문 원문에 있는 내용만으로 답하세요. 컨텍스트에 없는 사실(다른 법률, 금액, 해외 제도 등)을 덧붙이지 마세요.
2. 각 주장 뒤에 근거를 대괄호로 인용하세요. 형식: [제34조①], 호까지는 [제43조①1], 항이 없는 조문의 호는 [제2조4].
3. 컨텍스트에서 답을 찾을 수 없으면 "제공된 조문에서는 확인할 수 없습니다"라고만 답하고 인용을 붙이지 마세요.
4. 세부 사항이 대통령령에 위임된 경우("대통령령으로 정한다/정하는") 그 사실을 표시하세요.
5. "노력하여야 한다"는 노력의무, "하여야 한다"는 의무로 구분해 표현하세요.
6. 한국어로 간결하게(5문장 이내) 답하세요.

[컨텍스트]
{ctx}

질문: {q}"""

_llm = {}


def llm(max_tokens: int):
    if max_tokens not in _llm:
        _llm[max_tokens] = get_llm_model(max_tokens=max_tokens)
    return _llm[max_tokens]


# ---------- 1. Query Rewriting ----------
def rewrite(q: str) -> dict:
    """{"rewrite", "subqueries"}. JSON이 깨지면 원 질문으로 대체 (검색은 계속)."""
    text = llm(300).invoke(REWRITE_P.format(q=q)).content
    try:
        rw = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        subs = [s for s in rw.get("subqueries", []) if isinstance(s, str) and s.strip()][:3]
        return {"rewrite": str(rw.get("rewrite") or q), "subqueries": subs}
    except (AttributeError, json.JSONDecodeError):
        return {"rewrite": q, "subqueries": []}


def queries(q: str, rw: dict) -> list[str]:
    return [q, rw["rewrite"], *rw["subqueries"]]


# ---------- 2·3. Retrieval + Reranking ----------
def retrieve(q: str, rw: dict, vecs: list) -> dict:
    """vecs = queries(q, rw)의 임베딩. Multi-Query Hybrid(S3) → Reranking(S4)."""
    r = get_retriever()
    s3 = r.multi(q, queries(q, rw), vecs)
    scored = rerank(q, s3[:RERANK_POOL], [c["text"] for c in r.chunks])
    s4 = r.boost(q, [i for i, _ in scored])
    return {"s3": s3, "s4": s4, "scores": dict(scored)}


# ---------- 4. Context Engineering ----------
_REF_RE = re.compile(r"제(\d+)조(?:의(\d+))?((?:제\d+항)?(?:(?:ㆍ|및|\s)*제\d+항)*)(?:제(\d+)호)?")


def strip_external(t: str) -> str:
    """「타법」 제N조…, 같은 법 제N조… 참조는 이 법 조문이 아니므로 제거."""
    return re.sub(r"(「[^」]*」|같은 법)\s*제\d+조(?:의\d+)?[^\s,]*", "", t)


def _unit(a: dict, para: int | None = None, item: str | None = None) -> dict | None:
    mark = PARA_MARKS[para - 1] if para else None
    for u in units(a):
        if u["para"] == mark and u["item"] == item:
            return u
    return None


def _terms(arts: list[dict]) -> dict[str, dict]:
    """정의어 → 정의한 단위. 제2조 각 호 + 본문의 '(이하 "X"라 한다)'(예: 제3조⑤ 인공지능취약계층)."""
    out = {}
    for a in arts:
        for u in a["items"] + a["paras"]:  # 호 먼저: 가장 구체적인 단위가 정의 출처
            m = re.match(r"\d+\.\s*“([^”]+)”", u["text"]) if a["no"] == "2" else None
            if m:
                out[m.group(1)] = u
            for t in re.findall(r"이하 “([^”]+)”(?:이)?라 한다", u["text"]):
                out.setdefault(t, u)
    out.pop("인공지능", None)  # 거의 모든 질문에 붙어 노이즈
    return {t: u for t, u in out.items() if len(t) >= 4}


def _reverse_index(by: dict) -> dict[str, list[dict]]:
    """피참조 조문 → 그것을 가리키는 제재·조사 조문의 단위 (가장 구체적인 단위 하나씩)."""
    idx: dict[str, list[dict]] = {}
    for no in ENFORCEMENT:
        a = by[no]
        seen = set()
        for u in a["items"] + a["paras"]:  # 호 먼저: 같은 참조면 더 구체적인 쪽만
            for m in _REF_RE.finditer(strip_external(u["text"])):
                ref = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
                if ref == no or ref not in by or (ref, u["para"]) in seen and u["item"] is None:
                    continue
                seen.add((ref, u["para"]))
                if u not in idx.setdefault(ref, []):
                    idx[ref].append(u)
    return idx


_cache: dict = {}


def _law():
    """조문·정의어·역참조 색인 (임베딩·DB 없이 법 본문만으로)."""
    if not _cache:
        arts = load()
        by = {a["no"]: a for a in arts}
        _cache.update(by=by, terms=_terms(arts), rev=_reverse_index(by))
    return _cache


def _unit_text(a: dict, u: dict) -> str:
    """호면 소속 항의 첫 줄을 앞에 붙여 맥락 유지 (예: '① …과태료를 부과한다.' + '1. 제31조제1항을 위반하여…')."""
    if u["item"] is None:
        return u["text"]
    para = next(p for p in a["paras"] if p["para"] == u["para"])
    return para["text"].split("\n")[0] + "\n" + u["text"]


def build_context(q: str, rw: dict, top: list[str], reverse: bool = True) -> tuple[str, dict]:
    """상위 조문 원문 + 정의 + 정방향 참조(1단계) + 역참조(제재·조사 조문)."""
    law = _law()
    by = law["by"]
    blocks = [f"[{by[n]['crumb']} > {by[n]['article']}({by[n]['title']})]\n{by[n]['text']}" for n in top]
    meta = {"articles": [by[n]["article"] for n in top], "definitions": [], "references": [], "reverse_references": [],
            "attached_articles": []}
    qside = " ".join(queries(q, rw))

    for t, u in law["terms"].items():
        no = u["no"]
        if t in qside and no not in top and len(meta["definitions"]) < MAX_DEFS:
            blocks.append(f"[정의 · {u['label']} “{t}”]\n{u['text']}")
            meta["definitions"].append(u["label"])
            meta["attached_articles"].append(no)

    seen = set()
    for n in top:
        for m in _REF_RE.finditer(strip_external(by[n]["text"])):
            ref = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
            if ref in top or ref == n or ref not in by:
                continue
            paras = [int(x) for x in re.findall(r"제(\d+)항", m.group(3))] or [None]
            for p in paras:
                u = _unit(by[ref], p, m.group(4)) or (_unit(by[ref], p) if p else None)
                key = u["label"] if u else by[ref]["article"]
                if key in seen or len(meta["references"]) >= MAX_REFS:
                    continue
                seen.add(key)
                text = _unit_text(by[ref], u) if u else by[ref]["text"]
                blocks.append(f"[참조 조문 · {key}({by[ref]['title']})]\n{text}")
                meta["references"].append(key)
                meta["attached_articles"].append(ref)

    if reverse:
        for n in top:
            for u in law["rev"].get(n, []):
                src = u["no"]
                if src in top or u["label"] in seen:
                    continue
                seen.add(u["label"])
                blocks.append(f"[역참조 · {u['label']}({by[src]['title']}) → {by[n]['article']}]\n{_unit_text(by[src], u)}")
                meta["reverse_references"].append(u["label"])
                meta["attached_articles"].append(src)
    return "\n\n".join(blocks), meta


# ---------- 5. 인용 → sources ----------
_CITE_RE = re.compile(r"제(\d+)조(?:의(\d+))?\s*(?:([①-⑳])|제(\d+)항)?\s*(?:제?(\d+(?:의\d+)?)호?)?")


def parse_sources(answer: str) -> list[dict]:
    """답변의 [제34조①2] 류 인용을 원문 단위로 해석. 같은 단위는 한 번만 (등장 순)."""
    by = _law()["by"]
    out: dict[str, dict] = {}
    for marker in re.findall(r"\[([^\]]*제\d+조[^\]]*)\]", answer):
        for part in re.split(r"[,·ㆍ/]|및", marker):
            m = _CITE_RE.search(part)
            if not m:
                continue
            no = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
            if no not in by:
                continue
            a = by[no]
            para = (PARA_MARKS.index(m.group(3)) + 1) if m.group(3) else (int(m.group(4)) if m.group(4) else None)
            u = _unit(a, para, m.group(5)) or (_unit(a, para) if para else None) or (_unit(a, None, m.group(5)) if m.group(5) else None)
            label = u["label"] if u else a["article"]
            src = out.setdefault(label, {"article": label, "title": a["title"],
                                         "content": _unit_text(a, u) if u else a["text"], "markers": []})
            if marker not in src["markers"]:
                src["markers"].append(marker)
    return list(out.values())


# ---------- 전체 ----------
def stream(q: str, rw: dict | None = None, vecs: list | None = None):
    """(event, data) 제너레이터: rewrite → retrieve → rerank → context → token* → done.

    rw·vecs를 주면 Query Rewriting·임베딩을 건너뛴다 (평가에서 캐시 재사용)."""
    from rag.vectorstore import embed_queries
    t0 = time.perf_counter()
    r = get_retriever()
    rw = rw or rewrite(q)
    yield "rewrite", rw
    res = retrieve(q, rw, vecs or embed_queries(queries(q, rw)))
    cand = r.to_articles(res["s3"][:RERANK_POOL])
    yield "retrieve", {"candidates": [{"article": r.by[n]["article"], "title": r.by[n]["title"]} for n in cand]}
    top = r.to_articles(res["s4"])[:TOP_K]
    best = {}
    for i, s in res["scores"].items():
        best[r.chunk_no[i]] = max(best.get(r.chunk_no[i], -1e9), s)
    yield "rerank", {"top": [{"article": r.by[n]["article"], "title": r.by[n]["title"], "score": round(best.get(n, 0.0), 3)}
                             for n in top]}
    ctx, meta = build_context(q, rw, top)
    prompt = ANSWER_P.format(ctx=ctx, q=q)
    yield "context", {k: v for k, v in meta.items() if k != "attached_articles"} | {"chars": len(prompt)}
    text, usage = "", {}
    for chunk in llm(1500).stream(prompt, stream_usage=True):
        if chunk.content:
            text += chunk.content
            yield "token", {"text": chunk.content}
        if chunk.usage_metadata:
            usage = dict(chunk.usage_metadata)
    sources = parse_sources(text)
    refused = REFUSAL in text and not sources
    yield "done", {"answer": text, "sources": sources, "refused": refused,
                   "tokens": usage.get("input_tokens"), "latency_ms": int((time.perf_counter() - t0) * 1000)}


def ask(q: str) -> dict:
    """POST /ask: {"answer", "sources": [{"article", "content", ...}]}"""
    for ev, data in stream(q):
        if ev == "done":
            return data
    raise RuntimeError("pipeline ended without answer")

