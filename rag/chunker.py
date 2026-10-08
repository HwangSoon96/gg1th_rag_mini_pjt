"""조문 → 검색 청크.

구조 청킹: 조문 1개 = 청크 1개. 제2조(정의, 약 1,900자)만 호 단위로 쪼개고 정의 머리말을 붙인다.
모든 청크 앞에 [장 > 절 > 제N조(제목)] breadcrumb.
naive_chunks()는 비교 기준선(S0)용 500자 고정 분할.
"""
from rag.loader import load

DEF_HEAD = "이 법에서 사용하는 용어의 뜻은 다음과 같다."


def head(a: dict) -> str:
    return f"[{a['crumb']} > {a['article']}({a['title']})]"


def chunks(arts: list[dict] | None = None) -> list[dict]:
    """[{"text", "no", "label"}] — label은 청크가 대표하는 조·호."""
    out = []
    for a in arts or load():
        if a["no"] == "2":
            for it in a["items"]:
                out.append({"text": f"{head(a)} {DEF_HEAD}\n{it['text']}", "no": "2", "label": it["label"]})
        else:
            out.append({"text": f"{head(a)}\n{a['text']}", "no": a["no"], "label": a["article"]})
    return out


def naive_chunks(arts: list[dict] | None = None, size: int = 500, overlap: int = 50) -> list[dict]:
    """법 전문을 이어 붙여 size자 창으로 자른다. 각 청크가 걸친 조문 번호를 nos에 기록."""
    flat, spans, chapter = "", [], None
    for a in arts or load():
        if a["crumb"] != chapter:
            flat += a["crumb"] + "\n\n"
            chapter = a["crumb"]
        spans.append((len(flat), len(flat) + len(a["text"]), a["no"]))
        flat += a["text"] + "\n\n"
    out, step = [], size - overlap
    for s in range(0, len(flat), step):
        e = min(s + size, len(flat))
        out.append({"text": flat[s:e], "nos": [no for a, b, no in spans if a < e and b > s]})
        if e == len(flat):
            break
    return out
