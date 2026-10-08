"""Golden Test Set(docs/eval_set/golden_set.jsonl)로 검색 단계별(S0~S4)·Context(C0~C2)·답변 품질을 잰다.

  uv run python eval/evaluate.py                  # 검색 지표만 (질문당 LLM 1회: Query Rewriting)
  uv run python eval/evaluate.py --answers        # + 답변 생성·LLM 채점 (질문당 LLM 2회 추가)
  uv run python eval/evaluate.py --answers --ids g21 g22

MonoRouter 월 크레딧 보호: 시작 전·질문마다 잔여 크레딧을 확인하고 --budget(이번 실행 사용 상한)을 넘으면 멈춘다.
결과는 질문마다 eval/details.json에 저장 → 중단 후 다시 실행하면 이어서 (같은 질문은 재호출 안 함).
채점 LLM 오판 수정은 eval/overrides.json [{id, cond:"D2", set:{...}, why}] → 집계에 반영.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.credits import credits
from rag import pipeline as P
from rag.retriever import get_retriever
from rag.vectorstore import embed_queries

GOLDEN = ROOT / "docs/eval_set/golden_set.jsonl"
DET = ROOT / "eval/details.json"
RES = ROOT / "eval/results.json"
OV = ROOT / "eval/overrides.json"
OLD = ROOT / "docs/experiment/results.json"
STAGES = {"S0": "500자 고정 분할 + Dense", "S1": "+ 조문 단위 Chunking", "S2": "+ Hybrid(BM25·RRF)",
          "S3": "+ Query Rewriting·Multi-Query", "S4": "+ Reranking"}
CONTEXTS = {"C0": "상위 5개 조문만", "C1": "+ 정의·정방향 참조", "C2": "+ 역참조(제재·조사 조문)"}

JUDGE_P = """당신은 법률 QA 채점자입니다. 한국 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」에 관한 질문에 대한 모델 답변을 채점하세요.
[질문] {q}
[범위 밖 질문 여부] {oos}
[정답 요지] {ref}
[정답 근거 조문] {gold}
[관련 조문 원문 (정답 근거 + 답변이 인용한 조문)]
{texts}
[모델 답변]
{ans}

채점 기준:
- correct: 1 = 정답 요지의 핵심을 모두 맞게 담음, 0.5 = 일부만 맞거나 핵심 일부 누락/부정확, 0 = 틀림·핵심 누락·(범위 내 질문인데) 답변 거부.
  범위 밖 질문은 "이 법에는 해당 내용이 없다/답할 수 없다"는 취지를 분명히 하면 1, 이 법의 내용인 것처럼 답하면 0.
- citations_valid: 답변이 조문 번호를 인용하지 않았으면 "na". 인용했다면 인용한 조문(항·호)이 원문상 실제로 그 주장을 뒷받침하면 true, 하나라도 틀린 조문을 인용했으면 false.
- hallucination: 위 조문 원문이나 정답 요지와 모순되거나 원문에 없는 내용을 이 법의 내용처럼 단정하면 true (예: 없는 금액·의무·조문 번호, 노력의무를 강행의무로 단정).
- refused_correctly: 범위 밖 질문이면 이 법의 범위 밖임을 밝혔는지 true/false, 범위 내 질문이면 "na".
JSON만 출력: {{"correct": 0|0.5|1, "citations_valid": true|false|"na", "hallucination": true|false, "refused_correctly": true|false|"na", "reason": "한 문장"}}"""


def golden() -> list[dict]:
    return [json.loads(x) for x in GOLDEN.read_text().splitlines() if x.strip()]


def metrics(gold: list[str], ranked: list[str]) -> dict:
    """Hit@3, Recall@5, MRR (조 단위)."""
    first = next((i for i, n in enumerate(ranked) if n in gold), None)
    return {"hit3": float(any(n in gold for n in ranked[:3])),
            "recall5": len(set(gold) & set(ranked[:5])) / len(gold),
            "mrr": 1 / (first + 1) if first is not None else 0.0}


def retrieval(g: dict) -> dict:
    r, q = get_retriever(), g["question"]
    rw = P.rewrite(q)
    qs = P.queries(q, rw)
    vecs = embed_queries(qs)
    res = P.retrieve(q, rw, vecs)
    ranks = {"S0": r.naive(vecs[0]), "S1": r.to_articles(r.dense(vecs[0])), "S2": r.to_articles(r.hybrid(q, vecs[0])),
             "S3": r.to_articles(res["s3"]), "S4": r.to_articles(res["s4"])}
    top = ranks["S4"][:P.TOP_K]
    _, m1 = P.build_context(q, rw, top, reverse=False)
    _, m2 = P.build_context(q, rw, top)
    ctx = {"C0": top, "C1": list(dict.fromkeys(top + m1["attached_articles"])),
           "C2": list(dict.fromkeys(top + m2["attached_articles"]))}
    best = {}
    for i, s in res["scores"].items():
        best[r.chunk_no[i]] = round(max(best.get(r.chunk_no[i], -1e9), s), 3)
    return {"q": q, "rewrite": rw, "ranks": {k: v[:10] for k, v in ranks.items()}, "ctx": ctx,
            "attached": {k: v for k, v in m2.items() if k != "attached_articles"}, "rerank": best}


def judge(g: dict, ans: str) -> dict:
    by = P._law()["by"]
    cited = {a + (f"의{b}" if b else "") for a, b in re.findall(r"제(\d+)조(?:의(\d+))?", ans)}
    nos = list(dict.fromkeys([n for n in g["gold_articles"] if n in by] + sorted(c for c in cited if c in by)))
    texts = "\n\n".join(f"[{by[n]['article']}({by[n]['title']})]\n{by[n]['text']}" for n in nos) or "(없음)"
    p = JUDGE_P.format(q=g["question"], oos=g["out_of_scope"], ref=g["reference_answer"],
                       gold=g["gold_articles"] or "없음", texts=texts, ans=ans)
    return json.loads(re.search(r"\{.*\}", P.llm(600).invoke(p).content, re.DOTALL).group(0))


def answer(g: dict, d: dict) -> dict:
    """서비스와 같은 stream()으로 답변 (Query Rewriting·임베딩은 검색 단계 캐시 재사용)."""
    q, rw = g["question"], d["rewrite"]
    out = next(data for ev, data in P.stream(q, rw, embed_queries(P.queries(q, rw))) if ev == "done")
    return {"text": out["answer"], "sources": [s["article"] for s in out["sources"]], "refused": out["refused"],
            "input_tokens": out["tokens"], "judge": judge(g, out["answer"])}


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def aggregate(G: list[dict], det: dict) -> dict:
    ov = json.loads(OV.read_text()) if OV.exists() else []
    for o in ov:
        a = det.get(o["id"], {}).get("ans")
        if a and o.get("cond", "D2") == "D2":
            a["judge"].update(o["set"])
    rows = [(g, det[g["id"]]) for g in G if g["id"] in det]
    ins = [(g, d) for g, d in rows if not g["out_of_scope"]]

    def stage_stats(sub, key, src):
        ms = [metrics(g["gold_articles"], d[src][key]) for g, d in sub]
        return {m: mean(x[m] for x in ms) for m in ("hit3", "recall5", "mrr")}

    types = sorted({g["type"] for g, _ in ins})
    sets = sorted({g.get("set", "") for g, _ in ins})
    by_stage = {s: {"label": STAGES[s], **stage_stats(ins, s, "ranks"),
                    "by_type": {t: stage_stats([x for x in ins if x[0]["type"] == t], s, "ranks") for t in types},
                    "by_set": {t: stage_stats([x for x in ins if x[0].get("set") == t], s, "ranks") for t in sets}}
                for s in STAGES}

    def ctx_recall(sub, c):
        return mean(len(set(g["gold_articles"]) & set(d["ctx"][c])) / len(g["gold_articles"]) for g, d in sub)

    def ctx_full(sub, c):
        return mean(float(set(g["gold_articles"]) <= set(d["ctx"][c])) for g, d in sub)

    context = {c: {"label": CONTEXTS[c], "recall": ctx_recall(ins, c), "all_gold": ctx_full(ins, c),
                   "by_type": {t: ctx_recall([x for x in ins if x[0]["type"] == t], c) for t in types}}
               for c in CONTEXTS}

    answered = [(g, d) for g, d in rows if "ans" in d]
    answers = None
    if answered:
        J = [(g, d["ans"]["judge"]) for g, d in answered]
        cv = [j["citations_valid"] for _, j in J if j["citations_valid"] in (True, False)]
        oos = [j for g, j in J if g["out_of_scope"]]
        answers = {"n": len(J), "accuracy": mean(float(j["correct"]) for _, j in J),
                   "accuracy_by_type": {t: mean(float(j["correct"]) for g, j in J if g["type"] == t)
                                        for t in sorted({g["type"] for g, _ in J})},
                   "accuracy_by_set": {s: mean(float(j["correct"]) for g, j in J if g.get("set") == s)
                                       for s in sorted({g.get("set", "") for g, _ in J})},
                   "hallucination_rate": mean(float(j["hallucination"] is True) for _, j in J),
                   "citation_valid_rate": mean(float(x) for x in cv), "citations_judged": len(cv),
                   "refusal_correct": f"{sum(j['refused_correctly'] is True for j in oos)}/{len(oos)}",
                   "avg_input_tokens": mean(d["ans"]["input_tokens"] or 0 for _, d in answered)}

    per_q = []
    for g, d in rows:
        row = {"id": g["id"], "set": g.get("set"), "type": g["type"], "question": g["question"],
               "gold": g["gold_articles"], "oos": g["out_of_scope"], "rewrite": d["rewrite"],
               **{s: d["ranks"][s][:5] for s in STAGES}, **{c: d["ctx"][c] for c in CONTEXTS},
               "attached": d["attached"], "rerank": d["rerank"]}
        if "ans" in d:
            row["answer"] = {k: d["ans"][k] for k in ("text", "sources", "refused", "input_tokens")} | {
                "judge": d["ans"]["judge"]}
        per_q.append(row)

    old = json.loads(OLD.read_text()) if OLD.exists() else {}
    return {"golden": {"path": str(GOLDEN.relative_to(ROOT)), "count": len(G), "evaluated": len(rows),
                       "in_scope": len(ins), "types": {t: sum(g["type"] == t for g, _ in rows) for t in
                                                       sorted({g["type"] for g, _ in rows})}},
            "stages": by_stage, "context": context, "answers": answers,
            "baseline_20": {"note": "2026-10-08 실험(docs/experiment, base20 20문항): A LLM 단독 · B 법 전문 · C Naive RAG · "
                                    "D Advanced RAG(역참조 없음). 같은 모델(gpt-5.4-mini)·같은 채점 루브릭",
                            "answers": old.get("answers"), "stages": old.get("by_stage")},
            "overrides": ov, "per_question": per_q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="답변 생성·채점까지")
    ap.add_argument("--ids", nargs="*", help="이 문항만")
    ap.add_argument("--budget", type=int, default=1500, help="이번 실행에서 쓸 MonoRouter 크레딧 상한")
    a = ap.parse_args()

    G = golden()
    det = json.loads(DET.read_text()) if DET.exists() else {}
    start = credits()["remaining_credits"]
    print(f"남은 크레딧 {start:,} · 이번 실행 상한 {a.budget:,}", flush=True)
    for g in G:
        if a.ids and g["id"] not in a.ids:
            continue
        d = det.get(g["id"])
        need_ret = not d or d["q"] != g["question"]
        need_ans = a.answers and (need_ret or "ans" not in d)
        if not (need_ret or need_ans):
            continue
        used = start - credits()["remaining_credits"]
        if used >= a.budget:
            print(f"크레딧 상한 도달 (사용 {used:,}) → 중단. 다시 실행하면 이어서 진행", flush=True)
            break
        for attempt in range(5):  # 같은 키를 다른 곳(노트북·팀원)이 쓰면 429가 길게 이어질 수 있다 → 문항 단위로 재시도
            try:
                if need_ret:
                    d = det[g["id"]] = retrieval(g)
                    need_ret = False
                if need_ans:
                    d["ans"] = answer(g, d)
                break
            except Exception as ex:
                if "429" not in str(ex) or attempt == 4:
                    raise
                print(f"{g['id']} 429 지속 → 90초 대기 후 재시도 ({attempt + 1}/4)", flush=True)
                time.sleep(90)
        DET.write_text(json.dumps(det, ensure_ascii=False, indent=1))
        j = d.get("ans", {}).get("judge", {})
        print(g["id"], "S4", d["ranks"]["S4"][:5], "gold", g["gold_articles"], "C2", d["ctx"]["C2"][:8],
              "correct", j.get("correct", "-"), flush=True)
    end = credits()["remaining_credits"]
    res = aggregate(G, det)
    res["credits"] = {"this_run_used": start - end, "remaining": end}
    RES.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({"stages": {s: {k: v[k] for k in ("hit3", "recall5", "mrr")} for s, v in res["stages"].items()},
                      "context": {c: v["recall"] for c, v in res["context"].items()}, "answers": res["answers"]},
                     ensure_ascii=False, indent=1))
    print(f"이번 실행 사용 크레딧 {start - end:,} · 남은 크레딧 {end:,}")
    import runpy  # docs/report.html·docs/final_report.md 표 갱신
    runpy.run_path(str(ROOT / "eval/report.py"))


if __name__ == "__main__":
    main()
