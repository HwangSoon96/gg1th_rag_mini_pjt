"""Aggregate details.json (+ overrides.json, notes.json) into results.json."""
import json
from pathlib import Path
EXP = Path(__file__).parent
det = json.load(open(EXP / "details.json"))
ov = json.load(open(EXP / "overrides.json")) if (EXP / "overrides.json").exists() else []
meta = json.load(open(EXP / "meta.json"))
for o in ov:
    d = next(x for x in det if x["id"] == o["id"]); d["ans"][o["cond"]]["judge"].update(o["set"])

mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
ins = [d for d in det if not d["oos"]]

def stage(S, rows):
    return {"hit3": mean([any(a in d["ret"][S][:3] for a in d["gold"]) for d in rows]),
            "recall5": mean([len(set(d["gold"]) & set(d["ret"][S][:5])) / len(d["gold"]) for d in rows])}
types = sorted({d["type"] for d in ins})
by_stage = {S: {**stage(S, ins), "by_type": {t: stage(S, [d for d in ins if d["type"] == t]) for t in types}} for S in ["S0", "S1", "S2", "S3", "S4"]}

answers = {}
for c in ["A", "B", "C", "D", "D_nogate"]:
    J = [d["ans"][c]["judge"] for d in det]
    cv = [j["citations_valid"] for j in J if j["citations_valid"] in (True, False)]
    answers[c] = {"accuracy": mean([float(j["correct"]) for j in J]),
                  "accuracy_in_scope": mean([float(d["ans"][c]["judge"]["correct"]) for d in ins]),
                  "accuracy_by_type": {t: mean([float(d["ans"][c]["judge"]["correct"]) for d in det if d["type"] == t]) for t in sorted({d["type"] for d in det})},
                  "citation_valid_rate": mean(cv), "citations_judged": len(cv),
                  "hallucination_rate": mean([bool(j["hallucination"]) for j in J]),
                  "refusal_correct": f"{sum(d['ans'][c]['judge']['refused_correctly'] is True for d in det if d['oos'])}/{sum(d['oos'] for d in det)}",
                  "avg_input_tokens": round(mean([d["ans"][c]["tokens"] for d in det]))}
answers["D"]["extra_rewrite_tokens_per_q"] = round(mean([d["rewrite_tokens"] for d in det]))

res = {"golden_count": len(det), "in_scope_count": len(ins), "by_stage": by_stage, "answers": answers,
       "reranker_used": meta["reranker_used"], "notes": meta["notes"], "judge_overrides": ov, "oos_threshold_logit": 0.90625, "calibration": json.load(open(EXP / "calibration.json")),
       "per_question": [{"id": d["id"], "type": d["type"], "gold": d["gold"], **{S: d["ret"][S] for S in d["ret"]}, "rerank_max": round(d["rerank_max"], 2),
                         **{f"correct_{c}": d["ans"][c]["judge"]["correct"] for c in d["ans"]}} for d in det],
       "examples": [{"id": d["id"], "question": d["question"], "gold": d["gold"], "S0_top5": d["ret"]["S0"], "S4_top5": d["ret"]["S4"],
                     "C_answer": d["ans"]["C"]["text"][:200], "C_correct": d["ans"]["C"]["judge"]["correct"],
                     "D_answer": d["ans"]["D"]["text"][:200], "D_correct": d["ans"]["D"]["judge"]["correct"],
                     "D_nogate_answer": d["ans"]["D_nogate"]["text"][:200], "D_nogate_correct": d["ans"]["D_nogate"]["judge"]["correct"],
                     "B_correct": d["ans"]["B"]["judge"]["correct"], "rerank_max_logit": round(d["rerank_max"], 2)}
                    for i in meta["example_ids"] for d in det if d["id"] == i]}
json.dump(res, open(EXP / "results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ("by_stage", "answers")}, ensure_ascii=False, indent=1))
