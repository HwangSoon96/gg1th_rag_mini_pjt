"""Pick the S4 out-of-scope threshold on held-out probe questions (NOT the golden set)."""
import json, sys; sys.path.insert(0, __file__.rsplit("/", 1)[0]); import run as r
P = {"in": ["인공지능 윤리원칙은 누가 정하고 어떤 내용이 들어가?", "AI 데이터센터를 지으면 정부 지원을 받을 수 있어?",
            "AI 스타트업 창업하면 나라에서 뭘 도와줘?", "인공지능집적단지 지정이 취소되는 경우는?"],
     "out": ["저작권법상 AI 학습 데이터 공정이용 기준은 뭐야?", "미국 AI 행정명령에는 어떤 내용이 있어?",
             "자율주행차 사고가 나면 보험 책임은 누구한테 있어?", "중국 생성형 AI 관리 규정 위반 벌금은 얼마야?"]}
out = {k: [r.s4_chunks(q, r.s3_chunks(q)[0])[1] for q in qs] for k, qs in P.items()}
thr = (min(out["in"]) + max(out["out"])) / 2
print(json.dumps({"probes": P, "max_scores": out, "threshold": thr}, ensure_ascii=False))
json.dump({"probes": P, "max_scores": out, "threshold_midpoint": thr}, open(r.EXP / "calibration.json", "w"), ensure_ascii=False, indent=1)
