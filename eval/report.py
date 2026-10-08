"""eval/results.json → docs/report.html (한 장짜리 성능 리포트) + docs/final_report.md의 <!-- AUTO:이름 --> 표 갱신.
숫자는 전부 results.json에서만 가져온다.

  uv run python eval/report.py
"""
import html
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "eval/results.json").read_text())
OUT = ROOT / "docs/report.html"
e = html.escape

STEP = {  # 단계마다: 노린 문제 / 바꾼 것 / 추가 비용
    "S1": ("500자 창이 정의·목록을 중간에서 잘라 '무엇에 대한 과태료인지' 같은 맥락이 사라짐",
           "조문 1개 = 청크 1개, 제2조만 정의 호 단위 + breadcrumb", "없음"),
    "S2": ("임베딩은 '제36조'·'국내대리인' 같은 고유명사·번호를 뭉개서 놓침",
           "BM25(공백 단어 + 글자 bigram) + Dense를 RRF(k=60)로 합침, 질문의 제N조는 맨 앞", "없음 (로컬 계산)"),
    "S3": ("'가게 챗봇'처럼 일상어는 법률용어('인공지능사업자', '고지')와 멀고, 쟁점이 여러 개면 하나만 찾음",
           "LLM이 법률용어 질의 1개 + Sub-query ≤3개로 Query Rewriting → 각각 검색 후 RRF", "질문당 LLM 1회"),
    "S4": ("후보 20개 안에는 정답이 있는데 순위가 뒤로 밀림",
           "Qwen3-Reranker-0.6B가 원 질문과 각 후보를 직접 비교해 20 → 5", "로컬 GPU 약 2초"),
    "C1": ("상위 5개 조문만 넣으면 '고영향 인공지능'의 정의(제2조)나 본문이 가리키는 조문이 빠짐",
           "질의에 나온 정의어의 정의 호 + 본문 속 제N조 참조를 1단계 첨부", "입력 토큰 증가"),
    "C2": ("정방향 참조는 'A가 가리키는 B'만 가져옴. 이 법은 제재(제40·42·43조)가 의무 조문을 가리켜 '어기면?'에서 제재가 빠짐",
           "제재·조사 조문 → 피참조 조문 역색인. 상위 조문을 가리키는 제재 조항을 해당 호만 첨부", "입력 토큰 소폭 증가"),
}


def pct(x):
    return "–" if x is None else f"{x * 100:.0f}%"


def delta(a, b, unit="%p"):
    if a is None or b is None:
        return ""
    d = (b - a) * 100
    cls = "up" if d > 0.5 else "down" if d < -0.5 else "flat"
    sign = "+" if d > 0 else ""
    return f'<span class="d {cls}">{sign}{d:.0f}{unit}</span>'


def bar(v, hi=False, label=None):
    w = 0 if v is None else max(v * 100, 1.5)
    return f'<div class="bar"><i class="{"hi" if hi else ""}" style="width:{w:.1f}%"></i><b>{label or pct(v)}</b></div>'


Q = R["per_question"]
INS = [q for q in Q if not q["oos"]]
STAGES = list(R["stages"])
CTX = list(R["context"])


def recall(q, key):
    return len(set(q["gold"]) & set(q[key][:5] if key.startswith("S") else q[key])) / len(q["gold"])


def flips(a, b):
    up = [q for q in INS if recall(q, b) > recall(q, a)]
    down = [q for q in INS if recall(q, b) < recall(q, a)]
    return up, down


def qchips(qs, cls):
    return "".join(f'<span class="qc {cls}" title="{e(q["question"])}">{q["id"]}</span>' for q in qs) or '<span class="none">없음</span>'


# ---------- 섹션: 검색 ----------
s = R["stages"]
first, last = s[STAGES[0]], s[STAGES[-1]]
stage_rows = "".join(
    f'<tr><th><span class="tag">{k}</span>{e(v["label"])}</th><td>{bar(v["hit3"], k == STAGES[-1])}</td>'
    f'<td>{bar(v["recall5"], k == STAGES[-1])}</td><td>{bar(v["mrr"], k == STAGES[-1], f"{v['mrr']:.2f}")}</td></tr>'
    for k, v in s.items())
# C1·C2: 검색(S4 상위 5개)은 그대로이고 Context에 조문을 더 붙인 것 → 순위 지표(Hit@3·MRR)는 없고 Recall만 있다.
# C0 = S4 상위 5개라 S4 Recall@5와 같아서 행을 따로 두지 않는다.
stage_rows += "".join(
    f'<tr class="ctx"><th><span class="tag">{k}</span>{e(R["context"][k]["label"])}</th><td class="na">–</td>'
    f'<td>{bar(R["context"][k]["recall"], k == CTX[-1])}</td><td class="na">–</td></tr>' for k in CTX[1:])
types = list(first["by_type"])
type_rows = "".join(
    f'<tr><th>{e(t)} <small>{R["golden"]["types"].get(t, "")}문항</small></th>'
    + "".join(f'<td class="num">{pct(s[k]["by_type"][t]["recall5"])}</td>' for k in STAGES)
    + "".join(f'<td class="num">{pct(R["context"][k]["by_type"].get(t))}</td>' for k in CTX[1:]) + "</tr>" for t in types)
# 유형별 값은 문항 수 가중 평균하면 위 표의 전체 값이 된다 (예: C2 98% = multi-hop 93%×9 + 나머지 100%×26)
type_rows += (f'<tr class="total"><th>전체 <small>{R["golden"]["in_scope"]}문항</small></th>'
              + "".join(f'<td class="num">{pct(s[k]["recall5"])}</td>' for k in STAGES)
              + "".join(f'<td class="num">{pct(R["context"][k]["recall"])}</td>' for k in CTX[1:]) + "</tr>")

# ---------- 섹션: 개선 단계 카드 ----------
cards = []
chain = [(STAGES[i - 1], STAGES[i]) for i in range(1, len(STAGES))] + [("C0", "C1"), ("C1", "C2")]
for a, b in chain:
    prob, change, cost = STEP[b]
    up, down = flips(a, b)
    if b.startswith("S"):
        m = (f'Hit@3 {pct(s[a]["hit3"])}→{pct(s[b]["hit3"])} {delta(s[a]["hit3"], s[b]["hit3"])} · '
             f'Recall@5 {pct(s[a]["recall5"])}→{pct(s[b]["recall5"])} {delta(s[a]["recall5"], s[b]["recall5"])} · '
             f'MRR {s[a]["mrr"]:.2f}→{s[b]["mrr"]:.2f}')
        label = s[b]["label"]
    else:
        c = R["context"]
        m = (f'Recall(Context 전체) {pct(c[a]["recall"])}→{pct(c[b]["recall"])} {delta(c[a]["recall"], c[b]["recall"])} · '
             f'정답 조문 전부 포함 {pct(c[a]["all_gold"])}→{pct(c[b]["all_gold"])} {delta(c[a]["all_gold"], c[b]["all_gold"])}')
        label = c[b]["label"]
    cards.append(f"""<article class="step"><header><span class="tag">{b}</span><h3>{e(label)}</h3></header>
<dl><dt>문제</dt><dd>{e(prob)}</dd><dt>바꾼 것</dt><dd>{e(change)}</dd><dt>추가 비용</dt><dd>{e(cost)}</dd></dl>
<p class="metric">{m}</p>
<p class="flip"><span>나아진 문항</span>{qchips(up, "up")}</p><p class="flip"><span>나빠진 문항</span>{qchips(down, "down")}</p></article>""")

# ---------- 섹션: 문항별 히트맵 ----------
cols = STAGES + ["C1", "C2"]
ans_on = R["answers"] is not None


def cell(v):
    cls = "full" if v == 1 else "part" if v > 0 else "zero"
    return f'<td class="hm {cls}" title="정답 조문 {v * 100:.0f}%"></td>'


def ans_cell(q):
    a = q.get("answer")
    if not a:
        return '<td class="hm na" title="채점 안 함"></td>'
    c = float(a["judge"]["correct"])
    cls = "full" if c == 1 else "part" if c > 0 else "zero"
    return f'<td class="hm {cls}" title="정답성 {c}"></td>'


heat = "".join(
    f'<tr><th title="{e(q["question"])}"><span class="qid">{q["id"]}</span><span class="qt">{e(q["question"])}</span></th>'
    f'<td class="ty">{e(q["type"])}</td>' + "".join(cell(recall(q, k)) for k in cols)
    + (ans_cell(q) if ans_on else "") + "</tr>" for q in INS)

# ---------- 섹션: 답변 ----------
base = R["baseline_20"]["answers"] or {}
NAMES = {"A": "LLM 단독", "B": "법 전문 통째로", "C": "Naive RAG (500자, dense top4)", "D_nogate": "Advanced RAG (역참조 없음)"}  # D(점수 차단)는 폐기한 방식
ans_rows = "".join(
    f'<tr><th>{e(NAMES[k])} <small>20문항 · 이전 실험</small></th><td>{bar(base[k]["accuracy"])}</td>'
    f'<td>{bar(base[k]["hallucination_rate"])}</td><td>{bar(base[k]["citation_valid_rate"])}</td>'
    f'<td class="num">{base[k]["avg_input_tokens"]:,}</td></tr>' for k in NAMES if k in base)
A = R["answers"]
if A:
    ans_rows += (f'<tr class="me"><th>이번 서비스 파이프라인 <small>{A["n"]}문항</small></th><td>{bar(A["accuracy"], True)}</td>'
                 f'<td>{bar(A["hallucination_rate"], True)}</td><td>{bar(A["citation_valid_rate"], True)}</td>'
                 f'<td class="num">{int(A["avg_input_tokens"]):,}</td></tr>')

examples = []
ex_ids = [i for i in ("g12", "g30", "g02") if any(x["id"] == i and x.get("answer") for x in Q)]
for i in ex_ids:
    q = next(x for x in Q if x["id"] == i)
    a = q["answer"]
    examples.append(f"""<article class="ex"><p class="q">{e(q["question"])}</p><p class="a">{e(a["text"])}</p>
<p class="src">근거 조문 {" ".join(f'<span class="tag">{e(x)}</span>' for x in a["sources"]) or "없음"} · 정답 조문 {", ".join("제" + g + "조" for g in q["gold"])}
 · 검색 상위 5 {" ".join(q["S4"])} · 역참조 {", ".join(q["attached"]["reverse_references"]) or "없음"}</p></article>""")

c0, c1, c2 = (R["context"][k] for k in CTX)
mh = {k: R["context"][k]["by_type"].get("multi-hop") for k in CTX}
kpis = [("상위 3위 안에 정답 조문", pct(first["hit3"]), pct(last["hit3"]), f"{STAGES[0]} → {STAGES[-1]}"),
        ("필요한 조문 확보율 (Recall@5)", pct(first["recall5"]), pct(last["recall5"]), f"{STAGES[0]} → {STAGES[-1]}"),
        ("LLM에 넘긴 Context의 Recall", pct(c0["recall"]), pct(c2["recall"]), "상위 5개만(C0) → 참조·역참조 추가(C2)"),
        ("multi-hop 문항 Recall", pct(mh["C0"]), pct(mh["C2"]), "C0 → C2 (9문항만)")]
if A:
    kpis.append(("답변 정확도", pct(base.get("A", {}).get("accuracy")), pct(A["accuracy"]), "LLM 단독 → 이번 서비스"))
kpi_html = "".join(f'<div class="kpi"><p>{e(t)}</p><strong><s>{a}</s> {b}</strong><small>{e(n)}</small></div>'
                   for t, a, b, n in kpis)
g = R["golden"]
credit = R.get("credits", {})

doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>성능 리포트 · AI 기본법 QA</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%233182f6'/%3E%3Ctext x='16' y='22' font-size='16' font-weight='700' text-anchor='middle' fill='white' font-family='sans-serif'%3E%C2%A7%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>
:root{{--g9:#191F28;--g7:#4E5968;--g5:#8B95A1;--g3:#D1D6DB;--g2:#E5E8EB;--g1:#F2F4F6;--blue:#3182F6;--blue-bg:#E8F3FF;--red:#F04452;--green:#03A94D}}
*{{box-sizing:border-box}}body{{margin:0;font-family:"Pretendard Variable",Pretendard,-apple-system,"Malgun Gothic",sans-serif;color:var(--g9);line-height:1.6;word-break:keep-all;-webkit-font-smoothing:antialiased}}
main{{max-width:960px;margin:0 auto;padding:48px 24px 96px}}
a{{color:var(--blue)}}
.top{{display:flex;justify-content:space-between;align-items:center;font-size:14px;color:var(--g5)}}
h1{{font-size:34px;line-height:1.3;margin:20px 0 8px;letter-spacing:-.02em}}
.lede{{font-size:18px;color:var(--g7);margin:0 0 28px}}
nav.qs{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:36px}}
nav.qs a{{display:block;padding:14px 16px;border-radius:14px;background:var(--g1);color:var(--g9);text-decoration:none;font-weight:600;font-size:15px}}
nav.qs a small{{display:block;color:var(--g5);font-weight:500;font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:56px}}
.kpi{{border:1px solid var(--g2);border-radius:16px;padding:18px}}
.kpi p{{margin:0;color:var(--g7);font-size:14px}}.kpi strong{{display:block;font-size:30px;letter-spacing:-.02em;color:var(--blue)}}
.kpi s{{color:var(--g5);font-size:18px;text-decoration:none;font-weight:500}}.kpi s::after{{content:" →";}}
.kpi small{{color:var(--g5);font-size:12px}}
section{{margin-top:64px}}
section>.eyebrow{{color:var(--blue);font-weight:700;font-size:14px;margin:0}}
section>h2{{font-size:26px;margin:4px 0 6px;letter-spacing:-.02em}}
section>.sub{{color:var(--g7);margin:0 0 20px}}
table{{width:100%;border-collapse:collapse}}
.bt th,.bt td{{padding:10px 8px;border-bottom:1px solid var(--g1);text-align:left;font-size:14px;vertical-align:middle}}
.bt thead th{{color:var(--g5);font-weight:600;font-size:13px}}
.bt tbody th{{font-weight:600;width:34%}}.bt th small{{display:block;color:var(--g5);font-weight:500}}
.bt tr.me th{{color:var(--blue)}}
.bar{{position:relative;height:22px;background:var(--g1);border-radius:6px;overflow:hidden}}
.bar i{{position:absolute;inset:0 auto 0 0;background:var(--g3);border-radius:6px}}.bar i.hi{{background:var(--blue)}}
.bar b{{position:absolute;left:8px;top:1px;font-size:13px}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.tag{{display:inline-block;font-size:12px;font-weight:700;color:var(--blue);background:var(--blue-bg);border-radius:6px;padding:1px 7px;margin-right:6px}}
.note{{color:var(--g5);font-size:13px;margin-top:10px}}
.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}
.step{{border:1px solid var(--g2);border-radius:16px;padding:18px}}
.step header{{display:flex;align-items:center}}.step h3{{font-size:16px;margin:0}}
.step dl{{margin:12px 0 0;font-size:14px}}.step dt{{color:var(--g5);font-size:12px;font-weight:600;margin-top:8px}}.step dd{{margin:0;color:var(--g7)}}
.metric{{font-size:14px;font-weight:600;margin:14px 0 8px;padding-top:12px;border-top:1px solid var(--g1)}}
.d{{font-size:12px;font-weight:700;padding:0 4px;border-radius:4px}}.d.up{{color:var(--blue)}}.d.down{{color:var(--red)}}.d.flat{{color:var(--g5)}}
.flip{{margin:4px 0;font-size:12px;display:flex;flex-wrap:wrap;gap:4px;align-items:center}}.flip>span:first-child{{color:var(--g5);width:72px;flex:none}}
.qc{{padding:1px 6px;border-radius:5px;font-weight:600;cursor:help}}.qc.up{{background:var(--blue-bg);color:var(--blue)}}.qc.down{{background:#FFEEEE;color:var(--red)}}.none{{color:var(--g3)}}
.hmwrap{{overflow-x:auto}}
.hmt th,.hmt td{{font-size:13px;padding:3px 4px}}
.hmt thead th{{color:var(--g5);font-weight:600;text-align:center}}
.hmt tbody th{{text-align:left;font-weight:500;max-width:420px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.qid{{font-weight:700;margin-right:8px}}.qt{{color:var(--g7)}}.ty{{color:var(--g5);white-space:nowrap}}
tr.total th,tr.total td{{font-weight:700;border-top:2px solid var(--g1,#e5e8eb)}}
td.na{{color:var(--g3,#8b95a1);text-align:center}}tr.ctx th{{border-top:2px dashed var(--g1,#e5e8eb)}}tr.ctx td{{border-top:2px dashed var(--g1,#e5e8eb)}}
.hm{{width:44px;height:22px;border-radius:5px;border:2px solid #fff}}.hm.full{{background:var(--blue)}}.hm.part{{background:#90C2FF}}.hm.zero{{background:var(--g1)}}.hm.na{{background:repeating-linear-gradient(45deg,#fff,#fff 3px,var(--g1) 3px,var(--g1) 6px)}}
.legend{{display:flex;gap:16px;font-size:13px;color:var(--g7);margin:10px 0}}.legend i{{display:inline-block;width:14px;height:14px;border-radius:4px;vertical-align:-2px;margin-right:4px}}
.ex{{border-left:3px solid var(--blue);padding:4px 0 4px 16px;margin:18px 0}}.ex .q{{font-weight:700;margin:0}}.ex .a{{margin:6px 0;color:var(--g7);white-space:pre-line}}.ex .src{{font-size:13px;color:var(--g5);margin:0}}
.method{{background:var(--g1);border-radius:16px;padding:20px 22px;font-size:14px;color:var(--g7)}}.method li{{margin:4px 0}}
@media (max-width:640px){{h1{{font-size:26px}}nav.qs{{grid-template-columns:1fr}}.bt tbody th{{width:40%}}}}
</style></head><body><main>
<div class="top"><span>AI 기본법 QA · 성능 리포트</span><a href="/">서비스로 돌아가기</a></div>
<h1>이 답변은 믿을 만한가,<br>숫자로 확인했어요</h1>
<p class="lede">Golden Test Set {g["count"]}문항(범위 안 {g["in_scope"]}개)으로 검색 단계를 하나씩 쌓으며 무엇이 실제로 나아졌는지 쟀어요.</p>
<nav class="qs"><a href="#q2">필요한 조문을 제대로 찾았나<small>검색 품질</small></a><a href="#q3">검색 품질을 어떻게 개선했나<small>단계별 변화</small></a><a href="#q1">답변은 어떤 조문에 근거했나<small>답변 품질</small></a></nav>
<div class="kpis">{kpi_html}</div>

<section id="q2"><p class="eyebrow">질문 1</p><h2>필요한 법률 조항을 제대로 찾았나</h2>
<p class="sub">단계를 하나씩 더할 때마다 정답 조문이 상위권에 얼마나 들어왔는지예요.</p>
<table class="bt"><thead><tr><th>단계</th><th>Hit@3</th><th>Recall@5 · C는 Context 전체</th><th>MRR</th></tr></thead><tbody>{stage_rows}</tbody></table>
<p class="note">Hit@3: 상위 3개 조문에 정답 조문이 하나라도 있는 비율 · Recall@5: 정답 조문 중 상위 5개에 든 비율 · MRR: 첫 정답 조문 순위의 역수 평균 · 범위 안 {g["in_scope"]}문항, 조 단위<br>C1·C2는 검색 순위는 S4 그대로 두고 LLM에 넘기는 Context에 조문을 더 붙인 단계라, 순위 지표(Hit@3·MRR)는 없고 Recall을 상위 5개 대신 Context 전체 기준으로 재었어요. C0(상위 5개만)는 S4와 같아요.</p>
<h3 style="margin-top:28px;font-size:17px">유형별 Recall</h3>
<table class="bt"><thead><tr><th>유형</th>{"".join(f'<th class="num">{k}</th>' for k in STAGES + CTX[1:])}</tr></thead><tbody>{type_rows}</tbody></table>
</section>

<section id="q3"><p class="eyebrow">질문 2</p><h2>검색 품질을 어떻게 개선했나</h2>
<p class="sub">단계마다 노린 문제, 바꾼 것, 그리고 그 결과 나아지거나 나빠진 문항이에요. 문항 번호에 마우스를 올리면 질문이 보여요.</p>
<div class="steps">{"".join(cards)}</div>
<h3 style="margin-top:36px;font-size:17px">문항별로 보기</h3>
<div class="legend"><span><i style="background:var(--blue)"></i>정답 조문 전부</span><span><i style="background:#90C2FF"></i>일부</span><span><i style="background:var(--g1)"></i>못 찾음</span>{'<span><i class="hm na" style="border:0"></i>채점 안 함</span>' if ans_on else ""}</div>
<div class="hmwrap"><table class="hmt"><thead><tr><th>문항</th><th>유형</th>{"".join(f"<th>{k}</th>" for k in cols)}{"<th>답변</th>" if ans_on else ""}</tr></thead><tbody>{heat}</tbody></table></div>
<p class="note">S0~S4 칸은 상위 5개 조문, C1·C2 칸은 답변에 실제로 넣은 Context 전체 기준이에요.</p>
</section>

<section id="q1"><p class="eyebrow">질문 3</p><h2>이 답변은 어떤 문서를 근거로 만들어졌나</h2>
<p class="sub">같은 모델(gpt-5.4-mini)에 근거를 주는 방식만 바꿔 답변을 채점했어요.</p>
<table class="bt"><thead><tr><th>방식</th><th>정확도</th><th>할루시네이션 비율</th><th>인용 정확</th><th class="num">질문당 입력 토큰</th></tr></thead><tbody>{ans_rows}</tbody></table>
<p class="note">할루시네이션 비율은 낮을수록 좋아요. 답변 채점은 이전 실험(Golden Test Set g01~g20)의 결과예요. 이번엔 크레딧을 아끼려고 답변을 다시 채점하지 않았고, 대신 검색과 Context를 40문항으로 재었어요. 채점 LLM 판정은 사람이 재검토해 고쳤어요.</p>
{"".join(examples)}
</section>

<section><h2 style="font-size:20px">측정 방법과 한계</h2><div class="method"><ul>
<li>대상: 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」 법률 제21311호(2026. 7. 21. 시행), 46개 조문. HWPX 원문을 직접 파싱해 법제처 Open API 본문과 전부 일치함을 확인했어요.</li>
<li>Golden Test Set: <code>{e(g["path"])}</code> {g["count"]}문항 ({", ".join(f"{k} {v}" for k, v in g["types"].items())}). 1문항이 {100 / max(g["in_scope"], 1):.1f}%p를 움직이니 경향으로 봐 주세요.</li>
<li>multi-hop: 답하려면 조문 여러 개를 이어 봐야 하는 질문이에요. 예를 들어 '국내대리인을 안 두면?'은 의무(제36조)와 과태료(제43조)를 함께 봐야 해요.</li>
<li>모델: Query Rewriting·답변·채점 gpt-5.4-mini(temperature 0) · 임베딩 text-embedding-3-small · Reranker Qwen3-Reranker-0.6B(로컬).</li>
<li>채점 LLM은 같은 계열이라 관대할 수 있어요. 판정은 사람이 다시 확인해 <code>eval/overrides.json</code>에 이유와 함께 남겨요.</li>
<li>시행령(대통령령 제36580호)은 아직 넣지 않았어요. 위임된 세부 기준은 답변에 '대통령령으로 정한다'고 표시해요.</li>
<li>생성일 {date.today().isoformat()} · 이번 평가 실행에 쓴 MonoRouter 크레딧 {credit.get("this_run_used", "–")}</li>
</ul></div></section>
</main></body></html>"""
OUT.write_text(doc)
print("wrote", OUT.relative_to(ROOT), len(doc), "bytes")


# ---------- final_report.md 자동 표 ----------
def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def ids(qs):
    return ", ".join(q["id"] for q in qs) or "없음"


auto = {
    "summary": "\n".join(f"- **{t}**: {a} → **{b}** ({n})" for t, a, b, n in kpis),
    "meta": (f"Golden Test Set `{g['path']}` {g['count']}문항 중 {g['evaluated']}문항 평가(범위 안 {g['in_scope']}문항) · "
             + " · ".join(f"{k} {v}" for k, v in g["types"].items()) + f" · 생성일 {date.today().isoformat()}"),
    "stages": md_table(["단계", "바꾼 것", "Hit@3", "Recall@5", "MRR", "나아진 문항", "나빠진 문항"],
                       [[k, v["label"], pct(v["hit3"]), pct(v["recall5"]), f'{v["mrr"]:.2f}',
                         *(map(ids, flips(STAGES[i - 1], k)) if i else ("–", "–"))]
                        for i, (k, v) in enumerate(s.items())]),
    "types": md_table(["유형", *STAGES, *CTX[1:]], [[f"{t} ({R['golden']['types'].get(t, '')})",
                                              *(pct(s[k]["by_type"][t]["recall5"]) for k in STAGES),
                                              *(pct(R["context"][k]["by_type"].get(t)) for k in CTX[1:])] for t in types]
                      + [[f"**전체 ({R['golden']['in_scope']})**", *(f'**{pct(s[k]["recall5"])}**' for k in STAGES),
                          *(f'**{pct(R["context"][k]["recall"])}**' for k in CTX[1:])]]),
    "context": md_table(["Context", "구성", "Recall(Context 전체)", "정답 조문 전부 포함", "multi-hop Recall", "나아진 문항"],
                        [[k, v["label"], pct(v["recall"]), pct(v["all_gold"]), pct(v["by_type"].get("multi-hop")),
                          ids(flips(CTX[i - 1], k)[0]) if i else "–"] for i, (k, v) in enumerate(R["context"].items())]),
    "answers": md_table(["방식", "문항", "정확도", "할루시네이션", "인용 정확", "질문당 입력 토큰"],
                        [[NAMES[k], "20 (이전 실험)", pct(base[k]["accuracy"]), pct(base[k]["hallucination_rate"]),
                          pct(base[k]["citation_valid_rate"]), f'{base[k]["avg_input_tokens"]:,}'] for k in NAMES if k in base]
                        + ([["이번 서비스 파이프라인", A["n"], pct(A["accuracy"]), pct(A["hallucination_rate"]),
                             pct(A["citation_valid_rate"]), f'{int(A["avg_input_tokens"]):,}']] if A else [])),
}
MD = ROOT / "docs/final_report.md"
if MD.exists():
    import re
    text = MD.read_text()
    for name, body in auto.items():
        # 블록 안은 비어 있거나 여러 줄 — 다음 /AUTO를 넘어 본문을 먹지 않도록 [^<]로 제한하지 않고 줄 단위로 매칭
        text = re.sub(rf"(<!-- AUTO:{name} -->\n)(?:(?!<!-- /AUTO -->).*\n)*(<!-- /AUTO -->)",
                      lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text)
    MD.write_text(text)
    print("updated", MD.relative_to(ROOT))
