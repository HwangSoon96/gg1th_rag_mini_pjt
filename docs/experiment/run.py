"""RAG staircase experiment on 인공지능기본법. Run: cd repo && uv run python exp/run.py
Writes details.json (per-question) and results.json (aggregates). overrides.json (manual judge fixes) is applied if present."""
import json, math, os, re, sys
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
EXP = Path(__file__).parent
sys.path.insert(0, "/home/aiuser/rag_agent_ex"); sys.path.insert(0, str(EXP))
import numpy as np, tiktoken
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common_config import llm_connect, get_embeddings
import law

ARTS = law.load(); BY = {a["no"]: a for a in ARTS}
G = json.load(open(EXP / "golden.json"))
ENC = tiktoken.get_encoding("cl100k_base"); ntok = lambda s: len(ENC.encode(s))
EMB = get_embeddings()
LLM = llm_connect(max_tokens=1500); JUDGE = llm_connect(max_tokens=600)
# MonoRouter 한도(30회/분, 초과 시 retry-after 60s)를 LLM+임베딩 합산으로 넘지 않게 클라이언트에서 미리 조절
from langchain_core.rate_limiters import InMemoryRateLimiter
RL = InMemoryRateLimiter(requests_per_second=0.4, check_every_n_seconds=0.1, max_bucket_size=1)  # 24회/분
LLM.rate_limiter = RL; JUDGE.rate_limiter = RL  # 캐시 히트에는 적용 안 됨
for _m in ("embed_documents", "embed_query"):
    _f = getattr(EMB.underlying_embeddings, _m)
    object.__setattr__(EMB.underlying_embeddings, _m, (lambda f: lambda *a, **k: (RL.acquire(), f(*a, **k))[1])(_f))
RERANK_THRESHOLD = 0.90625  # reranker logit; midpoint of held-out probe questions (calibrate.py), not tuned on golden set

def head(a): return f"[{a['crumb']} > 제{a['no']}조({a['title']})]"
def art_block(a): return f"{head(a)}\n{a['text']}"
def unit(v): v = np.array(v); return v / np.linalg.norm(v, axis=-1, keepdims=True)
def jparse(s): return json.loads(re.search(r"\{.*\}", s, re.S).group(0))
def dedupe(seq): return list(dict.fromkeys(x for xs in seq for x in xs))
def rrf(rank_lists, k=60):
    sc = {}
    for rl in rank_lists:
        for r, i in enumerate(rl): sc[i] = sc.get(i, 0) + 1 / (k + r + 1)
    return sorted(sc, key=lambda i: -sc[i])

# ---------- S0: naive flat chunks ----------
flat, spans, ch = "", [], None
for a in ARTS:
    if a["crumb"] != ch: flat += a["crumb"] + "\n\n"; ch = a["crumb"]
    spans.append((len(flat), len(flat) + len(a["text"]), a["no"])); flat += a["text"] + "\n\n"
S0_DOCS = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True).create_documents([flat])
S0_ARTS = [[no for s, e, no in spans if s < d.metadata["start_index"] + len(d.page_content) and e > d.metadata["start_index"]] for d in S0_DOCS]
S0_V = unit(EMB.embed_documents([d.page_content for d in S0_DOCS]))

# ---------- S1: structural chunks ----------
CH, CH_ART = [], []
for a in ARTS:
    if a["no"] == "2":
        for it in a["items"]: CH.append(f"{head(a)} 이 법에서 사용하는 용어의 뜻은 다음과 같다.\n{it}"); CH_ART.append("2")
    else: CH.append(art_block(a)); CH_ART.append(a["no"])
CH_V = unit(EMB.embed_documents(CH))
def tok(s): z = re.sub(r"\s+", "", s); return [z[i:i + 2] for i in range(len(z) - 1)] + s.split()
BM = BM25Okapi([tok(c) for c in CH])

def dense(q, V): return list(np.argsort(-(V @ unit(EMB.embed_query(q)))))
def boost(q, chunk_rank):  # explicit "제N조" in the question → that article's chunks first
    want = re.findall(r"제(\d+)조", q)
    return [i for n in want for i in chunk_rank if CH_ART[i] == n] + [i for i in chunk_rank if CH_ART[i] not in want]
def s0(q): return dedupe(S0_ARTS[i] for i in dense(q, S0_V))
def s1_chunks(q): return dense(q, CH_V)
def s2_chunks(q): return boost(q, rrf([dense(q, CH_V), list(np.argsort(-BM.get_scores(tok(q))))]))
def to_arts(chunks): return dedupe([CH_ART[i]] for i in chunks)

REWRITE = """너는 한국 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」 조문 검색을 위한 질의 재작성기다.
사용자 질문을 법률 조문 검색에 적합한 법률 용어 질의로 바꾸고, 질문에 쟁점이 여러 개면 하위 질의(최대 3개)로 나눠라.
JSON만 출력: {{"rewrite": "...", "subqueries": ["...", "..."]}}
질문: {q}"""
def s3_chunks(q):
    rw = jparse(LLM.invoke(REWRITE.format(q=q)).content)
    qs = [q, rw["rewrite"]] + rw.get("subqueries", [])[:3]
    return boost(q, rrf([s2_chunks(x) for x in qs])), rw, ntok(REWRITE.format(q=q))

from sentence_transformers import CrossEncoder
RR = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")
RR_PROMPT = "Given a Korean question about Korea's AI Basic Act, judge whether this law article passage helps answer the question"
def s4_chunks(q, s3):
    cand = s3[:20]
    sc = RR.predict([(q, CH[i]) for i in cand], prompt=RR_PROMPT)
    order = [cand[j] for j in np.argsort(-sc)]
    return boost(q, order), float(sc.max()), {CH_ART[c]: round(float(s), 2) for c, s in sorted(zip(cand, sc), key=lambda x: x[1])}

# ---------- answers ----------
BASE = "한국 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」(법률 제21311호)에 관한 질문에 한국어로 간결하게(5문장 이내) 답하세요. 근거 조문이 있으면 밝히세요.\n"
GROUNDED = """당신은 한국 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법」 질의응답 도우미입니다. 규칙:
1. 아래 [컨텍스트]의 조문 원문에 있는 내용만으로 답하세요. 컨텍스트에 없는 사실(다른 법률, 금액, 해외 제도 등)을 덧붙이지 마세요.
2. 각 주장 뒤에 근거를 [제34조①] 또는 [제2조제4호] 형식으로 인용하세요.
3. 컨텍스트에서 답을 찾을 수 없으면 "제공된 조문에서는 확인할 수 없습니다"라고 답하세요.
4. 세부 사항이 대통령령에 위임된 경우("대통령령으로 정한다/정하는") 그 사실을 표시하세요.
5. 한국어로 간결하게(5문장 이내) 답하세요.

[컨텍스트]
{ctx}

질문: {q}"""
REFUSAL = "제공된 조문에서는 확인할 수 없습니다. (검색된 조문의 관련도가 기준 미만이라 답변하지 않음)"
TERMS = {re.match(r'\d+\. "([^"]+)"', it).group(1): it for it in BY["2"]["items"]}
TERMS.pop("인공지능")  # too generic: would attach to every question

def strip_external(t):  # drop references into other laws: 「X법」 제N조…, 같은 법 제N조…
    return re.sub(r"(「[^」]*」|같은 법)\s*제\d+조[^\s,]*", "", t)
def d_context(q, rw, arts):
    top = arts[:5]; blocks = [art_block(BY[n]) for n in top]
    qside = " ".join([q, rw["rewrite"], *rw.get("subqueries", [])])
    defs = [f"[정의: 제2조 \"{t}\"] {it}" for t, it in TERMS.items() if t in qside and "2" not in top]
    refs, seen = [], set()
    for n in top:
        for m in re.finditer(r"제(\d+)조((?:제\d+항(?:ㆍ|및|\s)*)*)", strip_external(BY[n]["text"])):
            r = m.group(1)
            if r in top or r == n or r not in BY: continue
            hs = re.findall(r"제(\d+)항", m.group(2)) or [None]
            for h in hs:
                if (r, h) in seen: continue
                seen.add((r, h)); a = BY[r]
                txt = a["paras"][int(h) - 1] if h and len(a["paras"]) >= int(h) else a["text"]
                refs.append(f"[참조 조문: 제{r}조{'제' + h + '항' if h else ''}({a['title']})] {txt}")
    return "\n\n".join(blocks + defs + refs[:6]), defs, refs[:6]

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
def judge(g, ans):
    cited = set(re.findall(r"제(\d+)조", ans)) - set(g["gold_articles"])
    nos = [n for n in g["gold_articles"] + sorted(cited, key=int) if n in BY]
    texts = "\n\n".join(art_block(BY[n]) for n in nos) or "(없음)"
    p = JUDGE_P.format(q=g["question"], oos=g["out_of_scope"], ref=g["reference_answer"], gold=g["gold_articles"] or "없음", texts=texts, ans=ans)
    return jparse(JUDGE.invoke(p).content)

def main():
    det = []
    for g in G:
        q = g["question"]; d = {"id": g["id"], "type": g["type"], "question": q, "gold": g["gold_articles"], "oos": g["out_of_scope"]}
        s3c, rw, rw_tok = s3_chunks(q); s4c, mx, scores = s4_chunks(q, s3c)
        d["ret"] = {"S0": s0(q)[:5], "S1": to_arts(s1_chunks(q))[:5], "S2": to_arts(s2_chunks(q))[:5], "S3": to_arts(s3c)[:5], "S4": to_arts(s4c)[:5]}
        d["rewrite"], d["rewrite_tokens"], d["rerank_max"], d["rerank_scores"] = rw, rw_tok, mx, scores
        s0_top4 = "\n\n---\n\n".join(S0_DOCS[i].page_content for i in dense(q, S0_V)[:4])
        ctx, defs, refs = d_context(q, rw, to_arts(s4c))
        prompts = {"A": BASE + f"\n질문: {q}", "B": BASE + f"\n[법률 전문]\n{flat}\n질문: {q}",
                   "C": BASE + f"\n[참고 문서]\n{s0_top4}\n\n질문: {q}", "D": GROUNDED.format(ctx=ctx, q=q)}
        prompts["D_nogate"] = prompts["D"]  # same as D but without the out-of-scope score gate
        d["d_attached"] = {"defs": [x[:40] for x in defs], "refs": [x[:40] for x in refs]}
        d["ans"] = {}
        for c, p in prompts.items():
            refuse = c == "D" and mx < RERANK_THRESHOLD
            text = REFUSAL if refuse else LLM.invoke(p).content
            d["ans"][c] = {"text": text, "tokens": 0 if refuse else ntok(p), "refused_by_threshold": refuse, "judge": judge(g, text)}
        det.append(d); print(g["id"], {k: v[:5] for k, v in d["ret"].items()}, round(mx, 2), {c: v["judge"]["correct"] for c, v in d["ans"].items()}, flush=True)
    json.dump(det, open(EXP / "details.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
