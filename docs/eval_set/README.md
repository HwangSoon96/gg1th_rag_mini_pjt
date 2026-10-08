# AI 기본법 RAG 평가셋 (v1)

버전: v1 (2026-10-08, 법률 제21311호 기준)

## 목적
모든 단계를 같은 문항, 같은 채점 기준으로 비교해요. 근거는 모두 `data/ai_basic_law/articles.json`(46개 조문) 원문이에요.

## 파일
| 파일 | 설명 |
|---|---|
| `golden_set.jsonl` | Golden Test Set 40문항 (base20 + ext20). 성능 비교에 써요 |
| `probe_set.jsonl` | Reranker 점수 임계값 보정용 8문항(범위 내 4, 범위 밖 4). Golden Test Set과 분리된 held-out이라 Golden Test Set로 임계값을 튜닝하지 않아요 |
| `README.md` | 이 문서 |

실행: `uv run python eval/evaluate.py`

## golden_set.jsonl 스키마
한 줄에 JSON 하나예요. `ensure_ascii=False`로 저장해요.

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | string | `g01`~`g40`. 재사용 금지 |
| `type` | string | 일상어, 조문번호/고유명사, multi-hop, 정의/세부, out-of-scope |
| `question` | string | 사용자 질문 |
| `gold_articles` | string[] | 답하는 데 꼭 필요한 조문 번호. `"31"`, `"22의2"`처럼 '제'/'조' 없이 적어요. 범위 밖이면 `[]` |
| `reference_answer` | string | 정답 요지. 근거를 `(제N조①)` 식으로 표기해요 |
| `out_of_scope` | bool | 이 법에 없는 내용을 묻는 질문인지 |
| `set` | string | `base20`(동결) 또는 `ext20` |
| `evidence` | object[] | `{"article","quote"}`. 정답 요지의 핵심 주장마다 원문 조문의 연속된 일부(20~120자)를 그대로 옮겨요. 범위 밖이면 `[]` |

`evidence.quote`는 공백·따옴표를 지운 뒤 해당 조문 `content`의 부분문자열이어야 해요.

`probe_set.jsonl` 필드: `id`(p01~p08), `question`, `out_of_scope`, `purpose`.

## 유형별 문항 수
| 유형 | base20 | ext20 | 합계 |
|---|---|---|---|
| 일상어 | 6 | 6 | 12 |
| 조문번호/고유명사 | 4 | 3 | 7 |
| multi-hop | 5 | 4 | 9 |
| 정의/세부 | 3 | 4 | 7 |
| out-of-scope | 2 | 3 | 5 |
| 합계 | 20 | 20 | 40 |

## 문항 수를 40으로 정한 이유
- 1문항이 2.5%p예요. 20문항(5%p)보다 차이를 더 잘게 볼 수 있어요.
- base20은 기존 실험 기준선과 비교할 수 있게 동결했어요.
- ext20은 base20이 다루지 않은 조문과 '의무 조문 -> 제재 조문(제40·42·43조)' 연결 질문을 채워요.

## 검색 지표
조 단위로 비교해요(항·호는 보지 않아요). 범위 내 문항(`out_of_scope=false`)만 계산해요.

| 지표 | 정의 |
|---|---|
| Hit@3 | 상위 3개 조문에 gold가 하나라도 있으면 1, 없으면 0 |
| Recall@5 | gold 조문 중 상위 5개 조문에 든 비율 |
| MRR | 첫 gold 조문 순위의 역수 (없으면 0) |

## 답변 채점 루브릭
채점 LLM 기준은 `docs/experiment/run.py`의 `JUDGE_P`를 따라요.

| 항목 | 값 | 기준 |
|---|---|---|
| `correct` | 0 / 0.5 / 1 | 1: 정답 요지 핵심을 모두 맞게 담음. 0.5: 일부만 맞거나 핵심이 일부 빠지거나 부정확함. 0: 틀림, 핵심 누락, 범위 내 질문인데 답변 거부. 범위 밖 질문은 '이 법에 없다/답할 수 없다'를 분명히 하면 1, 이 법의 내용처럼 답하면 0 |
| `citations_valid` | true / false / "na" | 조문 번호를 인용하지 않으면 "na". 인용한 조문(항·호)이 원문상 주장을 뒷받침하면 true, 하나라도 틀리면 false |
| `hallucination` | true / false | 원문이나 정답 요지와 모순되거나 원문에 없는 내용(없는 금액·의무·조문 번호 등)을 이 법의 내용처럼 단정하면 true. 노력의무를 의무로 단정하는 것도 포함해요 |
| `refused_correctly` | true / false / "na" | 범위 밖 질문에서 이 법의 범위 밖임을 밝혔는지. 범위 내 질문이면 "na" |

## 채점 LLM 오판 재검토
채점 LLM이 잘못 채점한 건 사람이 확인한 뒤 `overrides.json`에 적어 덮어써요. 예시는 `docs/experiment/overrides.json`을 참고해요.

| 필드 | 설명 |
|---|---|
| `id` | 문항 id |
| `cond` | 조건 이름(예: `D`) |
| `set` | 덮어쓸 값. 예: `{"correct": 0}` |
| `why` | 덮어쓴 이유 한 줄 |

## 문항 추가·수정 규칙
- id는 다시 쓰지 않아요. 지운 문항의 id도 비워 둬요.
- base20(g01~g20)은 수정하지 않아요.
- 문항을 추가하거나 고치면 버전을 올리고(v2, v3 ...) 날짜와 변경 내용을 이 문서에 남겨요.
- `evidence.quote`는 반드시 `articles.json` 원문과 일치해야 해요.
- 노력의무('노력하여야 한다')와 의무('하여야 한다')를 구분해서 적어요.

## 버전
| 버전 | 날짜 | 기준 | 내용 |
|---|---|---|---|
| v1 | 2026-10-08 | 법률 제21311호 | base20 동결 + ext20 추가, evidence 도입 |
