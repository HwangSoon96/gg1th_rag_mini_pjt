"""LLM·임베딩 없이 검증 가능한 파이프라인 로직: 컨텍스트 첨부(정의·참조·역참조), 인용 파싱, 토크나이저/RRF."""
from rag.pipeline import _law, build_context, parse_sources, strip_external
from rag.retriever import cited_articles, rrf, tokenize

RW = {"rewrite": "", "subqueries": []}


def test_reverse_index_points_sanctions_to_duties():
    rev = _law()["rev"]
    assert [u["label"] for u in rev["36"]] == ["제43조①2"]
    assert {u["label"] for u in rev["31"]} == {"제43조①1", "제40조①1", "제40조①2"}  # 제40조①의 1호(제31조·제32조·제34조 위반)·2호(신고)
    assert [u["label"] for u in rev["7"]] == ["제42조"]
    assert "40" in {u["no"] for u in rev["34"]}


def test_reverse_refs_attach_penalty_for_duty_article():
    # 제36조(국내대리인)만 상위에 있어도 제43조①2(미지정 시 과태료)가 붙어야 한다
    ctx, meta = build_context("국내대리인 안 두면?", RW, ["36"])
    assert "제43조①2" in meta["reverse_references"]
    assert "3천만원 이하의 과태료" in ctx and "국내대리인을 지정하지 아니한 자" in ctx
    _, meta0 = build_context("국내대리인 안 두면?", RW, ["36"], reverse=False)
    assert meta0["reverse_references"] == [] and "43" not in meta0["attached_articles"]


def test_definitions_and_external_refs():
    _, meta = build_context("인공지능취약계층 참여는?", RW, ["6"])
    assert "제3조⑤" in meta["definitions"]
    _, meta = build_context("고영향 인공지능 사업자 책무", RW, ["34"])
    assert "제2조4" in meta["definitions"]
    assert "제2조" not in strip_external("「에너지법」 제2조제1호에 따른")
    assert "인공지능" not in _law()["terms"]


def test_forward_refs_resolve_units():
    _, meta = build_context("과태료", RW, ["43"])
    assert {"제31조①", "제36조①", "제40조③"} <= set(meta["references"])


def test_parse_sources_variants():
    ans = ("고지해야 합니다[제31조①]. 어기면 과태료입니다[제43조①1]. 대출 심사는 고영향 영역입니다[제2조제4호사목]. "
           "국내대리인[제36조 제1항, 제43조①2]. 없는 조문[제99조]. 다시[제31조①].")
    src = parse_sources(ans)
    assert [s["article"] for s in src] == ["제31조①", "제43조①1", "제2조4", "제36조①", "제43조①2"]
    assert src[1]["content"].startswith("① 다음 각 호") and "제31조제1항을 위반하여" in src[1]["content"]
    assert src[0]["markers"] == ["제31조①"] and src[0]["title"] == "인공지능 투명성 확보 의무"
    assert parse_sources("제공된 조문에서는 확인할 수 없습니다.") == []


def test_retrieval_helpers():
    assert cited_articles("제36조와 제22조의2는?") == ["36", "22의2"]
    assert rrf([[1, 2], [2, 3]])[0] == 2
    assert "고영" in tokenize("고영향 AI") and "AI" in tokenize("고영향 AI")
