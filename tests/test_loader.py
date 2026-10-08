"""HWPX 파서 결과가 법제처 Open API 조문(articles.json)과 본문까지 일치하는지 대조."""
import json
import re
from pathlib import Path

from rag.loader import load, units

API = Path(__file__).resolve().parent.parent / "data/ai_basic_law/articles.json"


def norm(s: str) -> str:
    s = re.sub(r"<(?:개정|신설)[^>]*>|\[[^\]]*(?:개정|신설|시행일)[^\]]*\]", "", s)
    return re.sub(r"[\s\"'“”‘’]", "", s)


ARTS = load()
API_ARTS = {a["article"]: a for a in json.loads(API.read_text())}


def test_count_and_numbers():
    assert len(ARTS) == 46
    nos = [a["no"] for a in ARTS]
    assert nos[:2] == ["1", "2"] and nos[-1] == "43"
    assert {"17의2", "22의2", "22의3"} <= set(nos)
    assert len(set(nos)) == 46  # 부칙 제1조 등이 섞이지 않음


def test_text_matches_open_api():
    for a in ARTS:
        api = API_ARTS[a["article"]]
        assert a["title"] == api["title"], a["article"]
        assert norm(a["text"]) == norm(api["content"]), a["article"]


def test_breadcrumb_and_no_metadata_leak():
    by = {a["no"]: a for a in ARTS}
    assert by["1"]["chapter"] == "제1장 총칙"
    assert len({a["chapter"] for a in ARTS}) == 6
    assert {a["chapter"] for a in ARTS if a["section"]} == {"제3장 인공지능기술 개발 및 산업 육성"}
    assert len({a["section"] for a in ARTS if a["section"]}) == 2
    for a in ARTS:
        assert not re.search(r"제\d+[장절]\s", a["text"].splitlines()[-1]), a["article"]
        assert "<개정" not in a["text"] and "법제처" not in a["text"]


def test_units():
    by = {a["no"]: a for a in ARTS}
    labels = [u["label"] for u in units(by["43"])]
    assert labels == ["제43조①", "제43조②", "제43조①1", "제43조①2", "제43조①3"]
    two = {u["label"]: u["text"] for u in by["2"]["items"]}
    assert len(two) == 12 and two["제2조4"].count("\n") == 11  # 4호 = 머리 + 가~카목 11개
    assert by["3"]["paras"][4]["label"] == "제3조⑤" and "인공지능취약계층" in by["3"]["paras"][4]["text"]
    assert all(len(a["paras"]) >= 1 for a in ARTS)
