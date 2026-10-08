"""인공지능기본법 HWPX → 조문 리스트.

HWPX(zip) 안 Contents/section*.xml의 문단(hp:p)을 한 줄씩 읽어 장·절·조·항·호·목 구조로 묶는다.
표준 라이브러리만 사용 (구조가 XML에 이미 있어 Docling 불필요).
"""
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HWPX = Path(__file__).resolve().parent.parent / "data/ai_basic_law/ai_basic_law_21311_20260721.hwpx"
_P = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"

ARTICLE_RE = re.compile(r"^제(\d+)조(?:의(\d+))?\(([^)]+)\)\s*(.*)$")
CHAPTER_RE = re.compile(r"^제\d+장\s")
SECTION_RE = re.compile(r"^제\d+절\s")
PARA_MARKS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
ITEM_RE = re.compile(r"^(\d+(?:의\d+)?)\.\s")      # 호: "1. ", "4의2. "
SUBITEM_RE = re.compile(r"^([가-힣])\.\s")          # 목: "가. "
# 개정·신설 표시와 [본조신설 …]/[시행일 …]은 본문이 아니다
REVISION_RE = re.compile(r"\s*(?:<(?:개정|신설|전문개정|타법개정)[^>]*>|\[[^\]]*(?:개정|신설|시행일|종전)[^\]]*\])")


def article_label(no: str) -> str:
    """"17의2" → "제17조의2" (법령 표기)."""
    base, _, sub = no.partition("의")
    return f"제{base}조" + (f"의{sub}" if sub else "")


def _lines(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if re.fullmatch(r"Contents/section\d+\.xml", n))
        return ["".join(t.text or "" for t in p.iter(_P + "t"))
                for n in names for p in ET.fromstring(z.read(n)).iter(_P + "p")]


def _clean(s: str) -> str:
    return REVISION_RE.sub("", s).rstrip()


def _structure(no: str, body: list[str]) -> tuple[list[dict], list[dict]]:
    """본문 줄 → 항(paras)과 호(items). 각 원소는 {"no","label","para","item","text"}.

    label 예: 제34조①, 제34조①2, 제2조4(항 없는 조문의 호). 목은 호 text에 포함.
    """
    paras, items = [], []
    para, item = None, None
    for raw in body:
        s = raw.strip()
        if not s:
            continue
        if s[0] in PARA_MARKS:
            para = {"no": no, "label": f"{article_label(no)}{s[0]}", "para": s[0], "item": None, "text": s}
            paras.append(para)
            item = None
            continue
        if para is None:  # 항 없는 조문(제2조 등): 조문 머리를 이름 없는 항으로
            para = {"no": no, "label": article_label(no), "para": None, "item": None, "text": s}
            paras.append(para)
            continue
        m = ITEM_RE.match(s)
        if m and raw.startswith("  ") and not raw.startswith("    "):
            item = {"no": no, "label": f"{para['label']}{m.group(1)}", "para": para["para"], "item": m.group(1), "text": s}
            items.append(item)
        elif SUBITEM_RE.match(s) and item is not None:
            item["text"] += "\n  " + s
        elif item is not None:
            item["text"] += " " + s
        para["text"] += "\n" + ("  " + s if (item is not None) else s)
    return paras, items


def load(path: Path = HWPX) -> list[dict]:
    """조문 46개. 각 조문: no, article, title, chapter, section, text, paras, items."""
    arts, chapter, section, cur = [], "", "", None
    started = False
    for raw in _lines(path):
        line = _clean(raw)
        s = line.strip()
        if s.startswith("부칙"):  # 부칙에도 제1조~가 있어 번호가 겹친다 → 본문 끝
            break
        if CHAPTER_RE.match(s):
            chapter, section, started = s, "", True
            continue
        if SECTION_RE.match(s):
            section = s
            continue
        if not started:  # 머리말(법령명·법제처 표기·담당부서)
            continue
        m = ARTICLE_RE.match(s)
        if m:
            no = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
            cur = {"no": no, "article": article_label(no), "title": m.group(3), "chapter": chapter,
                   "section": section, "head": f"{article_label(no)}({m.group(3)})", "body": [m.group(4)] if m.group(4) else []}
            arts.append(cur)
        elif cur is not None and s:
            cur["body"].append(line)
    for a in arts:
        body = a.pop("body")
        a["paras"], a["items"] = _structure(a["no"], body)
        a["text"] = "\n".join([f"{a.pop('head')} {body[0].strip()}" if body else a["article"]]
                              + [b.rstrip() for b in body[1:]])
        a["crumb"] = " > ".join(x for x in (a["chapter"], a["section"]) if x)
    return arts


def units(a: dict) -> list[dict]:
    """조문의 인용 가능한 최소 단위(항·호) 전부."""
    return a["paras"] + a["items"]


if __name__ == "__main__":
    arts = load()
    print(len(arts), sum(len(a["text"]) for a in arts))
    for a in arts[:3]:
        print(a["crumb"], a["article"], a["title"], [u["label"] for u in units(a)][:8])
