"""Parse 인공지능기본법 XML into articles. Run directly to dump text."""
import re, xml.etree.ElementTree as ET
from pathlib import Path
XML = Path(__file__).parent.parent / "law/api.xml"

def t(e, tag):
    x = e.find(tag)
    # 현행 XML은 본문에 <개정 2026.1.20>·<신설 …> 표시가 섞여 있어 제거 (구 XML엔 없었음)
    return re.sub(r"\s*<(?:개정|신설)[^>]*>", "", x.text or "").strip() if x is not None else ""

def load():
    arts, chapter, section = [], "", ""
    for u in ET.parse(XML).getroot().iter("조문단위"):
        body = t(u, "조문내용")
        if t(u, "조문여부") == "전문":
            if re.match(r"제\d+장", body): chapter, section = body, ""
            else: section = body
            continue
        no = t(u, "조문번호") + (f"의{t(u,'조문가지번호')}" if t(u, "조문가지번호") else "")
        lines, items, paras = [body], [], []   # items: per-호 pieces (제2조); paras: per-항 text
        for h in u.findall("항"):
            pl = [t(h, "항내용")] if t(h, "항내용") else []
            for ho in h.findall("호"):
                hl = [t(ho, "호내용")] + ["  " + t(m, "목내용") for m in ho.findall("목")]
                pl += hl; items.append("\n".join(hl))
            lines += pl; paras.append("\n".join(pl))
        crumb = " > ".join(x for x in (chapter, section) if x)
        arts.append(dict(no=no, title=t(u, "조문제목"), crumb=crumb, text="\n".join(lines), items=items, paras=paras))
    return arts

if __name__ == "__main__":
    a = load(); print(len(a), sum(len(x["text"]) for x in a))
    for x in a: print(f"### [{x['crumb']}] {x['no']}\n{x['text']}\n")
