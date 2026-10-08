import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.pipeline import ask as run_ask
from rag.pipeline import stream as run_stream
from rag.retriever import get_retriever

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
REPORT = ROOT / "docs/report.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_retriever()  # 조문 로딩·Qdrant 컬렉션 확인을 첫 요청 전에
    yield


app = FastAPI(title="AI 기본법 QA", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Source(BaseModel):
    article: str
    content: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]


def _friendly(e: Exception) -> str:
    if "429" in str(e) or "rate" in str(e).lower():
        return "요청이 몰려 잠시 쉬고 있어요. 1분 뒤에 다시 시도해 주세요."
    return "답변을 만드는 중에 문제가 생겼어요. 잠시 후 다시 시도해 주세요."


@app.post("/ask", response_model=Answer)
def ask(body: Question):
    try:
        return run_ask(body.question.strip())
    except Exception as e:
        raise HTTPException(503, _friendly(e)) from e


@app.post("/ask/stream")
def ask_stream(body: Question):
    def events():
        try:
            for ev, data in run_stream(body.question.strip()):
                yield f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': _friendly(e)}, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _blocks(a: dict) -> list[dict]:
    """조문 패널용 비중복 블록: 항이 있으면 항 단위(호 포함), 항 없는 조문(제2조 등)은 머리 줄 + 호 단위."""
    if a["paras"][0]["para"]:
        return [{"label": p["label"], "text": p["text"]} for p in a["paras"]]
    head = a["paras"][0]
    return [{"label": head["label"], "text": head["text"].split("\n")[0]}] + [
        {"label": i["label"], "text": i["text"]} for i in a["items"]]


@app.get("/articles")
def articles():
    return [{"no": a["no"], "article": a["article"], "title": a["title"], "chapter": a["chapter"], "section": a["section"],
             "text": a["text"], "units": _blocks(a)} for a in get_retriever().arts]


@app.get("/report", include_in_schema=False)
def report():
    if not REPORT.exists():
        raise HTTPException(404, "평가 리포트가 아직 없어요. uv run python eval/report.py 로 만들 수 있어요.")
    return FileResponse(REPORT)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
