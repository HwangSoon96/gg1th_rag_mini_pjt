import time

import httpx
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from common.config import (
    API_KEY,
    BASE_URL,
    EMBEDDING_MODEL,
    MODEL,
)

# MonoRouter는 API 키당 30회/분 제한 (초과 시 429 + retry-after 60초).
# SDK 재시도는 최대 8초 간격이라 60초를 못 기다리고 실패 요청으로 한도만 더 소모하므로,
# LLM과 임베딩이 이 리미터 하나를 공유해 미리 24회/분 이하로 보낸다.
RATE_LIMITER = InMemoryRateLimiter(requests_per_second=0.4, check_every_n_seconds=0.1, max_bucket_size=1)


class RetryAfterTransport(httpx.HTTPTransport):
    """429면 retry-after(초)만큼 기다렸다 재시도. 리미터는 이 프로세스만 막으므로,
    같은 키를 쓰는 다른 프로세스(노트북·팀원)가 한도를 채워도 실패 요청으로 한도를 더 소모하지 않게 한다."""

    def __init__(self, tries: int = 4, **kw):
        super().__init__(**kw)
        self.tries = tries

    def handle_request(self, request):
        for _ in range(self.tries - 1):
            resp = super().handle_request(request)
            if resp.status_code != 429:
                return resp
            wait = float(resp.headers.get("retry-after") or 60)
            resp.close()
            print(f"[MonoRouter 429] {wait:.0f}초 대기 후 재시도", flush=True)
            time.sleep(wait + 1)
        return super().handle_request(request)


HTTP_CLIENT = httpx.Client(transport=RetryAfterTransport(), timeout=httpx.Timeout(120, connect=10))


class RateLimitedEmbeddings(OpenAIEmbeddings):
    # ponytail: embed_documents 1회 = 요청 1회로 계산. 텍스트가 chunk_size(기본 1000)를 넘으면 실제 요청은 여러 번
    def embed_documents(self, texts, chunk_size=None, **kwargs):
        RATE_LIMITER.acquire()
        return super().embed_documents(texts, chunk_size=chunk_size, **kwargs)

    def embed_query(self, text, **kwargs):
        RATE_LIMITER.acquire()
        return super().embed_query(text, **kwargs)


def get_llm_model(
    model: str = MODEL,
    api_key: str = API_KEY,
    temperature: float = 0,
    max_tokens: int = 512
):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
        rate_limiter=RATE_LIMITER,
        http_client=HTTP_CLIENT,
        max_retries=0,  # 429 재시도는 RetryAfterTransport가 담당
    )


def get_embedding_model():
    embedding_model = EMBEDDING_MODEL
    embeddings = RateLimitedEmbeddings(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=embedding_model,
        http_client=HTTP_CLIENT,
        max_retries=0,
        )
    # print(embeddings)
    return embeddings
