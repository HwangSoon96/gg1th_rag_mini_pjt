import os

from dotenv import load_dotenv

load_dotenv(override=True)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
TEMPERATURE = os.getenv("LLM_TEMPERATURE", 2)
MAX_TOKENS = os.getenv("LLM_MAX_TOKENS", 2086)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

# 국가법령정보 Open API 인증값(OC). "test"는 공용 샘플 계정이라 실사용 시 본인 OC로 교체
LAW_OC = os.getenv("LAW_OC", "test")
