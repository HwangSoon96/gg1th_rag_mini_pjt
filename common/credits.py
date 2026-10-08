"""MonoRouter 잔여 크레딧 조회 (LLM 호출 아님 — 한도 소모 없음). 월 한도라 평가처럼 호출이 많은 작업 전후로 확인할 것."""
import requests

from common.config import API_KEY, BASE_URL


def credits() -> dict:
    """{"max_credits", "used_credits", "remaining_credits", ...}"""
    r = requests.get(f"{BASE_URL.rstrip('/')}/credits", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    c = credits()
    print(f"남은 크레딧 {c['remaining_credits']:,} / {c['max_credits']:,} (사용 {c['used_credits']:,})")
