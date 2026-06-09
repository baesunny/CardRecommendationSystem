"""
OpenAI API 키 로딩 및 클라이언트 생성 (공통)
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

_REPO_ROOT = Path(__file__).parent.parent.parent
_PROJECT_ROOT = Path(__file__).parent.parent
_GPT_DIR = Path(__file__).parent

# 모듈 상수로 직접 넣을 때 사용 (비우면 .env / 환경변수 사용)
OPENAI_API_KEY = ""


def _setup_ssl() -> None:
    """Windows/학교망 등에서 Python SSL 검증 실패 방지."""
    try:
        import truststore

        truststore.inject_into_ssl()
        return
    except ImportError:
        pass
    try:
        import certifi

        bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    except ImportError:
        pass


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=True)
    load_dotenv(_PROJECT_ROOT / ".env", override=True)
    load_dotenv(_GPT_DIR / ".env", override=True)


def reload_env() -> None:
    """앱 시작 시 .env를 다시 읽는다."""
    _load_dotenv()


_setup_ssl()
_load_dotenv()


def _clean_key(value: str) -> str:
    key = value.strip().strip('"').strip("'")
    if key.startswith("\ufeff"):
        key = key[1:]
    return key


def resolve_api_key() -> str | None:
    """API 키 조회. 없으면 None."""
    reload_env()
    for candidate in (
        OPENAI_API_KEY,
        os.environ.get("OPENAI_API_KEY", ""),
    ):
        if candidate:
            cleaned = _clean_key(str(candidate))
            if cleaned:
                return cleaned
    return None


def require_api_key() -> str:
    key = resolve_api_key()
    if not key:
        raise ValueError(
            "OpenAI API 키가 설정되지 않았다.\n"
            "저장소 루트 또는 v2/.env 파일에 OPENAI_API_KEY=your-api-key-here 를 설정해야 한다."
        )
    return key


def create_client() -> OpenAI:
    return OpenAI(
        api_key=require_api_key(),
        timeout=120.0,
        max_retries=3,
        http_client=httpx.Client(timeout=120.0),
    )


def test_connection() -> tuple[bool, str]:
    """연결 테스트. (성공 여부, 메시지)"""
    try:
        client = create_client()
        client.models.list()
        return True, "OpenAI API 연결 성공"
    except Exception as exc:
        return False, format_api_error(exc)


def format_api_error(exc: Exception) -> str:
    if isinstance(exc, ValueError) and "API 키" in str(exc):
        return str(exc)
    if isinstance(exc, AuthenticationError):
        return (
            "OpenAI API 키가 올바르지 않다. "
            ".env 파일의 OPENAI_API_KEY 값을 확인해야 한다."
        )
    msg = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in msg or "certificate verify failed" in msg.lower():
        return (
            "SSL 인증서 검증에 실패했다 (학교/회사망에서 자주 발생).\n"
            "• 터미널에서 `py -m pip install truststore` 실행 후 앱을 재시작해야 한다.\n"
            "• VPN/프록시 사용 중이면 해제 후 다시 시도해야 한다."
        )
    if isinstance(exc, APIConnectionError):
        return (
            "OpenAI 서버에 연결할 수 없다 (Connection error).\n"
            "• 인터넷 연결과 방화벽/프록시(VPN) 설정을 확인해야 한다.\n"
            "• .env 파일에 OPENAI_API_KEY가 올바르게 설정되어 있는지 확인해야 한다."
        )
    if isinstance(exc, RateLimitError):
        return "OpenAI API 호출 한도를 초과했다. 잠시 후 다시 시도해야 한다."
    return msg
