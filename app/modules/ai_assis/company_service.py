import time
import httpx
from bs4 import BeautifulSoup


_COMPANY_PROFILE_CACHE = {
    "content": None,
    "expired_at": 0,
}


async def fetch_company_profile_context() -> str:
    now = time.time()

    if (
        _COMPANY_PROFILE_CACHE["content"]
        and _COMPANY_PROFILE_CACHE["expired_at"] > now
    ):
        return _COMPANY_PROFILE_CACHE["content"]

    url = "https://www.entraining.net/about.php"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 AI Assistant"
            }
        )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript"
    ]):
        tag.decompose()

    text = soup.get_text("\n")

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if len(line) <= 1:
            continue

        lines.append(line)

    content = "\n".join(lines)

    # กัน context ยาวเกิน
    content = content[:8000]

    _COMPANY_PROFILE_CACHE["content"] = content
    _COMPANY_PROFILE_CACHE["expired_at"] = now + 3600

    return content