import time
import re
import httpx
from bs4 import BeautifulSoup


_INSTRUCTOR_CACHE = {
    "items": None,
    "expired_at": 0,
}

_INSTRUCTOR_DETAIL_CACHE = {
    "items": {},
}

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_instructor_url(instructor_name: str) -> str:
    instructor_name = (instructor_name or "").strip()

    if not instructor_name:
        return ""

    slug_name = instructor_name.strip()

    prefixes = [
        "อ.",
        "อ. ",
        "อ ",
        "อาจารย์",
        "ดร.",
        "ดร. ",
        "ดร ",
        "คุณ",
        "รศ.",
        "รศ.ดร.",
        "นพ.",
        "ภญ.",
        "นสพ.",
    ]

    for prefix in prefixes:
        if slug_name.startswith(prefix):
            slug_name = slug_name[len(prefix):].strip()
            break

    slug = re.sub(r"\s+", "-", slug_name)
    slug = re.sub(r"-+", "-", slug)

    return f"https://www.entraining.net/expert/{slug}/"


def clean_style_name(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    replacements = [
        "วิทยากรที่ดำเนินการสอน",
        "หลักสูตร",
        "style",
        "Style",
    ]

    for r in replacements:
        text = text.replace(r, "")

    return text.strip(" -")


async def fetch_instructor_context() -> list[dict]:
    now = time.time()

    if (
        _INSTRUCTOR_CACHE["items"]
        and _INSTRUCTOR_CACHE["expired_at"] > now
    ):
        return _INSTRUCTOR_CACHE["items"]

    url = "https://entstaffs.entraining.net/api/expert/"

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            url,
            headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "th-TH,th;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.entraining.net/",
    "Connection": "keep-alive",
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

    instructors = []
    seen = set()

    # หา heading ของแต่ละ style
    headings = soup.select("h2.section-sub-title")

    for heading in headings:

        style_name = clean_style_name(
            heading.get_text(" ", strip=True)
        )

        # หา row ถัดจาก heading
        row = heading.find_parent().find_next("div", class_="row")

        if not row:
            continue

        # หา card วิทยากรทั้งหมดใน section นี้
        cards = row.select("div.col-6")

        for card in cards:

            link = card.select_one("h3.ts-name a")

            if not link:
                continue

            instructor_name = link.get_text(" ", strip=True)

            if not instructor_name:
                continue

            instructor_url = link.get("href", "").strip()

            # fallback
            if not instructor_url:
                instructor_url = build_instructor_url(
                    instructor_name
                )

            designation_el = card.select_one(
                "p.ts-designation"
            )

            designation = ""

            if designation_el:
                designation = designation_el.get_text(
                    " ",
                    strip=True
                )

            image_el = card.select_one("img")

            image_url = ""

            if image_el:
                image_url = image_el.get("src", "").strip()

            unique_key = (
                instructor_name.lower(),
                style_name.lower(),
            )

            if unique_key in seen:
                continue

            seen.add(unique_key)

            instructors.append({
                "instructor_name": instructor_name,
                "instructor_url": instructor_url,
                "style": style_name,
                "designation": designation,
                "image_url": image_url,
            })

    _INSTRUCTOR_CACHE["items"] = instructors
    _INSTRUCTOR_CACHE["expired_at"] = now + 3600

    return instructors

async def fetch_instructor_detail(instructor_url: str) -> dict:
    instructor_url = (instructor_url or "").strip()

    if not instructor_url:
        return {}

    now = time.time()

    cached = _INSTRUCTOR_DETAIL_CACHE["items"].get(
        instructor_url
    )

    if cached and cached["expired_at"] > now:
        return cached["data"]

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            instructor_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "th-TH,th;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.entraining.net/",
                "Connection": "keep-alive",
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

    title = ""

    title_el = (
        soup.select_one("h1")
        or soup.select_one("h2")
        or soup.select_one("title")
    )

    if title_el:
        title = clean_text(
            title_el.get_text(" ", strip=True)
        )

    profile_blocks = []

    # เก็บ text จาก profile section
    selectors = [
        ".entry-content",
        ".post-content",
        ".profile-content",
        ".elementor-widget-container",
        ".ts-team-content",
        "article",
    ]

    for selector in selectors:
        for container in soup.select(selector):

            for br in container.select("br"):
                br.replace_with("\n")

            raw_text = container.get_text(
                "\n",
                strip=True
            )

            for line in raw_text.splitlines():
                line = clean_text(line)

                if not line:
                    continue

                if len(line) <= 2:
                    continue

                profile_blocks.append(line)

    # fallback
    if not profile_blocks:

        for selector in [
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
        ]:
            for el in soup.select(selector):

                text = clean_text(
                    el.get_text(" ", strip=True)
                )

                if not text:
                    continue

                if len(text) <= 2:
                    continue

                profile_blocks.append(text)

    seen = set()
    clean_blocks = []

    for text in profile_blocks:

        if text in seen:
            continue

        seen.add(text)

        clean_blocks.append(text)

    profile_detail = "\n".join(clean_blocks)

    image_url = ""

    image_el = soup.select_one("img")

    if image_el:
        image_url = image_el.get("src", "").strip()

    detail = {
        "instructor_url": instructor_url,
        "title": title,
        "profile_detail": profile_detail[:12000],
        "detail_image_url": image_url,
    }

    _INSTRUCTOR_DETAIL_CACHE["items"][instructor_url] = {
        "data": detail,
        "expired_at": now + 3600,
    }

    return detail