import time
import re
import httpx
from bs4 import BeautifulSoup


_INSTRUCTOR_CACHE = {
    "items": None,
    "expired_at": 0,
}


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

    url = "https://www.entraining.net/expert/"

    async with httpx.AsyncClient(timeout=20) as client:
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