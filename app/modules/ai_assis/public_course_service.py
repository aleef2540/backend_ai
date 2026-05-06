import time
import re
import httpx
from bs4 import BeautifulSoup


_PUBLIC_COURSE_CACHE = {
    "items": None,
    "expired_at": 0,
}

_PUBLIC_COURSE_DETAIL_CACHE = {
    "items": {},
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


async def fetch_public_course_context() -> list[dict]:
    now = time.time()

    if (
        _PUBLIC_COURSE_CACHE["items"]
        and _PUBLIC_COURSE_CACHE["expired_at"] > now
    ):
        return _PUBLIC_COURSE_CACHE["items"]

    url = "https://www.entraining.net/public-course/plan/all/"

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

    courses = []
    current_month = ""

    container = soup.select_one("#myROW_block") or soup

    for child in container.find_all("div", recursive=False):

        month_heading = child.select_one("h4")

        if month_heading:
            current_month = clean_text(month_heading.get_text(" ", strip=True))
            current_month = current_month.replace("เดือน", "").strip()
            continue

        card = child.select_one(".public-item")

        if not card:
            continue

        title_el = card.select_one("h3.course-name-th a")
        title_en_el = card.select_one("p.course-name-en")
        image_el = card.select_one("img")
        date_el = card.select_one("h5")

        register_el = card.select_one('a[href*="/public-course/register"]')
        brochure_el = card.select_one('a[href*="/public-course/download"]')

        price_el = None
        for div in card.select("div"):
            text = clean_text(div.get_text(" ", strip=True))
            if "บาท" in text:
                price_el = div

        badge_el = child.select_one("span")

        course_name = clean_text(title_el.get_text(" ", strip=True)) if title_el else ""
        course_url = title_el.get("href", "").strip() if title_el else ""

        course_name_en = clean_text(title_en_el.get_text(" ", strip=True)) if title_en_el else ""
        course_name_en = course_name_en.strip("() ")

        image_url = image_el.get("src", "").strip() if image_el else ""

        course_date = clean_text(date_el.get_text(" ", strip=True)) if date_el else ""
        course_date = course_date.replace("วันที่", "").strip()

        register_url = register_el.get("href", "").strip() if register_el else ""
        brochure_url = brochure_el.get("href", "").strip() if brochure_el else ""

        badge = clean_text(badge_el.get_text(" ", strip=True)) if badge_el else ""

        price = ""
        if price_el:
            price_text = clean_text(price_el.get_text(" ", strip=True))
            price_match = re.search(r"[\d,]+\s*บาท", price_text)
            if price_match:
                price = price_match.group(0)

        view_count = ""
        download_count = ""

        stats_text = clean_text(card.get_text(" ", strip=True))
        numbers = re.findall(r"\b\d+\b", stats_text)

        if numbers:
            # ไม่ reliable 100% เพราะมี id/date/price ปนได้ ถ้าต้องการแม่นให้ใช้ DOM icon แยกเพิ่ม
            pass

        if not course_name:
            continue

        courses.append({
            "month": current_month,
            "course_name": course_name,
            "course_name_en": course_name_en,
            "course_url": course_url,
            "course_date": course_date,
            "price": price,
            "badge": badge,
            "image_url": image_url,
            "register_url": register_url,
            "brochure_url": brochure_url,
        })

    _PUBLIC_COURSE_CACHE["items"] = courses
    _PUBLIC_COURSE_CACHE["expired_at"] = now + 1800

    return courses

async def fetch_public_course_detail(course_url: str) -> dict:
    course_url = (course_url or "").strip()

    if not course_url:
        return {}

    now = time.time()

    cached = _PUBLIC_COURSE_DETAIL_CACHE["items"].get(course_url)

    if cached and cached["expired_at"] > now:
        return cached["data"]

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            course_url,
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

    title = ""

    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text(" ", strip=True))

    course_content = soup.select_one(".course-content")

    course_outline = ""

    if course_content:
        for img in course_content.select("img"):
            img.replace_with("\n")

        raw_text = course_content.get_text("\n", strip=True)

        lines = []

        for line in raw_text.splitlines():
            line = clean_text(line)

            if not line:
                continue

            if len(line) <= 1:
                continue

            lines.append(line)

        course_outline = "\n".join(lines)

    detail = {
        "course_url": course_url,
        "title": title,
        "course_outline": course_outline[:12000],
    }

    _PUBLIC_COURSE_DETAIL_CACHE["items"][course_url] = {
        "data": detail,
        "expired_at": now + 3600,
    }

    return detail