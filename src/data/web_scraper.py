"""Scrape Lex Fridman Podcast transcripts from lexfridman.com."""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config.settings import (
    LEX_PODCAST_URL,
    RAW_SCRAPED_DIR,
    SCRAPE_DELAY_SECONDS,
    SCRAPE_MAX_RETRIES,
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


def _fetch_with_retry(url: str) -> requests.Response | None:
    """Fetch a URL with retries and polite delay."""
    for attempt in range(SCRAPE_MAX_RETRIES):
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt < SCRAPE_MAX_RETRIES - 1:
                wait = SCRAPE_DELAY_SECONDS * (attempt + 1)
                print(f"  Retry {attempt + 1}/{SCRAPE_MAX_RETRIES} for {url}: {e}")
                time.sleep(wait)
            else:
                print(f"  Failed to fetch {url}: {e}")
                return None


def _cache_path(slug: str, suffix: str = "") -> Path:
    """Path for cached raw HTML."""
    return RAW_SCRAPED_DIR / f"{slug}{suffix}.html"


def get_episode_urls() -> list[dict]:
    """
    Scrape the podcast listing page to find all episode page URLs.

    The listing page has links like lexfridman.com/guest-slug with
    companion transcript links at lexfridman.com/guest-slug-transcript.

    Returns:
        List of dicts with keys: slug, episode_url, transcript_url
    """
    print(f"Fetching episode list from {LEX_PODCAST_URL}...")
    resp = _fetch_with_retry(LEX_PODCAST_URL)
    if resp is None:
        raise RuntimeError(f"Failed to fetch podcast listing from {LEX_PODCAST_URL}")

    soup = BeautifulSoup(resp.text, "lxml")

    # Collect all lexfridman.com links
    all_hrefs = set()
    for link in soup.find_all("a", href=True):
        href = link["href"].strip().rstrip("/")
        if href.startswith("/"):
            href = f"https://lexfridman.com{href}"
        all_hrefs.add(href)

    # Find transcript URLs — these end with -transcript
    transcript_urls = set()
    for href in all_hrefs:
        if href.endswith("-transcript") and "lexfridman.com/" in href:
            transcript_urls.add(href)

    # Derive episode URLs and slugs from transcript URLs
    # This is the most reliable signal — if there's a transcript link, it's an episode
    episodes = []
    skip_slugs = {"podcast", "about", "contact", "books", "sponsors", "guest",
                  "favorite", "feed", "category", "page", "tag"}

    for t_url in transcript_urls:
        # e.g. https://lexfridman.com/rick-beato-transcript -> slug=rick-beato
        slug = t_url.split("lexfridman.com/")[-1].replace("-transcript", "")
        if not slug or slug in skip_slugs:
            continue

        episode_url = f"https://lexfridman.com/{slug}"
        episodes.append({
            "slug": slug,
            "episode_url": episode_url,
            "transcript_url": t_url,
        })

    # Also check for episode URLs that might not have transcript links on the listing
    for href in all_hrefs:
        if "lexfridman.com/" not in href:
            continue
        if href.endswith("-transcript"):
            continue
        # Skip known non-episode paths
        slug = href.split("lexfridman.com/")[-1].strip("/")
        if not slug or slug in skip_slugs or "/" in slug:
            continue
        if any(href.endswith(ext) for ext in [".xml", ".mp3", ".png", ".jpg"]):
            continue
        # Skip if already found via transcript
        if any(e["slug"] == slug for e in episodes):
            continue
        # This might be an episode without a transcript link on the listing page
        episodes.append({
            "slug": slug,
            "episode_url": href,
            "transcript_url": f"{href}-transcript",
        })

    episodes.sort(key=lambda x: x["slug"])
    print(f"Found {len(episodes)} potential episode URLs")
    return episodes


def _extract_episode_info_from_page(html: str) -> dict:
    """Extract episode number, title, guest, and date from an episode page."""
    soup = BeautifulSoup(html, "lxml")
    info = {"episode_id": None, "title": None, "guest": None, "date": None}

    # Episode number: look for "#NNN" pattern in headings or title
    title_el = soup.find("title")
    page_title = title_el.get_text(strip=True) if title_el else ""

    # Try og:title or page title — format: "Guest: Topic | Lex Fridman Podcast #492"
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") or meta.get("name", "")
        if prop in ("og:title", "twitter:title"):
            page_title = meta.get("content", page_title)
            break

    # Extract episode number from title
    ep_match = re.search(r"#(\d+)", page_title)
    if ep_match:
        info["episode_id"] = int(ep_match.group(1))

    # Extract title (part before " | Lex Fridman")
    if "|" in page_title:
        info["title"] = page_title.split("|")[0].strip()
    else:
        info["title"] = page_title.strip()

    # Extract guest from title (part before ":")
    title_text = info["title"] or ""
    if ":" in title_text:
        info["guest"] = title_text.split(":")[0].strip()
    else:
        info["guest"] = title_text.strip()

    # Date from meta tags or time element
    time_el = soup.find("time")
    if time_el:
        info["date"] = time_el.get("datetime", time_el.get_text(strip=True))
    else:
        for meta in soup.find_all("meta"):
            prop = meta.get("property", "") or meta.get("name", "")
            if "date" in prop.lower() or "published" in prop.lower():
                info["date"] = meta.get("content")
                break

    return info


def _extract_transcript_from_transcript_page(html: str) -> str | None:
    """
    Extract transcript text from a dedicated transcript page.

    Transcript format:
    - Speaker labels like "Lex Fridman (00:00:00)" with timestamp links
    - Chapter headings as h2/h3 with dashes underneath
    - Paragraphs of conversation
    """
    soup = BeautifulSoup(html, "lxml")

    # Find the main content area
    content_div = soup.find("div", class_=re.compile(
        r"entry-content|post-content|article-content"
    ))
    if not content_div:
        content_div = soup.find("article") or soup

    # Extract all paragraphs, stripping timestamp links but keeping speaker labels
    transcript_parts = []
    for element in content_div.find_all(["p", "h2", "h3"]):
        text = element.get_text(strip=True)
        if not text:
            continue

        # Clean up timestamp patterns like "(00:00:00)" that remain after stripping links
        text = re.sub(r"\(?\d{1,2}:\d{2}:\d{2}\)?", "", text).strip()

        # Skip very short lines (nav elements, etc.)
        if len(text) < 5:
            continue

        # Skip table of contents entries (lines that are just timestamps)
        if re.match(r"^[\d:]+$", text):
            continue

        transcript_parts.append(text)

    if not transcript_parts:
        return None

    transcript = "\n\n".join(transcript_parts)

    # Only return if we got substantial content
    if len(transcript) < 500:
        return None

    return transcript


def scrape_episode(episode_info: dict) -> dict | None:
    """
    Scrape a single episode: fetch episode page for metadata, then transcript page.

    Uses cached HTML for both pages if available.
    """
    slug = episode_info["slug"]
    episode_url = episode_info["episode_url"]
    transcript_url = episode_info["transcript_url"]

    # Fetch episode page (for metadata: episode number, guest, date)
    ep_cache = _cache_path(slug, "_episode")
    if ep_cache.exists():
        ep_html = ep_cache.read_text(encoding="utf-8")
    else:
        time.sleep(SCRAPE_DELAY_SECONDS)
        resp = _fetch_with_retry(episode_url)
        if resp is None:
            return None
        ep_html = resp.text
        ep_cache.parent.mkdir(parents=True, exist_ok=True)
        ep_cache.write_text(ep_html, encoding="utf-8")

    info = _extract_episode_info_from_page(ep_html)
    if info["episode_id"] is None:
        # Can't use this episode without an ID
        return None

    # Fetch transcript page
    tr_cache = _cache_path(slug, "_transcript")
    if tr_cache.exists():
        tr_html = tr_cache.read_text(encoding="utf-8")
    else:
        time.sleep(SCRAPE_DELAY_SECONDS)
        resp = _fetch_with_retry(transcript_url)
        if resp is None:
            return None
        tr_html = resp.text
        tr_cache.parent.mkdir(parents=True, exist_ok=True)
        tr_cache.write_text(tr_html, encoding="utf-8")

    transcript = _extract_transcript_from_transcript_page(tr_html)
    if transcript is None:
        return None

    return {
        "episode_id": info["episode_id"],
        "title": info["title"] or slug.replace("-", " ").title(),
        "guest": info["guest"] or slug.replace("-", " ").title(),
        "date": info["date"],
        "full_transcript": transcript,
        "source": "scraped",
        "transcript_length": len(transcript),
    }


def scrape_all_transcripts() -> pd.DataFrame:
    """
    Scrape all available podcast transcripts from lexfridman.com.

    Returns:
        DataFrame with columns: episode_id, title, guest, date,
        full_transcript, source, transcript_length
    """
    episode_urls = get_episode_urls()
    episodes = []

    for ep_info in tqdm(episode_urls, desc="Scraping episodes"):
        result = scrape_episode(ep_info)
        if result is not None:
            episodes.append(result)

    result = pd.DataFrame(episodes)
    if not result.empty:
        # Deduplicate by episode_id (keep first occurrence)
        result = result.drop_duplicates(subset="episode_id", keep="first")
        result = result.sort_values("episode_id").reset_index(drop=True)
    print(f"Successfully scraped {len(result)} episodes")
    return result
