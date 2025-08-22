import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Dict, Optional
import feedparser
from typing import Any




def _to_iso(published: Any) -> Optional[str]:
    if not published:
        return None
    try:
        return published.isoformat()
    except AttributeError:
        return None

def fetch_rss_headlines(feeds: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for url in feeds:
        d = feedparser.parse(url)
        for e in d.entries[:limit]:
            
            title = getattr(e, "title", "")
            link = getattr(e, "link", "")
            published = getattr(e, "published_parsed", None) or getattr(e, "published", None)
            src = getattr(d.feed, "title", "") or getattr(e, "source", "")

            summary_raw = getattr(e, "summary", None)
            if summary_raw:
                try:
                    summary = BeautifulSoup(summary_raw, "html.parser").get_text(" ", strip=True)
                except Exception:
                    summary = summary_raw
            else:
                summary = None

            items.append({
                "title": title,
                "url": link,
                "published": _to_iso(published),  # <<< 统一为 published
                "source": src,
                "summary": summary
            })

   
    seen, uniq = set(), []
    for it in items:
        k = it.get("url")
        if k and k not in seen:
            seen.add(k)
            uniq.append(it)
    uniq.sort(key=lambda x: x.get("published") or "", reverse=True)
    return uniq[:limit]