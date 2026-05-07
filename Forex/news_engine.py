# ==============================================================================
#  NEWS ENGINE — Real-time Forex news, RSS feeds & sentiment analysis
#  Feeds: ForexFactory, FXStreet, Reuters, Investing.com, DailyFX
# ==============================================================================

import logging
import time
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger("ForexBot.News")

# ---------------------------------------------------------------------------
# FinBERT — financial-domain transformer NLP
# Significantly outperforms VADER on financial text (earnings, macro news, etc.)
# Install: pip install transformers torch  (or transformers tensorflow)
# Model card: https://huggingface.co/ProsusAI/finbert
# ---------------------------------------------------------------------------
try:
    from transformers import pipeline as _hf_pipeline  # type: ignore
    _finbert_pipeline = _hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,          # CPU; set to 0 for GPU
        top_k=None,
        truncation=True,
        max_length=512,
    )
    FINBERT_AVAILABLE = True
    log.info("[News] FinBERT (ProsusAI/finbert) loaded — domain-specific financial NLP active.")
except Exception as _finbert_err:
    _finbert_pipeline = None
    FINBERT_AVAILABLE = False
    log.info(f"[News] FinBERT unavailable ({_finbert_err}) — VADER+forex-lexicon active.")

# ---------------------------------------------------------------------------
# RSS feed sources (public, no API key needed)
# ---------------------------------------------------------------------------
RSS_FEEDS = {
    "ForexFactory":   "https://www.forexfactory.com/ff_calendar_thisweek.xml",
    "FXStreet":       "https://www.fxstreet.com/rss/news",
    "Investing":      "https://www.investing.com/rss/news_25.rss",
    "DailyFX":        "https://www.dailyfx.com/feeds/all",
    "Reuters_Forex":  "https://feeds.Reuters.com/reuters/businessNews",
    "MarketWatch":    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "EconCalendar":   "https://www.fxstreet.com/rss/economic-calendar",
}

# Keywords mapping to currency pairs for relevance filtering
CURRENCY_KEYWORDS = {
    "USD": ["dollar", "fed", "federal reserve", "powell", "nonfarm", "cpi", "inflation",
            "gdp", "unemployment", "fomc", "interest rate", "us economy", "treasury"],
    "EUR": ["euro", "ecb", "european central bank", "lagarde", "eurozone", "germany",
            "france", "italy", "spain", "eu economy", "draghi"],
    "GBP": ["pound", "sterling", "boe", "bank of england", "bailey", "uk economy",
            "brexit", "british", "england", "scotland"],
    "JPY": ["yen", "boj", "bank of japan", "ueda", "kuroda", "japan", "japanese",
            "nikkei", "abenomics"],
    "CHF": ["franc", "snb", "swiss national bank", "switzerland", "swiss"],
    "AUD": ["aussie", "rba", "reserve bank australia", "australia", "australian",
            "commodities", "iron ore", "china trade"],
    "CAD": ["loonie", "boc", "bank of canada", "canada", "canadian", "oil price",
            "crude", "macklem"],
    "NZD": ["kiwi", "rbnz", "reserve bank new zealand", "new zealand"],
}

# High-impact economic events
HIGH_IMPACT_EVENTS = [
    "nonfarm payroll", "nfp", "fomc", "interest rate decision", "cpi", "gdp",
    "retail sales", "ism manufacturing", "ism services", "pce", "ppi",
    "unemployment rate", "ecb rate", "boe rate", "rba rate", "boj rate",
    "snb rate", "employment change", "trade balance", "current account",
    "central bank", "rate hike", "rate cut", "quantitative easing", "tapering",
]


class NewsItem:
    """Represents a single news article with sentiment scoring."""

    __slots__ = ("title", "summary", "url", "published", "source",
                 "sentiment", "currencies", "impact", "raw_score")

    def __init__(
        self,
        title: str,
        summary: str,
        url: str,
        published: datetime,
        source: str,
    ):
        self.title = title
        self.summary = summary
        self.url = url
        self.published = published
        self.source = source
        self.sentiment: float = 0.0      # -1 bearish … +1 bullish
        self.currencies: List[str] = []  # affected currency codes
        self.impact: str = "low"         # low / medium / high
        self.raw_score: Dict = {}

    def to_dict(self) -> dict:
        return {
            "title":      self.title,
            "source":     self.source,
            "published":  self.published.isoformat() if self.published else "",
            "sentiment":  round(self.sentiment, 4),
            "currencies": self.currencies,
            "impact":     self.impact,
            "url":        self.url,
        }


class SentimentScorer:
    """
    Financial text sentiment analyser.

    Primary:  FinBERT (ProsusAI/finbert) — purpose-trained on ~10,000 financial
              news sentences; significantly outperforms VADER on earnings,
              central bank communication, and macro news text.
    Fallback: VADER + 30-term forex-specific lexicon extensions.

    Both modes return a VADER-compatible dict:
        {"compound": float[-1,1], "pos": float, "neg": float, "neu": float}
    """

    # Forex-specific word boosts (used by VADER fallback only)
    FOREX_LEXICON = {
        # Bullish terms → positive
        "hawkish": 2.5, "rate hike": 2.0, "tightening": 1.5,
        "strong growth": 2.0, "beat expectations": 2.5,
        "better than expected": 2.0, "surging": 2.0, "rally": 1.5,
        "bullish": 2.0, "risk-on": 1.5, "upbeat": 1.5,
        "optimistic": 1.5, "recovery": 1.2,
        # Bearish terms → negative
        "dovish": -2.5, "rate cut": -2.0, "easing": -1.5,
        "recession": -2.5, "contraction": -2.0,
        "miss expectations": -2.5, "worse than expected": -2.0,
        "slump": -2.0, "bearish": -2.0, "risk-off": -1.5,
        "pessimistic": -1.5, "crisis": -2.5, "default": -3.0,
        "collapse": -3.0,
    }

    def __init__(self):
        if FINBERT_AVAILABLE and _finbert_pipeline is not None:
            self._mode = "finbert"
            log.debug("[SentimentScorer] Using FinBERT.")
        else:
            self._mode = "vader"
            self._vader = SentimentIntensityAnalyzer()
            self._vader.lexicon.update(self.FOREX_LEXICON)
            log.debug("[SentimentScorer] Using VADER with forex lexicon.")

    def score(self, text: str) -> Dict:
        """Return sentiment scores with compound in [-1, +1]."""
        if not text:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        if self._mode == "finbert":
            return self._score_finbert(text)
        return self._vader.polarity_scores(text.lower())

    def _score_finbert(self, text: str) -> Dict:
        """Run FinBERT inference; return VADER-compatible dict."""
        try:
            # FinBERT max is 512 tokens; truncate at char level as safe upper bound
            raw = _finbert_pipeline(text[:512])[0]  # list of {label, score}
            scores = {r["label"].lower(): r["score"] for r in raw}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)
            compound = round(pos - neg, 4)   # map to [-1, +1]
            return {"compound": compound, "pos": pos, "neg": neg, "neu": neu}
        except Exception as exc:
            log.debug(f"[FinBERT] inference error: {exc} — falling back to VADER")
            if not hasattr(self, "_vader"):
                self._vader = SentimentIntensityAnalyzer()
                self._vader.lexicon.update(self.FOREX_LEXICON)
            return self._vader.polarity_scores(text.lower())


class NewsEngine:
    """
    Fetches, parses and scores forex news from multiple RSS feeds.

    Usage:
        engine = NewsEngine()
        items = engine.fetch_all()
        score = engine.pair_sentiment("EURUSD")
    """

    def __init__(self, max_age_hours: int = 24, request_timeout: int = 10):
        self.max_age_hours = max_age_hours
        self.request_timeout = request_timeout
        self.scorer = SentimentScorer()
        self._cache: List[NewsItem] = []
        self._last_fetch: float = 0.0
        self._cache_ttl: int = 900  # 15 minutes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self, force: bool = False) -> List[NewsItem]:
        """Fetch all RSS feeds and return scored NewsItem list."""
        now = time.time()
        if not force and self._cache and (now - self._last_fetch) < self._cache_ttl:
            return self._cache

        items: List[NewsItem] = []
        for source, url in RSS_FEEDS.items():
            try:
                fetched = self._fetch_feed(source, url)
                items.extend(fetched)
                log.debug(f"[News] {source}: {len(fetched)} items")
            except Exception as exc:
                log.debug(f"[News] {source} error: {exc}")

        # Deduplicate by title
        seen: set = set()
        unique: List[NewsItem] = []
        for item in items:
            key = item.title[:80].lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)

        # Score and tag
        for item in unique:
            self._score_item(item)

        # Sort newest first
        unique.sort(
            key=lambda x: x.published if x.published else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        self._cache = unique
        self._last_fetch = now
        log.info(f"[News] Fetched {len(unique)} unique articles from {len(RSS_FEEDS)} feeds.")
        return unique

    def pair_sentiment(self, pair: str) -> Dict:
        """
        Return an aggregated sentiment dict for a currency pair.

        Returns:
            {
              "pair":     "EURUSD",
              "base_bias":  float,  # -1..+1  (base currency view)
              "quote_bias": float,  # -1..+1  (quote currency view)
              "net":        float,  # base_bias - quote_bias → trade direction
              "news_count": int,
              "high_impact": bool,
              "latest_headlines": [str, ...]
            }
        """
        articles = self.fetch_all()
        base  = pair[:3].upper()
        quote = pair[3:].upper()

        base_scores:  List[float] = []
        quote_scores: List[float] = []
        high_impact = False
        headlines: List[str] = []

        for item in articles:
            if base in item.currencies:
                base_scores.append(item.sentiment)
                if item.impact == "high":
                    high_impact = True
                headlines.append(f"[{base}] {item.title}")
            if quote in item.currencies:
                quote_scores.append(item.sentiment)
                if item.impact == "high":
                    high_impact = True
                if f"[{quote}] {item.title}" not in headlines:
                    headlines.append(f"[{quote}] {item.title}")

        base_bias  = float(sum(base_scores)  / len(base_scores))  if base_scores  else 0.0
        quote_bias = float(sum(quote_scores) / len(quote_scores)) if quote_scores else 0.0
        net        = round(base_bias - quote_bias, 4)

        return {
            "pair":             pair,
            "base_bias":        round(base_bias,  4),
            "quote_bias":       round(quote_bias, 4),
            "net":              net,
            "news_count":       len(base_scores) + len(quote_scores),
            "high_impact":      high_impact,
            "latest_headlines": headlines[:10],
        }

    def high_impact_pending(self, currencies: Optional[List[str]] = None) -> bool:
        """
        Return True if a high-impact event is detected for listed currencies.
        Used as a trading pause flag.
        """
        articles = self.fetch_all()
        for item in articles:
            if item.impact != "high":
                continue
            if currencies is None:
                return True
            if any(c in item.currencies for c in currencies):
                return True
        return False

    def summary_report(self, pairs: List[str]) -> str:
        """Print a human-readable sentiment summary for a list of pairs."""
        lines = ["=" * 60, "  NEWS SENTIMENT REPORT", "=" * 60]
        for pair in pairs:
            data = self.pair_sentiment(pair)
            direction = "BULLISH" if data["net"] > 0.1 else ("BEARISH" if data["net"] < -0.1 else "NEUTRAL")
            lines.append(
                f"  {pair:8s} | {direction:8s} | net={data['net']:+.3f} | "
                f"articles={data['news_count']} | high_impact={data['high_impact']}"
            )
            for headline in data["latest_headlines"][:3]:
                lines.append(f"    • {headline[:90]}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_feed(self, source: str, url: str) -> List[NewsItem]:
        """Parse a single RSS feed URL and return NewsItem list."""
        headers = {"User-Agent": "Mozilla/5.0 (ForexBot/2.0)"}
        try:
            resp = requests.get(url, timeout=self.request_timeout, headers=headers)
            feed = feedparser.parse(resp.text)
        except Exception:
            feed = feedparser.parse(url)  # fallback: let feedparser handle

        items: List[NewsItem] = []
        cutoff = datetime.now(timezone.utc).timestamp() - self.max_age_hours * 3600

        for entry in feed.entries:
            title   = self._clean(getattr(entry, "title", ""))
            summary = self._clean(getattr(entry, "summary", ""))
            url_    = getattr(entry, "link", "")
            pub     = self._parse_date(entry)

            if pub and pub.timestamp() < cutoff:
                continue  # skip old articles

            item = NewsItem(title, summary, url_, pub, source)
            items.append(item)

        return items

    def _score_item(self, item: NewsItem):
        """Fill sentiment, currencies, and impact fields."""
        text = f"{item.title} {item.summary}".lower()

        # Sentiment
        scores = self.scorer.score(text)
        item.raw_score = scores
        item.sentiment = round(scores["compound"], 4)

        # Currency tagging
        for currency, keywords in CURRENCY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                item.currencies.append(currency)

        # Impact detection
        if any(evt in text for evt in HIGH_IMPACT_EVENTS):
            item.impact = "high"
        elif item.sentiment != 0.0:
            item.impact = "medium"

    @staticmethod
    def _clean(text: str) -> str:
        """Strip HTML tags and normalise whitespace."""
        if not text:
            return ""
        soup = BeautifulSoup(text, "lxml")
        clean = soup.get_text(separator=" ")
        return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def _parse_date(entry) -> Optional[datetime]:
        """Parse published/updated timestamp from feed entry."""
        for field in ("published_parsed", "updated_parsed", "created_parsed"):
            val = getattr(entry, field, None)
            if val:
                try:
                    return datetime(*val[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Economic Calendar Scraper (ForexFactory public calendar)
# ---------------------------------------------------------------------------

class EconomicCalendar:
    """
    Scrapes upcoming high-impact events from ForexFactory calendar.
    Falls back to an empty list if unavailable.
    """

    FF_URL = "https://www.forexfactory.com/calendar"

    def __init__(self, request_timeout: int = 12):
        self.timeout = request_timeout
        self._events: List[Dict] = []
        self._last_fetch: float = 0.0
        self._ttl = 3600  # 1 hour

    def get_events(self, force: bool = False) -> List[Dict]:
        """Return list of upcoming economic events."""
        if not force and self._events and (time.time() - self._last_fetch) < self._ttl:
            return self._events

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            resp = requests.get(self.FF_URL, timeout=self.timeout, headers=headers)
            soup = BeautifulSoup(resp.text, "lxml")
            events = self._parse_ff(soup)
            self._events = events
            self._last_fetch = time.time()
            log.info(f"[Calendar] Loaded {len(events)} upcoming events.")
        except Exception as exc:
            log.debug(f"[Calendar] Could not fetch calendar: {exc}")
            self._events = []

        return self._events

    def has_high_impact_soon(self, within_hours: float = 2.0) -> bool:
        """Return True if a HIGH-impact event is within `within_hours`."""
        events = self.get_events()
        now = datetime.now(timezone.utc)
        for ev in events:
            if ev.get("impact") != "high":
                continue
            ev_time = ev.get("time")
            if not ev_time:
                continue
            delta_h = (ev_time - now).total_seconds() / 3600
            if -0.5 <= delta_h <= within_hours:  # allow 30 min past
                return True
        return False

    @staticmethod
    def _parse_ff(soup: BeautifulSoup) -> List[Dict]:
        """Parse ForexFactory calendar table rows."""
        events = []
        rows = soup.select("tr.calendar__row")
        current_date = None

        for row in rows:
            # Date header row
            date_el = row.select_one("td.calendar__date span")
            if date_el:
                try:
                    current_date = datetime.strptime(
                        date_el.get_text(strip=True), "%a%b %d"
                    ).replace(year=datetime.now().year, tzinfo=timezone.utc)
                except Exception:
                    pass

            # Event row
            impact_el = row.select_one("td.calendar__impact span")
            name_el   = row.select_one("td.calendar__event span.calendar__eventcontent")
            time_el   = row.select_one("td.calendar__time")
            curr_el   = row.select_one("td.calendar__currency")

            if not name_el:
                continue

            impact_class = impact_el.get("class", []) if impact_el else []
            if any("red" in c for c in impact_class):
                impact = "high"
            elif any("orange" in c for c in impact_class):
                impact = "medium"
            else:
                impact = "low"

            time_str = time_el.get_text(strip=True) if time_el else ""
            ev_time = current_date
            if current_date and time_str and ":" in time_str:
                try:
                    h, m = map(int, time_str.replace("am", "").replace("pm", "").split(":"))
                    if "pm" in time_str.lower() and h != 12:
                        h += 12
                    ev_time = current_date.replace(hour=h, minute=m)
                except Exception:
                    pass

            events.append({
                "name":     name_el.get_text(strip=True),
                "currency": curr_el.get_text(strip=True) if curr_el else "",
                "impact":   impact,
                "time":     ev_time,
            })

        return events


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    engine = NewsEngine()
    print(engine.summary_report(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]))

    cal = EconomicCalendar()
    events = cal.get_events()
    print(f"\nUpcoming events ({len(events)} total):")
    for ev in events[:10]:
        print(f"  [{ev['impact'].upper():6s}] {ev.get('currency','???'):4s} {ev['name'][:50]}")
