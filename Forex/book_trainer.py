# ==============================================================================
#  BOOK TRAINER — Ingest trading books, PDFs & text files into knowledge base
#
#  Extracts rules, patterns, and entry/exit criteria from text.
#  Provides a searchable knowledge store usable as extra features.
# ==============================================================================

import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path

log = logging.getLogger("ForexBot.BookTrainer")

# Knowledge base file location
KNOWLEDGE_DB = "models/knowledge_base.json"

# Patterns to extract trading rules from text
RULE_PATTERNS = [
    # Entry rules
    r"(?:buy|long)\s+when\s+([^.!?]{15,120})",
    r"(?:sell|short)\s+when\s+([^.!?]{15,120})",
    r"enter\s+(?:a\s+)?(?:long|short|buy|sell|trade)\s+(?:when|if|after|as)\s+([^.!?]{15,120})",
    r"(?:open|place)\s+(?:a\s+)?(?:buy|sell|long|short)\s+(?:order|trade|position)\s+(?:when|if)\s+([^.!?]{10,120})",

    # Exit rules
    r"(?:exit|close)\s+(?:the\s+)?(?:trade|position)\s+(?:when|if|once|after)\s+([^.!?]{10,120})",
    r"take\s+profit\s+(?:when|at|if)\s+([^.!?]{10,100})",
    r"stop\s+loss\s+(?:should\s+be\s+)?(?:placed\s+)?(?:at|below|above)\s+([^.!?]{10,100})",

    # Indicator rules
    r"(?:when|if)\s+(?:the\s+)?(?:rsi|macd|ema|sma|adx|stochastic|bollinger|atr|cci)\s+([^.!?]{10,120})",
    r"(?:rsi|macd|ema|sma|adx|stochastic|atr|cci)\s+(?:crosses?|above|below|diverges?)\s+([^.!?]{5,100})",

    # Risk rules
    r"(?:risk|never\s+risk)\s+(?:more\s+than\s+)?(\d[\d.]*\s*%\s*[^.!?]{0,80})",
    r"reward.to.risk\s+(?:ratio|of)\s+([^.!?]{5,60})",
    r"position\s+size\s+(?:should\s+be\s+)?([^.!?]{10,100})",
]

# Category keywords for classifying extracted rules
RULE_CATEGORIES = {
    "entry":      ["buy", "sell", "enter", "long", "short", "open", "signal"],
    "exit":       ["exit", "close", "profit", "stop", "loss", "target"],
    "indicator":  ["rsi", "macd", "ema", "sma", "adx", "stochastic", "bollinger",
                   "atr", "cci", "ichimoku", "fibonacci", "volume", "momentum"],
    "risk":       ["risk", "reward", "ratio", "drawdown", "position size", "lot", "leverage"],
    "pattern":    ["head and shoulders", "double top", "double bottom", "triangle",
                   "flag", "breakout", "support", "resistance", "trend line"],
    "fundamental":["interest rate", "cpi", "gdp", "nfp", "inflation", "central bank",
                   "fomc", "ecb", "boe", "boj", "news", "event"],
}


class TradingRule:
    """A single trading rule extracted from source material."""

    def __init__(self, text: str, source: str, category: str = "general", confidence: float = 0.5):
        self.text = text.strip()
        self.source = source
        self.category = category
        self.confidence = confidence
        self.id = hashlib.md5(text.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "text":       self.text,
            "source":     self.source,
            "category":   self.category,
            "confidence": self.confidence,
        }


class KnowledgeBase:
    """
    Persistent JSON store for trading rules and concepts.
    Supports search and retrieval by category or keyword.
    """

    def __init__(self, db_path: str = KNOWLEDGE_DB):
        self.db_path = db_path
        self.rules: List[TradingRule] = []
        self._ingested_files: Dict[str, str] = {}  # path → mtime hash
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.rules = [
                    TradingRule(
                        r["text"], r["source"], r.get("category", "general"),
                        r.get("confidence", 0.5)
                    )
                    for r in data.get("rules", [])
                ]
                self._ingested_files = data.get("ingested_files", {})
                log.info(f"[Knowledge] Loaded {len(self.rules)} rules from {self.db_path}")
            except Exception as exc:
                log.warning(f"[Knowledge] Could not load DB: {exc}")
                self.rules = []

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "rules":          [r.to_dict() for r in self.rules],
            "ingested_files": self._ingested_files,
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"[Knowledge] Saved {len(self.rules)} rules → {self.db_path}")

    def add(self, rule: TradingRule):
        # Deduplicate by id
        existing_ids = {r.id for r in self.rules}
        if rule.id not in existing_ids:
            self.rules.append(rule)

    def search(self, query: str, category: Optional[str] = None, top_n: int = 10) -> List[TradingRule]:
        """Return the top N most relevant rules for a query string."""
        q = query.lower()
        scored: List[Tuple[float, TradingRule]] = []

        for rule in self.rules:
            if category and rule.category != category:
                continue
            text = rule.text.lower()
            # Simple keyword overlap score
            q_words = set(q.split())
            t_words = set(text.split())
            overlap = len(q_words & t_words)
            if overlap > 0:
                score = overlap / (len(q_words) + 1) * rule.confidence
                scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_n]]

    def get_by_category(self, category: str) -> List[TradingRule]:
        return [r for r in self.rules if r.category == category]

    def stats(self) -> Dict:
        cats: Dict[str, int] = {}
        for r in self.rules:
            cats[r.category] = cats.get(r.category, 0) + 1
        return {"total": len(self.rules), "by_category": cats}

    def sentiment_for_indicator(self, indicator: str) -> float:
        """
        Search knowledge base for rules about an indicator and return
        a net signal bias: +1 = bullish context, -1 = bearish, 0 = neutral.
        Used to adjust ML model confidence via knowledge-based overlay.
        """
        rules = self.search(indicator, top_n=20)
        if not rules:
            return 0.0

        bull_kw = {"buy", "long", "bullish", "above", "rising", "upward", "positive"}
        bear_kw = {"sell", "short", "bearish", "below", "falling", "downward", "negative"}

        score = 0.0
        for rule in rules:
            words = set(rule.text.lower().split())
            bull_hits = len(words & bull_kw)
            bear_hits = len(words & bear_kw)
            score += (bull_hits - bear_hits) * rule.confidence

        return max(-1.0, min(1.0, score / len(rules)))


class BookTrainer:
    """
    Ingests PDF files, text files, and raw strings containing trading knowledge.
    Extracts rules and populates the KnowledgeBase.

    Usage:
        trainer = BookTrainer()
        trainer.ingest_pdf("path/to/trading_book.pdf")
        trainer.ingest_text("path/to/notes.txt")
        trainer.ingest_string("Buy when RSI crosses above 30 from oversold territory.")
        trainer.save()
    """

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.kb = knowledge_base or KnowledgeBase()
        self._compiled = [re.compile(p, re.IGNORECASE) for p in RULE_PATTERNS]

    # ------------------------------------------------------------------
    # Public ingestion methods
    # ------------------------------------------------------------------

    def ingest_pdf(self, path: str) -> int:
        """Extract text from PDF and parse trading rules. Returns rules added."""
        if not os.path.exists(path):
            log.error(f"[BookTrainer] PDF not found: {path}")
            return 0

        source = Path(path).stem
        text = self._read_pdf(path)
        if not text:
            log.warning(f"[BookTrainer] Empty PDF: {path}")
            return 0

        count = self._parse_and_add(text, source)
        log.info(f"[BookTrainer] {path} → extracted {count} rules.")
        return count

    def ingest_text(self, path: str) -> int:
        """Read a plain text file and parse trading rules. Returns rules added."""
        if not os.path.exists(path):
            log.error(f"[BookTrainer] File not found: {path}")
            return 0

        source = Path(path).stem
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as exc:
            log.error(f"[BookTrainer] Read error: {exc}")
            return 0

        count = self._parse_and_add(text, source)
        log.info(f"[BookTrainer] {path} → extracted {count} rules.")
        return count

    def ingest_string(self, text: str, source: str = "manual") -> int:
        """Parse trading rules from a raw string. Returns rules added."""
        count = self._parse_and_add(text, source)
        log.info(f"[BookTrainer] String ({source}) → extracted {count} rules.")
        return count

    def ingest_folder(self, folder: str) -> int:
        """
        Recursively ingest all PDF and .txt files from a folder.
        Skips files already ingested (tracked by last-modified time).
        Returns total NEW rules added.
        """
        if not os.path.isdir(folder):
            log.error(f"[BookTrainer] Folder not found: {folder}")
            return 0

        total = 0
        for root, _, files in os.walk(folder):
            for fname in files:
                fpath = os.path.join(root, fname)
                ext   = fname.lower().rsplit(".", 1)[-1]
                if ext not in ("pdf", "txt", "md"):
                    continue

                # Skip if already ingested — compare last-modified timestamp
                try:
                    mtime_key = str(os.path.getmtime(fpath))
                except OSError:
                    mtime_key = "?"

                abs_path = os.path.abspath(fpath)
                if self.kb._ingested_files.get(abs_path) == mtime_key:
                    log.debug(f"[BookTrainer] Already ingested, skipping: {fname}")
                    continue

                added = self.ingest_pdf(fpath) if ext == "pdf" else self.ingest_text(fpath)
                if added >= 0:
                    self.kb._ingested_files[abs_path] = mtime_key
                total += added

        log.info(f"[BookTrainer] Folder '{folder}' → {total} new rules added.")
        return total

    def save(self):
        """Persist knowledge base to disk."""
        self.kb.save()

    def report(self) -> str:
        """Return a text summary of the knowledge base."""
        stats = self.kb.stats()
        lines = [
            "=" * 60,
            "  KNOWLEDGE BASE REPORT",
            f"  Total rules: {stats['total']}",
            "  By category:",
        ]
        for cat, n in sorted(stats["by_category"].items()):
            lines.append(f"    {cat:15s}: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_pdf(self, path: str) -> str:
        """Extract text from PDF using pdfplumber, fallback to PyPDF2."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            return "\n".join(text_parts)
        except Exception:
            pass

        try:
            import PyPDF2
            text_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            return "\n".join(text_parts)
        except Exception as exc:
            log.error(f"[BookTrainer] PDF read error: {exc}")
            return ""

    def _parse_and_add(self, text: str, source: str) -> int:
        """Extract rules using regex patterns and add to knowledge base."""
        added = 0
        # Split into sentences for pattern matching
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            # Try all rule patterns
            for pattern in self._compiled:
                match = pattern.search(sentence)
                if match:
                    rule_text = sentence[:300]  # Cap length
                    category  = self._classify(rule_text)
                    confidence = self._confidence(rule_text)
                    rule = TradingRule(rule_text, source, category, confidence)
                    self.kb.add(rule)
                    added += 1
                    break  # One rule per sentence

        return added

    def _classify(self, text: str) -> str:
        """Determine the category of a rule based on keyword presence."""
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for cat, keywords in RULE_CATEGORIES.items():
            scores[cat] = sum(1 for kw in keywords if kw in text_lower)

        best = max(scores, key=lambda c: scores[c])
        return best if scores[best] > 0 else "general"

    def _confidence(self, text: str) -> float:
        """
        Assign a confidence score based on specificity of the rule.
        Higher confidence for rules with specific numbers, indicators, or clear actions.
        """
        score = 0.3  # base
        if re.search(r"\d", text):
            score += 0.2    # has numbers → more specific
        if re.search(r"(rsi|macd|ema|sma|adx|stochastic|atr|cci|bollinger)", text, re.I):
            score += 0.2    # mentions specific indicator
        if re.search(r"(buy|sell|long|short|entry|exit)", text, re.I):
            score += 0.15   # clear action
        if re.search(r"(when|if|once|after)\s", text, re.I):
            score += 0.15   # conditional logic
        return min(1.0, score)


# ---------------------------------------------------------------------------
# Built-in trading knowledge (Classic rules pre-loaded into KB)
# ---------------------------------------------------------------------------

CLASSIC_TRADING_RULES = """
Buy when RSI crosses above 30 after being in oversold territory below 30.
Sell when RSI crosses below 70 after being in overbought territory above 70.
RSI divergence: if price makes a lower low but RSI makes a higher low, it signals a potential bullish reversal.
RSI divergence: if price makes a higher high but RSI makes a lower high, it signals a potential bearish reversal.

Enter a long trade when MACD line crosses above the signal line and both are below zero.
Enter a short trade when MACD line crosses below the signal line and both are above zero.
MACD histogram turning from negative to positive confirms upward momentum.

Buy when price crosses above the 50 EMA with increasing volume.
Sell when price crosses below the 50 EMA with increasing volume.
When price is above both the 21 EMA and 50 EMA, the trend is bullish.
When price is below both the 21 EMA and 50 EMA, the trend is bearish.

Buy when price bounces off the lower Bollinger Band with bullish candlestick confirmation.
Sell when price touches the upper Bollinger Band after a strong rally.
Bollinger Band squeeze indicates low volatility and potential breakout.

Buy when ADX is above 25 and the +DI line is above the -DI line.
Sell short when ADX is above 25 and the -DI line is above the +DI line.
When ADX is below 20 the market is ranging and trend-following strategies should be avoided.

Buy when stochastic %K crosses above %D from below 20 oversold zone.
Sell when stochastic %K crosses below %D from above 80 overbought zone.

Enter a long trade after a double bottom pattern forms with strong volume on the second bottom.
Enter a short trade after a double top pattern forms with declining volume.
Head and shoulders pattern signals a bearish reversal; enter short on neckline break.
Inverse head and shoulders signals bullish reversal; enter long on neckline breakout.

Buy on a flag or pennant breakout above the upper channel boundary after a strong uptrend.
Sell on a bear flag breakdown below the lower channel boundary.

Enter a long position on a bullish engulfing candle after a downtrend.
Enter a short position on a bearish engulfing candle after an uptrend.
Doji candles signal indecision; wait for confirmation before entering a trade.
Hammer and shooting star patterns signal potential reversals at support and resistance levels.

Never risk more than 1-2% of account balance on any single trade.
Reward to risk ratio should be at least 2:1 before entering any trade.
Set stop loss below the most recent swing low for long trades.
Set stop loss above the most recent swing high for short trades.
Take profit at the next major resistance level for long trades.
Take profit at the next major support level for short trades.

Avoid trading during major news events such as NFP, FOMC rate decisions, and CPI releases.
Wait at least 30 minutes after high-impact news before entering a trade.
Interest rate decisions by central banks are the single most impactful event for currency pairs.
When the Fed raises interest rates the USD typically strengthens against most currency pairs.
When the ECB cuts interest rates the EUR typically weakens against major pairs.

Trade in the direction of the higher timeframe trend.
If the daily chart shows an uptrend, only take long trades on the H4 and H1 charts.
The London session open between 8am and 10am GMT is the highest volume period for EUR and GBP pairs.
The New York session overlap with London between 1pm and 4pm GMT is the most volatile period.
Avoid trading during the Asian session for EUR, GBP, and other European pairs as spreads are wider.

Support turns into resistance and resistance turns into support once broken.
A breakout with high volume above a resistance level is a strong bullish signal.
A breakdown with high volume below a support level is a strong bearish signal.
False breakouts or stop hunts often occur just beyond obvious support and resistance levels.

Fibonacci retracements of 38.2%, 50%, and 61.8% are common reversal zones in a trending market.
The 61.8% Fibonacci level is the golden ratio and the most significant retracement level.
Enter long trades at the 50% or 61.8% Fibonacci retracement in a strong uptrend.

Ichimoku cloud: buy when price is above the cloud and the Tenkan-sen is above the Kijun-sen.
Ichimoku cloud: sell when price is below the cloud and the Tenkan-sen is below the Kijun-sen.
When price is inside the Ichimoku cloud, the trend is unclear and trading should be avoided.
"""


def load_classic_rules(kb: Optional[KnowledgeBase] = None) -> KnowledgeBase:
    """Pre-load the classic trading rules into the knowledge base."""
    if kb is None:
        kb = KnowledgeBase()

    trainer = BookTrainer(kb)
    count = trainer.ingest_string(CLASSIC_TRADING_RULES, source="classic_rules")
    trainer.save()
    log.info(f"[BookTrainer] Pre-loaded {count} classic trading rules.")
    return kb


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    # Load classic rules and demo search
    kb = load_classic_rules()
    trainer = BookTrainer(kb)

    print(trainer.report())

    print("\nSearch: 'rsi oversold'")
    for rule in kb.search("rsi oversold", top_n=5):
        print(f"  [{rule.category}] {rule.text[:80]}")

    print("\nSearch: 'stop loss'")
    for rule in kb.search("stop loss", category="risk", top_n=5):
        print(f"  [{rule.category}] {rule.text[:80]}")
