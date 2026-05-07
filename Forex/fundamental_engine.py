# ==============================================================================
#  FUNDAMENTAL ENGINE — Macro economic data, interest rates, central bank stances
#
#  Sources: yfinance (free, no API key needed)
#  Data: interest rate differentials, economic indicators, multi-pair quotes
# ==============================================================================

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("ForexBot.Fundamental")

# ---------------------------------------------------------------------------
# Central bank interest rate stances (manually maintained / fetched)
# Key rates: updated from public sources
# ---------------------------------------------------------------------------

# Approximate key policy rates — kept as fallback when live data unavailable
# Updated: April 2026
_FALLBACK_RATES = {
    "USD": 4.25,   # Federal Reserve (cut from 5.25 through 2025)
    "EUR": 2.50,   # ECB (cut from 4.00 through 2025)
    "GBP": 4.50,   # Bank of England
    "JPY": 0.50,   # Bank of Japan (raised from negative)
    "CHF": 0.25,   # SNB
    "AUD": 4.10,   # RBA
    "CAD": 2.75,   # Bank of Canada (cut aggressively)
    "NZD": 3.50,   # RBNZ
}

# Pairs to track in multi-pair mode (28 majors/minors + gold)
ALL_PAIRS = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    # Crosses (EUR)
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    # Crosses (GBP)
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    # Crosses (JPY)
    "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
    # Crosses (AUD/NZD/CAD)
    "AUDCAD", "AUDCHF", "AUDNZD",
    "CADCHF", "NZDCAD", "NZDCHF",
    # Commodities
    "XAUUSD",   # Gold vs USD
]

# Yahoo Finance ticker map — fx pairs and commodities
YF_PAIR_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",    "USDCAD": "CAD=X",    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X",
    "EURCHF": "EURCHF=X", "GBPCHF": "GBPCHF=X", "USDCNH": "USDCNH=X",
    # Commodities
    "XAUUSD": "GC=F",     # Gold vs USD (gold futures)
    "XAGUSD": "SI=F",     # Silver vs USD
}

# Economic data tickers from Yahoo Finance
ECONOMIC_TICKERS = {
    "DXY":     "DX-Y.NYB",    # US Dollar Index
    "GOLD":    "GC=F",         # Gold futures
    "OIL":     "CL=F",         # Crude oil futures
    "SP500":   "^GSPC",        # S&P 500
    "VIX":     "^VIX",         # Volatility Index
    "US10Y":   "^TNX",         # US 10-yr Treasury yield
    "DE10Y":   "^TECDAX",      # German 10yr (proxy)
}


class InterestRateDifferential:
    """
    Calculates interest rate differentials between base/quote currencies.
    A positive differential means the base currency earns higher yield → bullish.
    """

    def __init__(self):
        self._rates: Dict[str, float] = dict(_FALLBACK_RATES)
        self._last_update: float = 0.0

    def get_rate(self, currency: str) -> float:
        return self._rates.get(currency.upper(), 0.0)

    def differential(self, pair: str) -> float:
        """Return base - quote interest rate differential."""
        base  = pair[:3].upper()
        quote = pair[3:].upper()
        diff  = self.get_rate(base) - self.get_rate(quote)
        return round(diff, 3)

    def carry_bias(self, pair: str) -> int:
        """
        Return carry trade bias: +1 buy base, -1 sell base, 0 neutral.
        Carry trade: buy high-yield vs sell low-yield.
        """
        diff = self.differential(pair)
        if diff > 0.5:
            return 1
        if diff < -0.5:
            return -1
        return 0

    def rates_table(self) -> str:
        lines = ["Central Bank Interest Rates:", "-" * 30]
        for ccy, rate in sorted(self._rates.items()):
            lines.append(f"  {ccy}: {rate:+.2f}%")
        return "\n".join(lines)


class MultiPairQuotes:
    """
    Fetches live and historical price data for all major forex pairs
    using yfinance (free, no API key required).
    Falls back gracefully when offline.
    """

    def __init__(self, pairs: Optional[List[str]] = None, cache_ttl: int = 60):
        self.pairs = pairs or ALL_PAIRS
        self.cache_ttl = cache_ttl
        self._quote_cache: Dict[str, float] = {}
        self._cache_time: float = 0.0

    def get_quotes(self, force: bool = False) -> Dict[str, float]:
        """
        Return {pair: current_price} for all configured pairs.
        Uses yfinance for live data.
        """
        if not force and self._quote_cache and (time.time() - self._cache_time) < self.cache_ttl:
            return self._quote_cache

        try:
            import yfinance as yf
        except ImportError:
            log.warning("[Fundamental] yfinance not installed. Cannot fetch live quotes.")
            return {}

        tickers = [YF_PAIR_MAP.get(p, f"{p}=X") for p in self.pairs]
        quotes: Dict[str, float] = {}

        try:
            data = yf.download(
                tickers,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=True,
            )
            if "Close" in data.columns:
                last_close = data["Close"].iloc[-1]
                for pair in self.pairs:
                    ticker = YF_PAIR_MAP.get(pair, f"{pair}=X")
                    try:
                        val = float(last_close[ticker])
                        if not np.isnan(val):
                            quotes[pair] = round(val, 5)
                    except (KeyError, TypeError):
                        pass
        except Exception as exc:
            log.debug(f"[Fundamental] Quote fetch error: {exc}")

        if quotes:
            self._quote_cache = quotes
            self._cache_time  = time.time()
            log.info(f"[Fundamental] Fetched {len(quotes)} live quotes.")
        else:
            log.debug("[Fundamental] No live quotes retrieved — using cache.")

        return self._quote_cache or {}

    def get_ohlcv(
        self,
        pair: str,
        period: str = "3mo",
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data for a single pair via yfinance.
        Useful as a data source when MT5 is not connected.
        """
        try:
            import yfinance as yf
        except ImportError:
            log.warning("[Fundamental] yfinance not installed.")
            return pd.DataFrame()

        ticker = YF_PAIR_MAP.get(pair.upper(), f"{pair}=X")
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return pd.DataFrame()

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "time"
            log.info(f"[Fundamental] yfinance {pair} {interval}: {len(df)} bars")
            return df
        except Exception as exc:
            log.warning(f"[Fundamental] yfinance {pair} error: {exc}")
            return pd.DataFrame()

    def quotes_table(self) -> str:
        """Return a formatted table of current quotes."""
        quotes = self.get_quotes()
        if not quotes:
            return "No quotes available."

        lines = [f"{'PAIR':10s} {'PRICE':>10s}"]
        lines.append("-" * 22)
        for pair in self.pairs:
            if pair in quotes:
                lines.append(f"{pair:10s} {quotes[pair]:>10.5f}")
        return "\n".join(lines)


class MacroDataFetcher:
    """
    Fetches macro economic indicators via yfinance.
    Provides correlation context for trading decisions.
    """

    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_times: Dict[str, float] = {}

    def get(self, ticker_key: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """Fetch a macro indicator time series."""
        cache_key = f"{ticker_key}_{period}"
        if cache_key in self._cache:
            age = time.time() - self._cache_times.get(cache_key, 0)
            if age < self.cache_ttl:
                return self._cache[cache_key]

        try:
            import yfinance as yf
            ticker = ECONOMIC_TICKERS.get(ticker_key, ticker_key)
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                self._cache[cache_key] = df
                self._cache_times[cache_key] = time.time()
                return df
        except Exception as exc:
            log.debug(f"[Fundamental] MacroData {ticker_key} error: {exc}")
        return None

    def dxy_trend(self) -> int:
        """
        Returns DXY (Dollar Index) trend: +1 = rising, -1 = falling, 0 = flat.
        A rising DXY generally correlates with USD strength (sell EUR/GBP/AUD).
        """
        df = self.get("DXY", period="1mo")
        if df is None or len(df) < 5:
            return 0
        close = df["Close"].values
        # Simple 5-period vs 20-period SMA comparison
        sma5  = close[-5:].mean()
        sma20 = close[-20:].mean() if len(close) >= 20 else close.mean()
        if sma5 > sma20 * 1.001:
            return 1
        if sma5 < sma20 * 0.999:
            return -1
        return 0

    def risk_sentiment(self) -> str:
        """
        Gauge market risk sentiment from VIX and S&P 500.
        Returns: "risk-on", "risk-off", or "neutral"
        """
        vix_df = self.get("VIX", period="5d")
        if vix_df is not None and not vix_df.empty:
            vix = float(vix_df["Close"].iloc[-1])
            if vix > 25:
                return "risk-off"
            if vix < 15:
                return "risk-on"
        return "neutral"


class FundamentalEngine:
    """
    High-level fundamental analysis facade.
    Combines interest rates, macro data, and multi-pair quotes
    into a trading signal overlay.
    """

    def __init__(self, pairs: Optional[List[str]] = None):
        self.ird    = InterestRateDifferential()
        self.quotes = MultiPairQuotes(pairs)
        self.macro  = MacroDataFetcher()
        self._last_full_update: float = 0.0

    def fundamental_bias(self, pair: str) -> Dict:
        """
        Return a composite fundamental bias for a currency pair.

        Returns:
            {
              "pair":         str,
              "carry_bias":   int,     # +1 buy, -1 sell, 0 neutral
              "ird":          float,   # interest rate differential
              "dxy_trend":    int,     # +1 USD rising, -1 falling
              "risk_mode":    str,     # "risk-on" / "risk-off" / "neutral"
              "composite":    float,   # -1 to +1 overall fundamental score
              "notes":        [str]
            }
        """
        pair = pair.upper()
        base  = pair[:3]
        quote = pair[3:]

        carry  = self.ird.carry_bias(pair)
        ird    = self.ird.differential(pair)
        dxy    = self.macro.dxy_trend()
        risk   = self.macro.risk_sentiment()

        notes: List[str] = []
        score = 0.0

        # Carry contribution
        score += carry * 0.4
        if carry != 0:
            notes.append(f"Carry: {'+' if carry > 0 else ''}{ird:.2f}% IRD → {'buy' if carry > 0 else 'sell'} {base}")

        # DXY contribution (affects USD pairs)
        if "USD" in (base, quote):
            usd_side = 1 if base == "USD" else -1
            score += dxy * usd_side * 0.3
            if dxy != 0:
                notes.append(f"DXY {'rising' if dxy > 0 else 'falling'} → {'bullish' if usd_side * dxy > 0 else 'bearish'} USD")

        # Risk sentiment contribution
        # Risk-on: AUD/NZD/CAD gain vs JPY/CHF
        risk_currencies = {
            "risk-on":  {"buy": {"AUD", "NZD", "CAD"}, "sell": {"JPY", "CHF"}},
            "risk-off": {"buy": {"JPY", "CHF"}, "sell": {"AUD", "NZD", "CAD"}},
        }
        if risk in risk_currencies:
            buy_set  = risk_currencies[risk]["buy"]
            sell_set = risk_currencies[risk]["sell"]
            if base in buy_set or quote in sell_set:
                score += 0.3
                notes.append(f"Risk-{risk[-2:]}: favours {base}/{quote} long")
            elif base in sell_set or quote in buy_set:
                score -= 0.3
                notes.append(f"Risk-{risk[-2:]}: favours {base}/{quote} short")

        composite = max(-1.0, min(1.0, score))

        return {
            "pair":       pair,
            "carry_bias": carry,
            "ird":        ird,
            "dxy_trend":  dxy,
            "risk_mode":  risk,
            "composite":  round(composite, 3),
            "notes":      notes,
        }

    def all_quotes(self) -> Dict[str, float]:
        """Return current prices for all configured pairs."""
        return self.quotes.get_quotes()

    def best_pairs(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Rank all configured pairs by absolute fundamental signal strength.
        Returns list of (pair, composite_score) sorted by |score| descending.
        """
        scores: List[Tuple[str, float]] = []
        for pair in self.quotes.pairs:
            try:
                bias = self.fundamental_bias(pair)
                scores.append((pair, bias["composite"]))
            except Exception:
                pass
        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return scores[:n]

    def summary_report(self, pairs: Optional[List[str]] = None) -> str:
        """Print a formatted fundamental analysis report."""
        pairs = pairs or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        lines = [
            "=" * 65,
            "  FUNDAMENTAL ANALYSIS REPORT",
            "=" * 65,
            self.ird.rates_table(),
            "-" * 65,
            f"  DXY Trend  : {'RISING' if self.macro.dxy_trend() > 0 else 'FALLING' if self.macro.dxy_trend() < 0 else 'FLAT'}",
            f"  Risk Mode  : {self.macro.risk_sentiment().upper()}",
            "-" * 65,
        ]
        for pair in pairs:
            b = self.fundamental_bias(pair)
            direction = "BUY" if b["composite"] > 0.15 else ("SELL" if b["composite"] < -0.15 else "NEUTRAL")
            lines.append(
                f"  {pair:8s} | {direction:7s} | score={b['composite']:+.2f} | "
                f"IRD={b['ird']:+.2f}% | carry={b['carry_bias']:+d}"
            )
            for note in b["notes"]:
                lines.append(f"    • {note}")
        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    engine = FundamentalEngine()

    print(engine.summary_report(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"]))

    print("\nAll pair quotes:")
    print(engine.quotes.quotes_table())

    print("\nTop 5 pairs by fundamental signal:")
    for pair, score in engine.best_pairs(5):
        print(f"  {pair}: {score:+.3f}")
