# ==============================================================================
#  TECHNICAL ENGINE — Comprehensive technical analysis
#
#  Adds 40+ indicators, multi-timeframe analysis, and chart pattern detection.
#  Designed as a drop-in replacement / extension of the original FeatureEngineer.
# ==============================================================================

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# ta library (already installed)
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator,
    IchimokuIndicator, PSARIndicator,
)
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, VolumeWeightedAveragePrice

log = logging.getLogger("ForexBot.Technical")


# ---------------------------------------------------------------------------
# Extended feature column list (replaces original 15-column FEATURE_COLS)
# ---------------------------------------------------------------------------
EXTENDED_FEATURE_COLS = [
    # Trend
    "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_9", "ema_12", "ema_21", "ema_26", "ema_50",
    "ema_cross_9_21",    # 1 = 9 EMA above 21 EMA
    "ema_cross_21_50",   # 1 = 21 EMA above 50 EMA
    "price_vs_sma50",    # close / SMA50 - 1
    "price_vs_sma200",   # close / SMA200 - 1
    "adx", "adx_pos", "adx_neg",
    "psar_bull",         # 1 if PSAR below price (bullish)

    # Momentum
    "rsi",
    "rsi_sma",           # 14-bar SMA of RSI
    "stoch_k", "stoch_d",
    "williams_r",
    "roc_5", "roc_10",
    "macd", "macd_signal", "macd_diff",
    "macd_above_zero",   # 1 if macd line above 0
    "cci",

    # Volatility
    "atr", "atr_norm",   # ATR / close
    "bb_width", "bb_pct",
    "kc_width",          # Keltner channel width

    # Volume
    "obv_norm",          # OBV normalised to rolling mean
    "mfi",               # Money Flow Index

    # Ichimoku
    "ichi_above_cloud",  # 1 if price above cloud
    "ichi_ten_above_kij",# 1 if Tenkan > Kijun

    # Candle patterns (binary flags)
    "doji",
    "bullish_engulf",
    "bearish_engulf",
    "hammer",
    "shooting_star",
    "morning_star",
    "evening_star",

    # Price action
    "ret_1", "ret_3", "ret_5", "ret_10",
    "close_norm",

    # Session flag (0=Asia, 1=London, 2=NY, 3=Overlap)
    "session",

    # Chart patterns (detected over rolling window)
    "support_bounce",    # price near support
    "resist_reject",     # price rejected at resistance
    "higher_high",
    "lower_low",
    "breakout_up",
    "breakout_down",
]


class TechnicalEngine:
    """
    Comprehensive technical analysis engine.

    Replaces and extends the original FeatureEngineer.build() method
    with 50+ features across trend, momentum, volatility, volume,
    candle patterns, and chart patterns.
    """

    # Default feature set (can override to use EXTENDED_FEATURE_COLS)
    FEATURE_COLS = EXTENDED_FEATURE_COLS

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    # ------------------------------------------------------------------
    # Main build method
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators on a OHLCV DataFrame.
        Returns enriched DataFrame with all FEATURE_COLS + 'target'.
        """
        d = df.copy()
        d = self._trend_indicators(d)
        d = self._momentum_indicators(d)
        d = self._volatility_indicators(d)
        d = self._volume_indicators(d)
        d = self._ichimoku(d)
        d = self._candle_patterns(d)
        d = self._price_action(d)
        d = self._session_flag(d)
        d = self._chart_patterns(d)
        d = self._target(d)

        before = len(d)
        d.dropna(inplace=True)
        log.debug(f"[Technical] {before} bars → {len(d)} after dropna ({before - len(d)} dropped)")
        return d

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _trend_indicators(self, d: pd.DataFrame) -> pd.DataFrame:
        close = d["Close"]

        d["sma_10"]  = SMAIndicator(close, window=10).sma_indicator()
        d["sma_20"]  = SMAIndicator(close, window=20).sma_indicator()
        d["sma_50"]  = SMAIndicator(close, window=50).sma_indicator()
        d["sma_200"] = SMAIndicator(close, window=200).sma_indicator()

        d["ema_9"]  = EMAIndicator(close, window=9).ema_indicator()
        d["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
        d["ema_21"] = EMAIndicator(close, window=21).ema_indicator()
        d["ema_26"] = EMAIndicator(close, window=26).ema_indicator()
        d["ema_50"] = EMAIndicator(close, window=50).ema_indicator()

        d["ema_cross_9_21"]  = (d["ema_9"]  > d["ema_21"]).astype(int)
        d["ema_cross_21_50"] = (d["ema_21"] > d["ema_50"]).astype(int)
        d["price_vs_sma50"]  = close / d["sma_50"]  - 1
        d["price_vs_sma200"] = close / d["sma_200"] - 1

        adx_obj = ADXIndicator(d["High"], d["Low"], close)
        d["adx"]     = adx_obj.adx()
        d["adx_pos"] = adx_obj.adx_pos()
        d["adx_neg"] = adx_obj.adx_neg()

        # Parabolic SAR
        psar = PSARIndicator(d["High"], d["Low"], close)
        d["psar_bull"] = (psar.psar_down().isna() | (close > psar.psar())).astype(int)

        return d

    def _momentum_indicators(self, d: pd.DataFrame) -> pd.DataFrame:
        close = d["Close"]

        rsi_obj  = RSIIndicator(close, window=14)
        d["rsi"] = rsi_obj.rsi()
        d["rsi_sma"] = d["rsi"].rolling(14).mean()

        stoch = StochasticOscillator(d["High"], d["Low"], close)
        d["stoch_k"] = stoch.stoch()
        d["stoch_d"] = stoch.stoch_signal()

        d["williams_r"] = WilliamsRIndicator(d["High"], d["Low"], close).williams_r()

        d["roc_5"]  = ROCIndicator(close, window=5).roc()
        d["roc_10"] = ROCIndicator(close, window=10).roc()

        macd = MACD(close)
        d["macd"]        = macd.macd()
        d["macd_signal"] = macd.macd_signal()
        d["macd_diff"]   = macd.macd_diff()
        d["macd_above_zero"] = (d["macd"] > 0).astype(int)

        d["cci"] = CCIIndicator(d["High"], d["Low"], close).cci()

        return d

    def _volatility_indicators(self, d: pd.DataFrame) -> pd.DataFrame:
        close = d["Close"]

        atr_obj = AverageTrueRange(d["High"], d["Low"], close)
        d["atr"]      = atr_obj.average_true_range()
        d["atr_norm"] = d["atr"] / close

        bb = BollingerBands(close)
        d["bb_width"] = bb.bollinger_wband()
        d["bb_pct"]   = bb.bollinger_pband()

        kc = KeltnerChannel(d["High"], d["Low"], close)
        d["kc_width"] = (kc.keltner_channel_hband() - kc.keltner_channel_lband()) / close

        return d

    def _volume_indicators(self, d: pd.DataFrame) -> pd.DataFrame:
        close = d["Close"]
        vol   = d["Volume"]

        # Forex pairs from yfinance return zero volume — guard against all-zero
        has_volume = vol.sum() > 0

        if has_volume:
            obv = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
            d["obv_norm"] = obv / (obv.rolling(20).mean().abs() + 1e-9)
            d["mfi"] = MFIIndicator(d["High"], d["Low"], close, vol).money_flow_index()
        else:
            # No real volume data — neutral fill so dropna does not remove all rows
            d["obv_norm"] = 0.0
            d["mfi"]      = 50.0   # MFI mid-point = neutral

        return d

    def _ichimoku(self, d: pd.DataFrame) -> pd.DataFrame:
        ichi = IchimokuIndicator(d["High"], d["Low"])
        tenkan = ichi.ichimoku_conversion_line()
        kijun  = ichi.ichimoku_base_line()
        span_a = ichi.ichimoku_a()
        span_b = ichi.ichimoku_b()

        cloud_top    = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

        d["ichi_above_cloud"]  = (d["Close"] > cloud_top).astype(int)
        d["ichi_ten_above_kij"] = (tenkan > kijun).astype(int)

        return d

    def _candle_patterns(self, d: pd.DataFrame) -> pd.DataFrame:
        """Binary candle pattern flags."""
        o, h, l, c = d["Open"], d["High"], d["Low"], d["Close"]
        body   = (c - o).abs()
        upper  = h - pd.concat([c, o], axis=1).max(axis=1)
        lower  = pd.concat([c, o], axis=1).min(axis=1) - l
        hl     = h - l

        # Doji: very small body relative to range
        d["doji"] = ((body / (hl + 1e-9)) < 0.1).astype(int)

        # Bullish Engulfing
        prev_bear = (c.shift(1) < o.shift(1))
        curr_bull = (c > o)
        engulf_b  = (o < c.shift(1)) & (c > o.shift(1))
        d["bullish_engulf"] = (prev_bear & curr_bull & engulf_b).astype(int)

        # Bearish Engulfing
        prev_bull = (c.shift(1) > o.shift(1))
        curr_bear = (c < o)
        engulf_a  = (o > c.shift(1)) & (c < o.shift(1))
        d["bearish_engulf"] = (prev_bull & curr_bear & engulf_a).astype(int)

        # Hammer: small body at top, long lower wick, tiny upper wick
        d["hammer"] = (
            (lower > 2 * body) &
            (upper < body * 0.5) &
            (body > 0)
        ).astype(int)

        # Shooting Star: small body at bottom, long upper wick
        d["shooting_star"] = (
            (upper > 2 * body) &
            (lower < body * 0.5) &
            (body > 0)
        ).astype(int)

        # Morning Star (3-bar): bear, doji, bull
        d["morning_star"] = (
            (c.shift(2) < o.shift(2)) &
            ((body.shift(1) / (hl.shift(1) + 1e-9)) < 0.15) &
            (c > o) &
            (c > (o.shift(2) + c.shift(2)) / 2)
        ).astype(int)

        # Evening Star (3-bar): bull, doji, bear
        d["evening_star"] = (
            (c.shift(2) > o.shift(2)) &
            ((body.shift(1) / (hl.shift(1) + 1e-9)) < 0.15) &
            (c < o) &
            (c < (o.shift(2) + c.shift(2)) / 2)
        ).astype(int)

        return d

    def _price_action(self, d: pd.DataFrame) -> pd.DataFrame:
        c = d["Close"]
        d["ret_1"]      = c.pct_change(1)
        d["ret_3"]      = c.pct_change(3)
        d["ret_5"]      = c.pct_change(5)
        d["ret_10"]     = c.pct_change(10)
        d["close_norm"] = c / c.rolling(50).mean() - 1
        return d

    def _session_flag(self, d: pd.DataFrame) -> pd.DataFrame:
        """
        Encode trading session based on UTC hour.
          0 = Asia (22-08 UTC)
          1 = London (08-13 UTC)
          2 = NY (13-18 UTC)
          3 = Overlap London/NY (12-16 UTC)
        """
        if not isinstance(d.index, pd.DatetimeIndex):
            d["session"] = 0
            return d

        hour = d.index.hour

        def _session(h: int) -> int:
            if 12 <= h < 16:
                return 3   # overlap
            if 8  <= h < 13:
                return 1   # London
            if 13 <= h < 18:
                return 2   # NY
            return 0        # Asia

        d["session"] = [_session(h) for h in hour]
        return d

    def _chart_patterns(self, d: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Detect chart patterns over a rolling window."""
        c    = d["Close"]
        roll_high = c.rolling(window).max()
        roll_low  = c.rolling(window).min()

        atr = d["atr"] if "atr" in d.columns else c * 0.001

        # Support bounce: close is within 1x ATR of 20-bar low
        d["support_bounce"] = (c - roll_low < atr).astype(int)

        # Resistance reject: close is within 1x ATR of 20-bar high
        d["resist_reject"]  = (roll_high - c < atr).astype(int)

        # Higher high: current bar high > previous bar high
        d["higher_high"] = (d["High"] > d["High"].shift(1)).astype(int)

        # Lower low: current bar low < previous bar low
        d["lower_low"] = (d["Low"] < d["Low"].shift(1)).astype(int)

        # Breakout up: close beyond rolling high
        d["breakout_up"]   = (c > roll_high.shift(1)).astype(int)

        # Breakout down: close below rolling low
        d["breakout_down"] = (c < roll_low.shift(1)).astype(int)

        return d

    def _target(self, d: pd.DataFrame) -> pd.DataFrame:
        """
        ATR-magnitude target — world-class improvement over naive binary UP/DOWN.

        Labels:
          1 = next-bar forward return > 0.3 × ATR  (meaningful bullish move)
          0 = next-bar forward return < -0.3 × ATR (meaningful bearish move)
          NaN = move within ±0.3 × ATR (noise — excluded from training)

        Rationale: A +0.1 pip move and a +200 pip move both scored 1 in the
        old binary target.  The ATR threshold filters out micro-fluctuations
        that are indistinguishable from spread + slippage, forcing the model
        to only learn on bars with genuine directional conviction.
        """
        fwd = d["Close"].shift(-1) - d["Close"]
        if "atr" in d.columns:
            thresh = (d["atr"] * 0.3).clip(lower=1e-9)
        else:
            # Fallback: use 0.03% of price (≈ 3 pips on standard forex)
            thresh = (d["Close"] * 0.0003).clip(lower=1e-9)

        d["target"] = np.where(fwd > thresh, 1,
                      np.where(fwd < -thresh, 0, np.nan))
        return d

    # ------------------------------------------------------------------
    # Multi-timeframe feature merge
    # ------------------------------------------------------------------

    def build_multi_timeframe(
        self,
        dfs: Dict[str, pd.DataFrame],
        primary_tf: str = "H1",
    ) -> pd.DataFrame:
        """
        Build features on multiple timeframes and merge into primary tf DataFrame.

        dfs: {"H1": df_h1, "H4": df_h4, "D1": df_d1}
        Returns primary tf DataFrame with higher-tf trend indicators merged in.
        """
        if primary_tf not in dfs:
            raise ValueError(f"primary_tf '{primary_tf}' not in dfs keys.")

        primary = self.build(dfs[primary_tf]).copy()

        for tf, df in dfs.items():
            if tf == primary_tf:
                continue
            try:
                tf_feat = self.build(df)
                # Select only trend/momentum cols for higher tf
                cols_to_merge = [c for c in ["adx", "rsi", "ema_cross_9_21",
                                              "macd_above_zero", "ichi_above_cloud"]
                                 if c in tf_feat.columns]
                suffix = f"_{tf.lower()}"
                rename = {c: c + suffix for c in cols_to_merge}
                tf_merge = tf_feat[cols_to_merge].rename(columns=rename)
                # Merge: forward-fill higher tf values into primary index
                primary = primary.merge(
                    tf_merge, how="left", left_index=True, right_index=True
                )
                for col in rename.values():
                    primary[col] = primary[col].ffill()
                log.debug(f"[Technical] Merged {tf} features into {primary_tf}.")
            except Exception as exc:
                log.warning(f"[Technical] Could not merge {tf}: {exc}")

        primary.dropna(inplace=True)
        return primary

    # ------------------------------------------------------------------
    # Signal interpretation for live use
    # ------------------------------------------------------------------

    def interpret_signals(self, df: pd.DataFrame) -> Dict:
        """
        Interprets the latest bar of a fully-built feature DataFrame.
        Returns a dictionary of trading signal summaries.
        """
        if df.empty:
            return {}

        row = df.iloc[-1]

        signals = {
            "trend":      self._read_trend(row),
            "momentum":   self._read_momentum(row),
            "volatility": self._read_volatility(row),
            "pattern":    self._read_pattern(row),
            "session":    self._read_session(row),
        }

        # Overall bias: -1 = strong sell, 0 = neutral, +1 = strong buy
        bull = sum(1 for v in signals.values() if v.get("bias", 0) > 0)
        bear = sum(1 for v in signals.values() if v.get("bias", 0) < 0)
        signals["overall_bias"] = round((bull - bear) / max(bull + bear, 1), 2)

        return signals

    def _read_trend(self, row: pd.Series) -> Dict:
        bias = 0
        notes = []
        if "ema_cross_9_21" in row and row["ema_cross_9_21"] == 1:
            bias += 1
            notes.append("9 EMA above 21 EMA (bullish)")
        elif "ema_cross_9_21" in row:
            bias -= 1
            notes.append("9 EMA below 21 EMA (bearish)")
        if "price_vs_sma50" in row:
            if row["price_vs_sma50"] > 0:
                bias += 1
                notes.append("Price above SMA50")
            else:
                bias -= 1
                notes.append("Price below SMA50")
        if "adx" in row and row["adx"] > 25:
            notes.append(f"ADX={row['adx']:.1f} (trending)")
        return {"bias": bias, "notes": notes}

    def _read_momentum(self, row: pd.Series) -> Dict:
        bias = 0
        notes = []
        if "rsi" in row:
            rsi = row["rsi"]
            if rsi < 30:
                bias += 1
                notes.append(f"RSI={rsi:.1f} oversold (bullish)")
            elif rsi > 70:
                bias -= 1
                notes.append(f"RSI={rsi:.1f} overbought (bearish)")
            else:
                notes.append(f"RSI={rsi:.1f} neutral")
        if "macd_diff" in row:
            if row["macd_diff"] > 0:
                bias += 1
                notes.append("MACD histogram positive")
            else:
                bias -= 1
                notes.append("MACD histogram negative")
        return {"bias": bias, "notes": notes}

    def _read_volatility(self, row: pd.Series) -> Dict:
        notes = []
        if "bb_pct" in row:
            bp = row["bb_pct"]
            if bp > 0.8:
                notes.append(f"BB%={bp:.2f} near upper band")
            elif bp < 0.2:
                notes.append(f"BB%={bp:.2f} near lower band")
        if "atr_norm" in row:
            notes.append(f"ATR norm={row['atr_norm']:.4f}")
        return {"bias": 0, "notes": notes}

    def _read_pattern(self, row: pd.Series) -> Dict:
        bias = 0
        notes = []
        for bull_pat in ("bullish_engulf", "hammer", "morning_star", "support_bounce", "breakout_up"):
            if bull_pat in row and row[bull_pat] == 1:
                bias += 1
                notes.append(f"Pattern: {bull_pat}")
        for bear_pat in ("bearish_engulf", "shooting_star", "evening_star", "resist_reject", "breakout_down"):
            if bear_pat in row and row[bear_pat] == 1:
                bias -= 1
                notes.append(f"Pattern: {bear_pat}")
        return {"bias": bias, "notes": notes}

    def _read_session(self, row: pd.Series) -> Dict:
        sessions = {0: "Asia", 1: "London", 2: "NY", 3: "London/NY Overlap"}
        if "session" in row:
            s = int(row["session"])
            return {"bias": 0, "notes": [f"Session: {sessions.get(s, 'Unknown')}"]}
        return {"bias": 0, "notes": []}

    # ------------------------------------------------------------------
    # Fibonacci levels
    # ------------------------------------------------------------------

    @staticmethod
    def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
        """
        Compute Fibonacci retracement and extension levels.
        Returns dict of level_name → price.
        """
        delta = high - low
        return {
            "0.0%":    round(high, 5),
            "23.6%":   round(high - 0.236 * delta, 5),
            "38.2%":   round(high - 0.382 * delta, 5),
            "50.0%":   round(high - 0.500 * delta, 5),
            "61.8%":   round(high - 0.618 * delta, 5),
            "78.6%":   round(high - 0.786 * delta, 5),
            "100.0%":  round(low, 5),
            "127.2%":  round(low  - 0.272 * delta, 5),
            "161.8%":  round(low  - 0.618 * delta, 5),
        }

    # ------------------------------------------------------------------
    # Support & Resistance detection
    # ------------------------------------------------------------------

    @staticmethod
    def find_support_resistance(
        df: pd.DataFrame, window: int = 20, n_levels: int = 5
    ) -> Dict[str, List[float]]:
        """
        Identify key support and resistance price levels using local pivots.
        Returns {"support": [...], "resistance": [...]}
        """
        highs = df["High"].values
        lows  = df["Low"].values

        pivots_high: List[float] = []
        pivots_low:  List[float] = []

        for i in range(window, len(df) - window):
            slh = highs[i - window: i + window + 1]
            sll = lows[i  - window: i + window + 1]
            if highs[i] == slh.max():
                pivots_high.append(highs[i])
            if lows[i] == sll.min():
                pivots_low.append(lows[i])

        # Cluster nearby levels
        def cluster(levels: List[float], tol: float = 0.0002) -> List[float]:
            if not levels:
                return []
            arr = sorted(set(levels))
            clustered = [arr[0]]
            for v in arr[1:]:
                if abs(v - clustered[-1]) > tol:
                    clustered.append(v)
            return clustered

        current = df["Close"].iloc[-1]
        support    = sorted([l for l in cluster(pivots_low)  if l < current], reverse=True)[:n_levels]
        resistance = sorted([h for h in cluster(pivots_high) if h > current])[:n_levels]

        return {"support": support, "resistance": resistance}


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    np.random.seed(42)
    n = 500
    price = 1.1000
    rows = []
    ts = pd.Timestamp("2023-01-01", tz="UTC")
    for _ in range(n):
        o = price
        h = o + abs(np.random.normal(0, 0.001))
        l = o - abs(np.random.normal(0, 0.001))
        c = np.random.uniform(l, h)
        rows.append((ts, o, h, l, c, np.random.randint(100, 5000)))
        price = c
        ts += pd.Timedelta(hours=1)
    df = pd.DataFrame(rows, columns=["time","Open","High","Low","Close","Volume"])
    df.set_index("time", inplace=True)

    engine = TechnicalEngine()
    result = engine.build(df)
    print(f"Feature matrix shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")

    signals = engine.interpret_signals(result)
    print(f"\nSignal interpretation:")
    for k, v in signals.items():
        if isinstance(v, dict):
            print(f"  {k}: bias={v.get('bias',0)} | {', '.join(v.get('notes',[]))}")
        else:
            print(f"  overall_bias: {v}")

    fib = TechnicalEngine.fibonacci_levels(1.1200, 1.0900)
    print(f"\nFibonacci levels:")
    for lvl, price_ in fib.items():
        print(f"  {lvl:8s} = {price_}")

    sr = TechnicalEngine.find_support_resistance(df)
    print(f"\nSupport:    {sr['support']}")
    print(f"Resistance: {sr['resistance']}")
