"""
================================================================================
 FOREX AI TRADING BOT v2.0 — MetaTrader 5 + LSTM/RF + Intelligence Layer
================================================================================
 Author  : AI-Generated (Production-Ready Demo Account Bot)
 Version : 2.0.0
 Python  : 3.10+

 Requirements:
     pip install MetaTrader5 pandas numpy scikit-learn tensorflow ta
                 matplotlib seaborn joblib pytz requests feedparser
                 beautifulsoup4 lxml vaderSentiment PyPDF2 pdfplumber
                 yfinance python-dotenv schedule

 New in v2.0:
   - Real-time news feed with sentiment scoring (news_engine.py)
   - Book/PDF training pipeline (book_trainer.py)
   - 50+ technical indicators + chart patterns (technical_engine.py)
   - Fundamental analysis + multi-pair quotes (fundamental_engine.py)
   - Multi-pair support (28 major/minor pairs)
   - Session-aware trading (London/NY/Asia/Overlap)
   - High-impact news blackout guard
   - Knowledge-base overlay on ML signals

 Usage:
     python forex_ai_bot.py --mode train
     python forex_ai_bot.py --mode backtest
     python forex_ai_bot.py --mode live
     python forex_ai_bot.py --mode train-books --books-folder books/
     python forex_ai_bot.py --mode quotes
     python forex_ai_bot.py --mode news
     python forex_ai_bot.py --mode fundamentals
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib

# Technical Analysis
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow / Keras (LSTM)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model  # type: ignore[import]
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional  # type: ignore[import]
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore[import]
    from tensorflow.keras.optimizers import Adam  # type: ignore[import]
    from tensorflow.keras.regularizers import l2 as _keras_l2  # type: ignore[import]
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    print("[WARNING] TensorFlow not found. LSTM/Ensemble models unavailable. "
          "Use --model RF or install TensorFlow: pip install tensorflow")

# MetaTrader 5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MetaTrader5 package not found. "
          "Install it with: pip install MetaTrader5  (Windows only)")

# ── Timezone ──────────────────────────────────────────────────────────────────
import pytz

# ── Intelligence layer (new in v2.0) ──────────────────────────────────────────
try:
    from news_engine import NewsEngine, EconomicCalendar
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    print("[WARNING] news_engine.py not found. News features disabled.")

try:
    from technical_engine import TechnicalEngine, EXTENDED_FEATURE_COLS
    TECH_ENGINE_AVAILABLE = True
except ImportError:
    TECH_ENGINE_AVAILABLE = False
    print("[WARNING] technical_engine.py not found. Using basic indicators.")

try:
    from fundamental_engine import FundamentalEngine, ALL_PAIRS
    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    FUNDAMENTAL_AVAILABLE = False
    print("[WARNING] fundamental_engine.py not found. Fundamental analysis disabled.")

try:
    from book_trainer import BookTrainer, KnowledgeBase, load_classic_rules
    BOOKS_AVAILABLE = True
except ImportError:
    BOOKS_AVAILABLE = False
    print("[WARNING] book_trainer.py not found. Book training disabled.")

try:
    from multi_account_manager import MultiAccountManager, AccountProfile
    MULTI_ACCOUNT_AVAILABLE = True
except ImportError:
    MULTI_ACCOUNT_AVAILABLE = False
    print("[WARNING] multi_account_manager.py not found. Multi-account disabled.")

try:
    from social_engine import SocialEngine, TradeRecord, PerformanceTracker
    SOCIAL_AVAILABLE = True
except ImportError:
    SOCIAL_AVAILABLE = False
    print("[WARNING] social_engine.py not found. Social features disabled.")

# ==============================================================================
#  LOAD CREDENTIALS FROM config.py
# ==============================================================================
try:
    import config as _cfg
    _MT5_LOGIN    = _cfg.MT5_LOGIN
    _MT5_PASSWORD = _cfg.MT5_PASSWORD
    _MT5_SERVER   = _cfg.MT5_SERVER
    _MT5_PATH     = _cfg.MT5_PATH
    _LIVE_MODE    = getattr(_cfg, "LIVE_MODE", False)
    print(f"[CONFIG] Loaded credentials — server: {_MT5_SERVER} "
          f"| mode: {'LIVE' if _LIVE_MODE else 'DEMO'}")
except ImportError:
    _MT5_LOGIN, _MT5_PASSWORD, _MT5_SERVER, _MT5_PATH, _LIVE_MODE = 0, "", "", "", False
    print("[CONFIG] config.py not found — MT5 credentials not set.")

# ==============================================================================
#  LOAD MULTI-ACCOUNT + SOCIAL CONFIG FROM accounts_config.py
# ==============================================================================
try:
    import accounts_config as _acfg
    _ACCOUNTS   = getattr(_acfg, "ACCOUNTS",            [])
    _TG_TOKEN   = getattr(_acfg, "TELEGRAM_BOT_TOKEN",  "")
    _TG_CHANNEL = getattr(_acfg, "TELEGRAM_CHANNEL_ID", "")
    _TG_CHAT    = getattr(_acfg, "TELEGRAM_CHAT_ID",    "")
    _BRANDING   = {
        "name":      getattr(_acfg, "BOT_DISPLAY_NAME", "ForexAI Bot"),
        "tagline":   getattr(_acfg, "BOT_TAGLINE",      "AI-Powered Forex Signals"),
        "website":   getattr(_acfg, "BOT_WEBSITE",      ""),
        "instagram": getattr(_acfg, "BOT_INSTAGRAM",    ""),
        "twitter":   getattr(_acfg, "BOT_TWITTER",      ""),
    }
    _active_accounts = [a for a in _ACCOUNTS if a.get("active", True)
                        and a.get("login", 0)]
    print(f"[CONFIG] accounts_config loaded — "
          f"{len(_active_accounts)} active account(s)")
except ImportError:
    _ACCOUNTS, _TG_TOKEN, _TG_CHANNEL, _TG_CHAT, _BRANDING = [], "", "", "", {}
    _active_accounts = []
    print("[CONFIG] accounts_config.py not found — single-account / no-social mode.")

# ==============================================================================
#  CONFIGURATION — Edit these before running
# ==============================================================================
CONFIG = {
    # ── MT5 Connection ─────────────────────────────────────────────────────
    "mt5_login":    _MT5_LOGIN,    # Loaded from config.py
    "mt5_password": _MT5_PASSWORD, # Loaded from config.py
    "mt5_server":   _MT5_SERVER,   # Loaded from config.py
    "mt5_path":     _MT5_PATH,     # Loaded from config.py

    # ── Symbol & Timeframe ─────────────────────────────────────────────────
    "symbol":       "EURUSD",
    "timeframe":    "H1",       # M1 M5 M15 M30 H1 H4 D1
    # Multi-pair mode: trades all 29 instruments (28 forex majors/minors + Gold)
    "multi_pair_mode": True,
    "pairs": [
        # Majors
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        # EUR crosses
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
        # GBP crosses
        "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
        # JPY crosses
        "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
        # AUD/NZD/CAD crosses
        "AUDCAD", "AUDCHF", "AUDNZD",
        "CADCHF", "NZDCAD", "NZDCHF",
        # Commodities
        "XAUUSD",   # Gold vs USD
    ],

    # ── Data ───────────────────────────────────────────────────────────────
    "history_bars": 10000,      # Bars to download for training

    # ── Model ──────────────────────────────────────────────────────────────
    "model_type":   "LSTM",     # "LSTM", "RF" (Random Forest), or "ENSEMBLE" (LSTM+RF vote)
    "lstm_lookback": 60,        # Sequence length fed to LSTM
    "model_path":   "models/",  # Directory to save/load model artefacts

    # ── Trading ────────────────────────────────────────────────────────────
    "lot_size":     0.01,       # Base lot size
    "sl_pips":      30,         # Stop-loss in pips (fallback when ATR unavailable)
    "tp_pips":      60,         # Take-profit in pips (fallback when ATR unavailable)
    "atr_sl_mult":  1.5,        # SL = 1.5 × ATR (dynamic — adapts to volatility)
    "atr_tp_mult":  3.0,        # TP = 3.0 × ATR = 2:1 RR minimum, auto-adjusting
    "max_spread":   3.0,        # Max allowed spread in pips before skipping (forex)
    # Per-instrument spread overrides (pips) — overrides max_spread for that symbol
    "max_spread_overrides": {
        "XAUUSD": 60.0,  # Gold: wider spread normal ($0.30–$0.60 = 30–60 pips)
        "GBPJPY": 5.0,   "GBPNZD": 6.0, "EURNZD": 5.0,
        "GBPAUD": 5.0,   "AUDNZD": 5.0, "NZDCHF": 5.0,
        "NZDCAD": 5.0,   "CADCHF": 4.0, "GBPCAD": 5.0,
    },
    "magic_number": 20240101,   # Unique EA identifier

    # ── Risk Management ────────────────────────────────────────────────────
    "max_drawdown_pct": 10.0,   # Stop trading if drawdown exceeds this %
    "risk_pct":         1.0,    # Risk % of balance per trade
    "max_open_trades":  5,      # Maximum simultaneous open positions (across all pairs)

    # ── Live Loop ──────────────────────────────────────────────────────────
    "poll_interval_sec": 60,    # Seconds between live checks

    # ── Intelligence Layer (v2.0) ───────────────────────────────────────────
    "use_news":           True,  # Enable real-time news sentiment overlay
    "use_fundamentals":   True,  # Enable fundamental analysis overlay
    "use_knowledge_base": True,  # Enable knowledge-base (book training) overlay
    "news_min_sentiment": 0.05,  # Minimum |sentiment| to apply news bias
    "news_blackout":      True,  # Pause trading before high-impact events
    "news_blackout_hours":2.0,   # Hours before event to pause
    "sentiment_weight":   0.15,  # News sentiment weight in combined signal
    "fundamental_weight": 0.15,  # Fundamental weight in combined signal
    "knowledge_weight":   0.10,  # Knowledge base weight in combined signal
    "books_folder":       "books/",  # Folder to auto-scan for PDF/txt books

    # ── Social & Multi-Account (v3.0) ──────────────────────────────────────
    "social": {
        "enabled":           True,   # Master switch for all social features
        "broadcast_signals": False,  # Broadcast every signal (even no-trades)
        "daily_report":      True,   # Send daily P&L summary to Telegram
        "weekly_report":     True,   # Send weekly P&L card to Telegram
        "report_hour_utc":   22,     # Hour (UTC) to send daily report
    },
    "telegram": {
        "bot_token":  _TG_TOKEN,
        "channel_id": _TG_CHANNEL,
        "chat_id":    _TG_CHAT,
    },
    "branding":  _BRANDING,
    "accounts":  _ACCOUNTS,         # Full list (includes inactive)
}

# ==============================================================================
#  LOGGING SETUP
# ==============================================================================
os.makedirs("logs", exist_ok=True)
os.makedirs(CONFIG["model_path"], exist_ok=True)

# Ensure UTF-8 output on Windows consoles (avoids UnicodeEncodeError for box-drawing chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/forex_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        ),
    ],
)
log = logging.getLogger("ForexBot")

# ==============================================================================
#  TIMEFRAME MAP
# ==============================================================================
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1  if MT5_AVAILABLE else 1,
    "M5":  mt5.TIMEFRAME_M5  if MT5_AVAILABLE else 5,
    "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    "H1":  mt5.TIMEFRAME_H1  if MT5_AVAILABLE else 60,
    "H4":  mt5.TIMEFRAME_H4  if MT5_AVAILABLE else 240,
    "D1":  mt5.TIMEFRAME_D1  if MT5_AVAILABLE else 1440,
}


# ==============================================================================
#  1. MT5 CONNECTION
# ==============================================================================
class MT5Connector:
    """Handles MetaTrader 5 connection lifecycle."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.connected = False

    def connect(self) -> bool:
        """Initialise and log in to MT5 terminal — retries 5 times with
        progressive delays so the bot survives a slow terminal startup."""
        if not MT5_AVAILABLE:
            log.error("MetaTrader5 library not installed.")
            return False

        mt5_path = self.cfg.get("mt5_path", "")

        for attempt in range(1, 6):
            log.info(f"MT5 connect attempt {attempt}/5 …")

            # Strategy 1: try without explicit path (attach to running terminal)
            ok = mt5.initialize()
            if not ok and mt5_path:
                # Strategy 2: launch via explicit path
                ok = mt5.initialize(path=mt5_path)

            if not ok:
                err = mt5.last_error()
                log.warning(f"mt5.initialize() failed [{attempt}/5]: {err}")
                if attempt < 5:
                    wait = attempt * 10
                    log.info(f"Retrying in {wait}s … (make sure MT5 is open and AutoTrading is enabled)")
                    time.sleep(wait)
                continue

            # Login
            if self.cfg.get("mt5_login"):
                ok = mt5.login(
                    login=self.cfg["mt5_login"],
                    password=self.cfg["mt5_password"],
                    server=self.cfg["mt5_server"],
                )
                if not ok:
                    log.warning(f"MT5 login failed [{attempt}/5]: {mt5.last_error()}")
                    mt5.shutdown()
                    if attempt < 5:
                        time.sleep(attempt * 10)
                    continue

            info = mt5.account_info()
            if info is None:
                log.warning(f"Could not retrieve account info [{attempt}/5].")
                mt5.shutdown()
                if attempt < 5:
                    time.sleep(attempt * 10)
                continue

            log.info(
                f"Connected ✓ | Account: {info.login} | "
                f"Server: {info.server} | Balance: {info.balance:.2f} {info.currency}"
            )
            self.connected = True
            return True

        log.critical(
            "MT5 connection failed after 5 attempts.\n"
            "  Fix checklist:\n"
            "  1. Open MetaTrader 5 terminal and log in\n"
            "  2. Click the 'AutoTrading' button in the MT5 toolbar (must be GREEN)\n"
            "  3. Tools → Options → Expert Advisors → tick 'Allow automated trading'\n"
            "  4. Make sure the toolbar shows a green connection dot (bottom right)\n"
            "  5. Re-run the bot"
        )
        return False

    def disconnect(self):
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            log.info("MT5 disconnected.")

    def account_balance(self) -> float:
        info = mt5.account_info()
        return info.balance if info else 0.0

    def account_equity(self) -> float:
        info = mt5.account_info()
        return info.equity if info else 0.0


# ==============================================================================
#  2. DATA DOWNLOADER
# ==============================================================================
class DataDownloader:
    """Downloads OHLCV bars from MT5 or yfinance (real market data)."""

    # yfinance ticker map for forex pairs and commodities
    _YF_MAP = {
        # Majors
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
        "USDCHF": "CHF=X",    "USDCAD": "CAD=X",    "AUDUSD": "AUDUSD=X",
        "NZDUSD": "NZDUSD=X",
        # EUR crosses
        "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "EURCHF": "EURCHF=X",
        "EURAUD": "EURAUD=X", "EURCAD": "EURCAD=X", "EURNZD": "EURNZD=X",
        # GBP crosses
        "GBPJPY": "GBPJPY=X", "GBPCHF": "GBPCHF=X", "GBPAUD": "GBPAUD=X",
        "GBPCAD": "GBPCAD=X", "GBPNZD": "GBPNZD=X",
        # JPY crosses
        "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X", "CHFJPY": "CHFJPY=X",
        "NZDJPY": "NZDJPY=X",
        # AUD/NZD/CAD crosses
        "AUDCAD": "AUDCAD=X", "AUDCHF": "AUDCHF=X", "AUDNZD": "AUDNZD=X",
        "CADCHF": "CADCHF=X", "NZDCAD": "NZDCAD=X", "NZDCHF": "NZDCHF=X",
        # Commodities
        "XAUUSD": "GC=F",     # Gold vs USD (gold futures)
        "XAGUSD": "SI=F",     # Silver vs USD
        "USOIL":  "CL=F",     # Crude Oil WTI
    }

    # Pip size per instrument (price units per 1 pip)
    _PIP_SIZE = {
        # JPY pairs: 0.01
        "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01, "AUDJPY": 0.01,
        "CADJPY": 0.01, "CHFJPY": 0.01, "NZDJPY": 0.01,
        # Commodities
        "XAUUSD": 0.01,   # Gold: $0.01 per pip (price ~$2000–3000)
        "XAGUSD": 0.001,  # Silver: $0.001 per pip
        "USOIL":  0.01,   # Oil: $0.01 per pip
    }
    _PIP_SIZE_DEFAULT = 0.0001  # All standard forex pairs

    @classmethod
    def pip_size(cls, symbol: str) -> float:
        """Return the pip size (price movement per 1 pip) for a given symbol."""
        return cls._PIP_SIZE.get(symbol.upper(), cls._PIP_SIZE_DEFAULT)
    # yfinance interval map from internal timeframe names
    _YF_INTERVAL = {
        "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
        "H1": "1h", "H4": "4h", "D1":  "1d",
    }
    # yfinance period that gives enough bars for each interval
    _YF_PERIOD = {
        "M1": "7d", "M5": "60d", "M15": "60d", "M30": "60d",
        "H1": "730d", "H4": "730d", "D1": "5y",
    }

    def __init__(self, symbol: str, timeframe: str, n_bars: int):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_bars = n_bars
        self.tf_code = TF_MAP.get(timeframe, TF_MAP["H1"])

    def fetch(self) -> pd.DataFrame:
        """Download historical bars — MT5 first, yfinance second, synthetic last."""
        log.info(f"Downloading {self.n_bars} bars of {self.symbol} {self.timeframe}…")

        # 1) Try MT5
        if MT5_AVAILABLE:
            rates = mt5.copy_rates_from_pos(self.symbol, self.tf_code, 0, self.n_bars)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("time", inplace=True)
                df.rename(columns={
                    "open": "Open", "high": "High",
                    "low": "Low",  "close": "Close",
                    "tick_volume": "Volume"
                }, inplace=True)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                log.info(f"MT5 data: {len(df)} bars. Range: {df.index[0]} → {df.index[-1]}")
                return df
            log.warning(f"MT5 returned no data ({mt5.last_error()}) — trying yfinance.")

        # 2) Try yfinance (real historical data, no API key needed)
        df = self._yfinance_data()
        if not df.empty:
            return df

        # 3) Last resort: synthetic data
        log.warning("All real data sources failed — using synthetic data for demo only.")
        return self._synthetic_data()

    def fetch_higher_tf(self) -> pd.DataFrame:
        """
        Fetch H4 data for the same symbol — used as a trend filter.
        Returns DataFrame with same columns as fetch().
        """
        higher = "H4" if self.timeframe in ("M1","M5","M15","M30","H1") else "D1"
        dl = DataDownloader(self.symbol, higher, 500)
        return dl.fetch()

    def _yfinance_data(self) -> pd.DataFrame:
        """Download data via yfinance. Returns empty DataFrame on failure."""
        try:
            import yfinance as yf
        except ImportError:
            return pd.DataFrame()

        ticker   = self._YF_MAP.get(self.symbol.upper(), f"{self.symbol}=X")
        interval = self._YF_INTERVAL.get(self.timeframe, "1h")
        period   = self._YF_PERIOD.get(self.timeframe, "730d")

        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()

            # yfinance ≥ 0.2.x returns multi-level columns — flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "time"
            # Trim to requested bar count
            if len(df) > self.n_bars:
                df = df.iloc[-self.n_bars:]
            log.info(f"yfinance data: {len(df)} bars of {self.symbol} {self.timeframe}. "
                     f"Range: {df.index[0]} → {df.index[-1]}")
            return df
        except Exception as exc:
            log.warning(f"yfinance error: {exc}")
            return pd.DataFrame()

    # ── Synthetic fallback (last resort — poor for training) ──────────────
    @staticmethod
    def _synthetic_data(n: int = 5000) -> pd.DataFrame:
        log.warning("Using SYNTHETIC data — results will not reflect real market behaviour.")
        np.random.seed(42)
        price = 1.1000
        rows = []
        ts = pd.Timestamp("2022-01-01", tz="UTC")
        for _ in range(n):
            o = price
            h = o + abs(np.random.normal(0, 0.0010))
            l = o - abs(np.random.normal(0, 0.0010))
            c = np.random.uniform(l, h)
            v = np.random.randint(100, 5000)
            rows.append((ts, o, h, l, c, v))
            price = c
            ts += pd.Timedelta(hours=1)
        df = pd.DataFrame(rows, columns=["time","Open","High","Low","Close","Volume"])
        df.set_index("time", inplace=True)
        return df


# ==============================================================================
#  3. FEATURE ENGINEERING
# ==============================================================================
class FeatureEngineer:
    """
    Adds technical indicators and constructs the ML feature matrix.

    v2.0: Delegates to TechnicalEngine (50+ indicators) when available.
    Falls back to the original 15-indicator set if technical_engine.py
    is not present.
    """

    # Populated dynamically based on which engine is used
    FEATURE_COLS: List[str] = []

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.scaler = MinMaxScaler()

        if TECH_ENGINE_AVAILABLE:
            self._tech = TechnicalEngine(lookback=lookback)
            self.FEATURE_COLS = list(EXTENDED_FEATURE_COLS)
            log.info(f"[FeatureEngineer] v2.0 — using TechnicalEngine ({len(self.FEATURE_COLS)} features)")
        else:
            self._tech = None
            self.FEATURE_COLS = [
                "sma_10", "sma_20", "sma_50",
                "ema_12", "ema_26",
                "rsi",
                "macd", "macd_signal", "macd_diff",
                "bb_width", "atr",
                "ret_1", "ret_3", "ret_5",
                "close_norm",
            ]
            log.info(f"[FeatureEngineer] v1 fallback — {len(self.FEATURE_COLS)} features")

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return enriched DataFrame."""
        if TECH_ENGINE_AVAILABLE and self._tech is not None:
            result = self._tech.build(df)
            # Ensure only known FEATURE_COLS are present (for scaler consistency)
            available = [c for c in self.FEATURE_COLS if c in result.columns]
            self.FEATURE_COLS = available
            return result

        # ── Fallback: original 15-indicator build ─────────────────────────
        d = df.copy()

        d["sma_10"] = SMAIndicator(d["Close"], window=10).sma_indicator()
        d["sma_20"] = SMAIndicator(d["Close"], window=20).sma_indicator()
        d["sma_50"] = SMAIndicator(d["Close"], window=50).sma_indicator()
        d["ema_12"] = EMAIndicator(d["Close"], window=12).ema_indicator()
        d["ema_26"] = EMAIndicator(d["Close"], window=26).ema_indicator()

        d["rsi"] = RSIIndicator(d["Close"], window=14).rsi()

        macd_obj = MACD(d["Close"])
        d["macd"]        = macd_obj.macd()
        d["macd_signal"] = macd_obj.macd_signal()
        d["macd_diff"]   = macd_obj.macd_diff()

        from ta.volatility import BollingerBands, AverageTrueRange
        bb = BollingerBands(d["Close"])
        d["bb_width"] = bb.bollinger_wband()
        d["atr"] = AverageTrueRange(d["High"], d["Low"], d["Close"]).average_true_range()

        d["ret_1"] = d["Close"].pct_change(1)
        d["ret_3"] = d["Close"].pct_change(3)
        d["ret_5"] = d["Close"].pct_change(5)
        d["close_norm"] = d["Close"] / d["Close"].rolling(50).mean() - 1

        # ATR-magnitude target (same logic as TechnicalEngine._target)
        fwd = d["Close"].shift(-1) - d["Close"]
        thresh = (d["atr"] * 0.3).clip(lower=1e-9) if "atr" in d.columns else (d["Close"] * 0.0003)
        d["target"] = np.where(fwd > thresh, 1, np.where(fwd < -thresh, 0, np.nan))
        d.dropna(inplace=True)
        return d

    def scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Min-max scale the feature matrix."""
        shape = X.shape
        flat = X.reshape(-1, shape[-1])
        if fit:
            flat = self.scaler.fit_transform(flat)
        else:
            flat = self.scaler.transform(flat)
        return flat.reshape(shape)

    def save_scaler(self, path: str):
        joblib.dump(self.scaler, path)
        log.info(f"Scaler saved → {path}")

    def load_scaler(self, path: str):
        self.scaler = joblib.load(path)
        log.info(f"Scaler loaded ← {path}")

    # ── Prepare sequences for LSTM ────────────────────────────────────────
    def make_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) where X has shape (n, lookback, features)."""
        X_raw = df[self.FEATURE_COLS].values
        y_raw = df["target"].values

        X_scaled = self.scale(X_raw, fit=False)

        X, y = [], []
        for i in range(self.lookback, len(X_scaled)):
            X.append(X_scaled[i - self.lookback: i])
            y.append(y_raw[i])

        return np.array(X), np.array(y)

    # ── Prepare flat features for Random Forest ───────────────────────────
    def make_flat(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = df[self.FEATURE_COLS].values
        y = df["target"].values
        X = self.scale(X, fit=False)
        return X, y


# ==============================================================================
#  4a. LSTM MODEL
# ==============================================================================
class LSTMModel:
    """Bidirectional LSTM classifier for price direction prediction."""

    def __init__(self, lookback: int, n_features: int, model_dir: str):
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow\n"
                "Or switch to Random Forest: set model_type = 'RF' in CONFIG."
            )
        self.lookback   = lookback
        self.n_features = n_features
        self.model_dir  = model_dir
        self.model_path = os.path.join(model_dir, "lstm_model.h5")
        self.model      = None

    def build(self):
        reg = _keras_l2(1e-4)
        model = Sequential([
            Bidirectional(
                LSTM(128, return_sequences=True, kernel_regularizer=reg),
                input_shape=(self.lookback, self.n_features)
            ),
            Dropout(0.3),
            BatchNormalization(),

            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=reg)),
            Dropout(0.3),
            BatchNormalization(),

            LSTM(32, kernel_regularizer=reg),
            Dropout(0.2),

            Dense(32, activation="relu", kernel_regularizer=reg),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        log.info(f"LSTM built (Bidirectional + L2): {model.count_params():,} parameters")
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50):
        if self.model is None:
            self.build()

        # Balance classes — prevents the model from predicting the majority class
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_map = dict(zip(classes.tolist(), weights.tolist()))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(self.model_path, save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                              min_lr=1e-6, verbose=1),
        ]

        log.info("Training LSTM…")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weight_map,
            verbose=1,
        )
        log.info("LSTM training complete.")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of UP move."""
        return self.model.predict(X, verbose=0).flatten()

    def save(self):
        self.model.save(self.model_path)
        log.info(f"LSTM saved → {self.model_path}")

    def load(self):
        self.model = load_model(self.model_path)
        log.info(f"LSTM loaded ← {self.model_path}")


# ==============================================================================
#  4b. RANDOM FOREST MODEL
# ==============================================================================
class RFModel:
    """Random Forest Classifier for price direction prediction."""

    def __init__(self, model_dir: str):
        self.model_dir  = model_dir
        self.model_path = os.path.join(model_dir, "rf_model.pkl")
        self.model: Optional[RandomForestClassifier] = None

    def build(self):
        base = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        # Isotonic calibration turns raw RF scores into true probabilities,
        # making the 0.58 live threshold statistically meaningful.
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        log.info("Random Forest model initialised (with isotonic calibration).")

    def train(self, X_train, y_train):
        if self.model is None:
            self.build()
        log.info("Training Random Forest…")
        self.model.fit(X_train, y_train)
        log.info("Random Forest training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of UP move."""
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, self.model_path)
        log.info(f"RF model saved → {self.model_path}")

    def load(self):
        self.model = joblib.load(self.model_path)
        log.info(f"RF model loaded ← {self.model_path}")


# ==============================================================================
#  4c. ENSEMBLE MODEL  (LSTM + RF weighted vote)
# ==============================================================================
class EnsembleModel:
    """
    Combines LSTM and Random Forest predictions via weighted voting.
    A trade signal is only generated when BOTH models agree above threshold.
    Default weight: LSTM 60%, RF 40%.
    """

    def __init__(self, lookback: int, n_features: int, model_dir: str,
                 lstm_weight: float = 0.6, rf_weight: float = 0.4):
        self.lstm_model  = LSTMModel(lookback, n_features, model_dir)
        self.rf_model    = RFModel(model_dir)
        self.lstm_weight = lstm_weight
        self.rf_weight   = rf_weight
        self.model_dir   = model_dir

    def train(self, feat_df, engineer, epochs: int = 50):
        """Train both sub-models on the same feature dataframe."""
        n = len(feat_df)
        # Scaler already fitted by caller; just use it
        X_seq, y_seq = engineer.make_sequences(feat_df)
        split = int(len(X_seq) * 0.85)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]
        log.info("Ensemble — training LSTM sub-model…")
        self.lstm_model.train(X_tr, y_tr, X_val, y_val, epochs=epochs)
        self.lstm_model.save()

        X_flat, y_flat = engineer.make_flat(feat_df)
        split_f = int(len(X_flat) * 0.85)
        log.info("Ensemble — training RF sub-model…")
        self.rf_model.train(X_flat[:split_f], y_flat[:split_f])
        self.rf_model.save()

        # Validation accuracy of combined vote
        if len(X_val) > 0:
            combined = self._combine(X_val, X_flat[split_f:], engineer)
            y_true   = y_val
            min_len  = min(len(combined), len(y_true))
            preds    = (combined[:min_len] > 0.5).astype(int)
            acc = (preds == y_true[:min_len]).mean()
            log.info(f"Ensemble validation accuracy: {acc:.3f}")

    def predict(self, feat_df, engineer) -> np.ndarray:
        """Return combined LSTM+RF probability array (last N samples)."""
        X_seq, _ = engineer.make_sequences(feat_df)
        X_flat, _ = engineer.make_flat(feat_df)
        return self._combine(X_seq, X_flat, engineer)

    def predict_latest(self, feat_df, engineer) -> float:
        """Return combined probability for the most recent bar (scalar)."""
        lookback = self.lstm_model.lookback
        window   = feat_df[engineer.FEATURE_COLS].values[-lookback:]
        window_s = engineer.scale(window.reshape(1, *window.shape))
        lstm_p   = float(self.lstm_model.predict(window_s)[0])

        flat_w   = feat_df[engineer.FEATURE_COLS].values[-1:]
        flat_s   = engineer.scale(flat_w)
        rf_p     = float(self.rf_model.predict(flat_s)[0])

        combined = self.lstm_weight * lstm_p + self.rf_weight * rf_p
        log.debug(f"Ensemble: LSTM={lstm_p:.3f} RF={rf_p:.3f} → {combined:.3f}")
        return combined

    def _combine(self, X_seq: np.ndarray, X_flat: np.ndarray, engineer) -> np.ndarray:
        lstm_proba = self.lstm_model.predict(X_seq)
        rf_proba   = self.rf_model.predict(X_flat[-len(lstm_proba):])
        min_len    = min(len(lstm_proba), len(rf_proba))
        return (self.lstm_weight * lstm_proba[:min_len] +
                self.rf_weight   * rf_proba[:min_len])

    def save(self):
        self.lstm_model.save()
        self.rf_model.save()

    def load(self):
        self.lstm_model.load()
        self.rf_model.load()


# ==============================================================================
#  5. RISK MANAGER
# ==============================================================================
class RiskManager:
    """
    Handles position sizing, drawdown checks and spread filtering.

    Position sizing: Risk-pct of balance / (SL_pips * pip_value)
    """

    def __init__(self, cfg: dict, connector: MT5Connector):
        self.cfg       = cfg
        self.connector = connector
        self.peak_equity = connector.account_equity() if connector.connected else 0

    def compute_lot_size(self, sl_pips: float, symbol: str,
                         ml_proba: Optional[float] = None) -> float:
        """
        Dynamic lot sizing — three-layer formula:

        1. Base: risk-% of balance / (SL_pips × pip_value)  [unchanged core]
        2. Confidence scale: lots scale 0.70 → 1.30 with ML signal strength.
           Borderline signals (prob ≈ 0.58) trade smaller; high-conviction
           signals (prob ≈ 0.80+) trade larger — aligns size with edge quality.
        3. Progressive drawdown scaling: when equity drawdown exceeds 5 %,
           lot size is linearly reduced toward 50 % at the 10 % drawdown limit.
           This creates a natural circuit-breaker during losing stretches.

        Falls back to cfg['lot_size'] if MT5 is unavailable.
        """
        if not MT5_AVAILABLE or not self.connector.connected:
            return self.cfg["lot_size"]

        balance = self.connector.account_balance()
        risk_amount = balance * (self.cfg["risk_pct"] / 100)

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return self.cfg["lot_size"]

        # pip value in account currency per lot
        pip_value = sym_info.trade_tick_value * 10  # 1 pip = 10 ticks for 5-digit

        if pip_value <= 0 or sl_pips <= 0:
            return self.cfg["lot_size"]

        base_lot = risk_amount / (sl_pips * pip_value)

        # ── Layer 2: confidence scaling ────────────────────────────────────
        if ml_proba is not None:
            confidence = abs(ml_proba - 0.5) * 2   # 0 = borderline, 1 = certain
            # Linear scale: 0.70 at min confidence, 1.30 at max
            conf_scale = 0.70 + confidence * 0.60
            base_lot *= conf_scale

        # ── Layer 3: progressive drawdown scaling ──────────────────────────
        equity = self.connector.account_equity()
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
            if dd_pct > 5.0:
                # Linearly shrink from 1.0× at 5% DD to 0.5× at 10% DD
                dd_scale = max(0.5, 1.0 - (dd_pct - 5.0) / 10.0)
                base_lot *= dd_scale
                log.debug(
                    f"[Risk] DD={dd_pct:.1f}% → lot scaled by {dd_scale:.2f}"
                )

        lot = max(sym_info.volume_min,
                  min(round(base_lot, 2), sym_info.volume_max))
        return lot

    def check_drawdown(self) -> bool:
        """Return True if within acceptable drawdown limits."""
        if not MT5_AVAILABLE or not self.connector.connected:
            return True

        equity = self.connector.account_equity()
        self.peak_equity = max(self.peak_equity, equity)
        drawdown_pct = (self.peak_equity - equity) / self.peak_equity * 100

        if drawdown_pct >= self.cfg["max_drawdown_pct"]:
            log.warning(
                f"⚠️  Max drawdown reached: {drawdown_pct:.2f}% "
                f"(limit {self.cfg['max_drawdown_pct']}%). Trading paused."
            )
            return False
        return True

    def spread_ok(self, symbol: str) -> bool:
        """Return True if current spread is within allowed limit.

        Uses the instrument's pip size for the conversion so that JPY pairs
        (pip=0.01), Gold (pip=0.01) and standard forex (pip=0.0001) are all
        handled correctly.  Per-instrument overrides in cfg['max_spread_overrides']
        take priority over the global cfg['max_spread'].
        """
        if not MT5_AVAILABLE:
            return True
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        pip = DataDownloader.pip_size(symbol)
        spread_pips = (tick.ask - tick.bid) / pip
        limit = self.cfg.get("max_spread_overrides", {}).get(
            symbol.upper(), self.cfg["max_spread"]
        )
        if spread_pips > limit:
            log.debug(f"Spread {spread_pips:.1f} pips > limit {limit} for {symbol}. Skipping.")
            return False
        return True

    def open_positions_count(self, symbol: str) -> int:
        if not MT5_AVAILABLE:
            return 0
        positions = mt5.positions_get(symbol=symbol)
        return len(positions) if positions else 0


# ==============================================================================
#  6. ORDER EXECUTOR
# ==============================================================================
class OrderExecutor:
    """Opens and closes market orders on MT5."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _pip_factor(self, symbol: str) -> float:
        """Returns the pip size for a given symbol."""
        if not MT5_AVAILABLE:
            return 0.0001
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0001
        # Standard: 5-digit pair → 1 pip = 10 * point
        return info.point * 10

    def open_buy(self, symbol: str, lot: float, sl_pips: float, tp_pips: float) -> bool:
        """Market buy with SL and TP."""
        if not MT5_AVAILABLE:
            log.info(f"[SIM] BUY {lot} lots {symbol}")
            return True

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            log.error("Could not get tick for BUY order.")
            return False

        pip = self._pip_factor(symbol)
        price = tick.ask
        sl    = round(price - sl_pips * pip, 5)
        tp    = round(price + tp_pips * pip, 5)

        request = {
            "action":        mt5.TRADE_ACTION_DEAL,
            "symbol":        symbol,
            "volume":        lot,
            "type":          mt5.ORDER_TYPE_BUY,
            "price":         price,
            "sl":            sl,
            "tp":            tp,
            "deviation":     10,
            "magic":         self.cfg["magic_number"],
            "comment":       "ForexAIBot_BUY",
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"BUY failed: {result.retcode} — {result.comment}")
            return False

        log.info(f"✅ BUY  | {symbol} | {lot} lots | price={price} sl={sl} tp={tp}")
        return True

    def open_sell(self, symbol: str, lot: float, sl_pips: float, tp_pips: float) -> bool:
        """Market sell with SL and TP."""
        if not MT5_AVAILABLE:
            log.info(f"[SIM] SELL {lot} lots {symbol}")
            return True

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            log.error("Could not get tick for SELL order.")
            return False

        pip = self._pip_factor(symbol)
        price = tick.bid
        sl    = round(price + sl_pips * pip, 5)
        tp    = round(price - tp_pips * pip, 5)

        request = {
            "action":        mt5.TRADE_ACTION_DEAL,
            "symbol":        symbol,
            "volume":        lot,
            "type":          mt5.ORDER_TYPE_SELL,
            "price":         price,
            "sl":            sl,
            "tp":            tp,
            "deviation":     10,
            "magic":         self.cfg["magic_number"],
            "comment":       "ForexAIBot_SELL",
            "type_time":     mt5.ORDER_TIME_GTC,
            "type_filling":  mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"SELL failed: {result.retcode} — {result.comment}")
            return False

        log.info(f"✅ SELL | {symbol} | {lot} lots | price={price} sl={sl} tp={tp}")
        return True

    def close_all(self, symbol: str):
        """Close all open positions for a symbol (emergency exit)."""
        if not MT5_AVAILABLE:
            return
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        for pos in positions:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       pos.volume,
                "type":         close_type,
                "position":     pos.ticket,
                "price":        price,
                "deviation":    10,
                "magic":        self.cfg["magic_number"],
                "comment":      "ForexAIBot_CLOSE",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"Closed position #{pos.ticket}")
            else:
                log.error(f"Close failed #{pos.ticket}: {result.comment}")


# ==============================================================================
#  6b.  MARKET REGIME DETECTOR
# ==============================================================================
class MarketRegimeDetector:
    """
    Classifies the current market into one of three regimes every tick:

      TRENDING  — strong directional move (ADX > 25)
      RANGING   — low volatility, price contained in band (ADX < 20 + narrow BB)
      VOLATILE  — wide bands + high ATR, no clear direction

    Regime is used by the agent to:
      - TRENDING : use standard threshold (0.58) — trend-following works
      - RANGING  : raise threshold to 0.62 (harder to call direction)
      - VOLATILE : raise threshold to 0.65 and halve lot size (high risk)
    """

    TRENDING  = "TRENDING"
    RANGING   = "RANGING"
    VOLATILE  = "VOLATILE"

    def detect(self, feat_df: pd.DataFrame) -> str:
        """Return regime string based on the last bar of feat_df."""
        try:
            adx = self._adx(feat_df)
            bb_width = feat_df["bb_width"].iloc[-1] if "bb_width" in feat_df.columns else None
            atr_norm = self._atr_norm(feat_df)

            if adx is not None and adx > 25:
                return self.TRENDING

            if bb_width is not None and bb_width < 0.03 and (adx is None or adx < 20):
                return self.RANGING

            if atr_norm is not None and atr_norm > 1.5:
                return self.VOLATILE

            return self.TRENDING  # default: treat as trending
        except Exception:
            return self.TRENDING

    @staticmethod
    def _adx(df: pd.DataFrame) -> Optional[float]:
        """Approximate ADX from high/low/close if available."""
        if len(df) < 30:
            return None
        if "adx" in df.columns:
            v = df["adx"].iloc[-1]
            return float(v) if not pd.isna(v) else None
        # Compute a simplified ADX on the fly
        try:
            from ta.trend import ADXIndicator
            adx_ind = ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
            v = adx_ind.adx().iloc[-1]
            return float(v) if not pd.isna(v) else None
        except Exception:
            return None

    @staticmethod
    def _atr_norm(df: pd.DataFrame) -> Optional[float]:
        """ATR relative to its 20-bar rolling mean (values > 1 = above-average vol)."""
        if "atr" not in df.columns or len(df) < 21:
            return None
        atr_series = df["atr"].dropna()
        if len(atr_series) < 21:
            return None
        mean = atr_series.rolling(20).mean().iloc[-1]
        if mean == 0 or pd.isna(mean):
            return None
        return float(atr_series.iloc[-1]) / float(mean)

    def threshold_for(self, regime: str, base: float = 0.58) -> float:
        """Return the adjusted confidence threshold for the given regime."""
        return {
            self.TRENDING:  base,
            self.RANGING:   min(base + 0.04, 0.70),
            self.VOLATILE:  min(base + 0.07, 0.75),
        }.get(regime, base)

    def lot_multiplier_for(self, regime: str) -> float:
        """Position size multiplier: 1.0 normal, 0.5 for volatile."""
        return {
            self.TRENDING:  1.0,
            self.RANGING:   0.75,
            self.VOLATILE:  0.5,
        }.get(regime, 1.0)


# ==============================================================================
#  6c.  AGENT MEMORY  (adaptive threshold from recent win-rate)
# ==============================================================================
class AgentMemory:
    """
    Lightweight rolling memory of the last N trade outcomes.
    Raises the confidence threshold when win-rate is low, and relaxes it
    slightly when it is high — a simple RL-style feedback mechanism.

    Usage:
        memory.record_outcome(won=True)
        adjusted = memory.adjust_threshold(base_threshold)
    """

    def __init__(self, window: int = 20):
        self._outcomes: List[bool] = []   # True = won, False = lost
        self._window  = window

    def record_outcome(self, won: bool):
        self._outcomes.append(won)
        if len(self._outcomes) > self._window:
            self._outcomes.pop(0)

    @property
    def win_rate(self) -> float:
        if not self._outcomes:
            return 0.5
        return sum(self._outcomes) / len(self._outcomes)

    def adjust_threshold(self, base: float) -> float:
        """
        Win-rate < 40% → raise threshold by 0.05 (be more selective).
        Win-rate > 65% → lower threshold by 0.02 (take more trades).
        Otherwise       → no change.
        """
        wr = self.win_rate
        if len(self._outcomes) < 5:
            return base          # not enough history yet
        if wr < 0.40:
            return min(base + 0.05, 0.80)
        if wr > 0.65:
            return max(base - 0.02, 0.52)
        return base

    def summary(self) -> str:
        n = len(self._outcomes)
        return (f"AgentMemory: {n} trades | win-rate={self.win_rate:.0%} "
                f"(window={self._window})")


# ==============================================================================
#  6d.  TRAILING STOP MANAGER
# ==============================================================================
class TrailingStopManager:
    """
    Manages trailing stops for open positions.

    Two-stage trailing logic per position:
      Stage 1 — Move SL to break-even once price moves 1× ATR in our favour.
      Stage 2 — Trail SL at 1.5× ATR behind price once in 2× ATR profit.

    Call `update_all(symbol, atr_pips)` once per live tick for each pair.
    """

    def __init__(self):
        # Tickets that have already received a partial close — prevents double-close
        self._partial_closed: set = set()

    def update_all(self, symbol: str, atr_pips: float):
        """Iterate open positions for symbol and apply trailing + partial-close logic."""
        if not MT5_AVAILABLE:
            return
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        for pos in positions:
            try:
                self._update_one(pos, atr_pips)
            except Exception as exc:
                log.debug(f"Trailing stop update error for #{pos.ticket}: {exc}")

    def _update_one(self, pos, atr_pips: float):
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return

        sym_info = mt5.symbol_info(pos.symbol)
        if sym_info is None:
            return

        pip = DataDownloader.pip_size(pos.symbol)
        atr_dist = atr_pips * pip
        is_buy   = (pos.type == mt5.ORDER_TYPE_BUY)
        cur_price = tick.bid if is_buy else tick.ask

        profit_dist = (cur_price - pos.price_open) if is_buy else (pos.price_open - cur_price)

        # ── Stage 0: Partial close at 1.5× ATR profit ─────────────────────
        # Harvest 50 % of the position early to lock in profit; let the
        # remaining half run to full TP.  Only fires once per position.
        half_vol = round(pos.volume / 2, 2)
        if (
            profit_dist >= atr_dist * 1.5
            and pos.ticket not in self._partial_closed
            and pos.volume > sym_info.volume_min * 2     # need at least 2 × min lot
            and half_vol >= sym_info.volume_min
        ):
            if TrailingStopManager._partial_close(pos, half_vol):
                self._partial_closed.add(pos.ticket)
                log.info(
                    f"[PartialClose] #{pos.ticket} {pos.symbol} — "
                    f"closed {half_vol} lots at 1.5×ATR profit, remainder runs."
                )
            return

        # Stage 1: move to break-even
        be_sl  = round(pos.price_open + (sym_info.point * 2) * (1 if is_buy else -1), 5)
        if profit_dist >= atr_dist and (
            (is_buy and pos.sl < pos.price_open) or
            (not is_buy and (pos.sl > pos.price_open or pos.sl == 0.0))
        ):
            new_sl = be_sl
            TrailingStopManager._send_sl_update(pos, new_sl)
            log.debug(f"Trailing [{pos.symbol} #{pos.ticket}] → BE at {new_sl:.5f}")
            return

        # Stage 2: trail at 1.5× ATR
        trail_dist = atr_dist * 1.5
        if profit_dist >= atr_dist * 2:
            if is_buy:
                new_sl = round(cur_price - trail_dist, 5)
                if new_sl > pos.sl:
                    TrailingStopManager._send_sl_update(pos, new_sl)
                    log.debug(f"Trailing [{pos.symbol} #{pos.ticket}] BUY SL→{new_sl:.5f}")
            else:
                new_sl = round(cur_price + trail_dist, 5)
                if pos.sl == 0.0 or new_sl < pos.sl:
                    TrailingStopManager._send_sl_update(pos, new_sl)
                    log.debug(f"Trailing [{pos.symbol} #{pos.ticket}] SELL SL→{new_sl:.5f}")

    @staticmethod
    def _partial_close(pos, close_volume: float) -> bool:
        """Send a partial close order for `close_volume` lots of `pos`."""
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       close_volume,
            "type":         close_type,
            "position":     pos.ticket,
            "price":        price,
            "deviation":    10,
            "comment":      "ForexAIBot_PARTIAL",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    @staticmethod
    def _send_sl_update(pos, new_sl: float):
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol":   pos.symbol,
            "sl":       new_sl,
            "tp":       pos.tp,
        }
        mt5.order_send(request)


# ==============================================================================
#  6f.  CORRELATION FILTER
# ==============================================================================
# Static currency-bloc map.  Pairs within the same bloc move in lockstep when
# the shared currency is driven by a macro event (e.g. USD NFP).  Opening
# multiple positions from the same bloc effectively multiplies leverage on a
# single view.
_CORR_BLOCS: Dict[str, List[str]] = {
    "USD_LONG":   ["USDCAD", "USDJPY", "USDCHF"],          # USD goes up with these
    "USD_SHORT":  ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],# USD goes up = these go down
    "EUR_BLOCK":  ["EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD"],
    "GBP_BLOCK":  ["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD"],
    "ANTIPODEAN": ["AUDUSD", "NZDUSD", "AUDNZD", "AUDCAD", "AUDCHF",
                   "NZDCAD", "NZDCHF", "NZDJPY", "AUDJPY"],
}


class CorrelationFilter:
    """
    Blocks or scales down new trades that are correlated with already-open positions.

    Logic per new signal on `pair`:
      0 correlated positions open → scale = 1.0   (full lot)
      1 correlated position open  → scale = 0.60  (60% lot)
      2 correlated positions open → scale = 0.35  (35% lot)
      3+ correlated positions     → scale = 0.0   (blocked)

    A "correlated position" is one that shares at least one currency bloc with
    the candidate pair.
    """

    def lot_scale_factor(self, new_pair: str, open_symbols: List[str]) -> float:
        overlap = self._count_correlated(new_pair, open_symbols)
        if overlap == 0:
            return 1.0
        if overlap == 1:
            return 0.60
        if overlap == 2:
            return 0.35
        return 0.0   # 3 or more correlated — block entirely

    @staticmethod
    def _count_correlated(pair: str, open_symbols: List[str]) -> int:
        pair_up = pair.upper()
        blocs = [b for b, members in _CORR_BLOCS.items() if pair_up in members]
        if not blocs:
            return 0
        count = 0
        seen: set = set()
        for sym in open_symbols:
            sym_up = sym.upper()
            if sym_up == pair_up or sym_up in seen:
                continue
            for bloc in blocs:
                if sym_up in _CORR_BLOCS[bloc]:
                    count += 1
                    seen.add(sym_up)
                    break
        return count


# ==============================================================================
#  7. BACKTESTER
# ==============================================================================
class Backtester:
    """
    Vectorised backtester.
    Simulates the bot's trading signals on historical data and reports:
      - Win rate, total trades
      - Net PnL in pips
      - Max drawdown
      - Sharpe Ratio
      - Monte Carlo significance test (5,000 permutations)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        threshold: float = 0.55,
    ) -> pd.DataFrame:
        """
        df       : DataFrame with Close prices (and 'atr' column if available)
        signals  : probability of UP move [0,1] per bar
        threshold: confidence threshold for entering a trade

        Dynamic SL/TP: if 'atr' column present, SL = atr_sl_mult * ATR,
        TP = atr_tp_mult * ATR.  Falls back to fixed pips if ATR unavailable.
        """
        pip = DataDownloader.pip_size(self.cfg.get("symbol", "EURUSD"))
        lot = self.cfg["lot_size"]
        has_atr = "atr" in df.columns

        atr_sl_mult = self.cfg.get("atr_sl_mult", 1.5)
        atr_tp_mult = self.cfg.get("atr_tp_mult", 3.0)

        results = []
        balance = 10_000.0
        equity_curve = [balance]
        peak = balance

        for i in range(len(signals)):
            signal = signals[i]
            close  = df["Close"].iloc[i]

            if signal > threshold:
                direction = 1
            elif signal < (1 - threshold):
                direction = -1
            else:
                equity_curve.append(balance)
                continue

            if i + 1 >= len(df):
                break

            # ── Dynamic SL/TP based on ATR ────────────────────────────────
            if has_atr:
                atr_val  = df["atr"].iloc[i]
                sl_dist  = atr_sl_mult * atr_val
                tp_dist  = atr_tp_mult * atr_val
                sl_pips_ = sl_dist / pip
                tp_pips_ = tp_dist / pip
            else:
                sl_pips_ = self.cfg["sl_pips"]
                tp_pips_ = self.cfg["tp_pips"]
                sl_dist  = sl_pips_ * pip
                tp_dist  = tp_pips_ * pip

            next_close = df["Close"].iloc[i + 1]
            price_move = (next_close - close) * direction

            if price_move >= tp_dist:
                pnl_pips = tp_pips_
                outcome  = "TP"
            elif price_move <= -sl_dist:
                pnl_pips = -sl_pips_
                outcome  = "SL"
            else:
                pnl_pips = price_move / pip
                outcome  = "Close"

            pnl_usd = pnl_pips * 10 * lot
            balance += pnl_usd
            peak     = max(peak, balance)
            dd       = (peak - balance) / peak * 100

            results.append({
                "time":      df.index[i],
                "direction": "BUY" if direction == 1 else "SELL",
                "signal":    round(signal, 4),
                "sl_pips":   round(sl_pips_, 1),
                "tp_pips":   round(tp_pips_, 1),
                "pnl_pips":  round(pnl_pips, 1),
                "pnl_usd":   round(pnl_usd, 2),
                "balance":   round(balance, 2),
                "drawdown":  round(dd, 2),
                "outcome":   outcome,
            })
            equity_curve.append(balance)

        trades_df = pd.DataFrame(results)
        if trades_df.empty:
            log.warning("Backtester: No trades generated.")
            return trades_df

        self._print_stats(trades_df, equity_curve)
        self._plot(trades_df, df)
        return trades_df

    # ── Stats ─────────────────────────────────────────────────────────────
    def _print_stats(self, trades: pd.DataFrame, equity: list):
        total   = len(trades)
        wins    = (trades["pnl_pips"] > 0).sum()
        losses  = (trades["pnl_pips"] < 0).sum()
        win_pct = wins / total * 100 if total else 0
        net_pip = trades["pnl_pips"].sum()
        net_usd = trades["pnl_usd"].sum()
        max_dd  = trades["drawdown"].max()

        # Sharpe (annualised, daily returns approx)
        returns = trades["pnl_usd"].values
        sharpe  = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)

        # Profit factor
        gross_profit = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum()
        gross_loss   = abs(trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy per trade
        expectancy = net_usd / total if total else 0.0

        log.info("=" * 60)
        log.info("  BACKTEST RESULTS")
        log.info("=" * 60)
        log.info(f"  Total trades  : {total}")
        log.info(f"  Wins / Losses : {wins} / {losses}  ({win_pct:.1f}% win rate)")
        log.info(f"  Net PnL       : {net_pip:.1f} pips  /  ${net_usd:.2f}")
        log.info(f"  Expectancy    : ${expectancy:.2f} per trade")
        log.info(f"  Profit Factor : {profit_factor:.2f}")
        log.info(f"  Max Drawdown  : {max_dd:.2f}%")
        log.info(f"  Sharpe Ratio  : {sharpe:.2f}")

        # ── Monte Carlo significance test ──────────────────────────────────
        mc = self._monte_carlo(returns, sharpe)
        if mc["p_value"] is not None:
            sig = "SIGNIFICANT" if mc["significant"] else "NOT significant"
            log.info(
                f"  Monte Carlo   : p={mc['p_value']:.4f} ({sig} at 5%) | "
                f"95th-pct random Sharpe={mc['sharpe_95th']:.3f}"
            )
        log.info("=" * 60)

    @staticmethod
    def _monte_carlo(returns: np.ndarray, observed_sharpe: float,
                     n: int = 5000) -> Dict:
        """
        Permutation test: shuffle trade P&L sequence `n` times and compute
        Sharpe for each shuffle.  Reports p-value = fraction of random Sharpes
        that exceed the observed Sharpe.  p < 0.05 means the strategy edge
        is unlikely to be random luck.
        """
        if len(returns) < 10:
            return {"p_value": None, "sharpe_95th": None, "significant": False}
        rng = np.random.default_rng(42)
        shuffled = np.array([
            (np.mean(rng.permutation(returns)) /
             (np.std(rng.permutation(returns)) + 1e-9)) * np.sqrt(252)
            for _ in range(n)
        ])
        p_value    = float(np.mean(shuffled >= observed_sharpe))
        sharpe_95  = float(np.percentile(shuffled, 95))
        return {
            "p_value":     round(p_value, 4),
            "sharpe_95th": round(sharpe_95, 3),
            "significant": p_value < 0.05,
        }

    # ── Equity curve plot ─────────────────────────────────────────────────
    def _plot(self, trades: pd.DataFrame, price_df: pd.DataFrame):
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            fig.suptitle(
                f"Backtest — {CONFIG['symbol']} {CONFIG['timeframe']}",
                fontsize=14, fontweight="bold"
            )

            # Price chart with trade markers
            ax1 = axes[0]
            ax1.plot(price_df.index, price_df["Close"], linewidth=0.8,
                     color="#1a1a2e", label="Close")
            buys  = trades[trades["direction"] == "BUY"]
            sells = trades[trades["direction"] == "SELL"]
            ax1.scatter(buys["time"],  price_df.loc[buys["time"],  "Close"],
                        marker="^", color="#2ecc71", s=20, zorder=5, label="BUY")
            ax1.scatter(sells["time"], price_df.loc[sells["time"], "Close"],
                        marker="v", color="#e74c3c", s=20, zorder=5, label="SELL")
            ax1.set_ylabel("Price")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)

            # Equity curve
            ax2 = axes[1]
            ax2.plot(trades["time"], trades["balance"],
                     color="#3498db", linewidth=1.5)
            ax2.fill_between(trades["time"], trades["balance"].min(),
                             trades["balance"], alpha=0.15, color="#3498db")
            ax2.set_ylabel("Balance ($)")
            ax2.set_title("Equity Curve")
            ax2.grid(alpha=0.3)

            # Drawdown
            ax3 = axes[2]
            ax3.fill_between(trades["time"], 0, -trades["drawdown"],
                             color="#e74c3c", alpha=0.5)
            ax3.set_ylabel("Drawdown (%)")
            ax3.set_xlabel("Time")
            ax3.set_title("Drawdown")
            ax3.grid(alpha=0.3)

            plt.tight_layout()
            out = "logs/backtest_results.png"
            plt.savefig(out, dpi=120)
            log.info(f"Backtest chart saved → {out}")
            plt.close()
        except Exception as e:
            log.warning(f"Chart generation skipped: {e}")


# ==============================================================================
#  8. MAIN BOT ORCHESTRATOR
# ==============================================================================
class ForexAIBot:
    """
    Top-level orchestrator v2.0:
      - train        : download data → engineer features → train model
      - backtest     : train + vectorised backtest
      - live         : connect MT5 → poll with intelligence layer → trade
      - train-books  : ingest PDFs/text books into knowledge base
      - quotes       : display live prices for all 28 pairs
      - news         : fetch and display current news sentiment
      - fundamentals : display fundamental analysis report
    """

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.connector  = MT5Connector(cfg)
        self.downloader = DataDownloader(cfg["symbol"], cfg["timeframe"], cfg["history_bars"])
        self.engineer   = FeatureEngineer(cfg["lstm_lookback"])
        self.executor   = OrderExecutor(cfg)
        self.risk       = RiskManager(cfg, self.connector)
        self.backtester = Backtester(cfg)

        # Choose model — fallback to RF automatically if TF not installed
        n_feat = len(self.engineer.FEATURE_COLS)
        model_type = cfg["model_type"]
        if model_type in ("LSTM", "ENSEMBLE") and not TF_AVAILABLE:
            log.warning(
                "[Bot] TensorFlow not installed — falling back to Random Forest. "
                "Install TF with: pip install tensorflow"
            )
            model_type = "RF"
        self._model_type = model_type   # resolved type (may differ from cfg)
        if model_type == "LSTM":
            self.model = LSTMModel(cfg["lstm_lookback"], n_feat, cfg["model_path"])
        elif model_type == "ENSEMBLE":
            self.model = EnsembleModel(cfg["lstm_lookback"], n_feat, cfg["model_path"])
        else:
            self.model = RFModel(cfg["model_path"])

        self.scaler_path = os.path.join(cfg["model_path"], "scaler.pkl")

        # ── Agent components (v3.1) ───────────────────────────────────────
        self.regime_detector  = MarketRegimeDetector()
        self.agent_memory     = AgentMemory(window=cfg.get("agent_memory_window", 20))
        self.trailing_stops   = TrailingStopManager()
        self.correlation_filter = CorrelationFilter()

        # ── Intelligence Layer (v2.0) ─────────────────────────────────────
        self.news_engine: Optional[object] = None
        self.ecal: Optional[object]        = None
        self.fundamental: Optional[object] = None
        self.knowledge_base: Optional[object] = None

        if NEWS_AVAILABLE and cfg.get("use_news", True):
            self.news_engine = NewsEngine()
            self.ecal        = EconomicCalendar()
            log.info("[Bot] News engine loaded.")

        if FUNDAMENTAL_AVAILABLE and cfg.get("use_fundamentals", True):
            self.fundamental = FundamentalEngine(pairs=cfg.get("pairs", None))
            log.info("[Bot] Fundamental engine loaded.")

        if BOOKS_AVAILABLE and cfg.get("use_knowledge_base", True):
            self.knowledge_base = KnowledgeBase()
            # Auto-load classic rules if KB is empty
            if len(self.knowledge_base.rules) == 0:
                log.info("[Bot] Knowledge base empty — loading classic trading rules.")
                load_classic_rules(self.knowledge_base)
            # Auto-scan books folder on init
            books_folder = cfg.get("books_folder", "books/")
            if os.path.isdir(books_folder):
                trainer = BookTrainer(self.knowledge_base)
                added = trainer.ingest_folder(books_folder)
                if added > 0:
                    trainer.save()
            log.info(f"[Bot] Knowledge base: {len(self.knowledge_base.rules)} rules loaded.")

        # ── Social Engine (v3.0) ──────────────────────────────────────────────
        self.social: Optional[SocialEngine] = None
        self.paused: bool = False   # set True via /pause Telegram command
        if SOCIAL_AVAILABLE and cfg.get("social", {}).get("enabled", True):
            self.social = SocialEngine(cfg)
            log.info("[Bot] Social engine loaded.")

        # ── Multi-Account Manager (v3.0) ──────────────────────────────────────
        self.multi_account: Optional[MultiAccountManager] = None
        active_accts = [a for a in cfg.get("accounts", [])
                        if a.get("active", True) and a.get("login", 0)]
        if MULTI_ACCOUNT_AVAILABLE and len(active_accts) > 1:
            self.multi_account = MultiAccountManager(active_accts)
            log.info(f"[Bot] Multi-account manager: {len(active_accts)} account(s).")

    # ── A. Train ──────────────────────────────────────────────────────────
    def train(self):
        log.info("─── MODE: TRAIN ───────────────────────────────────────")

        # Fetch data
        raw_df = self.downloader.fetch()
        if raw_df.empty:
            log.error("No data to train on. Exiting.")
            return

        # Feature engineering
        feat_df = self.engineer.build(raw_df)
        log.info(f"Feature matrix: {feat_df.shape}")

        if self._model_type == "LSTM":
            # ── Walk-forward validation (3 expanding folds) ──────────────
            # Each fold trains on [0:train_end], validates on [train_end:val_end]
            # Final model is trained on the first 85% and the last 15% is hold-out
            folds = [
                (0.50, 0.65),   # fold 1: train on 50%, validate on next 15%
                (0.60, 0.75),   # fold 2: train on 60%, validate on next 15%
                (0.70, 0.85),   # fold 3: train on 70%, validate on next 15%
            ]
            n = len(feat_df)
            # Fit scaler on first 50% only to avoid future leakage
            scaler_end = int(n * 0.50)
            X_raw_scaler = feat_df.iloc[:scaler_end][self.engineer.FEATURE_COLS].values
            self.engineer.scale(X_raw_scaler, fit=True)
            self.engineer.save_scaler(self.scaler_path)

            val_scores: List[float] = []
            for fold_idx, (tr_ratio, val_ratio) in enumerate(folds, 1):
                tr_end  = int(n * tr_ratio)
                val_end = int(n * val_ratio)
                fold_df_tr  = feat_df.iloc[:tr_end]
                fold_df_val = feat_df.iloc[tr_end:val_end]
                X_tr, y_tr   = self.engineer.make_sequences(fold_df_tr)
                X_val, y_val = self.engineer.make_sequences(fold_df_val)
                if len(X_tr) < 50 or len(X_val) < 10:
                    continue
                log.info(f"Walk-forward fold {fold_idx}: train={len(X_tr)}, val={len(X_val)}")
                self.model.train(X_tr, y_tr, X_val, y_val, epochs=30)
                preds = (self.model.predict(X_val) > 0.5).astype(int)
                acc   = (preds == y_val).mean()
                log.info(f"  Fold {fold_idx} val accuracy: {acc:.3f}")
                val_scores.append(acc)

            if val_scores:
                log.info(f"Walk-forward mean accuracy: {np.mean(val_scores):.3f} "
                         f"(±{np.std(val_scores):.3f})")

            # Final model: train on 85%, hold-out test on last 15%
            final_end = int(n * 0.85)
            X_final, y_final   = self.engineer.make_sequences(feat_df.iloc[:final_end])
            X_holdout, y_hold  = self.engineer.make_sequences(feat_df.iloc[final_end:])
            val_data = (X_holdout, y_hold) if len(X_holdout) > 10 else None
            split = int(len(X_final) * 0.9)
            Xf_tr, Xf_val = X_final[:split], X_final[split:]
            yf_tr, yf_val = y_final[:split], y_final[split:]
            log.info(f"Final model: train={len(Xf_tr)}, val={len(Xf_val)}, hold-out={len(X_holdout)}")
            self.model.train(Xf_tr, yf_tr, Xf_val, yf_val, epochs=50)
            self.model.save()

            if val_data is not None:
                h_preds = (self.model.predict(X_holdout) > 0.5).astype(int)
                log.info(f"Hold-out accuracy: {(h_preds == y_hold).mean():.3f}")
                log.info("\n" + classification_report(y_hold, h_preds, target_names=["DOWN", "UP"]))

        elif self._model_type == "ENSEMBLE":
            self._train_ensemble(feat_df)

        else:  # Random Forest
            n = len(feat_df)
            X_raw = feat_df[self.engineer.FEATURE_COLS].values
            y_raw = feat_df["target"].values
            # Fit scaler on first 50% to avoid future leakage
            self.engineer.scale(X_raw[:int(n * 0.50)], fit=True)
            self.engineer.save_scaler(self.scaler_path)
            X = self.engineer.scale(X_raw)
            # Walk-forward: 70/15/15 split
            tr_end  = int(n * 0.70)
            val_end = int(n * 0.85)
            X_tr,  y_tr  = X[:tr_end],      y_raw[:tr_end]
            X_val, y_val = X[tr_end:val_end], y_raw[tr_end:val_end]
            X_hld, y_hld = X[val_end:],      y_raw[val_end:]
            self.model.train(X_tr, y_tr)
            for label, Xp, yp in [("Val", X_val, y_val), ("Hold-out", X_hld, y_hld)]:
                if len(Xp) == 0:
                    continue
                p = (self.model.predict(Xp) > 0.5).astype(int)
                log.info(f"{label} report:\n" + classification_report(yp, p, target_names=["DOWN", "UP"]))
            self.model.save()

        log.info("Training complete ✓")

    # ── A2. Train All Pairs ───────────────────────────────────────────────
    def train_all_pairs(self):
        """
        Train a separate model for every pair in cfg['pairs'].
        Each pair gets its own model directory: models/{PAIR}/
        """
        pairs = self.cfg.get("pairs", [self.cfg["symbol"]])
        log.info(f"─── MODE: TRAIN-ALL ({len(pairs)} pairs) ──────────────────")
        ok, failed = [], []
        for pair in pairs:
            log.info(f"\n{'═'*55}\n  Training: {pair}\n{'═'*55}")
            try:
                self._train_or_backtest_pair(pair, mode="train")
                ok.append(pair)
            except Exception as exc:
                log.error(f"{pair} training failed: {exc}")
                failed.append(f"{pair}: {exc}")
        self._all_pairs_summary("TRAIN-ALL", ok, failed)

    def _train_or_backtest_pair(self, pair: str, mode: str = "train"):
        """
        Create an isolated sub-bot for `pair`, sharing intelligence layers
        from self, then run train() or backtest() on it.
        Each pair's artefacts are stored in models/{PAIR}/.
        """
        pair_model_dir = os.path.join(self.cfg["model_path"].rstrip("/\\"), pair) + os.sep
        os.makedirs(pair_model_dir, exist_ok=True)

        pair_cfg = dict(self.cfg)
        pair_cfg["symbol"]     = pair
        pair_cfg["model_path"] = pair_model_dir

        # Build a lightweight sub-bot — skip re-creating intelligence engines
        sub = ForexAIBot.__new__(ForexAIBot)
        sub.cfg         = pair_cfg
        sub.connector   = self.connector
        sub.executor    = self.executor
        sub.risk        = self.risk
        sub.news_engine    = self.news_engine
        sub.ecal           = self.ecal
        sub.fundamental    = self.fundamental
        sub.knowledge_base = self.knowledge_base
        sub.downloader  = DataDownloader(pair, pair_cfg["timeframe"], pair_cfg["history_bars"])
        sub.engineer    = FeatureEngineer(pair_cfg["lstm_lookback"])
        sub.backtester  = Backtester(pair_cfg)
        n_feat = len(sub.engineer.FEATURE_COLS)
        if pair_cfg["model_type"] == "LSTM":
            sub.model = LSTMModel(pair_cfg["lstm_lookback"], n_feat, pair_model_dir)
        elif pair_cfg["model_type"] == "ENSEMBLE":
            sub.model = EnsembleModel(pair_cfg["lstm_lookback"], n_feat, pair_model_dir)
        else:
            sub.model = RFModel(pair_model_dir)
        sub.scaler_path = os.path.join(pair_model_dir, "scaler.pkl")

        if mode == "train":
            sub.train()
        else:
            sub.backtest()

    @staticmethod
    def _all_pairs_summary(label: str, ok: list, failed: list):
        log.info(f"\n─── {label} SUMMARY ─────────────────────────────────")
        for p in ok:
            log.info(f"  {p:12s} ✓")
        for p in failed:
            log.info(f"  {p}")
        log.info(f"  Done: {len(ok)} succeeded, {len(failed)} failed.")

    def _train_ensemble(self, feat_df):
        """Walk-forward training for the EnsembleModel sub-models."""
        n = len(feat_df)
        scaler_end = int(n * 0.50)
        X_raw_sc = feat_df.iloc[:scaler_end][self.engineer.FEATURE_COLS].values
        self.engineer.scale(X_raw_sc, fit=True)
        self.engineer.save_scaler(self.scaler_path)

        # 3-fold walk-forward val
        folds = [(0.50, 0.65), (0.60, 0.75), (0.70, 0.85)]
        val_scores: List[float] = []
        for fold_idx, (tr_ratio, val_ratio) in enumerate(folds, 1):
            tr_end  = int(n * tr_ratio)
            val_end = int(n * val_ratio)
            fold_tr  = feat_df.iloc[:tr_end]
            fold_val = feat_df.iloc[tr_end:val_end]
            X_seq_tr, y_tr   = self.engineer.make_sequences(fold_tr)
            X_seq_val, y_val = self.engineer.make_sequences(fold_val)
            X_flat_tr, _     = self.engineer.make_flat(fold_tr)
            X_flat_val, _    = self.engineer.make_flat(fold_val)
            if len(X_seq_tr) < 50 or len(X_seq_val) < 5:
                continue
            log.info(f"Ensemble fold {fold_idx}: lstm_tr={len(X_seq_tr)}, rf_tr={len(X_flat_tr)}")
            self.model.lstm_model.train(X_seq_tr, y_tr, X_seq_val, y_val, epochs=25)
            self.model.rf_model.train(X_flat_tr[:len(X_seq_tr)], y_tr)
            # Combined val score
            lstm_p = self.model.lstm_model.predict(X_seq_val)
            rf_len = min(len(X_flat_val), len(lstm_p))
            rf_p   = self.model.rf_model.predict(X_flat_val[-rf_len:])
            min_l  = min(len(lstm_p), len(rf_p))
            comb   = (0.6 * lstm_p[:min_l] + 0.4 * rf_p[:min_l])
            preds  = (comb > 0.5).astype(int)
            acc    = (preds == y_val[:min_l]).mean()
            log.info(f"  Fold {fold_idx} ensemble val accuracy: {acc:.3f}")
            val_scores.append(acc)

        if val_scores:
            log.info(f"Ensemble walk-forward mean accuracy: {np.mean(val_scores):.3f}")

        # Final training on 85%
        final_end = int(n * 0.85)
        self.model.train(feat_df.iloc[:final_end], self.engineer, epochs=50)
        self.model.save()

    # ── B. Backtest ───────────────────────────────────────────────────────
    def backtest(self):
        log.info("─── MODE: BACKTEST ────────────────────────────────────")
        self.train()   # train first

        raw_df  = self.downloader.fetch()
        feat_df = self.engineer.build(raw_df)

        self.engineer.load_scaler(self.scaler_path)
        self.model.load()

        if self._model_type == "LSTM":
            X, _ = self.engineer.make_sequences(feat_df)
            signals = self.model.predict(X)
            # Align with original df (first `lookback` rows were consumed)
            aligned_df = feat_df.iloc[self.cfg["lstm_lookback"]: self.cfg["lstm_lookback"] + len(signals)]
        elif self._model_type == "ENSEMBLE":
            signals    = self.model.predict(feat_df, self.engineer)
            lookback   = self.cfg["lstm_lookback"]
            aligned_df = feat_df.iloc[lookback: lookback + len(signals)]
        else:
            X, _ = self.engineer.make_flat(feat_df)
            signals = self.model.predict(X)
            aligned_df = feat_df.iloc[:len(signals)]

        self.backtester.run(aligned_df, signals, threshold=0.51)

    # ── B2. Backtest All Pairs ────────────────────────────────────────────
    def backtest_all_pairs(self):
        """Run train+backtest for every pair in cfg['pairs']."""
        pairs = self.cfg.get("pairs", [self.cfg["symbol"]])
        log.info(f"─── MODE: BACKTEST-ALL ({len(pairs)} pairs) ────────────────")
        ok, failed = [], []
        for pair in pairs:
            log.info(f"\n{'═'*55}\n  Backtesting: {pair}\n{'═'*55}")
            try:
                self._train_or_backtest_pair(pair, mode="backtest")
                ok.append(pair)
            except Exception as exc:
                log.error(f"{pair} backtest failed: {exc}")
                failed.append(f"{pair}: {exc}")
        self._all_pairs_summary("BACKTEST-ALL", ok, failed)

    # ── C. Live Trading ───────────────────────────────────────────────────
    def live(self, stop_event=None):
        log.info("─── MODE: LIVE ────────────────────────────────────────")

        # Connect MT5
        if not self.connector.connect():
            log.critical("MT5 connection failed. Exiting.")
            return

        # Startup: Telegram announcement + account balance table
        pairs = self.cfg.get("pairs", [self.cfg["symbol"]])
        if self.social is not None:
            self.social.announce_startup(pairs, "live")
        if self.multi_account is not None:
            acct_stats = self.multi_account.get_all_account_stats()
            self.multi_account.print_accounts_table(acct_stats)
            # Reconnect master after querying slave accounts
            if not self.connector.connect():
                log.critical("MT5 master reconnect failed after account scan.")
                return

        if self.cfg.get("multi_pair_mode"):
            self._live_multi(stop_event=stop_event)
        else:
            # Single-pair mode — load model and loop
            try:
                self.model.load()
                self.engineer.load_scaler(self.scaler_path)
            except Exception as e:
                log.error(f"Could not load model/scaler: {e}. Run --mode train first.")
                self.connector.disconnect()
                return
            log.info(f"Live (single-pair) | {self.cfg['symbol']} | "
                     f"polling every {self.cfg['poll_interval_sec']}s")
            log.info("Press Ctrl+C to stop.")
            try:
                while stop_event is None or not stop_event.is_set():
                    if self.social is not None:
                        self.social.handle_commands(self)
                    self._check_closed_positions()
                    if not self.paused:
                        self._live_tick()
                    else:
                        log.debug("[Live] Trading paused — skipping tick.")
                    if self.social is not None:
                        self.social.scheduled_report()
                    _interval = self.cfg["poll_interval_sec"]
                    if stop_event is not None:
                        stop_event.wait(timeout=_interval)
                    else:
                        time.sleep(_interval)
            except KeyboardInterrupt:
                log.info("Interrupted by user.")
            finally:
                self.connector.disconnect()

    def _live_multi(self, stop_event=None):
        """
        Multi-pair live loop.
        Loads a separate model+scaler+downloader for each pair in cfg['pairs'],
        then polls every configured interval, checking each pair in turn.
        Each pair's model must have been trained via --mode train-all first.
        """
        pairs = self.cfg.get("pairs", [self.cfg["symbol"]])
        log.info(f"Live (multi-pair) | {len(pairs)} pairs | "
                 f"polling every {self.cfg['poll_interval_sec']}s")
        log.info(f"  Pairs: {', '.join(pairs)}")
        log.info("Press Ctrl+C to stop.")

        # Load per-pair (model, engineer, downloader, scaler_path)
        pair_states: dict = {}
        n_feat = len(self.engineer.FEATURE_COLS)
        for pair in pairs:
            pair_model_dir = os.path.join(
                self.cfg["model_path"].rstrip("/\\"), pair) + os.sep
            scaler_path = os.path.join(pair_model_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                log.warning(f"{pair}: no trained model found at {pair_model_dir} — skipping. "
                            f"Run --mode train-all first.")
                continue
            try:
                mt = self._model_type
                if mt == "LSTM":
                    m = LSTMModel(self.cfg["lstm_lookback"], n_feat, pair_model_dir)
                elif mt == "ENSEMBLE":
                    m = EnsembleModel(self.cfg["lstm_lookback"], n_feat, pair_model_dir)
                else:
                    m = RFModel(pair_model_dir)
                m.load()
                eng = FeatureEngineer(self.cfg["lstm_lookback"])
                eng.load_scaler(scaler_path)
                dl  = DataDownloader(pair, self.cfg["timeframe"], self.cfg["history_bars"])
                pair_states[pair] = (m, eng, dl, scaler_path)
                log.info(f"  {pair}: model loaded ✓")
            except Exception as exc:
                log.warning(f"  {pair}: failed to load model — {exc}")

        if not pair_states:
            log.error("No pair models loaded. Run --mode train-all first.")
            self.connector.disconnect()
            return

        try:
            while stop_event is None or not stop_event.is_set():
                if self.social is not None:
                    self.social.handle_commands(self)
                self._check_closed_positions()
                if not self.paused:
                    for pair, (model, engineer, downloader, scaler_path) in pair_states.items():
                        try:
                            self._live_tick(
                                symbol=pair,
                                model=model,
                                engineer=engineer,
                                downloader=downloader,
                            )
                        except Exception as exc:
                            log.error(f"Live tick error for {pair}: {exc}")
                else:
                    log.debug("[Live] Trading paused — skipping all pairs.")
                if self.social is not None:
                    self.social.scheduled_report()
                _interval = self.cfg["poll_interval_sec"]
                if stop_event is not None:
                    stop_event.wait(timeout=_interval)
                else:
                    time.sleep(_interval)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            self.connector.disconnect()

    # ── Social + Multi-account helpers ────────────────────────────────────

    def _post_trade_hooks(self, symbol: str, direction: str, lot: float,
                          sl_pips: float, tp_pips: float,
                          confidence: float, overlay_str: str):
        """Called after a trade is successfully opened on the master account."""
        # 1) Social notification
        if SOCIAL_AVAILABLE and self.social is not None:
            trade = self._build_trade_record(symbol, direction, lot,
                                              sl_pips, tp_pips)
            if trade is not None:
                self.social.on_trade_open(trade, confidence, overlay_str)

        # 2) Copy to slave accounts
        if MULTI_ACCOUNT_AVAILABLE and self.multi_account is not None:
            results = self.multi_account.copy_trade_to_slaves(
                symbol, direction, lot, sl_pips, tp_pips,
                self.cfg["magic_number"],
            )
            if results:
                ok  = [k for k, v in results.items() if v]
                bad = [k for k, v in results.items() if not v]
                if ok:
                    log.info(f"[MultiAccount] Copied to: {', '.join(ok)}")
                if bad:
                    log.warning(f"[MultiAccount] Failed for: {', '.join(bad)}")
            # Reconnect to master after slave connections
            if self.multi_account.has_slaves:
                self.connector.connect()

    def _build_trade_record(self, symbol: str, direction: str, lot: float,
                             sl_pips: float, tp_pips: float
                             ) -> "Optional[TradeRecord]":
        """Build a TradeRecord for the position that was just opened."""
        if not SOCIAL_AVAILABLE:
            return None
        pip = DataDownloader.pip_size(symbol)
        if MT5_AVAILABLE and self.connector.connected:
            tick  = mt5.symbol_info_tick(symbol)
            price = (tick.ask if direction == "BUY" else tick.bid) if tick else 1.0
        else:
            price = 1.0  # simulation

        if direction == "BUY":
            sl_price = round(price - sl_pips * pip, 5)
            tp_price = round(price + tp_pips * pip, 5)
        else:
            sl_price = round(price + sl_pips * pip, 5)
            tp_price = round(price - tp_pips * pip, 5)

        # Prefer real MT5 ticket; fall back to timestamp-based synthetic id
        ticket = int(datetime.now(timezone.utc).timestamp() * 1000) % 100_000_000
        if MT5_AVAILABLE and self.connector.connected:
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                ticket = max(p.ticket for p in positions)

        return TradeRecord(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            open_time=datetime.now(timezone.utc),
            open_price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            lot=lot,
        )

    def _check_closed_positions(self):
        """
        Detect positions closed by MT5 (SL/TP hit or manual close).
        Notifies social engine and records the close.
        """
        if not SOCIAL_AVAILABLE or self.social is None:
            return
        if not MT5_AVAILABLE or not self.connector.connected:
            return
        open_tracked = [t for t in self.social.tracker.trades if not t.is_closed]
        for trade in open_tracked:
            try:
                positions = mt5.positions_get(ticket=trade.ticket)
                if positions:
                    continue   # still open
                # Position gone — fetch deal history
                deals = mt5.history_deals_get(position=trade.ticket)
                if deals:
                    close_deal  = max(deals, key=lambda d: d.time)
                    close_price = close_deal.price
                    close_time  = datetime.fromtimestamp(close_deal.time,
                                                          tz=timezone.utc)
                    profit      = float(sum(d.profit for d in deals))
                    pip         = DataDownloader.pip_size(trade.symbol)
                    raw_pips    = (close_price - trade.open_price) / pip
                    pips        = raw_pips if trade.direction == "BUY" else -raw_pips
                else:
                    close_price = trade.open_price
                    close_time  = datetime.now(timezone.utc)
                    profit, pips = 0.0, 0.0

                self.social.on_trade_close(
                    ticket=trade.ticket,
                    close_price=close_price,
                    close_time=close_time,
                    profit=profit,
                    pips=pips,
                )
                # Correct the optimistic win recorded at trade open
                self.agent_memory.record_outcome(won=(profit > 0))
                log.info(
                    f"[Social] #{trade.ticket} {trade.symbol} closed — "
                    f"P&L={profit:+.2f}  pips={pips:+.1f}"
                )
            except Exception as exc:
                log.debug(f"Close-check error for #{trade.ticket}: {exc}")

    def _live_tick(self, symbol: Optional[str] = None,
                   model=None, engineer=None, downloader=None):
        """
        Single live iteration: fetch → predict → intelligence overlay → trade.
        When called from multi-pair mode, pass the pair-specific overrides.
        """
        symbol     = symbol     or self.cfg["symbol"]
        model      = model      or self.model
        engineer   = engineer   or self.engineer
        downloader = downloader or self.downloader

        # ── Drawdown guard ────────────────────────────────────────────────
        if not self.risk.check_drawdown():
            return

        # ── Spread guard ──────────────────────────────────────────────────
        if not self.risk.spread_ok(symbol):
            return

        # ── Too many open trades ──────────────────────────────────────────
        if self.risk.open_positions_count(symbol) >= self.cfg["max_open_trades"]:
            log.debug("Max open positions reached. Waiting…")
            return

        # ── News blackout guard ───────────────────────────────────────────
        if self.cfg.get("news_blackout") and self.ecal is not None:
            try:
                if self.ecal.has_high_impact_soon(self.cfg.get("news_blackout_hours", 2.0)):
                    log.warning("[Bot] High-impact event imminent — trading paused.")
                    return
            except Exception:
                pass

        # ── Fetch latest bars ─────────────────────────────────────────────
        raw_df = downloader.fetch()
        if raw_df.empty or len(raw_df) < self.cfg["lstm_lookback"] + 50:
            log.warning("Insufficient data.")
            return

        feat_df = engineer.build(raw_df)

        # ── ML Prediction ─────────────────────────────────────────────────
        if self._model_type == "LSTM":
            window = feat_df[engineer.FEATURE_COLS].values[-self.cfg["lstm_lookback"]:]
            window_s = engineer.scale(window.reshape(1, *window.shape))
            ml_proba = float(model.predict(window_s)[0])
        elif self._model_type == "ENSEMBLE":
            ml_proba = model.predict_latest(feat_df, engineer)
        else:
            window = feat_df[engineer.FEATURE_COLS].values[-1:]
            window_s = engineer.scale(window)
            ml_proba = float(model.predict(window_s)[0])

        # ── Intelligence Layer Overlay ────────────────────────────────────
        combined = ml_proba
        overlay_notes: List[str] = []

        # News sentiment overlay
        if self.news_engine is not None and self.cfg.get("use_news"):
            try:
                news_data = self.news_engine.pair_sentiment(symbol)
                news_net  = news_data["net"]           # -1..+1
                if abs(news_net) >= self.cfg.get("news_min_sentiment", 0.05):
                    w = self.cfg.get("sentiment_weight", 0.15)
                    # Convert news net (-1..+1) to probability contribution
                    news_contrib = (news_net + 1) / 2   # → 0..1
                    combined = combined * (1 - w) + news_contrib * w
                    overlay_notes.append(
                        f"news={news_net:+.3f} ({news_data['news_count']} articles)"
                    )
                    if news_data.get("high_impact"):
                        overlay_notes.append("HIGH-IMPACT news detected")
            except Exception as exc:
                log.debug(f"News overlay error: {exc}")

        # Fundamental overlay
        if self.fundamental is not None and self.cfg.get("use_fundamentals"):
            try:
                fund_data = self.fundamental.fundamental_bias(symbol)
                fund_score = fund_data["composite"]    # -1..+1
                w = self.cfg.get("fundamental_weight", 0.15)
                fund_contrib = (fund_score + 1) / 2    # → 0..1
                combined = combined * (1 - w) + fund_contrib * w
                overlay_notes.append(f"fundamental={fund_score:+.3f}")
            except Exception as exc:
                log.debug(f"Fundamental overlay error: {exc}")

        # Knowledge base overlay
        if self.knowledge_base is not None and self.cfg.get("use_knowledge_base"):
            try:
                # Query KB with current RSI and MACD context
                rsi_val  = feat_df["rsi"].iloc[-1]  if "rsi"  in feat_df.columns else None
                macd_val = feat_df["macd_diff"].iloc[-1] if "macd_diff" in feat_df.columns else None
                kb_bias  = 0.0
                if rsi_val is not None:
                    indicator = "rsi oversold" if rsi_val < 35 else ("rsi overbought" if rsi_val > 65 else "rsi")
                    kb_bias += self.knowledge_base.sentiment_for_indicator(indicator)
                if macd_val is not None:
                    indicator = "macd bullish" if macd_val > 0 else "macd bearish"
                    kb_bias += self.knowledge_base.sentiment_for_indicator(indicator)
                if kb_bias != 0.0:
                    w = self.cfg.get("knowledge_weight", 0.10)
                    kb_contrib = (kb_bias + 1) / 2
                    combined = combined * (1 - w) + kb_contrib * w
                    overlay_notes.append(f"knowledge_bias={kb_bias:+.3f}")
            except Exception as exc:
                log.debug(f"Knowledge overlay error: {exc}")

        # Technical signal interpretation (informational)
        if TECH_ENGINE_AVAILABLE and engineer._tech is not None:
            try:
                signals  = engineer._tech.interpret_signals(feat_df)
                tech_bias = signals.get("overall_bias", 0)
                overlay_notes.append(f"tech_bias={tech_bias:+.2f}")
            except Exception:
                pass

        overlay_str = " | ".join(overlay_notes) if overlay_notes else "no overlay"
        log.info(
            f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} | "
            f"ML={ml_proba:.3f} | combined={combined:.3f} | {overlay_str}"
        )

        # ── H4 trend filter ───────────────────────────────────────────────
        h4_bull: Optional[bool] = None
        try:
            h4_df   = downloader.fetch_higher_tf()
            h4_feat = engineer.build(h4_df)
            if "ema_cross_9_21" in h4_feat.columns:
                h4_bull = bool(h4_feat["ema_cross_9_21"].iloc[-1] == 1)
        except Exception as _e:
            log.debug(f"H4 filter unavailable: {_e}")

        # ── ATR-based SL/TP ───────────────────────────────────────────────
        pip = DataDownloader.pip_size(symbol)
        atr_val: float
        if "atr" in feat_df.columns and not pd.isna(feat_df["atr"].iloc[-1]):
            atr_val  = float(feat_df["atr"].iloc[-1])
            sl_pips  = (self.cfg["atr_sl_mult"] * atr_val) / pip
            tp_pips  = (self.cfg["atr_tp_mult"] * atr_val) / pip
        else:
            atr_val  = self.cfg["sl_pips"] * pip   # fallback: estimate from pip
            sl_pips  = self.cfg["sl_pips"]
            tp_pips  = self.cfg["tp_pips"]

        # ── Agent: regime detection + adaptive threshold ──────────────────
        regime    = self.regime_detector.detect(feat_df)
        threshold = self.regime_detector.threshold_for(regime, base=0.58)
        threshold = self.agent_memory.adjust_threshold(threshold)
        lot_mult  = self.regime_detector.lot_multiplier_for(regime)
        lot       = round(self.risk.compute_lot_size(sl_pips, symbol, ml_proba=combined) * lot_mult, 2)
        lot       = max(lot, 0.01)    # never below broker minimum

        # ── Correlation filter — scale or block correlated positions ──────
        if MT5_AVAILABLE and self.connector.connected:
            open_syms = [p.symbol for p in (mt5.positions_get() or [])]
            corr_scale = self.correlation_filter.lot_scale_factor(symbol, open_syms)
            if corr_scale == 0.0:
                log.info(
                    f"[CorrelationFilter] {symbol} blocked — 3+ correlated positions open."
                )
                return
            if corr_scale < 1.0:
                lot = max(round(lot * corr_scale, 2), 0.01)
                log.info(
                    f"[CorrelationFilter] {symbol} lot → {lot} (scale={corr_scale:.2f})"
                )

        broadcast = self.cfg.get("social", {}).get("broadcast_signals", False)
        log.info(
            f"[Agent] regime={regime} | threshold={threshold:.2f} | "
            f"{self.agent_memory.summary()}"
        )

        # ── Update trailing stops on existing positions first ─────────────
        atr_pips = atr_val / pip
        self.trailing_stops.update_all(symbol, atr_pips)

        if combined > threshold:
            if h4_bull is False:
                log.info("Signal: BUY blocked by H4 trend filter (H4 is bearish)")
                if broadcast and self.social:
                    self.social.on_signal(symbol, "BUY", combined,
                                          f"Blocked — H4 bearish | {overlay_str}")
                return
            log.info(f"Signal: BUY (combined {combined:.2%}, SL={sl_pips:.1f}p, TP={tp_pips:.1f}p, regime={regime})")
            ok = self.executor.open_buy(symbol, lot, sl_pips, tp_pips)
            if ok:
                self._post_trade_hooks(symbol, "BUY", lot, sl_pips, tp_pips,
                                       combined, overlay_str)
            elif broadcast and self.social:
                self.social.on_signal(symbol, "BUY", combined, overlay_str)

        elif combined < (1 - threshold):
            if h4_bull is True:
                log.info("Signal: SELL blocked by H4 trend filter (H4 is bullish)")
                if broadcast and self.social:
                    self.social.on_signal(symbol, "SELL", 1 - combined,
                                          f"Blocked — H4 bullish | {overlay_str}")
                return
            log.info(f"Signal: SELL (combined {1-combined:.2%}, SL={sl_pips:.1f}p, TP={tp_pips:.1f}p, regime={regime})")
            ok = self.executor.open_sell(symbol, lot, sl_pips, tp_pips)
            if ok:
                self._post_trade_hooks(symbol, "SELL", lot, sl_pips, tp_pips,
                                       combined, overlay_str)
            elif broadcast and self.social:
                self.social.on_signal(symbol, "SELL", 1 - combined, overlay_str)
        else:
            log.info(f"Signal: NEUTRAL — no trade (regime={regime}, threshold={threshold:.2f}).")

    # ── D. Book Training ──────────────────────────────────────────────────
    def train_books(self, folder: Optional[str] = None):
        """Ingest PDFs and text files to build the knowledge base."""
        if not BOOKS_AVAILABLE:
            log.error("book_trainer.py not found. Cannot train from books.")
            return

        log.info("─── MODE: TRAIN-BOOKS ─────────────────────────────────")
        kb      = self.knowledge_base or KnowledgeBase()
        trainer = BookTrainer(kb)

        # Load built-in classic rules first
        load_classic_rules(kb)

        # Scan provided folder or config books_folder
        target = folder or self.cfg.get("books_folder", "books/")
        os.makedirs(target, exist_ok=True)
        added = trainer.ingest_folder(target)
        trainer.save()

        log.info(trainer.report())
        log.info(f"Books training complete. {added} new rules added. "
                 f"Total: {len(kb.rules)} rules in knowledge base.")

    # ── E. Quotes Report ──────────────────────────────────────────────────
    def show_quotes(self):
        """Display live prices for all configured forex pairs."""
        log.info("─── MODE: QUOTES ──────────────────────────────────────")
        if not FUNDAMENTAL_AVAILABLE:
            log.error("fundamental_engine.py not found.")
            return
        engine = FundamentalEngine(pairs=self.cfg.get("pairs", None))
        print(engine.quotes.quotes_table())

    # ── F. News Report ────────────────────────────────────────────────────
    def show_news(self):
        """Fetch and display current news sentiment for active pairs."""
        log.info("─── MODE: NEWS ────────────────────────────────────────")
        if not NEWS_AVAILABLE:
            log.error("news_engine.py not found.")
            return
        engine = NewsEngine()
        pairs  = self.cfg.get("pairs", [self.cfg["symbol"]])
        print(engine.summary_report(pairs))

        cal    = EconomicCalendar()
        events = cal.get_events()
        if events:
            print(f"\nUpcoming Economic Events ({len(events)} total):")
            for ev in events[:15]:
                tag = "*** HIGH ***" if ev["impact"] == "high" else ev["impact"].upper()
                t   = ev.get("time")
                tstr = t.strftime("%a %b %d %H:%M UTC") if t else "TBD"
                print(f"  [{tag:11s}] {ev.get('currency','???'):4s}  {tstr}  {ev['name'][:50]}")

    # ── G. Fundamentals Report ────────────────────────────────────────────
    def show_fundamentals(self):
        """Display fundamental analysis for active pairs."""
        log.info("─── MODE: FUNDAMENTALS ────────────────────────────────")
        if not FUNDAMENTAL_AVAILABLE:
            log.error("fundamental_engine.py not found.")
            return
        engine = FundamentalEngine(pairs=self.cfg.get("pairs", None))
        pairs  = self.cfg.get("pairs", [self.cfg["symbol"]])
        print(engine.summary_report(pairs))

    # ── H. Performance Stats ──────────────────────────────────────────────
    def show_stats(self):
        """Print performance statistics for all tracked trades."""
        log.info("─── MODE: STATS ───────────────────────────────────────")
        if not SOCIAL_AVAILABLE:
            log.error("social_engine.py not found — stats unavailable.")
            return
        tracker = (self.social.tracker if self.social is not None
                   else PerformanceTracker())
        for label, d in [("ALL TIME", None), ("LAST 30 DAYS", 30),
                          ("LAST 7 DAYS", 7), ("TODAY", 1)]:
            s = tracker.stats(days=d)
            wins   = s.get("wins",   0)
            losses = s.get("losses", 0)
            wr     = s.get("win_rate", 0.0)
            bar    = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))
            log.info(
                f"\n{'═'*50}\n"
                f"  {label}\n"
                f"{'─'*50}\n"
                f"  Trades:        {s['total']}  ({wins}W / {losses}L)\n"
                f"  Win Rate:      {wr:.1f}%  {bar}\n"
                f"  Profit Factor: {s['profit_factor']:.2f}\n"
                f"  Net P&L:       {s['total_profit']:+.2f}\n"
                f"  Total Pips:    {s['total_pips']:+.1f}\n"
                f"  Max Drawdown:  -{s['max_drawdown']:.2f}\n"
                f"  Sharpe Ratio:  {s['sharpe']:.2f}\n"
                f"{'═'*50}"
            )

    # ── I. Accounts Overview ──────────────────────────────────────────────
    def show_accounts(self):
        """Display balance / equity for every configured account."""
        log.info("─── MODE: ACCOUNTS ────────────────────────────────────")
        if not MULTI_ACCOUNT_AVAILABLE:
            log.error("multi_account_manager.py not found.")
            return
        if self.multi_account is None:
            log.error("No multi-account manager (add ≥ 2 accounts in accounts_config.py).")
            return
        self.multi_account.print_accounts_table()

    # ── J. Post performance card to Telegram ─────────────────────────────
    def post_card(self, period: str = "weekly"):
        """Generate a performance card and post it to the Telegram channel."""
        log.info("─── MODE: POST-CARD ───────────────────────────────────")
        if not SOCIAL_AVAILABLE:
            log.error("social_engine.py not found.")
            return
        if self.social is None:
            log.error("Social engine not initialised.")
            return
        days   = {"today": 1, "daily": 1, "weekly": 7, "monthly": 30}.get(
            period.lower(), 7)
        label  = period.title()
        self.social._send_report(days=days, period=label)
        log.info(f"Performance card ({label}) posted to Telegram channel.")


# ==============================================================================
#  CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Forex AI Trading Bot v3.0 — Multi-Account + Social Media",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train          Download data, engineer features, train model (single pair)
  train-all      Train a separate model for every pair in CONFIG['pairs']
  backtest       Train then run vectorised backtest with chart (single pair)
  backtest-all   Train + backtest every pair in CONFIG['pairs']
  live           Connect to MT5, run live trading (single or multi-pair)
  train-books    Ingest PDF/text books into the knowledge base
  quotes         Show live prices for all configured forex pairs
  news           Show current news sentiment and economic calendar
  fundamentals   Show fundamental analysis (interest rates, DXY, carry)
  stats          Show trade performance statistics (all time / weekly / daily)
  accounts       Show balance & equity for all configured MT5 accounts
  post-card      Generate & post a performance card to Telegram

Examples:
  python forex_ai_bot.py --mode train-all
  python forex_ai_bot.py --mode live
  python forex_ai_bot.py --mode stats
  python forex_ai_bot.py --mode accounts
  python forex_ai_bot.py --mode post-card --period weekly
  python forex_ai_bot.py --mode train-books --books-folder books/
        """
    )
    parser.add_argument(
        "--mode",
        choices=["train", "train-all", "backtest", "backtest-all",
                 "live", "train-books", "quotes", "news", "fundamentals",
                 "stats", "accounts", "post-card"],
        default="backtest",
        help="Operating mode (default: backtest)",
    )
    parser.add_argument(
        "--books-folder",
        default=None,
        help="Folder containing PDF/text trading books to ingest",
    )
    parser.add_argument(
        "--period",
        default="weekly",
        choices=["today", "daily", "weekly", "monthly"],
        help="Period for post-card (default: weekly)",
    )
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("  FOREX AI TRADING BOT v3.0")
    if CONFIG.get("multi_pair_mode"):
        pairs = CONFIG.get("pairs", [])
        log.info(f"  Pairs     : ALL ({len(pairs)} instruments) | {CONFIG['timeframe']}")
        log.info(f"              {', '.join(pairs)}")
    else:
        log.info(f"  Symbol    : {CONFIG['symbol']} {CONFIG['timeframe']}")
    log.info(f"  Model     : {CONFIG['model_type']}")
    log.info(f"  Mode      : {args.mode.upper()}")
    acct_count = len(_active_accounts)
    log.info(f"  Accounts  : {acct_count} active | "
             f"Multi-acc: {'ON' if acct_count > 1 and MULTI_ACCOUNT_AVAILABLE else 'OFF'}")
    log.info(f"  Social    : {'ON' if SOCIAL_AVAILABLE and CONFIG.get('social',{}).get('enabled') else 'OFF'} | "
             f"Telegram: {'configured' if CONFIG.get('telegram',{}).get('bot_token') else 'not set'}")
    log.info(f"  News      : {'ON' if CONFIG.get('use_news') and NEWS_AVAILABLE else 'OFF'}")
    log.info(f"  Fundament.: {'ON' if CONFIG.get('use_fundamentals') and FUNDAMENTAL_AVAILABLE else 'OFF'}")
    log.info(f"  Knowledge : {'ON' if CONFIG.get('use_knowledge_base') and BOOKS_AVAILABLE else 'OFF'}")
    log.info("=" * 65)

    bot = ForexAIBot(CONFIG)

    if args.mode == "train":
        bot.train()
    elif args.mode == "train-all":
        bot.train_all_pairs()
    elif args.mode == "backtest":
        bot.backtest()
    elif args.mode == "backtest-all":
        bot.backtest_all_pairs()
    elif args.mode == "live":
        bot.live()
    elif args.mode == "train-books":
        bot.train_books(folder=args.books_folder)
    elif args.mode == "quotes":
        bot.show_quotes()
    elif args.mode == "news":
        bot.show_news()
    elif args.mode == "fundamentals":
        bot.show_fundamentals()
    elif args.mode == "stats":
        bot.show_stats()
    elif args.mode == "accounts":
        bot.show_accounts()
    elif args.mode == "post-card":
        bot.post_card(period=args.period)


if __name__ == "__main__":
    main()
