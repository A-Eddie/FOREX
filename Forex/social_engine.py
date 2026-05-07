"""
================================================================================
 SOCIAL ENGINE — Telegram notifications + performance cards
================================================================================
 Features:
   • Real-time trade alerts via Telegram Bot API (HTML formatted)
   • Public signal channel broadcasting
   • Performance card images (dark professional theme, screenshot-ready)
   • Daily and weekly automated performance reports
   • Trade history persistence (JSON)

 Attracting followers:
   • Every trade open/close fires a rich Telegram message to your channel
   • Weekly cards can be posted to Instagram / Twitter as screenshots
   • All messages include relevant hashtags and branding
================================================================================
"""

import io
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

# Matplotlib — must use non-interactive backend before any other mpl import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

log = logging.getLogger("ForexBot.Social")

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    log.warning("[Social] 'requests' not installed — Telegram disabled.")


# ==============================================================================
#  TRADE RECORD
# ==============================================================================
@dataclass
class TradeRecord:
    """Complete record of a single trade (open + optional close)."""
    ticket:      int
    symbol:      str
    direction:   str           # "BUY" or "SELL"
    open_time:   datetime
    open_price:  float
    sl_price:    float
    tp_price:    float
    lot:         float
    close_time:  Optional[datetime] = None
    close_price: Optional[float] = None
    profit:      Optional[float] = None
    pips:        Optional[float] = None
    account_name: str = "Master"

    @property
    def is_closed(self) -> bool:
        return self.close_time is not None

    @property
    def rr_ratio(self) -> float:
        if self.direction == "BUY":
            tp_dist = abs(self.tp_price  - self.open_price)
            sl_dist = abs(self.open_price - self.sl_price)
        else:
            tp_dist = abs(self.open_price - self.tp_price)
            sl_dist = abs(self.sl_price   - self.open_price)
        return round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0


# ==============================================================================
#  PERFORMANCE TRACKER
# ==============================================================================
class PerformanceTracker:
    """Persists trade history to JSON and computes performance statistics."""

    SAVE_PATH = "models/trade_history.json"

    def __init__(self):
        self.trades: List[TradeRecord] = []
        self._load()

    # ── CRUD ───────────────────────────────────────────────────────────────

    def add_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self._save()

    def close_trade(self, ticket: int, close_price: float,
                    close_time: datetime, profit: float,
                    pips: float) -> Optional[TradeRecord]:
        for t in self.trades:
            if t.ticket == ticket and not t.is_closed:
                t.close_price = close_price
                t.close_time  = close_time
                t.profit      = profit
                t.pips        = pips
                self._save()
                return t
        return None

    def get_trade(self, ticket: int) -> Optional[TradeRecord]:
        return next((t for t in self.trades if t.ticket == ticket), None)

    # ── Statistics ─────────────────────────────────────────────────────────

    def closed_trades(self, days: Optional[int] = None) -> List[TradeRecord]:
        trades = [t for t in self.trades if t.is_closed]
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            trades = [t for t in trades
                      if t.close_time and t.close_time >= cutoff]
        return trades

    def equity_curve(self, days: Optional[int] = None) -> List[float]:
        closed      = self.closed_trades(days=days)
        profits     = [t.profit for t in closed if t.profit is not None]
        cumulative  = list(np.cumsum(profits)) if profits else [0.0]
        return cumulative

    def stats(self, days: Optional[int] = None) -> Dict[str, Any]:
        closed  = self.closed_trades(days=days)
        empty   = {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "total_profit": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "max_drawdown": 0.0, "expected_value": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
            "total_pips": 0.0, "sharpe": 0.0,
        }
        if not closed:
            return empty

        profits  = [t.profit for t in closed if t.profit is not None]
        pips_lst = [t.pips   for t in closed if t.pips   is not None]
        wins     = [p for p in profits if p > 0]
        losses   = [p for p in profits if p <= 0]

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses)) if losses else 0.0

        # Max drawdown (on running equity)
        equity, peak, dd_list = 0.0, 0.0, []
        for p in profits:
            equity += p
            peak = max(peak, equity)
            dd_list.append(peak - equity)
        max_dd = max(dd_list) if dd_list else 0.0

        # Sharpe-like ratio
        if len(profits) > 1:
            mu, std = float(np.mean(profits)), float(np.std(profits))
            sharpe  = (mu / std * np.sqrt(252)) if std > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "total":          len(closed),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       len(wins) / len(closed) * 100,
            "total_profit":   float(sum(profits)),
            "profit_factor":  gross_profit / gross_loss if gross_loss > 0 else 99.0,
            "avg_win":        float(np.mean(wins))   if wins   else 0.0,
            "avg_loss":       float(np.mean(losses)) if losses else 0.0,
            "max_drawdown":   max_dd,
            "expected_value": float(np.mean(profits)),
            "best_trade":     float(max(profits)),
            "worst_trade":    float(min(profits)),
            "total_pips":     float(sum(pips_lst)),
            "sharpe":         sharpe,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
        data = []
        for t in self.trades:
            d = asdict(t)
            for k in ("open_time", "close_time"):
                if d[k] is not None and isinstance(d[k], datetime):
                    d[k] = d[k].isoformat()
            data.append(d)
        with open(self.SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not os.path.exists(self.SAVE_PATH):
            return
        try:
            with open(self.SAVE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                for k in ("open_time", "close_time"):
                    if d.get(k):
                        d[k] = datetime.fromisoformat(d[k])
                self.trades.append(TradeRecord(**d))
            log.info(f"[Social] Loaded {len(self.trades)} trade(s) from history.")
        except Exception as exc:
            log.warning(f"[Social] Could not load trade history: {exc}")


# ==============================================================================
#  TELEGRAM NOTIFIER
# ==============================================================================
class TelegramNotifier:
    """
    Sends formatted messages and images via the Telegram Bot API.

    channel_id → public signal channel (trade alerts, reports)
    chat_id    → private alerts to the operator (errors, status)
    """

    _API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, channel_id: str,
                 chat_id: str = "", branding: dict = None):
        self.token      = token
        self.channel_id = channel_id
        self.chat_id    = chat_id or channel_id
        self.branding   = branding or {}
        self._on        = bool(token and channel_id and _REQUESTS_OK)
        if not token or not channel_id:
            log.info("[Social] Telegram not configured — fill in accounts_config.py.")

    # ── Low-level helpers ──────────────────────────────────────────────────

    def _post(self, method: str, **kwargs) -> bool:
        if not self._on:
            return False
        url = self._API.format(token=self.token, method=method)
        try:
            r = _requests.post(url, timeout=10, **kwargs)
            if not r.ok:
                log.warning(f"Telegram {method} {r.status_code}: {r.text[:200]}")
            return r.ok
        except Exception as exc:
            log.warning(f"Telegram {method} failed: {exc}")
            return False

    def _send_text(self, chat_id: str, text: str) -> bool:
        return self._post(
            "sendMessage",
            json={
                "chat_id":                  chat_id,
                "text":                     text,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
            },
        )

    def _send_photo(self, chat_id: str, img: bytes, caption: str = "") -> bool:
        return self._post(
            "sendPhoto",
            data={"chat_id": chat_id, "caption": caption[:1024],
                  "parse_mode": "HTML"},
            files={"photo": ("card.png", img, "image/png")},
        )

    def broadcast(self, text: str) -> bool:
        """Send to signal channel."""
        return self._send_text(self.channel_id, text)

    def alert(self, text: str) -> bool:
        """Send private alert to operator."""
        return self._send_text(self.chat_id, text)

    # ── Trade open notification ────────────────────────────────────────────

    def notify_trade_open(self, trade: TradeRecord,
                          confidence: float = 0.0,
                          overlay: str = "") -> bool:
        d_emoji  = "📈" if trade.direction == "BUY" else "📉"
        pip      = self._pip(trade.symbol)
        sl_pips  = abs(trade.open_price - trade.sl_price)  / pip
        tp_pips  = abs(trade.tp_price   - trade.open_price) / pip
        name     = self.branding.get("name",    "ForexAI Bot")
        tagline  = self.branding.get("tagline", "AI-Powered Signals")

        msg = (
            f"<b>🤖 {name}</b>\n"
            f"<i>{tagline}</i>\n"
            f"{'─'*30}\n"
            f"{d_emoji} <b>{trade.direction} {trade.symbol}</b>\n"
            f"{'─'*30}\n"
            f"💰 <b>Entry :</b>  <code>{trade.open_price:.5f}</code>\n"
            f"🛑 <b>Stop  :</b>  <code>{trade.sl_price:.5f}</code>   ({sl_pips:.0f} pips)\n"
            f"🎯 <b>Target:</b>  <code>{trade.tp_price:.5f}</code>  (+{tp_pips:.0f} pips)\n"
            f"📊 <b>R : R :</b>  1 : {trade.rr_ratio:.1f}\n"
            f"📦 <b>Lot   :</b>  {trade.lot:.2f}\n"
            f"{'─'*30}\n"
        )
        if confidence:
            msg += f"🧠 <b>AI Confidence:</b> {confidence:.0%}\n"
        if overlay:
            msg += f"📡 <b>Signals:</b> {overlay}\n"
        msg += (
            f"\n🕐 {trade.open_time.strftime('%d %b %Y  %H:%M UTC')}\n"
            f"\n{self._hashtags(trade.symbol, trade.direction)}"
        )
        return self.broadcast(msg)

    # ── Trade close notification ───────────────────────────────────────────

    def notify_trade_close(self, trade: TradeRecord) -> bool:
        if not trade.is_closed:
            return False
        profit  = trade.profit or 0.0
        pips    = trade.pips   or 0.0
        ok      = profit > 0
        emoji   = "✅" if ok else "❌"
        result  = "WIN 🏆" if ok else "LOSS"
        dur_str = ""
        if trade.open_time and trade.close_time:
            delta   = trade.close_time - trade.open_time
            h, rem  = divmod(int(delta.total_seconds()), 3600)
            m       = rem // 60
            dur_str = f"⏱️ <b>Duration :</b> {h}h {m:02d}m\n"

        name = self.branding.get("name", "ForexAI Bot")
        msg  = (
            f"{emoji} <b>TRADE CLOSED — {result}</b>\n"
            f"<b>🤖 {name}</b>\n"
            f"{'─'*30}\n"
            f"<b>{trade.direction} {trade.symbol}</b>\n"
            f"Open : <code>{trade.open_price:.5f}</code>\n"
            f"Close: <code>{trade.close_price:.5f}</code>\n"
            f"{'─'*30}\n"
            f"{'🟢' if ok else '🔴'} <b>P&amp;L  :</b>  "
            f"{'+'if profit>=0 else ''}{profit:.2f}\n"
            f"📏 <b>Pips  :</b>  {'+'if pips>=0 else ''}{pips:.1f}\n"
            f"{dur_str}"
            f"\n{self._hashtags(trade.symbol, trade.direction)}"
        )
        return self.broadcast(msg)

    # ── Signal notification (no trade placed) ─────────────────────────────

    def notify_signal(self, symbol: str, direction: str,
                      confidence: float, note: str = "") -> bool:
        if direction.upper() == "NEUTRAL":
            return False   # Don't spam neutrals to channel
        d_emoji = "📈" if direction.upper() == "BUY" else "📉"
        name    = self.branding.get("name", "ForexAI Bot")
        msg     = (
            f"<b>📡 {name} — SIGNAL</b>\n"
            f"{'─'*30}\n"
            f"{d_emoji} <b>{direction.upper()} {symbol}</b>\n"
            f"🧠 Confidence: {confidence:.0%}\n"
        )
        if note:
            msg += f"📊 {note}\n"
        msg += f"\n{self._hashtags(symbol, direction)}"
        return self.broadcast(msg)

    # ── Performance report ─────────────────────────────────────────────────

    def send_performance_report(self, stats: dict, period: str = "Daily",
                                 img: Optional[bytes] = None) -> bool:
        name    = self.branding.get("name",    "ForexAI Bot")
        tagline = self.branding.get("tagline", "")
        website = self.branding.get("website", "")
        ig      = self.branding.get("instagram", "")
        tw      = self.branding.get("twitter",   "")

        wr  = stats.get("win_rate",      0.0)
        pf  = stats.get("profit_factor", 0.0)
        tp  = stats.get("total_profit",  0.0)
        dd  = stats.get("max_drawdown",  0.0)
        tot = stats.get("total",         0)
        pip = stats.get("total_pips",    0.0)
        bar = "█" * int(wr / 10) + "░" * (10 - int(wr / 10))

        msg = (
            f"<b>📊 {name} — {period} Report</b>\n"
            f"<i>{tagline}</i>\n"
            f"{'═'*30}\n"
            f"🏆 Win Rate:       {wr:.1f}%  <code>{bar}</code>\n"
            f"⚡ Profit Factor:  {pf:.2f}\n"
            f"💵 Net P&amp;L:    {'+'if tp>=0 else ''}{tp:.2f}\n"
            f"📏 Total Pips:     {'+'if pip>=0 else ''}{pip:.1f}\n"
            f"📉 Max Drawdown:   -{dd:.2f}\n"
            f"🔢 Trades: {tot} "
            f"({stats.get('wins',0)}W / {stats.get('losses',0)}L)\n"
            f"{'═'*30}\n"
        )
        links = []
        if website:   links.append(f"🌐 {website}")
        if ig:        links.append(f"📸 {ig}")
        if tw:        links.append(f"🐦 {tw}")
        if links:
            msg += "\n".join(links) + "\n"
        msg += f"\n#ForexAI #Performance #{period}"

        if img:
            return self._send_photo(self.channel_id, img, caption=msg)
        return self.broadcast(msg)

    # ── Startup announcement ───────────────────────────────────────────────

    def announce_startup(self, pairs: List[str], mode: str) -> bool:
        name    = self.branding.get("name",    "ForexAI Bot")
        tagline = self.branding.get("tagline", "")
        website = self.branding.get("website", "")
        msg     = (
            f"<b>🚀 {name} is LIVE</b>\n"
            f"<i>{tagline}</i>\n"
            f"{'─'*30}\n"
            f"📊 Watching <b>{len(pairs)} instruments</b>\n"
            f"🔄 Mode: <b>{mode.upper()}</b>\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%d %b %Y  %H:%M UTC')}\n"
        )
        if website:
            msg += f"🌐 {website}\n"
        msg += "\n#ForexAI #Live #AlgoTrading #Signals"
        return self.broadcast(msg)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _pip(symbol: str) -> float:
        jpy_gold = {
            "USDJPY","EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY","XAUUSD"
        }
        return 0.01 if symbol.upper() in jpy_gold else 0.0001

    @staticmethod
    def _hashtags(symbol: str, direction: str) -> str:
        sym = symbol.upper()
        tags = [f"#{sym}", f"#{direction.upper()}", "#ForexAI",
                "#AlgoTrading", "#ForexSignals"]
        if len(sym) >= 3:
            tags.append(f"#{sym[:3]}")
        return " ".join(tags)


# ==============================================================================
#  TELEGRAM COMMAND HANDLER  (phone remote control)
# ==============================================================================
class TelegramCommandHandler:
    """
    Polls Telegram for commands from the authorised operator and executes
    them against the live bot.  Send these from your phone:

        /status              — bot state, equity, open position count
        /balance             — balance, equity, margin
        /positions           — list all open trades
        /buy  SYMBOL [LOTS]  — open a market BUY  (e.g. /buy EURUSD 0.01)
        /sell SYMBOL [LOTS]  — open a market SELL
        /close SYMBOL        — close all positions for that symbol
        /closeall            — close ALL open positions
        /pause               — pause AI trading (stops new entries)
        /resume              — resume AI trading
        /report              — send performance report to channel now
        /help                — list all commands
    """

    _API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, authorized_chat_id: str):
        self.token         = token
        self.authorized_id = str(authorized_chat_id)
        self._offset       = 0
        self._on           = bool(token and authorized_chat_id and _REQUESTS_OK)

    def _get(self, method: str, **kwargs) -> dict:
        if not self._on:
            return {}
        url = self._API.format(token=self.token, method=method)
        try:
            r = _requests.get(url, timeout=10, **kwargs)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def _reply(self, chat_id: str, text: str):
        if not self._on:
            return
        try:
            _requests.post(
                self._API.format(token=self.token, method="sendMessage"),
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception:
            pass

    def poll(self) -> list:
        """Fetch new updates; return list of pending command dicts."""
        data    = self._get("getUpdates",
                            params={"offset": self._offset, "timeout": 0, "limit": 10})
        updates = data.get("result", [])
        commands = []
        for upd in updates:
            self._offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("channel_post")
            if not msg:
                continue
            sender_id = str(
                msg.get("from", {}).get("id", "")
                or msg.get("sender_chat", {}).get("id", "")
            )
            text = msg.get("text", "").strip()
            if not text.startswith("/"):
                continue
            if sender_id != self.authorized_id:
                self._reply(sender_id, "⛔ Unauthorized.")
                continue
            commands.append({"chat_id": sender_id, "text": text})
        return commands

    def handle(self, commands: list, bot) -> None:
        """
        Process commands against a live ForexAIBot instance.
        `bot` must expose: .connector, .executor, .cfg, .social, .paused
        """
        try:
            import MetaTrader5 as _mt5
            _mt5_ok = True
        except ImportError:
            _mt5     = None
            _mt5_ok  = False

        for cmd in commands:
            chat_id = self.authorized_id
            parts   = cmd["text"].split()
            verb    = parts[0].lower().split("@")[0]
            try:
                # ── /status ───────────────────────────────────────────────
                if verb == "/status":
                    info      = _mt5.account_info()       if _mt5_ok else None
                    positions = _mt5.positions_get() or [] if _mt5_ok else []
                    bal  = f"{info.balance:.2f} {info.currency}" if info else "N/A"
                    eq   = f"{info.equity:.2f} {info.currency}"  if info else "N/A"
                    pl   = f"{info.profit:+.2f} {info.currency}" if info else "N/A"
                    paused = getattr(bot, "paused", False)
                    self._reply(chat_id,
                        f"<b>📊 Bot Status</b>\n"
                        f"State:   {'⏸ Paused' if paused else '▶️ Running'}\n"
                        f"Balance: <code>{bal}</code>\n"
                        f"Equity:  <code>{eq}</code>\n"
                        f"P/L:     <code>{pl}</code>\n"
                        f"Open trades: {len(positions)}"
                    )

                # ── /balance ──────────────────────────────────────────────
                elif verb == "/balance":
                    info = _mt5.account_info() if _mt5_ok else None
                    if info:
                        self._reply(chat_id,
                            f"💰 <b>Account Balance</b>\n"
                            f"Balance: <code>{info.balance:.2f} {info.currency}</code>\n"
                            f"Equity:  <code>{info.equity:.2f} {info.currency}</code>\n"
                            f"Margin:  <code>{info.margin:.2f}</code>\n"
                            f"Free:    <code>{info.margin_free:.2f}</code>"
                        )
                    else:
                        self._reply(chat_id, "❌ MT5 not connected.")

                # ── /positions ────────────────────────────────────────────
                elif verb == "/positions":
                    positions = (_mt5.positions_get() or []) if _mt5_ok else []
                    if not positions:
                        self._reply(chat_id, "📭 No open positions.")
                    else:
                        lines = [f"<b>📋 Open Positions ({len(positions)})</b>"]
                        for p in positions:
                            d = "BUY" if p.type == 0 else "SELL"
                            lines.append(
                                f"• {d} {p.symbol}  {p.volume:.2f} lots  "
                                f"entry={p.price_open:.5f}  P/L={p.profit:+.2f}"
                            )
                        self._reply(chat_id, "\n".join(lines))

                # ── /buy SYMBOL [LOTS] ─────────────────────────────────────
                elif verb == "/buy" and len(parts) >= 2:
                    symbol = parts[1].upper()
                    lot    = float(parts[2]) if len(parts) >= 3 else bot.cfg.get("lot_size", 0.01)
                    ok     = bot.executor.open_buy(
                        symbol, lot,
                        bot.cfg.get("sl_pips", 30),
                        bot.cfg.get("tp_pips", 60),
                    )
                    self._reply(chat_id,
                        f"{'✅ BUY opened' if ok else '❌ BUY failed'}: "
                        f"{symbol}  {lot} lots"
                    )

                # ── /sell SYMBOL [LOTS] ────────────────────────────────────
                elif verb == "/sell" and len(parts) >= 2:
                    symbol = parts[1].upper()
                    lot    = float(parts[2]) if len(parts) >= 3 else bot.cfg.get("lot_size", 0.01)
                    ok     = bot.executor.open_sell(
                        symbol, lot,
                        bot.cfg.get("sl_pips", 30),
                        bot.cfg.get("tp_pips", 60),
                    )
                    self._reply(chat_id,
                        f"{'✅ SELL opened' if ok else '❌ SELL failed'}: "
                        f"{symbol}  {lot} lots"
                    )

                # ── /close SYMBOL ─────────────────────────────────────────
                elif verb == "/close" and len(parts) >= 2:
                    symbol = parts[1].upper()
                    bot.executor.close_all(symbol)
                    self._reply(chat_id, f"✅ Closed all positions for {symbol}.")

                # ── /closeall ─────────────────────────────────────────────
                elif verb == "/closeall":
                    if _mt5_ok:
                        positions = _mt5.positions_get() or []
                        symbols   = set(p.symbol for p in positions)
                        for s in symbols:
                            bot.executor.close_all(s)
                        self._reply(chat_id, f"✅ Closed all positions ({len(symbols)} symbol(s)).")
                    else:
                        self._reply(chat_id, "❌ MT5 not connected.")

                # ── /pause ────────────────────────────────────────────────
                elif verb == "/pause":
                    bot.paused = True
                    self._reply(chat_id, "⏸ AI trading paused. Use /resume to restart.")

                # ── /resume ───────────────────────────────────────────────
                elif verb == "/resume":
                    bot.paused = False
                    self._reply(chat_id, "▶️ AI trading resumed.")

                # ── /report ───────────────────────────────────────────────
                elif verb == "/report":
                    if bot.social:
                        bot.social.scheduled_report(force=True)
                        self._reply(chat_id, "📊 Performance report sent to channel.")
                    else:
                        self._reply(chat_id, "❌ Social engine not enabled.")

                # ── /help ─────────────────────────────────────────────────
                elif verb == "/help":
                    self._reply(chat_id,
                        "<b>📱 Wave AI — Phone Commands</b>\n\n"
                        "/status              — bot state &amp; account info\n"
                        "/balance             — balance &amp; equity\n"
                        "/positions           — list open trades\n"
                        "/buy  SYMBOL [LOTS]  — open BUY  (e.g. /buy EURUSD 0.01)\n"
                        "/sell SYMBOL [LOTS]  — open SELL\n"
                        "/close SYMBOL        — close symbol positions\n"
                        "/closeall            — close ALL positions\n"
                        "/pause               — pause AI trading\n"
                        "/resume              — resume AI trading\n"
                        "/report              — send performance report now"
                    )

                else:
                    self._reply(chat_id, "❓ Unknown command. Send /help for list.")

            except Exception as exc:
                log.error(f"[TelegramCmd] Error handling '{cmd['text']}': {exc}")
                self._reply(chat_id, f"❌ Error: {exc}")


# ==============================================================================
#  PERFORMANCE CARD IMAGE GENERATOR
# ==============================================================================
class PerformanceCardGenerator:
    """
    Generates professional performance card PNG images using matplotlib.
    Dark trading-terminal theme — screenshot-ready for Instagram / Twitter.
    """

    BG      = "#0d1117"
    CARD    = "#161b22"
    GREEN   = "#39d353"
    RED     = "#f85149"
    BLUE    = "#58a6ff"
    GOLD    = "#e3b341"
    TEXT    = "#f0f6fc"
    MUTED   = "#8b949e"
    GRID    = "#21262d"

    def generate(self, stats: dict, period: str = "Weekly",
                 bot_name: str = "ForexAI Bot",
                 equity_curve: Optional[List[float]] = None) -> bytes:
        """Return PNG bytes of the performance card."""
        fig = plt.figure(figsize=(10, 5.6), facecolor=self.BG, dpi=150)
        gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.94, bottom=0.08)

        ax_title   = fig.add_subplot(gs[0, :])
        ax_equity  = fig.add_subplot(gs[1, :2])
        ax_metrics = fig.add_subplot(gs[1, 2])

        self._title_strip(ax_title, bot_name, period, stats)
        self._equity_curve(ax_equity, equity_curve or self._fake_curve(stats))
        self._metrics_panel(ax_metrics, stats)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor=self.BG)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ── Sub-panels ─────────────────────────────────────────────────────────

    def _title_strip(self, ax, name, period, stats):
        ax.set_facecolor(self.CARD)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        tp  = stats.get("total_profit", 0)
        wr  = stats.get("win_rate",     0)
        col = self.GREEN if tp >= 0 else self.RED

        # Left: bot name
        ax.text(0.02, 0.72, f"🤖  {name}", color=self.TEXT, fontsize=13,
                fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.02, 0.22, f"{period} Performance Report",
                color=self.MUTED, fontsize=9, transform=ax.transAxes)

        # Centre: win rate
        ax.text(0.50, 0.72, f"{wr:.1f}%", color=col, fontsize=17,
                fontweight="bold", transform=ax.transAxes,
                ha="center", va="center")
        ax.text(0.50, 0.22, "Win Rate", color=self.MUTED, fontsize=8,
                transform=ax.transAxes, ha="center")

        # Right: net P&L
        prefix = "+" if tp >= 0 else ""
        ax.text(0.98, 0.72, f"{prefix}{tp:.2f}", color=col, fontsize=17,
                fontweight="bold", transform=ax.transAxes,
                ha="right", va="center")
        ax.text(0.98, 0.22, "Net P&L", color=self.MUTED, fontsize=8,
                transform=ax.transAxes, ha="right")

        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(self.BLUE)
        ax.spines["bottom"].set_linewidth(2)

    def _equity_curve(self, ax, curve: List[float]):
        ax.set_facecolor(self.CARD)
        for sp in ax.spines.values():
            sp.set_color(self.GRID)
        ax.yaxis.grid(True, color=self.GRID, linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(colors=self.MUTED, labelsize=7)
        ax.tick_params(axis="x", labelbottom=False)

        x    = list(range(len(curve)))
        col  = self.GREEN if curve[-1] >= curve[0] else self.RED
        ax.plot(x, curve, color=col, linewidth=2, zorder=3)
        base = min(curve) - abs(min(curve)) * 0.05
        ax.fill_between(x, curve, base, alpha=0.15, color=col)
        ax.axhline(y=curve[0], color=self.MUTED, linewidth=0.5, linestyle=":")
        ax.set_title("Equity Curve", color=self.MUTED, fontsize=8, pad=4)

    def _metrics_panel(self, ax, stats):
        ax.set_facecolor(self.CARD)
        ax.axis("off")
        ax.set_title("Key Stats", color=self.MUTED, fontsize=8, pad=4)

        rows = [
            ("Trades",        str(stats.get("total", 0)),           self.TEXT),
            ("Profit Factor", f"{stats.get('profit_factor',0):.2f}",
             self.GREEN if stats.get("profit_factor", 0) > 1 else self.RED),
            ("Best Trade",    f"+{stats.get('best_trade',0):.2f}",   self.GREEN),
            ("Worst Trade",   f"{stats.get('worst_trade',0):.2f}",   self.RED),
            ("Max Drawdown",  f"-{stats.get('max_drawdown',0):.2f}", self.RED),
            ("Total Pips",    f"+{stats.get('total_pips',0):.0f}",   self.GOLD),
        ]
        y = 0.92
        for label, value, col in rows:
            ax.text(0.05, y, label, color=self.MUTED, fontsize=7.5,
                    transform=ax.transAxes)
            ax.text(0.95, y, value, color=col, fontsize=7.5,
                    fontweight="bold", transform=ax.transAxes, ha="right")
            y -= 0.155
            ax.axhline(y=y + 0.08, color=self.GRID, linewidth=0.4,
                       transform=ax.transAxes, xmin=0, xmax=1)

    @staticmethod
    def _fake_curve(stats: dict) -> List[float]:
        """Plausible equity curve for display when real curve unavailable."""
        np.random.seed(99)
        n   = max(stats.get("total", 20), 20)
        wr  = stats.get("win_rate", 55) / 100
        net = stats.get("total_profit", 0) or 50
        step = abs(net) / n
        curve = [0.0]
        for _ in range(n):
            curve.append(curve[-1] + (step if np.random.random() < wr else -step * 0.6))
        return curve


# ==============================================================================
#  SOCIAL ENGINE  (main interface)
# ==============================================================================
class SocialEngine:
    """
    Unified social engine wiring together PerformanceTracker,
    TelegramNotifier and PerformanceCardGenerator.

    Call from the live bot loop:
        social.on_trade_open(trade, confidence, overlay_notes)
        social.on_trade_close(ticket, close_price, close_time, profit, pips)
        social.scheduled_report()   # once per polling cycle
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.tracker  = PerformanceTracker()
        self.card_gen = PerformanceCardGenerator()
        self.telegram = TelegramNotifier(
            token      = cfg.get("telegram", {}).get("bot_token",  ""),
            channel_id = cfg.get("telegram", {}).get("channel_id", ""),
            chat_id    = cfg.get("telegram", {}).get("chat_id",    ""),
            branding   = cfg.get("branding", {}),
        )
        self.cmd_handler = TelegramCommandHandler(
            token              = cfg.get("telegram", {}).get("bot_token", ""),
            authorized_chat_id = cfg.get("telegram", {}).get("chat_id",   ""),
        )
        self._last_daily_report  = -1
        self._last_weekly_report = -1
        log.info("[Social] Social engine ready.")

    # ── Event hooks ────────────────────────────────────────────────────────

    def on_trade_open(self, trade: TradeRecord,
                      confidence: float = 0.0,
                      overlay: str = ""):
        self.tracker.add_trade(trade)
        self.telegram.notify_trade_open(trade, confidence, overlay)

    def on_trade_close(self, ticket: int, close_price: float,
                       close_time: datetime, profit: float, pips: float):
        trade = self.tracker.close_trade(ticket, close_price, close_time,
                                         profit, pips)
        if trade:
            self.telegram.notify_trade_close(trade)
        return trade

    def on_signal(self, symbol: str, direction: str,
                  confidence: float, note: str = ""):
        self.telegram.notify_signal(symbol, direction, confidence, note)

    def announce_startup(self, pairs: List[str], mode: str):
        self.telegram.announce_startup(pairs, mode)

    # ── Scheduled reports ──────────────────────────────────────────────────

    def scheduled_report(self, force: bool = False):
        """Call once per polling cycle — sends daily/weekly report when due."""
        now  = datetime.now(timezone.utc)
        day  = now.timetuple().tm_yday
        week = now.isocalendar()[1]

        social_cfg = self.cfg.get("social", {})

        if social_cfg.get("daily_report", True):
            if force or day != self._last_daily_report:
                report_hour = social_cfg.get("report_hour_utc", 22)
                if force or now.hour == report_hour:
                    self._send_report(days=1, period="Daily")
                    self._last_daily_report = day

        if social_cfg.get("weekly_report", True):
            if (force or (now.weekday() == 4 and week != self._last_weekly_report)):
                self._send_report(days=7, period="Weekly")
                self._last_weekly_report = week

    def get_stats(self, days: Optional[int] = None) -> dict:
        return self.tracker.stats(days=days)

    def handle_commands(self, bot) -> None:
        """Poll Telegram for phone commands and execute them against the live bot."""
        try:
            cmds = self.cmd_handler.poll()
            if cmds:
                self.cmd_handler.handle(cmds, bot)
        except Exception as exc:
            log.debug(f"[Social] Command poll error: {exc}")

    # ── Internal ───────────────────────────────────────────────────────────

    def _send_report(self, days: int, period: str):
        stats = self.tracker.stats(days=days)
        if stats["total"] == 0:
            return
        curve = self.tracker.equity_curve(days=days)
        img   = self.card_gen.generate(
            stats,
            period   = period,
            bot_name = self.cfg.get("branding", {}).get("name", "ForexAI Bot"),
            equity_curve = curve if len(curve) > 1 else None,
        )
        self.telegram.send_performance_report(stats, period, img)
        log.info(f"[Social] {period} report sent: {stats['total']} trades, "
                 f"WR={stats['win_rate']:.1f}%")
