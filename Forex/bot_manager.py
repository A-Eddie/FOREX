"""
Thread-safe manager that starts / stops one ForexAIBot per platform user.
Each user's bot runs in a daemon thread and can be stopped cleanly via a
threading.Event which is passed to ForexAIBot.live(stop_event=...).
"""
import copy
import logging
import threading
from datetime import datetime
from typing import Dict, Optional

log = logging.getLogger("BotManager")


class _BotInstance:
    """State holder for a single user's running bot."""
    __slots__ = (
        "bot", "thread", "stop_event",
        "started_at", "status", "error",
    )

    def __init__(self) -> None:
        self.bot        = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.started_at: Optional[datetime]     = None
        self.status: str = "stopped"   # stopped | starting | running | stopping
        self.error: Optional[str]       = None


class BotManager:
    """
    Public API
    ----------
    start(user_id, cfg)     -> (bool, str)   start a bot for this user
    stop(user_id)           -> bool          send stop signal
    get_status(user_id)     -> dict
    get_account_info(user_id) -> dict        live MT5 account figures
    """

    def __init__(self) -> None:
        self._instances: Dict[int, _BotInstance] = {}
        self._lock = threading.Lock()

    # ── start ─────────────────────────────────────────────────────────────
    def start(self, user_id: int, cfg: dict) -> tuple:
        with self._lock:
            inst = self._instances.get(user_id)
            if inst and inst.status in ("running", "starting"):
                return False, "Bot is already running."
            # Reuse or create instance
            inst = _BotInstance()
            self._instances[user_id] = inst
            inst.status = "starting"

        thread = threading.Thread(
            target=self._run,
            args=(user_id, inst, copy.deepcopy(cfg)),
            daemon=True,
            name=f"forex-bot-{user_id}",
        )
        inst.thread = thread
        thread.start()
        log.info(f"[BotManager] Bot starting for user {user_id}.")
        return True, "Bot is starting up — this may take a few seconds."

    # ── stop ──────────────────────────────────────────────────────────────
    def stop(self, user_id: int) -> bool:
        inst = self._instances.get(user_id)
        if inst is None or inst.status not in ("running", "starting"):
            return False
        inst.stop_event.set()
        inst.status = "stopping"
        log.info(f"[BotManager] Stop signal sent for user {user_id}.")
        return True

    # ── status ────────────────────────────────────────────────────────────
    def get_status(self, user_id: int) -> dict:
        inst = self._instances.get(user_id)
        if inst is None:
            return {"status": "stopped", "started_at": None, "error": None}
        # Sync status if the thread has died
        if inst.thread and not inst.thread.is_alive() and inst.status != "stopped":
            inst.status = "stopped"
        return {
            "status":     inst.status,
            "started_at": inst.started_at.isoformat() if inst.started_at else None,
            "error":      inst.error,
        }

    # ── live account info ─────────────────────────────────────────────────
    def get_account_info(self, user_id: int) -> dict:
        inst = self._instances.get(user_id)
        if inst is None or inst.bot is None or inst.status != "running":
            return {}
        try:
            connector = inst.bot.connector
            if not connector.connected:
                return {}
            import MetaTrader5 as mt5  # noqa
            info = mt5.account_info()
            if info is None:
                return {}
            return {
                "login":    info.login,
                "server":   info.server,
                "balance":  round(float(info.balance), 2),
                "equity":   round(float(info.equity), 2),
                "profit":   round(float(info.profit), 2),
                "currency": info.currency,
                "leverage": f"1:{info.leverage}",
            }
        except Exception as exc:
            log.debug(f"get_account_info error for user {user_id}: {exc}")
            return {}

    # ── internal runner (executes in daemon thread) ───────────────────────
    def _run(self, user_id: int, inst: _BotInstance, cfg: dict) -> None:
        try:
            from forex_ai_bot import ForexAIBot  # noqa
            inst.bot        = ForexAIBot(cfg)
            inst.status     = "running"
            inst.started_at = datetime.utcnow()
            log.info(f"[BotManager] Bot running for user {user_id}.")
            inst.bot.live(stop_event=inst.stop_event)
        except Exception as exc:
            inst.error = str(exc)
            log.error(f"[BotManager] Bot error for user {user_id}: {exc}", exc_info=True)
        finally:
            inst.status = "stopped"
            log.info(f"[BotManager] Bot stopped for user {user_id}.")
