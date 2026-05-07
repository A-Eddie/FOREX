"""
================================================================================
 MULTI-ACCOUNT MANAGER — Trade copying across multiple MT5 accounts
================================================================================
 Architecture:
   Master account  →  generates signals via the main bot
   Slave accounts  →  automatically receive a scaled copy of every master trade

 MT5 Limitation:
   The MetaTrader5 Python library supports only ONE active connection per
   Python process.  This manager uses a "sequential switcher" strategy:
     1.  Master places trade (bot already connected to master)
     2.  For each slave:
             mt5.shutdown()
             mt5.initialize(path=slave.mt5_path)
             mt5.login(slave credentials)
             place trade
             mt5.shutdown()
     3.  Reconnect to master  (bot calls connector.connect() after this returns)

   For fully simultaneous connections, run multiple MT5 terminal executables
   on the same PC at different paths and set mt5_path per slave account.
================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

log = logging.getLogger("ForexBot.MultiAccount")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


# ==============================================================================
#  ACCOUNT PROFILE
# ==============================================================================
@dataclass
class AccountProfile:
    """Definition of a single MT5 trading account."""
    name:            str
    login:           int
    password:        str
    server:          str
    mt5_path:        str   = ""
    is_master:       bool  = False
    risk_pct:        float = 1.0
    lot_multiplier:  float = 1.0
    notify_telegram: bool  = False
    active:          bool  = True

    _FIELDS = {
        "name", "login", "password", "server", "mt5_path",
        "is_master", "risk_pct", "lot_multiplier", "notify_telegram", "active",
    }

    @classmethod
    def from_dict(cls, d: dict) -> "AccountProfile":
        return cls(**{k: v for k, v in d.items() if k in cls._FIELDS})


# ==============================================================================
#  ACCOUNT SESSION
# ==============================================================================
class AccountSession:
    """Manages the MT5 connection lifecycle for one account."""

    def __init__(self, profile: AccountProfile):
        self.profile   = profile
        self.connected = False

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            log.warning(f"[{self.profile.name}] MT5 library not available.")
            return False

        init_kwargs: Dict[str, Any] = {}
        if self.profile.mt5_path:
            init_kwargs["path"] = self.profile.mt5_path

        if not mt5.initialize(**init_kwargs):
            log.error(f"[{self.profile.name}] mt5.initialize() failed: {mt5.last_error()}")
            return False

        if self.profile.login:
            ok = mt5.login(
                login=self.profile.login,
                password=self.profile.password,
                server=self.profile.server,
            )
            if not ok:
                log.error(f"[{self.profile.name}] Login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        info = mt5.account_info()
        if info is None:
            log.error(f"[{self.profile.name}] Could not retrieve account info.")
            mt5.shutdown()
            return False

        log.info(
            f"[{self.profile.name}] Connected ✓  "
            f"#{info.login} | {info.server} | "
            f"Balance: {info.balance:.2f} {info.currency}"
        )
        self.connected = True
        return True

    def disconnect(self):
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            log.debug(f"[{self.profile.name}] Disconnected.")

    def account_info_dict(self) -> dict:
        if not MT5_AVAILABLE or not self.connected:
            return {}
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "login":    info.login,
            "server":   info.server,
            "balance":  info.balance,
            "equity":   info.equity,
            "profit":   info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
        }


# ==============================================================================
#  MULTI-ACCOUNT MANAGER
# ==============================================================================
class MultiAccountManager:
    """
    Orchestrates signal broadcasting from the master account to slave accounts.

    Typical usage inside the live bot loop:

        # After master places a trade:
        results = manager.copy_trade_to_slaves(symbol, "BUY", lot, sl_pips, tp_pips)
        # Then reconnect master:
        connector.connect()   ← the bot is responsible for this step
    """

    def __init__(self, accounts: List[dict]):
        self.profiles: List[AccountProfile] = [
            AccountProfile.from_dict(a)
            for a in accounts
            if a.get("active", True) and a.get("login", 0)
        ]
        self.master: Optional[AccountProfile] = next(
            (p for p in self.profiles if p.is_master),
            self.profiles[0] if self.profiles else None,
        )
        self.slaves: List[AccountProfile] = [
            p for p in self.profiles if not p.is_master
        ]
        log.info(
            f"[MultiAccount] {len(self.profiles)} account(s) loaded — "
            f"master: {self.master.name if self.master else 'none'} | "
            f"slaves: {len(self.slaves)}"
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def has_slaves(self) -> bool:
        return len(self.slaves) > 0

    def copy_trade_to_slaves(
        self,
        symbol:    str,
        direction: str,       # "BUY" or "SELL"
        master_lot: float,
        sl_pips:   float,
        tp_pips:   float,
        magic:     int = 20240101,
    ) -> Dict[str, bool]:
        """
        Connect to each active slave account and place the same trade,
        scaled by that slave's lot_multiplier.

        Returns {account_name: success_bool}  for every slave attempted.
        MT5 is left disconnected after this call — the caller MUST call
        connector.connect() to re-establish the master connection.
        """
        if not self.has_slaves:
            return {}

        results: Dict[str, bool] = {}
        for slave in self.slaves:
            if not slave.active:
                continue
            lot = max(0.01, round(master_lot * slave.lot_multiplier, 2))
            log.info(
                f"[MultiAccount] Copying {direction} {symbol} "
                f"({lot:.2f} lots) → {slave.name}"
            )
            ok = self._place_on_account(slave, symbol, direction,
                                        lot, sl_pips, tp_pips, magic)
            results[slave.name] = ok
            time.sleep(0.3)  # brief pause between connections

        return results

    def get_all_account_stats(self) -> List[dict]:
        """
        Connect to every account, fetch account info, disconnect.
        Returns list of dicts — each with account_name, is_master, balance, etc.
        """
        results: List[dict] = []
        for profile in self.profiles:
            session = AccountSession(profile)
            if session.connect():
                info = session.account_info_dict()
                info["account_name"] = profile.name
                info["is_master"]    = profile.is_master
                results.append(info)
                session.disconnect()
                time.sleep(0.4)
        return results

    def print_accounts_table(self, stats: Optional[List[dict]] = None):
        """Print a nicely formatted table of all account balances."""
        if stats is None:
            stats = self.get_all_account_stats()
        if not stats:
            log.info("[MultiAccount] No account data available.")
            return

        log.info(
            f"\n{'═'*72}\n"
            f"{'Account':<24} {'Type':<8} {'Login':>10} "
            f"{'Balance':>12} {'Equity':>12} {'Open P&L':>10}\n"
            f"{'─'*72}"
        )
        for a in stats:
            tag = "MASTER" if a.get("is_master") else "SLAVE"
            log.info(
                f"{a.get('account_name','?'):<24} {tag:<8} "
                f"{a.get('login',0):>10} "
                f"{a.get('balance',0):>12.2f} "
                f"{a.get('equity',0):>12.2f} "
                f"{a.get('profit',0):>10.2f}"
            )
        log.info("═" * 72)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _place_on_account(
        self,
        profile:   AccountProfile,
        symbol:    str,
        direction: str,
        lot:       float,
        sl_pips:   float,
        tp_pips:   float,
        magic:     int,
    ) -> bool:
        """Connect to a slave account, place the trade, then disconnect."""
        if not MT5_AVAILABLE:
            log.info(f"[SIM] Copy {direction} {symbol} {lot:.2f}L → {profile.name}")
            return True

        session = AccountSession(profile)
        if not session.connect():
            log.error(f"[MultiAccount] Could not connect to {profile.name}")
            return False

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                log.error(f"[{profile.name}] No tick data for {symbol}")
                return False

            sym_info   = mt5.symbol_info(symbol)
            pip_factor = (sym_info.point * 10) if sym_info else 0.0001

            if direction == "BUY":
                price   = tick.ask
                sl_price = round(price - sl_pips * pip_factor, 5)
                tp_price = round(price + tp_pips * pip_factor, 5)
                order_type = mt5.ORDER_TYPE_BUY
            else:
                price    = tick.bid
                sl_price = round(price + sl_pips * pip_factor, 5)
                tp_price = round(price - tp_pips * pip_factor, 5)
                order_type = mt5.ORDER_TYPE_SELL

            # Clamp to broker's min/max volume
            if sym_info:
                lot = max(sym_info.volume_min,
                          min(round(lot, 2), sym_info.volume_max))

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       lot,
                "type":         order_type,
                "price":        price,
                "sl":           sl_price,
                "tp":           tp_price,
                "deviation":    20,
                "magic":        magic,
                "comment":      f"ForexAI_Copy",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(
                    f"[{profile.name}] Copy failed: "
                    f"retcode={result.retcode} — {result.comment}"
                )
                return False

            log.info(
                f"✅ [{profile.name}] {direction} {symbol} "
                f"{lot:.2f} lots @ {price:.5f} | "
                f"SL {sl_price:.5f} | TP {tp_price:.5f}"
            )
            return True

        except Exception as exc:
            log.error(f"[{profile.name}] Copy exception: {exc}")
            return False
        finally:
            session.disconnect()
