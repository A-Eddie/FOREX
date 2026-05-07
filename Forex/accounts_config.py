# ============================================================
#  FOREX AI BOT v3.0 — Multi-Account + Social Media Config
# ============================================================
# NEVER commit this file to version control.
#
# QUICK START:
#   1. Fill in your MT5 account(s) below
#   2. Create a Telegram bot via @BotFather on Telegram
#   3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID
#   4. Customise BOT_DISPLAY_NAME and social handles
#   5. Run: python forex_ai_bot.py --mode live
# ─────────────────────────────────────────────────────────────

# ==============================================================================
#  ACCOUNTS
# ==============================================================================
# Add one dict per MT5 account.
#
# MASTER:
#   is_master = True → this account generates all trading signals.
#   Only the FIRST master account is used as signal source.
#
# SLAVES:
#   is_master = False → this account copies every trade placed by the master,
#   scaled by its own lot_multiplier and risk_pct.
#
# MULTIPLE MT5 TERMINALS:
#   If you have multiple MT5 installations on the same PC, point each slave to
#   its own terminal executable via mt5_path.
#   e.g. mt5_path = "C:\\MT5_Slave1\\terminal64.exe"
#   Leave blank to use the default (auto-detected) terminal.
#
# SINGLE TERMINAL:
#   If you only have one MT5 terminal, the bot will sequentially connect to each
#   account (disconnect → re-connect) when copying trades.  The whole copy cycle
#   takes ~1–3 seconds per slave, which is fine for H1 trading.
#
# ─────────────────────────────────────────────────────────────

ACCOUNTS = [
    # ── Master Account ───────────────────────────────────────────────────────
    {
        "name":            "Master — Main",
        "login":           1126947,
        "password":        "Eddiewave@79",
        "server":          "Equiti-Demo",
        "mt5_path":        "C:\\Program Files\\EGM Securities MetaTrader 5 Terminal\\terminal64.exe",
        "is_master":       True,    # This account generates the signals
        "risk_pct":        1.0,     # Risk 1 % of balance per trade
        "lot_multiplier":  1.0,     # Base lot multiplier (1.0 = no change)
        "notify_telegram": True,    # Broadcast trade alerts to Telegram channel
        "active":          True,
    },

    # ── Slave Account — add as many as you like ──────────────────────────────
    # Uncomment and fill in to activate a second account:
    # {
    #     "name":            "Slave — Conservative",
    #     "login":           0,
    #     "password":        "",
    #     "server":          "",
    #     "mt5_path":        "C:\\MT5_Slave1\\terminal64.exe",
    #     "is_master":       False,
    #     "risk_pct":        0.5,    # Half the master's risk per trade
    #     "lot_multiplier":  0.5,    # Copy at 50 % of master lot size
    #     "notify_telegram": False,
    #     "active":          True,
    # },
    # {
    #     "name":            "Slave — Aggressive",
    #     "login":           0,
    #     "password":        "",
    #     "server":          "",
    #     "mt5_path":        "C:\\MT5_Slave2\\terminal64.exe",
    #     "is_master":       False,
    #     "risk_pct":        2.0,    # Double the master's risk
    #     "lot_multiplier":  2.0,
    #     "notify_telegram": False,
    #     "active":          True,
    # },
]

# ==============================================================================
#  TELEGRAM SETUP
# ==============================================================================
# How to create your bot + channel:
#   1. Open Telegram → search @BotFather → /newbot → follow prompts
#   2. Copy the bot token shown by BotFather
#   3. Create a public channel (or private group) and add your bot as Admin
#   4. To get the channel/chat ID:
#      • Forward any message from your channel/group to @userinfobot
#      • The ID looks like: -1001234567890   (channels always negative & 13 digits)
#      • Or use your public @channel_username directly
#
# TELEGRAM_CHANNEL_ID  → used for public signal broadcasts (trade alerts, reports)
# TELEGRAM_CHAT_ID     → your personal chat ID for private error/monitoring alerts
#                        (optional — leave blank to use the channel for everything)

TELEGRAM_BOT_TOKEN  = "8231014650:AAEyt0lIBSSLWs0rFII0vBPt1Q_H1kpOF4s"
TELEGRAM_CHANNEL_ID = "-1003781121986"  # Wave AI channel
TELEGRAM_CHAT_ID    = "7611578274"      # Eddie Wave (personal alerts)

# ==============================================================================
#  BRANDING  — shown in every Telegram message + performance card
# ==============================================================================
BOT_DISPLAY_NAME = "ForexAI Bot"
BOT_TAGLINE      = "AI-Powered Forex Signals | 29 Pairs + Gold"
BOT_WEBSITE      = ""           # e.g. "https://t.me/your_channel"
BOT_INSTAGRAM    = ""           # e.g. "@yourhandle"
BOT_TWITTER      = ""           # e.g. "@yourhandle"
