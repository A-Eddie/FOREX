"""
================================================================================
 FOREX AI PLATFORM — Web Interface
================================================================================
 Run:
   pip install -r requirements_platform.txt
   python app.py

 Then open:  http://localhost:5000
================================================================================
"""
import copy
import os
import secrets
from urllib.parse import urljoin, urlparse

from flask import (
    Flask, abort, flash, jsonify, redirect,
    render_template, request, session, url_for,
)
from flask_login import (
    LoginManager, current_user, login_required,
    login_user, logout_user,
)

from bot_manager import BotManager
from crypto_utils import decrypt, encrypt
from models import MT5Account, User, db

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Persist the secret key so sessions survive restarts
_SECRET_FILE = os.path.join(os.path.dirname(__file__), ".flask_secret")
if os.path.exists(_SECRET_FILE):
    with open(_SECRET_FILE, "r") as _f:
        app.secret_key = _f.read().strip()
else:
    app.secret_key = secrets.token_hex(32)
    with open(_SECRET_FILE, "w") as _f:
        _f.write(app.secret_key)

app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:///forex_platform.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SAMESITE="Lax",
)

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access the dashboard."
login_manager.login_message_category = "warning"

bot_manager = BotManager()


# ── CSRF protection ───────────────────────────────────────────────────────────
@app.context_processor
def _inject_csrf():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return {"csrf_token": session["csrf_token"]}


def _check_csrf():
    if request.method == "POST":
        tok      = session.get("csrf_token", "")
        form_tok = request.form.get("csrf_token", "")
        if not tok or not form_tok or not secrets.compare_digest(tok, form_tok):
            abort(403)


def _is_safe_redirect_target(target: str) -> bool:
    if not target:
        return False
    host_url = urlparse(request.host_url)
    redirect_url = urlparse(urljoin(request.host_url, target))
    return redirect_url.scheme in ("http", "https") and host_url.netloc == redirect_url.netloc


# ── User loader ───────────────────────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


# ── Auth: register ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        _check_csrf()
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        error = None
        if not username or not email or not password:
            error = "All fields are required."
        elif len(username) < 3 or len(username) > 40:
            error = "Username must be 3–40 characters."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none():
            error = "Username already taken."
        elif db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none():
            error = "Email already registered."

        if error:
            flash(error, "danger")
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user, remember=True)
            flash("Account created! Enter your MT5 credentials below to start trading.", "success")
            return redirect(url_for("dashboard"))

    return render_template("register.html")


# ── Auth: login ───────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        _check_csrf()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = db.session.execute(
            db.select(User).filter_by(username=username)
        ).scalar_one_or_none()
        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get("next")
            if next_page and _is_safe_redirect_target(next_page):
                return redirect(next_page)
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")

    return render_template("login.html")


# ── Auth: logout ──────────────────────────────────────────────────────────────
@app.route("/logout")
@login_required
def logout():
    bot_manager.stop(current_user.id)
    logout_user()
    return redirect(url_for("login"))


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    acct   = db.session.execute(
        db.select(MT5Account).filter_by(user_id=current_user.id)
    ).scalar_one_or_none()
    status = bot_manager.get_status(current_user.id)
    return render_template("dashboard.html", acct=acct, status=status)


# ── Save MT5 credentials ──────────────────────────────────────────────────────
@app.route("/save-settings", methods=["POST"])
@login_required
def save_settings():
    _check_csrf()
    mt5_login    = request.form.get("mt5_login", "").strip()
    mt5_password = request.form.get("mt5_password", "").strip()
    mt5_server   = request.form.get("mt5_server", "").strip()
    live_mode    = request.form.get("live_mode") == "on"

    if not mt5_login.isdigit():
        flash("MT5 account number must be numeric.", "danger")
        return redirect(url_for("dashboard"))
    if not mt5_server:
        flash("Server name is required.", "danger")
        return redirect(url_for("dashboard"))

    acct = db.session.execute(
        db.select(MT5Account).filter_by(user_id=current_user.id)
    ).scalar_one_or_none()

    if acct is None:
        # First save — password is mandatory
        if not mt5_password:
            flash("Password is required for the first save.", "danger")
            return redirect(url_for("dashboard"))
        acct = MT5Account(user_id=current_user.id)
        db.session.add(acct)

    acct.mt5_login  = int(mt5_login)
    acct.mt5_server = mt5_server
    acct.live_mode  = live_mode
    if mt5_password:                    # Only update password when a new one is supplied
        acct.mt5_password_enc = encrypt(mt5_password)

    db.session.commit()
    flash("Credentials saved successfully.", "success")
    return redirect(url_for("dashboard"))


# ── Bot control: start ────────────────────────────────────────────────────────
@app.route("/start-bot", methods=["POST"])
@login_required
def start_bot():
    _check_csrf()
    acct = db.session.execute(
        db.select(MT5Account).filter_by(user_id=current_user.id)
    ).scalar_one_or_none()
    if acct is None or not acct.mt5_login:
        flash("Please save your MT5 credentials first.", "warning")
        return redirect(url_for("dashboard"))

    cfg = _build_cfg(acct)
    ok, msg = bot_manager.start(current_user.id, cfg)
    flash(msg, "success" if ok else "warning")
    return redirect(url_for("dashboard"))


# ── Bot control: stop ─────────────────────────────────────────────────────────
@app.route("/stop-bot", methods=["POST"])
@login_required
def stop_bot():
    _check_csrf()
    ok = bot_manager.stop(current_user.id)
    flash("Stop signal sent — bot will halt after the current tick." if ok
          else "Bot is not currently running.", "info")
    return redirect(url_for("dashboard"))


# ── API: live status (polled by dashboard JS) ─────────────────────────────────
@app.route("/api/status")
@login_required
def api_status():
    status   = bot_manager.get_status(current_user.id)
    acct_inf = bot_manager.get_account_info(current_user.id)
    return jsonify({**status, "account": acct_inf})


# ── API: test MT5 connection (called from dashboard "Test Connection" button) ─
@app.route("/api/test-connection", methods=["POST"])
@login_required
def api_test_connection():
    _check_csrf()
    acct = db.session.execute(
        db.select(MT5Account).filter_by(user_id=current_user.id)
    ).scalar_one_or_none()

    if acct is None or not acct.mt5_login:
        return jsonify({"ok": False, "error": "No MT5 credentials saved yet."})

    try:
        import MetaTrader5 as mt5  # noqa

        # Determine MT5 terminal path from config (optional)
        mt5_path = ""
        try:
            import config as _c
            mt5_path = getattr(_c, "MT5_PATH", "") or ""
        except ImportError:
            pass

        init_kwargs = {"path": mt5_path} if mt5_path else {}
        if not mt5.initialize(**init_kwargs):
            return jsonify({"ok": False, "error": f"MT5 terminal not found: {mt5.last_error()}"})

        ok = mt5.login(
            login=acct.mt5_login,
            password=decrypt(acct.mt5_password_enc),
            server=acct.mt5_server,
        )
        if not ok:
            err = mt5.last_error()
            mt5.shutdown()
            return jsonify({"ok": False, "error": f"Login failed: {err}"})

        info = mt5.account_info()
        mt5.shutdown()

        if info is None:
            return jsonify({"ok": False, "error": "Connected but could not retrieve account info."})

        return jsonify({
            "ok":       True,
            "login":    info.login,
            "name":     getattr(info, "name", ""),
            "server":   info.server,
            "balance":  round(float(info.balance), 2),
            "equity":   round(float(info.equity), 2),
            "currency": info.currency,
            "leverage": f"1:{info.leverage}",
            "mode":     "LIVE" if getattr(info, "trade_mode", 0) == 0 else "DEMO",
        })

    except ImportError:
        return jsonify({"ok": False, "error": "MetaTrader5 package not installed on server."})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


# ── Config builder ────────────────────────────────────────────────────────────
def _build_cfg(acct: MT5Account) -> dict:
    """Merge the user's stored MT5 credentials into a deep copy of the base CONFIG."""
    try:
        from forex_ai_bot import CONFIG as BASE  # noqa
    except ImportError:
        BASE = {}
    cfg = copy.deepcopy(BASE)
    cfg["mt5_login"]    = acct.mt5_login
    cfg["mt5_password"] = decrypt(acct.mt5_password_enc)
    cfg["mt5_server"]   = acct.mt5_server
    return cfg


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("\n  Forex AI Platform  →  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
