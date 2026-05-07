"""
SQLAlchemy models for the Forex AI Platform.
  - User          : platform account (login/password)
  - MT5Account    : the user's MT5 broker credentials (password stored encrypted)
"""
from datetime import datetime

import bcrypt
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    is_active     = db.Column(db.Boolean, default=True, nullable=False)

    mt5_account = db.relationship("MT5Account", backref="user", uselist=False,
                                  cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            self.password_hash.encode("utf-8"),
        )


class MT5Account(db.Model):
    __tablename__ = "mt5_accounts"

    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    mt5_login         = db.Column(db.Integer, nullable=False)
    mt5_password_enc  = db.Column(db.Text, nullable=False)   # Fernet-encrypted
    mt5_server        = db.Column(db.String(100), nullable=False)
    live_mode         = db.Column(db.Boolean, default=False, nullable=False)
    updated_at        = db.Column(db.DateTime, default=datetime.utcnow,
                                  onupdate=datetime.utcnow)
