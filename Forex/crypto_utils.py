"""
Encrypt / decrypt MT5 passwords stored in the database.
Uses Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` package.
The key is generated once and saved to .enc.key beside this file.
"""
import os
from cryptography.fernet import Fernet

_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".enc.key")


def _get_key() -> bytes:
    if os.path.exists(_KEY_FILE):
        with open(_KEY_FILE, "rb") as f:
            return f.read().strip()
    key = Fernet.generate_key()
    with open(_KEY_FILE, "wb") as f:
        f.write(key)
    return key


def encrypt(plaintext: str) -> str:
    """Return Fernet-encrypted ciphertext as a UTF-8 string."""
    return Fernet(_get_key()).encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(ciphertext: str) -> str:
    """Decrypt a previously encrypted string back to plaintext."""
    return Fernet(_get_key()).decrypt(ciphertext.encode("utf-8")).decode("utf-8")
