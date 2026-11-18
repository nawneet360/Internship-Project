"""
core/auth.py
------------
Zero-external-dependency authentication utilities:
  • PBKDF2-HMAC-SHA256 password hashing & constant-time verification
  • HS256 JSON Web Token (JWT) encode, decode, and verification
  • Secure random refresh token generation
  • Flask auth decorators (@jwt_required, @jwt_optional)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable

from flask import current_app, g, jsonify, request

from core import database as db

log = logging.getLogger("disasterai.auth")

# ── Password Hashing (PBKDF2-HMAC-SHA256) ──────────────────────────────────────

_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with a cryptographically secure salt."""
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"{salt.hex()}${key.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Constant-time password verification against stored salt$hash."""
    try:
        salt_hex, key_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        candidate_key = hashlib.pbkdf2_hmac(
            "sha256", plain_password.encode("utf-8"), salt, _ITERATIONS
        )
        return hmac.compare_digest(candidate_key, expected_key)
    except Exception as exc:
        log.warning("Password verification exception: %s", exc)
        return False


# ── HS256 JWT Implementation (Standard Library Only) ──────────────────────────

def _b64url_encode(data: bytes) -> str:
    """Base64 URL-safe encode without padding (=)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    """Base64 URL-safe decode adding padding if necessary."""
    rem = len(data) % 4
    if rem > 0:
        data += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data.encode("ascii"))


def create_jwt(payload: dict, secret: str, expires_in_seconds: int = 3600) -> str:
    """Create a signed HS256 JWT token."""
    now = int(time.time())
    full_payload = {
        **payload,
        "iat": now,
        "exp": now + expires_in_seconds,
    }

    header = {"alg": "HS256", "typ": "JWT"}

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(full_payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_jwt(token: str, secret: str) -> dict:
    """
    Verify and decode an HS256 JWT token.
    Raises ValueError on invalid token, invalid signature, or expired token.
    """
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format — expected 3 parts")

    header_b64, payload_b64, sig_b64 = parts

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid token signature")

    payload_raw = _b64url_decode(payload_b64)
    payload = json.loads(payload_raw.decode("utf-8"))

    now = int(time.time())
    if "exp" in payload and payload["exp"] < now:
        raise ValueError("Token has expired")

    return payload


# ── Token Factory Helpers ──────────────────────────────────────────────────────

def generate_tokens(user_id: str, username: str, email: str) -> tuple[str, str, str]:
    """
    Generate an access token and a refresh token.
    Saves the refresh token into SQLite.
    Returns: (access_token, refresh_token, refresh_expires_at_iso)
    """
    secret = current_app.config.get("JWT_SECRET_KEY", "fallback-secret-key")
    access_mins = current_app.config.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60)
    refresh_days = current_app.config.get("REFRESH_TOKEN_EXPIRE_DAYS", 30)

    access_token = create_jwt(
        payload={"sub": user_id, "username": username, "email": email},
        secret=secret,
        expires_in_seconds=access_mins * 60,
    )

    refresh_token = secrets.token_urlsafe(48)
    refresh_expires_dt = datetime.now(timezone.utc) + timedelta(days=refresh_days)
    refresh_expires_at = refresh_expires_dt.isoformat()

    db.save_refresh_token(refresh_token, user_id, refresh_expires_at)

    return access_token, refresh_token, refresh_expires_at


# ── Flask Decorators ───────────────────────────────────────────────────────────

def _extract_token_from_header() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None


def jwt_required(fn: Callable):
    """
    Decorator: ensures request has a valid Bearer JWT.
    Sets `g.current_user` to the authenticated user dict.
    Returns 401 on missing, expired, or invalid token.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _extract_token_from_header()
        if not token:
            return jsonify({"error": "Unauthorized", "details": "Authorization header with Bearer token is required"}), 401

        secret = current_app.config.get("JWT_SECRET_KEY", "fallback-secret-key")
        try:
            payload = decode_jwt(token, secret)
        except ValueError as err:
            return jsonify({"error": "Unauthorized", "details": str(err)}), 401

        user_id = payload.get("sub")
        user = db.get_user_by_id(user_id)
        if not user:
            return jsonify({"error": "Unauthorized", "details": "User account no longer exists or is inactive"}), 401

        g.current_user = {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "created_at": user["created_at"],
        }
        return fn(*args, **kwargs)

    return wrapper


def jwt_optional(fn: Callable):
    """
    Decorator: if a Bearer JWT is present and valid, sets `g.current_user`.
    If no token or invalid, sets `g.current_user = None` and proceeds without error.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        g.current_user = None
        token = _extract_token_from_header()
        if token:
            secret = current_app.config.get("JWT_SECRET_KEY", "fallback-secret-key")
            try:
                payload = decode_jwt(token, secret)
                user_id = payload.get("sub")
                user = db.get_user_by_id(user_id)
                if user:
                    g.current_user = {
                        "id": user["id"],
                        "username": user["username"],
                        "email": user["email"],
                        "created_at": user["created_at"],
                    }
            except Exception:
                g.current_user = None

        return fn(*args, **kwargs)

    return wrapper
