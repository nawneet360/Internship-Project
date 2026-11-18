"""
api/auth_routes.py
------------------
Authentication REST API:
  • POST /api/auth/register  → create new user account + return JWT
  • POST /api/auth/login     → verify credentials + return JWT
  • POST /api/auth/refresh   → exchange refresh token for new access token
  • POST /api/auth/logout    → revoke refresh token
  • GET  /api/auth/me        → get profile of currently logged-in user
  • GET  /api/auth/my-predictions → get saved predictions of current user
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from flask import Blueprint, current_app, g, jsonify, request

from core import auth
from core import database as db

log = logging.getLogger("disasterai.api.auth")

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

_EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{3,30}$")


def _error(message: str, details=None, code: int = 400):
    body = {"error": message}
    if details:
        body["details"] = details
    return jsonify(body), code


def _ok(data: dict, code: int = 200):
    return jsonify(data), code


# ── Register ───────────────────────────────────────────────────────────────────

@auth_bp.post("/register")
def register():
    """
    Register a new user account.
    Expects JSON: { "username": "...", "email": "...", "password": "..." }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("Request body must be valid JSON object", code=400)

    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    errors = {}
    if not username:
        errors["username"] = "Username is required"
    elif not _USERNAME_REGEX.match(username):
        errors["username"] = "Username must be 3-30 characters (letters, numbers, _ and -)"
    elif db.username_exists(username):
        errors["username"] = "Username is already taken"

    if not email:
        errors["email"] = "Email is required"
    elif not _EMAIL_REGEX.match(email):
        errors["email"] = "Please provide a valid email address"
    elif db.email_exists(email):
        errors["email"] = "An account with this email already exists"

    if not password:
        errors["password"] = "Password is required"
    elif len(password) < 6:
        errors["password"] = "Password must be at least 6 characters long"

    if errors:
        return _error("Validation failed", details=errors, code=400)

    user_id = str(uuid.uuid4())
    pw_hash = auth.hash_password(password)
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        user_info = db.create_user(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=pw_hash,
            created_at=now_iso,
        )
    except Exception as exc:
        log.exception("Error creating user: %s", exc)
        return _error("Could not register user", details=str(exc), code=500)

    access_token, refresh_token, expires_at = auth.generate_tokens(
        user_id=user_id, username=user_info["username"], email=user_info["email"]
    )

    log.info("Registered new user: %s (%s)", username, email)
    return _ok({
        "message": "Account created successfully",
        "user": user_info,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in_seconds": current_app.config.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60) * 60,
    }, code=201)


# ── Login ──────────────────────────────────────────────────────────────────────

@auth_bp.post("/login")
def login():
    """
    Authenticate with email/username + password.
    Expects JSON: { "login": "username or email", "password": "..." }
    or:           { "email": "...", "password": "..." }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("Request body must be valid JSON object", code=400)

    login_identifier = str(data.get("login") or data.get("email") or data.get("username") or "").strip()
    password = str(data.get("password", ""))

    if not login_identifier or not password:
        return _error("Email/username and password are required", code=400)

    user = db.get_user_by_identifier(login_identifier)
    if not user:
        return _error("Invalid email/username or password", code=401)

    if not auth.verify_password(password, user["password_hash"]):
        return _error("Invalid email/username or password", code=401)

    access_token, refresh_token, expires_at = auth.generate_tokens(
        user_id=user["id"], username=user["username"], email=user["email"]
    )

    user_info = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"],
    }

    log.info("User logged in: %s", user["username"])
    return _ok({
        "message": "Login successful",
        "user": user_info,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in_seconds": current_app.config.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60) * 60,
    })


# ── Refresh Token ──────────────────────────────────────────────────────────────

@auth_bp.post("/refresh")
def refresh():
    """
    Exchange a valid refresh token for a new access token.
    Expects JSON: { "refresh_token": "..." }
    """
    data = request.get_json(silent=True) or {}
    token = data.get("refresh_token") or request.headers.get("X-Refresh-Token")

    if not token:
        return _error("Refresh token is required", code=400)

    token_row = db.get_refresh_token(token)
    if not token_row:
        return _error("Invalid or revoked refresh token", code=401)

    # Check expiration
    try:
        exp_dt = datetime.fromisoformat(token_row["expires_at"])
        if datetime.now(timezone.utc) > exp_dt:
            db.revoke_refresh_token(token)
            return _error("Refresh token has expired — please sign in again", code=401)
    except Exception:
        pass

    user = db.get_user_by_id(token_row["user_id"])
    if not user:
        return _error("User no longer exists", code=401)

    secret = current_app.config.get("JWT_SECRET_KEY", "fallback-secret-key")
    access_mins = current_app.config.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60)

    new_access_token = auth.create_jwt(
        payload={"sub": user["id"], "username": user["username"], "email": user["email"]},
        secret=secret,
        expires_in_seconds=access_mins * 60,
    )

    return _ok({
        "access_token": new_access_token,
        "token_type": "Bearer",
        "expires_in_seconds": access_mins * 60,
    })


# ── Logout ─────────────────────────────────────────────────────────────────────

@auth_bp.post("/logout")
def logout():
    """
    Revoke a refresh token.
    Expects JSON: { "refresh_token": "..." } (optional)
    """
    data = request.get_json(silent=True) or {}
    token = data.get("refresh_token")
    if token:
        db.revoke_refresh_token(token)

    return _ok({"message": "Logged out successfully"})


# ── Get Current Profile ────────────────────────────────────────────────────────

@auth_bp.get("/me")
@auth.jwt_required
def me():
    """Get the profile of the currently signed-in user."""
    return _ok({
        "user": g.current_user
    })


# ── User's Saved Predictions ───────────────────────────────────────────────────

@auth_bp.get("/my-predictions")
@auth.jwt_required
def my_predictions():
    """Get the saved prediction history for the current user."""
    disaster_type = request.args.get("type")
    limit = min(int(request.args.get("limit", 20)), 100)
    preds = db.get_predictions(
        user_id=g.current_user["id"],
        disaster_type=disaster_type,
        limit=limit,
    )
    return _ok({
        "count": len(preds),
        "predictions": preds,
    })
