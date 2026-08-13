from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from fastapi import Header, HTTPException

from backend.app.core.config import settings


def _b64_encode(raw: bytes) -> str:
    return (
        base64.urlsafe_b64encode(raw)
        .rstrip(b"=")
        .decode("ascii")
    )


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)

    return base64.urlsafe_b64decode(
        value + padding
    )


def _sign(payload: bytes) -> str:
    signature = hmac.new(
        settings.admin_secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).digest()

    return _b64_encode(signature)


def verify_credentials(
    username: str,
    password: str,
) -> bool:
    """Compare les identifiants en temps constant."""
    username_ok = secrets.compare_digest(
        username.strip(),
        settings.admin_username,
    )
    password_ok = secrets.compare_digest(
        password,
        settings.admin_password,
    )

    return username_ok and password_ok


def create_admin_token(
    username: str,
) -> tuple[str, int]:
    """Crée un jeton signé et retourne (jeton, expiration)."""
    expires_at = int(
        time.time()
        + settings.admin_token_ttl_minutes * 60
    )

    payload = {
        "sub": username,
        "exp": expires_at,
        "nonce": secrets.token_hex(8),
    }

    payload_bytes = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    encoded_payload = _b64_encode(payload_bytes)
    signature = _sign(
        encoded_payload.encode("ascii")
    )

    return (
        f"{encoded_payload}.{signature}",
        expires_at,
    )


def decode_admin_token(token: str) -> dict[str, Any]:
    """Vérifie la signature et l'expiration du jeton."""
    try:
        encoded_payload, signature = token.split(".", 1)
    except ValueError as error:
        raise ValueError("Jeton malformé.") from error

    expected_signature = _sign(
        encoded_payload.encode("ascii")
    )

    if not hmac.compare_digest(
        signature,
        expected_signature,
    ):
        raise ValueError("Signature du jeton invalide.")

    try:
        payload = json.loads(
            _b64_decode(encoded_payload)
        )
    except Exception as error:
        raise ValueError(
            "Contenu du jeton illisible."
        ) from error

    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Jeton expiré.")

    return payload


def require_admin(
    authorization: str | None = Header(default=None),
) -> str:
    """Dépendance FastAPI protégeant l'espace administration."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentification requise.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="En-tête d'autorisation invalide.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_admin_token(token.strip())
    except ValueError as error:
        raise HTTPException(
            status_code=401,
            detail=str(error),
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    return str(payload.get("sub", ""))
