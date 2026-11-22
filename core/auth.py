import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

security = HTTPBearer()

# Supabase issues access tokens with aud="authenticated" for a signed-in user.
# PyJWT rejects any token carrying an `aud` claim unless it's told which
# audience to expect, so this must be passed explicitly.
SUPABASE_AUDIENCE = "authenticated"

# Real Supabase tokens are signed with ES256 — asymmetric keys, verified
# against the public keys the project publishes at its JWKS endpoint. The
# older shared-secret (HS256) scheme does NOT verify them: a genuine token
# fails with InvalidAlgorithmError.
#
# HS256 is still accepted, but only in development, so locally minted test
# tokens (see /dev/token) keep working. A deployed environment accepts
# nothing but real Supabase-signed tokens.
_ASYMMETRIC_ALGORITHMS = ["ES256", "RS256"]

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    """Built lazily so the app still starts if SUPABASE_URL is missing, and
    so no network call happens at import time. PyJWKClient caches the keys
    it fetches, so this is not a per-request round trip."""
    global _jwks_client
    if _jwks_client is None:
        base = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        if not base:
            raise HTTPException(status_code=500, detail="SUPABASE_URL is not configured")
        _jwks_client = PyJWKClient(f"{base}/auth/v1/.well-known/jwks.json")
    return _jwks_client


def _decode_asymmetric(token: str) -> dict:
    signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=_ASYMMETRIC_ALGORITHMS,
        audience=SUPABASE_AUDIENCE,
    )


def _decode_shared_secret(token: str) -> dict:
    return jwt.decode(
        token,
        os.getenv("SUPABASE_JWT_SECRET"),
        algorithms=["HS256"],
        audience=SUPABASE_AUDIENCE,
    )


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials

    try:
        algorithm = jwt.get_unverified_header(token).get("alg")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        if algorithm in _ASYMMETRIC_ALGORITHMS:
            payload = _decode_asymmetric(token)
        elif algorithm == "HS256" and os.getenv("ENVIRONMENT") == "development":
            payload = _decode_shared_secret(token)
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload.get("sub")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception:
        # A JWKS fetch can fail (network, misconfigured URL). That's a server
        # problem, not a bad token — saying "invalid token" would send the
        # caller chasing the wrong thing.
        raise HTTPException(status_code=503, detail="Could not verify token right now")
