"""
Exercises verify_token directly against realistically-shaped tokens.

The endpoint tests override the auth dependency so they can focus on
behavior, which means verify_token itself goes unexercised there. That gap
is how a real bug shipped: tokens were hand-rolled with only `sub` and
`exp`, while real Supabase access tokens also carry `aud`, and PyJWT
rejects an `aud` it wasn't told to expect. These tests use the real token
shape so that can't silently happen again.
"""

import time
import uuid

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from core.auth import SUPABASE_AUDIENCE, verify_token

# At least 32 bytes — shorter HMAC keys make PyJWT warn on every call.
SECRET = "test-secret-not-the-real-one-padded-to-32-bytes-plus"
PATIENT_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)


def _supabase_token(secret: str = SECRET, **overrides) -> str:
    """A token shaped like what Supabase actually issues for a signed-in user."""
    now = int(time.time())
    payload = {
        "aud": SUPABASE_AUDIENCE,
        "exp": now + 3600,
        "iat": now,
        "iss": "https://wxghtvsywlonwlswqnur.supabase.co/auth/v1",
        "sub": PATIENT_ID,
        "email": "patient@example.com",
        "role": "authenticated",
        "aal": "aal1",
        "session_id": str(uuid.uuid4()),
        "is_anonymous": False,
    }
    payload.update(overrides)
    return jwt.encode(payload, secret, algorithm="HS256")


def _verify(token: str) -> str:
    return verify_token(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    )


def test_accepts_a_real_shaped_supabase_token():
    assert _verify(_supabase_token()) == PATIENT_ID


def test_rejects_token_signed_with_the_wrong_secret():
    with pytest.raises(HTTPException) as exc:
        _verify(_supabase_token(secret="a-different-secret-also-32-bytes-long-ok"))
    assert exc.value.status_code == 401


def test_rejects_expired_token():
    past = int(time.time()) - 60
    with pytest.raises(HTTPException) as exc:
        _verify(_supabase_token(exp=past, iat=past - 3600))
    assert exc.value.status_code == 401


def test_rejects_token_for_a_different_audience():
    with pytest.raises(HTTPException) as exc:
        _verify(_supabase_token(aud="some-other-service"))
    assert exc.value.status_code == 401


def test_rejects_garbage_token():
    with pytest.raises(HTTPException) as exc:
        _verify("not-a-jwt-at-all")
    assert exc.value.status_code == 401
