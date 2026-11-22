"""
Exercises verify_token directly against realistically-shaped tokens.

The endpoint tests override the auth dependency so they can focus on
behavior, which means verify_token itself goes unexercised there. That gap
let two real bugs ship, both found only against a genuine token from the
app:

1. Hand-rolled tokens carried only `sub` and `exp`; real ones also carry
   `aud`, which PyJWT rejects unless told to expect it.
2. Hand-rolled tokens were HS256; real Supabase tokens are signed ES256
   against a published JWKS key, which HS256 verification cannot check at
   all.

So these tests sign with a real ES256 key pair, the way Supabase does.
"""

import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from core.auth import SUPABASE_AUDIENCE, verify_token

# At least 32 bytes — shorter HMAC keys make PyJWT warn on every call.
SECRET = "test-secret-not-the-real-one-padded-to-32-bytes-plus"
PATIENT_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# A throwaway P-256 key pair, matching the curve Supabase signs with.
_EC_PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    monkeypatch.setenv("ENVIRONMENT", "development")


def _es256_token(**overrides) -> str:
    """A token shaped and signed the way a real Supabase one is."""
    now = int(time.time())
    payload = {
        "aud": SUPABASE_AUDIENCE,
        "exp": now + 3600,
        "iat": now,
        "iss": "https://wxghtvsywlonwlswqnur.supabase.co/auth/v1",
        "sub": PATIENT_ID,
        "role": "authenticated",
    }
    payload.update(overrides)
    return jwt.encode(payload, _EC_PRIVATE_KEY, algorithm="ES256")


@pytest.fixture
def _accept_test_key(monkeypatch):
    """Point the JWKS lookup at our throwaway public key instead of the
    network, so the ES256 path is exercised for real without a live call."""

    class _Key:
        key = _EC_PRIVATE_KEY.public_key()

    class _Client:
        def get_signing_key_from_jwt(self, token):
            return _Key()

    monkeypatch.setattr("core.auth._get_jwks_client", lambda: _Client())


def test_accepts_a_real_es256_supabase_token(_accept_test_key):
    assert _verify(_es256_token()) == PATIENT_ID


def test_rejects_es256_token_for_the_wrong_audience(_accept_test_key):
    with pytest.raises(HTTPException) as exc:
        _verify(_es256_token(aud="some-other-service"))
    assert exc.value.status_code == 401


def test_rejects_expired_es256_token(_accept_test_key):
    past = int(time.time()) - 60
    with pytest.raises(HTTPException) as exc:
        _verify(_es256_token(exp=past, iat=past - 3600))
    assert exc.value.status_code == 401


def test_shared_secret_tokens_are_refused_outside_development(monkeypatch):
    # HS256 is a local-testing convenience only; a deployed environment must
    # accept nothing but genuinely Supabase-signed tokens.
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(HTTPException) as exc:
        _verify(_supabase_token())
    assert exc.value.status_code == 401


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
