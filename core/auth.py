import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

# Supabase issues access tokens with aud="authenticated" for a signed-in user.
# PyJWT rejects any token carrying an `aud` claim unless it's told which
# audience to expect, so this must be passed explicitly — without it, every
# real token from the app is rejected as invalid.
SUPABASE_AUDIENCE = "authenticated"


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("SUPABASE_JWT_SECRET"),
            algorithms=["HS256"],
            audience=SUPABASE_AUDIENCE,
        )
        return payload.get("sub")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
