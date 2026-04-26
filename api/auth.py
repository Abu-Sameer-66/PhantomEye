import os
import uuid
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Config
SECRET_KEY = os.getenv("SECRET_KEY", "phantomeye-secret-key-change-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()

# In-memory API key store — replace with PostgreSQL in production
API_KEYS_DB: dict[str, dict] = {}

# Pre-registered demo keys for testing
DEMO_KEYS = {
    "PE-DEMO-KEY-2026-FREE": {
        "client": "demo_user",
        "tier": "free",
        "rate_limit": 100,
        "calls_today": 0,
        "created_at": datetime.utcnow().isoformat(),
        "active": True,
    },
    "PE-PROD-KEY-2026-PRO": {
        "client": "pro_user",
        "tier": "pro",
        "rate_limit": 10000,
        "calls_today": 0,
        "created_at": datetime.utcnow().isoformat(),
        "active": True,
    }
}
API_KEYS_DB.update(DEMO_KEYS)


def generate_api_key(client_name: str, tier: str = "free") -> str:
    """Generate a unique PhantomEye API key."""
    unique = str(uuid.uuid4()).replace("-", "").upper()[:16]
    key = f"PE-{tier.upper()}-{unique}"
    API_KEYS_DB[key] = {
        "client": client_name,
        "tier": tier,
        "rate_limit": 100 if tier == "free" else 10000,
        "calls_today": 0,
        "created_at": datetime.utcnow().isoformat(),
        "active": True,
    }
    return key


def validate_api_key(api_key: str) -> dict:
    """Validate API key and check rate limit."""
    if api_key not in API_KEYS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    key_data = API_KEYS_DB[api_key]
    if not key_data["active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is disabled"
        )
    if key_data["calls_today"] >= key_data["rate_limit"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded — {key_data['rate_limit']} calls/day for {key_data['tier']} tier"
        )
    API_KEYS_DB[api_key]["calls_today"] += 1
    return key_data


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> dict:
    """Verify JWT token from Authorization header."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        client = payload.get("sub")
        if client is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_api_key_stats() -> dict:
    """Return stats about all registered API keys."""
    return {
        "total_keys": len(API_KEYS_DB),
        "active_keys": sum(1 for k in API_KEYS_DB.values() if k["active"]),
        "keys": [
            {
                "key": k[:12] + "****",
                "client": v["client"],
                "tier": v["tier"],
                "calls_today": v["calls_today"],
                "rate_limit": v["rate_limit"],
                "active": v["active"],
            }
            for k, v in API_KEYS_DB.items()
        ]
    }