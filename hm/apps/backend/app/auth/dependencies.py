from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import decode_access_token
from .models import User

bearer = HTTPBearer()
optional_bearer = HTTPBearer(auto_error=False)


def current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)) -> User:
    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator permission required")
    return user


def optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer), db: Session = Depends(get_db)
) -> User | None:
    """Like `current_user`, but returns None instead of raising when no (or an
    invalid) token is present -- for endpoints that personalize when logged in
    but still work for anonymous browsing."""
    if credentials is None:
        return None
    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError:
        return None
    return db.get(User, user_id)
