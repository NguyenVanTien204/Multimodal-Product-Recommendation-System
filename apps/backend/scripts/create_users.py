"""Create the admin and demo accounts (idempotent). Run inside the backend container:

    docker exec datn-backend python scripts/create_users.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from app.auth.models import User
from app.cart import models as _cart  # noqa: F401  (every model must be imported so relationship('Cart') resolves)
from app.catalog import models as _catalog  # noqa: F401
from app.orders import models as _orders  # noqa: F401
from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password

ACCOUNTS = [
    ("admin@shopsense.vn", "Quản Trị Viên ShopSense", "adminpassword123", True),
    ("demo@shopsense.vn", "Nguyễn Văn Demo", "demopassword123", False),
]


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        for email, name, password, is_admin in ACCOUNTS:
            if db.scalar(select(User).where(User.email == email)) is None:
                db.add(User(email=email, full_name=name, password_hash=hash_password(password), is_admin=is_admin))
                print("created", email)
            else:
                print("exists ", email)
        db.commit()


if __name__ == "__main__":
    main()
