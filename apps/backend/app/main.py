from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth.router import router as auth_router
from .cart.router import router as cart_router
from .catalog.router import router as catalog_router
from .core.config import settings
from .core.database import Base, engine
from .orders.router import router as order_router
from .recommendations.router import router as recommendation_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Mini Marketplace API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[item.strip() for item in settings.cors_origins.split(",") if item.strip()], allow_credentials=True, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type"])
app.include_router(auth_router)
app.include_router(catalog_router)
app.include_router(cart_router)
app.include_router(order_router)
app.include_router(recommendation_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "mini-marketplace"}
