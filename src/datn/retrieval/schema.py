from __future__ import annotations

from ..vectordb.schema import VECTOR_IMAGE, VECTOR_TEXT

PRODUCTS_COLLECTION = "products"
REVIEWS_COLLECTION = "reviews"

# Jina CLIP v2 latent size shared by the image, text and review vectors.
VECTOR_SIZE = 1024

# Same conversion the marketplace importer (apps/backend/scripts/import_all_152k.py)
# used to turn the catalog's USD price into the VND price the shop displays.
USD_TO_VND = 25_000

# `products` payload. Point id == Postgres products.id == items.parquet row + 1.
P_PRODUCT_ID = "product_id"
P_ITEM_ID = "item_id"  # ASIN / Postgres sku -- the id the recommender checkpoint uses
P_TITLE = "title"
P_BRAND = "brand"
P_CATEGORY = "category"  # original Amazon category
P_CATEGORY_ID = "category_id"  # shop category (Postgres)
P_CATEGORY_SLUG = "category_slug"
P_PRICE = "price"  # VND, as displayed by the shop
P_PRICE_ESTIMATED = "price_estimated"  # True when the source had no real price
P_IMAGE_URL = "image_url"
P_DESCRIPTION = "description"
P_FEATURES = "features"
P_REVIEW_COUNT = "review_count"
P_AVG_RATING = "avg_rating"
P_HAS_IMAGE = "has_image"
P_HAS_TEXT = "has_text"
P_IMAGE_FALLBACK = "is_image_fallback"

# `reviews` payload.
R_PRODUCT_ID = "product_id"
R_ITEM_ID = "item_id"
R_RATING = "rating"
R_HELPFUL = "helpful_vote"
R_VERIFIED = "verified_purchase"
R_TITLE = "title"
R_TEXT = "text"
