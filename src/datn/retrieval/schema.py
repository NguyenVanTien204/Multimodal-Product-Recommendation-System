from __future__ import annotations

import os

from ..vectordb.schema import VECTOR_IMAGE, VECTOR_TEXT

# The catalog is selectable per deployment: Amazon (defaults: 1024-d vectors) or H&M
# (DATN_PRODUCTS_COLLECTION=hm_products DATN_REVIEWS_COLLECTION=hm_reviews DATN_VECTOR_SIZE=512).
PRODUCTS_COLLECTION = os.environ.get("DATN_PRODUCTS_COLLECTION", "products")
REVIEWS_COLLECTION = os.environ.get("DATN_REVIEWS_COLLECTION", "reviews")

# Jina CLIP v2 latent size shared by the image, text and review vectors (H&M embeddings are Matryoshka-cut to 512).
VECTOR_SIZE = int(os.environ.get("DATN_VECTOR_SIZE", "1024"))

# Payload field the "popular products" fallback sorts by (needs an integer payload index). H&M: sold_28d.
POPULARITY_KEY = os.environ.get("DATN_POPULARITY_KEY", "review_count")

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

# H&M-only `products` payload fields (the catalog has no brand: everything is H&M).
P_AUDIENCE = "audience"  # women | men | divided | kids | baby | other (derived from section_name)
P_PRODUCT_TYPE = "product_type"
P_PRODUCT_GROUP = "product_group"
P_COLOUR = "colour"
P_APPEARANCE = "appearance"
P_SECTION = "section"
P_DEPARTMENT = "department"
P_REVIEWS_MOCK = "reviews_are_mock"  # True: the rating/reviews of this product are borrowed demo data
P_SOLD_28D = "sold_28d"  # units sold in the 28 days before the shop clock (best-seller signal)

# `reviews` payload.
R_PRODUCT_ID = "product_id"
R_ITEM_ID = "item_id"
R_RATING = "rating"
R_HELPFUL = "helpful_vote"
R_VERIFIED = "verified_purchase"
R_TITLE = "title"
R_TEXT = "text"
R_IS_MOCK = "is_mock"  # True: review borrowed from another catalog (H&M has no review text)
