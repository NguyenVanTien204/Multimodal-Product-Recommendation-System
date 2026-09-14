from __future__ import annotations

# Named vectors inside the single `products` collection. Image and text embeddings
# share the same Jina CLIP v2 latent space (D=1024), so they live as two named
# vectors on one point rather than two separate collections: a point can carry
# only the "image" vector today and gain "text" later via a partial upsert,
# without re-writing payload or picking a new point id.
VECTOR_IMAGE = "image"
VECTOR_TEXT = "text"

# Payload field names, kept as constants so the importer and any query code
# can't drift apart on typos.
FIELD_ITEM_ID = "item_id"
FIELD_TITLE = "title"
FIELD_BRAND = "brand"
FIELD_CATEGORY = "category"
FIELD_PRICE = "price"
FIELD_IMAGE_URL = "image_url"
FIELD_HAS_IMAGE = "has_image"
FIELD_HAS_TEXT = "has_text"
FIELD_IS_IMAGE_FALLBACK = "is_image_fallback"

# (field, schema type) pairs that get an explicit Qdrant payload index so filtered
# search (brand/category facets, price ranges, excluding fallback-image items) stays
# fast instead of falling back to a full payload scan.
KEYWORD_INDEX_FIELDS = (FIELD_ITEM_ID, FIELD_BRAND, FIELD_CATEGORY)
FLOAT_INDEX_FIELDS = (FIELD_PRICE,)
BOOL_INDEX_FIELDS = (FIELD_HAS_IMAGE, FIELD_HAS_TEXT, FIELD_IS_IMAGE_FALLBACK)
