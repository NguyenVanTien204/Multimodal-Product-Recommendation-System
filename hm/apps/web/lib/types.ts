export interface User {
  id: number;
  email: string;
  full_name: string;
  is_admin: boolean;
}

export interface Category {
  id: number;
  name: string;
  slug: string;
}

export interface Product {
  id: number;
  sku: string;
  name: string;
  description: string;
  price: number;
  stock_quantity: number;
  image_url: string | null;
  category_id: number | null;
  is_active: boolean;
  audience?: string | null;
  attributes?: {
    product_type?: string;
    product_group?: string;
    colour?: string;
    appearance?: string;
    section?: string;
    department?: string;
    sold_28d?: number;
  } | null;
}

export interface PaginatedProducts {
  items: Product[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CartItem {
  product: Product;
  quantity: number;
}

export interface Cart {
  items: CartItem[];
  total_amount: number;
}

export interface OrderItem {
  product_id: number;
  product_name: string;
  unit_price: number;
  quantity: number;
}

export interface Order {
  id: number;
  status: "PENDING" | "PAID" | "PROCESSING" | "SHIPPED" | "CANCELLED";
  shipping_address: string;
  total_amount: number;
  created_at: string;
  items: OrderItem[];
}

export interface SimilarProduct {
  product_id: number;
  score: number;
}

export interface AiRecommendationItem {
  product: Product;
  score: number;
}

export interface ChatEvidence {
  tag: string | null;
  review_id: number;
  rating: number;
  helpful_vote: number;
  title: string | null;
  text: string;
  is_mock?: boolean;
}

export interface ChatProduct {
  product: Product;
  score: number;
  brand?: string | null;
  price_estimated: boolean;
  avg_rating: number | null;
  review_count: number;
  reviews_mock?: boolean;
  audience?: string | null;
  colour?: string | null;
  product_type?: string | null;
  reasons: string[];
  evidence: ChatEvidence[];
}

export interface ChatApiResponse {
  session_id: string;
  reply: string;
  action: string;
  lang: string;
  products: ChatProduct[];
  filter_chips: string[];
  suggestions: string[];
  citations: string[];
  meta: Record<string, unknown>;
  warnings: string[];
}

export interface ChatRequestPayload {
  message: string;
  session_id?: string | null;
  image_base64?: string | null;
  action?: {
    type: "explain" | "compare" | "similar" | "recommend" | "feedback" | "forget";
    product_id?: number;
    product_ids?: number[];
    kind?: "like" | "dislike";
  } | null;
}

/** What the assistant remembers about the shopper (GET /me/preferences). */
export interface PreferencesSummary {
  liked: Product[];
  disliked: Product[];
  event_count: number;
}

/** `meta.preferences` of a chat answer: how the remembered taste was used for this answer. */
export interface PreferenceUsage {
  liked_or_engaged: number;
  disliked: number;
  history_used: number;
  excluded: number;
  penalised: number;
}

export interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  content: string;
  timestamp: string;
  imagePreview?: string;
  chatProducts?: ChatProduct[];
  filterChips?: string[];
  suggestions?: string[];
  meta?: Record<string, unknown>;
  warnings?: string[];
  isError?: boolean;
}

export interface SystemHealth {
  api: boolean;
  postgres: boolean;
  qdrant: boolean;
}
