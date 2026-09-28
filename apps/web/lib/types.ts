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

export interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  content: string;
  timestamp: string;
  products?: Product[];
  actionSuggestion?: string;
}

export interface SystemHealth {
  api: boolean;
  postgres: boolean;
  qdrant: boolean;
}
