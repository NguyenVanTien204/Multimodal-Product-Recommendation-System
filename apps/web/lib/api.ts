import { AiRecommendationItem, Cart, Category, Order, PaginatedProducts, Product, SimilarProduct, SystemHealth, User } from "./types";
import { DEMO_CATEGORIES, DEMO_PRODUCTS } from "./demo-fixtures";


const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TOKEN_KEY = "shopsense_jwt_token";

export const getStoredToken = (): string | null => {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
};

export const setStoredToken = (token: string): void => {
  if (typeof window !== "undefined") {
    localStorage.setItem(TOKEN_KEY, token);
  }
};

export const clearStoredToken = (): void => {
  if (typeof window !== "undefined") {
    localStorage.removeItem(TOKEN_KEY);
  }
};

async function fetchWithAuth(endpoint: string, options: RequestInit = {}): Promise<Response> {
  const token = getStoredToken();
  const headers = new Headers(options.headers || {});

  if (!headers.has("Content-Type") && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const url = `${API_BASE}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
  return fetch(url, {
    ...options,
    headers,
  });
}

// ----------------- HEALTH & SYSTEM -----------------
export async function getSystemHealth(): Promise<SystemHealth> {
  try {
    const healthRes = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const healthData = await healthRes.json();
    const isApiOk = healthRes.ok && healthData.status === "ok";

    let isQdrantOk = false;
    try {
      const qRes = await fetch(`${API_BASE}/recommendations/health`, { signal: AbortSignal.timeout(3000) });
      const qData = await qRes.json();
      isQdrantOk = qRes.ok && !!qData.qdrant_available;
    } catch {
      isQdrantOk = false;
    }

    return {
      api: isApiOk,
      postgres: isApiOk,
      qdrant: isQdrantOk,
    };
  } catch {
    return {
      api: false,
      postgres: false,
      qdrant: false,
    };
  }
}

// ----------------- AUTHENTICATION -----------------
export async function loginUser(email: string, password: string): Promise<{ access_token: string; user: User }> {
  const res = await fetchWithAuth("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Đăng nhập thất bại" }));
    throw new Error(err.detail || "Email hoặc mật khẩu không chính xác");
  }

  const data = await res.json();
  setStoredToken(data.access_token);

  // fetch user info
  const user = await getCurrentUser();
  return { access_token: data.access_token, user };
}

export async function registerUser(email: string, password: string, fullName: string): Promise<User> {
  const res = await fetchWithAuth("/auth/register", {
    method: "POST",
    body: JSON.stringify({
      email,
      password,
      full_name: fullName,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Đăng ký thất bại" }));
    throw new Error(err.detail || "Không thể đăng ký tài khoản");
  }

  return res.json();
}

export async function getCurrentUser(): Promise<User> {
  const res = await fetchWithAuth("/auth/me");
  if (!res.ok) {
    throw new Error("Không thể xác thực người dùng");
  }
  return res.json();
}

export async function updateProfile(fullName: string): Promise<User> {
  const res = await fetchWithAuth("/auth/me", {
    method: "PATCH",
    body: JSON.stringify({ full_name: fullName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Không thể cập nhật hồ sơ" }));
    throw new Error(err.detail || "Không thể cập nhật hồ sơ");
  }
  return res.json();
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  const res = await fetchWithAuth("/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Không thể đổi mật khẩu" }));
    throw new Error(err.detail || "Không thể đổi mật khẩu");
  }
}

// ----------------- CATALOG & PRODUCTS -----------------
export async function getCategories(): Promise<Category[]> {
  try {
    const res = await fetchWithAuth("/categories", { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to load categories");
    return await res.json();
  } catch (error) {
    console.warn("Backend categories unavailable, using demo fixtures:", error);
    return DEMO_CATEGORIES;
  }
}

export async function getProducts(params?: {
  q?: string;
  category_id?: number;
  page?: number;
  page_size?: number;
}): Promise<PaginatedProducts> {
  const page = params?.page || 1;
  const pageSize = params?.page_size || 24;

  try {
    const queryParts: string[] = [
      `page=${page}`,
      `page_size=${pageSize}`,
    ];
    if (params?.q) queryParts.push(`q=${encodeURIComponent(params.q)}`);
    if (params?.category_id) queryParts.push(`category_id=${params.category_id}`);

    const queryString = `?${queryParts.join("&")}`;
    const res = await fetchWithAuth(`/products${queryString}`, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to load products");
    const data = await res.json();

    // Check if response is paginated object
    if (data && typeof data === "object" && "items" in data && "total" in data) {
      return data as PaginatedProducts;
    }

    // Otherwise if it's an array
    const arr = Array.isArray(data) ? data : [];
    return {
      items: arr,
      total: arr.length,
      page: 1,
      page_size: arr.length,
      total_pages: 1,
    };
  } catch (error) {
    console.warn("Backend products unavailable, using demo fixtures:", error);
    let list = DEMO_PRODUCTS;
    if (params?.category_id) {
      list = list.filter((p) => p.category_id === params.category_id);
    }
    if (params?.q) {
      const qLower = params.q.toLowerCase();
      list = list.filter((p) => p.name.toLowerCase().includes(qLower) || p.description.toLowerCase().includes(qLower));
    }
    const total = list.length;
    const offset = (page - 1) * pageSize;
    const paginatedItems = list.slice(offset, offset + pageSize);
    return {
      items: paginatedItems,
      total: total,
      page: page,
      page_size: pageSize,
      total_pages: Math.max(1, Math.ceil(total / pageSize)),
    };
  }
}

export async function getAiRecommendationsForYou(limit: number = 8): Promise<AiRecommendationItem[]> {
  try {
    const res = await fetchWithAuth(`/recommendations/for-you?limit=${limit}`, { cache: "no-store" });
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}


export async function getProductDetail(productId: number): Promise<Product> {
  try {
    const res = await fetchWithAuth(`/products/${productId}`, { cache: "no-store" });
    if (!res.ok) throw new Error("Product not found");
    return await res.json();
  } catch (error) {
    const found = DEMO_PRODUCTS.find((p) => p.id === productId);
    if (found) return found;
    throw new Error("Sản phẩm không tồn tại");
  }
}

export async function createProduct(payload: {
  sku: string;
  name: string;
  description: string;
  price: number;
  stock_quantity: number;
  image_url?: string;
  category_id?: number;
}): Promise<Product> {
  const res = await fetchWithAuth("/products", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Không thể thêm sản phẩm" }));
    throw new Error(err.detail || "Lỗi tạo sản phẩm");
  }
  return res.json();
}

// ----------------- CART -----------------
export async function getCart(): Promise<Cart> {
  const res = await fetchWithAuth("/cart");
  if (!res.ok) {
    throw new Error("Chưa đăng nhập hoặc không thể tải giỏ hàng");
  }
  return res.json();
}

export async function addToCart(productId: number, quantity: number = 1): Promise<Cart> {
  const res = await fetchWithAuth("/cart/items", {
    method: "POST",
    body: JSON.stringify({ product_id: productId, quantity }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Không thể thêm vào giỏ" }));
    throw new Error(err.detail || "Lỗi khi thêm sản phẩm vào giỏ");
  }
  return res.json();
}

export async function removeCartItem(productId: number): Promise<void> {
  const res = await fetchWithAuth(`/cart/items/${productId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error("Không thể xóa sản phẩm khỏi giỏ");
  }
}

// ----------------- ORDERS -----------------
export async function checkoutOrder(shippingAddress: string): Promise<Order> {
  const res = await fetchWithAuth("/orders/checkout", {
    method: "POST",
    body: JSON.stringify({ shipping_address: shippingAddress }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Không thể thanh toán" }));
    throw new Error(err.detail || "Lỗi trong quá trình thanh toán");
  }
  return res.json();
}

export async function getOrders(): Promise<Order[]> {
  const res = await fetchWithAuth("/orders");
  if (!res.ok) {
    throw new Error("Không thể tải danh sách đơn hàng");
  }
  return res.json();
}

export async function updateOrderStatus(orderId: number, status: string): Promise<Order> {
  const res = await fetchWithAuth(`/orders/${orderId}/status`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
  if (!res.ok) {
    throw new Error("Không thể cập nhật trạng thái đơn hàng");
  }
  return res.json();
}

// ----------------- RECOMMENDATIONS (AI & QDRANT) -----------------
export async function getSimilarProducts(productId: number, limit: number = 6): Promise<SimilarProduct[]> {
  try {
    const res = await fetchWithAuth(`/recommendations/products/${productId}/similar?limit=${limit}`);
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}

export function formatVND(amount: number): string {
  return new Intl.NumberFormat("vi-VN", {
    style: "currency",
    currency: "VND",
  }).format(amount);
}
