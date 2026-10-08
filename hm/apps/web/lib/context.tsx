"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { Cart, Product, User } from "./types";
import {
  addToCart as apiAddToCart,
  changePassword as apiChangePassword,
  clearStoredToken,
  getCart as apiGetCart,
  getCurrentUser,
  getStoredToken,
  loginUser,
  registerUser,
  removeCartItem as apiRemoveCartItem,
  updateProfile as apiUpdateProfile,
} from "./api";

// ---------------- TOAST ----------------
interface Toast {
  id: string;
  type: "success" | "error" | "info";
  message: string;
}

interface ToastContextType {
  showToast: (message: string, type?: "success" | "error" | "info") => void;
}

const ToastContext = createContext<ToastContextType>({ showToast: () => {} });

// ---------------- AUTH ----------------
interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, pass: string) => Promise<void>;
  register: (email: string, pass: string, name: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  updateProfile: (fullName: string) => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  login: async () => {},
  register: async () => {},
  logout: () => {},
  refreshUser: async () => {},
  updateProfile: async () => {},
  changePassword: async () => {},
});

// ---------------- CART ----------------
interface CartContextType {
  cart: Cart | null;
  loading: boolean;
  itemCount: number;
  isDrawerOpen: boolean;
  setIsDrawerOpen: (open: boolean) => void;
  addItem: (productId: number, quantity?: number) => Promise<void>;
  removeItem: (productId: number) => Promise<void>;
  refreshCart: () => Promise<void>;
}

const CartContext = createContext<CartContextType>({
  cart: null,
  loading: false,
  itemCount: 0,
  isDrawerOpen: false,
  setIsDrawerOpen: () => {},
  addItem: async () => {},
  removeItem: async () => {},
  refreshCart: async () => {},
});

// ---------------- COMPARE ----------------
interface CompareContextType {
  compareItems: Product[];
  addToCompare: (product: Product) => void;
  removeFromCompare: (productId: number) => void;
  clearCompare: () => void;
  isCompared: (productId: number) => boolean;
}

const CompareContext = createContext<CompareContextType>({
  compareItems: [],
  addToCompare: () => {},
  removeFromCompare: () => {},
  clearCompare: () => {},
  isCompared: () => false,
});

// ---------------- ROOT PROVIDER ----------------
export function AppProvider({ children }: { children: React.ReactNode }) {
  // Toasts
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((message: string, type: "success" | "error" | "info" = "info") => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }, []);

  // Auth State
  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    const token = getStoredToken();
    if (!token) {
      setUser(null);
      setAuthLoading(false);
      return;
    }
    try {
      const u = await getCurrentUser();
      setUser(u);
    } catch {
      clearStoredToken();
      setUser(null);
    } finally {
      setAuthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = async (email: string, pass: string) => {
    try {
      const result = await loginUser(email, pass);
      setUser(result.user);
      showToast(`Xin chào ${result.user.full_name}!`, "success");
      refreshCart();
    } catch (err: any) {
      showToast(err.message || "Đăng nhập thất bại", "error");
      throw err;
    }
  };

  const register = async (email: string, pass: string, name: string) => {
    try {
      await registerUser(email, pass, name);
      showToast("Đăng ký thành công! Hãy đăng nhập.", "success");
    } catch (err: any) {
      showToast(err.message || "Đăng ký thất bại", "error");
      throw err;
    }
  };

  const logout = () => {
    clearStoredToken();
    setUser(null);
    setCart(null);
    showToast("Đã đăng xuất tài khoản", "info");
  };

  const updateProfile = async (fullName: string) => {
    try {
      const updated = await apiUpdateProfile(fullName);
      setUser(updated);
      showToast("Đã cập nhật hồ sơ", "success");
    } catch (err: any) {
      showToast(err.message || "Không thể cập nhật hồ sơ", "error");
      throw err;
    }
  };

  const changePassword = async (currentPassword: string, newPassword: string) => {
    try {
      await apiChangePassword(currentPassword, newPassword);
      showToast("Đã đổi mật khẩu thành công", "success");
    } catch (err: any) {
      showToast(err.message || "Không thể đổi mật khẩu", "error");
      throw err;
    }
  };

  // Cart State
  const [cart, setCart] = useState<Cart | null>(null);
  const [cartLoading, setCartLoading] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const refreshCart = useCallback(async () => {
    if (!getStoredToken()) {
      setCart(null);
      return;
    }
    try {
      setCartLoading(true);
      const c = await apiGetCart();
      setCart(c);
    } catch {
      // not logged in or empty
    } finally {
      setCartLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user) {
      refreshCart();
    }
  }, [user, refreshCart]);

  const addItem = async (productId: number, quantity: number = 1) => {
    if (!user) {
      showToast("Vui lòng đăng nhập để thêm sản phẩm vào giỏ hàng!", "info");
      return;
    }
    try {
      const updated = await apiAddToCart(productId, quantity);
      setCart(updated);
      showToast("Đã thêm sản phẩm vào giỏ hàng!", "success");
    } catch (err: any) {
      showToast(err.message || "Không thể thêm vào giỏ hàng", "error");
    }
  };

  const removeItem = async (productId: number) => {
    try {
      await apiRemoveCartItem(productId);
      showToast("Đã xóa sản phẩm khỏi giỏ hàng", "info");
      await refreshCart();
    } catch (err: any) {
      showToast(err.message || "Lỗi khi xóa", "error");
    }
  };

  const itemCount = cart ? cart.items.reduce((sum, item) => sum + item.quantity, 0) : 0;

  // Compare State
  const [compareItems, setCompareItems] = useState<Product[]>([]);

  const addToCompare = (product: Product) => {
    if (compareItems.some((item) => item.id === product.id)) {
      setCompareItems((prev) => prev.filter((i) => i.id !== product.id));
      showToast(`Đã bỏ so sánh "${product.name}"`, "info");
      return;
    }
    if (compareItems.length >= 4) {
      showToast("Bạn chỉ có thể so sánh tối đa 4 sản phẩm cùng lúc", "error");
      return;
    }
    setCompareItems((prev) => [...prev, product]);
    showToast(`Đã thêm "${product.name}" vào danh sách so sánh`, "success");
  };

  const removeFromCompare = (productId: number) => {
    setCompareItems((prev) => prev.filter((i) => i.id !== productId));
  };

  const clearCompare = () => {
    setCompareItems([]);
  };

  const isCompared = (productId: number) => {
    return compareItems.some((item) => item.id === productId);
  };

  return (
    <ToastContext.Provider value={{ showToast }}>
      <AuthContext.Provider
        value={{ user, loading: authLoading, login, register, logout, refreshUser, updateProfile, changePassword }}
      >
        <CartContext.Provider
          value={{
            cart,
            loading: cartLoading,
            itemCount,
            isDrawerOpen,
            setIsDrawerOpen,
            addItem,
            removeItem,
            refreshCart,
          }}
        >
          <CompareContext.Provider
            value={{
              compareItems,
              addToCompare,
              removeFromCompare,
              clearCompare,
              isCompared,
            }}
          >
            {children}

            {/* TOAST CONTAINER */}
            <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
              {toasts.map((toast) => (
                <div
                  key={toast.id}
                  className={`pointer-events-auto px-4 py-3 rounded-xl shadow-xl text-sm font-medium transition-all duration-300 transform translate-y-0 flex items-center gap-3 backdrop-blur-md border ${
                    toast.type === "success"
                      ? "bg-emerald-950/90 border-emerald-600/50 text-emerald-100"
                      : toast.type === "error"
                      ? "bg-rose-950/90 border-rose-600/50 text-rose-100"
                      : "bg-slate-900/90 border-slate-700 text-slate-100"
                  }`}
                >
                  <span className="text-base">
                    {toast.type === "success" ? "✓" : toast.type === "error" ? "⚠" : "ℹ"}
                  </span>
                  <span>{toast.message}</span>
                </div>
              ))}
            </div>
          </CompareContext.Provider>
        </CartContext.Provider>
      </AuthContext.Provider>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);
export const useAuth = () => useContext(AuthContext);
export const useCart = () => useContext(CartContext);
export const useCompare = () => useContext(CompareContext);
