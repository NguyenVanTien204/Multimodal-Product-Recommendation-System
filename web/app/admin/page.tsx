"use client";

import React, { useState, useEffect } from "react";
import { 
  ShieldCheck, 
  PlusCircle, 
  Package, 
  Layers, 
  TrendingUp,
  RefreshCw,
  Cpu
} from "lucide-react";
import { useAuth, useToast } from "@/lib/context";
import { createProduct, getCategories, getOrders, getProducts, getSystemHealth, updateOrderStatus, formatVND } from "@/lib/api";
import { Category, Order, Product, SystemHealth } from "@/lib/types";

export default function AdminPage() {
  const { user, login } = useAuth();
  const { showToast } = useToast();

  const [categories, setCategories] = useState<Category[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [totalProducts, setTotalProducts] = useState(0);

  const [orders, setOrders] = useState<Order[]>([]);
  const [health, setHealth] = useState<SystemHealth>({ api: true, postgres: true, qdrant: true });
  const [loading, setLoading] = useState(true);

  // New product form
  const [sku, setSku] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [price, setPrice] = useState(250000);
  const [stock, setStock] = useState(50);
  const [imageUrl, setImageUrl] = useState("");
  const [categoryId, setCategoryId] = useState<number | undefined>(undefined);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const loadAdminData = async () => {
    try {
      setLoading(true);
      const [cats, prods, ords, h] = await Promise.all([
        getCategories(),
        getProducts(),
        user?.is_admin ? getOrders() : Promise.resolve([]),
        getSystemHealth(),
      ]);
      setCategories(cats);
      setProducts(prods.items);
      setTotalProducts(prods.total);
      setOrders(ords);
      setHealth(h);
      if (cats.length > 0 && !categoryId) {
        setCategoryId(cats[0].id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAdminData();
  }, [user]);

  const handleCreateProduct = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sku.trim() || !name.trim()) {
      showToast("Vui lòng điền đủ SKU và Tên sản phẩm", "error");
      return;
    }

    try {
      setIsSubmitting(true);
      await createProduct({
        sku: sku.trim(),
        name: name.trim(),
        description: description.trim(),
        price: Number(price),
        stock_quantity: Number(stock),
        image_url: imageUrl.trim() || undefined,
        category_id: categoryId,
      });

      showToast("Thêm sản phẩm thành công vào PostgreSQL!", "success");
      setSku("");
      setName("");
      setDescription("");
      setImageUrl("");
      await loadAdminData();
    } catch (err: any) {
      showToast(err.message || "Lỗi tạo sản phẩm", "error");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStatusChange = async (orderId: number, newStatus: string) => {
    try {
      await updateOrderStatus(orderId, newStatus);
      showToast(`Đã chuyển đơn #${orderId} sang ${newStatus}`, "success");
      await loadAdminData();
    } catch (err: any) {
      showToast(err.message || "Lỗi cập nhật trạng thái đơn", "error");
    }
  };

  if (!user || !user.is_admin) {
    return (
      <div className="max-w-md mx-auto px-4 py-24 text-center">
        <div className="w-16 h-16 rounded-3xl bg-amber-50 border border-amber-200 text-amber-700 flex items-center justify-center mx-auto mb-4 shadow-xs">
          <ShieldCheck className="w-8 h-8" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-2">Quyền Quản Trị Viên</h2>
        <p className="text-xs sm:text-sm text-slate-500 mb-6">
          Trang này dành riêng cho tài khoản Quản trị viên của hệ thống ShopSense.
        </p>
        <button
          onClick={() => login("admin@shopsense.vn", "adminpassword123")}
          className="w-full py-3.5 rounded-xl font-bold text-sm bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-all"
        >
          Đăng Nhập Tài Khoản Admin Mẫu
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* HEADER */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-6 border-b border-slate-200">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-700 mb-1">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>Khu Vực Quản Trị Hệ Thống</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
            Bảng Điều Khiển ShopSense
          </h1>
        </div>

        <button
          onClick={loadAdminData}
          className="p-2.5 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-2 shadow-2xs transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin text-emerald-600" : ""}`} />
          <span>Đồng bộ dữ liệu</span>
        </button>
      </div>

      {/* METRIC CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="p-5 rounded-2xl bg-white border border-slate-200 space-y-2 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Sản Phẩm Trong DB</span>
            <Package className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-slate-900">
            {totalProducts.toLocaleString("vi-VN")}
          </div>
          <span className="text-[11px] text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded">PostgreSQL Catalog</span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 space-y-2 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Danh Mục Ngành Hàng</span>
            <Layers className="w-4 h-4 text-teal-600" />
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-slate-900">
            {categories.length}
          </div>
          <span className="text-[11px] text-teal-700 font-semibold bg-teal-50 px-2 py-0.5 rounded">Phân loại thời trang</span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 space-y-2 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Đơn Hàng Ghi Nhận</span>
            <TrendingUp className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-slate-900">
            {orders.length}
          </div>
          <span className="text-[11px] text-amber-700 font-semibold bg-amber-50 px-2 py-0.5 rounded">Row-locked transactions</span>
        </div>

        <div className="p-5 rounded-2xl bg-white border border-slate-200 space-y-2 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Vector DB & Qdrant</span>
            <Cpu className="w-4 h-4 text-purple-600" />
          </div>
          <div className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span>{health.qdrant ? "Trực Tuyến" : "Chưa kết nối"}</span>
          </div>
          <span className="text-[11px] text-purple-700 font-semibold bg-purple-50 px-2 py-0.5 rounded">152k product_embeddings</span>
        </div>
      </div>

      {/* TWO COLUMNS: ADD PRODUCT & ORDER MANAGEMENT */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* ADD PRODUCT FORM */}
        <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 space-y-5 shadow-xs">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200">
              <PlusCircle className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-900">Thêm Sản Phẩm Mới Vào DB</h3>
              <p className="text-xs text-slate-500">
                Thêm trực tiếp vào bảng products của PostgreSQL thông qua API FastAPI
              </p>
            </div>
          </div>

          <form onSubmit={handleCreateProduct} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Mã SKU</label>
                <input
                  type="text"
                  required
                  value={sku}
                  onChange={(e) => setSku(e.target.value)}
                  placeholder="Vd: MEN-JEANS-004"
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-900 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 font-mono transition-all"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Danh Mục</label>
                <select
                  value={categoryId || ""}
                  onChange={(e) => setCategoryId(Number(e.target.value))}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-900 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 cursor-pointer transition-all"
                >
                  {categories.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">Tên Sản Phẩm</label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Tên đầy đủ của sản phẩm..."
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-900 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Giá Bán (VNĐ)</label>
                <input
                  type="number"
                  required
                  min={1000}
                  step={1000}
                  value={price}
                  onChange={(e) => setPrice(Number(e.target.value))}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-900 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Số Lượng Tồn Kho</label>
                <input
                  type="number"
                  required
                  min={0}
                  value={stock}
                  onChange={(e) => setStock(Number(e.target.value))}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-900 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
                />
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">URL Ảnh Sản Phẩm</label>
              <input
                type="url"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="https://images.unsplash.com/..."
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-900 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">Mô Tả Sản Phẩm</label>
              <textarea
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Chất liệu, form dáng, tính năng nổi bật..."
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-900 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 resize-none transition-all"
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full py-3.5 rounded-xl font-bold text-xs sm:text-sm bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-all flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <span>Đang ghi dữ liệu vào PostgreSQL...</span>
              ) : (
                <>
                  <PlusCircle className="w-4 h-4" />
                  <span>Xác Nhận Thêm Sản Phẩm</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* ORDER MANAGEMENT */}
        <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 space-y-5 shadow-xs">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-blue-50 text-blue-700 border border-blue-200">
              <Package className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-900">Quản Lý Trạng Thái Đơn Hàng</h3>
              <p className="text-xs text-slate-500">
                Cập nhật tiến trình đơn hàng (PENDING → PAID → SHIPPED)
              </p>
            </div>
          </div>

          <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
            {orders.length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-400">
                Chưa có đơn hàng nào cần xử lý.
              </div>
            ) : (
              orders.map((o) => (
                <div
                  key={o.id}
                  className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-2 text-xs shadow-2xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-slate-900">
                      Đơn #{o.id}
                    </span>
                    <span className="font-mono text-emerald-700 font-bold">
                      {formatVND(Number(o.total_amount))}
                    </span>
                  </div>

                  <div className="text-slate-500 text-[11px] truncate">
                    Giao tới: {o.shipping_address}
                  </div>

                  <div className="flex items-center justify-between pt-2 border-t border-slate-200">
                    <span className="text-[11px] text-slate-600 font-medium">
                      Trạng thái hiện tại: <b className="text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded">{o.status}</b>
                    </span>

                    <select
                      value={o.status}
                      onChange={(e) => handleStatusChange(o.id, e.target.value)}
                      className="px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-[11px] text-slate-800 focus:outline-none cursor-pointer font-medium shadow-2xs"
                    >
                      <option value="PENDING">PENDING</option>
                      <option value="PAID">PAID</option>
                      <option value="PROCESSING">PROCESSING</option>
                      <option value="SHIPPED">SHIPPED</option>
                      <option value="CANCELLED">CANCELLED</option>
                    </select>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
