"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { 
  X, 
  MapPin, 
  CheckCircle2, 
  ShieldCheck, 
  ArrowRight,
  PackageCheck
} from "lucide-react";
import { useCart, useToast } from "@/lib/context";
import { checkoutOrder, formatVND } from "@/lib/api";
import { Order } from "@/lib/types";

interface CheckoutModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export function CheckoutModal({ isOpen, onClose, onSuccess }: CheckoutModalProps) {
  const router = useRouter();
  const { cart, refreshCart } = useCart();
  const { showToast } = useToast();

  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [createdOrder, setCreatedOrder] = useState<Order | null>(null);

  if (!isOpen) return null;

  const total = cart?.total_amount || 0;
  const items = cart?.items || [];

  const handleCheckout = async (e: React.FormEvent) => {
    e.preventDefault();
    if (address.trim().length < 10) {
      showToast("Vui lòng nhập địa chỉ nhận hàng chi tiết (tối thiểu 10 ký tự)", "error");
      return;
    }

    try {
      setLoading(true);
      const order = await checkoutOrder(address.trim());
      setCreatedOrder(order);
      await refreshCart();
      showToast("Đặt hàng thành công! Đơn hàng đã được lưu vào cơ sở dữ liệu.", "success");
    } catch (err: any) {
      showToast(err.message || "Thanh toán thất bại", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleFinish = () => {
    onSuccess();
    router.push("/orders");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-fade-in">
      <div 
        className="relative w-full max-w-lg bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 rounded-full bg-slate-100 text-slate-500 hover:text-slate-900 hover:bg-slate-200 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {createdOrder ? (
          /* SUCCESS SCREEN */
          <div className="text-center py-6 space-y-5 animate-slide-up">
            <div className="w-16 h-16 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200 flex items-center justify-center mx-auto shadow-sm">
              <CheckCircle2 className="w-10 h-10" />
            </div>

            <div>
              <h3 className="text-2xl font-extrabold text-slate-900">Đặt Hàng Thành Công!</h3>
              <p className="text-xs text-slate-500 mt-1">
                Giao dịch ACID đã trừ tồn kho sản phẩm trong PostgreSQL
              </p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 text-left text-xs space-y-2">
              <div className="flex justify-between">
                <span className="text-slate-500">Mã đơn hàng:</span>
                <span className="font-mono font-bold text-emerald-700">#{createdOrder.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Trạng thái:</span>
                <span className="font-bold text-amber-600 bg-amber-50 px-2 py-0.5 rounded">{createdOrder.status}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Tổng thanh toán:</span>
                <span className="font-bold text-slate-900">{formatVND(Number(createdOrder.total_amount))}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Địa chỉ giao:</span>
                <span className="font-medium text-slate-700 text-right truncate max-w-[220px]">
                  {createdOrder.shipping_address}
                </span>
              </div>
            </div>

            <button
              onClick={handleFinish}
              className="w-full py-3.5 rounded-xl font-bold text-sm bg-slate-900 hover:bg-slate-800 text-white flex items-center justify-center gap-2 shadow-xs transition-all"
            >
              <PackageCheck className="w-5 h-5" />
              <span>Xem Chi Tiết Đơn Hàng</span>
            </button>
          </div>
        ) : (
          /* CHECKOUT FORM */
          <form onSubmit={handleCheckout} className="space-y-5">
            <div>
              <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-emerald-600" />
                <span>Xác Nhận & Thanh Toán</span>
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Kiểm tra thông tin giao hàng và đơn hàng trước khi tạo giao dịch.
              </p>
            </div>

            {/* ORDER SUMMARY PREVIEW */}
            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-2.5 max-h-48 overflow-y-auto">
              {items.map(({ product, quantity }) => (
                <div key={product.id} className="flex justify-between items-center text-xs">
                  <span className="text-slate-700 font-medium truncate max-w-[240px]">
                    {quantity}x {product.name}
                  </span>
                  <span className="font-mono text-emerald-700 font-bold">
                    {formatVND(Number(product.price) * quantity)}
                  </span>
                </div>
              ))}
              <div className="pt-2 border-t border-slate-200 flex justify-between items-center font-bold text-sm">
                <span className="text-slate-700">Tổng thanh toán:</span>
                <span className="text-emerald-600 text-base">{formatVND(Number(total))}</span>
              </div>
            </div>

            {/* ADDRESS INPUT */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                <MapPin className="w-3.5 h-3.5 text-emerald-600" />
                <span>Địa chỉ nhận hàng (bắt buộc)</span>
              </label>
              <textarea
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder="Ví dụ: Số 123 Đường Cầu Giấy, Phường Quan Hoa, Quận Cầu Giấy, Hà Nội"
                rows={3}
                required
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all resize-none"
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setAddress("Tòa nhà FPT Tower, Số 10 Phạm Văn Bạch, Cầu Giấy, Hà Nội")}
                  className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-[11px] text-slate-600 font-medium border border-slate-200 transition-colors"
                >
                  Gợi ý: Cầu Giấy, Hà Nội
                </button>
                <button
                  type="button"
                  onClick={() => setAddress("Ký túc xá Đại học Quốc Gia, Khu phố 6, TP. Thủ Đức, TP. Hồ Chí Minh")}
                  className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-[11px] text-slate-600 font-medium border border-slate-200 transition-colors"
                >
                  Gợi ý: TP. Thủ Đức, TP.HCM
                </button>
              </div>
            </div>

            {/* SUBMIT BUTTON */}
            <button
              type="submit"
              disabled={loading || items.length === 0}
              className="w-full py-3.5 rounded-xl font-bold text-sm bg-slate-900 hover:bg-slate-800 text-white shadow-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {loading ? (
                <span>Đang xử lý khóa tồn kho & tạo đơn...</span>
              ) : (
                <>
                  <span>Hoàn Tất Đặt Hàng ({formatVND(Number(total))})</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
