"use client";

import React, { useState } from "react";
import Image from "next/image";
import { X, ShoppingBag, Trash2, ArrowRight, ShieldCheck } from "lucide-react";
import { useCart, useAuth } from "@/lib/context";
import { formatVND } from "@/lib/api";
import { CheckoutModal } from "./checkout-modal";

export function CartDrawer() {
  const { cart, isDrawerOpen, setIsDrawerOpen, removeItem, addItem, loading } = useCart();
  const { user } = useAuth();
  const [checkoutOpen, setCheckoutOpen] = useState(false);

  if (!isDrawerOpen) return null;

  const items = cart?.items || [];
  const total = cart?.total_amount || 0;

  return (
    <>
      <div className="fixed inset-0 z-50 overflow-hidden animate-fade-in">
        {/* BACKDROP */}
        <div
          onClick={() => setIsDrawerOpen(false)}
          className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm transition-opacity"
        />

        <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
          <div className="w-screen max-w-md bg-white border-l border-slate-200 shadow-2xl flex flex-col justify-between">
            {/* HEADER */}
            <div className="p-6 border-b border-slate-100 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-700">
                  <ShoppingBag className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-lg text-slate-900">Giỏ Hàng Của Bạn</h3>
                  <span className="text-xs text-slate-500">
                    {items.reduce((s, i) => s + i.quantity, 0)} món đồ
                  </span>
                </div>
              </div>
              <button
                onClick={() => setIsDrawerOpen(false)}
                className="p-2 rounded-xl bg-slate-100 text-slate-500 hover:text-slate-900 hover:bg-slate-200 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* BODY: ITEM LIST */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {!user ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                  <ShoppingBag className="w-16 h-16 text-slate-300 stroke-1" />
                  <h4 className="font-semibold text-slate-800">Chưa đăng nhập</h4>
                  <p className="text-xs text-slate-500 max-w-xs">
                    Đăng nhập tài khoản để đồng bộ và lưu trữ giỏ hàng trong cơ sở dữ liệu PostgreSQL.
                  </p>
                </div>
              ) : items.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                  <ShoppingBag className="w-16 h-16 text-slate-300 stroke-1" />
                  <h4 className="font-semibold text-slate-800">Giỏ hàng đang trống</h4>
                  <p className="text-xs text-slate-500 max-w-xs">
                    Khám phá kho sản phẩm và thêm những món đồ ưng ý vào giỏ nhé!
                  </p>
                </div>
              ) : (
                items.map(({ product, quantity }) => (
                  <div
                    key={product.id}
                    className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/90 flex gap-3.5 items-center justify-between shadow-2xs"
                  >
                    <div className="relative w-16 h-16 rounded-xl overflow-hidden bg-white flex-shrink-0 border border-slate-200">
                      {product.image_url ? (
                        <Image
                          src={product.image_url}
                          alt={product.name}
                          fill
                          className="object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-slate-400 text-xs font-mono">
                          {product.sku}
                        </div>
                      )}
                    </div>

                    <div className="flex-1 min-w-0">
                      <h4 className="text-xs sm:text-sm font-semibold text-slate-900 line-clamp-1 mb-1">
                        {product.name}
                      </h4>
                      <div className="text-xs font-bold text-emerald-600 mb-2">
                        {formatVND(Number(product.price))}
                      </div>

                      {/* QUANTITY CHANGER */}
                      <div className="flex items-center gap-2">
                        <div className="flex items-center border border-slate-200 rounded-lg bg-white text-xs shadow-2xs">
                          <button
                            onClick={() => {
                              if (quantity > 1) {
                                addItem(product.id, -1);
                              } else {
                                removeItem(product.id);
                              }
                            }}
                            className="px-2.5 py-0.5 text-slate-600 hover:text-slate-950 hover:bg-slate-100 rounded-l"
                          >
                            -
                          </button>
                          <span className="px-2 font-mono font-bold text-slate-900">{quantity}</span>
                          <button
                            onClick={() => addItem(product.id, 1)}
                            className="px-2.5 py-0.5 text-slate-600 hover:text-slate-950 hover:bg-slate-100 rounded-r"
                            disabled={quantity >= product.stock_quantity}
                          >
                            +
                          </button>
                        </div>

                        <span className="text-[11px] text-slate-400 font-medium">
                          Kho: {product.stock_quantity}
                        </span>
                      </div>
                    </div>

                    {/* DELETE */}
                    <button
                      onClick={() => removeItem(product.id)}
                      className="p-2 rounded-xl text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
                      title="Xóa khỏi giỏ"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))
              )}
            </div>

            {/* FOOTER: TOTAL & CHECKOUT */}
            {items.length > 0 && (
              <div className="p-6 border-t border-slate-200 bg-slate-50/70 space-y-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600 font-medium">Tạm tính giỏ hàng:</span>
                  <span className="text-xl font-extrabold text-emerald-600">
                    {formatVND(Number(total))}
                  </span>
                </div>

                <div className="flex items-center gap-2 text-[11px] text-slate-500">
                  <ShieldCheck className="w-4 h-4 text-emerald-600" />
                  <span>Giao dịch khóa dòng Postgres an toàn và chuẩn xác</span>
                </div>

                <button
                  onClick={() => setCheckoutOpen(true)}
                  disabled={loading}
                  className="w-full py-3.5 rounded-xl font-bold text-sm bg-slate-900 hover:bg-slate-800 text-white shadow-xs flex items-center justify-center gap-2 transition-all"
                >
                  <span>Thanh Toán Đơn Hàng</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* CHECKOUT MODAL */}
      <CheckoutModal
        isOpen={checkoutOpen}
        onClose={() => setCheckoutOpen(false)}
        onSuccess={() => {
          setCheckoutOpen(false);
          setIsDrawerOpen(false);
        }}
      />
    </>
  );
}
