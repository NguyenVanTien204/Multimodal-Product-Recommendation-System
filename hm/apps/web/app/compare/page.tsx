"use client";

import React from "react";
import Image from "next/image";
import Link from "next/link";
import { Scale, Trash2, ShoppingCart, ArrowLeft, Check, Sparkles } from "lucide-react";
import { useCompare, useCart } from "@/lib/context";
import { formatVND } from "@/lib/api";

export default function ComparePage() {
  const { compareItems, removeFromCompare, clearCompare } = useCompare();
  const { addItem } = useCart();

  if (compareItems.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-24 text-center">
        <div className="w-16 h-16 rounded-3xl bg-white border border-slate-200 text-slate-400 flex items-center justify-center mx-auto mb-4 shadow-xs">
          <Scale className="w-8 h-8" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-2">Chưa Có Sản Phẩm So Sánh</h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto mb-8">
          Chọn biểu tượng chiếc cân trên thẻ sản phẩm để đưa sản phẩm vào bảng so sánh chi tiết.
        </p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-sm bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-all"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Quay Lại Khám Phá Sản Phẩm</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* HEADER */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8 pb-6 border-b border-slate-200">
        <div>
          <div className="flex items-center gap-2 text-xs text-emerald-700 font-semibold mb-1">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Phân Tích Đa Chiều</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 flex items-center gap-2.5">
            <Scale className="w-7 h-7 text-emerald-600" />
            <span>Bảng So Sánh Sản Phẩm ({compareItems.length}/4)</span>
          </h1>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={clearCompare}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-white hover:bg-rose-50 border border-slate-200 text-rose-600 transition-colors flex items-center gap-1.5 shadow-2xs"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Xóa Tất Cả</span>
          </button>
          <Link
            href="/"
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-white shadow-2xs transition-colors"
          >
            Thêm sản phẩm khác
          </Link>
        </div>
      </div>

      {/* COMPARISON MATRIX */}
      <div className="overflow-x-auto pb-6">
        <div className="min-w-full bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="w-48 p-5 text-left text-xs font-bold uppercase text-slate-500 bg-slate-50">
                  Thuộc Tính
                </th>
                {compareItems.map((item) => (
                  <th
                    key={item.id}
                    className="min-w-[260px] p-5 text-center bg-white relative border-l border-slate-100"
                  >
                    <button
                      onClick={() => removeFromCompare(item.id)}
                      className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-100 text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
                      title="Xóa sản phẩm này"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>

                    <div className="relative aspect-square w-36 mx-auto rounded-xl overflow-hidden bg-slate-50 mb-3 border border-slate-200">
                      {item.image_url ? (
                        <Image src={item.image_url} alt={item.name} fill className="object-cover" />
                      ) : null}
                    </div>

                    <h3 className="text-sm font-bold text-slate-900 line-clamp-2 mb-2">
                      {item.name}
                    </h3>

                    <div className="text-base font-extrabold text-emerald-600 mb-3">
                      {formatVND(Number(item.price))}
                    </div>

                    <button
                      onClick={() => addItem(item.id, 1)}
                      disabled={item.stock_quantity <= 0}
                      className="w-full py-2.5 px-3 rounded-xl text-xs font-bold bg-slate-900 hover:bg-slate-800 text-white shadow-xs flex items-center justify-center gap-1.5 disabled:opacity-40 transition-all"
                    >
                      <ShoppingCart className="w-3.5 h-3.5" />
                      <span>{item.stock_quantity > 0 ? "Thêm vào giỏ" : "Hết hàng"}</span>
                    </button>
                  </th>
                ))}
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-100 text-xs sm:text-sm">
              {/* SKU */}
              <tr className="hover:bg-slate-50/50">
                <td className="p-4 font-semibold text-slate-500 bg-slate-50/50">Mã SKU</td>
                {compareItems.map((item) => (
                  <td key={item.id} className="p-4 text-center font-mono text-slate-700 border-l border-slate-100">
                    {item.sku}
                  </td>
                ))}
              </tr>

              {/* STOCK STATUS */}
              <tr className="hover:bg-slate-50/50">
                <td className="p-4 font-semibold text-slate-500 bg-slate-50/50">Tồn kho</td>
                {compareItems.map((item) => (
                  <td key={item.id} className="p-4 text-center font-medium border-l border-slate-100">
                    {item.stock_quantity > 0 ? (
                      <span className="text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded">Còn {item.stock_quantity} sp</span>
                    ) : (
                      <span className="text-rose-600 font-semibold bg-rose-50 px-2 py-0.5 rounded">Hết hàng</span>
                    )}
                  </td>
                ))}
              </tr>

              {/* DESCRIPTION */}
              <tr className="hover:bg-slate-50/50">
                <td className="p-4 font-semibold text-slate-500 bg-slate-50/50">Mô tả sản phẩm</td>
                {compareItems.map((item) => (
                  <td key={item.id} className="p-4 text-slate-600 text-xs leading-relaxed text-left align-top border-l border-slate-100">
                    {item.description || "Chưa có mô tả."}
                  </td>
                ))}
              </tr>

              {/* STATUS */}
              <tr className="hover:bg-slate-50/50">
                <td className="p-4 font-semibold text-slate-500 bg-slate-50/50">Trạng thái kinh doanh</td>
                {compareItems.map((item) => (
                  <td key={item.id} className="p-4 text-center border-l border-slate-100">
                    <span className="inline-flex items-center gap-1 text-xs text-emerald-700 font-semibold bg-emerald-50 px-2.5 py-1 rounded-full">
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                      Chính hãng ShopSense
                    </span>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
