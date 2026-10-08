"use client";

import React, { useState, useEffect } from "react";
import { Sparkles, ShieldCheck, Zap } from "lucide-react";
import { Product } from "@/lib/types";
import { getProducts } from "@/lib/api";
import { AiChatAssistant } from "@/components/ai-chat-assistant";
import { ProductDetailModal } from "@/components/product-detail-modal";

export default function AssistantPage() {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);

  useEffect(() => {
    getProducts({ page: 1, page_size: 50 }).then((res) => setProducts(res.items)).catch(console.error);
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 flex flex-col">
      {/* HEADER BANNER */}
      <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 sm:p-8 bg-white rounded-3xl border border-slate-200 shadow-xs">
        <div>
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 mb-2.5">
            <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
            <span>Tư Vấn Phong Cách Trực Tuyến</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
            Stylist Thời Trang Cá Nhân
          </h1>
          <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-xl leading-relaxed">
            Nhận gợi ý phối đồ theo dịp, tư vấn chọn form dáng, chất liệu hoặc tải lên hình ảnh trang phục mẫu bạn yêu thích để tìm kiếm thiết kế phù hợp nhất tại ShopSense.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 text-xs text-slate-600">
          <div className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-50 border border-slate-200 shadow-2xs font-medium">
            <Zap className="w-4 h-4 text-emerald-600" />
            <span>Tư vấn tức thì</span>
          </div>
          <div className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-50 border border-slate-200 shadow-2xs font-medium">
            <ShieldCheck className="w-4 h-4 text-teal-600" />
            <span>Chuẩn gu thời trang</span>
          </div>
        </div>
      </div>

      {/* CHAT CONTAINER EMBEDDED */}
      <div className="flex-1 min-h-[640px] flex flex-col">
        <AiChatAssistant
          products={products}
          onSelectProduct={(p) => setSelectedProduct(p)}
          isFloating={false}
        />
      </div>

      {/* PRODUCT MODAL */}
      <ProductDetailModal
        product={selectedProduct}
        allProducts={products}
        onClose={() => setSelectedProduct(null)}
        onSelectProduct={(p) => setSelectedProduct(p)}
      />
    </div>
  );
}
