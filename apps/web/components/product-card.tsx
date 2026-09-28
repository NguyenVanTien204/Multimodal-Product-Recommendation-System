"use client";

import React, { useState } from "react";
import Image from "next/image";
import { ShoppingCart, Check, Scale, Eye, Sparkles } from "lucide-react";
import { Product } from "@/lib/types";
import { formatVND } from "@/lib/api";
import { useCart, useCompare } from "@/lib/context";

interface ProductCardProps {
  product: Product;
  categoryName?: string;
  onOpenDetail?: (product: Product) => void;
  isAiRecommended?: boolean;
  score?: number;
}

export function ProductCard({
  product,
  categoryName,
  onOpenDetail,
  isAiRecommended,
  score,
}: ProductCardProps) {
  const { addItem } = useCart();
  const { addToCompare, isCompared } = useCompare();
  const [isAdding, setIsAdding] = useState(false);
  const [justAdded, setJustAdded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const compared = isCompared(product.id);

  const handleAddToCart = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (product.stock_quantity <= 0) return;
    setIsAdding(true);
    await addItem(product.id, 1);
    setIsAdding(false);
    setJustAdded(true);
    setTimeout(() => setJustAdded(false), 1800);
  };

  const handleToggleCompare = (e: React.MouseEvent) => {
    e.stopPropagation();
    addToCompare(product);
  };

  // Stock status
  const isOutOfStock = product.stock_quantity <= 0;
  const isLowStock = product.stock_quantity > 0 && product.stock_quantity <= 10;

  return (
    <div
      onClick={() => onOpenDetail && onOpenDetail(product)}
      className="group relative flex flex-col rounded-2xl bg-white border border-slate-200/90 hover:border-emerald-500/50 transition-all duration-300 overflow-hidden cursor-pointer shadow-xs hover:shadow-xl hover:-translate-y-1"
    >
      {/* BADGES TOP */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-1.5 items-start pointer-events-none">
        {isAiRecommended && (
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-600 text-white shadow-sm">
            <Sparkles className="w-3 h-3 text-emerald-200" />
            <span>AI Gợi Ý {score ? `(${(score * 100).toFixed(0)}%)` : ""}</span>
          </span>
        )}
        {categoryName && (
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-white/90 text-slate-700 border border-slate-200/90 shadow-xs backdrop-blur-md">
            {categoryName}
          </span>
        )}
      </div>

      {/* COMPARE BUTTON TOP RIGHT */}
      <button
        onClick={handleToggleCompare}
        title={compared ? "Bỏ khỏi so sánh" : "Thêm vào so sánh"}
        className={`absolute top-3 right-3 z-10 p-2 rounded-xl backdrop-blur-md border transition-all ${
          compared
            ? "bg-emerald-600 border-emerald-500 text-white shadow-sm"
            : "bg-white/90 border-slate-200/90 text-slate-600 hover:text-slate-900 hover:bg-white shadow-xs"
        }`}
      >
        <Scale className="w-4 h-4" />
      </button>

      {/* IMAGE CONTAINER */}
      <div className="relative w-full aspect-square bg-slate-50 overflow-hidden">
        {product.image_url && !imageError ? (
          <Image
            src={product.image_url}
            alt={product.name}
            fill
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
            className="object-cover object-center group-hover:scale-105 transition-transform duration-500"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-slate-400 p-4 text-center">
            <ShoppingCart className="w-12 h-12 mb-2 stroke-1" />
            <span className="text-xs font-mono">{product.sku}</span>
          </div>
        )}

        {/* HOVER OVERLAY QUICK VIEW */}
        <div className="absolute inset-0 bg-slate-950/20 backdrop-blur-[2px] opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
          <span className="px-3.5 py-1.5 rounded-xl bg-white/95 border border-slate-200 text-xs font-bold text-slate-900 flex items-center gap-1.5 shadow-md">
            <Eye className="w-3.5 h-3.5 text-emerald-600" />
            Xem nhanh
          </span>
        </div>
      </div>

      {/* CONTENT */}
      <div className="p-4 sm:p-5 flex-1 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5">
            <span className="font-mono text-[11px] text-slate-400">{product.sku}</span>
            {isOutOfStock ? (
              <span className="text-rose-600 font-semibold bg-rose-50 px-2 py-0.5 rounded-md">Hết hàng</span>
            ) : isLowStock ? (
              <span className="text-amber-700 font-semibold bg-amber-50 px-2 py-0.5 rounded-md">Còn {product.stock_quantity} sp</span>
            ) : (
              <span className="text-emerald-700 font-medium bg-emerald-50 px-2 py-0.5 rounded-md">Sẵn hàng</span>
            )}
          </div>

          <h3 className="font-semibold text-sm sm:text-base text-slate-900 group-hover:text-emerald-700 transition-colors line-clamp-2 mb-2 leading-snug">
            {product.name}
          </h3>
        </div>

        {/* PRICE & ADD TO CART */}
        <div className="pt-3.5 mt-auto border-t border-slate-100 flex items-center justify-between gap-2">
          <div>
            <span className="text-[11px] text-slate-400 uppercase font-semibold block">Giá bán</span>
            <span className="text-base sm:text-lg font-extrabold text-emerald-600">
              {formatVND(Number(product.price))}
            </span>
          </div>

          <button
            onClick={handleAddToCart}
            disabled={isOutOfStock || isAdding}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-1.5 transition-all ${
              justAdded
                ? "bg-emerald-600 text-white shadow-sm"
                : isOutOfStock
                ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                : "bg-slate-900 hover:bg-emerald-600 text-white shadow-xs hover:shadow-sm"
            }`}
          >
            {justAdded ? (
              <>
                <Check className="w-3.5 h-3.5" />
                <span>Đã thêm</span>
              </>
            ) : isAdding ? (
              <span className="animate-spin text-sm">⏳</span>
            ) : isOutOfStock ? (
              <span>Hết hàng</span>
            ) : (
              <>
                <ShoppingCart className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Thêm giỏ</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
