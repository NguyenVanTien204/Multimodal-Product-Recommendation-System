"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import { 
  X, 
  ShoppingCart, 
  Check, 
  Scale, 
  Sparkles, 
  ShieldCheck, 
  Truck, 
  RotateCcw
} from "lucide-react";
import { Product, SimilarProduct } from "@/lib/types";
import { formatVND, getProductDetail, getSimilarProducts } from "@/lib/api";
import { useCart, useCompare } from "@/lib/context";

interface ProductDetailModalProps {
  product: Product | null;
  allProducts: Product[];
  onClose: () => void;
  onSelectProduct: (p: Product) => void;
}

export function ProductDetailModal({
  product,
  allProducts,
  onClose,
  onSelectProduct,
}: ProductDetailModalProps) {
  const { addItem } = useCart();
  const { addToCompare, isCompared } = useCompare();

  const [quantity, setQuantity] = useState(1);
  const [isAdding, setIsAdding] = useState(false);
  const [justAdded, setJustAdded] = useState(false);
  const [similarItems, setSimilarItems] = useState<{ product: Product; score: number }[]>([]);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  useEffect(() => {
    if (!product) return;
    setQuantity(1);
    setLoadingSimilar(true);

    getSimilarProducts(product.id, 6)
      .then(async (similars: SimilarProduct[]) => {
        const enriched = await Promise.all(
          similars.map(async (s) => {
            let found = allProducts.find((p) => p.id === s.product_id);
            if (!found) {
              try {
                found = await getProductDetail(s.product_id);
              } catch {
                found = undefined;
              }
            }
            return found ? { product: found, score: s.score } : null;
          })
        );
        const validItems = enriched.filter((item): item is { product: Product; score: number } => item !== null);
        setSimilarItems(validItems);
      })
      .catch(() => setSimilarItems([]))
      .finally(() => setLoadingSimilar(false));
  }, [product, allProducts]);

  if (!product) return null;

  const compared = isCompared(product.id);
  const isOutOfStock = product.stock_quantity <= 0;

  const handleAddToCart = async () => {
    if (isOutOfStock) return;
    setIsAdding(true);
    await addItem(product.id, quantity);
    setIsAdding(false);
    setJustAdded(true);
    setTimeout(() => setJustAdded(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-10 bg-slate-900/40 backdrop-blur-sm animate-fade-in">
      <div 
        className="relative w-full max-w-4xl max-h-[92vh] overflow-y-auto bg-white rounded-3xl border border-slate-200 shadow-2xl p-6 sm:p-8 flex flex-col gap-8"
        onClick={(e) => e.stopPropagation()}
      >
        {/* CLOSE BUTTON */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2.5 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-900 transition-colors border border-slate-200 z-10"
        >
          <X className="w-5 h-5" />
        </button>

        {/* TOP SECTION: IMAGE + DETAILS */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* IMAGE */}
          <div className="relative aspect-square w-full rounded-2xl overflow-hidden bg-slate-50 border border-slate-200">
            {product.image_url ? (
              <Image
                src={product.image_url}
                alt={product.name}
                fill
                sizes="(max-width: 768px) 100vw, 50vw"
                className="object-cover object-center"
                priority
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-slate-400">
                <ShoppingCart className="w-16 h-16 stroke-1 mb-2" />
                <span>Không có ảnh</span>
              </div>
            )}
            <span className="absolute top-4 left-4 px-3 py-1 rounded-full text-xs font-mono font-bold bg-white/90 text-slate-700 border border-slate-200 shadow-xs">
              {product.sku}
            </span>
          </div>

          {/* INFO & ACTIONS */}
          <div className="flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2.5 py-0.5 rounded-md text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                  Chính Hãng ShopSense
                </span>
                {isOutOfStock ? (
                  <span className="px-2.5 py-0.5 rounded-md text-xs font-semibold bg-rose-50 text-rose-600 border border-rose-200">
                    Hết Hàng
                  </span>
                ) : (
                  <span className="px-2.5 py-0.5 rounded-md text-xs font-semibold bg-slate-100 text-slate-700">
                    Còn {product.stock_quantity} sản phẩm
                  </span>
                )}
              </div>

              <h2 className="text-xl sm:text-2xl font-bold text-slate-900 mb-3 leading-snug">
                {product.name}
              </h2>

              <div className="text-2xl sm:text-3xl font-extrabold text-emerald-600 mb-5">
                {formatVND(Number(product.price))}
              </div>

              <div className="border-t border-slate-100 pt-4 mb-6">
                <h4 className="text-xs uppercase font-bold text-slate-400 mb-2 tracking-wider">
                  Mô Tả Sản Phẩm
                </h4>
                <p className="text-sm text-slate-600 leading-relaxed max-h-40 overflow-y-auto">
                  {product.description || "Chưa có mô tả chi tiết cho sản phẩm này."}
                </p>
              </div>

              {/* PERKS */}
              <div className="grid grid-cols-3 gap-2 py-3 px-4 rounded-xl bg-slate-50 border border-slate-200 text-center mb-6">
                <div className="flex flex-col items-center gap-1 text-[11px] text-slate-700 font-medium">
                  <Truck className="w-4 h-4 text-emerald-600" />
                  <span>Giao siêu tốc</span>
                </div>
                <div className="flex flex-col items-center gap-1 text-[11px] text-slate-700 font-medium border-x border-slate-200">
                  <ShieldCheck className="w-4 h-4 text-emerald-600" />
                  <span>Bảo hành 12T</span>
                </div>
                <div className="flex flex-col items-center gap-1 text-[11px] text-slate-700 font-medium">
                  <RotateCcw className="w-4 h-4 text-emerald-600" />
                  <span>Đổi trả 7 ngày</span>
                </div>
              </div>
            </div>

            {/* ACTION BUTTONS */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="flex items-center border border-slate-200 rounded-xl bg-slate-50 overflow-hidden shadow-xs">
                  <button
                    onClick={() => setQuantity(Math.max(1, quantity - 1))}
                    className="px-3.5 py-2 text-slate-600 hover:bg-slate-200 transition-colors font-bold text-base"
                    disabled={quantity <= 1 || isOutOfStock}
                  >
                    -
                  </button>
                  <span className="px-3 py-2 text-sm font-bold text-slate-900 min-w-[36px] text-center">
                    {quantity}
                  </span>
                  <button
                    onClick={() => setQuantity(Math.min(product.stock_quantity, quantity + 1))}
                    className="px-3.5 py-2 text-slate-600 hover:bg-slate-200 transition-colors font-bold text-base"
                    disabled={quantity >= product.stock_quantity || isOutOfStock}
                  >
                    +
                  </button>
                </div>

                <button
                  onClick={() => addToCompare(product)}
                  className={`flex-1 py-2.5 px-4 rounded-xl text-xs sm:text-sm font-semibold flex items-center justify-center gap-2 border transition-all ${
                    compared
                      ? "bg-emerald-50 border-emerald-300 text-emerald-800"
                      : "bg-slate-100 hover:bg-slate-200/80 border-slate-200 text-slate-700"
                  }`}
                >
                  <Scale className="w-4 h-4" />
                  <span>{compared ? "Đang so sánh" : "Thêm vào so sánh"}</span>
                </button>
              </div>

              <button
                onClick={handleAddToCart}
                disabled={isOutOfStock || isAdding}
                className={`w-full py-3.5 rounded-xl font-bold text-sm sm:text-base flex items-center justify-center gap-2 shadow-xs transition-all ${
                  justAdded
                    ? "bg-emerald-600 text-white"
                    : isOutOfStock
                    ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                    : "bg-slate-900 hover:bg-slate-800 text-white shadow-sm"
                }`}
              >
                {justAdded ? (
                  <>
                    <Check className="w-5 h-5" />
                    <span>Đã Thêm {quantity} Sản Phẩm Vào Giỏ!</span>
                  </>
                ) : isAdding ? (
                  <span className="animate-spin text-base">⏳ Đang xử lý...</span>
                ) : isOutOfStock ? (
                  <span>Tạm Hết Hàng</span>
                ) : (
                  <>
                    <ShoppingCart className="w-5 h-5 text-white" />
                    <span>Thêm Vào Giỏ Hàng — {formatVND(Number(product.price) * quantity)}</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* BOTTOM SECTION: AI SIMILAR PRODUCTS (QDRANT VECTOR SEARCH) */}
        <div className="border-t border-slate-100 pt-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 text-base sm:text-lg flex items-center gap-2">
                  <span>Sản Phẩm Tương Tự</span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-mono">
                    Qdrant Vector AI
                  </span>
                </h3>
                <p className="text-xs text-slate-500">
                  Tìm kiếm dựa trên độ tương đồng ngữ nghĩa vector đa phương thức
                </p>
              </div>
            </div>
          </div>

          {loadingSimilar ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-6 text-center text-xs text-slate-500">
              <div className="col-span-full animate-pulse">Đang tìm vector tương đồng trong Qdrant...</div>
            </div>
          ) : similarItems.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {similarItems.slice(0, 4).map(({ product: sp, score }) => (
                <div
                  key={sp.id}
                  onClick={() => onSelectProduct(sp)}
                  className="p-3 rounded-xl bg-slate-50 hover:bg-white border border-slate-200 hover:border-emerald-500/50 cursor-pointer transition-all flex flex-col justify-between group shadow-2xs hover:shadow-md"
                >
                  <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-white mb-2 border border-slate-200/60">
                    {sp.image_url ? (
                      <Image
                        src={sp.image_url}
                        alt={sp.name}
                        fill
                        className="object-cover group-hover:scale-105 transition-transform"
                      />
                    ) : null}
                    <span className="absolute bottom-1 right-1 px-1.5 py-0.5 text-[9px] font-bold rounded bg-slate-900/80 text-white">
                      Độ khớp: {(score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <h5 className="text-xs font-semibold text-slate-800 line-clamp-1 group-hover:text-emerald-700 mb-1">
                    {sp.name}
                  </h5>
                  <span className="text-xs font-bold text-emerald-600">
                    {formatVND(Number(sp.price))}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-center text-xs text-slate-500">
              Chưa có dữ liệu vector tương đồng cho sản phẩm này trong kho Qdrant.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
