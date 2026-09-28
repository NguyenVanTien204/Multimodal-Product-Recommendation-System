"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { 
  ArrowLeft, 
  ShoppingCart, 
  Check, 
  Scale, 
  Sparkles, 
  ShieldCheck, 
  Truck, 
  RotateCcw,
  Package
} from "lucide-react";
import { Product, SimilarProduct } from "@/lib/types";
import { formatVND, getProductDetail, getProducts, getSimilarProducts } from "@/lib/api";
import { useCart, useCompare } from "@/lib/context";

export default function ProductDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { addItem } = useCart();
  const { addToCompare, isCompared } = useCompare();

  const productId = Number(params.id);
  const [product, setProduct] = useState<Product | null>(null);
  const [allProducts, setAllProducts] = useState<Product[]>([]);
  const [similarItems, setSimilarItems] = useState<{ product: Product; score: number }[]>([]);
  const [quantity, setQuantity] = useState(1);
  const [loading, setLoading] = useState(true);
  const [isAdding, setIsAdding] = useState(false);
  const [justAdded, setJustAdded] = useState(false);

  useEffect(() => {
    if (!productId) return;
    setLoading(true);

    Promise.all([getProductDetail(productId), getProducts({ page: 1, page_size: 50 }), getSimilarProducts(productId, 6)])
      .then(async ([prod, paginated, similars]) => {
        setProduct(prod);
        setAllProducts(paginated.items);

        // Map similar products by finding in loaded products or fetching individually
        const enrichedList = await Promise.all(
          similars.map(async (s: SimilarProduct) => {
            let found = paginated.items.find((p) => p.id === s.product_id);
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
        const validSimilars = enrichedList.filter((item): item is { product: Product; score: number } => item !== null);
        setSimilarItems(validSimilars);
      })
      .catch((err) => console.error("Error loading product detail:", err))
      .finally(() => setLoading(false));
  }, [productId]);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-16 animate-pulse space-y-8">
        <div className="h-6 w-32 bg-slate-200 rounded-lg" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="aspect-square bg-slate-200 rounded-3xl" />
          <div className="space-y-4">
            <div className="h-8 w-3/4 bg-slate-200 rounded-xl" />
            <div className="h-6 w-1/4 bg-slate-200 rounded-lg" />
            <div className="h-32 bg-slate-200 rounded-2xl" />
          </div>
        </div>
      </div>
    );
  }

  if (!product) {
    return (
      <div className="max-w-md mx-auto px-4 py-24 text-center">
        <Package className="w-16 h-16 text-slate-400 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-slate-900 mb-2">Không tìm thấy sản phẩm</h2>
        <p className="text-xs text-slate-500 mb-6">
          Sản phẩm có thể đã ngừng kinh doanh hoặc đường dẫn không hợp lệ.
        </p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-xs bg-slate-900 hover:bg-slate-800 text-white shadow-xs"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Về Trang Chủ</span>
        </Link>
      </div>
    );
  }

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
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* BACK BUTTON */}
      <button
        onClick={() => router.back()}
        className="inline-flex items-center gap-2 text-xs font-semibold text-slate-600 hover:text-slate-950 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Quay lại</span>
      </button>

      {/* MAIN DETAIL GRID */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
        {/* IMAGE */}
        <div className="relative aspect-square w-full rounded-3xl overflow-hidden bg-slate-50 border border-slate-200 shadow-sm">
          {product.image_url ? (
            <Image
              src={product.image_url}
              alt={product.name}
              fill
              className="object-cover"
              priority
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-400">
              <ShoppingCart className="w-16 h-16 mb-2 stroke-1" />
              <span>{product.sku}</span>
            </div>
          )}
          <span className="absolute top-4 left-4 px-3 py-1 rounded-full text-xs font-mono font-bold bg-white/90 text-slate-700 border border-slate-200 shadow-xs">
            {product.sku}
          </span>
        </div>

        {/* DETAILS & ACTIONS */}
        <div className="flex flex-col justify-between space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-3">
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

            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 mb-4 leading-snug">
              {product.name}
            </h1>

            <div className="text-3xl font-extrabold text-emerald-600 mb-6">
              {formatVND(Number(product.price))}
            </div>

            <div className="border-t border-slate-100 pt-4 mb-6">
              <h4 className="text-xs uppercase font-bold text-slate-400 mb-2 tracking-wider">
                Mô Tả Chi Tiết
              </h4>
              <p className="text-sm text-slate-600 leading-relaxed">
                {product.description || "Chưa có mô tả chi tiết."}
              </p>
            </div>

            <div className="grid grid-cols-3 gap-2 py-3 px-4 rounded-xl bg-slate-50 border border-slate-200 text-center">
              <div className="flex flex-col items-center gap-1 text-[11px] text-slate-700 font-medium">
                <Truck className="w-4 h-4 text-emerald-600" />
                <span>Giao hàng 24h</span>
              </div>
              <div className="flex flex-col items-center gap-1 text-[11px] text-slate-700 font-medium border-x border-slate-200">
                <ShieldCheck className="w-4 h-4 text-emerald-600" />
                <span>Cam kết chính hãng</span>
              </div>
              <div className="flex flex-col items-center gap-1 text-[11px] text-slate-700 font-medium">
                <RotateCcw className="w-4 h-4 text-emerald-600" />
                <span>Đổi trả 7 ngày</span>
              </div>
            </div>
          </div>

          {/* ACTIONS */}
          <div className="space-y-4 pt-4 border-t border-slate-100">
            <div className="flex items-center gap-4">
              <div className="flex items-center border border-slate-200 rounded-xl bg-slate-50 overflow-hidden shadow-2xs">
                <button
                  onClick={() => setQuantity(Math.max(1, quantity - 1))}
                  className="px-3.5 py-2 text-slate-600 hover:bg-slate-200 font-bold"
                  disabled={quantity <= 1 || isOutOfStock}
                >
                  -
                </button>
                <span className="px-4 py-2 text-sm font-bold text-slate-900 min-w-[40px] text-center">
                  {quantity}
                </span>
                <button
                  onClick={() => setQuantity(Math.min(product.stock_quantity, quantity + 1))}
                  className="px-3.5 py-2 text-slate-600 hover:bg-slate-200 font-bold"
                  disabled={quantity >= product.stock_quantity || isOutOfStock}
                >
                  +
                </button>
              </div>

              <button
                onClick={() => addToCompare(product)}
                className={`flex-1 py-2.5 px-4 rounded-xl text-xs sm:text-sm font-semibold flex items-center justify-center gap-2 border transition-all ${
                  compared
                    ? "bg-emerald-50 border-emerald-300 text-emerald-700"
                    : "bg-slate-100 hover:bg-slate-200 border-slate-200 text-slate-700"
                }`}
              >
                <Scale className="w-4 h-4" />
                <span>{compared ? "Đang so sánh" : "Thêm vào so sánh"}</span>
              </button>
            </div>

            <button
              onClick={handleAddToCart}
              disabled={isOutOfStock || isAdding}
              className={`w-full py-4 rounded-xl font-bold text-sm sm:text-base flex items-center justify-center gap-2 shadow-xs transition-all ${
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
                  <span>Đã Thêm {quantity} Món Vào Giỏ!</span>
                </>
              ) : isAdding ? (
                <span>Đang thêm...</span>
              ) : isOutOfStock ? (
                <span>Tạm Hết Hàng</span>
              ) : (
                <>
                  <ShoppingCart className="w-5 h-5" />
                  <span>Thêm Vào Giỏ — {formatVND(Number(product.price) * quantity)}</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* SIMILAR PRODUCTS SECTION (QDRANT VECTOR SEARCH) */}
      <div className="border-t border-slate-100 pt-8">
        <div className="flex items-center gap-2.5 mb-6">
          <div className="p-2 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <span>Sản Phẩm Tương Tự Qua Vector Embeddings</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-mono font-medium">
                Qdrant Cosine
              </span>
            </h2>
            <p className="text-xs text-slate-500">
              Gợi ý dựa trên khoảng cách vector đa phương thức trong cơ sở dữ liệu Qdrant
            </p>
          </div>
        </div>

        {similarItems.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {similarItems.map(({ product: sp, score }) => (
              <Link
                key={sp.id}
                href={`/products/${sp.id}`}
                className="p-3.5 rounded-2xl bg-white border border-slate-200 hover:border-emerald-500/50 shadow-2xs hover:shadow-md transition-all flex flex-col justify-between group"
              >
                <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-50 mb-2 border border-slate-100">
                  {sp.image_url ? (
                    <Image
                      src={sp.image_url}
                      alt={sp.name}
                      fill
                      className="object-cover group-hover:scale-105 transition-transform"
                    />
                  ) : null}
                  <span className="absolute bottom-1.5 right-1.5 px-2 py-0.5 text-[10px] font-bold rounded bg-slate-900/80 text-white">
                    Độ khớp {(score * 100).toFixed(0)}%
                  </span>
                </div>
                <h4 className="text-xs font-semibold text-slate-800 line-clamp-1 group-hover:text-emerald-700 mb-1">
                  {sp.name}
                </h4>
                <span className="text-xs font-bold text-emerald-600">
                  {formatVND(Number(sp.price))}
                </span>
              </Link>
            ))}
          </div>
        ) : (
          <div className="p-6 rounded-2xl bg-white border border-slate-200 text-center text-xs text-slate-500">
            Chưa tìm thấy vector tương đồng khác trong kho dữ liệu Qdrant.
          </div>
        )}
      </div>
    </div>
  );
}
