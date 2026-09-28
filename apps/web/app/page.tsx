"use client";

import React, { useState, useEffect, useMemo } from "react";
import { 
  Sparkles, 
  Search, 
  ArrowUpDown, 
  Layers, 
  TrendingUp, 
  RefreshCw,
  Cpu,
  CheckCircle2
} from "lucide-react";
import { AiRecommendationItem, Category, Product } from "@/lib/types";
import { getCategories, getProducts, getAiRecommendationsForYou } from "@/lib/api";
import { ProductCard } from "@/components/product-card";
import { ProductDetailModal } from "@/components/product-detail-modal";
import { AiChatAssistant } from "@/components/ai-chat-assistant";
import { Pagination } from "@/components/pagination";

export default function HomePage() {
  const [categories, setCategories] = useState<Category[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [aiRecommendations, setAiRecommendations] = useState<AiRecommendationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingRecs, setLoadingRecs] = useState(true);

  // Pagination & Filtering state
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 24;
  const [totalItems, setTotalItems] = useState(0);
  const [totalPages, setTotalPages] = useState(1);

  const [searchQuery, setSearchQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [selectedCategoryId, setSelectedCategoryId] = useState<number | null>(null);
  const [sortBy, setSortBy] = useState<"default" | "price-asc" | "price-desc">("default");
  const [onlyInStock, setOnlyInStock] = useState(false);

  // Selected product modal
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);

  // 1. Fetch Categories and AI Recommendations
  useEffect(() => {
    getCategories().then(setCategories).catch(console.error);

    setLoadingRecs(true);
    getAiRecommendationsForYou(8)
      .then((recs) => setAiRecommendations(recs))
      .catch((err) => console.error("Error loading AI recommendations:", err))
      .finally(() => setLoadingRecs(false));
  }, []);

  // 2. Fetch Paginated Products
  const loadProducts = async (page: number = currentPage) => {
    try {
      setLoading(true);
      const res = await getProducts({
        q: submittedQuery || undefined,
        category_id: selectedCategoryId || undefined,
        page: page,
        page_size: pageSize,
      });

      setProducts(res.items);
      setTotalItems(res.total);
      setCurrentPage(res.page);
      setTotalPages(res.total_pages);
    } catch (err) {
      console.error("Failed to load products:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProducts(currentPage);
  }, [selectedCategoryId, submittedQuery, currentPage]);

  const handleCategorySelect = (catId: number | null) => {
    setSelectedCategoryId(catId);
    setCurrentPage(1);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittedQuery(searchQuery.trim());
    setCurrentPage(1);
  };

  const handleClearFilters = () => {
    setSearchQuery("");
    setSubmittedQuery("");
    setSelectedCategoryId(null);
    setOnlyInStock(false);
    setSortBy("default");
    setCurrentPage(1);
  };

  // Client-side sorting for current page
  const displayedProducts = useMemo(() => {
    let list = [...products];

    if (onlyInStock) {
      list = list.filter((p) => p.stock_quantity > 0);
    }

    if (sortBy === "price-asc") {
      list.sort((a, b) => Number(a.price) - Number(b.price));
    } else if (sortBy === "price-desc") {
      list.sort((a, b) => Number(b.price) - Number(a.price));
    }

    return list;
  }, [products, onlyInStock, sortBy]);

  const getCategoryName = (catId: number | null) => {
    if (!catId) return undefined;
    return categories.find((c) => c.id === catId)?.name;
  };

  return (
    <div className="flex-1 pb-20">
      {/* ================= HERO SECTION ================= */}
      <section className="relative overflow-hidden py-14 sm:py-24 border-b border-slate-200/90 bg-gradient-to-b from-white via-slate-50/50 to-slate-100/50 bg-grid-subtle">
        {/* Soft Ambient Glows */}
        <div className="absolute top-0 left-1/3 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-1/3 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200/80 shadow-xs mb-6 animate-fade-in">
            <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
            <span>Mô Hình Gợi Ý Đa Phương Thức & Vector AI Trực Tuyến</span>
          </div>

          <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold text-slate-950 tracking-tight leading-tight max-w-4xl mx-auto mb-6">
            Mua Sắm Thông Minh Cùng{" "}
            <span className="bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-500 bg-clip-text text-transparent">
              Trí Tuệ Nhân Tạo
            </span>
          </h1>

          <p className="text-sm sm:text-lg text-slate-600 max-w-2xl mx-auto mb-10 leading-relaxed font-normal">
            Hệ sinh thái thương mại điện tử đồng bộ dữ liệu thời gian thực giữa PostgreSQL,
            Qdrant Vector Database và kho dữ liệu thời trang Amazon Reviews 2023.
          </p>

          {/* SEARCH FORM HERO */}
          <form
            onSubmit={handleSearchSubmit}
            className="max-w-2xl mx-auto relative flex items-center shadow-[0_8px_30px_rgb(0,0,0,0.06)] rounded-2xl bg-white p-2 border border-slate-200 focus-within:border-emerald-500 focus-within:ring-4 focus-within:ring-emerald-500/10 transition-all"
          >
            <div className="pl-3.5 text-slate-400">
              <Search className="w-5 h-5" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Tìm kiếm áo sơ mi, blazer, sneaker, đồng hồ, túi xách..."
              className="flex-1 bg-transparent px-3 py-2.5 text-xs sm:text-sm text-slate-900 placeholder-slate-400 focus:outline-none"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => {
                  setSearchQuery("");
                  setSubmittedQuery("");
                  setCurrentPage(1);
                }}
                className="text-xs text-slate-400 hover:text-slate-700 px-2"
              >
                Xóa
              </button>
            )}
            <button
              type="submit"
              className="px-5 py-2.5 rounded-xl font-bold text-xs sm:text-sm bg-slate-900 hover:bg-slate-800 text-white transition-all shadow-xs"
            >
              Tìm Kiếm
            </button>
          </form>

          {/* REAL STATS PILLS */}
          <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-6 mt-10 text-xs text-slate-600">
            <div className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white border border-slate-200/90 shadow-xs">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>{totalItems.toLocaleString("vi-VN")} sản phẩm thực từ Parquet</span>
            </div>
            <div className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white border border-slate-200/90 shadow-xs">
              <Cpu className="w-3.5 h-3.5 text-teal-600" />
              <span>152.086 Vector Embeddings 1024D Qdrant</span>
            </div>
            <div className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white border border-slate-200/90 shadow-xs">
              <Sparkles className="w-3.5 h-3.5 text-blue-600" />
              <span>Gợi ý Đa Phương Thức & Sequential AI</span>
            </div>
          </div>
        </div>
      </section>

      {/* ================= AI RECOMMENDED SECTION (LIVE QDRANT VECTOR AI) ================= */}
      {aiRecommendations.length > 0 && !selectedCategoryId && !submittedQuery && (
        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-12 sm:pt-14">
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-200/80">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-600 flex items-center justify-center shadow-xs">
                <TrendingUp className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-lg sm:text-xl font-bold text-slate-900 flex items-center gap-2">
                  <span>Gợi Ý Dành Riêng Cho Bạn</span>
                  <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Qdrant Vector AI
                  </span>
                </h2>
                <p className="text-xs text-slate-500">
                  Phân tích ngữ nghĩa không gian vector 1024 chiều đa phương thức từ cơ sở dữ liệu Qdrant
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {aiRecommendations.slice(0, 4).map((rec) => (
              <ProductCard
                key={`rec-${rec.product.id}`}
                product={rec.product}
                categoryName={getCategoryName(rec.product.category_id)}
                onOpenDetail={(prod) => setSelectedProduct(prod)}
                isAiRecommended={true}
                score={rec.score}
              />
            ))}
          </div>
        </section>
      )}

      {/* ================= CATALOG BROWSING, PAGINATION & FILTERS ================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-12 sm:pt-14">
        {/* SECTION HEADER & CONTROLS */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 border-b border-slate-200/80">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
              <Layers className="w-5 h-5 text-emerald-600" />
              <span>Danh Mục Sản Phẩm</span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Hiển thị <span className="text-emerald-700 font-bold">{totalItems.toLocaleString("vi-VN")}</span> sản phẩm trong cơ sở dữ liệu PostgreSQL
            </p>
          </div>

          {/* RIGHT FILTERS: SORT, STOCK TOGGLE, REFRESH */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* ONLY IN STOCK */}
            <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer select-none bg-white px-3.5 py-2 rounded-xl border border-slate-200 shadow-xs hover:bg-slate-50 transition-colors">
              <input
                type="checkbox"
                checked={onlyInStock}
                onChange={(e) => setOnlyInStock(e.target.checked)}
                className="rounded border-slate-300 text-emerald-600 focus:ring-0"
              />
              <span>Chỉ còn hàng</span>
            </label>

            {/* SORT */}
            <div className="flex items-center gap-1.5 bg-white px-3 py-2 rounded-xl border border-slate-200 shadow-xs text-xs text-slate-700">
              <ArrowUpDown className="w-3.5 h-3.5 text-slate-400" />
              <select
                value={sortBy}
                onChange={(e: any) => setSortBy(e.target.value)}
                className="bg-transparent text-slate-800 focus:outline-none cursor-pointer text-xs font-medium"
              >
                <option value="default">Mặc định</option>
                <option value="price-asc">Giá: Thấp đến Cao</option>
                <option value="price-desc">Giá: Cao đến Thấp</option>
              </select>
            </div>

            <button
              onClick={() => loadProducts(currentPage)}
              title="Làm mới dữ liệu từ backend"
              className="p-2.5 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-600 hover:text-slate-900 shadow-xs transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin text-emerald-600" : ""}`} />
            </button>
          </div>
        </div>

        {/* CATEGORY TABS */}
        <div className="flex items-center gap-2 overflow-x-auto py-5 no-scrollbar">
          <button
            onClick={() => handleCategorySelect(null)}
            className={`px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all whitespace-nowrap ${
              selectedCategoryId === null
                ? "bg-slate-900 text-white shadow-sm"
                : "bg-white hover:bg-slate-100 text-slate-600 border border-slate-200 shadow-xs"
            }`}
          >
            Tất Cả ({totalItems.toLocaleString("vi-VN")})
          </button>
          {categories.map((cat) => (
            <button
              key={cat.id}
              onClick={() => handleCategorySelect(cat.id)}
              className={`px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all whitespace-nowrap ${
                selectedCategoryId === cat.id
                  ? "bg-slate-900 text-white shadow-sm"
                  : "bg-white hover:bg-slate-100 text-slate-600 border border-slate-200 shadow-xs"
              }`}
            >
              {cat.name}
            </button>
          ))}
        </div>

        {/* PRODUCTS GRID */}
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 py-10">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
              <div
                key={n}
                className="h-80 rounded-2xl bg-white border border-slate-200 animate-pulse shadow-xs"
              />
            ))}
          </div>
        ) : displayedProducts.length === 0 ? (
          <div className="py-20 text-center bg-white rounded-3xl border border-slate-200 my-6 shadow-xs">
            <Search className="w-12 h-12 text-slate-400 mx-auto mb-3 stroke-1" />
            <h3 className="text-lg font-bold text-slate-900 mb-1">
              Không tìm thấy sản phẩm phù hợp
            </h3>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              Thử tìm kiếm với từ khóa khác hoặc xóa bộ lọc danh mục hiện tại.
            </p>
            <button
              onClick={handleClearFilters}
              className="mt-4 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-white shadow-xs"
            >
              Xóa bộ lọc
            </button>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {displayedProducts.map((p) => (
                <ProductCard
                  key={p.id}
                  product={p}
                  categoryName={getCategoryName(p.category_id)}
                  onOpenDetail={(prod) => setSelectedProduct(prod)}
                />
              ))}
            </div>

            {/* PAGINATION COMPONENT */}
            <Pagination
              currentPage={currentPage}
              totalPages={totalPages}
              totalItems={totalItems}
              pageSize={pageSize}
              onPageChange={(newPage) => {
                setCurrentPage(newPage);
                window.scrollTo({ top: 500, behavior: "smooth" });
              }}
            />
          </>
        )}
      </section>

      {/* ================= PRODUCT DETAIL MODAL ================= */}
      <ProductDetailModal
        product={selectedProduct}
        allProducts={products}
        onClose={() => setSelectedProduct(null)}
        onSelectProduct={(p) => setSelectedProduct(p)}
      />

      {/* ================= FLOATING AI ASSISTANT ================= */}
      <AiChatAssistant
        products={products}
        onSelectProduct={(p) => setSelectedProduct(p)}
        isFloating={true}
      />
    </div>
  );
}
