"use client";

import React from "react";
import Link from "next/link";
import { Sparkles, ShieldCheck, Truck, RotateCcw, Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full bg-white border-t border-slate-200 mt-20 pt-14 pb-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* TOP VALUE PROPOSITIONS */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pb-12 mb-12 border-b border-slate-100">
          <div className="flex items-center gap-3.5 p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center flex-shrink-0">
              <Truck className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs sm:text-sm font-bold text-slate-900">Giao Hàng Nhanh Toàn Quốc</h4>
              <p className="text-xs text-slate-500">Miễn phí vận chuyển cho đơn hàng từ 500k</p>
            </div>
          </div>

          <div className="flex items-center gap-3.5 p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center flex-shrink-0">
              <RotateCcw className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs sm:text-sm font-bold text-slate-900">Đổi Trả Dễ Dàng</h4>
              <p className="text-xs text-slate-500">Hỗ trợ đổi trả miễn phí trong 7 ngày</p>
            </div>
          </div>

          <div className="flex items-center gap-3.5 p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center flex-shrink-0">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs sm:text-sm font-bold text-slate-900">Cam Kết Chất Lượng</h4>
              <p className="text-xs text-slate-500">100% sản phẩm có nguồn gốc minh bạch</p>
            </div>
          </div>
        </div>

        {/* MAIN FOOTER LINKS */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* BRAND COL */}
          <div className="space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-slate-950 flex items-center justify-center text-white font-bold shadow-sm">
                <span className="font-extrabold text-base">S</span>
              </div>
              <span className="font-extrabold text-xl text-slate-950 tracking-tight">
                ShopSense
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
              Điểm đến thời trang hiện đại với bộ sưu tập phong phú, chất lượng cao cấp và trải nghiệm mua sắm chuẩn gu được tối ưu riêng cho bạn.
            </p>
            <div className="flex items-center gap-2 text-xs text-slate-500 pt-1">
              <span>Hỗ trợ khách hàng:</span>
              <strong className="text-slate-800">1900 6868 (8:00 - 21:00)</strong>
            </div>
          </div>

          {/* CATALOG LINKS */}
          <div>
            <h4 className="font-bold text-xs uppercase text-slate-900 tracking-wider mb-3">
              Danh Mục Sản Phẩm
            </h4>
            <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
              <li>
                <Link href="/" className="hover:text-emerald-700 transition-colors">
                  Thời Trang Nữ
                </Link>
              </li>
              <li>
                <Link href="/" className="hover:text-emerald-700 transition-colors">
                  Thời Trang Nam
                </Link>
              </li>
              <li>
                <Link href="/" className="hover:text-emerald-700 transition-colors">
                  Bộ Sưu Tập Trẻ Em
                </Link>
              </li>
              <li>
                <Link href="/" className="hover:text-emerald-700 transition-colors">
                  Trang Phục Thể Thao
                </Link>
              </li>
              <li>
                <Link href="/" className="hover:text-emerald-700 transition-colors">
                  Phụ Kiện &amp; Giày Dép
                </Link>
              </li>
            </ul>
          </div>

          {/* CUSTOMER SUPPORT */}
          <div>
            <h4 className="font-bold text-xs uppercase text-slate-900 tracking-wider mb-3">
              Chăm Sóc Khách Hàng
            </h4>
            <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
              <li>
                <Link href="/orders" className="hover:text-emerald-700 transition-colors">
                  Tra Cứu Đơn Hàng
                </Link>
              </li>
              <li>
                <Link href="/compare" className="hover:text-emerald-700 transition-colors">
                  So Sánh Sản Phẩm
                </Link>
              </li>
              <li>
                <Link href="/assistant" className="hover:text-emerald-700 transition-colors">
                  Tư Vấn Phong Cách
                </Link>
              </li>
              <li>
                <span className="text-slate-600">Chính Sách Đổi Trả &amp; Bảo Hành</span>
              </li>
              <li>
                <span className="text-slate-600">Hướng Dẫn Chọn Kích Cỡ (Size Guide)</span>
              </li>
            </ul>
          </div>

          {/* ABOUT & POLICIES */}
          <div>
            <h4 className="font-bold text-xs uppercase text-slate-900 tracking-wider mb-3">
              Về ShopSense
            </h4>
            <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
              <li>
                <span className="text-slate-600">Câu Chuyện Thương Hiệu</span>
              </li>
              <li>
                <span className="text-slate-600">Hệ Thống Cửa Hàng</span>
              </li>
              <li>
                <span className="text-slate-600">Tiêu Chuẩn Bền Vững &amp; Chất Liệu</span>
              </li>
              <li>
                <span className="text-slate-600">Chính Sách Bảo Mật Thanh Toán</span>
              </li>
              <li>
                <span className="text-slate-600">Điều Khoản Dịch Vụ</span>
              </li>
            </ul>
          </div>
        </div>

        {/* BOTTOM COPYRIGHT */}
        <div className="pt-6 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-1.5">
            <span>© {new Date().getFullYear()} ShopSense. Tất cả các quyền được bảo lưu.</span>
          </div>
          <div className="flex items-center gap-3 text-slate-500">
            <span>Thanh toán bảo mật</span>
            <span>•</span>
            <span>Giao nhận toàn quốc</span>
            <span>•</span>
            <span>Hỗ trợ 24/7</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
