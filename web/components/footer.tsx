"use client";

import React from "react";
import Link from "next/link";
import { Sparkles, Database, Server, Cpu, ExternalLink } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full bg-white border-t border-slate-200 mt-20 pt-12 pb-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* BRAND COL */}
          <div className="md:col-span-2 space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 flex items-center justify-center text-white font-bold shadow-sm">
                <Sparkles className="w-5 h-5" />
              </div>
              <span className="font-extrabold text-xl text-slate-900 tracking-tight">
                ShopSense
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-600 max-w-md leading-relaxed">
              Nền tảng thương mại điện tử tích hợp hệ thống gợi ý sản phẩm đa phương thức
              (Multimodal Recommendation) và trợ lý mua sắm AI thông minh, kết nối trực tiếp
              FastAPI, PostgreSQL và cơ sở dữ liệu vector Qdrant.
            </p>

            <div className="flex flex-wrap gap-2 pt-2">
              <span className="px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-50 text-slate-700 border border-slate-200 flex items-center gap-1.5 shadow-2xs">
                <Server className="w-3.5 h-3.5 text-emerald-600" />
                FastAPI :8000
              </span>
              <span className="px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-50 text-slate-700 border border-slate-200 flex items-center gap-1.5 shadow-2xs">
                <Database className="w-3.5 h-3.5 text-blue-600" />
                PostgreSQL :5432
              </span>
              <span className="px-2.5 py-1 rounded-lg text-[11px] font-medium bg-slate-50 text-slate-700 border border-slate-200 flex items-center gap-1.5 shadow-2xs">
                <Cpu className="w-3.5 h-3.5 text-purple-600" />
                Qdrant Vector :6333
              </span>
            </div>
          </div>

          {/* QUICK LINKS */}
          <div>
            <h4 className="font-bold text-xs uppercase text-slate-900 tracking-wider mb-3">
              Điều Hướng
            </h4>
            <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
              <li>
                <Link href="/" className="hover:text-emerald-700 transition-colors">
                  Khám Phá Sản Phẩm
                </Link>
              </li>
              <li>
                <Link href="/compare" className="hover:text-emerald-700 transition-colors">
                  So Sánh Sản Phẩm
                </Link>
              </li>
              <li>
                <Link href="/orders" className="hover:text-emerald-700 transition-colors">
                  Quản Lý Đơn Hàng
                </Link>
              </li>
              <li>
                <Link href="/assistant" className="hover:text-emerald-700 transition-colors">
                  Trợ Lý AI Trò Chuyện
                </Link>
              </li>
              <li>
                <Link href="/admin" className="hover:text-emerald-700 transition-colors">
                  Trang Quản Trị Hệ Thống
                </Link>
              </li>
            </ul>
          </div>

          {/* RESEARCH & DOCS */}
          <div>
            <h4 className="font-bold text-xs uppercase text-slate-900 tracking-wider mb-3">
              Tài Liệu & Tích Hợp
            </h4>
            <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
              <li>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-emerald-700 transition-colors flex items-center gap-1"
                >
                  <span>FastAPI Swagger UI</span>
                  <ExternalLink className="w-3 h-3 text-slate-400" />
                </a>
              </li>
              <li>
                <a
                  href="http://localhost:6333/dashboard"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-emerald-700 transition-colors flex items-center gap-1"
                >
                  <span>Qdrant Dashboard</span>
                  <ExternalLink className="w-3 h-3 text-slate-400" />
                </a>
              </li>
              <li>
                <span className="text-slate-500">Catalog: Amazon Reviews 152k</span>
              </li>
              <li>
                <span className="text-slate-500">Mô hình: Vector RAG & SASRec</span>
              </li>
            </ul>
          </div>
        </div>

        {/* BOTTOM COPYRIGHT */}
        <div className="pt-6 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div>
            © {new Date().getFullYear()} ShopSense — Đồ Án Tốt Nghiệp: Multimodal Product Recommendation System.
          </div>
          <div>Bảo mật giao dịch ACID • Row Lock PostgreSQL</div>
        </div>
      </div>
    </footer>
  );
}
