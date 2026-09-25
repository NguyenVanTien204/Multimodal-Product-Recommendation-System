"use client";

import React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

export function Pagination({
  currentPage,
  totalPages,
  totalItems,
  pageSize,
  onPageChange,
}: PaginationProps) {
  if (totalPages <= 1) return null;

  const startItem = (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);

  // Helper to generate page numbers with ellipsis
  const getPageNumbers = () => {
    const pages: (number | string)[] = [];
    const maxVisible = 5;

    if (totalPages <= maxVisible + 2) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);

      let start = Math.max(2, currentPage - 1);
      let end = Math.min(totalPages - 1, currentPage + 1);

      if (currentPage <= 3) {
        start = 2;
        end = 4;
      } else if (currentPage >= totalPages - 2) {
        start = totalPages - 3;
        end = totalPages - 1;
      }

      if (start > 2) pages.push("...");
      for (let i = start; i <= end; i++) pages.push(i);
      if (end < totalPages - 1) pages.push("...");

      pages.push(totalPages);
    }

    return pages;
  };

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-8 border-t border-slate-200 mt-10">
      {/* TEXT SUMMARY */}
      <div className="text-xs sm:text-sm text-slate-500">
        Hiển thị <span className="font-semibold text-slate-900">{startItem.toLocaleString("vi-VN")}</span> -{" "}
        <span className="font-semibold text-slate-900">{endItem.toLocaleString("vi-VN")}</span> trên tổng số{" "}
        <span className="font-bold text-emerald-700">{totalItems.toLocaleString("vi-VN")}</span> sản phẩm
      </div>

      {/* PAGINATION CONTROLS */}
      <div className="flex items-center gap-1.5">
        {/* PREV BUTTON */}
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage <= 1}
          className="p-2 rounded-xl bg-white border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 disabled:opacity-30 disabled:cursor-not-allowed shadow-2xs transition-all"
          title="Trang trước"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>

        {/* PAGE NUMBERS */}
        {getPageNumbers().map((page, idx) => {
          if (page === "...") {
            return (
              <span key={`ellipsis-${idx}`} className="px-2 text-xs text-slate-400 select-none">
                ...
              </span>
            );
          }

          const pageNum = Number(page);
          const isCurrent = pageNum === currentPage;

          return (
            <button
              key={`page-${pageNum}`}
              onClick={() => onPageChange(pageNum)}
              className={`w-9 h-9 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
                isCurrent
                  ? "bg-slate-900 text-white font-bold shadow-xs border border-slate-900"
                  : "bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 shadow-2xs"
              }`}
            >
              {pageNum}
            </button>
          );
        })}

        {/* NEXT BUTTON */}
        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage >= totalPages}
          className="p-2 rounded-xl bg-white border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 disabled:opacity-30 disabled:cursor-not-allowed shadow-2xs transition-all"
          title="Trang sau"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
