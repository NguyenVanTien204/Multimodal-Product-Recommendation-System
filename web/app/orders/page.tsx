"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Package, Calendar, MapPin, CheckCircle, Clock, Truck, XCircle, ArrowRight, RefreshCw } from "lucide-react";
import { useAuth } from "@/lib/context";
import { getOrders, formatVND } from "@/lib/api";
import { Order } from "@/lib/types";

export default function OrdersPage() {
  const { user } = useAuth();
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStatus, setSelectedStatus] = useState<string>("ALL");

  const loadOrders = async () => {
    if (!user) {
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      const data = await getOrders();
      setOrders(data);
    } catch (err) {
      console.error("Failed to load orders:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOrders();
  }, [user]);

  const filteredOrders = orders.filter((o) => {
    if (selectedStatus === "ALL") return true;
    return o.status === selectedStatus;
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "PENDING":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200">
            <Clock className="w-3.5 h-3.5" />
            <span>Chờ Xác Nhận</span>
          </span>
        );
      case "PAID":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>Đã Thanh Toán</span>
          </span>
        );
      case "PROCESSING":
      case "SHIPPED":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
            <Truck className="w-3.5 h-3.5" />
            <span>Đang Vận Chuyển</span>
          </span>
        );
      case "CANCELLED":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200">
            <XCircle className="w-3.5 h-3.5" />
            <span>Đã Hủy</span>
          </span>
        );
      default:
        return (
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700">
            {status}
          </span>
        );
    }
  };

  if (!user) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-24 text-center">
        <div className="w-16 h-16 rounded-3xl bg-white border border-slate-200 text-slate-400 flex items-center justify-center mx-auto mb-4 shadow-xs">
          <Package className="w-8 h-8" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-2">Đăng Nhập Để Xem Đơn Hàng</h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto mb-6">
          Vui lòng đăng nhập để tra cứu lịch sử đơn hàng và tiến độ xử lý giao hàng.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* HEADER */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8 pb-6 border-b border-slate-200">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 flex items-center gap-2.5">
            <Package className="w-7 h-7 text-emerald-600" />
            <span>Lịch Sử Đơn Hàng Của Bạn</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Dữ liệu đơn hàng đồng bộ trực tiếp từ bảng orders PostgreSQL
          </p>
        </div>

        <button
          onClick={loadOrders}
          className="p-2.5 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-600 hover:text-slate-900 shadow-2xs transition-colors flex items-center gap-2 text-xs font-medium"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin text-emerald-600" : ""}`} />
          <span>Làm mới</span>
        </button>
      </div>

      {/* FILTER TABS */}
      <div className="flex items-center gap-2 overflow-x-auto pb-4 mb-6 no-scrollbar">
        {[
          { key: "ALL", label: "Tất Cả Đơn" },
          { key: "PENDING", label: "Chờ Xác Nhận" },
          { key: "PAID", label: "Đã Thanh Toán" },
          { key: "SHIPPED", label: "Đang Giao" },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setSelectedStatus(tab.key)}
            className={`px-4 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
              selectedStatus === tab.key
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-white text-slate-600 hover:text-slate-900 border border-slate-200 shadow-2xs"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ORDERS LIST */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((n) => (
            <div
              key={n}
              className="h-44 rounded-2xl bg-white border border-slate-200 animate-pulse shadow-xs"
            />
          ))}
        </div>
      ) : filteredOrders.length === 0 ? (
        <div className="py-20 text-center bg-white rounded-3xl border border-slate-200 shadow-xs">
          <Package className="w-12 h-12 text-slate-400 mx-auto mb-3 stroke-1" />
          <h3 className="text-lg font-bold text-slate-900 mb-1">Chưa có đơn hàng nào</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto mb-6">
            Bạn chưa thực hiện đơn đặt hàng nào trong trạng thái này.
          </p>
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-xs bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-all"
          >
            <span>Mua Sắm Ngay</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      ) : (
        <div className="space-y-5">
          {filteredOrders.map((order) => (
            <div
              key={order.id}
              className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 space-y-4 hover:border-slate-300 transition-all shadow-xs"
            >
              {/* ORDER HEADER */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-100">
                <div className="space-y-1">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-sm font-bold text-slate-900">
                      Đơn hàng #{order.id}
                    </span>
                    {getStatusBadge(order.status)}
                  </div>
                  <div className="flex items-center gap-1.5 text-xs text-slate-500">
                    <Calendar className="w-3.5 h-3.5 text-slate-400" />
                    <span>
                      {new Date(order.created_at).toLocaleDateString("vi-VN", {
                        hour: "2-digit",
                        minute: "2-digit",
                        day: "2-digit",
                        month: "2-digit",
                        year: "numeric",
                      })}
                    </span>
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-xs text-slate-400 block font-medium">Tổng tiền</span>
                  <span className="text-lg sm:text-xl font-extrabold text-emerald-600">
                    {formatVND(Number(order.total_amount))}
                  </span>
                </div>
              </div>

              {/* ITEMS */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                  Danh Sách Món Đồ
                </span>
                <div className="divide-y divide-slate-100">
                  {order.items.map((item, idx) => (
                    <div
                      key={idx}
                      className="py-2.5 flex items-center justify-between text-xs sm:text-sm"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded text-xs">
                          {item.quantity}x
                        </span>
                        <span className="text-slate-800 font-medium">{item.product_name}</span>
                      </div>
                      <span className="font-mono text-slate-700 font-semibold">
                        {formatVND(Number(item.unit_price) * item.quantity)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* SHIPPING ADDRESS */}
              <div className="pt-3 border-t border-slate-100 flex items-start gap-2 text-xs text-slate-500">
                <MapPin className="w-3.5 h-3.5 text-slate-400 mt-0.5 flex-shrink-0" />
                <span>Địa chỉ giao hàng: <strong className="text-slate-700">{order.shipping_address}</strong></span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
