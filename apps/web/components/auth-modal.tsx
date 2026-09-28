"use client";

import React, { useState } from "react";
import { X, Lock, Mail, User, Sparkles, Shield, UserCheck } from "lucide-react";
import { useAuth, useToast } from "@/lib/context";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AuthModal({ isOpen, onClose }: AuthModalProps) {
  const { login, register } = useAuth();
  const { showToast } = useToast();

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (mode === "login") {
        await login(email, password);
        onClose();
      } else {
        await register(email, password, fullName);
        setMode("login");
      }
    } catch {
      // toast shown in context
    } finally {
      setLoading(false);
    }
  };

  const fillDemoUser = () => {
    setEmail("demo@shopsense.vn");
    setPassword("demopassword123");
    showToast("Đã điền tài khoản người dùng demo", "info");
  };

  const fillAdminUser = () => {
    setEmail("admin@shopsense.vn");
    setPassword("adminpassword123");
    showToast("Đã điền tài khoản Quản trị viên (Admin)", "info");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-fade-in">
      <div 
        className="relative w-full max-w-md bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 rounded-full bg-slate-100 text-slate-500 hover:text-slate-900 hover:bg-slate-200 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* LOGO & HEADING */}
        <div className="text-center mb-6">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-tr from-emerald-600 to-teal-500 flex items-center justify-center mx-auto mb-3 shadow-md shadow-emerald-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-slate-900">
            {mode === "login" ? "Chào Mừng Trở Lại!" : "Tạo Tài Khoản Mới"}
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            {mode === "login"
              ? "Đăng nhập để lưu trữ giỏ hàng và lịch sử đơn hàng"
              : "Gia nhập nền tảng gợi ý mua sắm thông minh ShopSense"}
          </p>
        </div>

        {/* TABS */}
        <div className="flex rounded-xl bg-slate-100 p-1 border border-slate-200 mb-6">
          <button
            onClick={() => setMode("login")}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              mode === "login"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Đăng Nhập
          </button>
          <button
            onClick={() => setMode("register")}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
              mode === "register"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Đăng Ký
          </button>
        </div>

        {/* FORM */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === "register" && (
            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Họ và Tên
              </label>
              <div className="relative">
                <User className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Nguyễn Văn A"
                  className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
                />
              </div>
            </div>
          )}

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Email</label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Mật khẩu</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Tối thiểu 8 ký tự"
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl font-bold text-sm bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-all disabled:opacity-50"
          >
            {loading ? (
              <span>Đang xử lý...</span>
            ) : mode === "login" ? (
              <span>Đăng Nhập</span>
            ) : (
              <span>Tạo Tài Khoản</span>
            )}
          </button>
        </form>

        {/* QUICK DEMO CREDENTIALS AUTOFILL */}
        {mode === "login" && (
          <div className="mt-6 pt-5 border-t border-slate-100 text-center">
            <span className="text-[11px] text-slate-500 font-medium block mb-2.5">
              Tài khoản mẫu có sẵn trong PostgreSQL:
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={fillDemoUser}
                className="flex-1 py-1.5 px-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-[11px] text-emerald-700 font-semibold flex items-center justify-center gap-1.5 shadow-2xs transition-colors"
              >
                <UserCheck className="w-3.5 h-3.5 text-emerald-600" />
                <span>Demo User</span>
              </button>
              <button
                type="button"
                onClick={fillAdminUser}
                className="flex-1 py-1.5 px-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-[11px] text-amber-700 font-semibold flex items-center justify-center gap-1.5 shadow-2xs transition-colors"
              >
                <Shield className="w-3.5 h-3.5 text-amber-600" />
                <span>Admin User</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
