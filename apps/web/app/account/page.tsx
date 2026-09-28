"use client";

import React, { useEffect, useState } from "react";
import { User as UserIcon, Lock, Mail, ShieldCheck, Save, KeyRound } from "lucide-react";
import { useAuth } from "@/lib/context";

export default function AccountPage() {
  const { user, loading, updateProfile, changePassword } = useAuth();

  const [fullName, setFullName] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [savingPassword, setSavingPassword] = useState(false);

  useEffect(() => {
    if (user) setFullName(user.full_name);
  }, [user]);

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingProfile(true);
    try {
      await updateProfile(fullName);
    } catch {
      // toast shown in context
    } finally {
      setSavingProfile(false);
    }
  };

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      return;
    }
    setSavingPassword(true);
    try {
      await changePassword(currentPassword, newPassword);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch {
      // toast shown in context
    } finally {
      setSavingPassword(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-24 text-center text-sm text-slate-500">
        Đang tải...
      </div>
    );
  }

  if (!user) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-24 text-center">
        <div className="w-16 h-16 rounded-3xl bg-white border border-slate-200 text-slate-400 flex items-center justify-center mx-auto mb-4 shadow-xs">
          <UserIcon className="w-8 h-8" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-2">Đăng Nhập Để Xem Tài Khoản</h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto">
          Vui lòng đăng nhập để xem và chỉnh sửa thông tin cá nhân.
        </p>
      </div>
    );
  }

  const passwordMismatch = confirmPassword.length > 0 && newPassword !== confirmPassword;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* HEADER */}
      <div className="pb-6 border-b border-slate-200">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 flex items-center gap-2.5">
          <UserIcon className="w-7 h-7 text-emerald-600" />
          <span>Tài Khoản Của Tôi</span>
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Quản lý thông tin cá nhân và bảo mật đăng nhập
        </p>
      </div>

      {/* PROFILE CARD */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs">
        <div className="flex items-center gap-2 mb-5">
          <UserIcon className="w-4 h-4 text-emerald-600" />
          <h2 className="text-sm font-bold text-slate-900">Thông Tin Cá Nhân</h2>
        </div>

        <form onSubmit={handleProfileSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Email</label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="email"
                disabled
                value={user.email}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-500 text-xs sm:text-sm cursor-not-allowed"
              />
            </div>
            <p className="text-[11px] text-slate-400 mt-1">Email không thể thay đổi.</p>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Họ và Tên</label>
            <div className="relative">
              <UserIcon className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="text"
                required
                minLength={2}
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Vai Trò</label>
            <div className="flex items-center gap-1.5 text-xs">
              <ShieldCheck className={`w-3.5 h-3.5 ${user.is_admin ? "text-amber-600" : "text-emerald-600"}`} />
              <span className={`font-semibold ${user.is_admin ? "text-amber-700" : "text-emerald-700"}`}>
                {user.is_admin ? "Quản Trị Viên" : "Thành viên"}
              </span>
            </div>
          </div>

          <button
            type="submit"
            disabled={savingProfile || fullName.trim().length < 2}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-xs bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-all disabled:opacity-50"
          >
            <Save className="w-3.5 h-3.5" />
            <span>{savingProfile ? "Đang lưu..." : "Lưu Thay Đổi"}</span>
          </button>
        </form>
      </div>

      {/* CHANGE PASSWORD CARD */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs">
        <div className="flex items-center gap-2 mb-5">
          <KeyRound className="w-4 h-4 text-emerald-600" />
          <h2 className="text-sm font-bold text-slate-900">Đổi Mật Khẩu</h2>
        </div>

        <form onSubmit={handlePasswordSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Mật Khẩu Hiện Tại</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="password"
                required
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Mật Khẩu Mới</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="password"
                required
                minLength={8}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="Tối thiểu 8 ký tự"
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Xác Nhận Mật Khẩu Mới</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="password"
                required
                minLength={8}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className={`w-full pl-10 pr-3.5 py-2.5 rounded-xl bg-slate-50 border text-slate-900 text-xs sm:text-sm focus:outline-none focus:bg-white transition-all ${
                  passwordMismatch
                    ? "border-rose-300 focus:border-rose-500 focus:ring-4 focus:ring-rose-500/10"
                    : "border-slate-200 focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10"
                }`}
              />
            </div>
            {passwordMismatch && (
              <p className="text-[11px] text-rose-600 mt-1">Mật khẩu xác nhận không khớp.</p>
            )}
          </div>

          <button
            type="submit"
            disabled={savingPassword || passwordMismatch || newPassword.length < 8}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-xs bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-all disabled:opacity-50"
          >
            <KeyRound className="w-3.5 h-3.5" />
            <span>{savingPassword ? "Đang xử lý..." : "Đổi Mật Khẩu"}</span>
          </button>
        </form>
      </div>
    </div>
  );
}
