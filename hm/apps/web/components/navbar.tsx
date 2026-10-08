"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  ShoppingBag, 
  Sparkles, 
  Scale, 
  Package, 
  User as UserIcon, 
  LogOut, 
  ShieldCheck, 
  Menu, 
  X 
} from "lucide-react";
import { useAuth, useCart, useCompare } from "@/lib/context";

export function Navbar({ onOpenAuth }: { onOpenAuth: () => void }) {
  const pathname = usePathname();
  const { user, logout } = useAuth();
  const { itemCount, setIsDrawerOpen } = useCart();
  const { compareItems } = useCompare();

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: "/", label: "Khám Phá", icon: ShoppingBag },
    { href: "/compare", label: "So Sánh", icon: Scale, badge: compareItems.length },
    { href: "/orders", label: "Đơn Hàng", icon: Package },
    { href: "/assistant", label: "Tư Vấn Phong Cách", icon: Sparkles, isHighlight: true },
  ];

  if (user?.is_admin) {
    navLinks.push({ href: "/admin", label: "Quản Trị", icon: ShieldCheck });
  }

  return (
    <header className="sticky top-0 z-40 w-full bg-white/90 backdrop-blur-md border-b border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* LOGO */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-9 h-9 rounded-xl bg-slate-950 flex items-center justify-center shadow-sm group-hover:bg-emerald-600 transition-colors duration-200">
            <span className="font-extrabold text-white text-base tracking-tighter">S</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-extrabold text-xl tracking-tight text-slate-950 group-hover:text-emerald-700 transition-colors">
              ShopSense
            </span>
          </div>
        </Link>

        {/* NAVIGATION DESKTOP */}
        <nav className="hidden md:flex items-center gap-1 lg:gap-1.5">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`relative px-3.5 py-2 rounded-xl text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
                  isActive
                    ? "bg-slate-100 text-slate-950 font-semibold shadow-xs"
                    : link.isHighlight
                    ? "text-emerald-700 hover:text-emerald-800 hover:bg-emerald-50/80 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/70"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-emerald-600" : ""}`} />
                <span>{link.label}</span>
                {link.badge !== undefined && link.badge > 0 && (
                  <span className="ml-1 px-1.5 py-0.2 text-[10px] font-bold rounded-full bg-emerald-600 text-white shadow-xs">
                    {link.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* RIGHT ACTIONS: CART, AUTH */}
        <div className="flex items-center gap-2.5">

          {/* CART BUTTON */}
          <button
            onClick={() => setIsDrawerOpen(true)}
            className="relative p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200/80 border border-slate-200/80 text-slate-700 hover:text-slate-950 transition-colors"
            aria-label="Xem giỏ hàng"
          >
            <ShoppingBag className="w-4 h-4 sm:w-5 sm:h-5" />
            {itemCount > 0 && (
              <span className="absolute -top-1.5 -right-1.5 bg-emerald-600 text-white text-[11px] font-bold w-5 h-5 rounded-full flex items-center justify-center shadow-sm animate-fade-in">
                {itemCount}
              </span>
            )}
          </button>

          {/* AUTH BUTTON / USER MENU */}
          {user ? (
            <div className="flex items-center gap-2 pl-2 border-l border-slate-200">
              <Link
                href="/account"
                title="Tài khoản của tôi"
                className={`hidden sm:flex flex-col text-right px-2 py-1 rounded-lg transition-colors ${
                  pathname === "/account" ? "bg-slate-100" : "hover:bg-slate-100"
                }`}
              >
                <span className="text-xs font-bold text-slate-900 leading-tight">
                  {user.full_name}
                </span>
                <span className="text-[10px] text-emerald-600 font-semibold">
                  {user.is_admin ? "Quản Trị Viên" : "Thành viên"}
                </span>
              </Link>
              <Link
                href="/account"
                title="Tài khoản của tôi"
                className="sm:hidden p-2 rounded-xl bg-slate-100 hover:bg-slate-200/80 border border-slate-200/80 text-slate-700 transition-colors"
              >
                <UserIcon className="w-4 h-4" />
              </Link>
              <button
                onClick={logout}
                title="Đăng xuất"
                className="p-2 rounded-xl bg-slate-100 hover:bg-rose-50 hover:text-rose-600 border border-slate-200 text-slate-500 transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={onOpenAuth}
              className="px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold bg-slate-900 hover:bg-slate-800 text-white shadow-xs hover:shadow-sm transition-all flex items-center gap-1.5"
            >
              <UserIcon className="w-4 h-4 text-slate-200" />
              <span>Đăng Nhập</span>
            </button>
          )}

          {/* MOBILE MENU TOGGLE */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-xl bg-slate-100 border border-slate-200 text-slate-700"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* MOBILE MENU DROPDOWN */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white/95 backdrop-blur-md border-b border-slate-200 px-4 pt-2 pb-4 space-y-1 shadow-lg">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium ${
                  isActive
                    ? "bg-slate-100 text-slate-950 font-semibold"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className="w-4 h-4" />
                  <span>{link.label}</span>
                </div>
                {link.badge !== undefined && link.badge > 0 && (
                  <span className="px-2 py-0.5 text-xs font-bold rounded-full bg-emerald-600 text-white">
                    {link.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
}
