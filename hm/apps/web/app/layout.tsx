import type { Metadata } from "next";
import "./globals.css";
import { AppProvider } from "@/lib/context";
import { LayoutWrapper } from "@/components/layout-wrapper";

export const metadata: Metadata = {
  title: "ShopSense — Nền Tảng Thời Trang & Mua Sắm Cá Nhân Hóa",
  description:
    "ShopSense: Khám phá hàng chục ngàn mẫu thời trang đa phong cách cùng trải nghiệm mua sắm thông minh, gợi ý chuẩn gu và dịch vụ tận tâm.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi" className="light">
      <body className="bg-slate-50/70 text-slate-900 antialiased selection:bg-emerald-500 selection:text-white min-h-screen flex flex-col font-sans">
        <AppProvider>
          <LayoutWrapper>{children}</LayoutWrapper>
        </AppProvider>
      </body>
    </html>
  );
}
