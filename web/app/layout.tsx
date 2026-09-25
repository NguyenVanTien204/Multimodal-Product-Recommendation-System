import type { Metadata } from "next";
import "./globals.css";
import { AppProvider } from "@/lib/context";
import { LayoutWrapper } from "@/components/layout-wrapper";

export const metadata: Metadata = {
  title: "ShopSense — Sàn Thương Mại Điện Tử & Gợi Ý AI Đa Phương Thức",
  description:
    "ShopSense: Nền tảng thương mại điện tử tích hợp hệ sinh thái gợi ý tuần tự (Sequential Recommender), tìm kiếm vector Qdrant và trợ lý ảo AI mua sắm.",
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
