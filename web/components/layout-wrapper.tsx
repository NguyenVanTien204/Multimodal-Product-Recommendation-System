"use client";

import React, { useState } from "react";
import { Navbar } from "./navbar";
import { CartDrawer } from "./cart-drawer";
import { AuthModal } from "./auth-modal";
import { Footer } from "./footer";

export function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const [authModalOpen, setAuthModalOpen] = useState(false);

  return (
    <>
      <Navbar onOpenAuth={() => setAuthModalOpen(true)} />
      <main className="flex-1 flex flex-col">{children}</main>
      <CartDrawer />
      <AuthModal isOpen={authModalOpen} onClose={() => setAuthModalOpen(false)} />
      <Footer />
    </>
  );
}
