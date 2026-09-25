"use client";

import React, { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { 
  Bot, 
  Send, 
  Sparkles, 
  X, 
  ShoppingBag
} from "lucide-react";
import { ChatMessage, Product } from "@/lib/types";
import { formatVND } from "@/lib/api";
import { useCart } from "@/lib/context";

interface AiChatAssistantProps {
  products: Product[];
  onSelectProduct?: (p: Product) => void;
  isFloating?: boolean;
}

export function AiChatAssistant({
  products,
  onSelectProduct,
  isFloating = false,
}: AiChatAssistantProps) {
  const { addItem } = useCart();
  const [isOpen, setIsOpen] = useState(!isFloating);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome-1",
      sender: "assistant",
      content:
        "Xin chào! Tôi là Trợ Lý Mua Sắm AI của ShopSense. Tôi có thể giúp bạn tìm kiếm trang phục, gợi ý phối đồ theo phong cách hoặc giải thích các sản phẩm phù hợp nhất trong kho dữ liệu thời trang.",
      timestamp: "Vừa xong",
    },
  ]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async (customPrompt?: string) => {
    const textToSend = customPrompt || input.trim();
    if (!textToSend) return;

    const userMsg: ChatMessage = {
      id: Math.random().toString(),
      sender: "user",
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!customPrompt) setInput("");
    setIsTyping(true);

    // AI response generator based on catalog data and query intent
    setTimeout(() => {
      const q = textToSend.toLowerCase();
      let reply = "";
      let matchedProds: Product[] = [];

      if (q.includes("công sở") || q.includes("nam") || q.includes("sơ mi") || q.includes("blazer")) {
        matchedProds = products.filter(
          (p) => p.category_id === 1 || p.name.toLowerCase().includes("sơ mi") || p.name.toLowerCase().includes("blazer")
        );
        reply =
          "Dựa trên yêu cầu của bạn, tôi gợi ý phong cách công sở lịch lãm với sự kết hợp giữa Áo Sơ Mi Oxford Slim Fit và Áo Blazer. Đây là các sản phẩm có chất liệu cotton thoáng mát và đứng form chuẩn.";
      } else if (q.includes("giày") || q.includes("sneaker") || q.includes("thể thao")) {
        matchedProds = products.filter((p) => p.category_id === 3 || p.category_id === 7);
        reply =
          "Về giày dép thể thao, bạn có thể tham khảo các mẫu Sneaker chạy bộ đệm khí bảo vệ khớp chân, hoặc đôi Sneaker White Classic cực kỳ dễ phối cùng mọi loại trang phục.";
      } else if (q.includes("nữ") || q.includes("váy") || q.includes("đầm") || q.includes("vest")) {
        matchedProds = products.filter((p) => p.category_id === 2);
        reply =
          "Bộ sưu tập thời trang nữ hiện có các mẫu Váy Xòe và Đầm Voan Vintage đang được đánh giá rất cao về độ tôn dáng và chất liệu vải nhập khẩu cao cấp.";
      } else if (q.includes("túi") || q.includes("balo") || q.includes("laptop")) {
        matchedProds = products.filter((p) => p.category_id === 4);
        reply =
          "Dành cho nhu cầu đi làm hoặc du lịch: Túi Tote Da Thật và Balo Chống Thấm Nước tích hợp ngăn chống sốc tiện lợi.";
      } else if (q.includes("đồng hồ") || q.includes("nhẫn") || q.includes("trang sức")) {
        matchedProds = products.filter((p) => p.category_id === 5);
        reply =
          "Các phụ kiện cao cấp đang sẵn sàng gồm Đồng Hồ Chronograph chống nước và Trang Sức Bạc Đính Đá tinh xảo, rất thích hợp làm quà tặng hoặc diện đi tiệc.";
      } else {
        matchedProds = products.slice(0, 3);
        reply = `Tôi đã tìm kiếm trong kho dữ liệu và chọn ra những sản phẩm nổi bật nhất phù hợp với từ khóa "${textToSend}". Bạn có thể xem chi tiết bên dưới:`;
      }

      const botMsg: ChatMessage = {
        id: Math.random().toString(),
        sender: "assistant",
        content: reply,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        products: matchedProds.slice(0, 3),
      };

      setMessages((prev) => [...prev, botMsg]);
      setIsTyping(false);
    }, 600);
  };

  const quickPrompts = [
    "Gợi ý trang phục công sở nam",
    "Tìm giày sneaker phối đồ đẹp",
    "Túi đựng laptop đi làm",
    "Váy đầm nữ thanh lịch",
  ];

  if (isFloating && !isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-40 p-4 rounded-2xl bg-slate-900 hover:bg-slate-800 text-white shadow-2xl hover:scale-105 transition-all flex items-center gap-2.5 font-bold text-sm border border-slate-700/60 glow-emerald"
      >
        <div className="w-7 h-7 rounded-lg bg-emerald-600 flex items-center justify-center text-white">
          <Bot className="w-4 h-4" />
        </div>
        <span className="hidden sm:inline">Trợ Lý AI ShopSense</span>
      </button>
    );
  }

  const containerClasses = isFloating
    ? "fixed bottom-6 right-6 z-40 w-[94vw] sm:w-[440px] h-[600px] max-h-[85vh] bg-white rounded-3xl border border-slate-200 shadow-2xl flex flex-col overflow-hidden animate-slide-up"
    : "w-full h-full min-h-[600px] bg-white rounded-3xl border border-slate-200 flex flex-col overflow-hidden shadow-xs";

  return (
    <div className={containerClasses}>
      {/* HEADER */}
      <div className="p-4 sm:p-5 border-b border-slate-100 bg-slate-50/80 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 flex items-center justify-center text-white shadow-xs">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-bold text-sm sm:text-base text-slate-900 flex items-center gap-1.5">
              <span>Trợ Lý Mua Sắm AI</span>
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
            </h4>
            <span className="text-[11px] text-emerald-700 font-medium flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Sẵn sàng giải đáp & gợi ý
            </span>
          </div>
        </div>

        {isFloating && (
          <button
            onClick={() => setIsOpen(false)}
            className="p-2 rounded-xl bg-slate-100 text-slate-500 hover:text-slate-900 hover:bg-slate-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* CHAT MESSAGES */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.sender === "user" ? "items-end" : "items-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-xs sm:text-sm leading-relaxed ${
                msg.sender === "user"
                  ? "bg-slate-900 text-white font-medium rounded-tr-xs shadow-xs"
                  : "bg-slate-100 text-slate-800 border border-slate-200/80 rounded-tl-xs shadow-2xs"
              }`}
            >
              {msg.content}
            </div>

            {/* EMBEDDED PRODUCT CARDS IN BOT MESSAGE */}
            {msg.products && msg.products.length > 0 && (
              <div className="mt-3 w-full grid grid-cols-1 gap-2.5">
                {msg.products.map((p) => (
                  <div
                    key={p.id}
                    onClick={() => onSelectProduct && onSelectProduct(p)}
                    className="p-2.5 rounded-xl bg-slate-50 hover:bg-white border border-slate-200 hover:border-emerald-500/50 cursor-pointer transition-all flex items-center justify-between gap-3 group shadow-2xs hover:shadow-xs"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className="w-12 h-12 rounded-lg bg-white border border-slate-200/80 flex-shrink-0 overflow-hidden relative">
                        {p.image_url ? (
                          <Image
                            src={p.image_url}
                            alt={p.name}
                            fill
                            className="object-cover group-hover:scale-105 transition-transform"
                          />
                        ) : null}
                      </div>
                      <div className="min-w-0">
                        <h6 className="text-xs font-semibold text-slate-800 truncate group-hover:text-emerald-700">
                          {p.name}
                        </h6>
                        <span className="text-xs font-bold text-emerald-600">
                          {formatVND(Number(p.price))}
                        </span>
                      </div>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        addItem(p.id, 1);
                      }}
                      className="px-2.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-[11px] font-bold text-white flex-shrink-0 flex items-center gap-1 shadow-2xs transition-colors"
                    >
                      <ShoppingBag className="w-3 h-3" />
                      <span>Thêm</span>
                    </button>
                  </div>
                ))}
              </div>
            )}

            <span className="text-[10px] text-slate-400 mt-1 px-1">
              {msg.timestamp}
            </span>
          </div>
        ))}

        {isTyping && (
          <div className="flex items-center gap-2 text-xs text-slate-500 p-2">
            <Bot className="w-4 h-4 text-emerald-600 animate-spin" />
            <span>ShopSense AI đang suy nghĩ...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* QUICK PROMPT CHIPS */}
      <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50/70 flex items-center gap-2 overflow-x-auto text-[11px] no-scrollbar">
        {quickPrompts.map((p) => (
          <button
            key={p}
            onClick={() => handleSend(p)}
            className="px-3 py-1 rounded-full bg-white hover:bg-slate-100 text-slate-600 hover:text-slate-900 whitespace-nowrap border border-slate-200 shadow-2xs transition-colors font-medium"
          >
            {p}
          </button>
        ))}
      </div>

      {/* INPUT BAR */}
      <div className="p-3 sm:p-4 border-t border-slate-100 bg-white">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="relative flex items-center"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Hỏi về sản phẩm, phối đồ hoặc gợi ý..."
            className="w-full pl-4 pr-12 py-3 rounded-2xl bg-slate-50 border border-slate-200 text-xs sm:text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="absolute right-2 p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white disabled:opacity-30 transition-all shadow-xs"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
