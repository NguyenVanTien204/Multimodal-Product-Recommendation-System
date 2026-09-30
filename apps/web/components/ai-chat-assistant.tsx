"use client";

import React, { useState, useRef, useEffect } from "react";
import Image from "next/image";
import {
  Bot,
  Send,
  Sparkles,
  X,
  ShoppingBag,
  ImagePlus,
  Star,
  Scale,
  MessageSquareQuote,
  RotateCcw,
} from "lucide-react";
import { ChatMessage, ChatProduct, ChatRequestPayload, Product } from "@/lib/types";
import { formatVND, sendChat } from "@/lib/api";
import { useCart } from "@/lib/context";

interface AiChatAssistantProps {
  products?: Product[];
  onSelectProduct?: (p: Product) => void;
  isFloating?: boolean;
}

const WELCOME: ChatMessage = {
  id: "welcome-1",
  sender: "assistant",
  content:
    "Xin chào! Mình là trợ lý mua sắm ShopSense. Bạn có thể mô tả món đồ cần tìm hoặc tải lên một bức ảnh, mình sẽ tìm sản phẩm giống/phù hợp, giải thích lý do dựa trên đánh giá của người mua và so sánh giúp bạn.",
  timestamp: "Vừa xong",
  suggestions: [
    "Giày chạy bộ nam màu đen dưới 500k",
    "Áo sơ mi trắng công sở",
    "Túi xách nữ da thật",
    "Gợi ý cho tôi",
  ],
};

const nowLabel = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

/** Downscale + JPEG-encode an uploaded photo so the request stays small (max 1024px side). */
async function fileToBase64(file: File): Promise<{ dataUrl: string; preview: string }> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
  return { dataUrl, preview: dataUrl };
}

/** Minimal inline formatting: **bold** and [P1]/[R1.2] citation markers. */
function renderText(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[(?:P\d+|R\d+\.\d+)\])/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (/^\[(P\d+|R\d+\.\d+)\]$/.test(part))
      return (
        <sup key={i} className="mx-0.5 px-1 rounded bg-emerald-100 text-emerald-700 text-[9px] font-bold align-super">
          {part.slice(1, -1)}
        </sup>
      );
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

function EvidenceList({ item }: { item: ChatProduct }) {
  const [open, setOpen] = useState(false);
  if (!item.evidence.length) return null;
  return (
    <div className="mt-1.5">
      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="text-[10px] font-semibold text-emerald-700 hover:underline flex items-center gap-1"
      >
        <MessageSquareQuote className="w-3 h-3" />
        {open ? "Ẩn" : "Xem"} nhận xét người mua ({item.evidence.length})
      </button>
      {open && (
        <ul className="mt-1 space-y-1">
          {item.evidence.map((e) => (
            <li key={e.review_id} className="text-[11px] text-slate-600 bg-white border border-slate-200 rounded-lg px-2 py-1.5">
              <span className="font-semibold text-amber-600">{e.rating.toFixed(0)}★</span>{" "}
              {e.tag && <span className="text-emerald-700 font-bold">[{e.tag}] </span>}
              {e.title ? <span className="font-semibold">{e.title} — </span> : null}“{e.text}”
              {e.helpful_vote > 0 && <span className="text-slate-400"> · {e.helpful_vote} hữu ích</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function AiChatAssistant({ onSelectProduct, isFloating = false }: AiChatAssistantProps) {
  const { addItem } = useCart();
  const [isOpen, setIsOpen] = useState(!isFloating);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pendingImage, setPendingImage] = useState<{ dataUrl: string; preview: string } | null>(null);
  const [compareSel, setCompareSel] = useState<Record<string, number[]>>({});
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const send = async (payload: ChatRequestPayload, userLabel: string, preview?: string) => {
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), sender: "user", content: userLabel, timestamp: nowLabel(), imagePreview: preview },
    ]);
    setIsTyping(true);
    try {
      const data = await sendChat({ ...payload, session_id: sessionId });
      setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          sender: "assistant",
          content: data.reply,
          timestamp: nowLabel(),
          chatProducts: data.products,
          filterChips: data.filter_chips,
          suggestions: data.suggestions,
          meta: data.meta,
          warnings: data.warnings,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          sender: "assistant",
          isError: true,
          content: `Không kết nối được trợ lý (${err instanceof Error ? err.message : "lỗi không xác định"}). Bạn thử lại sau nhé.`,
          timestamp: nowLabel(),
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSend = async (customPrompt?: string) => {
    const text = (customPrompt ?? input).trim();
    if ((!text && !pendingImage) || isTyping) return;
    const image = pendingImage;
    setInput("");
    setPendingImage(null);
    await send({ message: text, image_base64: image?.dataUrl ?? null }, text || "🖼️ Tìm sản phẩm giống ảnh này", image?.preview);
  };

  const handleFile = async (file?: File) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return;
    try {
      setPendingImage(await fileToBase64(file));
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), sender: "assistant", isError: true, content: "Không đọc được ảnh này, bạn thử ảnh khác nhé.", timestamp: nowLabel() },
      ]);
    }
  };

  const toggleCompare = (msgId: string, productId: number) =>
    setCompareSel((prev) => {
      const cur = prev[msgId] ?? [];
      return { ...prev, [msgId]: cur.includes(productId) ? cur.filter((id) => id !== productId) : [...cur, productId].slice(-4) };
    });

  const resetChat = () => {
    setSessionId(null);
    setMessages([WELCOME]);
    setCompareSel({});
    setPendingImage(null);
  };

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
    ? "fixed bottom-6 right-6 z-40 w-[94vw] sm:w-[480px] h-[640px] max-h-[88vh] bg-white rounded-3xl border border-slate-200 shadow-2xl flex flex-col overflow-hidden animate-slide-up"
    : "w-full h-full min-h-[600px] bg-white rounded-3xl border border-slate-200 flex flex-col overflow-hidden shadow-xs";

  const lastSuggestions = [...messages].reverse().find((m) => m.sender === "assistant" && m.suggestions?.length)?.suggestions ?? [];

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
              Tìm kiếm đa phương thức · RAG từ đánh giá thật
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={resetChat}
            title="Cuộc trò chuyện mới"
            className="p-2 rounded-xl bg-slate-100 text-slate-500 hover:text-slate-900 hover:bg-slate-200 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          {isFloating && (
            <button
              onClick={() => setIsOpen(false)}
              className="p-2 rounded-xl bg-slate-100 text-slate-500 hover:text-slate-900 hover:bg-slate-200 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* CHAT MESSAGES */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
        {messages.map((msg) => {
          const selected = compareSel[msg.id] ?? [];
          const sourceLabel =
            msg.meta?.answer_source === "llm"
              ? "Sinh bởi LLM, đã kiểm chứng nguồn"
              : msg.meta?.answer_source === "template"
                ? "Trả lời dựng trực tiếp từ dữ liệu"
                : msg.meta?.answer_source === "keyword_fallback"
                  ? "Chế độ dự phòng (tìm theo từ khóa)"
                  : null;
          return (
            <div key={msg.id} className={`flex flex-col ${msg.sender === "user" ? "items-end" : "items-start"}`}>
              {msg.imagePreview && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={msg.imagePreview} alt="Ảnh đã gửi" className="mb-1.5 w-28 h-28 rounded-xl object-cover border border-slate-200" />
              )}
              <div
                className={`max-w-[92%] rounded-2xl px-4 py-3 text-xs sm:text-sm leading-relaxed whitespace-pre-wrap ${
                  msg.sender === "user"
                    ? "bg-slate-900 text-white font-medium rounded-tr-xs shadow-xs"
                    : msg.isError
                      ? "bg-rose-50 text-rose-800 border border-rose-200 rounded-tl-xs"
                      : "bg-slate-100 text-slate-800 border border-slate-200/80 rounded-tl-xs shadow-2xs"
                }`}
              >
                {msg.sender === "assistant" ? renderText(msg.content) : msg.content}
              </div>

              {msg.filterChips && msg.filterChips.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {msg.filterChips.map((c) => (
                    <span key={c} className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px] font-semibold">
                      {c}
                    </span>
                  ))}
                </div>
              )}

              {/* PRODUCT CARDS */}
              {msg.chatProducts && msg.chatProducts.length > 0 && (
                <div className="mt-3 w-full grid grid-cols-1 gap-2.5">
                  {msg.chatProducts.map((item, idx) => {
                    const p = item.product;
                    return (
                      <div
                        key={p.id}
                        onClick={() => onSelectProduct?.(p)}
                        className="p-2.5 rounded-xl bg-slate-50 hover:bg-white border border-slate-200 hover:border-emerald-500/50 cursor-pointer transition-all group shadow-2xs hover:shadow-xs"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-start gap-2.5 min-w-0">
                            <div className="w-14 h-14 rounded-lg bg-white border border-slate-200/80 flex-shrink-0 overflow-hidden relative">
                              {p.image_url ? (
                                <Image src={p.image_url} alt={p.name} fill sizes="56px" className="object-cover group-hover:scale-105 transition-transform" />
                              ) : null}
                              <span className="absolute top-0 left-0 bg-slate-900/80 text-white text-[9px] font-bold px-1 rounded-br">
                                {idx + 1}
                              </span>
                            </div>
                            <div className="min-w-0">
                              <h6 className="text-xs font-semibold text-slate-800 line-clamp-2 group-hover:text-emerald-700">{p.name}</h6>
                              <div className="flex items-center flex-wrap gap-x-2 gap-y-0.5 mt-0.5">
                                <span className="text-xs font-bold text-emerald-600">{formatVND(Number(p.price))}</span>
                                {item.price_estimated && (
                                  <span title="Sản phẩm chưa có giá gốc, giá hiển thị là giá tham khảo" className="text-[9px] text-amber-700 bg-amber-50 border border-amber-200 rounded px-1">
                                    giá tham khảo
                                  </span>
                                )}
                                {item.brand && <span className="text-[10px] text-slate-500">{item.brand}</span>}
                                {item.avg_rating != null && item.review_count > 0 && (
                                  <span className="text-[10px] text-amber-600 flex items-center gap-0.5 font-semibold">
                                    <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                                    {item.avg_rating.toFixed(1)} ({item.review_count})
                                  </span>
                                )}
                              </div>
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

                        {item.reasons.length > 0 && (
                          <ul className="mt-1.5 space-y-0.5">
                            {item.reasons.slice(0, 3).map((r) => (
                              <li key={r} className="text-[10px] text-slate-500 leading-snug">
                                • {r}
                              </li>
                            ))}
                          </ul>
                        )}
                        <EvidenceList item={item} />

                        <div className="mt-2 flex items-center gap-1.5">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              send({ message: `Tại sao nên chọn sản phẩm ${idx + 1}?`, action: { type: "explain", product_id: p.id } }, `Giải thích sản phẩm ${idx + 1}`);
                            }}
                            className="px-2 py-1 rounded-md bg-white border border-slate-200 hover:border-emerald-500 text-[10px] font-semibold text-slate-600 hover:text-emerald-700 transition-colors"
                          >
                            Vì sao chọn?
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              send({ message: "", action: { type: "similar", product_id: p.id } }, `Tìm sản phẩm giống ${idx + 1}`);
                            }}
                            className="px-2 py-1 rounded-md bg-white border border-slate-200 hover:border-emerald-500 text-[10px] font-semibold text-slate-600 hover:text-emerald-700 transition-colors"
                          >
                            Tương tự
                          </button>
                          {msg.chatProducts!.length > 1 && (
                            <label
                              onClick={(e) => e.stopPropagation()}
                              className="ml-auto flex items-center gap-1 text-[10px] font-semibold text-slate-500 cursor-pointer select-none"
                            >
                              <input
                                type="checkbox"
                                checked={selected.includes(p.id)}
                                onChange={() => toggleCompare(msg.id, p.id)}
                                className="accent-emerald-600"
                              />
                              So sánh
                            </label>
                          )}
                        </div>
                      </div>
                    );
                  })}
                  {selected.length >= 2 && (
                    <button
                      onClick={() => send({ message: "So sánh các sản phẩm đã chọn", action: { type: "compare", product_ids: selected } }, `So sánh ${selected.length} sản phẩm`)}
                      className="self-start px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-[11px] font-bold flex items-center gap-1.5"
                    >
                      <Scale className="w-3.5 h-3.5" />
                      So sánh {selected.length} sản phẩm
                    </button>
                  )}
                </div>
              )}

              <span className="text-[10px] text-slate-400 mt-1 px-1 flex items-center gap-1.5">
                {msg.timestamp}
                {sourceLabel && <span className="text-slate-400">· {sourceLabel}</span>}
                {msg.meta?.personalized === true && <span className="text-emerald-600 font-semibold">· cá nhân hóa</span>}
              </span>
            </div>
          );
        })}

        {isTyping && (
          <div className="flex items-center gap-2 text-xs text-slate-500 p-2">
            <Bot className="w-4 h-4 text-emerald-600 animate-spin" />
            <span>ShopSense AI đang tìm kiếm & đọc đánh giá...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* SUGGESTION CHIPS */}
      {lastSuggestions.length > 0 && (
        <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50/70 flex items-center gap-2 overflow-x-auto text-[11px] no-scrollbar">
          {lastSuggestions.map((p) => (
            <button
              key={p}
              onClick={() => handleSend(p)}
              disabled={isTyping}
              className="px-3 py-1 rounded-full bg-white hover:bg-slate-100 text-slate-600 hover:text-slate-900 whitespace-nowrap border border-slate-200 shadow-2xs transition-colors font-medium disabled:opacity-40"
            >
              {p}
            </button>
          ))}
        </div>
      )}

      {/* INPUT BAR */}
      <div className="p-3 sm:p-4 border-t border-slate-100 bg-white">
        {pendingImage && (
          <div className="mb-2 flex items-center gap-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={pendingImage.preview} alt="Ảnh sẽ gửi" className="w-12 h-12 rounded-lg object-cover border border-slate-200" />
            <span className="text-[11px] text-slate-500">Ảnh sẽ được dùng để tìm sản phẩm giống</span>
            <button onClick={() => setPendingImage(null)} className="ml-auto text-slate-400 hover:text-slate-700">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="relative flex items-center gap-2"
        >
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => handleFile(e.target.files?.[0])} />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            title="Tìm bằng ảnh"
            className="p-3 rounded-2xl bg-slate-50 border border-slate-200 text-slate-500 hover:text-emerald-700 hover:border-emerald-500 transition-colors"
          >
            <ImagePlus className="w-4 h-4" />
          </button>
          <div className="relative flex-1">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Mô tả món đồ, hoặc hỏi “rẻ hơn”, “tại sao sản phẩm 2?”..."
              className="w-full pl-4 pr-12 py-3 rounded-2xl bg-slate-50 border border-slate-200 text-xs sm:text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10 transition-all"
            />
            <button
              type="submit"
              disabled={(!input.trim() && !pendingImage) || isTyping}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white disabled:opacity-30 transition-all shadow-xs"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
