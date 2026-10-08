import { Product } from "@/lib/types";

/** Who a product is for (H&M `audience`, derived from the section name). */
export const AUDIENCES: { key: string; label: string }[] = [
  { key: "women", label: "Nữ" },
  { key: "men", label: "Nam" },
  { key: "divided", label: "Teen" },
  { key: "kids", label: "Trẻ em" },
  { key: "baby", label: "Em bé" },
];

export const audienceLabel = (key?: string | null) =>
  AUDIENCES.find((a) => a.key === key)?.label ?? (key === "other" ? "Khác" : key ?? "");

/** "Vest top · Black" -- what replaces the brand line of the Amazon catalog. */
export const productSubtitle = (p: Pick<Product, "attributes">) =>
  [p.attributes?.product_type, p.attributes?.colour].filter(Boolean).join(" · ");

/** Product photos served by our own backend are already 512 px JPEGs: skip the Next image optimizer. */
export const isLocalImage = (src?: string | null) => !!src && src.startsWith("/api/");
