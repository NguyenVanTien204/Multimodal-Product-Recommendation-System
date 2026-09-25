# UML cho web demo

Các biểu đồ dùng Mermaid để render trực tiếp trong Markdown và dễ đồng bộ với tài liệu luận văn.

## Use-case diagram

```mermaid
flowchart LR
  Shopper([Người mua demo])
  Researcher([Nghiên cứu viên])
  Web[Web storefront]
  Rec[Recommender API]
  Agent[Agent + RAG API]

  Shopper -->|Tìm / lọc sản phẩm| Web
  Shopper -->|Nhận gợi ý| Web
  Shopper -->|Tinh chỉnh theo phiên| Web
  Shopper -->|Chat, giải thích, so sánh| Web
  Researcher -->|Xem trạng thái model| Web
  Web --> Rec
  Web --> Agent
```

## Sequence — refinement tuần tự

```mermaid
sequenceDiagram
  actor U as Người mua
  participant W as Web
  participant A as FastAPI
  participant S as Session manager
  participant R as Recommender/Reranker

  U->>W: “Tìm giày đen dưới $100”
  W->>A: POST /search(query, filters, session_id)
  A->>S: Lấy/khởi tạo preference
  A->>R: Retrieval + hard filter + ranking
  R-->>A: Top-K + metadata + source
  A->>S: Lưu last_result_item_ids
  A-->>W: products + understood_preferences
  U->>W: “Ít thể thao hơn”
  W->>A: POST /refine(session_id, preference_delta)
  A->>S: Merge preference
  A->>R: Rerank candidates / retrieve mới
  A-->>W: Top-K mới, không retrain
```

## Sequence — chatbot agentic có evidence

```mermaid
sequenceDiagram
  actor U as Người mua
  participant W as Web
  participant A as /chat
  participant G as Recommendation Agent
  participant T as Allowed tools
  participant R as RAG context builder
  participant L as LLM

  U->>W: “Tại sao sản phẩm #2 phù hợp?”
  W->>A: POST /chat(session_id, message)
  A->>G: Parse intent + state
  G->>T: get_product_context(item #2)
  T->>R: metadata + relevant reviews of item #2
  R-->>T: evidence
  T-->>G: validated context
  G->>L: answer constrained to context
  L-->>G: explanation + citations
  G-->>A: typed chat response
  A-->>W: message, products, evidence
```

## State — session trên web

```mermaid
stateDiagram-v2
  [*] --> NewSession
  NewSession --> Browsing: recommend/search
  Browsing --> ResultsReady: Top-K received
  ResultsReady --> Refining: preference delta
  Refining --> ResultsReady: reranked
  ResultsReady --> Asking: chat/explain/compare
  Asking --> ResultsReady: response with products
  Asking --> Browsing: informational response
  ResultsReady --> Expired: TTL reached
  Browsing --> Expired: TTL reached
  Expired --> NewSession: start again
```

## Component diagram

```mermaid
flowchart TB
  UI[Next.js + Tailwind CSS]
  BFF[Next.js route handler]
  API[FastAPI]
  Session[Session state]
  Rec[Retrieval + Recommender + Reranker]
  Catalog[(Parquet / DuckDB)]
  Index[(FAISS)]
  Agent[Single recommendation agent]
  RAG[Metadata/review context]
  LLM[LLM optional]

  UI --> BFF --> API
  API --> Session
  API --> Rec
  Rec --> Catalog
  Rec --> Index
  API --> Agent
  Agent --> Rec
  Agent --> RAG
  RAG --> Catalog
  RAG --> Index
  Agent -. optional .-> LLM
```
