# Triển khai DATN lên Azure for Students ($100 credit)

Mục tiêu: chạy cả stack (postgres, qdrant, backend, recommender, rag) trên **một VM 8 GB**, tốn ít credit nhất.
Các script ở đây chưa từng được chạy thật trên subscription của bạn: chạy từng bước và đọc kỹ đầu ra.

## Điều cần biết về gói
- $100 dùng trong 12 tháng, không cần thẻ. Hết credit thì subscription **bị vô hiệu hóa, không tính thêm tiền**
  (không thể nợ), nhưng bạn mất VM/dữ liệu nếu không nâng cấp lên pay-as-you-go.
- Chỉ được triển khai ở khoảng 5 region (khác nhau theo từng người), không xin tăng quota, nhiều dòng VM bị chặn.
- Marketplace không dùng được credit. Credit cũng không liên quan tới Gemini (dùng free tier của AI Studio riêng).

## Ngân sách (giá bán lẻ công khai, Southeast Asia, Linux, lấy từ prices.azure.com)
| Hạng mục | Giá |
|---|---|
| VM `Standard_B2as_v2` (2 vCPU, 8 GB) | $0.0944/giờ |
| VM `Standard_D2as_v5` / `Standard_B2ms` (dự phòng) | $0.108 / $0.106 mỗi giờ |
| Đĩa Standard SSD 64 GB (E6) | $4.80/tháng (tính cả khi VM tắt) |
| Public IP tĩnh (Standard) | $0.005/giờ = $3.65/tháng (tính cả khi VM tắt) |
| Băng thông ra | 0 cho phần đầu, sau đó khoảng $0.06-0.07/GB |

Chi phí cố định khi VM tắt: khoảng **$8.45/tháng**.

| Cách dùng | Ước tính/tháng | $100 kéo dài |
|---|---|---|
| Chạy 24/7 | khoảng $77 | khoảng 1.3 tháng |
| 10 giờ/ngày, 22 ngày | khoảng $29 | khoảng 3.4 tháng |
| 4 giờ/ngày, 22 ngày | khoảng $17 | khoảng 6 tháng |

Khuyến nghị: dev hằng ngày ở máy local (miễn phí), chỉ bật VM khi cần tích hợp hoặc demo. VM tự tắt lúc 22:00 giờ VN.

## Trạng thái đã triển khai (30/09/2026)
- Region **koreacentral** (subscription chỉ cho phép koreacentral, eastasia, indiasouthcentral, malaysiawest, indonesiacentral).
- VM `vm-datn` (Standard_B2as_v2, 20.41.100.79 nếu chưa đổi IP), user `datn`, repo tại `~/datn`, `.env` ở `~/datn/.env` (chmod 600).
- Stack: postgres, qdrant, recommender, rag, backend, **web** (Next.js, container `datn-web`) — tất cả bind 127.0.0.1.
- Dữ liệu đã chuyển: volume Postgres + Qdrant (`backend_postgres_data`, `backend_qdrant_data`), embedding, artifact, dataset.

## Dùng hằng ngày (chạy từ thư mục DATN trong Git Bash)
    bash deploy/azure/vm.sh start     # bật VM (~1 phút), container tự chạy lại (restart: unless-stopped)
    bash deploy/azure/vm.sh tunnel    # giữ cửa sổ này mở; sau đó mở http://localhost:3000 (web), :8000/docs (backend)
    bash deploy/azure/vm.sh ps        # trạng thái container + RAM
    bash deploy/azure/vm.sh logs rag  # log theo dõi trực tiếp (backend|rag|recommender|web|qdrant|postgres; bỏ trống = tất cả)
    bash deploy/azure/vm.sh stop      # TẮT (deallocate) - dừng tính tiền compute
Lưu ý Windows: dùng `127.0.0.1` thay `localhost` nếu trình duyệt/curl bị treo. Auto-shutdown hằng ngày mặc định 15:00 UTC (22:00 VN).

## Cập nhật code lên VM
    tar --exclude=.git --exclude=.venv --exclude=node_modules --exclude=.next --exclude=__pycache__ --exclude=./data --exclude=.env --exclude=deploy/azure/config.env -cf - . | ssh datn@<ip> 'tar -xf - -C ~/datn'
    bash deploy/azure/vm.sh ssh   # rồi: cd ~/datn && docker compose -f docker-compose.yml -f deploy/azure/docker-compose.azure.yml --profile ai up -d --build <service>

## Lưu ý khi chạy script trên Windows (Git Bash)
`az` là bản Windows nên nhận đường dẫn kiểu `/c/...` là sai: dùng `C:/Users/...` cho `SSH_PUBLIC_KEY_FILE`, `01-setup.sh` đã tự đổi `--custom-data` bằng `cygpath -m`.
Nếu cloud-init không chạy (Docker chưa có), cài tay: `curl -fsSL https://get.docker.com | sudo sh; sudo usermod -aG docker datn`.

## Các bước
1. Cài CLI và đăng nhập: `winget install Microsoft.AzureCLI`, rồi `az login`.
2. `cp deploy/azure/config.env.example deploy/azure/config.env`, chỉnh `AZ_LOCATION` cho đúng region được phép
   (xem Portal > Subscriptions > Policies, hoặc để script tự báo khi size không dùng được).
3. `deploy/azure/01-setup.sh` : kiểm tra size khả dụng, hiển thị giá, hỏi xác nhận rồi mới tạo VM.
4. **Tạo budget thủ công trong Portal** (script không làm được việc này một cách đáng tin cậy):
   Cost Management + Billing > Budgets > Add. Đặt $60, cảnh báo ở 50% / 80% / 100%, gửi email cho bạn.
   Budget chỉ cảnh báo, không chặn chi tiêu. Sau khi tạo subscription có thể mất tới 48 giờ mới xem được chi phí.
5. Hằng ngày: `vm.sh start`, làm việc, `vm.sh stop` (là *deallocate*; `az vm stop` vẫn tính tiền compute).
6. Truy cập dịch vụ qua tunnel, không mở cổng công khai: `vm.sh tunnel`, rồi dùng `localhost:8200`, `localhost:8000`...
7. Xong hẳn: `destroy.sh` xóa toàn bộ resource group (dừng mọi khoản phí cố định).

## Chạy stack trên VM
Trên VM, trong thư mục repo, tạo `.env` có `POSTGRES_PASSWORD` và `JWT_SECRET` do bạn tự sinh (overlay từ chối chạy nếu thiếu), rồi:

    docker compose -f docker-compose.yml -f deploy/azure/docker-compose.azure.yml --profile ai up -d

Overlay bind mọi cổng vào 127.0.0.1 nên public IP chỉ lộ SSH (NSG cũng chỉ mở 22 cho IP hiện tại của bạn).

## Ghi chú dữ liệu
- (Đã làm) Chuyển dữ liệu lên VM: `data/embedding` (1.8 GB), `data/artifacts/user_tower_balanced_v1` và `reranker_v2`,
  `data/processed/balanced_u5_i2_v1`, cache Jina (khoảng 1.7 GB, có thể để VM tự tải), và snapshot Qdrant.
  Tổng khoảng 6-8 GB, nên đĩa 64 GB là đủ.
