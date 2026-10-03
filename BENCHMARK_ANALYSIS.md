# Phân tích kết quả benchmark và bonus

File này ghi lại kết quả của `python src/benchmark.py` và giải thích các con số bằng cơ chế đã triển khai trong `src/`.

## Kết quả benchmark

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1,154 | 12,986 | 0.00 | 0.20 | 0 | 0 |
| Advanced | 1,532 | 23,242 | 1.00 | 1.00 | 449 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 231 | 21,834 | 0.00 | 0.20 | 0 | 0 |
| Advanced | 332 | 14,137 | 1.00 | 1.00 | 348 | 3 |

## 1. Persistent memory làm tăng cross-session recall

Ở cả Standard và Stress, `Cross-session recall` của Baseline là **0.00**, còn Advanced là **1.00**. Baseline chỉ giữ `SessionState.messages` theo `thread_id`, vì vậy recall thread mới không nhìn thấy fact ở thread cũ; điều này cũng giải thích vì sao `Memory growth` của Baseline bằng **0 byte** ở cả hai bảng.

Advanced đi theo đường dữ liệu khác: `extract_profile_updates()` lấy fact ổn định từ tin nhắn, `UserProfileStore.upsert_fact()` ghi chúng vào `state/profiles/<user>/User.md`, và `AdvancedAgent._offline_response()` đọc lại các fact này khi câu hỏi đến từ thread mới. Đổi lại, recall 1.00 phụ thuộc vào chất lượng extraction: nếu một câu nhiễu bị coi là fact ổn định, lỗi đó cũng được mang qua mọi session sau.

## 2. Advanced tốn hơn ở hội thoại thông thường

Trong Standard Benchmark, Advanced sinh **1,532 agent tokens**, cao hơn **378 token (32.8%)** so với **1,154** của Baseline. `Prompt tokens processed` của Advanced là **23,242**, cao hơn **10,256 token (79.0%)** so với **12,986** của Baseline; đồng thời `Compactions` vẫn bằng **0**.

Nguyên nhân là `_estimate_prompt_context_tokens()` của Advanced tính cả `User.md`, compact summary và các message gần nhất. Với các conversation ngắn, từng thread chưa vượt ngưỡng compact nên Advanced phải trả chi phí profile nhưng chưa nhận được lợi ích nén; thao tác ghi file không nằm trong token count nhưng là thêm I/O và độ phức tạp vận hành. Vì vậy persistent memory cải thiện recall, nhưng không miễn phí và không mặc định thắng về token ở ngữ cảnh ngắn.

## 3. Compact chủ yếu giảm prompt cost ở hội thoại dài

Trong Stress Benchmark, Baseline xử lý **21,834 prompt tokens**, trong khi Advanced xử lý **14,137**, giảm **7,697 token (35.3%)**. Advanced thực hiện **3 compaction**, còn Baseline không compact; `CompactMemoryManager` chuyển message cũ thành summary có giới hạn và chỉ giữ 6 message gần nhất.

Lợi ích này xuất hiện ở `Prompt tokens processed`, không phải `Agent tokens only`: Advanced vẫn sinh **332 agent tokens**, cao hơn **231** của Baseline. Compact làm giảm lượng lịch sử phải đọc lại ở các lượt sau, nhưng không trực tiếp làm câu trả lời ngắn hơn; số output token còn phụ thuộc nội dung phản hồi và lượng fact Advanced có thể trả lời.

## 4. Memory growth và chi phí của hệ thống mạnh hơn

Advanced làm `User.md` tăng **449 byte** trong Standard và **348 byte** trong Stress, trong khi Baseline luôn là **0 byte**. Stress có hội thoại dài hơn nhưng file nhỏ hơn Standard vì `User.md` chỉ lưu fact có cấu trúc, không lưu nguyên văn toàn bộ hội thoại; phần lịch sử dài được compact trong RAM. Standard có nhiều loại fact và preference hơn nên profile lớn hơn.

Nếu số field hoặc giá trị tăng không giới hạn, file profile vẫn phình theo thời gian và làm mọi prompt sau đắt hơn. Nguy hiểm hơn, một fact sai có thể tồn tại lâu hơn message đã tạo ra nó; hệ thống vì thế cần guardrail cho câu hỏi, nhiễu và correction. Compact cũng có rủi ro riêng: summary bị giới hạn có thể bỏ mất chi tiết cần cho một follow-up hiếm dù tổng prompt token thấp hơn.

## Bonus đã chọn: Conflict handling

### Vấn đề được giải quyết

Dataset đổi nơi ở từ Huế sang Đà Nẵng, đổi nghề từ backend engineer sang MLOps engineer, đồng thời chèn Hà Nội và product manager làm nhiễu. Nếu profile chỉ append text hoặc luôn tin mẩu text mới nhất, câu recall hiện tại sẽ chứa cả fact cũ lẫn fact sai.

### Cơ chế và tác động

`extract_profile_updates()` chỉ nhận các mẫu khai báo đủ rõ, loại giá trị mang từ nghi vấn như “gì/đâu/không” và bỏ câu nói rõ một địa điểm không phải nơi ở. `upsert_fact()` dùng mỗi key đúng một dòng Markdown và thay giá trị cũ khi có correction; response style được hợp nhất vì các preference có thể bổ sung nhau thay vì loại trừ nhau. Nhờ đó các câu recall correction trong cả hai dataset vẫn góp vào mức recall **1.00**, và `test_conflict_handling_keeps_latest_stable_facts()` kiểm tra riêng rằng MLOps/Đà Nẵng thắng backend/Huế cũng như nhiễu Hà Nội/product manager.

### Rủi ro mới

Conflict handling hiện dựa trên regex tiếng Việt và chiến lược “fact hợp lệ mới nhất thắng”. Nó có thể bỏ sót cách diễn đạt chưa có pattern, hoặc ghi đè fact đúng nếu người dùng đưa một câu khẳng định sai nhưng có hình thức hợp lệ. Bước nâng cấp hợp lý là lưu thêm confidence, nguồn và thời điểm cho mỗi fact; đổi lại schema, prompt và logic giải quyết xung đột sẽ phức tạp hơn và profile có thể lớn hơn.
