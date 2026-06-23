# Lab Day 19 — Xây dựng Hệ thống GraphRAG & Đánh giá Hiệu năng

### Thông tin sinh viên
* **Họ và tên:** Nguyễn Xuân Tài
* **Mã học viên:** 2A202600621
* **Khóa học:** AI Application Developer - VinUniversity

---

## Giới thiệu & Quy trình thực hiện của tôi

Trong bài lab này, tôi đã thiết kế và triển khai một hệ thống **GraphRAG** (Retrieval-Augmented Generation dựa trên Đồ thị Tri thức) hoàn chỉnh từ đầu, đồng thời tiến hành đối sánh trực tiếp với giải pháp **Flat RAG** truyền thống trên tập dữ liệu xe điện/năng lượng tái tạo và thông tin về các tập đoàn công nghệ lớn.

Dưới đây là các bước tôi đã nghiên cứu và triển khai chi tiết:

### Bước 1: Trích xuất thực thể và quan hệ (Entity & Relation Extraction)
Tôi đã xây dựng một mô-đun trích xuất dựa trên mô hình ngôn ngữ lớn (Gemini 2.5 Flash). Với mỗi đoạn văn bản trong kho ngữ liệu:
* Tôi thiết kế prompt yêu cầu LLM trích xuất các thực thể cụ thể (Company, Technology, Product, Place, Person, Metric) và các quan hệ kết nối chúng dưới dạng các bộ ba (triples) `(subject, relation, object)`.
* Để tối ưu hóa chi phí API và tăng tốc độ thử nghiệm, tôi đã thiết lập cơ chế lưu trữ kết quả trích xuất tạm thời vào tệp `triples.json`. Khi chạy lại, hệ thống của tôi sẽ ưu tiên tải dữ liệu từ tệp này thay vì gọi lại LLM.

### Bước 2: Xây dựng Đồ thị Tri thức & Khử trùng lặp (Knowledge Graph Construction & Deduplication)
* Tôi sử dụng thư viện `NetworkX` để biểu diễn đồ thị dưới dạng `MultiDiGraph` nhằm hỗ trợ nhiều loại quan hệ khác nhau giữa cùng một cặp thực thể.
* **Khử trùng lặp thực thể (Deduplication)**: Tôi nhận thấy trong văn bản thô, một thực thể có thể xuất hiện dưới nhiều biến thể (ví dụ: `Meta`, `META`, `Meta Platforms, Inc.`). Nếu để nguyên, đồ thị sẽ bị phân mảnh và làm gãy luồng duyệt thông tin. Do đó, tôi đã viết một bộ chuẩn hóa tên thực thể để gộp tất cả các biến thể này về một nút duy nhất.
* Tôi cũng tạo ra các vector embeddings cho từng nút trong đồ thị bằng mô hình `gemini-embedding` phục vụ cho việc tìm kiếm ngữ nghĩa sau này.
* Đồ thị tri thức được tôi trực quan hóa bằng `matplotlib` và lưu thành ảnh `knowledge_graph.png`.

### Bước 3: Triển khai Flat RAG (Baseline)
Tôi xây dựng lớp `FlatRAG` làm baseline đối chứng:
* Hệ thống thực hiện chia nhỏ văn bản thành các chunk (1200 ký tự, overlap 150 ký tự), tính toán vector embedding cho từng chunk và lưu vào một vector index đơn giản.
* Khi người dùng đặt câu hỏi, tôi tìm kiếm các chunk có độ tương đồng cosine cao nhất và ghép chúng thành context thô gửi cho LLM.

### Bước 4: Triển khai GraphRAG với Thuật toán Duyệt Đồ thị
Đây là phần cốt lõi trong hệ thống của tôi:
1. **Liên kết thực thể (Entity Linking)**: Khi nhận câu hỏi từ người dùng, tôi trích xuất các thực thể chính trong câu hỏi, sau đó tìm nút tương ứng trong đồ thị bằng cách đối chiếu chuỗi hoặc tìm kiếm ngữ nghĩa dựa trên vector embedding của nút.
2. **Duyệt đồ thị đa bước (BFS Traversal)**: Từ các nút hạt giống tìm được, tôi cho hệ thống duyệt đồ thị theo chiều rộng (BFS) trong phạm vi 2-hop để thu thập toàn bộ các mối quan hệ xung quanh.
3. **Chuyển đổi đồ thị thành văn bản (Textualization)**: Tôi chuyển đổi các cạnh và quan hệ duyệt được thành các câu diễn đạt tự nhiên làm context cho mô hình.
4. **Trả lời câu hỏi**: LLM sẽ kết hợp ngữ cảnh cấu trúc đồ thị này cùng với văn bản hỗ trợ để đưa ra câu trả lời cuối cùng.

---

## Kết quả đánh giá hệ thống của tôi

Tôi đã chạy thử nghiệm đánh giá độc lập hệ thống trên bộ câu hỏi benchmark:
* **Độ chính xác (Accuracy)**: Hệ thống GraphRAG của tôi đạt **100.0%** độ chính xác (20/20 câu hỏi đúng), trong khi Flat RAG chỉ đạt **90.0%**. Hệ thống GraphRAG của tôi đã giải quyết triệt để các câu hỏi đòi hỏi kết nối dữ liệu từ nhiều nguồn khác nhau (multi-hop) mà Flat RAG thường bị ảo giác hoặc báo thiếu thông tin.
* **Thời gian phản hồi (Latency)**: Flat RAG phản hồi nhanh hơn (~2.4s) so với GraphRAG (~5.5s) do không phải thực hiện các bước trích xuất thực thể từ câu hỏi và duyệt đồ thị.
* Báo cáo đánh giá chi tiết được tôi lưu trữ đầy đủ trong file **[benchmark_report.md](benchmark_report.md)**.

---

## Hướng dẫn chạy chương trình của tôi

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install networkx numpy pandas matplotlib requests tqdm google-generativeai
   ```
2. Tạo file `.env` cục bộ để lưu trữ API key của bạn:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
3. Chạy file script chính để thực thi pipeline và sinh báo cáo:
   ```bash
   python run_graphrag_lab.py
   ```
4. Bạn cũng có thể mở trực tiếp tệp **[graphrag_lab.ipynb](graphrag_lab.ipynb)** đã được tôi tinh chỉnh để chạy thử nghiệm trực tiếp trên Google Colab hoặc môi trường Jupyter local.
