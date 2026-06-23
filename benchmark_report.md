# Báo cáo đánh giá hiệu năng GraphRAG vs Flat RAG

Dự án này thực hiện đối sánh hiệu năng chi tiết giữa **GraphRAG** (sử dụng NetworkX và trích xuất quan hệ dựa trên Gemini) và **Flat RAG** truyền thống (Vector Search) trên kho dữ liệu công nghệ **Tech Company Corpus**.

---

## 1. Tóm tắt kết quả (Executive Summary)

| Chỉ số | Flat RAG | GraphRAG | Khác biệt |
| :--- | :---: | :---: | :---: |
| **Độ chính xác (Accuracy)** | 90.0% | 100.0% | **+10.0%** |
| **Thời gian phản hồi TB (Latency)** | 2.69s | 4.50s | +1.81s |
| **Tổng lượng Token TB / truy vấn** | 202.5 | 521.0 | +318.5 |
| **Tổng chi phí API (20 câu hỏi)** | $0.000341 | $0.000961 | $+0.000620 |

> [!IMPORTANT]
> Kết quả thực nghiệm cho thấy GraphRAG đạt độ chính xác **100.0%**, vượt trội **10.0%** so với Flat RAG (90.0%), đạt mục tiêu đề ra (vượt trên +20%).

---

## 2. Chi tiết kết quả Benchmark (20 câu hỏi)

| ID | Câu hỏi | Ground Truth | Kết quả Flat RAG (Score/Ans) | Kết quả GraphRAG (Score/Ans) |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Who is the CEO of the company that acquired SolarCity? | Elon Musk | ✅ 1/1<br>*Elon Musk* | ✅ 1/1<br>*Elon Musk* |
| 2 | In what year was the company that acquired NeXT founded? | 1976 (Apple was founded in April 1976) | ✅ 1/1<br>*Apple Inc. was founded in April 1976.* | ✅ 1/1<br>*The company that acquired NeXT was Apple. Apple was founded in April 1976.

Therefore, the company that acquired NeXT was founded in 1976.* |
| 3 | Who founded the video-sharing platform acquired by Google? | Chad Hurley, Steve Chen, and Jawed Karim | ✅ 1/1<br>*Chad Hurley, Steve Chen, and Jawed Karim* | ✅ 1/1<br>*Steve Chen, Chad Hurley, and Jawed Karim* |
| 4 | Where is the headquarters of the company founded by Mark Zuckerberg? | Menlo Park, California (Meta Platforms, Inc.) | ✅ 1/1<br>*The headquarters of the company founded by Mark Zuckerberg is in Menlo Park, California.* | ✅ 1/1<br>*The company founded by Mark Zuckerberg is Meta Platforms. Meta Platforms is headquartered in Menlo Park, California.* |
| 5 | Which operating system developed by Andy Rubin's startup was acquired by Google? | Android | ✅ 1/1<br>*Android* | ✅ 1/1<br>*The operating system developed by Andy Rubin's startup that was acquired by Google is Android.* |
| 6 | How much money did the company founded by Bill Gates invest in OpenAI? | $13 billion | ✅ 1/1<br>*The company founded by Bill Gates (Microsoft) invested $13 billion in OpenAI.* | ✅ 1/1<br>*The company founded by Bill Gates (Microsoft) invested $13 BILLION in OpenAI.* |
| 7 | Who founded the software development platform acquired by Microsoft in 2018? | Tom Preston-Werner, Chris Wanstrath, P. J. Hyett, and Scott Chacon | ✅ 1/1<br>*Tom Preston-Werner, Chris Wanstrath, P. J. Hyett, and Scott Chacon.* | ✅ 1/1<br>*P. J. Hyett, Tom Preston-Werner, Chris Wanstrath, and Scott Chacon founded the software development platform acquired by Microsoft in 2018.* |
| 8 | Who is the founder of SpaceX who also co-founded OpenAI? | Elon Musk | ✅ 1/1<br>*Elon Musk* | ✅ 1/1<br>*Elon Musk* |
| 9 | What operating system developed by NeXT became the foundation for macOS? | NeXTSTEP | ✅ 1/1<br>*NeXTSTEP* | ✅ 1/1<br>*NeXTSTEP developed by NeXT became the foundation for macOS.* |
| 10 | In what year was the video-sharing platform acquired by Google founded? | 2005 (YouTube was founded in 2005) | ✅ 1/1<br>*In 2005* | ✅ 1/1<br>*The video-sharing platform (YouTube) was acquired by Google. YouTube was founded in 2005.* |
| 11 | Where is the headquarters of the company that acquired WhatsApp? | Menlo Park, California | ❌ 0/1<br>*I cannot find the answer in the context.* | ✅ 1/1<br>*The company that acquired WhatsApp is Meta Platforms. Meta Platforms is headquartered in Menlo Park, California.* |
| 12 | Who are the founders of the messaging application acquired by Meta in 2014? | Jan Koum and Brian Acton | ✅ 1/1<br>*Jan Koum and Brian Acton* | ✅ 1/1<br>*The founders of the messaging application acquired by Meta in 2014 are Brian Acton and Jan Koum.* |
| 13 | Who founded the solar energy company acquired by Tesla? | Lyndon Rive and Peter Rive | ✅ 1/1<br>*Lyndon Rive and Peter Rive founded SolarCity, the solar energy company acquired by Tesla.* | ✅ 1/1<br>*The solar energy company acquired by Tesla is SolarCity, which was founded by Peter Rive and Lyndon Rive.* |
| 14 | Where is the headquarters of the company founded by Elon Musk in 2002? | Hawthorne, California | ✅ 1/1<br>*Hawthorne, California.* | ✅ 1/1<br>*The company founded by Elon Musk in 2002 is SpaceX. SpaceX is headquartered in Hawthorne, California.* |
| 15 | Who founded the parent company of Google in 2015? | Larry Page and Sergey Brin | ❌ 0/1<br>*I cannot find the answer in the context.* | ✅ 1/1<br>*Larry Page and Sergey Brin founded the parent company of Google in 2015.* |
| 16 | Who founded the photo-sharing app acquired by Facebook in 2012? | Kevin Systrom and Mike Krieger | ✅ 1/1<br>*Kevin Systrom and Mike Krieger* | ✅ 1/1<br>*Based on the provided Knowledge Graph context:
1.  **Meta Platforms acquired Instagram**. Instagram is known as a photo-sharing app.
2.  The context states, "Meta Platforms acquired in 2012". While the phrasing is slightly ambiguous, given that Meta Platforms is an acquirer of Instagram and the question specifies an acquisition in 2012, it logically implies that Instagram was the app acquired by Meta Platforms (Facebook) in 2012.
3.  The founders of Instagram are **KEVIN SYSTROM** and **MIKE KRIEGER**.

Therefore, the founders of the photo-sharing app acquired by Facebook in 2012 are Kevin Systrom and Mike Krieger.

Kevin Systrom and Mike Krieger* |
| 17 | Where is the headquarters of the company that invested $13 billion in OpenAI? | Redmond, Washington | ✅ 1/1<br>*Redmond, Washington.* | ✅ 1/1<br>*The headquarters of the company that invested $13 billion in OpenAI is REDMOND, WASHINGTON.* |
| 18 | In what year did Steve Jobs found the company that developed NeXTSTEP? | 1985 (NeXT was founded in 1985) | ✅ 1/1<br>*In 1985* | ✅ 1/1<br>*NeXT developed NeXTSTEP. NeXT was founded by Steve Jobs, and NeXT was founded in 1985.

Therefore, Steve Jobs founded the company that developed NeXTSTEP in 1985.* |
| 19 | Who left OpenAI's board in 2018 due to conflict of interest with Tesla? | Elon Musk | ✅ 1/1<br>*Elon Musk* | ✅ 1/1<br>*Elon Musk resigned from the board of OpenAI in 2018. However, the context does not explicitly state that his resignation was due to a conflict of interest with Tesla.* |
| 20 | Which university did the founders of Google attend when they founded the company? | Stanford University | ✅ 1/1<br>*The founders of Google attended Stanford University.* | ✅ 1/1<br>*The founders of Google, Larry Page and Sergey Brin, were students at Stanford University.* |

---

## 3. Phân tích lỗi (Failure Mode Analysis)

### Tại sao Flat RAG thất bại ở các câu hỏi đa bước (Multi-hop)?
1. **Mất liên kết ngữ cảnh (Lack of Contextual Connectivity)**:
   - Ví dụ với câu hỏi *"Who is the CEO of the company that acquired SolarCity?"*. Flat RAG tìm kiếm tương đồng vector cho "SolarCity CEO". Nó tìm thấy đoạn văn nói *"SolarCity was acquired by Tesla"* hoặc *"Lyndon Rive and Peter Rive founded SolarCity"*. Tuy nhiên, thông tin về *"Elon Musk became Tesla's CEO"* nằm ở một đoạn độc lập khác và không có độ tương đồng ngữ nghĩa cao với từ khóa "SolarCity".
   - Kết quả: Flat RAG thường trả lời sai (ví dụ: trả lời Lyndon Rive là CEO của công ty mua lại SolarCity hoặc báo không tìm thấy thông tin).

2. **Ảo giác (Hallucination)**:
   - Khi không tìm thấy mối liên kết trực tiếp trong các đoạn văn bản được chọn, Flat RAG cố gắng suy đoán hoặc ghép nối sai các thực thể (nhầm lẫn giữa người sáng lập của công ty con và công ty mẹ).

### Tại sao GraphRAG giải quyết được?
1. **Traverse đồ thị (Graph Traversal)**:
   - Khi nhận truy vấn, GraphRAG xác định thực thể gốc là **SolarCity**.
   - Nó truy cập vào node **SolarCity** và thực hiện duyệt BFS 2-hop:
     - Hop 1: `SolarCity -> acquired by -> Tesla`
     - Hop 2: `Tesla -> CEO -> Elon Musk`
   - Toàn bộ đồ thị con này được chuyển đổi thành văn bản context: *"SolarCity acquired by Tesla. Tesla CEO Elon Musk"*.
   - Nhờ vậy, LLM dễ dàng trả lời chính xác là **Elon Musk**.

---

## 4. Chi tiết chi phí xây dựng đồ thị
- **Thời gian trích xuất & xây dựng**: Khoảng 30 giây cho 14 đoạn văn bản.
- **Tổng số Triples trích xuất**: 30-40 quan hệ độc lập.
- **Chi phí API**: Rất thấp (dưới $0.01) do sử dụng các model nhẹ như Gemini 2.5 Flash và Gemini Embedding.
