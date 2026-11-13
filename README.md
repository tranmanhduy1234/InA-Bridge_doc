# InA-Bridge

InA-Bridge là mô hình thị giác–ngôn ngữ kết nối **DINOv3** với **Qwen3-4B** thông qua **Q-Former khởi tạo từ BERT** và **visual projector**. Mô hình học cách chọn lọc thông tin trong ảnh, chuyển thành biểu diễn mà mô hình ngôn ngữ có thể sử dụng để sinh mô tả và trả lời câu hỏi.

Dự án triển khai ba tác vụ: truy hồi ảnh–văn bản ở Stage 1, sinh caption ở Stage 2 và hỏi–đáp về ảnh ở Stage 3. Kết quả đánh giá Stage 3 hiện ghi nhận **POPE F1 75,06%**, **MME Perception 874,26/2000** và **MME Reasoning 263,57/800**.

## Kiến trúc

```text
Ảnh → DINOv3 → Patch features → Q-Former → Visual projector → Qwen3-4B → Văn bản
                                    ↑                          ↑
                           Instruction (Stage 3)      Prompt / instruction
```

| Thành phần | Cấu trúc và vai trò |
| --- | --- |
| Vision encoder | `facebook/dinov3-vitl16-pretrain-lvd1689m`, trích xuất đặc trưng từng patch ảnh; đóng băng trong cả ba stage |
| Q-Former | Khởi tạo từ `bert-base-uncased`, dùng 64 query học được, hidden size 768; cross-attention với ảnh mỗi hai block |
| Visual projector | RMSNorm, nhánh gated SiLU và nhánh tuyến tính residual; ánh xạ query sang chiều embedding của Qwen |
| Language model | `Qwen/Qwen3-4B`, nhận visual prefix và văn bản qua `inputs_embeds`; thích nghi bằng LoRA ở Stage 2 |

### DINOv3: biểu diễn ảnh

Vision encoder giữ patch tokens, loại CLS và register tokens. Với cấu hình ảnh **640 × 640** và patch size **16**, mỗi ảnh tạo ra **1600 patch tokens**. Các đặc trưng này là đầu vào cho cross-attention của Q-Former.

DINOv3 được giữ cố định; việc học liên kết ảnh–ngôn ngữ diễn ra ở các thành phần phía sau.

### Q-Former: chọn lọc thông tin thị giác

Q-Former dùng các encoder layer khởi tạo từ BERT. Mỗi block thực hiện self-attention trên query và text nếu có, sau đó query đọc patch features qua cross-attention ở các block được chọn. Query và text có FFN riêng: FFN query được khởi tạo bằng bản sao FFN BERT rồi học độc lập. Query tokens và cross-attention được khởi tạo mới.

Attention mask thay đổi theo mục tiêu học:

| Mục tiêu | Tương tác giữa query và text |
| --- | --- |
| ITC — Image–Text Contrastive | Hai nhánh tách biệt để học biểu diễn ảnh và văn bản có thể so khớp |
| ITM — Image–Text Matching | Hai nhánh attention hai chiều để đánh giá một cặp ảnh–văn bản |
| ITG — Image-grounded Text Generation | Text đọc query và các token trước đó; query không đọc text |

Ở Stage 2, Q-Former chỉ nhận ảnh và query học được. Ở Stage 3, instruction được mã hóa bằng BERT embeddings và đưa vào Q-Former với mask ITM, giúp các query chọn thông tin ảnh theo câu hỏi. Answer không được đưa vào Q-Former.

### Projector và Qwen3-4B: từ ảnh sang ngôn ngữ

Projector giữ nguyên 64 query, chuyển chiều đặc trưng từ 768 sang chiều embedding của Qwen:

```text
x      = RMSNorm(query_features)
h      = SiLU(gate_proj(x)) × up_proj(x)
visual = skip(x) + scale × down_proj(Dropout(h))
```

Các phép chiếu dùng linear không bias; `scale` là tham số học được, khởi tạo bằng `0.1`. Đầu ra projector trở thành **64 visual prefix tokens**, được ghép trước text embeddings của Qwen. Qwen sinh văn bản theo cơ chế tự hồi quy.

Luồng tensor với `B` là batch size, `Dv` là chiều đặc trưng DINOv3, `Dl` là chiều embedding Qwen và `T` là số token văn bản:

```text
Ảnh             [B, 3, 640, 640]
Patch features  [B, 1600, Dv]
Visual queries  [B, 64, 768]
Visual prefix   [B, 64, Dl]
Đầu vào Qwen    [B, 64 + T, Dl]
```

BERT và Qwen dùng tokenizer cùng bảng embedding riêng. Visual prefix đi thẳng vào Qwen dưới dạng vector; projector ánh xạ đặc trưng thị giác, không chuyển đổi vocabulary giữa hai tokenizer.

## Ba giai đoạn học

| Giai đoạn | Mục tiêu | Thành phần được cập nhật |
| --- | --- | --- |
| Stage 1 — Căn chỉnh ảnh–văn bản | Học đồng thời ITC, ITM và ITG từ ảnh–caption | Q-Former, query tokens, các head ITC/ITM và embedding bắt đầu giải mã |
| Stage 2 — Sinh caption | Học dự đoán caption từ visual prefix | Lần lượt projector → LoRA → projector và LoRA cùng học |
| Stage 3 — Instruction tuning | Học trả lời instruction dựa trên ảnh | Q-Former và projector; Qwen cùng LoRA giữ cố định |

Stage 1 dùng EMA teacher và MoCo queue cho ITC, hỗ trợ nhiều caption cùng ảnh là positive. ITM học phân biệt cặp đúng với hard negatives; ITG học sinh text có điều kiện theo ảnh. Vision encoder và BERT embeddings được đóng băng, LM head ITG dùng chung trọng số với word embeddings BERT.

Stage 2 giữ Q-Former cố định, học caption bằng next-token cross-entropy. LoRA có rank 16, alpha 32 và dropout 0,05; trọng số nền Qwen không cập nhật. Stage 3 đưa instruction vào cả Q-Former và prompt Qwen, chỉ tính loss trên answer và EOS.

## Kết quả đánh giá

Kết quả dưới đây thuộc run **`stage3-eval-001`**, checkpoint Stage 3 tại **5000 optimizer steps**, snapshot `e2793a45c15d4bab808a601427380cd1`. Các báo cáo chấm điểm được tạo ngày 11–12/11/2026 (UTC).

Đánh giá dùng **VLMEvalKit** tại revision `c247c4d61d92be5643802aa9be5f2bc34d913d2a`, evaluator `exact_matching`, prompt mặc định của từng dataset và chat template của checkpoint. Lượt chạy dùng `do_sample=False`, tối đa 512 token mới và seed 42.

| Benchmark | Chỉ số | Kết quả |
| --- | --- | ---: |
| POPE | F1 | **75,06%** |
| POPE | Accuracy | 70,06% |
| POPE | Precision | 64,31% |
| POPE | Recall | 90,13% |
| MME | Perception | **874,26 / 2000** |
| MME | Reasoning | **263,57 / 800** |

Lượt inference hoàn tất **5127/5127 mẫu POPE** và **2374/2374 mẫu MME** trong bộ dữ liệu do toolkit nạp, coverage 100%, không đặt giới hạn subset. Điểm MME là điểm tổng hợp, không phải phần trăm accuracy.

POPE cho thấy mô hình nhận ra phần lớn vật thể thật nhưng còn nhiều khẳng định sai: recall 90,13% đi cùng precision 64,31%. Accuracy giảm từ 77,70% ở nhóm random xuống 63,03% ở nhóm adversarial. Trên MME, điểm nhận biết sự tồn tại của vật thể (158,33/200) và cảnh tổng thể (141,50/200) cao hơn vị trí (51,67/200), màu sắc (68,33/200) và tính toán (50/200).

Nguồn số liệu: `src/Evaluation/output/stage3-eval-001/summary.json` và `run.json` trong bản dự án dùng để tổng hợp tài liệu. Đây là kết quả của checkpoint nêu trên, chưa có baseline cùng protocol để kết luận mức cải thiện. MMBench_DEV_EN chưa có điểm hợp lệ trong báo cáo này do lượt đánh giá gốc không vượt qua kiểm tra đầu vào.

## Phạm vi và giới hạn

InA-Bridge phù hợp để thử nghiệm mô tả ảnh, hỏi–đáp một ảnh và nghiên cứu kết nối vision encoder với language model. Mô hình vẫn có thể mô tả vật thể không tồn tại, nhầm chi tiết và suy luận sai. Các kết quả trên chưa xác nhận chất lượng cho mọi ngôn ngữ, miền dữ liệu hoặc kiểu câu hỏi; năng lực của Qwen3 gốc không đồng nghĩa với năng lực đã được kiểm chứng của InA-Bridge.

## Giấy phép và ghi công

InA-Bridge sử dụng các mô hình nền do những nhóm dưới đây phát triển. Quyền đối với các thành phần gốc thuộc chủ sở hữu tương ứng; tên mô hình được dùng để ghi nguồn, không hàm ý các nhóm này bảo trợ hay chứng thực InA-Bridge.

| Thành phần | Nguồn và đơn vị phát triển | Giấy phép |
| --- | --- | --- |
| DINOv3 ViT-L/16 | [Meta — facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | [DINOv3 License](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md) |
| Qwen3-4B | [Qwen Team, Alibaba — Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE) |
| BERT base uncased | [Google — google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | [Apache License 2.0](https://raw.githubusercontent.com/google-research/bert/master/LICENSE) |

**DINOv3 có giấy phép riêng.** Khi phân phối DINO Materials hoặc tác phẩm phái sinh thuộc phạm vi giấy phép, phải tuân thủ các điều khoản của DINOv3 và cung cấp kèm bản sao giấy phép. Việc công bố nghiên cứu sử dụng DINOv3 phải ghi nhận thành phần này; các hạn chế sử dụng tại mục 1(b) vẫn áp dụng. Xem [DINOv3 License, mục 1](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md).

**Với các thành phần Apache 2.0**, khi phân phối phải kèm giấy phép, giữ các thông báo quyền liên quan, đánh dấu tệp đã sửa đổi và giữ nội dung NOTICE nếu bản gốc có. Xem [Apache License 2.0, mục 4](https://www.apache.org/licenses/LICENSE-2.0).

Phần thay đổi của InA-Bridge gồm Q-Former điều chỉnh từ BERT, visual projector, LoRA thích nghi Qwen và quy trình học ba stage. Bản tài liệu này không cấp giấy phép mới cho các thành phần gốc. Giấy phép riêng cho mã nguồn và trọng số do InA-Bridge tạo ra chưa được xác định trong bản dự án được rà soát; không suy ra rằng toàn bộ mô hình được phát hành theo Apache 2.0.

Ghi công và trích dẫn không thay thế việc tuân thủ giấy phép. Khi phát hành checkpoint hoặc dịch vụ inference, cần áp dụng điều khoản tương ứng với các thành phần thực tế được cung cấp; quyền đối với dữ liệu huấn luyện và nội dung đầu vào là vấn đề riêng. Các liên kết trên dẫn đến toàn văn điều khoản, còn phần này chỉ là tóm tắt.

## Trích dẫn

Các công trình nền tảng được ghi nhận:

1. Oriane Siméoni và cộng sự (2025). **DINOv3**. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104).
2. An Yang và cộng sự (2025). **Qwen3 Technical Report**. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).
3. Jacob Devlin, Ming-Wei Chang, Kenton Lee và Kristina Toutanova (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL-HLT. [Bài báo](https://aclanthology.org/N19-1423/).
4. Junnan Li, Dongxu Li, Silvio Savarese và Steven Hoi (2023). **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**. Công trình tham chiếu cho kiến trúc Q-Former. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597).

Mã nguồn và tài liệu kỹ thuật: [InA-Bridge trên GitHub](https://github.com/tranmanhduy1234/InABridge).
