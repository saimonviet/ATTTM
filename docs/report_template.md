# BÁO CÁO ĐỀ TÀI 2.28
## Nghiên cứu và phòng thủ trước tấn công đối kháng lên mô hình nhận diện hình ảnh

**Môn:** An toàn thông tin – **Lớp:** Nh22.15
**Thành viên:** Phan Thanh Việt, Huỳnh Thị Thanh Nhàn
**Giảng viên hướng dẫn:** `{{TÊN GIẢNG VIÊN}}` – **Ngày nộp:** `{{NGÀY}}`

> **Cách dùng mẫu này.** Mọi chỗ ghi `{{...}}` là placeholder phải thay bằng số liệu/nội dung thật lấy từ `results/` sau khi chạy thí nghiệm. Mẫu **không chứa kết quả**; không điền số ước đoán. Nếu thí nghiệm chưa chạy, ghi `NOT_RUN` và nêu lý do. Nguồn số liệu được ghi trong ngoặc ở từng mục.

---

## Chương 1. Giới thiệu đề tài

### 1.1. Bối cảnh và vấn đề
Nêu vai trò của mô hình nhận diện ảnh trong các hệ thống thực tế và rủi ro an toàn thông tin khi mô hình bị đánh lừa bởi nhiễu đối kháng. `{{trích dẫn tài liệu tham khảo}}`

### 1.2. Mục tiêu và câu hỏi nghiên cứu
Liệt kê 8 câu hỏi nghiên cứu (xem `docs/experiment_plan.md`, mục 1) và mục tiêu của đề tài.

### 1.3. Phạm vi và giả định
Dataset CIFAR-10, ResNet18, tấn công white-box L∞ (FGSM, PGD), phòng thủ Adversarial Training và Gaussian Blur. Nêu rõ giả định về tài liệu giảng viên `{{có/không có tài liệu yêu cầu; tóm tắt}}` và những gì nằm ngoài phạm vi (certified robustness, AutoAttack…).

### 1.4. Phân công thành viên
`{{Phan Thanh Việt: …}}`, `{{Huỳnh Thị Thanh Nhàn: …}}`

---

## Chương 2. Cơ sở lý thuyết

### 2.1. Image classification
Bài toán phân loại ảnh, CNN/ResNet, hàm mất mát cross-entropy, CIFAR-10. `{{mô tả ngắn + tài liệu}}`

### 2.2. Adversarial examples
Định nghĩa, mô hình đe dọa (white-box/black-box), ràng buộc L∞/L2, ε trong pixel space [0,1].

### 2.3. FGSM
Công thức: `x_adv = clip_[0,1](x + ε · sign(∇ₓ L(f(x), y)))`. Ưu/nhược điểm một bước.

### 2.4. PGD
Công thức lặp có chiếu: `xₜ₊₁ = Π_{B∞(x,ε)∩[0,1]}(xₜ + α · sign(∇ₓ L(f(xₜ), y)))`, random start, các tham số (ε, α, số bước).

### 2.5. Adversarial Training
Bài toán min–max, hàm mục tiêu `L = λ·L_clean + (1−λ)·L_adv`, chi phí tính toán, khác biệt huấn luyện từ đầu và fine-tuning.

### 2.6. Gaussian Blur
Tiền xử lý làm mịn ảnh, kernel size và sigma; vì sao blur có thể giảm nhiễu tần số cao nhưng làm mất thông tin; khái niệm tấn công thích nghi (adaptive attack).

### 2.7. Các metric đánh giá robustness
Accuracy trên ảnh đối kháng, **ASR** = (đúng ban đầu và sai sau attack) / (đúng ban đầu), L∞ và L2 trung bình, thời gian. Nhấn mạnh Accuracy ≠ ASR và trường hợp không có mẫu đúng ban đầu.

---

## Chương 3. Thiết kế hệ thống và thực nghiệm

### 3.1. Kiến trúc hệ thống
Sơ đồ pipeline và cấu trúc mã nguồn (README mục 3–4). Quy ước Normalize nằm trong model để epsilon luôn ở pixel space.

### 3.2. Dữ liệu
Split `{{45.000/5.000/10.000}}`, seed `{{42}}`, augmentation, tập test chỉ dùng đánh giá cuối. (`experiment_manifest.json` → `dataset`)

### 3.3. Môi trường thực nghiệm
Phần cứng `{{GPU/VRAM}}`, phiên bản Python/PyTorch/CUDA `{{…}}`. (`experiment_manifest.json`)

### 3.4. Mô hình baseline và huấn luyện
Kiến trúc, optimizer, learning rate, epoch, batch size, scheduler, AMP. (`experiment_manifest.json` → `hyperparameters`)

### 3.5. Cấu hình tấn công
FGSM: ε `{{danh sách}}`. PGD: ε `{{danh sách}}`, số bước `{{…}}`, α `{{…}}`, random start `{{…}}`. (`configs/config.yaml`, `attack_configs`)

### 3.6. Cấu hình phòng thủ
Adversarial Training: phương pháp `{{…}}`, ε `{{…}}`, số bước `{{…}}`, λ `{{…}}`, chế độ `{{từ đầu / fine-tune}}`. Gaussian Blur: kernel `{{…}}`, sigma `{{…}}`, hai kịch bản (không thích nghi / thích nghi).

### 3.7. Thiết kế thí nghiệm và biến
Bảng biến độc lập/phụ thuộc/kiểm soát (`docs/experiment_plan.md`, mục 3–4). Mô tả Experiment 1–6.

### 3.8. Đảm bảo tính đúng đắn và tái lập
Kiểm thử đơn vị/tích hợp, seed, manifest (SHA256 checkpoint), cách tái lập (README mục 17).

---

## Chương 4. Kết quả và phân tích

> Mỗi bảng/hình phải ghi nguồn file trong `results/` và nêu rõ *attack được tạo trên model nào* và *đánh giá trên model nào*.

### 4.1. Baseline trên dữ liệu sạch
Clean accuracy `{{…}}`, loss `{{…}}`, thời gian train `{{…}}`, thời gian inference `{{…}}`. Per-class accuracy và class yếu nhất `{{…}}`. (`results/tables/baseline_best_clean.json`; hình `baseline_best_training_curves.png`, `baseline_best_confusion_matrix.png`)

### 4.2. Kết quả FGSM
Bảng accuracy/ASR/L∞/L2/thời gian theo ε. `{{điền từ results/tables/attack_table.csv, Model = baseline, Attack = fgsm}}`. Hình: `baseline_accuracy_vs_epsilon.png`, `baseline_asr_vs_epsilon.png`, `examples_baseline_fgsm_eps*.png`. Phân tích xu hướng theo ε `{{…}}`.

### 4.3. Kết quả PGD và so sánh với FGSM
Bảng PGD `{{…}}`; so sánh với FGSM ở cùng ε và chi phí thời gian `{{…}}`. Nhận xét dựa trên số liệu, không giả định PGD luôn mạnh hơn. `{{…}}`

### 4.4. Adversarial Training
Bảng clean/FGSM/PGD accuracy, ASR, thời gian train so với baseline `{{…}}`. Attack tạo trực tiếp trên model được đánh giá (white-box). Hình `comparison_models_accuracy_vs_epsilon.png`, `comparison_models_asr_vs_epsilon.png`, `advtrain_pgd_best_training_curves.png`. Phân tích đánh đổi clean accuracy và robustness `{{…}}`.

### 4.5. Gaussian Blur
Kịch bản không thích nghi và thích nghi, ảnh hưởng kernel/sigma, clean accuracy sau blur `{{…}}`. Phân tích mức giảm perturbation và mức mất thông tin (`results/tables/blur_analysis.csv`). Hình `baseline_blur_transfer_*.png`, `baseline_blur_adaptive_*.png`, `baseline_blur_examples.png`.

### 4.6. So sánh tổng hợp
Bảng defense (`results/tables/defense_table.csv`, ε chính `{{…}}`), hình `final_comparison_bars.png`, `tradeoff_clean_vs_pgd.png`. Trả lời lần lượt 8 câu hỏi nghiên cứu, mỗi câu dẫn chứng bằng số liệu `{{…}}`.

### 4.7. Độ tin cậy của kết quả
Số seed đã chạy `{{…}}`; mean ± std nếu có (`results/tables/seed_summary.csv`) `{{hoặc ghi rõ: chỉ một seed}}`. Các kiểm tra hợp lý (PGD so với FGSM, đơn điệu theo ε) và các hạn chế còn lại.

---

## Chương 5. Kết luận và hướng phát triển

### 5.1. Kết luận
Tóm tắt trả lời các câu hỏi nghiên cứu, chỉ nêu điều có bằng chứng thực nghiệm `{{…}}`. Không khẳng định defense hiệu quả ngoài phạm vi đã khảo sát; không tuyên bố certified robustness.

### 5.2. Giới hạn
`{{liệt kê, tham khảo README mục 18}}`

### 5.3. Hướng phát triển
Đánh giá bằng AutoAttack/attack mạnh hơn, nhiều seed, TRADES/AT với ε lớn hơn, các defense khác, dữ liệu/model khác, thử nghiệm black-box. `{{…}}`

---

## Tài liệu tham khảo
`{{Goodfellow et al. 2015 (FGSM); Madry et al. 2018 (PGD/Adversarial Training); He et al. 2016 (ResNet); Krizhevsky 2009 (CIFAR-10); các tài liệu khác}}`

## Phụ lục
- A. Cấu hình đầy đủ `configs/config.yaml` và `results/manifests/experiment_manifest.json`.
- B. Danh sách lệnh đã chạy.
- C. Hình ảnh bổ sung.
