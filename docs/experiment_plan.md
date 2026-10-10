# Kế hoạch thực nghiệm – Đề tài 2.28

Nghiên cứu và phòng thủ trước tấn công đối kháng lên mô hình nhận diện hình ảnh.
Nhóm: Phan Thanh Việt, Huỳnh Thị Thanh Nhàn – Lớp Nh22.15.

> Giả định: workspace không có tài liệu yêu cầu của giảng viên (không có thư mục `docs/` gốc), nên kế hoạch bám theo mô tả đề tài trong prompt.

## 1. Câu hỏi nghiên cứu

1. Mô hình ResNet18 hoạt động thế nào trên CIFAR-10 sạch?
2. FGSM và PGD làm giảm accuracy ra sao?
3. Epsilon ảnh hưởng thế nào đến khả năng phân loại?
4. Adversarial Training có cải thiện robustness không?
5. Gaussian Blur có giảm tác động của nhiễu đối kháng không?
6. Các defense ảnh hưởng thế nào đến clean accuracy?
7. Có đánh đổi giữa clean accuracy, adversarial accuracy và chi phí tính toán không?
8. Defense nào phù hợp hơn trong điều kiện đã khảo sát?

## 2. Giả thuyết (cần kiểm chứng bằng số liệu thật, không phải kết luận có sẵn)

- H1: Accuracy của baseline giảm khi epsilon tăng, cả với FGSM và PGD.
- H2: Với cùng epsilon, PGD thường cho accuracy thấp hơn FGSM; sẽ kiểm tra thực nghiệm, không mặc định.
- H3: PGD Adversarial Training tăng adversarial accuracy nhưng giảm clean accuracy và tăng thời gian huấn luyện.
- H4: Gaussian Blur có thể phục hồi một phần accuracy dưới tấn công *không thích nghi* (attack tạo trên model không biết đến blur) nhưng làm giảm clean accuracy; hiệu quả có thể biến mất dưới tấn công thích nghi.

## 3. Biến

| Loại | Biến |
|---|---|
| Độc lập | loại attack (FGSM/PGD), epsilon, số bước PGD, phương pháp defense (none / adv. training / Gaussian Blur), kernel size, sigma |
| Phụ thuộc | clean/adversarial accuracy, ASR, mean L∞, mean L2, thời gian attack, thời gian train, thời gian inference |
| Kiểm soát | dataset CIFAR-10, kiến trúc ResNet18, split train/val/test, seed, optimizer, số epoch, batch size, không gian epsilon (pixel [0,1]), test set dùng chung |

## 4. Thiết kế thực nghiệm

- **Dữ liệu**: 50k train tách thành 45k train + 5k validation (seed cố định); test 10k chỉ dùng để đánh giá cuối. Best checkpoint chọn theo validation.
- **Chuẩn hóa**: lớp Normalize nằm *trong* model; attack hoạt động trên pixel [0,1], epsilon là L∞ theo thang pixel [0,1] (0.03 ≈ 7.65/255).
- **Exp 1 Baseline**: train ResNet18, đánh giá clean (accuracy, loss, per-class, confusion matrix, thời gian).
- **Exp 2 FGSM**: ε ∈ {0, 0.01, 0.03, 0.05, 0.1}.
- **Exp 3 PGD**: ε ∈ {0.01, 0.03, 0.05}, 10 bước, α = ε/4, random start.
- **Exp 4 Adversarial Training**: PGD-AT (ε=0.03, 5 bước, λ=0 mặc định). Đánh giá bằng attack tạo *trực tiếp trên model được đánh giá* (white-box).
- **Exp 5 Gaussian Blur**: kernel {3,5} × sigma trong config, không train lại. Báo cáo hai kịch bản: (a) ảnh đối kháng tạo trên baseline rồi blur (không thích nghi), (b) tấn công thích nghi đi qua lớp blur (PGD với blur nằm trong đồ thị gradient).
- **Exp 6 So sánh cuối**: Baseline, AT model, Baseline+Blur trên cùng test set, cùng metric.

Mọi dòng kết quả ghi rõ *attack được tạo trên model nào* (`source_model`) và *đánh giá trên model nào* (`eval_model`).

## 5. Metric

- Accuracy = số dự đoán đúng / tổng số mẫu.
- ASR = số mẫu đúng ban đầu và sai sau attack / số mẫu đúng ban đầu. Nếu không có mẫu đúng ban đầu: ghi `NaN`/`N/A`.
- Mean L∞, mean L2 của perturbation (pixel space).
- Thời gian tạo adversarial, thời gian train, thời gian inference.
- Accuracy và ASR là hai metric khác nhau, luôn báo cáo cả hai.

## 6. Phân tích

- Vẽ accuracy/ASR theo epsilon, so sánh FGSM vs PGD.
- So sánh baseline vs AT vs Blur theo clean accuracy, adversarial accuracy, chi phí.
- Nếu chạy nhiều seed: báo mean ± std.
- Nêu giới hạn: chỉ white-box L∞, chưa đánh giá AutoAttack/certified robustness; không kết luận "robust tuyệt đối".
