# Adversarial Attack & Defense trên mô hình nhận diện ảnh (CIFAR-10)

**Đề tài 2.28** – Nghiên cứu và phòng thủ trước tấn công đối kháng lên mô hình nhận diện hình ảnh.
Môn An toàn thông tin – Lớp Nh22.15. Thành viên: **Phan Thanh Việt**, **Huỳnh Thị Thanh Nhàn**.

Project phục vụ học tập, thực nghiệm trong môi trường kiểm soát (chỉ dùng CIFAR-10 và model tự huấn luyện).

## 1. Giới thiệu

Mô hình phân loại ảnh có thể bị đánh lừa bằng những nhiễu rất nhỏ, khó thấy bằng mắt (adversarial examples). Project xây dựng pipeline đầy đủ để đo mức độ tổn thương của ResNet18 trên CIFAR-10 trước FGSM và PGD, rồi kiểm chứng hai hướng phòng thủ: **Adversarial Training** và **Gaussian Blur**.

> Giả định: workspace không có tài liệu yêu cầu của giảng viên, nên project bám theo mô tả đề tài. Nếu có tài liệu riêng, hãy đối chiếu lại các yêu cầu.

## 2. Mục tiêu

Trả lời bằng số liệu thực nghiệm (không tự tạo số liệu): clean accuracy của baseline; ảnh hưởng của FGSM/PGD và của epsilon; hiệu quả của Adversarial Training và Gaussian Blur; ảnh hưởng của defense lên clean accuracy; sự đánh đổi giữa clean accuracy, adversarial accuracy và chi phí tính toán.

## 3. Pipeline

```
CIFAR-10 -> Baseline ResNet18 -> FGSM / PGD -> Đánh giá attack (accuracy, ASR, L_inf, L2, thời gian)
         -> Defense: Adversarial Training (PGD-AT) | Gaussian Blur (transfer + adaptive)
         -> Đánh giá robustness -> Bảng + hình + manifest -> Báo cáo
```

Quy ước quan trọng: **model nhận ảnh pixel [0,1]; Normalize nằm bên trong model** (`models/resnet.py`), nên epsilon của FGSM/PGD luôn là epsilon L∞ trong không gian pixel [0,1] (0.03 ≈ 7.65/255), không bao giờ bị cộng nhầm vào tensor đã chuẩn hóa.

Định nghĩa metric: *Accuracy* = tỉ lệ đúng trên **toàn bộ** ảnh sau attack. *ASR* = số mẫu đúng ban đầu nhưng sai sau attack / số mẫu đúng ban đầu (NaN, ghi `N/A`, nếu không có mẫu đúng ban đầu). Hai metric này khác nhau và luôn được báo cáo cùng nhau. Mỗi dòng kết quả ghi rõ attack được **tạo trên model nào** (`source_model`) và **đánh giá trên model nào** (`eval_model`).

## 4. Cấu trúc thư mục

```
├── configs/config.yaml          cấu hình mặc định (YAML)
├── docs/                        experiment_plan.md, report_template.md
├── src/adversarial_lab/
│   ├── data/ models/            CIFAR-10, ResNet18
│   ├── attacks/                 base.py, fgsm.py, pgd.py
│   ├── defenses/                adversarial_training.py, gaussian_blur.py
│   ├── training/                engine.py (vòng train chung), train_baseline.py, train_adversarial.py
│   ├── evaluation/              evaluate_clean/attacks/defenses.py, build_reports.py, evaluate_all.py, run_*.py
│   ├── visualization/           visualize_attacks.py, plot_results.py
│   ├── utils/                   metrics, checkpoint, logging, cli, manifest
│   └── demo.py                  logic cho giao diện demo
├── scripts/                     train_baseline.py, run_fgsm.py, run_pgd.py, train_adversarial.py,
│                                run_blur.py, evaluate_all.py, run_pipeline.py
├── app/streamlit_app.py         giao diện demo
├── tests/                       unit + integration tests (pytest)
├── checkpoints/  data/  logs/   (không đưa vào Git)
└── results/{tables,figures,examples,manifests}/
```

## 5. Yêu cầu môi trường

- Python 3.11 hoặc 3.12 (đã chạy với 3.12.10), Windows/Linux.
- GPU NVIDIA tùy chọn: đã chạy trên **RTX 3050 Ti Laptop 4 GB VRAM** (cấu hình mặc định được chọn để vừa 4 GB). Không có GPU vẫn chạy được trên CPU nhưng chậm hơn nhiều; nên giảm `--max-samples`/số epoch.
- Dung lượng: khoảng 170 MB dữ liệu CIFAR-10, 43 MB mỗi checkpoint, vài GB cho PyTorch.

## 6. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate                       # Linux/macOS: source .venv/bin/activate
# PyTorch bản CUDA (đã dùng cu126; chọn bản phù hợp driver tại https://pytorch.org):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# hoặc bản CPU:  pip install torch torchvision
pip install -r requirements.txt
pip install -e .                             # tùy chọn
```

Kiểm tra GPU: `python -c "import torch; print(torch.cuda.is_available())"`. Mọi lệnh bên dưới chạy từ thư mục gốc project; thêm `--device cpu|cuda|auto` nếu cần (mặc định `auto`).

## 7. Tải dataset

CIFAR-10 được tải tự động bằng `torchvision` ở lần chạy đầu tiên vào `data/`. Nếu mạng chậm, tải tay `cifar-10-python.tar.gz` từ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz và đặt vào `data/`; torchvision sẽ kiểm tra checksum và giải nén. Train/validation được tách 45.000/5.000 từ tập train theo seed; **tập test 10.000 ảnh chỉ dùng để đánh giá cuối**, checkpoint tốt nhất chọn theo validation.

## 8. Huấn luyện baseline

```bash
python scripts/train_baseline.py --config configs/config.yaml
```

Mặc định: ResNet18 (conv1 3×3, bỏ maxpool), AdamW lr 1e-3, weight decay 5e-4, cosine, 20 epoch, batch 128, AMP, seed 42. Ghi `checkpoints/baseline_best.pth` (kèm metadata: kiến trúc, class mapping, normalization, cấu hình), `logs/baseline_best/history.csv` (+TensorBoard), `results/tables/baseline_best_clean.json` và hình đường cong train/confusion matrix. Tùy chọn: `--epochs`, `--lr`, `--batch-size`, `--seed`, `--max-samples` (smoke test), `--resume`, `--overwrite`, `--checkpoint`, `--output-dir`. Script từ chối ghi đè checkpoint đã có nếu không có `--overwrite`/`--resume`.

## 9. Chạy FGSM

```bash
python scripts/run_fgsm.py --checkpoint checkpoints/baseline_best.pth --epsilon 0.03
python scripts/run_fgsm.py --checkpoint checkpoints/baseline_best.pth    # epsilon theo config: 0 0.01 0.03 0.05 0.1
```

## 10. Chạy PGD

```bash
python scripts/run_pgd.py --checkpoint checkpoints/baseline_best.pth --epsilon 0.03 --steps 10
```

PGD L∞: α = ε × `alpha_ratio` (mặc định ε/4), random start bật (tắt bằng `--no-random-start`), projection và clamp [0,1] sau mỗi bước. Kết quả vào `results/tables/attack_results.csv` và `results/figures/` (ảnh gốc / perturbation phóng đại có ghi chú / ảnh đối kháng kèm nhãn và confidence; accuracy và ASR theo epsilon).

## 11. Adversarial Training

```bash
python scripts/train_adversarial.py --config configs/config.yaml
python scripts/train_adversarial.py --method fgsm --epsilon 0.03                 # FGSM-AT
python scripts/train_adversarial.py --lambda-clean 0.5 --adv-ratio 0.5            # L = λ·L_clean + (1-λ)·L_adv
python scripts/train_adversarial.py --mode finetune --init-checkpoint checkpoints/baseline_best.pth --epochs 5
```

Mặc định: **PGD-AT huấn luyện từ đầu**, ε=0.03, 5 bước, α=ε/2, random start, **λ=0 (adversarial-only)**, 20 epoch. Adversarial examples được sinh ngay trong vòng train trên model hiện tại (model ở eval mode khi sinh attack, rồi train mode khi cập nhật). Checkpoint tốt nhất chọn theo PGD accuracy trên validation. Chế độ `finetune` được ghi rõ trong metadata là fine-tuning, không phải train từ đầu. Sau khi train, script tự đánh giá FGSM/PGD white-box (tắt bằng `--no-eval`).

Lưu ý kỹ thuật: khi sinh attack lúc train, attack chạy bằng bfloat16 autocast. Đã đo fp16 làm ~11% gradient underflow về 0 và attack yếu hơn rõ rệt; bf16 cho dấu gradient khớp fp32 ~96%. Mọi bước **đánh giá** dùng fp32.

## 12. Gaussian Blur

```bash
python scripts/run_blur.py --checkpoint checkpoints/baseline_best.pth
python scripts/run_blur.py --checkpoint checkpoints/baseline_best.pth --kernel-sizes 3 5 --sigmas 0.5 1.0 --no-adaptive
```

Blur áp dụng trên pixel [0,1] rồi mới Normalize (không train lại model). Hai kịch bản: **không thích nghi** (adversarial tạo trên baseline không biết blur) và **thích nghi** (attack tạo trực tiếp qua blur, blur là phép vi phân được). Kèm `results/tables/blur_analysis.csv`: perturbation giảm bao nhiêu và ảnh sạch mất bao nhiêu thông tin sau blur.

## 13. Đánh giá tổng hợp

```bash
python scripts/evaluate_all.py --config configs/config.yaml     # chỉ chạy phần còn thiếu, rồi dựng bảng/hình/manifest
python scripts/evaluate_all.py --reports-only                   # dựng lại bảng/hình từ kết quả có sẵn
python scripts/evaluate_all.py --force                          # chạy lại mọi đánh giá
python scripts/run_pipeline.py --config configs/config.yaml     # train baseline -> train AT -> đánh giá (bỏ qua bước đã có checkpoint)
python scripts/run_pipeline.py --smoke                          # chạy thử rất nhỏ (512 mẫu, 1 epoch) vào results_smoke_pipeline/
```

Bảng chính được sinh tự động: `results/tables/attack_table.csv` (Model, Attack, Epsilon, Accuracy, ASR, Mean L_inf, Mean L2, Time) và `defense_table.csv` (Model, Defense, Clean/FGSM/PGD Accuracy, ASR, Training Time; ε chính = `report.main_epsilon` = 0.03). Mục chưa chạy ghi `NOT_RUN`.

## 14. Chạy Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Chọn checkpoint, ảnh CIFAR-10 hoặc ảnh tải lên (có cảnh báo ngoài phân phối), attack None/FGSM/PGD, epsilon, số bước PGD, defense None/Gaussian Blur/model adversarially trained. Nếu chưa có checkpoint, app chỉ hiển thị hướng dẫn train (không giả lập kết quả) và không train lại khi mở.

## 15. Vị trí lưu kết quả

| Nội dung | Vị trí |
|---|---|
| Checkpoint | `checkpoints/*.pth` (`*_last.pth` chỉ để resume) |
| Log huấn luyện | `logs/<tên checkpoint>/history.csv` |
| Bảng | `results/tables/` |
| Hình | `results/figures/` |
| Ảnh mẫu (.npz) | `results/examples/` |
| Manifest | `results/manifests/experiment_manifest.json` (thời gian, phiên bản, seed, cấu hình, SHA256 checkpoint, số mẫu đánh giá) |

## 16. Kết quả đã chạy (một lần chạy, seed 42, toàn bộ test set 10.000 ảnh)

ε=0.03 (L∞, pixel [0,1]); PGD 10 bước. Số liệu lấy từ `results/tables/defense_table.csv`.

| Cấu hình | Kịch bản | Clean | FGSM acc | PGD acc | ASR (PGD) | Train (s) |
|---|---|---|---|---|---|---|
| Baseline | white-box | 91.36% | 4.75% | 0.00% | 100% | 537 |
| PGD-AT (từ đầu) | white-box | 77.01% | 50.99% | 47.35% | 38.5% | 2254 |
| Baseline + blur k3 σ1.0 | không thích nghi | 59.57% | 22.06% | 26.91% | 55.2% | 0 |
| Baseline + blur k3 σ1.0 | thích nghi | 59.57% | 1.05% | 0.04% | 99.9% | 0 |
| Baseline + blur k5 σ1.0 | không thích nghi | 43.76% | 24.00% | 27.61% | 37.7% | 0 |
| Baseline + blur k5 σ1.0 | thích nghi | 43.76% | 0.83% | 0.03% | 99.9% | 0 |

Đọc kết quả thận trọng: Adversarial Training là cấu hình duy nhất giữ được accuracy đáng kể dưới tấn công thích nghi, với giá clean accuracy giảm ~14 điểm và chi phí train ~4,2 lần; Gaussian Blur chỉ giúp khi kẻ tấn công không biết đến nó và làm clean accuracy giảm mạnh. Đây là kết luận trong **phạm vi đã khảo sát**, xem mục 18.

## 17. Tái lập thí nghiệm

1. Cài đặt như mục 6, dùng cùng `configs/config.yaml` (đã đóng băng seed 42, split 45k/5k, các siêu tham số).
2. `python scripts/run_pipeline.py --config configs/config.yaml`.
3. So sánh `results/manifests/experiment_manifest.json` (phiên bản thư viện, SHA256 checkpoint).

Lưu ý: cuDNN benchmark/AMP/atomic ops khiến kết quả không giống từng bit giữa các máy; chênh lệch nhỏ là bình thường.

**Chạy nhiều seed để có mean ± std** (ví dụ seed 43):

```bash
python scripts/train_baseline.py --seed 43 --checkpoint checkpoints/baseline_seed43.pth
python scripts/run_fgsm.py --checkpoint checkpoints/baseline_seed43.pth --model-name baseline_seed43
python scripts/run_pgd.py  --checkpoint checkpoints/baseline_seed43.pth --model-name baseline_seed43
python scripts/train_adversarial.py --seed 43 --checkpoint checkpoints/advtrain_pgd_seed43.pth
python scripts/evaluate_all.py --reports-only     # tạo results/tables/seed_summary.csv khi có >= 2 seed cho cùng model
```

Tên model có hậu tố `_seed<N>` được gom nhóm cùng model gốc (model không hậu tố tính là một lần chạy). Hiện chưa có kết quả nhiều seed.

## 18. Giới hạn của phương pháp

- **Một seed** cho mọi kết quả chính; chưa có khoảng tin cậy. Kết quả có thể thay đổi giữa các lần chạy.
- Chỉ đánh giá **white-box L∞** với FGSM và PGD-10 (random start, một lần khởi tạo). Chưa dùng AutoAttack, C&W, attack black-box hay biến đổi khác; do đó độ robust của Adversarial Training có thể bị đánh giá lạc quan. Các kiểm tra hợp lý (PGD ≤ FGSM tại cùng ε, accuracy giảm đơn điệu theo ε) chưa đủ để loại trừ hoàn toàn gradient masking. **Không có certified robustness.**
- FGSM không giảm đơn điệu theo ε ở baseline (ε=0.1 cao hơn ε=0.05); chỉ ghi nhận, chưa phân tích nguyên nhân.
- Adversarial Training chỉ huấn luyện ở ε=0.03; độ robust giảm rõ khi ε lớn hơn (ε=0.05). Chưa khảo sát robust overfitting, λ>0, FGSM-AT hay fine-tuning bằng thí nghiệm đầy đủ (chỉ có code và test).
- Chỉ ResNet18, CIFAR-10, ảnh 32×32; kết quả không suy ra được cho dữ liệu/model khác. Demo trên ảnh ngoài CIFAR-10 chỉ mang tính minh họa.
- Gaussian Blur chỉ khảo sát vài cặp (kernel, σ); kết luận chỉ áp dụng cho các cấu hình đó.

## 19. Kiểm thử

```bash
python -m pytest -q
```

Gồm unit test cho FGSM, PGD (epsilon=0, khoảng [0,1], ràng buộc L∞, random start, projection, tham số model không đổi), metric (accuracy, ASR, trường hợp không có mẫu đúng, norm), dữ liệu/checkpoint, adversarial training, Gaussian Blur, bảng/manifest, demo, giao diện Streamlit (headless) và một smoke test tích hợp.
