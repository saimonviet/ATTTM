| Model | Defense | Scenario | Attack source model | Epsilon | Clean Accuracy | FGSM Accuracy | PGD Accuracy | ASR (FGSM) | ASR (PGD) | Training Time (s) | PGD attack time (s) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | None (baseline) | white-box | baseline | 0.03 | 91.36% | 4.75% | 0.00% | 94.80% | 100.00% | 536.9 | 50.56 | attack tạo và đánh giá trên cùng model |
| advtrain_pgd | Adversarial Training (PGD, from scratch) | white-box | advtrain_pgd | 0.03 | 77.01% | 50.99% | 47.35% | 33.79% | 38.51% | 2254 | 51.16 | attack tạo và đánh giá trên cùng model |
| baseline | gaussian_blur_k3_s0.5 | transfer (non-adaptive) | baseline | 0.03 | 89.47% | 6.46% | 0.01% | 92.78% | 99.99% | 0 | 49.66 | không train lại; adversarial tạo trên model không có blur |
| baseline | gaussian_blur_k3_s0.5 | adaptive | baseline+gaussian_blur_k3_s0.5 | 0.03 | 89.47% | 4.37% | 0.00% | 95.12% | 100.00% | 0 | 50.34 | không train lại; attack tạo trực tiếp qua blur |
| baseline | gaussian_blur_k3_s1 | transfer (non-adaptive) | baseline | 0.03 | 59.57% | 22.06% | 26.91% | 63.37% | 55.15% | 0 | 49.66 | không train lại; adversarial tạo trên model không có blur |
| baseline | gaussian_blur_k3_s1 | adaptive | baseline+gaussian_blur_k3_s1 | 0.03 | 59.57% | 1.05% | 0.04% | 98.25% | 99.93% | 0 | 50.43 | không train lại; attack tạo trực tiếp qua blur |
| baseline | gaussian_blur_k5_s0.5 | transfer (non-adaptive) | baseline | 0.03 | 89.48% | 6.48% | 0.01% | 92.76% | 99.99% | 0 | 49.66 | không train lại; adversarial tạo trên model không có blur |
| baseline | gaussian_blur_k5_s0.5 | adaptive | baseline+gaussian_blur_k5_s0.5 | 0.03 | 89.48% | 4.41% | 0.00% | 95.07% | 100.00% | 0 | 50.35 | không train lại; attack tạo trực tiếp qua blur |
| baseline | gaussian_blur_k5_s1 | transfer (non-adaptive) | baseline | 0.03 | 43.76% | 24.00% | 27.61% | 46.48% | 37.73% | 0 | 49.66 | không train lại; adversarial tạo trên model không có blur |
| baseline | gaussian_blur_k5_s1 | adaptive | baseline+gaussian_blur_k5_s1 | 0.03 | 43.76% | 0.83% | 0.03% | 98.10% | 99.93% | 0 | 50.37 | không train lại; attack tạo trực tiếp qua blur |
