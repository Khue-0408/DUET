# DUET (Polyp Segmentation) – Paper-to-Code Methodology Notes

Mục tiêu của tài liệu này:
1) Tóm tắt các điểm methodology/experimental design trong DUET liên quan trực tiếp đến source code.
2) Chuẩn hóa “paper spec” thành checklist + mapping sang các module/file trong repo.
3) Làm chuẩn đầu vào cho code review/verification (đối chiếu codebase với paper).

Lưu ý:
- Khi paper không nói rõ chi tiết triển khai (cutoff FFT, bán kính morphology, v.v.), phần này sẽ ghi “ASSUMPTION/TODO” để bạn (hoặc reviewer) biết chỗ cần xác minh/điều chỉnh.
- Công thức viết theo LaTeX để bạn đối chiếu trực tiếp với phần implementation (loss.py / fusion.py / metrics.py).

---

## 0. Notation (theo Section Methodology)

Input image:
\[
I \in \mathbb{R}^{3 \times H \times W}
\]

Output segmentation mask và uncertainty map:
\[
M \in \{0, 1\}^{H \times W}, \quad U \in \mathbb{R}^{H \times W}
\]

Bài toán: pixel-wise binary classification (polyp vs background) kèm ước lượng uncertainty cho từng pixel.

---

## 1. High-level pipeline (3 stages)

DUET gồm 3 stage (tham chiếu Figure tổng quan của paper):

Stage 1 – Boundary-aware Domain-adaptive Encoder Pretraining
- Pretrain một encoder \(E_\theta\) sao cho:
  - nhạy với boundary (biên polyp),
  - domain-invariant (giảm domain shift giữa center/modality/hospital).

Stage 2 – Dual-stream Feature Extraction
- Tách ảnh theo frequency:
  - nhánh high-frequency (chi tiết, biên),
  - nhánh low-frequency (ngữ cảnh, shape).
- Hai encoder khác loại:
  - CNN encoder cho high-frequency,
  - ViT encoder cho low-frequency.

Stage 3 – Evidential Segmentation + Evidence-level Fusion
- Mỗi nhánh dự đoán Dirichlet parameters (evidence-based).
- Hợp nhất ở mức evidence (không fuse xác suất kiểu trung bình).
- Sinh uncertainty map theo Dirichlet concentration.

---

## 2. Stage 1: Boundary-aware Domain-adaptive Encoder Pretraining

### 2.1. Inputs / data assumption
- Paper mô tả có tập unlabeled lớn:
\[
\mathcal{D}_U = \{I_i\}_{i=1}^N
\]
- Và tập labeled nhỏ:
\[
\mathcal{D}_L = \{(I_j, M_j)\}_{j=1}^n
\]
Paper có nhắc \(N \approx 15{,}000\) (unlabeled).

### 2.2. Boundary-aware region construction (core/boundary/background)
Mục tiêu: tạo 3 vùng để sampling contrastive pairs và/hoặc weighting.

Một cấu trúc thường gặp (cần đối chiếu đúng paper):
- Core region \(R_C\): vùng chắc chắn là polyp (morphological erosion của GT/pseudo-mask).
- Boundary region \(R_B\): dải biên quanh polyp (giữa dilation và erosion).
- Background region \(R_{BG}\): phần còn lại (ngoài dilation).

Một biểu diễn hay dùng (ASSUMPTION nếu paper không ghi rõ toán tử):
\[
R_C = \text{Erode}(M, r_e)
\]
\[
R_B = \text{Dilate}(M, r_d) \setminus \text{Erode}(M, r_e)
\]
\[
R_{BG} = 1 - \text{Dilate}(M, r_d)
\]

ASSUMPTION/TODO:
- paper có thể định nghĩa bằng pseudo-mask hoặc heuristic boundary; cần xác nhận đúng cách tạo \(R_C, R_B, R_{BG}\) và các bán kính \(r_e, r_d\).

### 2.3. Pixel-wise contrastive learning (InfoNCE – Eq. (1) trong paper)
Các pixel feature:
\[
z_{x} = \text{Proj}(E_\theta(I)) \quad \text{(sau projection head)}
\]

Loss dạng InfoNCE (khung đối chiếu; công thức chính xác phải match paper):
\[
\mathcal{L}_{\text{con}}(x) = -\log
\frac{\exp(\text{sim}(z_x, z_{x^+})/\tau)}
{\exp(\text{sim}(z_x, z_{x^+})/\tau) + \sum_{x^- \in \mathcal{N}(x)} \exp(\text{sim}(z_x, z_{x^-})/\tau)}
\]

Trong đó:
- \(x^+\) là positive sample (thường cùng region hoặc cùng boundary-class tùy paper).
- \(\mathcal{N}(x)\) là tập negative sample (khác region, đặc biệt boundary vs background/core).
- \(\text{sim}\) thường là cosine similarity.
- \(\tau\) là temperature.

ASSUMPTION/TODO:
- paper có thể quy định số pair sampling / image / iteration, và chính sách P/N cụ thể theo region. Cần đối chiếu đúng paper.

### 2.4. Auxiliary segmentation / boundary task (Eq. (2) trong paper)
Paper mô tả một auxiliary task trong pretraining để ép encoder học tín hiệu segmentation/boundary.
Một khung thường gặp:
- seg head dự đoán \(\hat{M}\) (pseudo-mask hoặc weak label).
- loss Dice hoặc BCE/Dice:
\[
\mathcal{L}_{\text{seg}} = 1 - \frac{2\sum \hat{M} M + \epsilon}{\sum \hat{M} + \sum M + \epsilon}
\]

ASSUMPTION/TODO:
- paper nói rõ dùng Dice hay combo; cần match y hệt trong code.

### 2.5. Domain-adversarial alignment (Eq. (3) trong paper)
Có domain classifier \(D_\phi\) và Gradient Reversal Layer (GRL):
- Encoder học feature \(f = E_\theta(I)\)
- Domain head dự đoán domain label \(\hat{d} = D_\phi(\text{GRL}(f))\)

Domain loss:
\[
\mathcal{L}_{\text{adv}} = \text{CE}(\hat{d}, d)
\]

Tổng loss Stage 1 (khung):
\[
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{con}} + \lambda_{\text{seg}}\mathcal{L}_{\text{seg}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}}
\]

ASSUMPTION/TODO:
- paper có thể set \(\lambda_{\text{adv}}\), \(\lambda_{\text{seg}}\). Cần khớp YAML config.

---

## 3. Stage 2: Dual-stream Feature Extraction (FFT split + dual encoders)

### 3.1. FFT-based frequency decomposition
Paper mô tả tách \(I\) thành 2 ảnh:
- \(I_{\text{low}}\): low-frequency (global structure)
- \(I_{\text{high}}\): high-frequency (edges/details)

Một cách tách chuẩn trong frequency domain:
\[
F = \mathcal{F}(I)
\]
\[
F_{\text{low}} = F \odot \mathbb{1}(\|k\| \le r)
\]
\[
I_{\text{low}} = \mathcal{F}^{-1}(F_{\text{low}})
\]
\[
I_{\text{high}} = I - I_{\text{low}}
\]

ASSUMPTION/TODO:
- paper có thể không nêu rõ cutoff \(r\) hoặc mask type (ideal / gaussian). Nếu thiếu, repo cần ghi rõ default cutoff và để configurable.

### 3.2. Dual encoders
- High-frequency branch:
\[
F_H = E_H(I_{\text{high}})
\]
Trong đó \(E_H\) là CNN encoder (paper nêu EfficientNet-B4), được khởi tạo từ Stage 1 pretraining.

- Low-frequency branch:
\[
F_L = E_L(I_{\text{low}})
\]
Trong đó \(E_L\) là ViT-Base (paper mô tả ViT + MAE pretraining).

ASSUMPTION/TODO:
- Cách lấy multi-scale features từ ViT (tokens -> feature map) cần match paper. Nếu paper không ghi, phải document rõ mapping.

---

## 4. Stage 3: Evidential Segmentation + Evidence-level Fusion

### 4.1. Evidential head (Dirichlet)
Mỗi nhánh decoder dự đoán logits:
\[
o_H(x,y), \quad o_L(x,y)
\]

Evidence (paper mô tả dùng Softplus):
\[
e_H = \text{Softplus}(o_H), \quad e_L = \text{Softplus}(o_L)
\]

Dirichlet parameters:
\[
\alpha_H = e_H + 1, \quad \alpha_L = e_L + 1
\]

Dirichlet mean probability (paper có Eq. (6) cho binary; dạng tổng quát):
\[
\hat{p}_H(\text{polyp}) = \frac{\alpha_{H,\text{polyp}}}{\sum_c \alpha_{H,c}}, \quad
\hat{p}_L(\text{polyp}) = \frac{\alpha_{L,\text{polyp}}}{\sum_c \alpha_{L,c}}
\]

Total evidence / concentration:
\[
S_H = \sum_c \alpha_{H,c}, \quad S_L = \sum_c \alpha_{L,c}
\]

Diễn giải: \(S\) thấp -> Dirichlet phẳng -> uncertainty cao.

### 4.2. Evidential loss với digamma + KL (Eq. (7)(8) trong paper)
Một dạng EDL hay dùng (phải match paper):
- Expected cross-entropy (digamma):
\[
\mathcal{L}_{\text{EDL-data}}(\alpha, y) = \sum_{c=1}^{C} y_c\left(\psi(S) - \psi(\alpha_c)\right)
\]
- Regularization về uniform prior:
\[
\mathcal{L}_{\text{KL}} = \text{KL}\left(\text{Dir}(\alpha)\;||\;\text{Dir}(\mathbf{1})\right)
\]
- Tổng:
\[
\mathcal{L}_{\text{EDL}} = \mathcal{L}_{\text{EDL-data}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}
\]

ASSUMPTION/TODO:
- paper ghi rõ \(\lambda_{\text{KL}}\) và form KL đóng (closed-form). Repo cần đúng closed-form và đúng prior.

### 4.3. Region-weighted evidential loss (Eq. (9) trong paper)
Paper nhấn mạnh weighting theo vùng (boundary-focused).

Một khung:
\[
\mathcal{L}_{\text{RW-EDL}} =
w_B \cdot \mathbb{E}_{(x,y)\in R_B}\left[\mathcal{L}_{\text{EDL}}(\alpha(x,y), y(x,y))\right]
+
w_C \cdot \mathbb{E}_{(x,y)\in R_C}\left[\mathcal{L}_{\text{EDL}}(\alpha(x,y), y(x,y))\right]
+
w_{BG} \cdot \mathbb{E}_{(x,y)\in R_{BG}}\left[\mathcal{L}_{\text{EDL}}(\alpha(x,y), y(x,y))\right]
\]

ASSUMPTION/TODO:
- paper có thể set \(w_B, w_C, w_{BG}\). YAML phải khớp.

### 4.4. Dice loss trên fused probability
Paper dùng Dice để tối ưu segmentation accuracy:
\[
\mathcal{L}_{\text{Dice}}(\hat{p}^*, M) = 1 - \frac{2\sum \hat{p}^* M + \epsilon}{\sum \hat{p}^* + \sum M + \epsilon}
\]
\[
\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{RW-EDL}} + \lambda_{\text{Dice}}\mathcal{L}_{\text{Dice}}
\]

### 4.5. Evidence-level fusion (Eq. (10)(11)(12) trong paper)
Fusion weights theo total evidence:
\[
\pi_H = \frac{S_H}{S_H + S_L + \epsilon}, \quad
\pi_L = \frac{S_L}{S_H + S_L + \epsilon}
\]

Fuse evidence:
\[
e^* = \pi_H e_H + \pi_L e_L
\]
\[
\alpha^* = e^* + 1
\]
\[
\hat{p}^*(\text{polyp}) = \frac{\alpha^*_{\text{polyp}}}{\sum_c \alpha^*_c}
\]

### 4.6. Uncertainty map \(U\)
Paper nói DUET xuất uncertainty map từ Dirichlet concentration.

Hai cách phổ biến (chỉ 1 cái là đúng theo paper; cần xác nhận):
- Uncertainty mass (EDL hay dùng):
\[
U = \frac{C}{S^*} \quad (C=2 \text{ cho binary})
\]
- Hoặc variance của Beta/Dirichlet (binary):
\[
\text{Var}(p) = \frac{\alpha_0 \alpha_1}{(S^*)^2 (S^* + 1)}
\]
Repo cần implement đúng định nghĩa mà paper sử dụng.

---

## 5. Experimental design liên quan tới code (datasets, shifts, metrics)

### 5.1. Datasets paper nhắc tới
Paper nêu các benchmark:
- Kvasir-SEG
- CVC-ClinicDB
- ETIS-Larib
- PolypDB
- PolypGen

### 5.2. Domain shift protocols
Paper mô tả multi-axis evaluation:
- Cross-center shift: LOCO splits trên PolypGen.
- Cross-modality shift: PolypDB (ví dụ train WLI test NBI/BLI tuỳ paper mô tả).
- Theo kích thước polyp: subset “small polyps” (paper nhắc < 5mm).

ASSUMPTION/TODO:
- Cách define center_id / split file cần rõ trong repo (CSV mapping hoặc folder naming).

### 5.3. Metrics
Paper nhấn mạnh vừa accuracy vừa calibration/clinical readiness.
Repo nên có:
- Dice, mIoU, Precision, Recall (pixel-level).
- Calibration metric: ECE (Expected Calibration Error).
- Clinical-like: NPV (Negative Predictive Value) cho optical diagnosis setting (paper nhắc tiêu chuẩn > 90% NPV).

ASSUMPTION/TODO:
- NPV có thể là image-level rule dựa trên max probability/threshold. Repo phải mô tả đúng rule paper dùng.

---

## 6. Paper-to-Code mapping (điền theo repo thực tế của bạn)

Mục này dùng để đối chiếu “paper component” -> “implementation path”.
Bạn điền/đối chiếu theo codebase hiện có.

Gợi ý mapping template:

1) Stage 1
- Region construction (core/boundary/background): src/losses/region_weighting.py hoặc src/utils/morphology.py
- Pixel-wise contrastive InfoNCE: src/losses/contrastive_infonce.py
- Projection head / feature extractor hooks: src/models/encoders/*
- Domain GRL + discriminator: src/models/domain/grl.py + domain_discriminator.py
- Stage1 trainer: src/train/train_stage1.py

2) Stage 2
- FFT split: src/models/freq/fft_split.py
- EfficientNet-B4 encoder: src/models/encoders/efficientnet_b4.py
- ViT/MAE encoder: src/models/encoders/vit_mae.py
- Feature alignment / multi-scale extraction: src/models/decoders/* (nếu có adapter)

3) Stage 3
- Evidential utilities (softplus -> evidence -> alpha): src/models/evidential/evidence.py
- EDL loss (digamma + KL): src/losses/edl.py
- Fusion (evidence-level): src/models/fusion/evidence_fusion.py
- Dice loss: src/losses/dice.py
- Main trainer: src/train/train_duet.py
- Eval + metrics: src/train/eval.py + src/utils/metrics.py

4) Configs
- Stage1 config: configs/stage1_pretrain.yaml
- DUET default config: configs/duet_default.yaml
- Ablations: configs/ablations/*.yaml

---

## 7. Verification checklist (dùng cho code reviewer)

Stage 1 checklist:
- [ ] Có đúng 3 vùng core/boundary/background theo paper không? (mask logic + radius)
- [ ] InfoNCE sampling có đúng định nghĩa positives/negatives theo region không?
- [ ] Có GRL đúng nghĩa (gradient sign flip) và domain classifier đúng input feature không?
- [ ] Tổng loss Stage1 đúng hệ số \(\lambda\) theo paper và log đầy đủ từng term?

Stage 2 checklist:
- [ ] FFT split đúng cách paper mô tả (mask type, cutoff) và được config hóa?
- [ ] EH dùng weight Stage1 (load checkpoint) đúng chỗ?
- [ ] EL (ViT) đúng phiên bản + đúng cách lấy feature map?

Stage 3 checklist:
- [ ] Evidence = Softplus(logits) và \(\alpha=e+1\) đúng?
- [ ] EDL-data term dùng digamma đúng công thức paper?
- [ ] KL Dirichlet closed-form implement đúng và stable (clamp/eps)?
- [ ] Region-weighted EDL đúng vùng + đúng trọng số?
- [ ] Fusion weights dựa trên \(S=\sum \alpha\) đúng Eq. (10)?
- [ ] Uncertainty map U đúng định nghĩa paper?
- [ ] Metrics ECE/NPV được tính đúng protocol paper?

Engineering checklist:
- [ ] Config-driven (YAML) cho toàn bộ hyperparams quan trọng.
- [ ] Determinism/seed control.
- [ ] Resume training + checkpoint best on val Dice.
- [ ] AMP + DDP scaffold (ít nhất có khung).
- [ ] Unit tests tối thiểu cho: FFT split, KL Dirichlet, EDL-data (digamma), fusion weights.

---

## 8. Improvement ideas (clean/clear/pro-grade)
- Tách rõ “paper math” và “engineering glue”: losses/ chỉ chứa đúng công thức, trainer/ chỉ orchestration.
- Tạo một file “paper_trace.yaml” mapping Eq/Fig -> function/class name (tăng auditability).
- Logging chuẩn: từng loss term + histogram của \(S\) (evidence) + reliability diagram cho calibration.
- Kiểm soát numerical stability: eps/clamp cho alpha, S, digamma input.
- Dùng dataclass config + schema validation (pydantic hoặc custom) để tránh sai YAML.




## 9. Prompt
Bạn là Senior ML Engineer + Research Reproducibility Reviewer. Hãy đọc file DUET_PAPER_TO_CODE.md (paper spec) và parse toàn bộ repo source code hiện tại trong workspace. Nhiệm vụ: (1) Lập “Component Coverage Report” theo từng Stage (Stage1/2/3) dựa trên checklist trong markdown: đánh dấu ✅/⚠️/❌ và chỉ ra chính xác file path + line range đang implement tương ứng. (2) Với mọi công thức/loss/fusion trong markdown, hãy đối chiếu xem code có đúng toán (đặc biệt: digamma-based EDL-data, KL Dirichlet closed-form, region-weighted masking, evidence-level fusion weights theo S=sum(alpha), FFT split). Nếu có lệch: mô tả lệch ở đâu, tác động gì, và đưa patch proposal (pseudo-diff hoặc code snippet) để sửa. (3) Kiểm tra config YAML: mọi hyperparameter quan trọng có được expose và có default hợp lý không; có hard-code chỗ nào cần loại bỏ không. (4) Đề xuất cải thiện để repo clean/professional hơn: kiến trúc module, type hints, docstrings, naming, logging (loss terms + calibration plots), numerical stability (eps/clamp), unit tests tối thiểu cho FFT/KL/EDL/fusion, và khả năng reproducibility (seed, determinism, checkpoint/resume, AMP/DDP scaffold). Output theo format: A) Coverage table; B) Mismatch table (Markdown Spec → Code reference); C) Actionable refactor plan theo mức ưu tiên (P0/P1/P2); D) 5 quick wins để code rõ ràng hơn mà không đổi thuật toán.

