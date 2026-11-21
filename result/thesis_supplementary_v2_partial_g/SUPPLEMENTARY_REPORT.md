# Báo cáo Phân tích Bổ sung (Post Hoc, CPU-Only)

## 1. Truy vết nguồn và kiểm chứng nền tảng
- **Kho mã nguồn**: `D:\Github\diabetic-retinopathy-grading` (`mode=cli_script`, `git_commit=fe81c9c07f50b9c840b741be3f9c08f4bfacdab2`, `dirty_relevant=True`)
- **Mã nguồn SHA-256 (start -> end)**: `449fb979e3e2...` -> `449fb979e3e2...` (`modified_during_run=False`)
- **Phân tích được chọn trong lượt chạy**: `G`
- **P0 cache (chỉ đọc)**: `D:\Github\diabetic-retinopathy-grading\result\p0_fp32_main_lodo_v1`
- **E0 cache (chỉ đọc)**: `D:\Github\diabetic-retinopathy-grading\result\p0_fp32_main_lodo_v1_e0_resume_v2`
- **E1 cache (chỉ đọc)**: `D:\Github\diabetic-retinopathy-grading\result\p0_fp32_main_lodo_v1_e1_resume_v2`
- **Archived reference / NPZ (chỉ đọc)**: `D:\Github\DR-evaluation-study2026\result\reference\lodo_summary_with_calibrated_screening.json` / `D:\Github\DR-evaluation-study2026\result\raw_preds`
- **Thư mục đầu ra mới**: `D:\Github\diabetic-retinopathy-grading\result\thesis_supplementary_v2_partial_g` (seed=42, bootstrap_reps=2000, phương pháp tổng hợp: equal-cohort mean `1/6` trừ khi ghi rõ `pooled`)

### Kiểm chứng baseline từ dữ liệu theo fold
- P0 G0 audit counts: observed `118 PASS / 74 DISCREPANCY (total 192)` vs expected `118 PASS / 74 DISCREPANCY (total 192)` (`matched=True`)
- V3 native target G3 / Within-1 (%): observed `32.2165% / 92.3376%` vs expected `32.22% / 92.34%` (`matched=True`)
- V3 CIBD target G3 / Within-1 (%): observed `41.0691% / 91.1763%` vs expected `41.07% / 91.18%` (`matched=True`)
- E1 probe z_v2 target G3 recall (%): observed `44.6396%` vs expected `44.64%` (`matched=True`)
- E1 probe z_v3 target G3 recall (%): observed `51.0868%` vs expected `51.09%` (`matched=True`)
- E1 probe s_v3 target G3 recall (%): observed `54.0930%` vs expected `54.09%` (`matched=True`)

## 2. Trạng thái thực thi (analysis_status.csv)

- **G (CIBD with 0.5 pp source-validation Within-1 loss budget)**: `COMPLETE` — Verified 12/12 exact reproduction of 1.0 pp baseline, then evaluated locked 0.5 pp source selections on target.

## 3. Tóm tắt phát hiện mới so với kết quả chính
### G. CIBD với ngân sách mất Within-1 trên nguồn 0,5 pp (post hoc) so với 1,0 pp (chính)
- `V2` `cibd_0.5pp_budget`: đổi tham số ở 0/6 fold, fallback=0/6; equal-cohort target G3=53.73% (delta=+10.10 pp), Within-1=91.63% (delta=+0.99 pp), QWK=0.7115; mean đạt đồng thời hai biên target H_D=True; cấp fold đạt biên: 4/6; cohort chưa đạt biên: Messidor-2, DDR.
- `V2` `cibd_1.0pp_budget`: đổi tham số ở 0/6 fold, fallback=0/6; equal-cohort target G3=53.73% (delta=+10.10 pp), Within-1=91.63% (delta=+0.99 pp), QWK=0.7115; mean đạt đồng thời hai biên target H_D=True; cấp fold đạt biên: 4/6; cohort chưa đạt biên: Messidor-2, DDR.
- `V3` `cibd_0.5pp_budget`: đổi tham số ở 6/6 fold, fallback=0/6; equal-cohort target G3=38.77% (delta=+6.56 pp), Within-1=91.63% (delta=-0.71 pp), QWK=0.7101; mean đạt đồng thời hai biên target H_D=True; cấp fold đạt biên: 4/6; cohort chưa đạt biên: DDR, EyePACS.
- `V3` `cibd_1.0pp_budget`: đổi tham số ở 0/6 fold, fallback=0/6; equal-cohort target G3=41.07% (delta=+8.85 pp), Within-1=91.18% (delta=-1.16 pp), QWK=0.7157; mean đạt đồng thời hai biên target H_D=False; cấp fold đạt biên: 3/6; cohort chưa đạt biên: APTOS 2019, DDR, EyePACS.

## 4. Đoạn LaTeX tiếng Anh sẵn sàng dán vào bản thảo

### Chương 4 (Phương pháp — Sensitivity & Post Hoc Checks)
```latex
\paragraph{Post hoc sensitivity analyses.}
To test the robustness of the main diagnostic findings without altering prespecified hypothesis decisions or model weights, we conducted post hoc CPU-only analyses (G) on the cached checkpoints and outputs.
For E0 budget sensitivity, distinct from the prespecified primary 1.0~pp source-validation Within-1 loss budget, we evaluated a stricter post hoc 0.5~pp source-validation budget while keeping the prespecified target $H_D$ evaluation thresholds (+3.0~pp Grade~3 recall and $-1.0$~pp Within-1 accuracy) unchanged.
```

### Chương 5 (Kết quả — Post Hoc Verification)
```latex
\paragraph{Post hoc verification and sensitivity findings.}
Tightening the source-validation CIBD Within-1 loss budget from 1.0~pp (primary) to 0.5~pp (post hoc) changed $V_3$ selections in 6/6 folds and yielded a post hoc equal-cohort target Grade~3 recall change of +6.56~pp alongside a Within-1 change of -0.71~pp (4/6 folds meeting both prespecified target $H_D$ margins).
Under the 1.0~pp primary rule reproduced from cached grids, $V_3$ equal-cohort target changes were +8.85~pp Grade~3 recall and -1.16~pp Within-1 accuracy.
```

### Chương 6 (Ảnh hưởng tới RQ2 và RQ3)
```latex
For RQ2, the prespecified 1.0~pp primary source budget leaves $H_D$ unsupported (+8.85~pp equal-cohort Grade~3 recall change, -1.16~pp equal-cohort Within-1 change).
Under the post hoc 0.5~pp source budget, the equal-cohort mean target trade-off met both target margins (+6.56~pp Grade~3 recall, -0.71~pp Within-1 accuracy), with 4/6 cohorts satisfying both criteria while DDR and EyePACS missed a prespecified target margin; this informs future source-selection safety margins without altering the prespecified primary 1.0~pp hypothesis decision.
```
