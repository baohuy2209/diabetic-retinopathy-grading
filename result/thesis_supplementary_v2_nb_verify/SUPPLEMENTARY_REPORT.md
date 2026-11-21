# Báo cáo Phân tích Bổ sung (Post Hoc, CPU-Only)

## 1. Truy vết nguồn và kiểm chứng nền tảng
- **Kho mã nguồn**: `D:\Github\diabetic-retinopathy-grading` (`mode=notebook`, `git_commit=fe81c9c07f50b9c840b741be3f9c08f4bfacdab2`, `dirty_relevant=True`)
- **Mã nguồn SHA-256 (start -> end)**: `a4f661adafb1...` -> `a4f661adafb1...` (`modified_during_run=False`)
- **Phân tích được chọn trong lượt chạy**: `A, B, C, D, E, F, G`
- **P0 cache (chỉ đọc)**: `D:\Github\diabetic-retinopathy-grading\result\p0_fp32_main_lodo_v1`
- **E0 cache (chỉ đọc)**: `D:\Github\diabetic-retinopathy-grading\result\p0_fp32_main_lodo_v1_e0_resume_v2`
- **E1 cache (chỉ đọc)**: `D:\Github\diabetic-retinopathy-grading\result\p0_fp32_main_lodo_v1_e1_resume_v2`
- **Archived reference / NPZ (chỉ đọc)**: `D:\Github\DR-evaluation-study2026\result\reference\lodo_summary_with_calibrated_screening.json` / `D:\Github\DR-evaluation-study2026\result\raw_preds`
- **Thư mục đầu ra mới**: `D:\Github\diabetic-retinopathy-grading\result\thesis_supplementary_v2_nb_verify` (seed=42, bootstrap_reps=2000, phương pháp tổng hợp: equal-cohort mean `1/6` trừ khi ghi rõ `pooled`)

### Kiểm chứng baseline từ dữ liệu theo fold
- P0 G0 audit counts: observed `118 PASS / 74 DISCREPANCY (total 192)` vs expected `118 PASS / 74 DISCREPANCY (total 192)` (`matched=True`)
- V3 native target G3 / Within-1 (%): observed `32.2165% / 92.3376%` vs expected `32.22% / 92.34%` (`matched=True`)
- V3 CIBD target G3 / Within-1 (%): observed `41.0691% / 91.1763%` vs expected `41.07% / 91.18%` (`matched=True`)
- E1 probe z_v2 target G3 recall (%): observed `44.6396%` vs expected `44.64%` (`matched=True`)
- E1 probe z_v3 target G3 recall (%): observed `51.0868%` vs expected `51.09%` (`matched=True`)
- E1 probe s_v3 target G3 recall (%): observed `54.0930%` vs expected `54.09%` (`matched=True`)

## 2. Trạng thái thực thi (analysis_status.csv)

- **A (Shared-decoder comparison of V2 and V3 on held-out targets)**: `COMPLETE` — All 6 folds and both variants verified against E0 target_decisions.csv.
- **B (Five-grade recall, G3 prediction ratio, and paired G3 transitions)**: `COMPLETE` — Computed 5-grade recalls, per-fold ratios, equal-cohort means, and image_key-matched G3 transitions.
- **C (P0 image-count discrepancies and ID-matched prediction agreement)**: `PARTIAL` — Verifiable image-count and support checks completed across 24 variant-fold-split rows (118 PASS / 74 DISCREPANCY of 192 metric checks). ID-matched agreement rows: 0 estimable, 0 disjoint zero-overlap, 24 NOT_ESTIMABLE.
- **D (Max p3 and Grade 3 decision-region overlap across 6 V3 folds)**: `COMPLETE` — Reconstructed float64 cloglog grids and interval overlaps across all 6 V3 folds.
- **E (Paired recorded-grouping-key bootstrap sensitivity analysis)**: `COMPLETE` — Completed 2000-replicate recorded-grouping-key bootstrap sensitivity analysis (seed=42). Primary patient-cluster CIs remain NOT_ESTIMABLE pending clinical identity verification.
- **F (Probe regularization C selected by source-validation QWK)**: `COMPLETE` — Evaluated all converged C candidates; s3 > z3 on mean target G3 recall preserved (53.25% vs 51.20%, ranking_preserved=True).
- **G (CIBD with 0.5 pp source-validation Within-1 loss budget)**: `COMPLETE` — Verified 12/12 exact reproduction of 1.0 pp baseline, then evaluated locked 0.5 pp source selections on target.

## 3. Tóm tắt phát hiện mới so với kết quả chính
### A. So sánh V2 và V3 dưới cùng decoder (equal-cohort mean trên 6 cohort, target)
- `argmax`: G3 V2=43.63%, V3=30.08% (V3-V2=-13.54 pp); Within-1 V2=90.64%, V3=84.45%.
- `posterior_median`: G3 V2=47.56%, V3=33.30% (V3-V2=-14.26 pp); Within-1 V2=91.08%, V3=91.02%.
- `rounded_expectation`: G3 V2=51.79%, V3=32.22% (V3-V2=-19.57 pp); Within-1 V2=91.71%, V3=92.34%.
- `cibd_selected`: G3 V2=53.73%, V3=41.07% (V3-V2=-12.66 pp); Within-1 V2=91.63%, V3=91.18%.
- `cutpoint_latent`: G3 V2=N/A, V3=45.84% (V3-V2=N/A); Within-1 V2=N/A, V3=89.46%.
### B. Recall 5 grade, tỷ lệ dự đoán G3 và dịch chuyển nhãn ghép cặp
- `native_v2`: G0=63.92%, G1=45.53%, G2=44.61%, G3=43.63%, G4=57.75%, G3 prec=39.65%, pred/true G3 ratio=1.45.
- `cibd_v2`: G0=61.27%, G1=45.28%, G2=40.98%, G3=53.73%, G4=52.05%, G3 prec=35.39%, pred/true G3 ratio=1.92.
- `native_v3`: G0=59.23%, G1=50.49%, G2=41.01%, G3=32.22%, G4=48.52%, G3 prec=31.71%, pred/true G3 ratio=1.31.
- `cibd_v3`: G0=67.11%, G1=43.99%, G2=40.42%, G3=41.07%, G4=53.95%, G3 prec=32.16%, pred/true G3 ratio=1.82.
- `probe_z_v2`: G0=58.22%, G1=43.02%, G2=47.26%, G3=44.64%, G4=61.19%, G3 prec=39.57%, pred/true G3 ratio=1.52.
- `probe_z_v3`: G0=46.18%, G1=47.52%, G2=41.23%, G3=51.09%, G4=49.18%, G3 prec=32.52%, pred/true G3 ratio=2.12.
- `probe_s_v3`: G0=44.93%, G1=46.37%, G2=40.06%, G3=54.09%, G4=50.17%, G3 prec=31.41%, pred/true G3 ratio=2.31.
- Chuyển sang G3 (`cibd_v2` vs `native_v2`, tổng 6 cohort): +944 ảnh mới được gán G3 (true G0=178, G1=48, G2=412, G3=165, G4=141; từ dự đoán cũ G2=545, G4=396); G3 đúng ròng +165.
- Chuyển sang G3 (`cibd_v3` vs `native_v3`, tổng 6 cohort): +1335 ảnh mới được gán G3 (true G0=109, G1=57, G2=775, G3=224, G4=170; từ dự đoán cũ G2=1335, G4=0); G3 đúng ròng +140.
- Chuyển sang G3 (`probe_z_v2` vs `native_v2`, tổng 6 cohort): +244 ảnh mới được gán G3 (true G0=5, G1=20, G2=165, G3=39, G4=15; từ dự đoán cũ G2=232, G4=6); G3 đúng ròng +5.
- Chuyển sang G3 (`probe_z_v3` vs `native_v3`, tổng 6 cohort): +1665 ảnh mới được gán G3 (true G0=89, G1=79, G2=901, G3=369, G4=227; từ dự đoán cũ G2=1523, G4=108); G3 đúng ròng +317.
- Chuyển sang G3 (`probe_s_v3` vs `native_v3`, tổng 6 cohort): +2004 ảnh mới được gán G3 (true G0=193, G1=66, G2=1111, G3=392, G4=242; từ dự đoán cũ G2=1966, G4=38); G3 đúng ròng +346.
### C. Kiểm tra P0 theo số ảnh và truy vết liên kết ID từng ảnh
- `V2` `source_val`: n_arch=41907, n_replay=41907 (khớp support 100%: True); status=`NOT_ESTIMABLE` (unavailable_missing_image_ids), ID linkage/agreement=`NOT_ESTIMABLE` (`NaN`); G3 đúng archived=915 vs replay=910 (chênh ròng -5); equal-cohort G3=62.32% -> 61.89%; G0 net=+337, G1 net=-29.
- `V2` `target`: n_arch=54883, n_replay=54883 (khớp support 100%: True); status=`NOT_ESTIMABLE` (unavailable_missing_image_ids), ID linkage/agreement=`NOT_ESTIMABLE` (`NaN`); G3 đúng archived=650 vs replay=648 (chênh ròng -2); equal-cohort G3=43.84% -> 43.63%; G0 net=-204, G1 net=+25.
- `V3` `source_val`: n_arch=41907, n_replay=41907 (khớp support 100%: True); status=`NOT_ESTIMABLE` (unavailable_missing_image_ids), ID linkage/agreement=`NOT_ESTIMABLE` (`NaN`); G3 đúng archived=475 vs replay=461 (chênh ròng -14); equal-cohort G3=31.73% -> 30.88%; G0 net=+345, G1 net=-36.
- `V3` `target`: n_arch=54883, n_replay=54883 (khớp support 100%: True); status=`NOT_ESTIMABLE` (unavailable_missing_image_ids), ID linkage/agreement=`NOT_ESTIMABLE` (`NaN`); G3 đúng archived=420 vs replay=422 (chênh ròng +2); equal-cohort G3=31.86% -> 32.22%; G0 net=+589, G1 net=-39.
- Biên độ chênh lệch G3 target theo từng fold: từ -4 đến +3 ảnh.
### D. Max p3 và độ chồng vùng G3 trên 6 fold V3
- Trên lưới latent float64: `max p3` nằm trong [0.2428, 0.4816] (trên ảnh target thực: [0.2428, 0.4816]).
- Dịch biên lên ở cả hai đầu xảy ra ở **6/6** fold; tỷ lệ chồng `overlap / cutpoint_length` trên lưới từ 22.2% đến 63.0% (IDRiD: 46.79%).
### E. Recorded-grouping-key bootstrap sensitivity analysis (2,000 lượt, seed=42)
- `cibd_v3_minus_native_v3` `recall_g3_pp`: point=+8.8526, 95% CI=[+6.1614, +11.6393] (valid_reps=2000/2000).
- `cibd_v3_minus_native_v3` `within1_pp`: point=-1.1613, 95% CI=[-1.3927, -0.9007] (valid_reps=2000/2000).
- `cibd_v3_minus_native_v3` `g3_precision_pp`: point=+0.4535, 95% CI=[-1.5883, +2.5586] (valid_reps=2000/2000).
- `cibd_v3_minus_native_v3` `qwk`: point=+0.0235, 95% CI=[+0.0201, +0.0269] (valid_reps=2000/2000).
- `cibd_v3_minus_native_v3` `mae`: point=-0.0263, 95% CI=[-0.0334, -0.0189] (valid_reps=2000/2000).
- `probe_s3_minus_native_v3` `recall_g3_pp`: point=+21.8764, 95% CI=[+18.7356, +24.9402] (valid_reps=2000/2000).
- `probe_s3_minus_native_v3` `within1_pp`: point=-0.8782, 95% CI=[-1.2179, -0.5310] (valid_reps=2000/2000).
- `probe_s3_minus_native_v3` `g3_precision_pp`: point=-0.2925, 95% CI=[-2.7723, +2.3143] (valid_reps=2000/2000).
- `probe_s3_minus_native_v3` `qwk`: point=-0.0040, 95% CI=[-0.0085, +0.0004] (valid_reps=2000/2000).
- `probe_s3_minus_native_v3` `mae`: point=+0.0522, 95% CI=[+0.0417, +0.0629] (valid_reps=2000/2000).
- Ghi chú: Khoảng tin cậy `patient_cluster` trong kết quả chính vẫn giữ `NOT_ESTIMABLE` do khóa nhóm là `legacy_unverified`.
### F. Chọn C của probe theo source-validation QWK
- `s_v3`: đổi C ở 3/6 fold (winner origin: cached=6, cpu_refit=0); mean target G3 recall 54.09% (macro-recall) -> 53.25% (QWK, delta=-0.85 pp), Within-1 delta=-0.05 pp, QWK delta=-0.0002.
- `z_v2`: đổi C ở 6/6 fold (winner origin: cached=6, cpu_refit=0); mean target G3 recall 44.64% (macro-recall) -> 43.96% (QWK, delta=-0.68 pp), Within-1 delta=-0.21 pp, QWK delta=-0.0074.
- `z_v3`: đổi C ở 5/6 fold (winner origin: cached=6, cpu_refit=0); mean target G3 recall 51.09% (macro-recall) -> 51.20% (QWK, delta=+0.12 pp), Within-1 delta=-0.28 pp, QWK delta=-0.0018.
- So sánh `s_v3` vs `z_v3` theo equal-cohort target G3 recall: `s_v3 > z_v3` (53.25% vs 51.20%; ranking preserved vs macro-recall: `True`).
### G. CIBD với ngân sách mất Within-1 trên nguồn 0,5 pp (post hoc) so với 1,0 pp (chính)
- `V2` `cibd_0.5pp_budget`: đổi tham số ở 0/6 fold, fallback=0/6; equal-cohort target G3=53.73% (delta=+10.10 pp), Within-1=91.63% (delta=+0.99 pp), QWK=0.7115; mean đạt đồng thời hai biên target H_D=True; cấp fold đạt biên: 4/6; cohort chưa đạt biên: Messidor-2, DDR.
- `V2` `cibd_1.0pp_budget`: đổi tham số ở 0/6 fold, fallback=0/6; equal-cohort target G3=53.73% (delta=+10.10 pp), Within-1=91.63% (delta=+0.99 pp), QWK=0.7115; mean đạt đồng thời hai biên target H_D=True; cấp fold đạt biên: 4/6; cohort chưa đạt biên: Messidor-2, DDR.
- `V3` `cibd_0.5pp_budget`: đổi tham số ở 6/6 fold, fallback=0/6; equal-cohort target G3=38.77% (delta=+6.56 pp), Within-1=91.63% (delta=-0.71 pp), QWK=0.7101; mean đạt đồng thời hai biên target H_D=True; cấp fold đạt biên: 4/6; cohort chưa đạt biên: DDR, EyePACS.
- `V3` `cibd_1.0pp_budget`: đổi tham số ở 0/6 fold, fallback=0/6; equal-cohort target G3=41.07% (delta=+8.85 pp), Within-1=91.18% (delta=-1.16 pp), QWK=0.7157; mean đạt đồng thời hai biên target H_D=False; cấp fold đạt biên: 3/6; cohort chưa đạt biên: APTOS 2019, DDR, EyePACS.

## 4. Đoạn LaTeX tiếng Anh sẵn sàng dán vào bản thảo

### Chương 4 (Phương pháp — Sensitivity & Post Hoc Checks)
```latex
\paragraph{Post hoc sensitivity analyses.}
To test the robustness of the main diagnostic findings without altering prespecified hypothesis decisions or model weights, we conducted post hoc CPU-only analyses (A, B, C, D, E, F, G) on the cached checkpoints and outputs.
For comparative decoder diagnostics, we evaluated $V_2$ and $V_3$ under identical fixed-output decoding rules and tabulated five-grade recall, predicted-to-reference Grade~3 ratios, and paired image-level transitions.
For archive reconciliation, we audited per-grade image counts between archived DMIP records and P0 replay while requiring strict non-null unique per-image identifier linkage for sample-wise agreement (reporting \texttt{NOT\_ESTIMABLE} when identifiers are unavailable).
For latent calibration geometry, we reconstructed float64 cumulative-link probabilities across $V_3$ latent grids to measure peak Grade~3 probability ($\max p_3$) and decision-interval overlap.
For uncertainty sensitivity, we computed a 2000-replicate recorded-grouping-key bootstrap analysis (seed~42, equal cohort weight $1/6$) for the target differences $V_3$~CIBD minus native $V_3$ and $s_3$ probe minus native $V_3$, leaving primary patient-cluster intervals as \texttt{NOT\_ESTIMABLE}.
For E1 hyperparameter selection sensitivity, we repeated source-validation regularization selection using QWK (tie-broken by macro recall, Grade~3 recall, lower NLL, and smaller $C$) before evaluating held-out target cohorts.
For E0 budget sensitivity, distinct from the prespecified primary 1.0~pp source-validation Within-1 loss budget, we evaluated a stricter post hoc 0.5~pp source-validation budget while keeping the prespecified target $H_D$ evaluation thresholds (+3.0~pp Grade~3 recall and $-1.0$~pp Within-1 accuracy) unchanged.
```

### Chương 5 (Kết quả — Post Hoc Verification)
```latex
\paragraph{Post hoc verification and sensitivity findings.}
Under matched fixed-output decoders, target equal-cohort Grade~3 recall differences ($V_3 - V_2$) were \texttt{argmax}: 30.08\% vs.\ 43.63\% (-13.54~pp); \texttt{posterior\_median}: 33.30\% vs.\ 47.56\% (-14.26~pp); \texttt{rounded\_expectation}: 32.22\% vs.\ 51.79\% (-19.57~pp); \texttt{cibd\_selected}: 41.07\% vs.\ 53.73\% (-12.66~pp).
Paired image-level transitions over native predictions yielded \texttt{cibd\_v2} (+165 net true Grade~3), \texttt{cibd\_v3} (+140 net true Grade~3), \texttt{probe\_z\_v2} (+5 net true Grade~3), \texttt{probe\_z\_v3} (+317 net true Grade~3), \texttt{probe\_s\_v3} (+346 net true Grade~3).
Archive reconciliation across target cohorts found V2 (-2 net Grade~3, linkage \texttt{NOT\_ESTIMABLE}); V3 (+2 net Grade~3, linkage \texttt{NOT\_ESTIMABLE}).
Across 6 $V_3$ folds, peak Grade~3 probability on the diagnostic latent grid remained bounded between 0.243 and 0.482, and native rounded expectation shifted the Grade~3 interval upward relative to the learned cutpoints in all 6 folds (grid overlap 22.2\%--63.0\%).
In the post hoc recorded-grouping-key bootstrap sensitivity analysis (2000 replicates, seed~42), $V_3$ CIBD changed equal-cohort Grade~3 recall by +8.85~pp (95\% CI [+6.16, +11.64]) and Within-1 accuracy by -1.16~pp ([-1.39, -0.90]), while the $s_3$ probe shifted Grade~3 recall by +21.88~pp ([+18.74, +24.94]).
Selecting E1 regularization by source-validation QWK instead of macro recall changed $C$ in 3/6 folds for $s_3$ and 5/6 folds for $\mathbf{z}_3$, yielding equal-cohort target Grade~3 recalls of 53.25\% ($s_3$) vs.\ 51.20\% ($\mathbf{z}_3$) and preserving the scalar-over-vector ordering.
Tightening the source-validation CIBD Within-1 loss budget from 1.0~pp (primary) to 0.5~pp (post hoc) changed $V_3$ selections in 6/6 folds and yielded a post hoc equal-cohort target Grade~3 recall change of +6.56~pp alongside a Within-1 change of -0.71~pp (4/6 folds meeting both prespecified target $H_D$ margins).
Under the 1.0~pp primary rule reproduced from cached grids, $V_3$ equal-cohort target changes were +8.85~pp Grade~3 recall and -1.16~pp Within-1 accuracy.
```

### Chương 6 (Ảnh hưởng tới RQ2 và RQ3)
```latex
For RQ2, the prespecified 1.0~pp primary source budget leaves $H_D$ unsupported (+8.85~pp equal-cohort Grade~3 recall change, -1.16~pp equal-cohort Within-1 change; 95\% recorded-grouping-key bootstrap CI $[-1.39, -0.90]$~pp).
Under the post hoc 0.5~pp source budget, the equal-cohort mean target trade-off met both target margins (+6.56~pp Grade~3 recall, -0.71~pp Within-1 accuracy), with 4/6 cohorts satisfying both criteria while DDR and EyePACS missed a prespecified target margin; this informs future source-selection safety margins without altering the prespecified primary 1.0~pp hypothesis decision.
Latent-grid inspection across 6 folds confirmed that bounded peak Grade~3 probability (max $p_3$ 0.243--0.482) and upward boundary displacement occurred in all 6 folds.
For RQ3, selecting probe regularization by source-validation QWK yielded higher equal-cohort target Grade~3 recall for $s_3$ (53.25\%) than for $\mathbf{z}_3$ (51.20\%), preserving the scalar-over-vector ordering and leaving $H_R$ unsupported under both selection criteria.
```
