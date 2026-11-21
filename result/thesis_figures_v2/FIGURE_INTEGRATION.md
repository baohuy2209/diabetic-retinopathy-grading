# Figure Integration Guide (`result/thesis_figures_v2/`)

Copy the `figures/` tree from this bundle directly into the Overleaf thesis project root. Manual draw.io diagrams (`figures/chapter03/architecture.pdf`, `figures/chapter04/evaluation.pdf`) remain untouched and are specified separately in `DRAWIO_SPECIFICATIONS.md`.

## Visual & Editorial Conventions Applied (T1-E / T1-F)
- Unified palette: $V_2$ blue (`#0072B2`), $V_3$ red (`#C0392B`), consistent Grade 3 highlight (`#B02A24`).
- Single readable font family (`DejaVu Sans`), minimum rendered text $\ge 8$~pt at $\approx 16$~cm insertion width, Unicode mathematical minus (`−`) and arrow (`→`), and true subscripts.
- Every chart uses short and full LaTeX captions (`\caption[short]{full}`) so the List of Figures remains concise.
- `decoding.pdf` uses `[t]` placement to prevent single-figure float pages; stale `Cohorts: 1 aptos, 2 idrid...` text is removed.

---

## Chapter 5 Quantitative Figures

### 1. Archived Architectural Trade-off (`figures/chapter05/tradeoff.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1/g0_comparison.csv` (`split=target`, `metric in {recall_g3, within1}`, `reference` column) -> `figures/chapter05/tradeoff_data.csv`
- **Section / Anchor**: `Ordinal Modeling and the Severity Trade-off` (`\label{subsec:5-3-3}`), existing `\IfFileExists{figures/chapter05/tradeoff.pdf}` block
- **LaTeX Label**: `\label{fig:tradeoff}` (placement `[htbp]`)
- **Introductory Sentence**: `Figure~\ref{fig:tradeoff} compares cohort-level changes in exact Grade~3 recall and Within-1 accuracy from the archived nominal configuration ($V_2$) to the ordinal cumulative-link configuration ($V_3$).`
- **Short & Full Caption**:
```latex
\caption[Archived Grade~3 recall and Within-1 changes ($V_3 - V_2$)]{%
  Archived target changes from $V_2$ (nominal) to $V_3$ (ordinal CLM) across six held-out cohorts and their equal-cohort mean ($1/6$), reported to one decimal place in percentage points ($100 \times (V_3 - V_2)$).
  Exact Grade~3 recall declines in every cohort ($-12.0$~pp mean), whereas Within-1 accuracy improves in five of six cohorts ($+1.5$~pp mean, declining only on DeepDRiD).}
```

### 2. Latent Severity and Grade 3 Decision Regions (`figures/chapter05/latent-regions.pdf`)
- **Source Files**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/units/V3_fold_1_idrid/fold_report/decision_grid_float64_diagnostic.csv`, `decision_intervals.csv`, `selected_params.json`, `p0_fp32_main_lodo_v1/lodo/V3_fold_1_idrid/target/predictions.csv`, verified against `analysis_d_v3_latent_regions_per_fold.csv` -> `figures/chapter05/latent-regions_data.csv`
- **Section / Anchor**: `Decision Regions and Grade 3 Error Redistribution` (`\label{subsec:5-7-2}`), first paragraph
- **LaTeX Label**: `\label{fig:latent-regions}` (placement `[htbp]`)
- **Introductory Sentence**: `Figure~\ref{fig:latent-regions} aligns the empirical IDRiD target score distributions, cumulative-link class probabilities, and Grade~3 decision bands on a shared latent severity axis.`
- **Short & Full Caption**:
```latex
\caption[IDRiD $V_3$ latent severity distributions and Grade~3 decision regions]{%
  Latent severity mechanism for the IDRiD-held-out $V_3$ checkpoint ($n=516$ target images; horizontal axis zoomed to $[-3.30, 4.35]$ with 33 images above $4.35$: 27 Grade~4, 3 Grade~3, 2 Grade~2, and 1 Grade~0, maximum $s=7.06$).
  Top: within-grade normalized empirical target score densities. Middle: cumulative-link probabilities with cutpoints $\theta_0=-1.504$, $\theta_1=-0.468$, $\theta_2=0.605$, $\theta_3=1.552$ and peak diagnostic-grid $p_3=0.336$ (observed image maximum $0.336$).
  Bottom: Grade~3 prediction bands under source-locked CIBD ($\alpha=0.5, \tau=1.0, q=2$), raw latent cutpoints, and native rounded expectation; the native rule overlaps 46.8\% of the raw-cutpoint width.}
```

### 3. Native Grade 3 Recall on Source and Target (`figures/chapter05/source-target-g3.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`decoder=native_replay`, `V2` & `V3`, `source_val` vs. `target`) -> `figures/chapter05/source-target-g3_data.csv`
- **Section / Anchor**: `Source--Target Comparison and Latent Severity Distributions` (`\label{subsec:5-6-2}`) or `Joint Diagnostic Evidence` (`\label{subsec:5-10-1}`); add reference to Appendix D Figure~\ref{fig:app-source-target-folds}
- **LaTeX Label**: `\label{fig:source-target-g3}` (placement `[htbp]`)
- **Introductory Sentence**: `Figure~\ref{fig:source-target-g3} compares the equal-cohort mean native Grade~3 recall of replayed $V_2$ and $V_3$ checkpoints on source validation and held-out targets.`
- **Short & Full Caption**:
```latex
\caption[Mean native Grade~3 recall on source validation and targets]{%
  Equal-cohort six-fold mean Grade~3 recall (\%) for replayed native $V_2$ (argmax) and $V_3$ (rounded expectation) checkpoints on source validation ($61.9\%$ versus $30.9\%$) and held-out targets ($43.6\%$ versus $32.2\%$).
  The $-31.0$~pp ordinal deficit already present on source validation narrows on targets mainly because $V_2$ recall declines under transfer.}
```

### 4. Cohort-Level Effects of Primary CIBD (`figures/chapter05/decoding.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`split=target`, `decoder in {native_replay, cibd_selected}`, `V2` and `V3`) -> `figures/chapter05/decoding_data.csv`
- **Section / Anchor**: `Cost-interpolated Bayes decoding within E0` — `Cohort Robustness and Limits of Decision Intervention` (`\label{sec:5-8-4}`), existing conditional block
- **LaTeX Label**: `\label{fig:decoding}` (**use `[t]` placement**; remove obsolete `Cohorts: 1 aptos, 2 idrid...` text)
- **Introductory Sentence**: `Figure~\ref{fig:decoding} plots paired target changes in Grade~3 recall and Within-1 accuracy from replayed native decoding to primary source-selected CIBD across all 12 model--cohort combinations.`
- **Short & Full Caption**:
```latex
\begin{figure}[t]
\centering\includegraphics[width=.90\linewidth,height=.40\textheight,keepaspectratio]{figures/chapter05/decoding.pdf}
\caption[Target changes from native decoding to primary source-selected CIBD]{%
  Paired target changes ($100 \times (\mathrm{CIBD} - \mathrm{native})$) under primary source-selected CIBD (1.0~pp source allowance) across six held-out cohorts for $V_2$ (blue squares; equal-cohort mean $+10.10$~pp Grade~3 recall, $+0.99$~pp Within-1) and $V_3$ (red circles; $+8.85$~pp, $-1.16$~pp).
  Dashed lines mark the practical margins ($x = +3$~pp recall, $y = -1$~pp Within-1) with the jointly satisfied region shaded; all six $V_2$ cohorts improve on both endpoints, whereas 4 of 6 $V_2$ cohorts (Messidor-2 and DDR gain $<3$~pp recall) and 3 of 6 $V_3$ cohorts satisfy both margins.}
\label{fig:decoding}
\end{figure}
```

### 5. Frozen Probe Recall and Precision (`figures/chapter05/probing.pdf`)
- **Source CSVs**: `p0_fp32_main_lodo_v1_e1_resume_v2/probe_metrics.csv` (`split=target`, primary probes) + `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`native_replay`) -> `figures/chapter05/probing_data.csv`
- **Section / Anchor**: `Linear Accessibility in Frozen Features and Scalar Scores` (`\label{subsec:5-9-1}`), existing conditional block
- **LaTeX Label**: `\label{fig:probing}` (placement `[htbp]`)
- **Introductory Sentence**: `Figure~\ref{fig:probing} compares held-out Grade~3 recall and precision across the six cohorts for replayed native classifiers and primary source-trained E1 probes.`
- **Short & Full Caption**:
```latex
\caption[Target Grade~3 recall and precision under frozen-input probing]{%
  Held-out Grade~3 recall (left) and precision (right) across six cohorts (small markers) and equal-cohort means (diamonds) for replayed native classifiers and primary E1 linear probes selected by source-validation macro recall.
  The scalar $s_3$ probe raises recall in every cohort ($54.09\%$ mean versus $32.22\%$ native $V_3$, $51.09\%$ $\mathbf{z}_3$, and $44.64\%$ $\mathbf{z}_2$), while six-cohort mean precision remains comparable ($31.41\%$ versus $31.71\%$ native) amid wide cohort variation ($9.4\%$--$59.4\%$).}
```

### 6. Integrated Operating-Point Map (`figures/chapter05/operating-points.pdf`)
- **Source CSVs**: `thesis_supplementary_v1/analysis_a_summary.csv` and `p0_fp32_main_lodo_v1_e1_resume_v2/equal_cohort_means.csv` -> `figures/chapter05/operating-points_data.csv`
- **Section / Anchor**: `Joint Diagnostic Evidence` (`\label{subsec:5-10-1}`), existing block
- **LaTeX Label**: `\label{fig:operating-points}` (placement `[htbp]`)
- **Introductory Sentence**: `Figure~\ref{fig:operating-points} synthesizes absolute equal-cohort target Within-1 accuracy and Grade~3 recall across conventional E0 decoders, primary source-selected CIBD, and primary E1 probes.`
- **Short & Full Caption**:
```latex
\caption[Equal-cohort target operating points of E0 decoders and E1 probes]{%
  Absolute equal-cohort six-fold target means for replayed $V_2$ (blue) and $V_3$ (red) E0 decision rules and primary E1 linear probes.
  Coincident rules are deduplicated ($V_2$ native coincides with argmax; $V_3$ native coincides with rounded expectation), and latent-cutpoint decoding applies only to $V_3$ CLM.
  Points represent distinct mappings rather than a continuous training trajectory; the scalar $s_3$ probe was fitted on source labels in E1 outside the prespecified E0 decoder test.}
```

---

## Appendix Companions & Localization Reuse Note

### A1. Six-Fold Latent Region Representativeness (`figures/appendix/latent-regions-folds.pdf`)
- **Source CSV**: `thesis_supplementary_v1/analysis_d_v3_latent_regions_per_fold.csv`
- **Section / Anchor**: Appendix E (`\label{app:decoder_results}`), existing block `\label{fig:latent-regions-folds}`
- **Short & Full Caption**:
```latex
\caption[Grade~3 interval overlap and peak probability by fold (post hoc)]{%
  Post hoc six-fold $V_3$ decision-interval overlap (left, percentage of raw latent-cutpoint width) and peak Grade~3 probability (right, diagnostic-grid circles versus observed target-image crosses).
  IDRiD (diamonds; $46.8\%$ overlap, grid $\max p_3=0.336$) lies near the center of the cross-fold range ($22.2\%$--$63.0\%$ overlap; $\max p_3$ $0.243$--$0.482$).}
```

### A2. Fold-Level Source vs. Target Native Grade 3 Recall (`figures/appendix/source-target-folds.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`decoder=native_replay`, `V2` & `V3`, `source_val` vs. `target`)
- **Section / Anchor**: Appendix D (`\label{app:reproduction}`), after `Table~\ref{tab:ch5-p0}`
- **LaTeX Label**: `\label{fig:app-source-target-folds}`
- **Short & Full Caption**:
```latex
\caption[Fold-level native Grade~3 recall on source mixtures and targets]{%
  Per-fold Grade~3 recall (\%) for replayed native $V_2$ (left) and $V_3$ (right), comparing each fold's five-cohort source-validation mixture (filled circles) with the held-out target cohort (open diamonds).
  Connecting segments denote within-fold comparisons and do not follow the same patients or images across domains.}
```

### A3. Post Hoc 0.5 pp vs. Primary 1.0 pp CIBD Sensitivity (`figures/appendix/decoding-05pp.pdf`)
- **Source CSV**: `thesis_supplementary_v1/analysis_g_cibd_05pp_per_fold.csv` (`variant=V3`)
- **Section / Anchor**: Appendix E (`\label{appsec:cibd-budget-sensitivity}`), existing block `\label{fig:cibd-budget-sensitivity}`
- **Short & Full Caption**:
```latex
\caption[CIBD source-allowance sensitivity (post hoc)]{%
  Post hoc sensitivity comparison of $V_3$ CIBD target changes under the primary 1.0~pp source-validation allowance (open circles) versus a stricter post hoc 0.5~pp allowance (filled squares), connected within each cohort.
  Under the 0.5~pp allowance, 4 of 6 cohorts (APTOS~2019, IDRiD, Messidor-2, and DeepDRiD) satisfy both target margins ($+6.56$~pp mean Grade~3 recall, $-0.71$~pp Within-1).}
```

### R1. Localization Grid Reuse Note (`figures/chapter05/localization.pdf`)
- **Verified Original Asset**: `D:\Github\diabetic-retinopathy-grading\result\figure_8_xai_qualitative_grid.pdf` (underlying images and masks preserved unchanged; copy/rename to `figures/chapter05/localization.pdf` in Overleaf).
- **Section / Anchor**: `Localization Evidence and Failure Cases` (`\label{subsec:5-5-2}`), immediately after `Table~\ref{tab:ch5-localization}` (`\label{fig:localization-grid}`).
- **Short & Full Caption**:
```latex
\caption[Selected IDRiD localization pointing-game misses]{%
  Qualitative overlays from the archived 81-image IDRiD lesion-localization evaluation showing local gated attention and global Grad-CAM maps for five selected pointing-game misses \citep{nguyen2026clinically}.
  These illustrative failure cases highlight misalignment between peak saliency and annotated lesion masks rather than representing the full 81-image distribution.}
```

