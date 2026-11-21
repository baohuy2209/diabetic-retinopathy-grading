# Figure Integration Guide (`result/thesis_figures_v1/`)

Copy the `figures/` tree from this bundle directly into the Overleaf thesis project root. Draw.io placeholders (`figures/chapter03/architecture.pdf`, `figures/chapter04/evaluation.pdf`) were intentionally left untouched.

## Summary & Discrepancy Notes vs. `UEL_Thesis_official_v1.zip`

1. **Chart 3 (`source-target-g3.pdf`)**: `chapters/05_results_discussion.tex` currently states the four native Grade 3 recall means (`61.9%`, `30.9%`, `43.6%`, `32.2%`) only in prose (§5.6.2 line 295 and §5.10.1 line 467), with no compact summary table. Therefore **both** the compact Chapter 5 paired-dot chart (`figures/chapter05/source-target-g3.pdf`) and the fold-level Appendix D companion (`figures/appendix/source-target-folds.pdf`) were generated.
2. **Chart 4 (`decoding.pdf`) vs. existing `chapters/05_results_discussion.tex` / `FIGURES_vi.md`**: The existing ZIP block (`subsec:5-8-4`, `\label{fig:decoding}`) describes only $V_3$ (`6` cohorts) and `FIGURES_vi.md` swaps x/y (`X=delta Within-1, Y=delta G3 recall`). Per supervisor feedback §7 and specification, the generated `decoding.pdf` plots **all 12 $V_2$ and $V_3$ cohort points** with **$x = \Delta$ Grade 3 recall (pp)** and **$y = \Delta$ Within-1 accuracy (pp)**, practical margins at $x=+3$~pp and $y=-1$~pp, and equal-cohort means. Update the LaTeX caption accordingly using the copy-ready caption below.
3. **Localization reuse asset**: Original figure verified at `D:\Github\diabetic-retinopathy-grading\result\figure_8_xai_qualitative_grid.pdf`; copy/rename it to `figures/chapter05/localization.pdf` in Overleaf.

---

## Chapter 5 Quantitative Figures

### 1. Archived Architectural Trade-off (`figures/chapter05/tradeoff.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1/g0_comparison.csv` (`split=target`, `metric in {recall_g3, within1}`, `reference` column) -> `figures/chapter05/tradeoff_data.csv`
- **Section Title**: `5.3.3 Ordinal Modeling and the Severity Trade-off` (`\label{subsec:5-3-3}`)
- **Manuscript Block Status**: Existing `\IfFileExists{figures/chapter05/tradeoff.pdf}` block in `chapters/05_results_discussion.tex` (lines 114–120)
- **Insertion Anchor**: Immediately after `\label{subsec:5-3-3}` heading
- **LaTeX Label**: `\label{fig:tradeoff}`
- **Introductory Sentence**: `Figure~\ref{fig:tradeoff} compares the cohort-level changes in Grade~3 recall and Within-1 accuracy from the archived nominal configuration ($V_2$) to the ordinal cumulative-link configuration ($V_3$).`
- **Self-Contained Caption**: `Archived target changes from $V_2$ (nominal) to $V_3$ (ordinal CLM) across six held-out cohorts and their equal-cohort mean ($1/6$), in percentage points ($100 \times (V_3 - V_2)$). Exact Grade~3 recall drops in every cohort ($-11.98$~pp mean) while Within-1 accuracy improves in five of six cohorts ($+1.46$~pp mean, declining only on DeepDRiD).`

### 2. Latent Severity and Grade 3 Decision Regions (`figures/chapter05/latent-regions.pdf`)
- **Source Files**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/units/V3_fold_1_idrid/fold_report/decision_grid_float64_diagnostic.csv`, `decision_intervals.csv`, `selected_params.json`, `p0_fp32_main_lodo_v1/lodo/V3_fold_1_idrid/target/predictions.csv`, verified against `analysis_d_v3_latent_regions_per_fold.csv` -> `figures/chapter05/latent-regions_data.csv`
- **Section Title**: `5.7.2 Decision Regions and Grade 3 Error Redistribution` (`\label{subsec:5-7-2}`)
- **Manuscript Block Status**: **New** figure block (wrap with `\IfFileExists{figures/chapter05/latent-regions.pdf}{...}{}`)
- **Insertion Anchor**: First paragraph of `\subsection{Decision Regions and Grade 3 Error Redistribution}` (`\label{subsec:5-7-2}`)
- **LaTeX Label**: `\label{fig:latent-regions}`
- **Introductory Sentence**: `Figure~\ref{fig:latent-regions} aligns the empirical IDRiD score distributions, cumulative-link class probabilities, and resulting Grade~3 decision bands on a shared latent severity axis.`
- **Self-Contained Caption**: `Latent severity mechanism for the IDRiD-held-out $V_3$ checkpoint ($n=516$ target images; horizontal view shown over $[-4, 5]$ with Grade~4 extending to $7.06$). Top: within-grade normalized empirical target score densities. Middle: cumulative-link class probabilities showing cutpoints $\theta_0=-1.504$, $\theta_1=-0.468$, $\theta_2=0.605$, $\theta_3=1.552$ and peak diagnostic-grid $p_3=0.336$ (observed image maximum $0.336$). Bottom: Grade~3 prediction intervals under source-locked CIBD ($\alpha=0.5, \tau=1.0, q=2$), raw latent cutpoints, median, argmax and native rounded expectation; the native rule overlaps 46.8\% of the raw-cutpoint interval width.`

### 3. Native Grade 3 Recall on Source and Target (`figures/chapter05/source-target-g3.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`decoder=native_replay`, `variant in {V2, V3}`, `split in {source_val, target}`) -> `figures/chapter05/source-target-g3_data.csv`
- **Section Title**: `5.6.2 Source--Target Comparison and Latent Severity Distributions` (`\label{subsec:5-6-2}`)
- **Manuscript Block Status**: **New** compact figure block (placed at §5.6.2 since Chapter 5 currently states these four means only in prose)
- **Insertion Anchor**: End of paragraph 1 of `\subsection{Source--Target Comparison and Latent Severity Distributions}` (`\label{subsec:5-6-2}`)
- **LaTeX Label**: `\label{fig:source-target-g3}`
- **Introductory Sentence**: `Figure~\ref{fig:source-target-g3} summarizes the equal-cohort mean Grade~3 recall of replayed $V_2$ and $V_3$ checkpoints on source validation and held-out targets.`
- **Self-Contained Caption**: `Equal-cohort six-fold mean Grade~3 recall (%) for replayed native $V_2$ (argmax) and $V_3$ (rounded expectation) checkpoints on source validation ($61.9\%$ vs.\ $30.9\%$) and held-out target cohorts ($43.6\%$ vs.\ $32.2\%$). The $-31.0$~pp ordinal deficit already present on source validation indicates the trade-off does not arise solely from cross-domain transfer.`

### 4. Cohort-Level Effects of CIBD (`figures/chapter05/decoding.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`split=target`, `decoder in {native_replay, cibd_selected}`, `V2` and `V3`) -> `figures/chapter05/decoding_data.csv`
- **Section Title**: `5.7.6 Cost-interpolated Bayes decoding within E0` — `Cohort Robustness and Limits of Decision Intervention` (`\label{sec:5-8-4}`)
- **Manuscript Block Status**: **Existing conditional block** (`\IfFileExists{figures/chapter05/decoding.pdf}`) in `chapters/05_results_discussion.tex` (lines 409–415) — **caption update needed** to reflect both $V_2$ and $V_3$ points and updated x/y axes
- **Insertion Anchor**: Beginning of `\subsubsection{Cohort Robustness and Limits of Decision Intervention}` (`chapters/05_results_discussion.tex`, line 409)
- **LaTeX Label**: `\label{fig:decoding}`
- **Introductory Sentence**: `Figure~\ref{fig:decoding} plots the paired target changes in Grade~3 recall and Within-1 accuracy under primary source-selected CIBD relative to replayed native decoding across all 12 model--cohort combinations.`
- **Self-Contained Caption**: `Target changes from replayed native decoding to primary source-selected CIBD (1.0~pp source budget) across all six held-out cohorts for $V_2$ (squares; equal-cohort mean $+10.10$~pp Grade~3 recall, $+0.99$~pp Within-1) and $V_3$ (circles; $+8.85$~pp, $-1.16$~pp). Dashed lines mark the prespecified practical margins ($x = +3$~pp Grade~3 recall, $y = -1$~pp Within-1 accuracy), with the jointly satisfied region lightly shaded; 4 of 6 $V_2$ cohorts (Messidor-2 and DDR fall below $+3$~pp recall despite positive changes in both metrics) and 3 of 6 $V_3$ cohorts satisfy both margins.`

### 5. Frozen Probe Recall and Precision (`figures/chapter05/probing.pdf`)
- **Source CSVs**: `p0_fp32_main_lodo_v1_e1_resume_v2/probe_metrics.csv` (`split=target`, `control_seed` empty, primary probes) paired with `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`native_replay`) -> `figures/chapter05/probing_data.csv`
- **Section Title**: `5.8.1 Linear Accessibility in Frozen Features and Scalar Scores` (`\label{subsec:5-9-1}`)
- **Manuscript Block Status**: **Existing conditional block** (`\IfFileExists{figures/chapter05/probing.pdf}`) in `chapters/05_results_discussion.tex` (lines 424–430)
- **Insertion Anchor**: Immediately after `\label{subsec:5-9-1}` (`chapters/05_results_discussion.tex`, line 424)
- **LaTeX Label**: `\label{fig:probing}`
- **Introductory Sentence**: `Figure~\ref{fig:probing} compares target Grade~3 recall and precision across all six held-out cohorts for replayed native classifiers and source-trained E1 linear probes.`
- **Self-Contained Caption**: `Target Grade~3 recall (left) and Grade~3 precision (right) across six held-out cohorts (small markers) and their equal-cohort means (diamonds) for replayed native classifiers and primary E1 probes selected by source-validation macro recall. The $s_3$ scalar probe raises recall in every cohort ($54.09\%$ mean vs.\ $32.22\%$ native $V_3$, $51.09\%$ $\mathbf{z}_3$, and $44.64\%$ $\mathbf{z}_2$), while six-cohort mean precision remains comparable across $V_3$ rules ($31.41\%$ vs.\ $31.71\%$ native) amid wide cohort variation ($9.4\%$--$59.4\%$).`

### 6. Integrated Operating-Point Map (`figures/chapter05/operating-points.pdf`)
- **Source CSVs**: `thesis_supplementary_v1/analysis_a_summary.csv` and `p0_fp32_main_lodo_v1_e1_resume_v2/equal_cohort_means.csv` -> `figures/chapter05/operating-points_data.csv`
- **Section Title**: `5.9.1 Joint Diagnostic Evidence` (`\label{subsec:5-10-1}`)
- **Manuscript Block Status**: **Existing commented block** (`% \begin{figure}...`) in `chapters/05_results_discussion.tex` (lines 487–491) — uncomment and wrap with `\IfFileExists` or insert directly
- **Insertion Anchor**: End of `\subsection{Joint Diagnostic Evidence}` (`chapters/05_results_discussion.tex`, line 487)
- **LaTeX Label**: `\label{fig:operating-points}`
- **Introductory Sentence**: `Figure~\ref{fig:operating-points} synthesizes the equal-cohort target performance of conventional E0 rules, primary source-selected CIBD, and primary E1 probes on shared Within-1 and Grade~3 recall axes.`
- **Self-Contained Caption**: `Equal-cohort six-fold target means for replayed $V_2$ (blue) and $V_3$ (orange-red) decision rules and source-trained E1 probes. Coincident rules are deduplicated ($V_2$ native coincides with argmax; $V_3$ native coincides with rounded expectation), and latent-cutpoint decoding applies only to $V_3$ CLM. Markers denote rule families rather than a continuous training trajectory; the scalar $s_3$ probe was fitted on source labels in E1 outside the prespecified E0 decoder test.`

---

## Appendix Companions & Localization Reuse Note

### A1. Six-Fold Latent Region Representativeness (`figures/appendix/latent-regions-folds.pdf`)
- **Source CSV**: `thesis_supplementary_v1/analysis_d_v3_latent_regions_per_fold.csv`
- **Section / Anchor**: Appendix G (`\label{app:derivations}`), after `\section{Cumulative probabilities and latent boundaries}` (or Appendix E end)
- **LaTeX Label**: `\label{fig:app-latent-regions-folds}`
- **Introductory Sentence**: `Figure~\ref{fig:app-latent-regions-folds} compares the native--cutpoint Grade~3 interval overlap and peak Grade~3 probability across all six $V_3$ folds to contextualize the IDRiD example.`
- **Self-Contained Caption**: `Six-fold $V_3$ decision-interval overlap fraction (left, percent of raw latent-cutpoint width) and peak Grade~3 probability (right, diagnostic grid circles vs.\ observed target-image maxima crosses). IDRiD (diamonds; $46.8\%$ overlap, grid $\max p_3=0.336$) lies near the center of the cross-fold range ($22.2\%$--$63.0\%$ overlap; $\max p_3$ $0.243$--$0.482$).`

### A2. Fold-Level Source vs. Target Native Grade 3 Recall (`figures/appendix/source-target-folds.pdf`)
- **Source CSV**: `p0_fp32_main_lodo_v1_e0_resume_v2/lodo/decision_metrics.csv` (`decoder=native_replay`, `V2` & `V3`, `source_val` vs. `target`)
- **Section / Anchor**: Appendix D (`\label{app:reproduction}`), end of `\section{Scope of the replay evidence}` after `Table~\ref{tab:ch5-p0}`
- **LaTeX Label**: `\label{fig:app-source-target-folds}`
- **Introductory Sentence**: `Figure~\ref{fig:app-source-target-folds} details the per-fold native Grade~3 recall on each fold's five-cohort source-validation mixture alongside its held-out target cohort.`
- **Self-Contained Caption**: `Per-fold Grade~3 recall (%) for replayed native $V_2$ (left) and $V_3$ (right) comparing the five-cohort source-validation mixture of each fold (filled circles) with the corresponding held-out target cohort (open diamonds). Connecting segments denote within-fold comparisons only and do not follow the same patients or images across domains.`

### A3. Post Hoc 0.5 pp vs. Primary 1.0 pp CIBD Sensitivity (`figures/appendix/decoding-05pp.pdf`)
- **Source CSV**: `thesis_supplementary_v1/analysis_g_cibd_05pp_per_fold.csv` (`variant=V3`)
- **Section / Anchor**: Appendix E (`\label{app:decoder_results}`), end of `\section{Parameter selection and transfer}`
- **LaTeX Label**: `\label{fig:app-decoding-05pp}`
- **Introductory Sentence**: `Figure~\ref{fig:app-decoding-05pp} contrasts the primary 1.0~pp source-budget $V_3$ CIBD operating points with the post hoc 0.5~pp sensitivity selections across all six held-out cohorts.`
- **Self-Contained Caption**: `Post hoc sensitivity comparison of $V_3$ CIBD target changes under the primary 1.0~pp source-validation Within-1 budget (open circles) versus a stricter post hoc 0.5~pp budget (filled circles), connected within each cohort. Under the post hoc 0.5~pp budget, 4 of 6 cohorts (APTOS~2019, IDRiD, Messidor-2, and DeepDRiD) meet both practical target margins ($+6.56$~pp mean Grade~3 recall, $-0.71$~pp Within-1).`

### R1. Localization Grid Reuse Note (`figures/chapter05/localization.pdf`)
- **Verified Original Asset**: `D:\Github\diabetic-retinopathy-grading\result\figure_8_xai_qualitative_grid.pdf`
- **Section / Anchor**: `5.5.2 Localization Evidence and Failure Cases` (`\label{subsec:5-5-2}`), immediately after `Table~\ref{tab:ch5-localization}` (`chapters/05_results_discussion.tex`, lines 269–273)
- **Manuscript Block Status**: Existing commented block (`% \begin{figure}...\label{fig:localization-grid}`) — copy verified asset to `figures/chapter05/localization.pdf` in Overleaf, uncomment, and replace caption.
- **Introductory Sentence**: `Figure~\ref{fig:localization-grid} illustrates five representative pointing-game failure cases drawn from the 81-image IDRiD lesion-localization evaluation.`
- **Self-Contained Caption**: `Qualitative saliency examples from the archived IDRiD lesion-localization evaluation (81 matched crops) showing local gated attention and global Grad-CAM overlays for five selected pointing-game misses \citep{nguyen2026clinically}. These illustrative failure cases highlight misalignment between top saliency responses and annotated lesion masks rather than the full 81-image distribution.`

