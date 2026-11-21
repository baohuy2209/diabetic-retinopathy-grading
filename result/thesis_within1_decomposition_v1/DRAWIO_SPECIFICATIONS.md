# Draw.io Specifications & Code-Verified Fitting/Selection Table

This document provides exact, draw.io-ready specifications for the two manual architectural and protocol diagrams (`figures/chapter03/architecture.pdf` and `figures/chapter04/evaluation.pdf`) per supervisor reviews **T1-E** and **T1-F**, together with the code-verified four-row fitting and selection table for Chapter 4. No raster architecture or protocol images are generated; export both diagrams manually from draw.io as cropped vector PDFs without embedded diagram XML.

---

## 1. Shared Visual & Export Conventions (Apply to Both Diagrams)

| Setting | Specification |
| --- | --- |
| **Canvas width** | Design at `16.0 cm` (or `640 px` scaled to `\linewidth` $\approx 15.5\text{--}16.0\text{ cm}$ in LaTeX) so font sizes map 1:1 to print. |
| **Font & minimum size** | Single clean sans-serif family (`Helvetica`, `Arial`, or `DejaVu Sans`). Minimum rendered text size $\ge 8\text{ pt}$ (`9–10 pt` for box titles, `8–9 pt` for dimensions/annotations, `10–11 pt` for panel titles). |
| **Box density** | Maximum **2 short lines** per box; avoid paragraph text inside boxes. |
| **Unified thesis palette** | Neutral/shared inputs & backbone: gray (`#F2F4F4` fill, `#566573` border); $V_2$ nominal branch: blue (`#EBF5FB` fill, `#0072B2` border); $V_3$ ordinal CLM branch: red (`#FDEDEC` fill, `#C0392B` border); E1 frozen-probing enclosure: green dashed (`#1E8449` border, `#EAFAF1` tint); E0 decoder-swap enclosure: orange-red dashed (`#D35400` border, `#FEF5E7` tint); held-out target cohort: orange frame (`#E67E22` border, `#FEF5E7` tint). |
| **Grayscale legibility** | Every branch and grouping must remain distinguishable in black-and-white via **line style** (solid vs. dashed vs. dotted), **explicit row/container labels**, and **badges** (`V0+`, `V1+`, `V2+`, `V3`). |
| **Typography** | Use true subscripts (`z₂`, `z₃`, `s₃`, `θ₀–θ₃`, `V₀–V₃`, `ŷ`) via HTML/Math formatting in draw.io, and proper Unicode minus (`−`, U+2212) and arrows (`→`, U+2192). |
| **Export format** | `File → Export as → PDF…` with **Crop** enabled, **Transparent Background** off (white background), and **Include a copy of my diagram** unchecked (no embedded XML bloat). |

---

## 2. Chapter 3 Diagram: Architecture and Post-Training Interventions

- **Target file**: `figures/chapter03/architecture.pdf`
- **LaTeX label**: `\label{fig:architecture}` (in `chapters/03_methodology.tex`, replacing the existing `architecture.pdf` figure block)
- **Scope boundary**: Show the model architecture, variant progression (`V0–V3`), and frozen-checkpoint diagnostic interventions (`P0`, `E0`, `E1`) **only**. Do **not** draw LODO cohort splits or source/target data partitioning here (those belong exclusively to Figure `\ref{fig:evaluation}` in Chapter 4).
- **Overall geometry**: Two stacked horizontal panels inside a `16.0 cm × 13.5 cm` canvas (Panel A height `~7.2 cm`, Panel B height `~5.8 cm`, `0.5 cm` vertical gap).

### Panel A: Feature Extraction, Progressive Variant Structure, and Separate $V_2$ / $V_3$ Checkpoints

1. **Panel header**: `Panel A: Multi-scale feature extraction and head variants (V0–V3)` (`10.5 pt` bold, top-left).
2. **Left-to-right shared backbone specification** (neutral gray fill `#F2F4F4`, border `#566573`):
   - **Box A1 (Input)**: `Fundus image` / `RGB 512 × 512` (gray).
   - **Box A2 (Backbone)**: `ResNet-50 backbone` / `Global 2,048-D + local C3–C5` (gray; badge `V0+` in top-right corner).
   - **Parallel dual-branch detail** immediately to the right of Box A2:
     - **Upper branch (gray, badge `V0+`)**: `Global GAP branch` / `2,048-D → fused 4,096-D` (or concisely: `Global branch (GAP)` / `2,048-D`).
     - **Lower branch (gray, badge `V1+`)**: `Multi-scale gated attention` / `C3 (512) + C4 (1,024) + C5 (2,048)` → `Local 2,560-D` (note: combined multiscale projection + global yields `4,096 + 2,560 = 6,656-D`; keep labels crisp: Box A3a `Global + multi-scale fusion` `2,048-D (V0) / 4,608-D (V1)` and Box A3b `Attention-gated fusion (V2/V3)` `4,096 + 2,560 = 6,656-D → 512-D`).
     - Exact dimension trail to display clearly (verified in `models/dr_model.py` and `evaluation/checkpoint_reference/`):
       - `V0`: `2,048-D` global pooled feature (`GAP`) directly to nominal classifier.
       - `V1`: `4,608-D` multi-scale concatenated feature (`2,048 + 2,560`) without local attention gating.
       - `V2` / `V3`: `4,096 + 2,560 = 6,656-D` (global `GAP+GMP` `4,096-D` + gated local multi-scale `2,560-D`) projected via linear+BN+ReLU+dropout to penultimate feature **`512-D`** (`z₂` for $V_2$, `z₃` for $V_3$).
   - **Shared-architecture vs. separate-weights bracket** (CRITICAL T1-F §1 requirement):
     - Draw a clean dashed outer frame spanning **both the backbone and head** for the two final rows, or place an explicit horizontal callout banner directly beneath the backbone-to-head split:
       > `V2 and V3 share architecture up to 512-D z, but are trained as two separate checkpoints (distinct backbone + head weights)`
     - This prevents readers from mistakenly inferring that $V_2$ and $V_3$ share one frozen backbone.
3. **Two stacked terminal head rows on the right** (clearly separated horizontally with row labels on the left of each row):
   - **Row 1 — Blue (`#EBF5FB` fill, `#0072B2` border, solid lines; row tag `V2 (Nominal, badge V2+)`)**:
     - `z₂ (512-D)` → `Linear head (5 logits)` → `Softmax → p = (p₀,…,p₄)` (merge Softmax and probabilities into a single clean box) → `Native rule: argmax` → `ŷ ∈ {0,…,4}`.
   - **Row 2 — Red (`#FDEDEC` fill, `#C0392B` border, solid lines; row tag `V3 (Ordinal CLM, badge V3)`)**:
     - `z₃ (512-D)` → **CLM head group** (light red rounded container titled `Cumulative-Link Model (CLM)`):
       - Inside CLM container: `Latent score s₃ = wᵀz₃` (scalar) → `cloglog link + θ₀–θ₃ → p = (p₀,…,p₄)` (explicitly keep `s₃ = wᵀz₃` inside the CLM grouping, followed by the cloglog cumulative-link mapping with ordered cutpoints `θ₀ < θ₁ < θ₂ < θ₃` producing `p`).
     - Exiting CLM container → `Native rule: round(E[Y | x])` (note: native $V_3$ uses rounded posterior expectation; raw latent cutpoints are a separate E0 decoder shown in Panel B) → `ŷ ∈ {0,…,4}`.

### Panel B: Frozen-Checkpoint Post-Training Interventions (`P0`, `E0`, `E1`)

1. **Panel header**: `Panel B: Frozen-checkpoint replay (P0), decision decoding (E0), and linear probing (E1)` (`10.5 pt` bold, top-left).
2. **Top provenance strip (neutral gray, full width)**:
   - `P0 Frozen-Checkpoint Replay (FP32, no backbone/head retraining): caches z₂, z₃, s₃, θ₀–θ₃, and p for every image`
   - Downward thin gray arrows drop from this P0 cache strip into the two aligned model lanes below.
3. **Two horizontally aligned frozen-output lanes** (matching Panel A colors and row order):
   - **Upper lane ($V_2$, blue)**: `z₂ (512-D)` → `p = (p₀,…,p₄)` → `[E0 Decoder]` → `ŷ`
   - **Lower lane ($V_3$, red)**: `z₃ (512-D)` → `s₃ (1-D) & θ₀–θ₃` → `p = (p₀,…,p₄)` → `[E0 Decoder]` → `ŷ`
4. **Two non-overlapping dashed intervention enclosures**:
   - **Left enclosure — Green dashed (`#1E8449`, fill `#EAFAF1`, label `E1: Frozen-Input Linear Probing`)**:
     - Encloses frozen `z₂`, `z₃`, and `s₃` (with a small snowflake/padlock icon or `(frozen)` tag).
     - Downward green arrows feed into a compact E1 box at the bottom-left:
       `StandardScaler + L2 LogisticRegression` / `Fitted on source train; C selected on source val` → `Probe ŷ (z₂, z₃, s₃)`.
   - **Right enclosure — Orange-red dashed (`#D35400`, fill `#FEF5E7`, label `E0: Fixed-Output Decoder Swap`)**:
     - Encloses the decoder stage (`p → ŷ`) for both lanes, with no parameter refitting of `z` or `s`.
     - Inside the E0 box, list the five decision rules compactly:
       - `argmax` ($V_2$ native) · `posterior median` · `rounded E[Y]` ($V_3$ native)
       - `latent cutpoints` (**$V_3$ only**) · `source-selected CIBD (α, τ, q)`
     - **CRITICAL T1-F §1 arrow**: Draw a **dashed red bypass arrow** directly from `s₃ & θ₀–θ₃` (skipping `p`) into the E0 decoder box labeled `cutpoints (V3 only)` so the reader sees that raw latent-cutpoint decoding acts directly on `(s₃, θ₀–θ₃)` rather than on `p`.

### Copy-Ready LaTeX Block & Short/Full Caption for `fig:architecture`

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/chapter03/architecture.pdf}
\caption[Model progression ($V_0$--$V_3$) and frozen-checkpoint interventions (P0, E0, E1)]{%
  Architecture and post-training intervention framework.
  \textbf{Panel~A}: Progressive feature-extraction variants ($V_0$--$V_3$) culminating in two separately trained checkpoints ($V_2$ nominal softmax head versus $V_3$ ordinal cumulative-link head with scalar score $s_3 = \mathbf{w}^\top \mathbf{z}_3$ and ordered cutpoints $\theta_0$--$\theta_3$); both share the 6,656-D to 512-D dual-branch architecture but hold distinct trained backbone and head weights.
  \textbf{Panel~B}: Frozen-checkpoint replay (P0) caches $(\mathbf{z}, s, \boldsymbol{\theta}, \mathbf{p})$ for fixed-output decoder replacement (E0, including direct latent-cutpoint decoding on $V_3$ only and source-selected CIBD) and frozen-input linear probing (E1 on $\mathbf{z}_2$, $\mathbf{z}_3$, and $s_3$).}
\label{fig:architecture}
\end{figure}
```

---

## 3. Chapter 4 Diagram: Six-Fold LODO Evaluation and Source-Only Selection Protocol

- **Target file**: `figures/chapter04/evaluation.pdf`
- **LaTeX label**: `\label{fig:evaluation}` (in `chapters/04_experimental_design.tex`, replacing the existing `evaluation.pdf` figure block)
- **Overall geometry**: Horizontal flow divided by an explicit **horizontal source/target leakage barrier** (`16.0 cm × 10.5 cm`).

### Exact Box-by-Box Layout

1. **Top Zone — Source Domain Container** (neutral gray rounded container `#F8F9F9`, border `#566573`, title `Source domain: five training cohorts (per LODO fold)`):
   - **Box E1 (Left)**: `Five source cohorts` / `Pooled source inventory` (gray).
   - **Box E2 (Split)**: `Grouped 85/15 split` / `by recorded grouping key` (gray; note in small italic below: `no grouping key shared across splits`).
   - **Two horizontal branches exiting Box E2**:
     - **Upper branch — Source training**:
       - **Box E3a**: `Class-balanced subsampling` / `(training branch only; alters prevalence, preserves grouping)` (dashed blue/gray accent border to highlight that subsampling applies ONLY to the training partition and never to validation or target).
       - **Box E3b**: `Source training partition` / `Fit V0–V3 weights, CIBD prior π̂, E1 scaler & probe` (distinct from selection!).
     - **Lower branch — Source validation**:
       - **Box E4a**: `Source validation partition` / `(unsubsampled natural source mixture)`.
       - **Box E4b**: `Source-only selection` / `Select checkpoints (QWK), RDR τ, CIBD (α, τ, q), E1 C` (distinct box downstream of `Source validation partition`, receiving fitted objects from Box E3b via a vertical/diagonal downward arrow labeled `fitted models / priors / probes`).

2. **Horizontal Source/Target Boundary Line** (full width, thick dashed crimson/dark-gray line `#7B241C` across the middle of the canvas):
   - Center label directly on the line: **`No target images or labels cross this line`**.
   - **CRITICAL T1-F §2 rule**: **ONLY ONE arrow** crosses this horizontal barrier from top to bottom:
     - Arrow exits `Box E4b (Source-only selection)` → passes through a bold badge box **`Lock evaluation configuration`** straddling the barrier → enters `Target inference` in the Bottom Zone.
     - Verify visually that zero arrows cross upward from Target to Source and zero secondary arrows bypass `Lock evaluation configuration`.

3. **Bottom Zone — Held-Out Target Container** (warm orange rounded container `#FEF5E7`, border `#E67E22`, title `Target domain: one held-out cohort (unseen during training and selection)`):
   - **Box E5 (Left, entering its own independent lane from the left)**: `Held-out target cohort` / `Images x (1 of 6 cohorts)` (orange fill/border).
   - **Two separate sub-lanes exiting Box E5**:
     - **Upper target lane (Images only)**: `Held-out target images` → **Box E6 (`Frozen target inference`)** (receives the single downward arrow from `Lock evaluation configuration`) → produces `Target predictions ŷ, p, s`.
     - **Lower target lane (Labels isolated)**: `Target reference grades y` enters from the left along the bottom edge and connects **exclusively** into **Box E7 (`Metric calculation`)** — never into inference, calibration, or selection.
   - **Box E7 (`Per-cohort target evaluation`)**: Receives `Target predictions` from Box E6 and `Target reference grades y` from the isolated lower lane → outputs per-cohort metrics (`QWK, MAE, Within-1, Grade 3 recall & precision, RDR sensitivity/specificity`).

4. **Right-Side Aggregation Column** (spanning the bottom-right exit from Box E7):
   - **Loop badge** attached to the outer LODO frame or between Box E7 and Box E8: **`×6 LODO folds`** (each of APTOS 2019, IDRiD, Messidor-2, DDR, EyePACS, and DeepDRiD held out once; 54,883 evaluated target images total).
   - **Box E8 (`Cross-domain summary`)**: `Per-cohort metrics + equal-cohort mean (1/6 weight)` / `(primary target aggregation; image-pooled reported separately)`.

### Copy-Ready LaTeX Block & Short/Full Caption for `fig:evaluation`

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/chapter04/evaluation.pdf}
\caption[Six-fold LODO evaluation and source-only selection protocol]{%
  Six-fold leave-one-dataset-out (LODO) evaluation and source-only model, decoder, and probe selection protocol.
  In each fold, five source cohorts undergo a grouped 85/15 split by recorded grouping key; class-balanced subsampling is applied exclusively to the source training partition (adjusting class prevalence while preserving grouping separation), while the unsubsampled source validation partition selects checkpoints, RDR thresholds, CIBD parameters, and probe regularization.
  Only the locked configuration crosses the horizontal boundary to the held-out target cohort, whose reference grades enter metric computation alone; results are reported per cohort and as the equal-cohort ($1/6$) mean across all six folds (a separate pooled six-cohort split is used solely for the head benchmark in Section~\ref{sec:5-2}).}
\label{fig:evaluation}
\end{figure}
```

---

## 4. Code-Verified Four-Row Fitting and Selection Table (Chapter 4)

Verified directly against repository source files:
- `V0–V3` training/checkpointing: `workflows/dmip/01_main_lodo_ablation.py` (and `training/trainer.py`)
- `RDR screening` threshold: `screening_threshold()` in `workflows/mlwa/01_E0_decision_optimization_resume_v2.py` and `workflows/mlwa/02_E1_representation_probes_resume_v2.py`
- `E0 CIBD` prior & selection: `prior_from_labels()`, `cibd()`, and `select_cibd()` in `workflows/mlwa/01_E0_decision_optimization_resume_v2.py`
- `E1 probe` fitting & selection: `StandardScaler` + L2 `LogisticRegression` and `macro_recall` selection in `workflows/mlwa/02_E1_representation_probes_resume_v2.py`

### Copy-Ready LaTeX Table (`tab:ch4-fitting-selection`)

```latex
\begin{table}[htbp]
\centering\small
\setlength{\tabcolsep}{4.5pt}
\caption[Source-training fitting and source-validation selection across study stages]{%
  Separation of parameter fitting (source training partition) and hyperparameter or rule selection (unsubsampled source validation partition) across all four primary study stages before locking configurations for held-out target evaluation.
  Post hoc sensitivity variants (the QWK-selected E1 probe and the 0.5~pp-allowance E0 CIBD rule) are analyzed separately from these prespecified primary rules.}
\label{tab:ch4-fitting-selection}
\begin{tabular}{@{}p{2.3cm}p{4.8cm}p{5.0cm}p{2.5cm}@{}}
\toprule
Stage & Fitted on source training & Selected on source validation & Locked for target \\
\midrule
$V_0$--$V_3$ models &
Backbone, fusion, and head parameters (after grouped 85/15 split and class-balanced training subsampling). &
Best training epoch checkpoint maximizing quadratic weighted kappa (QWK). &
Frozen FP32 checkpoint weights. \\
\addlinespace[4pt]
RDR screening &
None (uses frozen model or probe posterior $P(Y \geq 2)$). &
Probability threshold maximizing sensitivity subject to specificity $\geq 95\%$ (ties: higher specificity, then higher threshold). &
Scalar screening threshold $\tau_{\mathrm{RDR}}$. \\
\addlinespace[4pt]
E0 CIBD &
Laplace-smoothed training class prior $\hat{\pi}_c$ (\texttt{smoothing=1.0}) computed on the subsampled source training labels. &
Grid $(\alpha, \tau, q)$ maximizing exact Grade~3 recall subject to Within-1 loss $\leq 1.0$~pp vs.\ native replay (ties: higher Within-1, higher QWK, smaller $\alpha$, $|\tau - 1|$, smaller $q$). &
Prior $\hat{\pi}$ and triple $(\alpha^\star, \tau^\star, q^\star)$. \\
\addlinespace[4pt]
E1 linear probe &
Per-feature \texttt{StandardScaler} parameters and L2-regularized multinomial logistic regression weights on frozen $\mathbf{z}_2$, $\mathbf{z}_3$, or $s_3$. &
Regularization strength $C \in \{10^{-4},\dots,10^{2}\}$ maximizing five-class macro recall. &
Frozen scaler, weights, and $C^\star$. \\
\bottomrule
\end{tabular}
\end{table}
```

> **Note on post hoc variants (for adjacent prose in Section 4.9 / Table footnote)**: Two post hoc sensitivity rules use the same source-only leakage barrier while changing only the source-validation selection criterion: (i) the post hoc E1 probe selects $C$ by source-validation QWK rather than macro recall, and (ii) the post hoc E0 CIBD sensitivity rule tightens the source-validation Within-1 loss allowance from $1.0\text{ pp}$ to $0.5\text{ pp}$ (`workflows/mlwa/03_thesis_supplementary_analyses.py`, `workflows/mlwa/05_within1_transition_decomposition.py`). Neither replaces the prespecified primary rules in hypothesis testing.

