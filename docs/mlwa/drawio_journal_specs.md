# Draw.io Vector Diagram Specifications — MLWA Journal Extension

Publication-ready structural specifications for author-drawn vector diagrams in draw.io (`File -> Export as -> PDF`, cropped, embedded fonts, vector text). These complement — and stay separate from — the measured quantitative matplotlib charts produced by `experiments/mlwa/reporting.py` and the graduation-thesis figures.

---

## Diagram J1 — End-to-End Experimental Architecture (`RQ1`–`RQ4` & Protocol Map)

- **Target filename**: `figures/journal/protocol_overview.pdf`
- **Canvas**: 16:9 landscape (`1120 × 620 px`), 300 DPI / vector PDF, sans-serif (`Arial` or `Liberation Sans`, 10–12 pt), colorblind-safe Okabe–Ito accents (`#0072B2` Blue = Nominal/Softmax, `#D55E00` Vermilion = Ordinal/CLM, `#009E73` Bluish Green = Decision/Bayes, `#E69F00` Orange = Gated/Opt-in, `#6B7280` Neutral Gray = Frozen Assets).
- **Columns (Left -> Right)**:
  1. **Data & Split Contract (`LODO 6-Fold`)**
     - Box 1A: *Six Retinal Cohorts* (`APTOS`, `IDRiD`, `Messidor-2`, `DDR`, `EyePACS`, `DeepDRiD`; `55,570` inventory / `54,883` LODO targets; grouped keys `35,474`, status `legacy_unverified`).
     - Box 1B: *Leakage-Free Partition (`seed_split = 42`)* -> `natural_source_train` (85% grouped, exact natural prior $\pi_{\text{source\_train}}$) | `source_val` (15% grouped, model/CIBD/`C`/RDR selection ONLY) | Held-out `target` (100% of excluded cohort, zero leakage).
  2. **Stage 1: Controlled 2×2 Head × Objective Factorial (`E2`, `RQ2`, 72 GPU units)**
     - Shared Dual-Branch Backbone + Fusion (`512-D` $\mathbf{z}$, SHA-256 matched init per `(fold_id, seed_train)` across Cells A–D, `AdamW`, `CosineAnnealingLR`, symmetric `15`-epoch QWK warmup `0.0 -> 0.4`).
     - 2×2 Grid:
       - `Cell A`: Softmax + CE ($\lambda=0$)
       - `Cell B`: Softmax + CE + $\lambda(t)\mathcal{L}_{\mathrm{QWK}}$
       - `Cell C`: CLM(`cloglog`) + pure NLL ($\lambda=0$)
       - `Cell D`: CLM(`cloglog`) + NLL + $\lambda(t)\mathcal{L}_{\mathrm{QWK}}$
     - Side branches (Dashed gray connectors = reused baselines):
       - `E2b (18 units)`: `CLM(logit)` vs `CLM(cloglog)` link ablation (reuses `E2-D`).
       - `E4 (36 units)`: `Cui CB-CE` ($\beta=0.9999$) & `Menon Logit Adj` ($\tau=1.0$) (reuses `E2-A`).
  3. **Stage 2: Fixed-Posterior Decision & Representation Audits (`RQ1`, `RQ3`)**
     - Box 3A (`E0`, CPU): Decoders on locked probabilities $\mathbf{p}(x)$ — `argmax`, `posterior_median`, `rounded_expectation_bayes`, `cutpoint_latent` (CLM only), source-selected CIBD (`journal_dual_guardrail_v1` & `legacy_within1_1pp_v1`), RDR screening (`95% source_val spec`), plus `EXPLORATORY` target grid dashed separately.
     - Box 3B (`E1`, CPU): Frozen-encoder multinomial logistic probes on `z_nominal` (`512-D`), `z_ordinal` (`512-D`), `s_ordinal` (`1-D`), `source_train`-fitted scaler/weights, `source_val`-selected `C`, within-cohort permutations (`seed_probe_control`).
  4. **Stage 3: Gated & Opt-In Extensions (`RQ4`)**
     - Diamond Gate `H3` (`E3 BRC`): *Closed by default (`SKIPPED_GATE_NOT_MET`)* unless multi-seed paired $\mathbf{z}_{\mathrm{ord}} > s_{\mathrm{ord}}$ across $\ge 5/6$ cohorts -> if open, residual head + 7 capacity/reweight controls.
     - Diamond Gate `G2` (`E6 Generality`): Post-lock gate ($\Delta\mathrm{G3}\ge +5\text{ pp}$, $\ge 5/6$ cohorts, $\Delta\mathrm{Within1}\ge -1\text{ pp}$, beats E4) + pretraining exposure audit.
     - Dashed Callout (`E5 Transductive`, Opt-in `--enable-e5`): Saerens EM & BBSE using unlabelled $\mathbf{p}_{\text{target}}$ ONLY (strictly separate `TRANSDUCTIVE_ADAPTATION` regime; blocked from double-correcting with CIBD $\tau \neq 0$).
- **Caption Draft**: *MLWA journal experimental protocol isolating representation/head effects (`E2`, `E2b`), imbalance training baselines (`E4`), fixed-posterior decision rules (`E0`/CIBD), linear feature accessibility (`E1`), and gated/opt-in extensions (`E3`, `E5`, `E6`) under a strict source-only selection contract.*

---

## Diagram J2 — Head Geometry & Decision-Rule Mechanism (`CLM` vs `Softmax` × `CIBD`)

- **Target filename**: `figures/journal/head_and_cibd_geometry.pdf`
- **Canvas**: Two stacked horizontal panels (`980 × 520 px`).
- **Top Panel (`Softmax` vs `CLM` Output Geometry)**:
  - Left: `512-D` $\mathbf{z}$ -> Softmax (`5 × 512` unconstrained logits $f_0,\dots,f_4$) -> $p_j = \mathrm{softmax}(\mathbf{f})_j$ (per-grade boundaries, can decouple Grade 3 sensitivity from neighbor ordering).
  - Right: `512-D` $\mathbf{z}$ -> Shared direction $\mathbf{w}^\top \mathbf{z} = s$ (`1-D` bottleneck, rank-1 pre-threshold geometry) -> ordered cutpoints $\theta_0 < \theta_1 < \theta_2 < \theta_3$ via cumulative link $F$ (`cloglog`: $1-\exp(-\exp(\theta_k - s))$; `logit`: $\sigma(\theta_k - s)$) -> intervals $p_k = P(y \le k) - P(y \le k-1)$ (Grade 3 interval $[\theta_2, \theta_3]$ squeezed by asymmetric hazard under severe imbalance).
- **Bottom Panel (`Fixed-Posterior Decision Layer — E0 & CIBD`)**:
  - Inputs: locked $\mathbf{p}(x)$ + Laplace-smoothed `source_train` prior $\hat{\pi}$.
  - Step 1: Prior tilt $r_j \propto p_j \hat{\pi}_j^{-\tau}$ ($\tau \in \{0, 0.25, 0.5, 0.75, 1\}$).
  - Step 2: Blended cost $C_\alpha(k, j) = \alpha|k-j|^q + (1-\alpha)\mathbb{I}[k \neq j]$ showing special cases on clean labels: $(\alpha=0, \tau=0) \Rightarrow \text{argmax}$; $(\alpha=1, q=1, \tau=0) \Rightarrow \text{posterior median}$; $(\alpha=1, q=2, \tau=0) \Rightarrow \text{rounded expectation Bayes}$ (all breaking ties toward the lower grade).
  - Step 3: `source_val` constraint filter (`within1 >= baseline - margin`, `qwk >= baseline - 0.01`) before locking $(\alpha^*, \tau^*, q^*)$ for single-pass `target` evaluation.
- **Caption Draft**: *Mechanistic separation of output head parameterization (`Softmax` vs. cumulative link `CLM`) from fixed-posterior Bayesian decision decoding (`CIBD` and classical special cases).*
