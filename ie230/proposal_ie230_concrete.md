# IE 230 — Statistical Design and Analysis for Industrial Engineering
## Capstone / Mini-Project Proposal

**Course:** IE 230 (1st Semester 2026–2027) · 3 units
**Instructor:** Prof. Eugene Rex L. Jalao, Ph.D.
**Student:** Rhei Juan
**Proposal date:** 5 September 2026 (due 19 September 2026)
**Weight:** Project = 30% of final grade

---

## 1. Title

**Modeling and Optimization of Concrete Compressive Strength Using Regression, Factorial Design, and Response Surface Methodology**

## 2. Background & Motivation

Concrete is the backbone of construction, and its 28-day compressive strength is the
primary quality metric used to accept or reject a mix. Strength depends on several
controllable mix ingredients (cement, water, aggregates, admixtures) and on curing age.
Engineers routinely face two coupled questions:

1. **Which ingredients most influence strength, and how well can we predict it?** (regression)
2. **What ingredient combination maximizes strength under a cost / workability constraint?** (optimization)

This project treats the concrete dataset as an industrial-engineering problem and applies
the full statistical toolkit covered in IE 230 — sampling & hypothesis testing, multiple
regression with diagnostics, factorial design of experiments, and response surface
methodology — to answer both questions.

## 3. Objectives

- **General:** Perform a complete statistical design-and-analysis study on a real,
  publicly available engineering dataset, demonstrating all four course outcomes.
- **Specific:**
  1. Characterize the data and test differences in mean strength across groups (CO1).
  2. Build and validate a multiple-regression model predicting compressive strength (CO2).
  3. Design and analyze a factorial experiment on the dominant mix factors (CO3).
  4. Fit a second-order response-surface model and locate the strength-optimal mix (CO4).

## 4. Dataset & Source (readily available online)

- **Name:** Concrete Compressive Strength Dataset
- **Source (primary):** UCI Machine Learning Repository —
  https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
- **Mirror:** Kaggle — https://www.kaggle.com/datasets/iamashish/concrete-data
- **Size:** 8,038 observations × 9 columns (no missing values).
- **Format:** CSV (direct download, no registration required from UCI).

### Variables

| Variable | Role | Units |
|----------|------|-------|
| Cement component amount | Factor (continuous) | kg/m³ |
| Blast Furnace Slag | Factor (continuous) | kg/m³ |
| Fly Ash | Factor (continuous) | kg/m³ |
| Water | Factor (continuous) | kg/m³ |
| Superplasticizer | Factor (continuous) | kg/m³ |
| Coarse Aggregate | Factor (continuous) | kg/m³ |
| Fine Aggregate | Factor (continuous) | kg/m³ |
| Age (concrete age) | Factor (continuous) | days |
| **Concrete compressive strength** | **Response** | **MPa** |

## 5. Statistical Methodology (mapped to Course Outcomes)

### CO1 — Sampling, Estimation & Hypothesis Testing
- Descriptive statistics and sampling-distribution summaries of strength.
- Confidence intervals for the mean strength.
- One-way **ANOVA** / pairwise **t-tests** on mean strength across age groups
  (e.g., 7-day vs 28-day vs 90-day) and across cement-content groups.

### CO2 — Simple & Multiple Linear Regression (+ Diagnostics)
- Simple regression of strength on cement; then **multiple regression** on all seven inputs.
- **Stepwise / backward variable selection** to identify the significant contributors.
- **Model validation & diagnostics:** residual-vs-fitted, Q-Q plot, scale-location,
  Durbin–Watson (autocorrelation), VIF (multicollinearity), lack-of-fit, R² / adjusted R² / RMSE.

### CO3 — Factorial Design of Experiments
- Screen the dominant factors from CO2, then run a **2^k (fractional) factorial** design
  at low/high levels of the top ingredients (e.g., cement, water, slag, fly ash).
- Analyze main effects and interactions; use **confounding / fractional replication**
  to keep the run count practical; test factor significance via ANOVA.

### CO4 — Response Surface Methodology (Process Optimization)
- Fit a **second-order (quadratic) model** to the significant continuous factors.
- Use a **Central Composite Design (CCD)** or **Box–Behnken** design for the RSM stage.
- Generate contour / 3-D response surfaces; locate the **stationary (optimal) point**
  maximizing predicted strength, and verify it lies within the feasible region.

## 6. Tools

- **Python 3.x**
  - `pandas`, `numpy` — data handling
  - `scipy.stats` — t-tests, ANOVA, confidence intervals
  - `statsmodels` — OLS regression, diagnostics, VIF, Durbin–Watson
  - `pyDOE2` — factorial, CCD, Box–Behnken design generators
  - `matplotlib`, `seaborn` — residual plots, contour surfaces
- Reference: Montgomery, *Design and Analysis of Experiments*; Montgomery,
  *Applied Statistics and Probability for Engineers*.

## 7. Expected Deliverables

1. This proposal (1 page).
2. Jupyter notebook: data exploration → CO1–CO4 analyses with annotated figures.
3. Final report describing each analysis, the statistical rationale, and conclusions.
4. The identified optimal concrete mix with a predicted-strength estimate.

## 8. Timeline (aligned to syllabus)

| Date | Milestone |
|------|-----------|
| 19 Sep | **Proposal due** (this document) |
| 26 Sep – 3 Oct | Data acquisition, cleaning, EDA (CO1) |
| 10 – 24 Oct | Regression + diagnostics (CO2); factorial design (CO3) |
| 14 – 21 Nov | Response surface methodology (CO4) |
| 28 Nov | Project work / integration |
| 5 Dec | Finals (all topics) |
| TBA (Dec) | **Final project due** |

## 9. Success Criteria

- Regression model with adjusted R² ≥ 0.85 and no serious diagnostic violations.
- Factorial ANOVA clearly identifying ≥ 2 significant factors/interactions.
- RSM producing a feasible stationary point with a validated predicted maximum strength.
- All four course outcomes (CO1–CO4) demonstrably addressed.
