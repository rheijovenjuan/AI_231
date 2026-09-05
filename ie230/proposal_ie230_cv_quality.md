# IE 230 — Statistical Design and Analysis for Industrial Engineering
## Alternative Capstone Proposal (Computer-Vision / Image-Signal-Video Focus)

**Course:** IE 230 (1st Semester 2026–2027) · 3 units
**Instructor:** Prof. Eugene Rex L. Jalao, Ph.D.
**Student:** Rhei Juan
**Proposal date:** 5 September 2026 (due 19 September 2026)
**Note:** This is an *alternative* to the concrete-strength proposal, chosen to align
with a computer-vision interest. It still satisfies all four IE 230 course outcomes by
treating **image/signal quality metrics as the statistical response** and **capture /
processing parameters as the experimental factors.**

---

## 1. Title

**Optimizing Image / Video Compression Quality Using Designed Experiments, Regression, and Response Surface Methodology**

## 2. Background & Motivation

Every image, signal, or video pipeline trades **quality** against **cost** (file size,
bitrate, processing time). A JPEG/HEVC encoder, a denoising filter, or a down-sampler is
governed by several tunable parameters (quality factor, bitrate, block size, kernel size,
frame rate). Industrial engineers need a rigorous way to answer:

1. **Which parameters most affect perceived/objective quality, and can we predict it?** (regression)
2. **Which parameter setting maximizes quality for a target file size / bitrate?** (optimization)

This project measures objective image/signal quality across a controlled grid of
processing parameters and applies the full IE 230 toolkit — hypothesis testing, multiple
regression with diagnostics, factorial design, and response surface methodology.

## 3. Objectives

- **General:** Apply statistical design & analysis to a computer-vision / signal-processing
  quality dataset, demonstrating CO1–CO4.
- **Specific:**
  1. Compare quality across processing settings using hypothesis tests (CO1).
  2. Model quality as a function of processing parameters via multiple regression (CO2).
  3. Run a factorial experiment on the dominant parameters (CO3).
  4. Fit a response-surface model and find the quality-optimal setting at a size constraint (CO4).

## 4. Dataset & Source (readily available online)

Primary — **public image / video benchmarks** (all directly downloadable):

- **Kaggle "Image Compression / JPEG Artifacts"** and general image datasets
  (e.g., **Kaggle "Images for Image Processing"**) —
  https://www.kaggle.com/datasets (search: image compression, jpeg, mri, x-ray)
- **UCI / public medical imaging** (for a denoising variant):
  - Kaggle **MRI / X-ray** datasets — https://www.kaggle.com/datasets
  - NIH **Open Access Series of Imaging Studies (OAI)** — https://nihimage.github.io/
- **Video:** **Kaggle / public surveillance & driving video sets** (e.g., **BDD100K**
  https://www.bdd100k.com/, **CUHK** surveillance) for a frame-rate / resolution study.

**Experimental data generation (the core of the project):**
Rather than relying only on a labeled set, we *generate* the response by processing each
source image/signal across a grid of parameter settings and measuring objective quality.
This is legitimate under the syllabus ("perform a statistical analysis on your dataset"),
and the raw source images are freely downloadable.

### Measured response & factors

| Quantity | Role | Typical metric |
|----------|------|----------------|
| **Objective quality score** | **Response** | PSNR, SSIM, MS-SSIM, or VMAF (video) |
| Quality factor / bitrate | Factor (continuous) | 1–100 / kbps |
| Block / tile size | Factor (categorical) | 8 / 16 / 32 px |
| Kernel / filter size (denoise) | Factor (categorical) | 3 / 5 / 7 |
| Frame rate / resolution (video) | Factor | fps / H×W |
| Noise level (if denoising) | Factor (continuous) | σ |
| **File size / bitrate** | Constraint / covariate | bytes / kbps |

## 5. Statistical Methodology (mapped to Course Outcomes)

### CO1 — Sampling, Estimation & Hypothesis Testing
- Sample a representative subset of images/signals; summarize quality-score distributions.
- Confidence intervals for mean PSNR/SSIM per setting.
- **ANOVA / t-tests** comparing mean quality across quality-factor levels, block sizes,
  or codecs.

### CO2 — Simple & Multiple Linear Regression (+ Diagnostics)
- Simple regression of quality on bitrate/quality-factor.
- **Multiple regression** of quality on all processing parameters (with indicator
  variables for categorical factors such as block size / codec).
- **Stepwise selection** + **diagnostics:** residual plots, Q-Q, Durbin–Watson, VIF,
  adjusted R² / RMSE. (Directly exercises the "indicator variables & multicollinearity"
  session, 26 Sep.)

### CO3 — Factorial Design of Experiments
- A **2^k factorial** on the key parameters (e.g., quality factor × block size × noise
  level) at low/high levels.
- Use **fractional replication / confounding** to bound the number of encodes.
- ANOVA for main effects and interactions (e.g., does block size matter more at low
  quality?).

### CO4 — Response Surface Methodology (Process Optimization)
- Fit a **second-order model** to the continuous parameters (quality factor, bitrate).
- **Central Composite / Box–Behnken** design for the RSM stage.
- Contour surfaces of quality vs. parameters; find the **optimal setting** that
  maximizes PSNR/SSIM subject to a **file-size / bitrate constraint** — the classic
  quality-vs-cost trade-off.

## 6. Tools

- **Python 3.x**
  - `Pillow`, `opencv-python`, `scikit-image` — encode / denoise / resize, compute PSNR/SSIM
  - `av` (PyAV) / `ffmpeg` — video encode + **VMAF** for the video variant
  - `pandas`, `numpy`, `scipy.stats`, `statsmodels` — statistics & regression
  - `pyDOE2` — factorial / CCD / Box–Behnken designs
  - `matplotlib`, `seaborn` — residual plots, contour surfaces
- Reference: Montgomery, *Design and Analysis of Experiments*.

## 7. Expected Deliverables

1. This proposal (1 page).
2. Jupyter notebook: source-data prep → parameter-grid experiment → CO1–CO4 analyses.
3. Final report with the statistical rationale, figures, and the optimal setting.
4. A quality-vs-file-size Pareto chart and the recommended encoder/filter configuration.

## 8. Timeline (aligned to syllabus)

| Date | Milestone |
|------|-----------|
| 19 Sep | **Proposal due** (this document) |
| 26 Sep – 3 Oct | Acquire source images/video; build quality-measurement pipeline (CO1) |
| 10 – 24 Oct | Regression + diagnostics (CO2); factorial design (CO3) |
| 14 – 21 Nov | Response surface methodology (CO4) |
| 28 Nov | Project work / integration |
| 5 Dec | Finals (all topics) |
| TBA (Dec) | **Final project due** |

## 9. Success Criteria

- A reproducible parameter-grid experiment producing a clean quality dataset.
- Regression model with adjusted R² ≥ 0.80 and acceptable diagnostics.
- Factorial ANOVA identifying ≥ 2 significant parameters/interactions.
- RSM yielding a feasible optimum with a validated quality-vs-size trade-off curve.
- All four course outcomes (CO1–CO4) demonstrably addressed.

## 10. Variant Options (same structure, different domain)

| Variant | Response | Factors | Public data |
|---------|----------|---------|-------------|
| **Image compression** (default) | PSNR / SSIM | quality factor, block size, codec | Kaggle image sets |
| **Medical-image denoising** | SSIM vs. noise-free ref | kernel size, strength, noise σ | Kaggle MRI/X-ray, NIH OAI |
| **Video streaming** | VMAF | bitrate, frame rate, resolution, GOP | BDD100K / CUHK video |
| **Signal denoising** | SNR | filter order, cutoff, window | IEEE / Kaggle signal sets |
