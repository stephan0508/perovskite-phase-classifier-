# perovskite-phase-classifier-
Baseline machine learning pipeline for crystal structure classification in ABO3 perovskites using Random Forest."

# Perovskite Crystal Structure Classification (Baseline ML)

## 📌 Executive Summary
This repository contains a **Machine Learning (Random Forest)** baseline pipeline designed to classify crystal structures in $ABO_3$ perovskite oxides using physico-chemical descriptors (ionic radii, electronegativities, and tolerance factors).

The goal of this project is **not** to artificially inflate metrics, but to build an honest baseline, audit the limitations of experimental data, and critically analyze algorithm performance under severe class imbalance.

---

## 📊 Performance Metrics (Random Forest)
The model achieved a **77% Global Accuracy** (with a previous trial baseline at 58%), largely driven by the overwhelming dominance of the majority classes. However, the per-class breakdown exposes the real technical limitations of the system:

| Class | Precision | Recall | F1-Score | Support (Test Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **cubic** | 0.90 | 0.82 | 0.86 | 651 |
| **orthorhombic** | 0.68 | 0.78 | 0.72 | 315 |
| **rhombohedral** | 0.34 | 0.37 | 0.36 | 65 |
| **tetragonal** | 0.27 | 0.36 | 0.31 | 25 |

* **Global Accuracy:** 0.77
* **Macro Avg (The real metric exposing minority class failure):** 0.56
* **Weighted Avg:** 0.77

---

## 🔬 Findings & Critical Analysis

1. **The Accuracy Illusion:** A 77% accuracy is misleading. The **Macro F1 score of 0.56** proves that the model struggles critically when predicting minority classes (`rhombohedral` and `tetragonal`), where data support is extremely low (only 25 samples for the tetragonal phase).
2. **Crystallographic Confusion:** The normalized confusion matrix reveals that tetragonal phase errors concentrate mainly in the rhombohedral phase. This aligns with physical reality: both are distorted structures sitting close to each other in phase diagrams, causing severe overlap in the numerical feature space.
3. **Feature Importance (Physics vs. Noise):** 
   * The model validated underlying physical theory by assigning the highest importance to **Goldschmidt's Tolerance Factor ($tG$)** and the **electronegativity difference ($\Delta ENR$)**.
   * Modern descriptors like the $\tau$ factor were heavily penalized due to the impact of imputed null values during data preprocessing.

---

## 🚀 Conclusions & Next Steps (Phase 2)
This repository demonstrates that a black-box model cannot compensate for a scarcity of experimental data in rare classes purely through hyperparameter brute-force.

For a production version or future iteration, mandatory action items include:
* Implementing synthetic data augmentation techniques (e.g., **SMOTE** adapted to crystallochemical spaces) strictly on the training set.
* Incorporating a larger experimental volume of rare distorted phases to balance the sample space.
