# Hotel Booking Cancellation Prediction

A machine learning pipeline that predicts hotel booking cancellations using Genetic Algorithm (GA) feature selection applied independently to three classifiers: K-Nearest Neighbors, Decision Tree, and Neural Network (MLP).

**Best Result:** Decision Tree + GA — Test F1 = **0.7612** | Test Accuracy = **82.70%**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Summary](#pipeline-summary)
- [Feature Selection (GA)](#feature-selection-ga)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Team](#team)

---

## Project Overview

Hotel booking cancellations create significant operational challenges — lost revenue, inefficient inventory management, and poor demand forecasting. This project builds a full end-to-end ML pipeline to classify whether a hotel booking will be canceled (`is_canceled = 1`) or not (`is_canceled = 0`).

**Key design choice:** Instead of using a single GA run shared across all models, we run a **separate Genetic Algorithm feature selection for each model**. This is because wrapper-based feature selection is model-dependent — the optimal feature subset for a Decision Tree differs from that for KNN or MLP.

---

## Dataset

| Property | Value |
|---|---|
| Source | Hotel Booking Demand (public dataset) |
| Initial Records | 119,390 bookings |
| Initial Features | 33 columns |
| After Preprocessing | 119,288 records, 44 features |
| Target Variable | `is_canceled` (0 = not canceled, 1 = canceled) |
| Class Distribution | ~63% not canceled, ~37% canceled |

### Preprocessing Steps

- Dropped `company` column (94.31% missing)
- Filled missing values: `agent` → 0, `country` → mode, `children` → 0
- IQR-based outlier capping on numeric columns
- Removed 63 duplicate rows post-preprocessing
- **Encoding:** Label encoding for ordinal categoricals; One-Hot encoding for `city` (15 dummies)
- **Frequency encoding:** `hotel_freq`, `country_freq`
- **Feature engineering:** `arrival_dayofweek`, `arrival_month_num`, `total_guests`, `total_previous_bookings`, `got_reserved_room`
- Dropped `total_stay` (highly correlated)

### Train / Validation / Test Split

| Split | Records | Class Ratio |
|---|---|---|
| Train (60%) | 71,572 | 63:37 |
| Validation (20%) | 23,858 | 63:37 |
| Test (20%) | 23,858 | 63:37 |

**SMOTE** applied to training set only → balanced to 45,089 : 45,089 (1:1).

---

## Pipeline Summary

```
Raw Data (119,390 rows, 33 cols)
        |
        v
   Data Cleaning
  (missing values, duplicates, outliers)
        |
        v
  Feature Engineering
  (6 derived + 2 frequency + encoding)
        |
        v
  Train / Val / Test Split  +  SMOTE (train only)
        |
        +---> GA Feature Selection (KNN)   --> KNN Hyperparameter Tuning
        |
        +---> GA Feature Selection (DT)    --> DT Hyperparameter Tuning
        |
        +---> GA Feature Selection (MLP)   --> MLP Hyperparameter Tuning
        |
        v
   Final Evaluation on Test Set
```

---

## Feature Selection (GA)

`GAFeatureSelectionCV` from `sklearn-genetic-opt` — a wrapper-based method that evolves feature subsets using cross-validation F1-score as the fitness function.

| Model | Generations | CV Folds | Features Selected | Reduction |
|---|---|---|---|---|
| KNN | 6 | 3 | 28 | 36% |
| Decision Tree | 9 | 3 | 28 | 36% |
| MLP | 9 | 3 | 25 | 43% |

**Universal features selected by all 3 models:**
`market_segment`, `total_guests`, `total_of_special_requests`, `country_freq`, `got_reserved_room`, `is_repeated_guest`, `lead_time`

---

## Results

### Test Set Performance

| Model | Features | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| KNN + GA | 28 | 0.7459 | 0.6265 | 0.7755 | 0.6931 |
| MLP + GA | 25 | 0.8065 | 0.7317 | 0.7534 | 0.7424 |
| **Decision Tree + GA** | **28** | **0.8270** | **0.7781** | **0.7450** | **0.7612** |

### Best Model: Decision Tree + GA

- `max_depth=15`, `min_samples_leaf=10`, `min_samples_split=2`
- Minimal validation-to-test gap (0.7732 → 0.7612) — no overfitting
- Catches 74.5% of all actual cancellations (6,577 / 8,828)

### Confusion Matrix — Decision Tree (Test Set)

```
                  Predicted NOT Cancel    Predicted CANCEL
Actual NOT Cancel       13,154                1,876
Actual CANCEL            2,251                6,577
```

---

## Project Structure

```
PROJECT/
├── Final_AI_updated.ipynb              # Main notebook (run in Google Colab)
├── hotel_bookings.csv                  # Dataset (place in Google Drive)
├── Hotel Booking Cancellation Report.pdf        # Original report
├── Hotel_Booking_Cancellation_Report_Updated.pdf # Updated report (new results)
├── generate_report.py                  # Script used to generate updated PDF
└── README.md
```

---

## Requirements

This notebook is designed to run on **Google Colab**. It will not run locally without removing Colab-specific code (`google.colab` imports).

```
Python 3.x
scikit-learn
sklearn-genetic-opt
imbalanced-learn (SMOTE)
numpy
pandas
matplotlib
seaborn
```

Install dependencies (in Colab):
```bash
!pip install sklearn-genetic-opt imbalanced-learn
```

---

## How to Run

1. Upload `Final_AI_updated.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `hotel_bookings.csv` to your Google Drive under `MyDrive/`
3. Run all cells in order — the notebook will:
   - Mount Google Drive and load the dataset
   - Preprocess the data and engineer features
   - Run GA feature selection for each model (~45–60 min total)
   - Tune hyperparameters on the validation set
   - Evaluate final performance on the held-out test set

> Note: GA runs are stochastic — selected features may vary slightly between runs.

---

## Team

| Name | Student ID |
|---|---|
| Abdallah Hassan | 2023/04630 |
| Aly Mohamed | 2023/05500 |
| Ahmed Hamed | 2023/07596 |
| Moustafa Mohamed | 2023/06727 |
| Ahmed Ayman | 2023/03485 |

**Course:** Artificial Intelligence — Year 3, Semester 1

---

## References

1. Scikit-learn. Feature selection. https://scikit-learn.org/stable/modules/feature_selection.html
2. Brainalyst Academy. Wrapper Methods: Feature Selection Algorithm. https://brainalystacademy.com/wrapper-methods/
3. Goldberg, D. E. (1989). *Genetic algorithms in search, optimization, and machine learning.* Addison-Wesley.
4. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
