# EIA-CKD: End-to-End Internal Augmentation and Curriculum-Based Knowledge Distillation for Lightweight ECG Classification
Official implementation of EIA-CKD, a lightweight end-to-end framework for ECG classification using internal multi-view augmentation and curriculum-based knowledge distillation.

# Dataset

This directory contains documentation and instructions for obtaining and preparing the ECG datasets used in the experiments reported in the paper.

The proposed EIA-CKD framework was trained and evaluated using two publicly available benchmark datasets:

1. Chapman/Shaoxing 12-lead ECG Database
2. PTB-XL ECG Database

The original datasets are **not redistributed in this repository**. Users should obtain the datasets from their respective public sources and follow the preprocessing instructions provided in this repository.

---

## 1. Chapman/Shaoxing 12-lead ECG Database

The Chapman/Shaoxing dataset contains 12-lead ECG recordings sampled at 500 Hz with a duration of 10 seconds.

Four diagnostic classes were used in this study:

| Class | Description                          |
| ----- | ------------------------------------ |
| SR    | Normal Sinus Rhythm                  |
| SB    | Sinus Bradycardia                    |
| GSVT  | General Supraventricular Tachycardia |
| AFIB  | Atrial Fibrillation                  |

### Dataset source

The dataset is publicly available through Kaggle:

[Chapman/Shaoxing 12-lead ECG Database — Kaggle](https://www.kaggle.com/datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database?utm_source=chatgpt.com)

Please refer to the original dataset publication and source page for dataset licensing, attribution, and usage conditions.

---

## 2. PTB-XL ECG Dataset

PTB-XL is a large publicly available 12-lead ECG dataset containing clinical ECG recordings annotated by expert cardiologists.

In this study, the superdiagnostic classification scheme was adopted.

Five diagnostic classes were used:

| Class | Description            |
| ----- | ---------------------- |
| NORM  | Normal ECG             |
| MI    | Myocardial Infarction  |
| STTC  | ST/T Change            |
| CD    | Conduction Disturbance |
| HYP   | Hypertrophy            |

### Dataset source

The PTB-XL dataset is publicly available through PhysioNet:

[PTB-XL — PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/?utm_source=chatgpt.com)

Please refer to the official PhysioNet page for the dataset description, licensing, citation requirements, and access conditions.

---
