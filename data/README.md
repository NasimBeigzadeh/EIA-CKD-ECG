# Datasets

This study uses two publicly available benchmark ECG datasets:

1. Chapman-Shaoxing 12-lead ECG Database
2. PTB-XL ECG Dataset

The raw datasets are not redistributed in this repository.
Users should obtain the datasets from their official/public sources and follow
the corresponding dataset licenses and terms of use.

---

## 1. Chapman-Shaoxing 12-lead ECG Database

The Chapman-Shaoxing dataset contains 12-lead ECG recordings sampled at
500 Hz with a duration of 10 seconds.

For this study, four diagnostic/rhythm classes were considered:

| Class | Description |
|---|---|
| SR | Sinus Rhythm |
| SB | Sinus Bradycardia |
| GSVT | General Supraventricular Tachycardia |
| AFIB | Atrial Fibrillation |

### Dataset source

The dataset can be obtained from Kaggle:

[Chapman-Shaoxing 12-lead ECG Database](https://www.kaggle.com/datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database)

### Dataset split

The dataset was divided into training, validation, and test subsets.

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| SR | 1301 | 145 | 380 |
| SB | 2821 | 308 | 760 |
| GSVT | 1633 | 196 | 447 |
| AFIB | 1599 | 169 | 457 |

The original dataset was divided using an 80/20 train-test split.
The training portion was subsequently divided into 90% training and 10%
validation subsets.

---

## 2. PTB-XL

PTB-XL is a large publicly available 12-lead ECG dataset containing
21,799 clinical ECG records from 18,869 patients.

Each ECG recording has a duration of 10 seconds and was acquired at
100 Hz. The dataset provides cardiologist annotations and standardized
ECG statements.

### Official dataset source

[PTB-XL – PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)


### Classes

The superdiagnostic classification scheme was used, consisting of five
classes:

| Class | Description |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Change |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

### Dataset split used in this study

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| NORM | 7243 | 914 | 912 |
| CD | 1353 | 171 | 184 |
| HYP | 415 | 64 | 56 |
| STTC | 1903 | 255 | 242 |
| MI | 2043 | 233 | 256 |

---

## 3. ECG Image Representation

In this study, the raw ECG signals were converted into digital ECG images
before being provided to the deep learning models.

This representation enables the use of convolutional neural networks (CNNs)
for ECG classification.

The generated images were organized using the following directory structure:

```text
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── valid/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
