# Project Name : Stress-Testing of Convolutional Neural Networks

## Overview
This repository contains an ongoing project focused on experimentation, development, and analysis.  
The project is currently **under active development** and may contain incomplete components, placeholders, or experimental code.

The structure is designed to be modular so that different components (data, models, training scripts, utilities, etc.) can be developed and updated independently.

---
📁 Project Structure
----

cnn-stress-testing/
│
├── data/
│   └── cifar-10-batches-py/        # Official CIFAR-10 dataset (unchanged)
│
├── models/
│   └── resnet18.py                 # ResNet-18 architecture (modified for CIFAR-10)
│
├── train.py                        # Baseline model training
├── failure_analysis.py             # Extract confident failure cases
├── explainability.py               # Grad-CAM for baseline model
├── improvement.py                  # Improved training (Label Smoothing)
├── compare_models.py               # Baseline vs Improved comparison
├── explainability_improved.py      # Grad-CAM for improved model
│
├── experiments/
│   ├── baseline/
│   │   ├── best_model.pth
│   │   ├── metrics.csv
│   │   ├── accuracy_curve.png
│   │   └── loss_curve.png
│   │
│   ├── improved/
│   │   ├── best_model.pth
│   │   ├── metrics.csv
│   │   ├── accuracy_curve.png
│   │   └── loss_curve.png
│   │
│   └── comparison/
│       └── comparison_results.csv
│
├── failure_cases/
│   └── baseline/
│       ├── case_03/
│       │   ├── image.png
│       │   ├── meta.txt
│       │   ├── gradcam_baseline.png
│       │   └── gradcam_improved.png
│       │
│       ├── case_08/
│       └── case_11/
│
├── environment.yml                 # Conda environment configuration
└── README.md                       # Project documentation


To reproduce baseline model:
python train.py

To reproduce improved model:
python improvement_train.py



# Project Workflow Overview

This project performs stress-testing of a ResNet-18 model on CIFAR-10 by analyzing confident failure cases and evaluating the effect of a constrained improvement (label smoothing).

The workflow was executed in the following order:

---

## 1️⃣ `train.py` — Baseline Model Training

**Purpose:**
Train a ResNet-18 model from scratch on CIFAR-10.

**What it does:**

* Trains the baseline model
* Evaluates test accuracy each epoch
* Logs metrics to CSV
* Saves accuracy and loss plots
* Saves best model checkpoint

**Outputs saved to:**

```
experiments/baseline/
    best_model.pth
    metrics.csv
    accuracy_curve.png
    loss_curve.png
```

---

## 2️⃣ `failure_analysis.py` — Extract Confident Failure Cases

**Purpose:**
Identify high-confidence misclassifications from the trained baseline model.

**What it does:**

* Loads baseline model
* Runs inference on test set
* Filters wrong predictions with high confidence
* Saves selected failure images and metadata

**Outputs saved to:**

```
failure_cases/baseline/
    case_XX/
        image.png
        meta.txt
```

Three representative failure cases were selected for deeper analysis.

---

## 3️⃣ `explainability.py` — Grad-CAM for Baseline Model

**Purpose:**
Visualize which image regions influenced the baseline model’s predictions.

**What it does:**

* Loads baseline model
* Applies Grad-CAM to selected failure cases
* Generates heatmap overlays

**Outputs saved inside each case folder:**

```
gradcam_baseline.png
```

---

## 4️⃣ `improvement.py` — Train Improved Model (Label Smoothing)

**Purpose:**
Apply one constrained modification (label smoothing) and retrain the model.

**What it does:**

* Same architecture and training setup as baseline
* Only change: `label_smoothing=0.1` in loss function
* Trains improved model
* Logs metrics and saves plots
* Saves improved model checkpoint

**Outputs saved to:**

```
experiments/improved/
    best_model.pth
    metrics.csv
    accuracy_curve.png
    loss_curve.png
```

---

## 5️⃣ `compare_models.py` — Baseline vs Improved Comparison

**Purpose:**
Compare baseline and improved models on the same selected failure cases.

**What it does:**

* Loads both baseline and improved models
* Evaluates them on the same three failure images
* Compares predictions and confidence values
* Saves results in CSV format

**Outputs saved to:**

```
experiments/comparison/
    comparison_results.csv
```

---

## 6️⃣ `explainability_improved.py` — Grad-CAM for Improved Model

**Purpose:**
Analyze how attention patterns changed after applying label smoothing.

**What it does:**

* Loads improved model
* Applies Grad-CAM to the same three baseline failure cases
* Saves heatmaps for comparison

**Outputs saved inside each case folder:**

```
gradcam_improved.png
```

---

# Complete Experimental Flow

1. Train baseline → `train.py`
2. Extract failure cases → `failure_analysis.py`
3. Analyze baseline attention → `explainability.py`
4. Apply constrained improvement → `improvement.py`
5. Compare behavior on same failures → `compare_models.py`
6. Analyze improved attention → `explainability_improved.py`

---

This structured workflow ensures:

* Reproducibility
* Controlled experimentation
* Clear separation between baseline and improved models
* Proper behavioral analysis beyond accuracy

---

