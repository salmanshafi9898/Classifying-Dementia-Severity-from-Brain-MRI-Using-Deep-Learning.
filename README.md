# 🧠 Classifying Dementia Severity from Brain MRI Using Deep Learning

> **Patient-level CNN pipeline** on ~80K MRI slices — fine-tuned ResNet achieving 91.6% weighted test accuracy with rigorous subject-stratified validation to prevent data leakage under real-world conditions.

---

## 📌 Project Overview

Dementia affects tens of millions worldwide, and early, accurate classification of its severity is critical for timely clinical intervention. Yet most published deep learning benchmarks suffer from a subtle but serious flaw: **data leakage** — where MRI slices from the same patient appear in both training and test sets, inflating accuracy metrics that don't hold up in the real world.

This project tackles that problem head-on. By building a **patient-level validation pipeline**, every design decision — from data splitting to model evaluation — mirrors real clinical deployment conditions. The result is a model that's not just accurate, but trustworthy.

---

## 🎯 Clinical Problem

Given a brain MRI scan, classify dementia severity into one of four categories:

| Class | Description |
|---|---|
| Non-Demented | No cognitive impairment |
| Very Mild Demented | Earliest detectable decline |
| Mild Demented | Moderate cognitive decline |
| Moderate Demented | Significant impairment |

---

## 🗂️ Dataset

| Feature | Detail |
|---|---|
| Total MRI Slices | ~80,000 |
| Classes | 4 (Non / Very Mild / Mild / Moderate Demented) |
| Input Format | 2D grayscale brain MRI images |
| Validation Strategy | Subject-stratified split (patient-level) |
| Framework | PyTorch |

> ⚠️ **Critical design note:** Slices from the same patient were kept exclusively in either train or test — never both. This prevents the model from learning patient-specific features rather than disease indicators, a common source of inflated benchmarks in medical imaging literature.

---

## 🔬 Methodology

### 1. 🧹 Data Preprocessing
- Resized and normalized MRI slices to fit ResNet input requirements
- Applied data augmentation (flips, rotations, brightness jitter) to address class imbalance, particularly for the rare *Moderate Demented* class
- Split data at the **patient level** using subject-stratified sampling — ensuring zero overlap between train and test subjects

### 2. 🏗️ Model Architecture — Fine-Tuned ResNet
- Used a **pretrained ResNet** (ImageNet weights) as the backbone — leveraging transfer learning to extract robust low-level visual features
- Replaced the final classification head with a custom 4-class output layer
- Frozen early convolutional layers during initial training; selectively unfrozen deeper layers for fine-tuning
- Applied **weighted loss function** to penalize misclassification of underrepresented classes

### 3. 🔁 Training Strategy
- Optimizer: Adam with learning rate scheduling
- Loss: Weighted Cross-Entropy (to handle class imbalance)
- Validation: Patient-level held-out test set throughout training
- Monitored both **overall accuracy** and **per-class recall** to ensure the model didn't simply learn the majority class

### 4. 📊 Results

| Metric | Score |
|---|---|
| Weighted Test Accuracy | **91.6%** |
| Validation Strategy | Subject-stratified (no leakage) |
| Benchmark Comparison | Competitive with published results under real-world conditions |

> The 91.6% weighted accuracy is meaningful precisely because it was achieved without data leakage — making it a fair comparison against clinical benchmarks.

---

## 📁 Repository Structure

```
├── dataset/        # MRI image data (organized by class)
├── docs/           # Project report and documentation
├── script/         # Training, evaluation, and preprocessing scripts
├── README.md
└── placeholder
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, torchvision |
| Model | ResNet (pretrained, fine-tuned) |
| Data Handling | NumPy, PIL, torchvision transforms |
| Visualization | Matplotlib, Seaborn |
| Validation | Subject-stratified train/test split |

---

## 💡 Key Takeaways

- **Data leakage is the #1 silent killer of medical AI benchmarks.** Patient-level splitting is non-negotiable for any model intended to generalize to new patients
- **Transfer learning from ImageNet works surprisingly well on MRI data** — even though the domains are very different, low-level edge and texture features transfer effectively
- **Weighted loss functions matter more than oversampling** when class imbalance is moderate — they're simpler and less prone to overfitting on minority classes
- **91.6% weighted accuracy on a properly validated 4-class problem** is a strong result that holds up against published literature

---

## 👤 Author

**Salman Khan Shafi**
MS Business Analytics — Duke Fuqua '26
[LinkedIn](https://linkedin.com/in/salmankhanshafi) • [GitHub](https://github.com/salmanshafi9898)
