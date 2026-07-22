# Early Disease Finding Detection from Brain MRI Scans — ML

Classifying brain MRI scans to flag likely findings (glioma, meningioma,
pituitary tumor, or no tumor) using classical machine learning on
radiologist-relevant image features — a full pipeline from raw scans to a
usable prediction script.

> **Disclaimer:** This is an academic/portfolio project, not a medical device.
> It is not validated for clinical use and must not be used to make real
> diagnostic decisions. See [Limitations](#limitations--ethical-notes).

---

## 1. Problem Statement

Radiologists look for specific visual patterns in an MRI scan — the shape and
edges of an abnormal mass, and how "textured" vs. uniform a tissue region
looks — to flag findings like tumors. This project builds an ML pipeline that
extracts those same kinds of signals (edge/shape structure and texture) from
MRI images and trains a classifier to predict one of four findings:
`glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, or `no_tumor`.

**Why not a raw-pixel CNN?** A convolutional neural network is the standard
approach for image classification and would likely outperform this pipeline
given a GPU and a larger compute budget. This project deliberately uses
**classical ML on engineered image features** instead — it's fully
interpretable (you can inspect exactly which features drive a prediction), it
trains in seconds instead of hours, and it runs on any laptop with no GPU. See
[Extending This Project](#9-extending-this-project) for how to move to a CNN.

## 2. Dataset

**Brain Tumor Classification (MRI) Dataset** — 3,264 labeled brain MRI scans
across 4 classes.

| Split | glioma | meningioma | pituitary | no tumor | Total |
|---|---|---|---|---|---|
| Training | 826 | 822 | 827 | 395 | 2,870 |
| Testing | 100 | 115 | 74 | 105 | 394 |

Source: `https://github.com/sartajbhuvaji/brain-tumor-classification-dataset`
(images are 512×512 JPEGs; not committed to this repo due to size — run
`src/00_download_data.sh` to fetch them).

## 3. Methodology

### 3.1 Feature Extraction (`src/01_feature_extraction.py`)
Since classical ML models need a fixed-length numeric vector rather than a raw
pixel grid, each image is converted into two well-established feature
families used in medical image analysis:

- **HOG (Histogram of Oriented Gradients)** — captures shape and edge
  structure; picks up the boundary of an abnormal mass against normal tissue.
- **GLCM (Gray-Level Co-occurrence Matrix) texture features** — contrast,
  dissimilarity, homogeneity, energy, correlation, ASM, computed at multiple
  distances and angles; captures how textured vs. uniform a region is, which
  tends to differ between tumor and healthy tissue.

Each image → resized to 128×128 → **1,812-dimensional** feature vector.

### 3.2 Dimensionality Reduction
1,812 features on ~2,870 training images risks overfitting and makes SVM/KNN
slow. PCA reduces this to **363 components while retaining 95% of the
variance** (see `results/pca_variance.png`).

### 3.3 Models Compared
Six algorithms were trained on the PCA-reduced features and evaluated on the
held-out test set:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (RBF kernel)
- Random Forest
- Gradient Boosting
- XGBoost

### 3.4 Hyperparameter Tuning
XGBoost was the strongest baseline, so it was tuned further
(`src/02b_tune_best_model.py`) over `n_estimators`, `max_depth`, and
`learning_rate` using a held-out validation split (rather than full k-fold
cross-validation, which is too slow for this project's single-core compute
budget — see note in that script).

## 4. Results

| Model | Accuracy | F1-Macro |
|---|---|---|
| **XGBoost** | **0.782** | **0.734** |
| Random Forest | 0.766 | 0.716 |
| Logistic Regression | 0.764 | 0.714 |
| Gradient Boosting | 0.761 | 0.716 |
| Support Vector Machine | 0.718 | 0.674 |
| K-Nearest Neighbors | 0.500 | 0.458 |

**Final tuned model: XGBoost** — Accuracy 77.4%, F1-Macro 0.723 on the test set.

```
                  precision    recall  f1-score   support
    glioma_tumor       1.00      0.18      0.31       100
meningioma_tumor       0.82      0.98      0.89       115
        no_tumor       0.66      1.00      0.80       105
 pituitary_tumor       0.86      0.93      0.90        74
```

### An honest limitation, not a bug
The model is strong on meningioma, no-tumor, and pituitary scans (89–90% F1),
but **weak on glioma** (31% F1, 18% recall) — when it predicts glioma it's
always right (100% precision), but it misses most actual glioma cases,
usually confusing them with pituitary or no-tumor. This is a real, reportable
finding: gliomas are known to be visually diffuse and infiltrative (less
sharply bounded than meningiomas or pituitary tumors), so hand-engineered
edge/texture features struggle to capture them as reliably as a CNN's learned
features would. See `results/confusion_matrix.png`.

Full comparison: `results/model_comparison.csv` · Plots:
`results/model_comparison.png`, `results/pca_variance.png`,
`results/confusion_matrix.png`

## 5. Project Structure

```
mri-tumor-detection-ml/
├── data/
│   ├── raw_training/, raw_testing/    # MRI images (fetched via 00_download_data.sh, gitignored)
│   ├── X_train.npy, X_test.npy        # extracted HOG+GLCM features (committed — small)
│   └── y_train.csv, y_test.csv        # labels
├── src/
│   ├── 00_download_data.sh            # fetches the raw MRI dataset
│   ├── 01_feature_extraction.py       # image -> HOG+GLCM feature vector
│   ├── 02_train_models.py             # trains + compares 6 models, saves preprocessing artifacts
│   └── 02b_tune_best_model.py         # tunes XGBoost, saves final model + confusion matrix
│   └── 03_predict.py                  # predicts the finding for a new scan
├── models/                            # scaler, PCA transform, label encoder, trained model
├── results/                           # plots + metrics
├── requirements.txt
├── LICENSE
└── README.md
```

## 6. How to Run

```bash
git clone https://github.com/<your-username>/mri-tumor-detection-ml.git
cd mri-tumor-detection-ml
pip install -r requirements.txt

# 1. Get the raw MRI images (skip this if you only want to use the
#    already-extracted features included in data/)
bash src/00_download_data.sh

# 2. Extract features from the raw images (only needed if you ran step 1)
python3 src/01_feature_extraction.py

# 3. Train and compare 6 baseline models
python3 src/02_train_models.py

# 4. Tune the best model and produce the final confusion matrix
python3 src/02b_tune_best_model.py

# 5. Predict on a new scan
python3 src/03_predict.py path/to/scan.jpg
```

The extracted features (`data/X_train.npy`, `data/X_test.npy`) and labels are
already included in this repo, so **you can skip straight to steps 3–5** and
still reproduce the full result without downloading the raw images at all.

## 7. Example Prediction

```
$ python3 src/03_predict.py data/raw_testing/pituitary_tumor/image(10).jpg

Predicted finding: pituitary_tumor
Confidence: 0.91

Full class probabilities:
  glioma_tumor: 0.02
  meningioma_tumor: 0.05
  no_tumor: 0.02
  pituitary_tumor: 0.91
```

## 8. Limitations & Ethical Notes

- **Not a diagnostic tool.** A real clinical screening tool would need
  regulatory clearance, prospective validation on external hospital data, and
  radiologist oversight — this is a portfolio-scale demonstration only.
- **Known weak spot on glioma** (see Results above) — a model this size
  should not be trusted to rule out glioma in a real patient.
- **Single dataset, single source.** All images come from one public dataset;
  MRI scanner settings, patient demographics, and acquisition protocols vary
  across hospitals, and this model has not been tested on other sources.
- **No causal claims.** Feature importance reflects predictive association
  learned by the model, not a proven biological mechanism.

## 9. Extending This Project
- **Move to a CNN** (transfer learning on ResNet/EfficientNet pretrained on
  ImageNet, fine-tuned on this dataset) — would very likely close the glioma
  gap, since CNNs learn their own features instead of relying on hand-crafted
  HOG/GLCM descriptors.
- **Add patient report text** — combine the image model's prediction with
  structured fields from a patient's clinical report (age, symptoms, prior
  history) in a multimodal model.
- **Segmentation** instead of classification — localize *where* the
  abnormality is in the scan, not just whether one is present.

## 10. Tech Stack
Python · scikit-image (HOG, GLCM) · scikit-learn · XGBoost · Matplotlib ·
Seaborn · joblib

## Author
Kruthik Balasubramanian — [LinkedIn](https://linkedin.com/in/kruthik-balasubramanian-9baa15283)
