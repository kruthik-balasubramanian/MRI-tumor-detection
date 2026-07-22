"""
Step 1: MRI Image Feature Extraction
Early Disease Finding Detection from Brain MRI Scans — using ML

Why feature extraction instead of raw pixels?
Classical ML models (SVM, Random Forest, etc.) don't work directly on raw pixel
grids the way CNNs do — they need a fixed-length numeric feature vector that
captures the *texture and shape patterns* radiologists actually look for
(irregular masses, tissue density changes, edge patterns). This script extracts
two well-established, interpretable feature families used in medical image
classification literature:

  1. HOG (Histogram of Oriented Gradients) — captures shape/edge structure,
     good at picking up the boundary of an abnormal mass against normal tissue.
  2. GLCM (Gray-Level Co-occurrence Matrix) texture features — contrast,
     homogeneity, energy, correlation, dissimilarity, ASM — capture how
     "textured" vs "uniform" a region is, which differs between tumor and
     healthy tissue.
"""
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.util import img_as_ubyte

IMG_SIZE = (128, 128)
CLASSES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]


def extract_features(img_gray):
    """Given a grayscale image (0-1 float), return a combined HOG + GLCM feature vector."""
    img_resized = resize(img_gray, IMG_SIZE, anti_aliasing=True)

    # --- HOG: shape / edge structure ---
    hog_feat = hog(
        img_resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        orientations=9, feature_vector=True
    )

    # --- GLCM: texture ---
    img_ubyte = img_as_ubyte(img_resized)
    glcm = graycomatrix(
        img_ubyte, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256, symmetric=True, normed=True
    )
    texture_feat = np.hstack([
        graycoprops(glcm, prop).flatten()
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    ])

    return np.concatenate([hog_feat, texture_feat])


def process_split(split_dir, split_name):
    X, y, paths = [], [], []
    for label in CLASSES:
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(os.listdir(class_dir))
        print(f"  {split_name}/{label}: {len(files)} images")
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                img = imread(fpath, as_gray=True)
            except Exception as e:
                print(f"    skipped {fpath}: {e}")
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(label)
            paths.append(fpath)
    return np.array(X, dtype=np.float32), np.array(y), paths


if __name__ == "__main__":
    print("Extracting features from training set...")
    X_train, y_train, _ = process_split("data/raw_training", "train")
    print(f"Training feature matrix: {X_train.shape}")

    print("\nExtracting features from testing set...")
    X_test, y_test, _ = process_split("data/raw_testing", "test")
    print(f"Testing feature matrix: {X_test.shape}")

    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    pd.Series(y_train).to_csv("data/y_train.csv", index=False, header=["label"])
    pd.Series(y_test).to_csv("data/y_test.csv", index=False, header=["label"])

    print("\nSaved: data/X_train.npy, data/X_test.npy, data/y_train.csv, data/y_test.csv")
    print(f"Feature vector length: {X_train.shape[1]} (HOG + GLCM texture features)")
