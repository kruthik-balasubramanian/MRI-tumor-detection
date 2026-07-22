"""
Step 3: Inference — Classify a New Patient's Brain MRI Scan
Early Disease Finding Detection from MRI — using ML

Usage:
    python3 src/03_predict.py path/to/scan.jpg
    (or import predict_scan() in your own code)
"""
import sys
import json
import joblib
import numpy as np
from skimage.io import imread

sys.path.insert(0, "src")
from importlib import import_module
fe = import_module("01_feature_extraction")  # reuse extract_features()

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"
PCA_PATH = "models/pca.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"


def load_pipeline():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, pca, le


def predict_scan(image_path: str) -> dict:
    """
    image_path: path to a brain MRI scan (jpg/png), any size — it will be
    resized and processed exactly like the training data.

    Returns: dict with predicted class and per-class probabilities.
    """
    model, scaler, pca, le = load_pipeline()

    img = imread(image_path, as_gray=True)
    feat = fe.extract_features(img).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    feat_pca = pca.transform(feat_scaled)

    pred_idx = model.predict(feat_pca)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    proba = model.predict_proba(feat_pca)[0]

    proba_dict = {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)}

    return {
        "prediction": pred_label,
        "confidence": round(float(proba[pred_idx]), 3),
        "class_probabilities": proba_dict,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Fallback demo: pick a sample image from the test set
        import os
        demo_dir = "data/raw_testing/glioma_tumor"
        if os.path.isdir(demo_dir):
            demo_file = os.path.join(demo_dir, os.listdir(demo_dir)[0])
            print(f"No image path given — running on a demo scan: {demo_file}\n")
            image_path = demo_file
        else:
            print("Usage: python3 src/03_predict.py path/to/scan.jpg")
            sys.exit(1)
    else:
        image_path = sys.argv[1]

    result = predict_scan(image_path)
    print(f"Scan: {image_path}")
    print(f"\nPredicted finding: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print("\nFull class probabilities:")
    for cls, p in result["class_probabilities"].items():
        print(f"  {cls}: {p}")
