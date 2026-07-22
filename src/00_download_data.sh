#!/bin/bash
# Downloads the open-source Brain Tumor Classification (MRI) dataset into data/
# Source: https://github.com/sartajbhuvaji/brain-tumor-classification-dataset
# ~3,264 labeled MRI images across 4 classes (glioma, meningioma, pituitary, no tumor)

set -e
cd "$(dirname "$0")/.."

echo "Cloning dataset..."
git clone --depth 1 https://github.com/sartajbhuvaji/brain-tumor-classification-dataset.git /tmp/mri-dataset-tmp

mkdir -p data
mv /tmp/mri-dataset-tmp/Training data/raw_training
mv /tmp/mri-dataset-tmp/Testing data/raw_testing
rm -rf /tmp/mri-dataset-tmp

echo "Done. Dataset is in data/raw_training and data/raw_testing"
