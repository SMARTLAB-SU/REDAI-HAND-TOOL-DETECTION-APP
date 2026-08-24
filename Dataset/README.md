# Dataset Overview & Preprocessing Guidelines

This folder contains dataset specifications, annotation formats, and analysis reports for the REDAI Hand Tool Detection system.

## 📊 Dataset Specifications

- **Classes**: 14 distinct hand tool categories.
- **Format**: YOLO Darknet format (`.txt` bounding box coordinates normalized to `[0, 1]`).
- **Annotations**: Bounding box object detection and segmentation masks.

## 🔄 Preprocessing Steps

1. **Resolution Normalization**: Input images resized to standard 640x640 dimensions.
2. **Data Augmentation**: Mosaic augmentation, random horizontal flipping, color jittering, and rotation.
3. **Format Conversion**: Roboflow exported annotations validated against `data.yaml`.

## 📁 Subdirectory Contents

- `Report/`: Contains exploratory data analysis (EDA) reports, class distribution graphs, and annotation balance charts.
