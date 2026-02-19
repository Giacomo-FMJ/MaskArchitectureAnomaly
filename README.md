# Mask Architecture Anomaly Segmentation for Road Scenes

**Course Project:** [Machine Learning Course Project](https://drive.google.com/file/d/19gQ2uhI8jPxWIdGewkaA2DqihUnQLVfB/view?usp=drive_link)

This repository provides a code setup for the Real-Time Anomaly Segmentation project of the Machine Learning Course. It consists of the code base for training/testing ERFNet on the Cityscapes dataset and performing anomaly segmentation. It also contains code referring to EoMT.

---

## Branch: Pixel-wise with Visual Features Anomaly Head Extension

> **This branch implements an extended version of EoMT (EoMT-EXT) specifically designed for anomaly segmentation using a dedicated Pixel-wise Anomaly Head that includes also visual features.**

### Key Innovation

This extension introduces a **learnable Pixel-wise Anomaly Head** that operates on statistical uncertainty features **combined with visual features from the backbone** derived from frozen semantic heads (Mask Head and Class Head). The architecture classifies each pixel into two states:

- **Normal** (In-Distribution pixel)
- **Anomaly** (Out-of-Distribution pixel)

The core hypothesis is that **statistical uncertainty metrics** (Entropy, Max Probability, Energy, MSP) **combined with visual features** extracted from the frozen DinoV2 backbone contain sufficient information to distinguish between ID and OOD regions, without manual threshold tuning.

### Architecture Overview

- **Frozen Mask & Class Heads**: Repurposed as uncertainty signal generators
- **Statistical Feature Extraction**: Computes 4 uncertainty metrics from semantic predictions
- **Visual Features Fusion**: Extracts visual features from the final backbone layer and interpolates them to match spatial resolution
- **Pixel-wise MLP**: Learns to decode anomaly patterns from the concatenation of statistical features and visual features (hybrid approach)

**For complete architectural details, theoretical foundations, and training procedures, see:** [eomt/README.md](eomt/README.md)

---

## Validation & Inference

The notebook [eomt/model evaluation -COLAB.ipynb](eomt/model%20evaluation%20-COLAB.ipynb) provides:

- **Model loading** with the extended EoMT-EXT architecture including the Pixel-wise Anomaly Head
- **Inference pipeline** that utilizes learned anomaly predictions instead of traditional uncertainty baselines
- **Validation on 5 anomaly detection datasets**: FS LaF, FS Static, LostAndFound, RoadAnomaly21, RoadObstacle21
- **Metric computation**: AuPRC and FPR95 for anomaly detection performance
- **Visualization**: Anomaly heatmaps comparing predictions with ground truth

---

## Packages

For more instructions, please refer to the README in each folder:

- **[eval](eval)** - Contains tools for evaluating/visualizing ERFNet model's output and performing anomaly segmentation
- **[trained_models](trained_models)** - Contains the ERFNet trained models for the baseline evaluation
- **[eomt](eomt)** - Almost the original folder of the EoMT project with the pixel-wise anomaly head extension. Inside you will find code to train the extended architecture and pretrained checkpoints

