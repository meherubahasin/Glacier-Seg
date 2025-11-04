# Glacier-Seg: Lightweight Hybrid CNN–Transformer–Mamba Architecture for Glacier Segmentation

## Overview
This repository contains the code, data processing scripts, and model implementations for **Glacier-Seg**, a compact and high-performing hybrid architecture designed for glacier segmentation using multi-modal remote sensing data (SAR, optical, DEM). The project aims to achieve accurate, efficient, and near real-time glacier mapping for both satellite and drone imagery.

Key features include:
- Multi-modal data handling (SAR, optical, DEM)
- Geometric, radiometric, and photometric data augmentation
- CNN, Transformer, and hybrid model architectures
- Evaluation with IoU, Dice, Precision, Recall, Pixel Accuracy, and AUC-ROC
- Lightweight design for edge deployment

## Repository Structure
├── ablation/
│   └──              # Previously held ablation experiment notebook
│
├── data/
│   ├── GlacierSAR.py                          # Glacier SAR dataset class (e.g., for Sentinel-1 imagery)
│   └── GlacierDatasetHKH.py                   # Glacier dataset for HKH region (Hindu Kush Himalaya)
│
├── src/
│   └──         
│
├── tests/
│   └── temporal.ipynb                         # Temporal or time-series analysis/testing notebook
│
├── visualisations/
│   └── notebooks/                             # Visualization notebooks for qualitative results
│
├── README.md                                  # Project overview, setup, and usage guide
│
├── T2420368_Final_ViT_Glaciers.pdf            # Final report/paper on ViT-based glacier segmentation
└── T2420368_IEEE_Journal.pdf                  # IEEE-formatted version of the paper
