# Breast Cancer Classification in Deep Ultraviolet Fluorescence Images Using a Patch-Level Vision Transformer Framework

This repository implements the framework proposed in:

**Pouya Afshin, David Helminiak, Tongtong Lu, Tina Yen, Julie M. Jorns, Mollie Patton, Bing Yu, Dong Hye Ye.**  
*Breast Cancer Classification in Deep Ultraviolet Fluorescence Images Using a Patch-Level Vision Transformer Framework*  
[PDF link](https://www.researchgate.net/publication/398306055_Breast_Cancer_Classification_in_Deep_Ultraviolet_Fluorescence_Images_Using_a_Patch-Level_Vision_Transformer_Framework)

---

## **Project Overview**

Breast-conserving surgery (BCS) requires removing malignant tissue while preserving healthy tissue.  
This project uses a deep ultraviolet fluorescence scanning microscope (DUV-FSM) to acquire whole-slide images (WSIs).  
A patch-level Vision Transformer (ViT) captures both local and global features to classify patches as benign or malignant. Grad-CAM++ is applied for explainability.

---

## **Repository Structure**

- `data/`: DUV images and generated patches.
- `configs/`: Training configuration files for ViT and synthetic patch generation.
- `models/`: Vision Transformer and ResNet backbone.
- `notebooks/`: Jupyter notebooks for experiments and analysis.
- `scripts/`: Scripts for training, validation, and inference.
- `results/`: Outputs including Grad-CAM++ visualizations.
- `requirements.txt`: Python dependencies.

---

## **Installation**

```bash
git clone https://github.com/your-username/DUV-Patch-ViT-BreastCancer.git
cd DUV-Patch-ViT-BreastCancer
pip install -r requirements.txt
