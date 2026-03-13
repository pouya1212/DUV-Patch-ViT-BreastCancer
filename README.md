# Breast Cancer Classification in Deep Ultraviolet Fluorescence Images Using a Patch-Level Vision Transformer Framework

This repository implements the framework proposed in:

*Breast Cancer Classification in Deep Ultraviolet Fluorescence Images Using a Patch-Level Vision Transformer Framework*  
[PDF link](https://www.researchgate.net/publication/398306055_Breast_Cancer_Classification_in_Deep_Ultraviolet_Fluorescence_Images_Using_a_Patch-Level_Vision_Transformer_Framework)


---

## **Authors & Affiliations**

**Pouya Afshin, David Helminiak, Tongtong Lu, Tina Yen, Julie M. Jorns, Mollie Patton, Bing Yu, Dong Hye Ye 

1. Department of Computer Science, Georgia State University  
2. Department of Electrical and Computer Engineering, Marquette University  
3. Department of Bioengineering, Marquette University  
4. Department of Surgery, Medical College of Wisconsin  
5. Department of Pathology, Medical College of Wisconsin  

![Universities](figures/Univerisities.png)



---

## **Project Overview**

Breast-conserving surgery (BCS) requires removing malignant tissue while preserving healthy tissue.  

![BCS](figures/BCS.jpg)

Whole-slide images (WSIs) of excised breast tissue are acquired using a deep ultraviolet fluorescence scanning microscope (DUV-FSM), which provides high-contrast visualization of malignant and normal regions.

![WSI](figures/WSI_Project.png)

A patch-level Vision Transformer (ViT) framework is employed to address the challenges posed by high-resolution images and complex histopathology. Both local and global features are captured by the model to enable robust breast cancer classification. Additionally, Grad-CAM++ is used to generate saliency-based visualizations that highlight diagnostically relevant regions and enhance interpretability. The approach is evaluated using 5-fold cross-validation, and its performance is shown to surpass conventional deep learning methods, achieving a classification accuracy of 98.33% for distinguishing benign and malignant tissue.
---

## Pipeline: DUV WSI Classification

![system-model](figures/system-model.png)

1. Divide each DUV WSI into non-overlapping patches.

2. Subdivide each patch into smaller sub-patches and transform them into learnable position and class embeddings.

3. Pass embeddings through the Vision Transformer (ViT) encoder to update class embeddings.

4. Classify each patch using the MLP head.

5. Generate Grad-CAM++ maps with a fine-tuned CNN to obtain patch-level importance weights.

6. Fuse patch-level predictions with Grad-CAM++ weights to obtain the final WSI-level classification.

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
