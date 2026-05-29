# Glaucoma Classification Project Overview

This document serves as the central log and repository for the glaucoma classification project. It contains detailed dataset profiles, preprocessing configurations, classification model architectures, and training results. Use this information directly for report generation and PowerPoint (PPT) slides.

---

## 1. Project Goal & Executive Summary
The goal of this project is to build and benchmark a robust, generalizable computer vision pipeline to classify eye fundus images as either **`POSITIVE`** (Glaucoma) or **`NEGATIVE`** (Normal). 

To achieve clinical-grade reliability, the project combines **15 distinct ophthalmic databases** from global medical sources. We apply **4 image preprocessing techniques** (Original, HE, Gamma, CLAHE) and compare **5 state-of-the-art Convolutional Neural Network (CNN) architectures** (AlexNet, VGG-16, DenseNet-121, ResNet-18, MobileNetV2) trained using Stratified 5-Fold Cross-Validation on GPU.

---

## 2. Comprehensive Dataset Profile
The database contains **12,317 images** in total:
* **Train Set**: **9,290 images** (3,515 POSITIVE / 5,775 NEGATIVE)
* **Test Set**: **3,027 images** (1,273 POSITIVE / 1,754 NEGATIVE)

### Ophthalmic Database Breakdown (For PPT Slides)
Below is the clinical origin and file count for each of the 15 databases included in the combined train/test sets:

| Database Name | Clinical Origin & Description | Train Set Count | Test Set Count |
| :--- | :--- | :--- | :--- |
| **BEH** | **Beijing Eye Study**: Large population-based study in China. Standard fundus camera. Balanced demographics. | 461 total<br>(127 Pos / 334 Neg) | 154 total<br>(46 Pos / 108 Neg) |
| **CRFO** | **Clinica Rotger Fundus Ophthalmoscopy**: Registry data from private clinics in Spain. High-contrast images. | 52 total<br>(34 Pos / 18 Neg) | 13 total<br>(9 Pos / 4 Neg) |
| **DRH** | **Drishti-GS1 & DR-HAGIS**: Clinical datasets from India and the UK focusing on glaucoma and diabetic retinopathy. | 80 total<br>(58 Pos / 22 Neg) | 30 total<br>(18 Pos / 12 Neg) |
| **EyePACS** | **Eye Photo Analysis Group**: Multi-ethnic teleretinal screening platform in the US. Contains only positive glaucoma cases here. | 2,282 total<br>(2,282 Pos / 0 Neg) | 767 total<br>(767 Pos / 0 Neg) |
| **FIVES** | **Multi-Disease Fundus Image Dataset**: Quality-evaluated images with annotations for clinical screening. | 271 total<br>(103 Pos / 168 Neg) | 96 total<br>(40 Pos / 56 Neg) |
| **G1020** | **G1020 Dataset**: Large-scale public database annotated specifically for optic disc segmentation and glaucoma classification. | 709 total<br>(207 Pos / 502 Neg) | 245 total<br>(68 Pos / 177 Neg) |
| **HRF** | **High-Resolution Fundus Database**: High-fidelity fundus images from patients in Europe. | 21 total<br>(11 Pos / 10 Neg) | 13 total<br>(11 Pos / 2 Neg) |
| **JSIEC** | **Joint Shantou International Eye Center**: Large clinical database from Shantou University, China. Only negative cases in this split. | 32 total<br>(0 Pos / 32 Neg) | 8 total<br>(0 Pos / 8 Neg) |
| **LES** | **Lariboisière Eye Study**: Ophthalmic registry database from France. | 19 total<br>(8 Pos / 11 Neg) | 12 total<br>(8 Pos / 4 Neg) |
| **OIA** | **Ophthalmic Image Analysis**: Dataset compiled from standard clinical ODIR challenges (Baidu/Xunlei). | 3,103 total<br>(202 Pos / 2,901 Neg) | 1,047 total<br>(78 Pos / 969 Neg) |
| **ORIGA** | **Online Retinal Fundus Image Database**: Extracted from the Singapore Malay Eye Study (SiMES). Gold-standard annotations. | 650 total<br>(168 Pos / 482 Neg) | 226 total<br>(115 Pos / 111 Neg) |
| **ORIGA2** | **ORIGA Part 2**: Additional split/subset of SiMES. | 452 total<br>(127 Pos / 325 Neg) | (Included in ORIGA test split) |
| **PAPILA** | **PAPILA Dataset**: Open clinical database from Spain containing detailed clinical annotations for glaucoma. | 305 total<br>(62 Pos / 243 Neg) | 84 total<br>(18 Pos / 66 Neg) |
| **REFUGE** | **Retinal Fundus Glaucoma Challenge**: MICCAI challenge dataset. Very popular benchmarks for disc/cup segmentation. | 573 total<br>(56 Pos / 517 Neg) | 193 total<br>(25 Pos / 168 Neg) |
| **SJCHOI** | **Choi et al. Clinical Registry**: Anonymous database named after primary clinical investigators. | 280 total<br>(70 Pos / 210 Neg) | 139 total<br>(70 Pos / 69 Neg) |
| **TOTALS** | **Combined Database Pool** | **9,290 Images**<br>(3,515 Pos / 5,775 Neg) | **3,027 Images**<br>(1,273 Pos / 1,754 Neg) |

---

## 3. Data Challenges & Preprocessing Strategy

### 3.1 Addressing Class Imbalance
The training set is imbalanced: **5,775 Normal (62%)** vs **3,515 Glaucoma (38%)**. 
* **Impact**: Standard neural networks would overfit to the Normal class, causing missed glaucoma diagnoses (high False Negatives).
* **Solution**: We implement a PyTorch `WeightedRandomSampler` inside our DataLoader. This dynamically oversamples positive images during training batches, maintaining a **50/50 balance** per batch without copying files on disk.

### 3.2 Variance in Zoom/Field-of-View
Ophthalmic images from different studies vary from tight "zooms" of the optic nerve head to wide "full-field" retina shots.
* **Solution**: In the training dataloader, we implement advanced data augmentations:
  * `RandomRotation` (15 degrees)
  * `RandomAffine` (Zoom in/out from 80% to 120%, translation)
  * `RandomResizedCrop` (simulates varying closeups of the optic nerve)

---

## 4. Evaluated Pipeline Grid
We run a grid search evaluating **20 total configurations** (4 Preprocessing Methods $\times$ 5 Classifiers).

### 4.1 Preprocessing Methods (Image Processing)
1. **Original**: Normal resizing to $224 \times 224 \times 3$.
2. **Histogram Equalization (HE)**: Standard global contrast stretching.
3. **Gamma Correction**: Adjusting luminance scale using a power-law transformation.
4. **CLAHE**: Contrast Limited Adaptive Histogram Equalization applied to the L-channel of LAB color space (historically the best method).

### 4.2 Classification Models
1. **AlexNet** (Classic early deep network, baseline)
2. **VGG-16** (Deep architecture with small convolutional filters)
3. **DenseNet-121** (Densely connected layers, highly parameter-efficient for medical imaging)
4. **ResNet-18** (Residual skip connections, prevents gradient vanishing)
5. **MobileNetV2** (Modern, lightweight network using depthwise separable convolutions)

---

## 5. Benchmarking Results log (Placeholder)
*This section will be populated dynamically as training experiments conclude.*

### 5.1 Preprocessing Method Comparison
| Preprocessing Method | Best Model | Mean Accuracy | Mean Sensitivity | Mean Specificity | Mean AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Histogram Equalization** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Gamma Correction** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **CLAHE** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### 5.2 Top 5 Classification Model Benchmarks
*Overall results averaged across the 4 preprocessing methods.*

| Model | Accuracy (Mean ± SD) | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | AUC (Mean ± SD) | Training Time (per fold) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AlexNet** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **VGG-16** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **DenseNet-121** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **ResNet-18** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **MobileNetV2** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
