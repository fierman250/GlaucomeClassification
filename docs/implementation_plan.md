# Implementation Plan - Glaucoma Classification Python Migration

This plan outlines the architecture and implementation steps to migrate the MATLAB glaucoma classification codebase to Python. The new Python codebase is designed to meet all 5 grading rubric items:
1. **EDA and Data Preprocessing**
2. **Five Classifier Implementation**
3. **ROC Curves and Confusion Matrix**
4. **Cross-Validation Analysis**
5. **Feature Importance (Grad-CAM) and Conclusion**

To provide a premium and interactive experience, we will also build a **Streamlit Web Dashboard** to replace the MATLAB `.mlapp` GUI.

---

## Agreed Implementation Decisions

> [!NOTE]
> **1. Selection of the 5 Classifiers**
> We will implement the following 5 Deep Learning CNN architectures in PyTorch:
> * **Model 1: AlexNet** (Transfer learning via `torchvision.models.alexnet`)
> * **Model 2: VGG-16** (Transfer learning via `torchvision.models.vgg16`)
> * **Model 3: DenseNet-121** (Transfer learning via `torchvision.models.densenet121`)
> * **Model 4: ResNet-18** (Transfer learning via `torchvision.models.resnet18`)
> * **Model 5: MobileNetV2** (Transfer learning via `torchvision.models.mobilenet_v2`)
> All 5 models are loaded with pre-trained ImageNet weights and adapted for 2-class classification.

> [!NOTE]
> **2. Selection of the 4 Preprocessing Methods**
> We will apply and compare 4 distinct image processing configurations:
> * **Original**: Resize to `[224, 224]` with no additional filters.
> * **Histogram Equalization (HE)**: Standard global histogram equalization using OpenCV.
> * **Gamma Correction**: Luminance scaling (power-law transformation).
> * **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Local contrast enhancement (historically the best performer at 84.96%).
> *(Note: Low-Light Image Enhancement (LLIE) is excluded as per requirements).*

> [!NOTE]
> **3. Unified Benchmarking & Recording (4 Preprocessing x 5 Classifiers = 20 Experiments)**
> We will run a complete grid search of all 20 configurations. For each configuration, we will:
> * Run K-Fold Cross-Validation (5-fold).
> * Generate and save a **Confusion Matrix** plot (summed/averaged over folds).
> * Generate and save the **ROC Curve** plot (with mean AUC).
> * Calculate and record all standard performance metrics (Accuracy, Sensitivity, Specificity, Precision, F1-Score) across all folds.
> * Save all quantitative metrics in a master excel/csv workbook (e.g. `glaucoma_classification_benchmark.xlsx`) for easy analysis and comparison.

> [!NOTE]
> **4. Environment & GPU Acceleration**
> * Conda Environment: `SimTR` (Python 3.10.14)
> * Framework: PyTorch 2.5.1
> * GPU: NVIDIA GeForce RTX 3090 Ti (CUDA is active and verified)
> * Dependencies: All required libraries (`torch`, `torchvision`, `scikit-learn`, `opencv-python`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`) are already pre-installed.

> [!NOTE]
> **5. Dataset Path**
> * Training and Cross-Validation will run on the extracted images path. The scripts will keep paths parameterized so they can easily scale to larger datasets (e.g. after uncompressing `.rar` sources).

> [!NOTE]
> **6. Logging and Progress Monitoring**
> * We will integrate Python's standard `logging` module in all scripts, writing detailed records to both the console (stdout) and a persistent log file (`glaucoma_pipeline.log`).
> * The logs will track:
>   * **EDA**: Database file counts, class balance checks, and dataset validation.
>   * **Preprocessing**: File-by-file status, errors encountered, and enhancement output paths.
>   * **Training**: Epoch-by-epoch loss, validation accuracy, learning rates, fold transitions, and model checkpoint saves for all 20 configurations.
>   * **Error Handling**: Full tracebacks for exceptions, enabling easy debugging.

---

## Proposed Changes

We will create a new directory `Python Code` in the root workspace to contain all our files.

```
c:\Users\laimm\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\
├── Project Overviews.md           # Master documentation: dataset profiles, challenge notes, & benchmarking results log
└── Python Code\
    ├── eda_and_preprocessing.py   # Rubric 1: EDA, statistics, 4 preprocessing methods (HE, Gamma, CLAHE)
    ├── classifiers.py             # Rubric 2: PyTorch CNN configurations (AlexNet, VGG-16, DenseNet-121, ResNet-18, MobileNetV2)
    ├── train_and_evaluate.py      # Rubric 3 & 4: 20-experiment benchmarking grid, K-fold CV, ROC, Confusion Matrix, results recorder
    ├── explainability.py          # Rubric 5: Grad-CAM heatmap generator for CNN decisions
    ├── dashboard.py               # Streamlit App (replaces MATLAB GUI, displays benchmark comparisons, interactive testing)
    └── requirements.txt           # Python dependency checklist
```

### [Documentation & Config Modules]

#### [NEW] [Project Overviews.md](file:///c:/Users/laimm/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Project%20Overviews.md)
* Contains executive summary, comprehensive image counts and descriptions of all 15 clinical databases, challenge discussions (imbalance and zoom), and a detailed results log which we will update dynamically as soon as benchmarking runs finish.

### [Python Code Module]

#### [NEW] [requirements.txt](file:///c:/Users/laimm/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python%20Code/requirements.txt)
Defines the required Python dependencies:
* `torch` and `torchvision` (Deep Learning models)
* `opencv-python` (Image processing: CLAHE, resizing)
* `scikit-learn` (Metrics, cross-validation)
* `pandas` and `numpy` (Data manipulation and exporting results)
* `matplotlib` and `seaborn` (EDA, ROC, and Confusion Matrix plotting)
* `streamlit` (Interactive user dashboard)

#### [NEW] [eda_and_preprocessing.py](file:///c:/Users/laimm/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python%20Code/eda_and_preprocessing.py)
* **EDA**: Calculates dataset stats (class balance, mean aspect ratio, color channel distributions) and generates descriptive plots.
* **Preprocessing**: Implements exact Python equivalents of the 4 MATLAB operations:
  * Resize to `[224, 224]`.
  * Histogram Equalization (`cv2.equalizeHist` on L channel in LAB color space).
  * Gamma Correction (power-law lookup table).
  * CLAHE (`cv2.createCLAHE` on L channel in LAB, or green channel, matching MATLAB's best option).

#### [NEW] [classifiers.py](file:///c:/Users/laimm/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python%20Code/classifiers.py)
* Defines the 5 CNN architectures loaded via `torchvision.models` (DenseNet-121, AlexNet, VGG-16, ResNet-18, and MobileNetV2) and replaces their final classification heads with custom linear layers for 2-class output.

#### [NEW] [train_and_evaluate.py](file:///c:/Users/laimm/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python%20Code/train_and_evaluate.py)
* Runs the **20-experiment loop** matching all combinations of the 4 preprocessing methods and 5 classifiers.
* Runs **K-Fold cross-validation** (5-fold) using `StratifiedKFold`.
* **Class Imbalance Handling**: Implements a `WeightedRandomSampler` in the PyTorch `DataLoader` for the training split. This dynamically oversamples the minority class (POSITIVE) so each batch is balanced (approx. 50/50), preventing model bias.
* **PyTorch Data Augmentation**: For the training dataset, we apply an advanced augmentation pipeline using `torchvision.transforms`:
  - `RandomHorizontalFlip()` & `RandomVerticalFlip()` (reflection)
  - `RandomRotation(degrees=15)` (rotation to handle tilted images)
  - `RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2))` (translation and zoom in/out to handle varied fields of view)
  - `RandomResizedCrop(size=224, scale=(0.8, 1.0))` (random cropping and resizing)
* Evaluates cross-validation performance (Accuracy, Sensitivity, Specificity, Precision, F1-score).
* Plots and saves combined **Confusion Matrices** and **ROC Curves** (with mean AUC) for all 20 experiments under a structured folder structure (`results/<model>_<preprocessing>/`).
* Saves the quantitative metrics of all folds and experiments into `glaucoma_classification_benchmark.xlsx`.
* Saves the best-performing weights for each configuration.

#### [NEW] [explainability.py](file:///c:/Users/laimm/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python%20Code/explainability.py)
* Implements **Grad-CAM** or **Score-CAM** for the CNN architectures.
* Generates class activation maps indicating regions of interest (e.g. optic cup/disc) and overlays them on the input images.

#### [NEW] [dashboard.py](file:///c:/Users/laimm/OneDrive/A4_JobREF/9Z_CodeWork/2403CW_GlaucomeClassification/Python%20Code/dashboard.py)
A Streamlit web dashboard structured into tabs:
1. **EDA**: Dynamic statistics, charts, and sample images of the dataset.
2. **Preprocessing**: Upload a fundus image and side-by-side visualize original, HE, Gamma, and CLAHE.
3. **Training & Evaluation**: View model comparisons (accuracies, sensitivity, specificity, ROC curves, confusion matrices for the 20 configurations, reading directly from saved results and tables).
4. **Interactive Tester**: Select any of the 5 trained models and any of the 4 preprocessing methods, upload/select a test image, see the classification result, and view the Grad-CAM activation heatmap overlay.

---

## Verification Plan

### Automated Tests
* Run the preprocessing pipeline on sample test images and visually inspect saved output.
* Run a training session on a quick subset (e.g. 5 epochs) to verify the training loops, metric computations, and plot generations.
* Run Streamlit locally (`streamlit run dashboard.py`) and use the browser tool or manual inspection to check UI functionality.

### Manual Verification
* The user can open the local Streamlit dashboard to interactively test image enhancements and model classifications.
