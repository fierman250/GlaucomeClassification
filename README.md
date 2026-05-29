# Glaucoma Classification 

A modular, end-to-end Machine Learning pipeline for the detection and classification of Glaucoma using deep learning. This repository includes dataset preprocessing, dynamically optimized model training (via VRAM preloading), and an interactive UI Dashboard.

---

## 📁 Repository Structure

The core functionality of this project is housed inside the `Python_Code` directory, which follows a clean, modular design:

```text
Python_Code/
├── src/
│   ├── eda/                 # Data analysis and offline preprocessing filters (CLAHE, Gamma, etc.)
│   ├── data_processing/     # VRAM Dataset caching, DataLoader logic, and GPU-native oversampling
│   └── models/              # Pretrained CNN architectures, Trainer scripts, and Grad-CAM explainability
│
├── step_by_step_training.py # Interactive VS Code notebook script for step-by-step training
├── run_custom_experiments.py# Customizable batch training script to iterate over models and datasets
├── main_dashboard.py        # Streamlit-based interactive Web UI for clinical evaluation
└── requirements.txt         # Project dependencies
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r Python_Code/requirements.txt
```

### 2. Training the Models

This project supports two primary workflows depending on your needs:

#### A) Interactive Step-by-Step Training (The Playground)
If you want to manually step through the data loading, observe the dataset distribution, and watch the training loss decrease epoch-by-epoch, use the interactive script. 
Open `step_by_step_training.py` in VS Code and click **"Run Cell"** above each `# %%` block.

#### B) Automated Batch Training (The Workhorse)
If you want to queue up multiple experiments and walk away, use the batch runner. Open `run_custom_experiments.py`, edit the lists at the top of the file to choose your models/datasets, and run:
```bash
python Python_Code/run_custom_experiments.py
```
*Note: All results (Confusion Matrices, ROC Curves, `.pth` Weights, and an Excel Summary) will be saved to the `results/` folder.*

---

## 🖥️ Clinical UI Dashboard

This project features a fully interactive Streamlit web dashboard. The UI allows users to upload a raw fundus image, apply preprocessing filters in real-time, predict Glaucoma using trained models, and visualize the prediction via **Grad-CAM** heatmaps.

To launch the dashboard, run:
```bash
cd Python_Code
streamlit run main_dashboard.py
```

---

## 🧠 Key Technologies
- **PyTorch / Torchvision**: Deep Learning modeling and GPU execution.
- **Scikit-learn**: Data splitting and stratified balancing.
- **OpenCV**: Image preprocessing (Histogram Equalization, CLAHE, Gamma Correction).
- **Streamlit**: Web interface for visualization and evaluation.
- **Grad-CAM**: Explainable AI implementation to highlight clinically relevant areas of the fundus.

*(Note: Original raw datasets and large model weights are intentionally excluded from this repository via `.gitignore`)*
