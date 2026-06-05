import copy
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from src.data_processing.dataset import (
    RAMCachedDataset,
    # balance_classes_on_gpu,
    get_processed_dataset_stats,
    preload_to_device,
)

# Import modular components
from src.eda.preprocessing import setup_logger
from src.models.classifiers import get_model
from src.models.trainer import calculate_metrics, plot_confusion_matrix, plot_roc_curve
from torch.utils.data import DataLoader

# ==============================================================================
# USER OPTIONS (EDIT THESE LISTS)
# ==============================================================================
MODELS_TO_RUN = [
    # "mobilenet_v2",
    "resnet18",
    # "alexnet",
    "vgg16",
    "densenet121",
]  # Available: 'alexnet', 'vgg16', 'densenet121', 'resnet18', 'mobilenet_v2'
DATASETS_TO_RUN = [
    "original",
    # "he",
    # "clahe",
]  # Available: 'original', 'he', 'gamma', 'clahe'
K_FOLDS = 5  # Number of folds for cross-validation
# ==============================================================================

ROOT_DATA_DIR = r"C:\Users\laimm\Processed_Datasets"
RESULTS_DIR = "results/custom_experiments"
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-4


def run_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Custom Batch Training Started. Device: {device}")

    summary_path = os.path.join(RESULTS_DIR, "custom_experiments_summary.xlsx")
    if os.path.exists(summary_path):
        try:
            all_results = pd.read_excel(summary_path).to_dict("records")
            logger.info(
                f"Loaded {len(all_results)} previous results from summary file."
            )
        except Exception:
            all_results = []
    else:
        all_results = []

    # Outer loop: Datasets
    for dataset_name in DATASETS_TO_RUN:
        dataset_path = os.path.join(ROOT_DATA_DIR, dataset_name)
        logger.info(f"\n==================================================")
        logger.info(f"  PROCESSING DATASET: {dataset_name.upper()}")
        logger.info(f"====================================================")

        # Load Dataset Metadata
        df_train = get_processed_dataset_stats(dataset_path, split="train")
        if df_train.empty:
            logger.error(f"No train data found in {dataset_path}. Skipping dataset...")
            continue

        df_test = get_processed_dataset_stats(dataset_path, split="test")

        # VRAM Preloading for the entire training set
        logger.info("Preloading Full Train Set to VRAM...")
        all_images, all_labels = preload_to_device(df_train, logger, device)

        # VRAM preloading for the test set
        test_images, test_labels = None, None
        test_loader = None
        if not df_test.empty:
            logger.info("Preloading Test Set to VRAM...")
            test_images, test_labels = preload_to_device(df_test, logger, device)
            test_dataset = RAMCachedDataset(test_images, test_labels, augment=False)
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
            )

        # Model loop
        for model_name in MODELS_TO_RUN:
            exp_name = f"{dataset_name}_{model_name}"

            # --- RESUME LOGIC for combined experiment ---
            if any(r.get("Experiment") == exp_name for r in all_results):
                logger.info(
                    f"Experiment {exp_name} is already in the Excel summary. Moving to next model..."
                )
                continue

            logger.info(
                f"\n--- Starting {K_FOLDS}-Fold Cross-Validation for: {exp_name} ---"
            )

            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
            fold_results = []

            # --- Cross-Validation Loop ---
            for fold, (train_idx, val_idx) in enumerate(
                skf.split(all_images, all_labels.cpu().numpy())
            ):
                fold_exp_name = f"{exp_name}_fold{fold + 1}"
                fold_dir = os.path.join(RESULTS_DIR, fold_exp_name)
                os.makedirs(fold_dir, exist_ok=True)
                logger.info(f"\n--- Processing Fold {fold + 1}/{K_FOLDS} ---")

                # Data for this fold
                train_img_split = all_images[train_idx]
                train_lbl_split = all_labels[train_idx]
                val_img_split = all_images[val_idx]
                val_lbl_split = all_labels[val_idx]

                # DataLoaders for this fold
                train_dataset = RAMCachedDataset(
                    train_img_split, train_lbl_split, augment=True
                )
                val_dataset = RAMCachedDataset(
                    val_img_split, val_lbl_split, augment=False
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
                )

                # --- Model training for this fold ---
                weights_path = os.path.join(fold_dir, "best_weights.pth")
                if os.path.exists(weights_path):
                    logger.info(f"Found existing weights for {fold_exp_name}.")
                    model = get_model(model_name, num_classes=2, pretrained=False).to(
                        device
                    )
                    model.load_state_dict(torch.load(weights_path))
                else:
                    logger.info(f"Training new model for {fold_exp_name}...")
                    model = get_model(model_name, num_classes=2, pretrained=True).to(
                        device
                    )
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(
                        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
                    )
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.5, patience=5
                    )
                    scaler = torch.amp.GradScaler("cuda")

                    best_val_loss = float("inf")
                    best_model_weights = None
                    history = []

                    for epoch in range(EPOCHS):
                        model.train()
                        running_loss = 0.0
                        for inputs, labels in train_loader:
                            optimizer.zero_grad()
                            with torch.amp.autocast("cuda"):
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            running_loss += loss.item() * inputs.size(0)
                        epoch_loss = running_loss / len(train_loader.dataset)

                        model.eval()
                        val_loss = 0.0
                        correct, total = 0, 0
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item() * inputs.size(0)
                                preds = torch.argmax(outputs, dim=1)
                                correct += (preds == labels).sum().item()
                                total += labels.size(0)
                        val_loss /= len(val_loader.dataset)
                        val_acc = correct / total

                        scheduler.step(val_loss)
                        logger.info(
                            f"Fold {fold + 1} | Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                        )

                        history.append(
                            {
                                "epoch": epoch + 1,
                                "train_loss": epoch_loss,
                                "val_loss": val_loss,
                                "val_acc": val_acc,
                            }
                        )

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_weights = copy.deepcopy(model.state_dict())

                    if best_model_weights:
                        model.load_state_dict(best_model_weights)
                        torch.save(best_model_weights, weights_path)
                        logger.info(f"Saved best model weights to {weights_path}")

                    # Plotting and saving history for the fold
                    history_df = pd.DataFrame(history)
                    history_df.to_csv(
                        os.path.join(fold_dir, "training_history.csv"), index=False
                    )
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(
                        history_df["epoch"],
                        history_df["train_loss"],
                        label="Train Loss",
                    )
                    plt.plot(
                        history_df["epoch"], history_df["val_loss"], label="Val Loss"
                    )
                    plt.legend()
                    plt.title(f"Loss vs. Epochs (Fold {fold + 1})")
                    plt.subplot(1, 2, 2)
                    plt.plot(
                        history_df["epoch"], history_df["val_acc"], label="Val Accuracy"
                    )
                    plt.legend()
                    plt.title(f"Accuracy vs. Epochs (Fold {fold + 1})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(fold_dir, "learning_curves.png"))
                    plt.close()

                # --- Evaluation for this fold on its validation set ---
                logger.info(f"Evaluating fold {fold + 1} on its validation set...")
                model.eval()
                y_true, y_pred, y_prob = [], [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        preds = torch.argmax(outputs, dim=1)
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())
                        y_prob.extend(probs.cpu().numpy())

                metrics = calculate_metrics(y_true, y_pred, y_prob)
                fold_results.append(metrics)
                logger.info(
                    f"Fold {fold + 1} Validation | Acc: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f}"
                )

                del model
                gc.collect()
                torch.cuda.empty_cache()

            # --- Aggregating CV Results ---
            if fold_results:
                avg_metrics = {
                    k: np.mean([r[k] for r in fold_results]) for k in fold_results[0]
                }
                std_metrics = {
                    k: np.std([r[k] for r in fold_results]) for k in fold_results[0]
                }
                logger.info(
                    f"\n--- Average CV Results for {exp_name} ({K_FOLDS} folds) ---"
                )
                for key in avg_metrics:
                    if key not in ["cm", "fpr", "tpr"]:
                        logger.info(
                            f"  - Avg {key.capitalize()}: {avg_metrics[key]:.4f} (+/- {std_metrics[key]:.4f})"
                        )
            else:
                logger.error(
                    f"No fold results to aggregate for {exp_name}. Skipping final eval."
                )
                continue

            # --- Final Training on Full Set & Evaluation on Test Set ---
            final_test_metrics = {}
            if test_loader is not None:
                logger.info(f"\n--- Final Training on full dataset for {exp_name} ---")
                final_exp_dir = os.path.join(RESULTS_DIR, exp_name)
                os.makedirs(final_exp_dir, exist_ok=True)
                final_weights_path = os.path.join(
                    final_exp_dir, "final_model_weights.pth"
                )

                if os.path.exists(final_weights_path):
                    logger.info(f"Found existing final model weights for {exp_name}.")
                    final_model = get_model(
                        model_name, num_classes=2, pretrained=False
                    ).to(device)
                    final_model.load_state_dict(torch.load(final_weights_path))
                else:
                    logger.info("Training final model on full training data...")
                    full_train_dataset = RAMCachedDataset(
                        all_images, all_labels, augment=True
                    )
                    full_train_loader = DataLoader(
                        full_train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                    )

                    final_model = get_model(
                        model_name, num_classes=2, pretrained=True
                    ).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(
                        final_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
                    )
                    scaler = torch.amp.GradScaler("cuda")

                    for epoch in range(EPOCHS):  # No early stopping for final model
                        final_model.train()
                        running_loss = 0.0
                        for inputs, labels in full_train_loader:
                            optimizer.zero_grad()
                            with torch.amp.autocast("cuda"):
                                outputs = final_model(inputs)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            running_loss += loss.item() * inputs.size(0)
                        epoch_loss = running_loss / len(full_train_loader.dataset)
                        logger.info(
                            f"Final Train Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f}"
                        )
                    torch.save(final_model.state_dict(), final_weights_path)
                    logger.info(f"Saved final model weights to {final_weights_path}")

                logger.info(f"--- Evaluating final model on Test Set ---")
                final_model.eval()
                y_true, y_pred, y_prob = [], [], []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = final_model(inputs)
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        preds = torch.argmax(outputs, dim=1)
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())
                        y_prob.extend(probs.cpu().numpy())

                final_test_metrics = calculate_metrics(y_true, y_pred, y_prob)
                logger.info(f"Test Accuracy: {final_test_metrics['accuracy']:.4f}")
                logger.info(f"Test AUC: {final_test_metrics['auc']:.4f}")

                plot_confusion_matrix(
                    final_test_metrics["cm"],
                    "Final Test Confusion Matrix",
                    os.path.join(final_exp_dir, "test_confusion_matrix.png"),
                )
                plot_roc_curve(
                    final_test_metrics["fpr"],
                    final_test_metrics["tpr"],
                    final_test_metrics["auc"],
                    "Final Test ROC Curve",
                    os.path.join(final_exp_dir, "test_roc_curve.png"),
                )

                del final_model
                gc.collect()
                torch.cuda.empty_cache()

            # --- Consolidate and Save Results ---
            result_row = {
                "Experiment": exp_name,
                "Model": model_name,
                "Dataset": dataset_name,
            }
            # Add CV results
            for key in avg_metrics:
                if key not in ["cm", "fpr", "tpr"]:
                    result_row[f"CV_Val_{key.capitalize()}_Mean"] = avg_metrics[key]
                    result_row[f"CV_Val_{key.capitalize()}_Std"] = std_metrics[key]
            # Add Test results
            for key in final_test_metrics:
                if key not in ["cm", "fpr", "tpr"]:
                    result_row[f"Test_{key.capitalize()}"] = final_test_metrics[key]

            all_results.append(result_row)
            pd.DataFrame(all_results).to_excel(summary_path, index=False)

        # Clean up dataset from VRAM
        del all_images, all_labels
        if test_images is not None:
            del test_images, test_labels
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"All experiments completed! Summary saved to {summary_path}")


if __name__ == "__main__":
    run_experiments()
