import copy
import gc
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
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
    "mobilenet_v2",
    "resnet18",
    "alexnet",
    "vgg16",
    "densenet121",
]  # Available: 'alexnet', 'vgg16', 'densenet121', 'resnet18', 'mobilenet_v2'
DATASETS_TO_RUN = [
    "original",
    "he",
    "clahe",
]  # Available: 'original', 'he', 'gamma', 'clahe'
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

        # VRAM Preloading
        logger.info("Preloading Train Set to VRAM...")
        all_images, all_labels = preload_to_device(df_train, logger, device)

        test_images, test_labels = None, None
        if not df_test.empty:
            logger.info("Preloading Test Set to VRAM...")
            test_images, test_labels = preload_to_device(df_test, logger, device)
            test_dataset = RAMCachedDataset(test_images, test_labels, augment=False)
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
            )
        else:
            test_loader = None

        # Splitting
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(all_images, all_labels.cpu().numpy()))

        train_img_split = all_images[train_idx]
        train_lbl_split = all_labels[train_idx]
        val_img_split = all_images[val_idx]
        val_lbl_split = all_labels[val_idx]

        # Oversampling - DISABLED due to already balanced dataset
        # train_img_split, train_lbl_split = balance_classes_on_gpu(train_img_split, train_lbl_split, device, logger)

        # DataLoaders
        train_dataset = RAMCachedDataset(train_img_split, train_lbl_split, augment=True)
        val_dataset = RAMCachedDataset(val_img_split, val_lbl_split, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Inner loop: Models
        for model_name in MODELS_TO_RUN:
            exp_name = f"{dataset_name}_{model_name}"
            exp_dir = os.path.join(RESULTS_DIR, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            logger.info(f"\n--- Starting Experiment: {exp_name} ---")

            weights_path = os.path.join(exp_dir, "best_weights.pth")

            # --- RESUME LOGIC ---
            if os.path.exists(weights_path):
                logger.info(
                    f"Found existing weights for {exp_name}. Skipping training!"
                )
                model = get_model(model_name, num_classes=2, pretrained=False).to(
                    device
                )
                model.load_state_dict(torch.load(weights_path))

                # Check if this experiment is already in the summary file
                already_in_results = any(
                    r.get("Experiment") == exp_name for r in all_results
                )
                if already_in_results:
                    logger.info(
                        f"Experiment {exp_name} is already in the Excel summary. Moving to next model..."
                    )
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
            else:
                model = get_model(model_name, num_classes=2, pretrained=True).to(device)
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

                train_losses, val_losses, val_accuracies = [], [], []

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
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item() * inputs.size(0)
                            preds = torch.argmax(outputs, dim=1)
                            correct += (preds == labels).sum().item()
                            total += labels.size(0)

                    val_loss = val_loss / len(val_loader.dataset)
                    val_acc = correct / total

                    train_losses.append(epoch_loss)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)

                    scheduler.step(val_loss)

                    logger.info(
                        f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_weights = copy.deepcopy(model.state_dict())

                if best_model_weights is not None:
                    model.load_state_dict(best_model_weights)
                    torch.save(best_model_weights, weights_path)

                # Save Training History
                history_df = pd.DataFrame(
                    {
                        "epoch": range(1, EPOCHS + 1),
                        "train_loss": train_losses,
                        "val_loss": val_losses,
                        "val_acc": val_accuracies,
                    }
                )
                csv_path = os.path.join(exp_dir, "training_history.csv")
                history_df.to_csv(csv_path, index=False)
                logger.info(f"Saved training history to {csv_path}")

                # Plot Learning Curves
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
                plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training & Validation Loss")
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(
                    range(1, EPOCHS + 1),
                    val_accuracies,
                    label="Val Accuracy",
                    color="green",
                )
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title("Validation Accuracy")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, "learning_curves.png"))
                plt.close()

            # Final Evaluation on Test Set (if available) or Val set
            eval_loader = test_loader if test_loader is not None else val_loader
            eval_name = "Test" if test_loader is not None else "Validation"

            model.eval()
            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for inputs, labels in eval_loader:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = torch.argmax(outputs, dim=1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_prob.extend(probs.cpu().numpy())

            metrics = calculate_metrics(y_true, y_pred, y_prob)

            logger.info(f"--- {eval_name} Results for {exp_name} ---")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"AUC:      {metrics['auc']:.4f}")

            plot_confusion_matrix(
                metrics["cm"],
                f"{eval_name} Confusion Matrix",
                os.path.join(exp_dir, "confusion_matrix.png"),
            )
            plot_roc_curve(
                metrics["fpr"],
                metrics["tpr"],
                metrics["auc"],
                f"{eval_name} ROC Curve",
                os.path.join(exp_dir, "roc_curve.png"),
            )

            all_results.append(
                {
                    "Experiment": exp_name,
                    "Model": model_name,
                    "Dataset": dataset_name,
                    f"{eval_name}_Accuracy": metrics["accuracy"],
                    f"{eval_name}_AUC": metrics["auc"],
                    f"{eval_name}_Sensitivity": metrics["sensitivity"],
                    f"{eval_name}_Specificity": metrics["specificity"],
                }
            )

            # Clean up model
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Clean up dataset from VRAM to make room for the next dataset
        del all_images, all_labels
        if test_images is not None:
            del test_images, test_labels
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(all_results).to_excel(
        os.path.join(RESULTS_DIR, "custom_experiments_summary.xlsx"), index=False
    )
    logger.info(
        f"All experiments completed! Summary saved to {RESULTS_DIR}\\custom_experiments_summary.xlsx"
    )


if __name__ == "__main__":
    run_experiments()
