"""
Training loop for the skin lesion classifier.

Class imbalance handling (per team decision): BOTH techniques combined.
  1. WeightedRandomSampler (in dataset.get_train_loader) balances which
     images the model SEES each epoch.
  2. Weighted CrossEntropyLoss (via dataset.get_loss_class_weights) uses a
     softened sqrt(inverse-frequency) weighting so the two techniques
     don't stack into an over-correction.

"""

import time
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import mlflow
import mlflow.pytorch

import config
from model import build_model
from dataset import get_train_loader, get_val_loader, get_loss_class_weights


def set_seed(seed=config.SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, criterion, optimizer=None):
    """
    If optimizer is provided -> training mode (weights update).
    If optimizer is None     -> eval mode (no gradient, no update).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    torch.set_grad_enabled(True)  # restore default

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def train():
    set_seed()

    print(f"Device: {config.DEVICE}")

    train_loader = get_train_loader(use_weighted_sampler=True)
    val_loader = get_val_loader()

    class_weights = get_loss_class_weights().to(config.DEVICE)
    print("Loss class weights:", class_weights.cpu().numpy())

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.LR_SCHEDULER_PATIENCE
    )

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    mlflow.set_experiment("skin_lesion_classifier")

    with mlflow.start_run(run_name=f"{config.MODEL_NAME}_run"):

        mlflow.log_params({
            "model_name": config.MODEL_NAME,
            "num_classes": config.NUM_CLASSES,
            "embedding_dim": config.EMBEDDING_DIM,
            "pretrained": config.PRETRAINED,
            "freeze_backbone": config.FREEZE_BACKBONE,
            "image_size": config.IMAGE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "lr_scheduler_patience": config.LR_SCHEDULER_PATIENCE,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "optimizer": "AdamW",
            "seed": config.SEED,
        })

        for epoch in range(1, config.EPOCHS + 1):
            start = time.time()

            train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer=None)

            scheduler.step(val_loss)
            elapsed = time.time() - start

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"Epoch {epoch}/{config.EPOCHS} "
                f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                f"| {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                torch.save(best_model_state, config.BEST_MODEL_PATH)
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        model.load_state_dict(best_model_state)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(config.BEST_MODEL_PATH)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    train()
