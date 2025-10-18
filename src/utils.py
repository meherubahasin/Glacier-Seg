
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    jaccard_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)

# ======================================================
# 🧮 SEGMENTATION METRICS
# ======================================================
def segmentation_metrics(y_true, y_pred, num_classes=4):
    """
    Compute common segmentation metrics:
    - pixel accuracy
    - IoU (per-class + mean)
    - Dice (per-class + mean)
    - Precision, Recall, Confusion matrix
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pixel_acc = accuracy_score(y_true, y_pred)

    per_class_iou = jaccard_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    mean_iou = np.nanmean(per_class_iou)

    per_class_dice = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    mean_dice = np.nanmean(per_class_dice)

    precision = precision_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))
    recall = recall_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))

    return {
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }

# ======================================================
# 🕹️ EARLY STOPPING
# ======================================================
class EarlyStopping:
    """
    Stop training when validation metric doesn't improve after `patience` epochs.
    mode='max' for metrics like mIoU; 'min' for loss.
    """
    def __init__(self, patience=10, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, current):
        if self.best is None:
            self.best = current
            return False

        improved = (current > self.best + self.min_delta) if self.mode == 'max' else (current < self.best - self.min_delta)
        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# ======================================================
# ⚙️ OPTIMIZER / SCHEDULER BUILDERS
# ======================================================
def build_optimizer(model, opt_cfg=None):
    """
    Construct optimizer from dict config.
    Example:
        opt_cfg = {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4}
    """
    opt_cfg = opt_cfg or {"name": "AdamW", "lr": 3e-6, "weight_decay": 1e-4}
    name = opt_cfg.get("name", "AdamW").lower()
    lr = opt_cfg.get("lr", 3e-6)
    wd = opt_cfg.get("weight_decay", 1e-4)
    momentum = opt_cfg.get("momentum", 0.9)

    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, sch_cfg=None):
    """
    Default: ReduceLROnPlateau (mode='max', monitors val mIoU)
    """
    sch_cfg = sch_cfg or {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 5, "threshold": 1e-4, "min_lr": 1e-7}
    name = sch_cfg.get("name", "ReduceLROnPlateau").lower()

    if name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=sch_cfg.get("factor", 0.5),
            patience=sch_cfg.get("patience", 5),
            threshold=sch_cfg.get("threshold", 1e-4),
            cooldown=sch_cfg.get("cooldown", 0),
            min_lr=sch_cfg.get("min_lr", 1e-7)
        )
    elif name == "cosineannealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sch_cfg.get("T_max", 50), eta_min=sch_cfg.get("min_lr", 1e-7)
        )
    elif name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=sch_cfg.get("step_size", 30), gamma=sch_cfg.get("gamma", 0.1)
        )
    else:
        return None

# ======================================================
# 🔁 TRAINING / VALIDATION EPOCHS
# ======================================================
def train_epoch(loader, model, criterion, optimizer, device, binary=True):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        if binary:
            y = y.float().to(device)
        else:
            y = y.to(device).long()

        optimizer.zero_grad()
        outputs = model(x)

        if binary:
            loss = criterion(outputs, y)
        else:
            loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(loader, model, criterion, device, num_classes=4, binary=True):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x = x.to(device)
            if binary:
                y = y.float().to(device)
            else:
                y = y.to(device).long()

            outputs = model(x)

            if binary:
                loss = criterion(outputs, y)
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
                y_true = y.long().squeeze(1)
            else:
                loss = criterion(outputs, y)
                preds = torch.argmax(outputs, dim=1)
                y_true = y

            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_true.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    if binary:
        metrics = segmentation_metrics(y_true, y_pred, num_classes=2)
    else:
        metrics = segmentation_metrics(y_true, y_pred, num_classes=num_classes)

    return total_loss / len(loader), metrics


# ======================================================
# 🪵 OPTIONAL: CSV LOGGING
# ======================================================
import csv
import os

def log_metrics(epoch, train_loss, val_loss, val_metrics, csv_path="training_log.csv"):
    """
    Append per-epoch stats to CSV.
    Logs mean + per-class IoU/Dice + precision/recall.
    """
    # Extract core metrics
    mean_iou = val_metrics.get("mean_iou", 0.0)
    mean_dice = val_metrics.get("mean_dice", 0.0)
    precision = val_metrics.get("precision", 0.0)
    recall = val_metrics.get("recall", 0.0)

    # Per-class metrics
    per_class_iou = val_metrics.get("per_class_iou", [])
    per_class_dice = val_metrics.get("per_class_dice", [])

    # Dynamically build header for per-class metrics
    header = [
        "epoch", "train_loss", "val_loss",
        "mean_iou", "mean_dice", "precision", "recall"
    ]
    for i in range(len(per_class_iou)):
        header.append(f"class_{i}_iou")
    for i in range(len(per_class_dice)):
        header.append(f"class_{i}_dice")

    # Build data row
    data = [
        epoch, train_loss, val_loss,
        mean_iou, mean_dice, precision, recall
    ]
    data.extend(per_class_iou)
    data.extend(per_class_dice)

    # Check if we need to write header
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
