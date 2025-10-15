import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    confusion_matrix
)

def eval_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return total_loss/len(loader), correct/total


def segmentation_metrics(y_true, y_pred, num_classes=3):
    # Flatten arrays
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Pixel accuracy
    pixel_acc = accuracy_score(y_true, y_pred)

    # IoU (Jaccard Index)
    per_class_iou = jaccard_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    mean_iou = jaccard_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))

    # Dice (F1-score)
    per_class_dice = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    mean_dice = f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))

    # Precision & Recall
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
