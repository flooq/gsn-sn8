import torch

def get_train_metrics(loss, prefix: str = 'train'):
    metrics = {f"{prefix}/loss": loss}
    return metrics

def get_val_metrics(loss, pred, mask, class_loss, class_pred, class_mask, distance_transform_enabled, metrics_by_class: bool = False, prefix: str = 'val'):
    if distance_transform_enabled:
        assert pred.shape[1] == 9, f"Invalid flood prediction shape: {pred.shape}"
        pred_mask = torch.argmax(pred[:, :5, :, :], dim=1)
        gt_mask = torch.argmax(mask[:, :5, :, :], dim=1)
    else:
        assert pred.shape[1] == 5, f"Invalid flood prediction shape: {pred.shape}"
        pred_mask = torch.argmax(pred, dim=1)
        gt_mask = torch.argmax(mask, dim=1)

    pred_mask = pred_mask.to(mask.dtype)

    # Calculate true positives (TP), false positives (FP), and false negatives (FN)
    tp = torch.sum((pred_mask > 0) & (pred_mask == gt_mask)).item()
    fp = torch.sum((pred_mask > 0) & (pred_mask != gt_mask)).item()
    fn = torch.sum((pred_mask == 0) & (gt_mask > 0)).item()

    # Calculate precision, recall, and iou
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    f1 = _get_f1(precision,recall)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    metrics = {"iou": iou,
               "precision": precision,
               "recall": recall,
               "f1": f1,
               "dice": dice,
               "loss": loss}

    if class_loss is not None:
        metrics["class_loss"] = class_loss
        _class_pred = (torch.sigmoid(class_pred) >= 0.5).float()
        correct_predictions = torch.eq(_class_pred, class_mask).sum().item()
        class_accuracy = correct_predictions / class_pred.size(0)
        metrics["class_accuracy"] = class_accuracy

    if metrics_by_class:
        class_labels = {
            "non_flood_building": 1,
            "flood_building": 2,
            "non_flood_road": 3,
            "flood_road": 4
        }

        for class_name, class_label in class_labels.items():
            tp_class = torch.sum((pred_mask == class_label) & (gt_mask == class_label)).item()
            fp_class = torch.sum((pred_mask == class_label) & (gt_mask != class_label)).item()
            fn_class = torch.sum((pred_mask != class_label) & (gt_mask == class_label)).item()

            precision_class = tp_class / (tp_class + fp_class) if (tp_class + fp_class) > 0 else 1
            recall_class = tp_class / (tp_class + fn_class) if (tp_class + fn_class) > 0 else 1
            iou_class = tp_class / (tp_class + fp_class + fn_class) if (tp_class + fp_class + fn_class) > 0 else 1
            f1_class = _get_f1(precision_class,recall_class)
            dice_class = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

            metrics[f"{class_name}_precision"] = precision_class
            metrics[f"{class_name}_recall"] = recall_class
            metrics[f"{class_name}_iou"] = iou_class
            metrics[f"{class_name}_f1"] = f1_class
            metrics[f"{class_name}_dice"] = dice_class

    data = {}
    data.update({f"{prefix}/{metric}": metrics[metric] for metric in metrics})
    return data

def _get_f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0