import torch


def get_flood_segmentation_metrics(pred, gt_mask):
    assert pred.shape[1] == 5, f"invalid flood prediction shape: {pred.shape}"
    pred_mask = torch.argmax(pred, dim=1)
    tp = torch.sum(torch.logical_and(pred_mask > 0, pred_mask == gt_mask)).item()
    fp = torch.sum(torch.logical_and(pred_mask > 0, pred_mask != gt_mask)).item()
    fn = torch.sum(torch.logical_and(pred_mask == 0, gt_mask > 0)).item()
    precision = tp / (tp + fp) if tp + fp > 0 else 1
    recall = tp / (tp + fn) if tp + fn > 0 else 1
    accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1
    return {"accuracy": accuracy, "precision": precision, "recall": recall}
