import torch


soft_dice_loss_weight = 0.25  # road loss
focal_loss_weight = 0.75  # road loss
road_loss_weight = 0.5
building_loss_weight = 0.5

def soft_dice_loss(outputs, targets, per_image=False):
    batch_size = outputs.size()
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def focal(outputs, targets, gamma=2,  ignore_index=255):
    outputs = outputs.contiguous()
    targets = targets.contiguous()
    eps = 1e-8
    non_ignored = targets.view(-1) != ignore_index
    targets = targets.view(-1)[non_ignored].float()
    outputs = outputs.contiguous().view(-1)[non_ignored]
    outputs = torch.clamp(outputs, eps, 1. - eps)
    targets = torch.clamp(targets, eps, 1. - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return (-(1. - pt) ** gamma * torch.log(pt)).mean()


bceloss = torch.nn.BCEWithLogitsLoss()
celoss = torch.nn.CrossEntropyLoss()
