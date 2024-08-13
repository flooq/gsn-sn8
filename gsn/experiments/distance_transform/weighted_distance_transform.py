import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

class WeightedDistanceTransform(nn.Module):
    def __init__(self, weights=None, inverted: bool = False):
        super(WeightedDistanceTransform, self).__init__()
        self.weights = weights
        self.inverted = inverted

    def forward(self, flood_batch):
        return self._get_weighted_distance_transform(flood_batch)

    def _get_weighted_distance_transform(self, flood_batch):
        flood_np = flood_batch.cpu().numpy()
        distance_transforms = np.zeros_like(flood_np)
        for i in range(flood_np.shape[0]):  # iterate over batch
            for j in range(flood_np.shape[1]):  # iterate over the 4 masks
                distance_transform = distance_transform_edt(flood_np[i, j])
                if self.weights is not None:
                    weight = self.weights[j]
                    distance_transform *= weight
                # Normalize distance transform
                max_distance = distance_transform.max()
                if max_distance > 0:
                    if self.inverted:
                        distance_transform = max_distance - distance_transform
                    distance_transform /= max_distance
                distance_transforms[i, j] = distance_transform
        distance_transforms_tensor = torch.from_numpy(distance_transforms).to(flood_batch.device)
        return distance_transforms_tensor
