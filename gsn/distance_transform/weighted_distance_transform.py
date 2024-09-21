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
                max_distance = distance_transform.max()
                if max_distance > 0:
                    if self.inverted:
                        distance_transform = np.where(flood_np[i, j] > 0, max_distance - distance_transform, 0)
                    distance_transform /= max_distance
                # change to [0, weight]
                if self.weights is not None:
                    weight = self.weights[j]
                    distance_transform *= weight
                distance_transforms[i, j] = distance_transform
        distance_transforms_tensor = torch.from_numpy(distance_transforms).to(flood_batch.device)
        return distance_transforms_tensor

if __name__ == "__main__":
    flood_batch = torch.tensor([[
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 1, 1, 1]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 1, 1, 1]],

        [[0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]
    ]], dtype=torch.float32)

    wdt = WeightedDistanceTransform(weights=[2, 1, 1, 1], inverted=True)
    result = wdt(flood_batch)

    print("Input tensor:")
    print(flood_batch)
    print("\nDistance Transform Result:")
    print(result)