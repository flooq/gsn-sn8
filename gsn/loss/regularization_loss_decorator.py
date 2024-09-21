import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class RegularizationLossDecorator(nn.Module):

    def __init__(self,
                 base_loss,
                 gradient_alpha,
                 shape_alpha,
                 corner_alpha,
                 channels_to_apply=('non-flood-building', 'flood-building')):
        super(RegularizationLossDecorator, self).__init__()

        self.mappings = {
            'non-flood-building': 1,
            'flood-building': 2,
            'non-flood-road': 3,
            'flood-road': 4
        }

        self.base_loss = base_loss
        self.gradient_alpha = gradient_alpha
        self.shape_alpha = shape_alpha
        self.corner_alpha = corner_alpha
        self.channels_to_apply = []
        for channel in channels_to_apply:
            if channel not in self.mappings:
                raise ValueError(f"Channel '{channel}' is not in the mappings dictionary.")
            self.channels_to_apply.append(self.mappings[channel])

    def forward(self, pred, target):
        base_loss_value = self.base_loss(pred, target)
        grad_loss = self.gradient_alpha * self.gradient_regularization_loss(pred, self.channels_to_apply)
        shape_loss = self.shape_alpha * self.shape_regularization_loss(pred, self.channels_to_apply)
        #corner_loss = self.corner_alpha * self.corner_regularization_loss(pred, self.channels_to_apply)
        loss = base_loss_value + grad_loss + shape_loss # + corner_loss
        return loss

    def gradient_regularization_loss(self, predicted_mask, channels):
        # Sobel filters for detecting edges
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).float().unsqueeze(0).unsqueeze(0).cuda()
        sobel_y = sobel_x.transpose(2, 3)

        # Apply Sobel filters to each channel separately
        grad_x_list = []
        grad_y_list = []
        for c in channels:  # Loop over the number of channels
            channel = predicted_mask[:, c:c+1, :, :]  # Extract the i-th channel

            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)

            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)

        # Stack gradients along the channel dimension
        gradient_x = torch.cat(grad_x_list, dim=1)
        gradient_y = torch.cat(grad_y_list, dim=1)

        # Calculate the gradient magnitude
        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1e-8)

        # Return the mean gradient magnitude as the loss
        return torch.mean(gradient_magnitude)


    def shape_regularization_loss(self, predicted_mask, channels):
        # Lists to store vertical and horizontal differences for each channel
        vertical_diff_list = []
        horizontal_diff_list = []

        # Iterate over the specified channels
        for c in channels:
            channel = predicted_mask[:, c:c+1, :, :]  # Extract the specific channel

            # Calculate vertical and horizontal differences for this channel
            vertical_diff = torch.abs(channel[:, :, :-1, :] - channel[:, :, 1:, :])
            horizontal_diff = torch.abs(channel[:, :, :, :-1] - channel[:, :, :, 1:])

            # Append the differences to the lists
            vertical_diff_list.append(vertical_diff)
            horizontal_diff_list.append(horizontal_diff)

        # Stack differences along the channel dimension
        vertical_diff = torch.cat(vertical_diff_list, dim=1)
        horizontal_diff = torch.cat(horizontal_diff_list, dim=1)

        # Calculate the mean of the differences
        vertical_loss = torch.mean(vertical_diff)
        horizontal_loss = torch.mean(horizontal_diff)

        # Return the sum of vertical and horizontal shape regularization losses
        return vertical_loss + horizontal_loss + 1e-8


    def corner_regularization_loss(self, predicted_mask, channels):
        total_non_right_angles = 0

        for c in channels:
            # Extract the specific channel
            channel = predicted_mask[:, c:c+1, :, :]  # Extract the c-th channel

            # Convert channel to a format suitable for corner detection
            channel = channel.squeeze(1)  # Remove the channel dimension for corner detection

            # Harris corner detection for the channel
            corners = self.harris_corner_detection(channel)

            # Calculate angles at corners
            angles = self.calculate_angles_at_corners(corners)

            # Count non-right angles
            non_right_angles = [angle for angle in angles if not self.is_right_angle(angle)]
            total_non_right_angles += len(non_right_angles)

        return total_non_right_angles

    def harris_corner_detection(self, image):
        # Convert the image to a numpy array (if needed)
        image_np = image.cpu().numpy().squeeze()  # Assuming image is a PyTorch tensor
        image_np = (image_np * 255).astype(np.uint8)  # Assuming image is normalized [0, 1]

        # Apply Harris corner detection
        corners = cv2.cornerHarris(image_np, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)  # Dilate corner image to enhance corner points

        return torch.from_numpy(corners).float()  # Convert back to PyTorch tensor

    def calculate_angles_at_corners(self, corner_image):
        corners = self.extract_corners(corner_image)

        if len(corners) == 0:
            return []

        # Assuming a simple case where we just get orientations for all corners
        image = corner_image  # Replace this with your actual image if needed
        orientations = self.calculate_edge_orientations(image, corners)

        # Calculate angles between adjacent orientations
        angles = []
        for i in range(len(orientations)):
            for j in range(i + 1, len(orientations)):
                angle = abs(orientations[i] - orientations[j])
                angle = min(angle, 360 - angle)  # Get the smaller angle
                angles.append(angle)

        return angles

    def calculate_edge_orientations(self, image, corners):
        # Convert image to numpy if it's a tensor
        image_np = image.cpu().numpy()

        # Calculate gradients
        grad_x = np.gradient(image_np, axis=1)
        grad_y = np.gradient(image_np, axis=0)

        orientations = []

        for (y, x) in corners:
            grad_x_val = grad_x[y, x]
            grad_y_val = grad_y[y, x]

            orientation = np.arctan2(grad_y_val, grad_x_val) * 180 / np.pi  # Convert to degrees
            orientations.append(orientation)

        return orientations


    def extract_corners(self, corner_image):
        # Convert tensor to numpy array if needed
        corners_np = corner_image.cpu().numpy()

        # Apply threshold to find strong corners
        threshold = 0.01 * corners_np.max()
        strong_corners = (corners_np > threshold)

        # Find corner coordinates
        coordinates = np.argwhere(strong_corners)
        return coordinates


    def is_right_angle(self, angle, threshold=0.1):
        # Check if the angle is close to 90 degrees (right angle)
        return abs(angle - 90) < threshold


