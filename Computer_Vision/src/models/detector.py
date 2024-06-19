import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import RoIPool


class Detector(nn.Module):
    def __init__(
        self,
        num_classes,
        pool_to_feature_vector,
        bb_last_layer_shape,
        sampling_scale,
        device=torch.device("cuda"),
    ):
        super().__init__()

        self._input_features = 7 * 7 * bb_last_layer_shape[1]
        self.device = device
        # Define network
        self._roi_pool = RoIPool(
            output_size=(7, 7), spatial_scale=1.0 / sampling_scale
        )
        self._pool_to_feature_vector = pool_to_feature_vector

        # Matching roi_head_classifier
        self._roi_head_classifier = nn.Sequential(
            nn.Linear(self._input_features, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        # Matching cls_loc and score
        self._regressor = nn.Linear(4096, (num_classes) * 4)
        self._classifier = nn.Linear(4096, num_classes)

        # Initialize weights
        self._classifier.weight.data.normal_(mean=0.0, std=0.01)
        self._classifier.bias.data.zero_()
        self._regressor.weight.data.normal_(mean=0.0, std=0.001)
        self._regressor.bias.data.zero_()

        # Move layers to the specified device
        self.to(device)

    def forward(self, feature_map, proposals):
        # Ensure proper shapes
        assert feature_map.shape[0] == 1, "Batch size must be 1"
        batch_size, num_proposals, _ = proposals.shape

        # Batch indices for proposals
        batch_idxs = torch.zeros(
            (batch_size, num_proposals, 1),
            dtype=torch.float32,
            device=self.device,
        )

        # Concatenate batch indices with proposals
        indexed_proposals = torch.cat([batch_idxs, proposals], dim=-1)

        # Reorder proposals to (batch_idx, x1, y1, x2, y2)
        indexed_proposals = indexed_proposals[:, :, [0, 2, 1, 4, 3]]

        # RoI pooling expects (N, 5) tensor of (batch_idx, x1, y1, x2, y2)
        indexed_proposals = indexed_proposals.view(-1, 5)

        # RoI pooling: (N, feature_map_channels, 7, 7)
        rois = self._roi_pool(feature_map, indexed_proposals)

        # Convert RoIs to feature vectors
        y = self._pool_to_feature_vector(rois)

        # Reshape for classifier input (batch_size * num_proposals, -1)
        y = y.view(batch_size * num_proposals, -1)

        # Forward propagate through classifier
        y = self._roi_head_classifier(y)

        # Predict classes and box deltas
        classes_raw = self._classifier(y)
        classes = F.sigmoid(classes_raw)
        box_deltas = self._regressor(y)

        # Reshape outputs to match proposals shape
        classes = classes.view(batch_size, num_proposals)
        box_deltas = box_deltas.view(batch_size, num_proposals, -1)

        return classes, box_deltas


def regression_loss(predicted_box_deltas, y_true):
    """
    Computes detector regression loss.

    Parameters
    ----------
    predicted_box_deltas : torch.Tensor
      RoI predicted box delta regressions, (N, 4*(num_classes-1)).
      The background class is excluded and only the non-background classes are
      included. Each set of box deltas is stored in parameterized form as
      (ty,tx, th, tw).
    y_true : torch.Tensor
      RoI box delta regression ground truth labels, (N, 2, 4*(num_classes-1)).
      These are stored as mask values (1 or 0) in (:,0,:) and regression
      parameters in (:,1,:). Note that it is important to mask off the
      predicted and ground truth values because they may be set to invalid
      values.

    Returns
    -------
    torch.Tensor
      Scalar loss.
    """
    epsilon = 1e-7
    scale_factor = 1.0
    sigma = 1.0
    sigma_squared = sigma * sigma

    # We want to unpack the regression targets and the mask of valid targets into
    # tensors each of the same shape as the predicted:
    #   (num_proposals, 4*(num_classes-1))
    # y_true has shape:
    #   (num_proposals, 2, 4*(num_classes-1))
    y_mask = y_true[:, 0, :]
    y_true_targets = y_true[:, 1, :]

    # Compute element-wise loss using robust L1 function for all 4 regression targets
    x = y_true_targets - predicted_box_deltas
    x_abs = torch.abs(x)
    is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
    R_negative_branch = 0.5 * x * x * sigma_squared
    R_positive_branch = x_abs - 0.5 / sigma_squared
    losses = (
        is_negative_branch * R_negative_branch
        + (1.0 - is_negative_branch) * R_positive_branch
    )

    # Normalize to number of proposals (e.g., 128). Although this may not be
    # what the paper does, it seems to work. Other implementations do this.
    # Using e.g., the number of positive proposals will cause the loss to
    # behave erratically because sometimes N will become very small.
    N = y_true.shape[0] + epsilon
    relevant_loss_terms = y_mask * losses
    return scale_factor * torch.sum(relevant_loss_terms) / N


def class_loss(predicted_classes, y_true):
    """
    Computes detector class loss.

    Parameters
    ----------
    predicted_classes : torch.Tensor
      RoI predicted classes as categorical vectors, (N, num_classes).
    y_true : torch.Tensor
      RoI class labels as categorical vectors, (N, num_classes).

    Returns
    -------
    torch.Tensor
      Scalar loss.
    """
    epsilon = 1e-7
    scale_factor = 1.0
    cross_entropy_per_row = -(
        y_true * torch.log(predicted_classes + epsilon)
    ).sum(dim=1)
    N = cross_entropy_per_row.shape[0] + epsilon
    cross_entropy = torch.sum(cross_entropy_per_row) / N
    return scale_factor * cross_entropy
