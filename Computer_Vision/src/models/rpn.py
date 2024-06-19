from torch import nn
import torch
from . import anchors
import torchvision


class RPN(nn.Module):
    def __init__(
        self,
        feature_map_size,  # [1,512,50,50]
        allow_edge_proposals: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        # Constants
        self._allow_edge_proposals = allow_edge_proposals
        self.device = device

        # Layers
        self.num_anchors = 9
        channels = feature_map_size[1]
        self._rpn_conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
        )
        self._rpn_class = nn.Conv2d(
            in_channels=channels,
            out_channels=self.num_anchors,
            # can be required to make num_anchors*2(if we use softmax loss)
            kernel_size=(1, 1),
            stride=1,
            padding="same",
        )
        self._rpn_boxes = nn.Conv2d(
            in_channels=channels,
            out_channels=self.num_anchors * 4,
            kernel_size=(1, 1),
            stride=1,
            padding="same",
        )

        # Weight Initialization
        self._rpn_conv1.weight.data.normal_(mean=0, std=0.01)
        self._rpn_conv1.bias.data.zero_()
        self._rpn_class.weight.data.normal_(mean=0, std=0.01)
        self._rpn_class.bias.data.zero_()
        self._rpn_boxes.weight.data.normal_(mean=0, std=0.01)
        self._rpn_boxes.bias.data.zero_()

        # Move layers to the specified device
        self.to(device)

    def forward(
        self,
        feature_map,
        image_shape,
        all_anchor_bboxes,
        valid_anchor_bbox_indices,
        max_proposals_pre_nms,
        max_proposals_post_nms,
    ):
        # Forward pass through the RPN network
        x = nn.functional.relu(self._rpn_conv1(feature_map))

        objectness_score_map = torch.sigmoid(self._rpn_class(x))
        box_deltas_map = self._rpn_boxes(x)

        # Reshape box_deltas_map and objectness_score_map
        #   box_deltas_map->(batch_size,height,width,num_anchors*4)
        # [1, 36(9*4), 50, 50] => [1, 50,50,36]
        box_deltas_map = box_deltas_map.permute(0, 2, 3, 1).contiguous()

        # objectness_score_map -> (batch_size, height,width, \
        # num_anchors*num_classes (can be 1 or 2 depending on \
        # loss function, as loss is sigmoid we use 1))
        # [1, 9(9*1), 50, 50] => [1, 50,50,9]
        objectness_score_map = objectness_score_map.permute(
            0, 2, 3, 1
        ).contiguous()

        # Based on self._allow_edge_proposals, value,
        # we either return all the anchor bboxes or
        # only the valid anchors (those which lie in bounds of the img)
        anchor_map, objectness_score_map, box_deltas_map = self._extract_valid(
            anchor_map=all_anchor_bboxes,
            valid_anchor_indices=valid_anchor_bbox_indices,
            box_deltas_map=box_deltas_map,
            objectness_score_map=objectness_score_map,
        )
        # Here anchor_map is of shape (feature_map_h*feature_map_w*9,4)
        # Here objectness_score_map is of
        # shape (1,feature_map_h*feature_map_w*9)
        # Here box_deltas_map is of shape (1,feature_map_h*feature_map_w*9,4)

        # Convert deltas to box corners
        proposals = anchors.t_convert_deltas_to_boxes_torch(
            anchor_map=anchor_map, box_deltas_map=box_deltas_map
        )
        # proposals -> (1,feature_map_h*feature_map_w*9,4)
        # Clip proposals to image boundaries
        proposals[:, slice(0, 4, 2)] = torch.clamp(
            proposals[:, slice(0, 4, 2)], 0, image_shape[0]
        )
        proposals[:, slice(1, 4, 2)] = torch.clamp(
            proposals[:, slice(1, 4, 2)], 0, image_shape[1]
        )

        # Filter proposals with small height or width
        patch_size = min(image_shape[1:]) / min(self.feature_map_size[1:])
        hs = proposals[:, :, 2] - proposals[:, :, 0]
        ws = proposals[:, :, 3] - proposals[:, :, 1]
        valid_pred_indices = (hs >= patch_size) & (ws >= patch_size)

        # After indexing the batch dimension is lost
        proposals = proposals[valid_pred_indices]

        # Get the batch dimension back
        proposals = proposals.unsqueeze(0)
        proposal_scores = objectness_score_map[valid_pred_indices]

        # Sort the proposals based on the objectness score in descending order
        # Keep only the top-N scores. Note that we do not care whether the
        # proposals were labeled as objects (score > 0.5) and peform a simple
        # ranking among all of them.
        # Restricting them has a strong adverse impact
        # on training performance.
        # Sort proposals based on objectness scores
        proposal_scores_ordered_indices = torch.argsort(
            proposal_scores, descending=True
        )
        pre_nms_ordered_indices = proposal_scores_ordered_indices[
            :max_proposals_pre_nms
        ]
        pre_nms_proposal_scores = proposal_scores[pre_nms_ordered_indices]
        pre_nms_proposals = proposals[:, pre_nms_ordered_indices]

        # Perform NMS
        # NMS is th process in which we remove/merge extremely highly
        # overlapping bounding boxes. In this process the goal is to retain
        # the bboxes, which are unique and doesn't overlap much.
        # The threshold is kept at 0.7, and it means the minimum overlapping
        # area required to merge overlapping bboxes

        # selected_indices = anchors.nms(
        #     boxes=pre_nms_proposals,
        #     scores=pre_nms_proposal_scores,
        #     iou_threshold=0.7,
        # )

        # Much more efficient
        selected_indices = torchvision.ops.nms(
            boxes=pre_nms_proposals[0],
            scores=pre_nms_proposal_scores,
            iou_threshold=0.7,
        )

        # Select top proposals after NMS
        idxs = selected_indices[:max_proposals_post_nms]
        post_nms_proposals = pre_nms_proposals[:, idxs]

        # Return the proposals in form of tensors and are backprop compatible
        # objectness_score_map:(1,feature_map_h*feature_map_w*9)
        # box_deltas_map:(1,feature_map_h*feature_map_w*9,4)
        # post_nms_proposals:(1, max_proposals_post_nms, 4) (y1x1y2x2)
        return objectness_score_map, box_deltas_map, post_nms_proposals

    # def forward(
    #     self,
    #     feature_map,
    #     image_shape,
    #     all_anchor_bboxes,
    #     valid_anchor_bbox_indices,
    #     max_proposals_pre_nms,
    #     max_proposals_post_nms,
    # ):
    #     # Do a forward pass of the feature_map through the network
    #     x = nn.functional.relu(self._rpn_conv1(feature_map))

    #     objectness_score_map = torch.sigmoid(self._rpn_class(x))
    #     box_deltas_map = self._rpn_boxes(x)

    #     #   box_deltas_map->(batch_size,height,width,num_anchors*4)
    #     # [1, 36(9*4), 50, 50] => [1, 50,50,36]
    #     box_deltas_map = box_deltas_map.permute(0, 2, 3, 1).contiguous()

    #     # objectness_score_map -> (batch_size, height,width, \
    #     # num_anchors*num_classes (can be 1 or 2 depending on \
    #     # loss function, as loss is sigmoid we use 1))
    #     # [1, 9(9*1), 50, 50] => [1, 50,50,9]
    #     objectness_score_map = objectness_score_map.permute(
    #         0, 2, 3, 1
    #     ).contiguous()

    #     # Based on self._allow_edge_proposals, value,
    #     # we either return all the anchor bboxes or
    #     # only the valid anchors (those which lie in bounds of the img)
    #     anchor_map, objectness_score_map, box_deltas_map = self._extract_valid(
    #         anchor_map=all_anchor_bboxes,
    #         valid_anchor_indices=valid_anchor_bbox_indices,
    #         box_deltas_map=box_deltas_map,
    #         objectness_score_map=objectness_score_map,
    #     )
    #     # Here anchor_map is of shape (feature_map_h*feature_map_w*9,4)
    #     # Here objectness_score_map is of
    #     # shape (feature_map_h*feature_map_w*9,)
    #     # Here box_deltas_map is of shape (feature_map_h*feature_map_w*9,4)

    #     # Detach from graph to avoid backprop. This is to resolve the memory
    #     # leak involving t_convert_deltas_to_boxes(). Ultimately the numerical
    #     # results are not affected. Proposals returns from this function are
    #     # supposed to be constant and are fed into the detector stage.

    #     box_deltas_map = box_deltas_map.detach()

    #     # Convert to NumPy arrays if necessary
    #     if isinstance(anchor_map, torch.Tensor):
    #         anchor_map = anchor_map.cpu().numpy()
    #     if isinstance(box_deltas_map, torch.Tensor):
    #         box_deltas_map = box_deltas_map.cpu().numpy().squeeze(0)
    #     if isinstance(objectness_score_map, torch.Tensor):
    #         objectness_score_map = (
    #             objectness_score_map.detach().cpu().numpy().squeeze(0)
    #         )

    #     # Convert deltas to box corners
    #     # Proposals are of shape (feature_map_h*feature_map_w*9,4(y1x1y2x2))
    #     proposals = anchors.t_convert_deltas_to_boxes(
    #         anchor_map=anchor_map, box_deltas_map=box_deltas_map
    #     )

    #     # Clip the proposals to the image boundaries
    #     # Clip the y coordinates to 0 to y_max
    #     proposals[:, slice(0, 4, 2)] = np.clip(
    #         proposals[:, slice(0, 4, 2)], 0, image_shape[0]
    #     )
    #     # Clip the x coordinates to 0 to 800
    #     proposals[:, slice(1, 4, 2)] = np.clip(
    #         proposals[:, slice(1, 4, 2)], 0, image_shape[1]
    #     )

    #     patch_size = min(image_shape[1:]) / min(self.feature_map_size[1:])

    #     # Remove the predictions with either height or width < threshold
    #     hs = proposals[:, 2] - proposals[:, 0]
    #     ws = proposals[:, 3] - proposals[:, 1]
    #     valid_pred_indices = np.where((hs >= patch_size) & (ws >= patch_size))[
    #         0
    #     ]
    #     proposals = proposals[valid_pred_indices]
    #     proposal_scores = objectness_score_map[valid_pred_indices]

    #     # Sort the proposals based on the objectness score in descending order
    #     # Keep only the top-N scores. Note that we do not care whether the
    #     # proposals were labeled as objects (score > 0.5) and peform a simple
    #     # ranking among all of them.
    #     # Restricting them has a strong adverse impact
    #     # on training performance.
    #     proposal_scores_ordered_indices = np.argsort(proposal_scores)[::-1]

    #     # Take the max_proposals_pre_nms top proposals
    #     pre_nms_ordered_indices = proposal_scores_ordered_indices[
    #         :max_proposals_pre_nms
    #     ]
    #     pre_nms_proposal_scores = proposal_scores[pre_nms_ordered_indices]

    #     pre_nms_proposals = proposals[pre_nms_ordered_indices]

    #     # Perform NMS
    #     # NMS is the process in which we remove/merge extremely highly
    #     # overlapping bounding boxes. In this process the goal is to retain
    #     # the bboxes, which are unique and doesn't overlap much.
    #     # The threshold is kept at 0.7, and it means the minimum overlapping
    #     # area required to merge overlapping bboxes
    #     selected_indices = anchors.nms(
    #         boxes=pre_nms_proposals,
    #         scores=pre_nms_proposal_scores,
    #         iou_threshold=0.7,
    #     )

    #     # Take the max_proposals_post_nms top proposals
    #     idxs = selected_indices[:max_proposals_post_nms]
    #     post_nms_proposals = pre_nms_proposals[idxs]

    #     # Convert back to torch tensors and move to GPU
    #     post_nms_proposals = torch.tensor(
    #         post_nms_proposals, device=self.device
    #     )
    #     proposal_scores = torch.tensor(proposal_scores, device=self.device)
    #     objectness_score_map = torch.tensor(
    #         objectness_score_map, device=self.device
    #     )
    #     box_deltas_map = torch.tensor(box_deltas_map, device=self.device)
    #     # Return the proposals
    #     return objectness_score_map, box_deltas_map, post_nms_proposals

    def _extract_valid(
        self,
        anchor_map,
        valid_anchor_indices,
        box_deltas_map,
        objectness_score_map,
    ):
        assert objectness_score_map.shape[0] == 1
        # Only support for Batch size of 1

        # Reformat the box_deltas_map
        # [1,50,50,36] => [1,50*50*9,4]
        box_deltas_map = box_deltas_map.view(1, -1, 4)

        # Reformat the objectness_score_map
        # [1,50,50,9] => [1,50*50*9]
        objectness_score_map = objectness_score_map.view(1, -1)

        if self._allow_edge_proposals:
            return (
                anchor_map.unsqueeze(dim=0),
                objectness_score_map,
                box_deltas_map,
            )
        else:
            return (
                anchor_map[valid_anchor_indices].squeeze(dim=0),
                objectness_score_map[valid_anchor_indices],
                box_deltas_map[valid_anchor_indices],
            )


def class_loss(
    predicted_scores: torch.Tensor, objectness_score_target: torch.Tensor
) -> torch.Tensor:
    """
    Compute binary cross-entropy classification loss.

    @param torch.Tensor predicted_scores: Predicted objectness scores.
    @param torch.Tensor objectness_score_target: Target objectness scores.

    @return torch.Tensor: Scalar loss.
    """
    epsilon = 1e-7

    # Filter out ignored targets (-1)
    valid_indices = objectness_score_target != -1
    predicted_scores = predicted_scores[valid_indices]
    objectness_score_target = objectness_score_target[valid_indices]

    # Flatten the tensors to perform element-wise operations
    predicted_scores = predicted_scores.view(-1)
    objectness_score_target = objectness_score_target.view(-1)

    # Compute binary cross-entropy loss
    loss_all_elements = torch.nn.functional.binary_cross_entropy(
        predicted_scores,
        objectness_score_target,
        reduction="sum",
    )

    # Normalize the total loss by total positive and negative
    # samples being used in the training
    N_cls = torch.count_nonzero(valid_indices) + epsilon
    return (loss_all_elements) / N_cls


def regression_loss(
    predicted_box_deltas: torch.Tensor,
    box_deltas_target: torch.Tensor,
    objectness_score_target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute smooth L1 regression loss.

    @param torch.Tensor predicted_box_deltas: Predicted box deltas.
    @param torch.Tensor box_deltas_target: Target box deltas.
    @param torch.Tensor objectness_score_target: Target objectness scores.

    @return torch.Tensor: Scalar loss.
    """
    epsilon = 1e-7
    scale_factor = 1.0
    # hyper-parameter that controls magnitude of regression loss and is chosen
    # to make regression term comparable to class term
    sigma = 3.0  # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
    sigma_squared = sigma * sigma

    # Consider only positive samples
    valid_indices = objectness_score_target > 0
    predicted_box_deltas = predicted_box_deltas[valid_indices]
    box_deltas_target = box_deltas_target[valid_indices]

    # Flatten the tensors to perform element-wise operations
    predicted_box_deltas = predicted_box_deltas.view(-1, 4)
    box_deltas_target = box_deltas_target.view(-1, 4)

    # They used L1 loss instead of L2 loss because the values of predicted
    # regression head of RPN are not bounded. Regression loss is also applied
    # to the bounding boxes which have positive label
    # Compute smooth L1 loss
    x = box_deltas_target - predicted_box_deltas
    x_abs = torch.abs(x)
    is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
    R_negative_branch = 0.5 * x * x * sigma_squared
    R_positive_branch = x_abs - 0.5 / sigma_squared
    loss_all_anchors = (
        is_negative_branch * R_negative_branch
        + (1.0 - is_negative_branch) * R_positive_branch
    )

    # Normalize the total loss by total positive and negative
    # samples being used in the training
    N_cls = torch.count_nonzero(valid_indices) + epsilon

    # Zero out the ones which should not have been included
    relevant_loss_terms = loss_all_anchors
    return scale_factor * torch.sum(relevant_loss_terms) / N_cls
