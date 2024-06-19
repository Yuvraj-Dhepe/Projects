from dataclasses import dataclass
import torch
import numpy as np
from typing import Dict, Tuple
from torch import nn
from . import rpn, detector
from . import utils as model_utils
from . import anchors


class FasterRcnn(nn.Module):
    @dataclass
    class Loss:
        rpn_cls_loss: torch.Tensor
        rpn_reg_loss: torch.Tensor
        detector_cls_loss: torch.Tensor
        detector_reg_loss: torch.Tensor
        total: torch.Tensor

    def __init__(
        self,
        num_classes,
        backbone,
        feature_map_size,  # If used in RPN, it should be [1,512,50,50]
        # else for anchor generation it should be [50,50]
        rpn_minibatch_size=256,
        proposal_batch_size=128,
        allow_edge_proposals=True,
        pool_to_feature_vector=None,
        device: torch.device = torch.device("cuda"),  # Added device argument
    ):
        super().__init__()
        self._num_classes = num_classes
        self._rpn_minibatch_size = rpn_minibatch_size
        self._proposal_batch_size = proposal_batch_size
        self.device = device  # Store the device
        # self._detector_box_delta_means = [0.0, 0.0, 0.0, 0.0]
        # self._detector_box_delta_stds = [0.1, 0.1, 0.2, 0.2]

        # Adding the backbone & Feature map size
        self._stage1_feature_extractor = backbone.to(device)
        self.feature_map_size = feature_map_size

        self._stage2_rpn = rpn.RPN(
            feature_map_size=feature_map_size,
            allow_edge_proposals=allow_edge_proposals,
            device=device,
        )

        self._stage3_detector = detector.Detector(
            num_classes=num_classes,
            pool_to_feature_vector=pool_to_feature_vector,
            bb_last_layer_shape=feature_map_size,
            sampling_scale=800 // feature_map_size[-1],
        )

    # Image_data
    def forward(
        self, image_data, all_anchor_bboxes=None, anchor_valid_indices=None
    ):
        assert image_data.shape[0] == 1, "Only single image batch supported"
        image_shape = image_data.shape[-2:]

        # Anchor maps can be pre-computed and passed, else they are computed
        # here on fly

        if all_anchor_bboxes is None or anchor_valid_indices is None:
            # all_anchor_bboxes is of shape (H*W*A, 4),
            # valid_anchor_bbox_indices is of shape (len(Valid_indices),)
            all_anchor_bboxes, valid_anchor_bbox_indices = (
                anchors.generate_anchor_maps(
                    feature_map_size=(
                        self.feature_map_size[-2],
                        self.feature_map_size[-1],
                    ),
                    image_size=image_shape,
                    ratios=[0.5, 1, 2],
                    scales=[8, 16, 32],
                )
            )
        all_anchor_bboxes = torch.from_numpy(all_anchor_bboxes).to(self.device)
        valid_anchor_bbox_indices = torch.from_numpy(
            valid_anchor_bbox_indices
        ).to(self.device)

        # TODO: Check image data to be of shape [1,C,H,W], and also should be
        # TODO: sent to device, and converted to floats.
        # TODO: The feature extractor should also be sent to the device.
        # Run each stage
        feature_map = self._stage1_feature_extractor(image_data)
        # Feature map will be of size (1, 512, 50, 50) for VGG16

        # Objectness score is of shape: [1,f_map_h*f_map_w*9], bbox_deltas_map
        # is of size [1,9*f_map_h*f_map_w,4], rpn_proposals are of size
        # [1,2000,4 (y1x1y2x2)]
        objectness_scores, bbox_deltas_map, rpn_proposals = self._stage2_rpn(
            feature_map=feature_map,
            image_shape=image_shape,
            all_anchor_bboxes=all_anchor_bboxes,
            valid_anchor_bbox_indices=valid_anchor_bbox_indices,
            max_proposals_pre_nms=12000,
            max_proposals_post_nms=2000,
        )

        classes, box_deltas = self._stage3_detector(
            feature_map=feature_map, proposals=rpn_proposals
        )

        return rpn_proposals, classes, box_deltas

    @model_utils.no_grad
    def predict(
        self,
        image_data,
        score_threshold,
        anchor_map=None,
        anchor_valid_indices=None,
    ):
        self.eval()
        assert image_data.shape[0] == 1, "Batch Size of 1 is supported"

        image_shape = image_data.shape[-2:]
        # Forward Inference
        rpn_proposals, classes, box_deltas = self(
            image_data=image_data,
            anchor_map=anchor_map,
            anchor_valid_indices=anchor_valid_indices,
        )

        proposals = rpn_proposals.cpu().numpy()
        classes = classes.cpu().numpy()
        box_deltas = box_deltas.cpu().numpy()

        # Convert proposals from yxyx to cxcywh
        proposal_anchors = anchors.bbox_xyxy_to_cxcywh(proposals)

        # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
        boxes_and_scores_by_class_idx = {}
        for class_idx in range(
            1, classes.shape[1]
        ):  # skip class 0 (background)
            # Get the box deltas (ty, tx, th, tw) corresponding to this class,
            # for all proposals
            box_delta_idx = (class_idx - 1) * 4
            box_delta_params = box_deltas[
                :, (box_delta_idx + 0) : (box_delta_idx + 4)
            ]  # (N, 4)
            proposal_boxes_this_class = (
                anchors.t_convert_deltas_to_boxes_torch(
                    anchor_map=proposal_anchors,
                    box_deltas_map=box_delta_params,
                )
            )

            # Clip the proposals to the image boundaries
            # Clip the y coordinates to 0 to y_max
            proposal_boxes_this_class[:, slice(0, 4, 2)] = np.clip(
                proposal_boxes_this_class[:, slice(0, 4, 2)],
                0,
                image_shape[0],
            )
            # Clip the x coordinates to 0 to 800
            proposal_boxes_this_class[:, slice(1, 4, 2)] = np.clip(
                proposal_boxes_this_class[:, slice(1, 4, 2)],
                0,
                image_shape[1],
            )

            # Get the scores for this class. The class scores are returned in
            # normalized categorical form. Each row corresponds to a class.
            scores_this_class = classes[:, class_idx]

            # Keep only those scoring high enough
            sufficiently_scoring_idxs = np.where(
                scores_this_class > score_threshold
            )[0]
            proposal_boxes_this_class = proposal_boxes_this_class[
                sufficiently_scoring_idxs
            ]
            scores_this_class = scores_this_class[sufficiently_scoring_idxs]
            boxes_and_scores_by_class_idx[class_idx] = (
                proposal_boxes_this_class,
                scores_this_class,
            )

            # Perform NMS per class
            scored_boxes_by_class_idx = {}
            for class_idx, (
                boxes,
                scores,
            ) in boxes_and_scores_by_class_idx.items():
                idxs = (
                    anchors.nms(boxes=boxes, scores=scores, iou_threshold=0.3)
                    .cpu()
                    .numpy()
                )
                boxes = boxes[idxs]
                scores = np.expand_dims(scores[idxs], axis=0)  # (N,) -> (N,1)
                scored_boxes = np.hstack(
                    [boxes, scores.T]
                )  # (N,5), with each row: (y1, x1, y2, x2, score)
                scored_boxes_by_class_idx[class_idx] = scored_boxes

        return scored_boxes_by_class_idx

    def train_step(
        self,
        optimizer,
        image_data,
        anchor_boxes,
        valid_anchor_bbox_indices,
        gt_rpn_map,
        gt_rpn_labels,
        gt_boxes,
    ):
        self.train()

        # Clear the accumulated gradient
        optimizer.zero_grad()

        # For now we only work with batch_size of 1
        assert image_data.shape[0] == 1, "Batch size must be 1"
        assert gt_rpn_map.shape[0] == 1, "Batch size must be 1"
        image_shape = image_data.shape[1:]

        # Stage1: Extract features
        feature_map = self._stage1_feature_extractor(image_data=image_data)

        # Stage 2: Generate object proposals using RPN
        objectness_scores, bbox_deltas_map, rpn_proposals = self._stage2_rpn(
            feature_map=feature_map,
            image_shape=image_shape,
            all_anchor_bboxes=anchor_boxes,
            valid_anchor_bbox_indices=valid_anchor_bbox_indices,
            max_proposals_pre_nms=12000,
            max_proposals_post_nms=2000,
        )

        # Sample random mini-batch of anchors (for RPN training):
        gt_rpn_minibatch_target = self._sample_rpn_minibatch(
            anchor_target_labels=gt_rpn_labels,
            rpn_minibatch_size=self._rpn_minibatch_size,
            positive_sample_ratio=0.5,
        )
        # Assign labels to proposals and take random sample (for detector
        # training)
        proposals, gt_classes, gt_box_deltas = self._label_proposals(
            proposals=rpn_proposals,
            gt_boxes=gt_boxes,
            min_background_iou_threshold=0.0,
            min_object_iou_threshold=0.5,
        )
        proposals, gt_classes, gt_box_deltas = self._sample_proposals(
            proposals=proposals,
            gt_classes=gt_classes,
            gt_box_deltas=gt_box_deltas,
            max_proposals=self._proposal_batch_size,
            positive_fraction=0.25,
        )
        # Make sure RoI proposals and ground truths are detached from
        # computational graph so that gradients are not propagated through
        # them. They are treated as constant inputs into the detector stage.
        proposals = proposals.detach()
        gt_classes = gt_classes.detach()
        gt_box_deltas = gt_box_deltas.detach()

        # Stage 3: Detector
        detector_classes, detector_box_deltas = self._stage3_detector(
            feature_map=feature_map, proposals=rpn_proposals
        )

        # Compute losses
        rpn_class_loss = rpn.class_loss(
            predicted_scores=objectness_scores, y_true=gt_rpn_minibatch_target
        )
        rpn_regression_loss = rpn.regression_loss(
            predicted_box_deltas=bbox_deltas_map,
            y_true=gt_box_deltas.unsqueeze(dim = 0),
            objectness_score_target=gt_rpn_minibatch_target
        )
        detector_class_loss = detector.class_loss(
            predicted_classes=detector_classes, y_true=gt_classes
        )
        detector_regression_loss = detector.regression_loss(
            predicted_box_deltas=detector_box_deltas, y_true=gt_box_deltas
        )

        total_loss = (
            rpn_class_loss
            + rpn_regression_loss
            + detector_class_loss
            + detector_regression_loss
        )
        loss = FasterRcnn.Loss(
            rpn_class=rpn_class_loss.detach().cpu().item(),
            rpn_regression=rpn_regression_loss.detach().cpu().item(),
            detector_class=detector_class_loss.detach().cpu().item(),
            detector_regression=detector_regression_loss.detach().cpu().item(),
            total=total_loss.detach().cpu().item(),
        )

        # Backprop
        total_loss.backward()

        # Optimizer step
        optimizer.step()

        # Return losses and data useful for computing statistics
        return loss

    def _sample_rpn_minibatch(
    anchor_target_labels: torch.Tensor = None,
    rpn_minibatch_size: int = 256,
    positive_sample_ratio: float = 0.5,
) -> torch.Tensor:
        """
        Randomly sample positive and negative anchors for RPN training and
        generate target scores.

        @param torch.Tensor anchor_target_labels: Tensor containing target
        labels for anchors.
        @param int rpn_minibatch_size: Total size of the RPN minibatch.
        @param float positive_sample_ratio: Ratio of positive samples
        in the minibatch.

        :return torch.Tensor: Target objectness scores for the
        sampled minibatch,
        where the sampled targets have 0(-ve sample) or 1 (+ve sample) and -1 for
        samples to be ignored.
        """
        # Find indices of positive and negative samples
        positive_indices = torch.nonzero(anchor_target_labels == 1).squeeze().to("cuda")
        negative_indices = torch.nonzero(anchor_target_labels == 0).squeeze().to("cuda")

        # Calculate number of positive and negative samples to select
        num_positive_samples = min(
            int(rpn_minibatch_size * positive_sample_ratio),
            len(positive_indices),
        )  # Up to half of the samples should be positive if possible
        num_negative_samples = rpn_minibatch_size - num_positive_samples

        # Randomly sample positive and negative indices
        if num_positive_samples > 0:
            sampled_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:num_positive_samples]]
        else:
            sampled_positive_indices = torch.tensor([], dtype=torch.long)

        if num_negative_samples > 0:
            sampled_negative_indices = negative_indices[torch.randperm(len(negative_indices))[:num_negative_samples]]
        else:
            sampled_negative_indices = torch.tensor([], dtype=torch.long)

        # Concatenate sampled indices
        sampled_indices = torch.cat((sampled_positive_indices, sampled_negative_indices))

        # Generate target objectness scores initialized to -1
        objectness_score_target = -torch.ones_like(anchor_target_labels, dtype=torch.float).to("cuda")

        # We only make the sampled indices targets as valid ones.
        objectness_score_target[sampled_indices] = anchor_target_labels[sampled_indices].float()
        # Shape is [1,22500]
        return objectness_score_target.unsqueeze(dim = 0)

    def _label_proposals(
        proposals: torch.Tensor,
        gt_boxes: Dict[str, torch.Tensor],
        min_background_iou_threshold: float,
        min_object_iou_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Label proposals as positive, negative, or ignored based on their
        IoU withground truth boxes.

        This function assigns labels to the proposals generated by the RPN
        stage based on their Intersection over Union (IoU)
        with ground truth boxes. It also computes the regression targets for
        bounding box refinement.

        The function ensures that the proposals include some positive examples
        by adding ground truth boxes. It computes the IoU between each
        proposal and ground truth box, assigns labels based on the IoU
        thresholds, and calculates
        the regression targets for bounding box refinement.

        @param torch.Tensor proposals: Tensor of shape (N, 4) containing
        proposals in xyxy format.
        @param Dict[str, torch.Tensor] gt_boxes: Dictionary containing ground
        truth boxes with keys 'boxes' (tensor of shape (M, 4)) and
        'labels' (tensor of shape (M,)).
        @param float min_background_iou_threshold: Minimum IoU threshold with
        ground truth boxes below which proposals are ignored entirely.Proposals
        with an IoU threshold in the range
        [min_background_iou_threshold, min_object_iou_threshold) are labeled
        as background.
        @param float min_object_iou_threshold: Minimum IoU threshold for a
        proposal to be labeled as an object.

        :raises AssertionError: If min_background_iou_threshold is not less
        than min_object_iou_threshold.

        :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns a
        tuple containing the following:
            - proposals (torch.Tensor): Tensor of shape (N, 4) with the
            proposals
            in xyxy format.
            - gt_classes (torch.Tensor): One-hot encoded class labels of shape
            (N, num_classes) for each proposal.
            - box_delta_targets (torch.Tensor): Box delta regression targets of
            shape (N, 4) for each proposal.
        """
        # Ensure that the background threshold is less than the object
        # threshold
        assert (
        min_background_iou_threshold < min_object_iou_threshold
    ), "Object threshold must be greater than background threshold"

        # Extract boxes and labels from gt_boxes dictionary
        proposals = proposals.squeeze()
        gt_box_corners = torch.tensor(
        gt_boxes["boxes"], dtype=torch.float32
        ,device='cuda')
        # Convert the list of string labels to a list of integers
        gt_boxes_labels_int = [int(label) for label in gt_boxes['labels']]

        # Convert the list of integers to a PyTorch tensor
        gt_box_class_idxs = torch.tensor(gt_boxes_labels_int, dtype=torch.long, device="cuda")

        # Add ground truth boxes to the proposals to ensure some positive examples
        proposals = torch.vstack([proposals, gt_box_corners])

        # Compute IoU between each proposal and each ground truth box
        ious = anchors.calculate_iou(proposals, gt_box_corners)

        # Find the highest IoU for each proposal and the corresponding ground truth box index
        best_ious = torch.max(ious, dim=1).values
        box_idxs = torch.argmax(ious, dim=1)
        gt_box_class_idxs = gt_box_class_idxs[box_idxs]
        gt_box_corners = gt_box_corners[box_idxs]

        # Remove proposals with IoU less than the background threshold
        idxs = torch.where(best_ious >= min_background_iou_threshold)[0]
        proposals = proposals[idxs]
        best_ious = best_ious[idxs]
        gt_box_class_idxs = gt_box_class_idxs[idxs]
        gt_box_corners = gt_box_corners[idxs]

        # Label proposals with IoU less than the object threshold as background
        gt_box_class_idxs[best_ious < min_object_iou_threshold] = 0

        # One-hot encode the class labels for the proposals
        num_proposals = proposals.shape[0]
        num_classes = torch.max(gt_box_class_idxs) + 1  # Assumes class indices start at 0

        gt_classes = torch.zeros((num_proposals, num_classes), dtype=torch.float32, device="cuda")
        gt_classes[torch.arange(num_proposals), gt_box_class_idxs] = 1.0

        # Calculate center points and side lengths for proposals and ground truth boxes
        proposal_centers = 0.5 * (proposals[:, 0:2] + proposals[:, 2:4])
        proposal_sides = proposals[:, 2:4] - proposals[:, 0:2]
        gt_box_centers = 0.5 * (gt_box_corners[:, 0:2] + gt_box_corners[:, 2:4])
        gt_box_sides = gt_box_corners[:, 2:4] - gt_box_corners[:, 0:2]

        # Prepare the data in the required format for t_convert_boxes_to_deltas
        gt_data = {
            "height": gt_box_sides[:, 0],
            "width": gt_box_sides[:, 1],
            "ctr_x": gt_box_centers[:, 1],
            "ctr_y": gt_box_centers[:, 0],
        }
        anchor_data = {
            "height": proposal_sides[:, 0],
            "width": proposal_sides[:, 1],
            "ctr_x": proposal_centers[:, 1],
            "ctr_y": proposal_centers[:, 0],
        }

        # Calculate box delta regression targets using t_convert_boxes_to_deltas
        box_delta_targets = anchors.t_convert_boxes_to_deltas_torch(gt_data, anchor_data)
        box_delta_targets = torch.tensor(box_delta_targets, dtype=torch.float32).cuda()

        return proposals, gt_classes, box_delta_targets

    def _sample_proposals(
        proposals: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_box_deltas: torch.Tensor,
        max_proposals: int,
        positive_fraction: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample positive and negative proposals from the input proposals.

        @param torch.Tensor proposals: The input proposals.
        @param torch.Tensor gt_classes: The ground truth classes corresponding to the proposals.
        @param torch.Tensor gt_box_deltas: The ground truth box deltas corresponding to the proposals.
        @param int max_proposals: The maximum number of proposals to sample.
        @param float positive_fraction: The fraction of positive samples to select.

        @return Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the sampled proposals,
            their corresponding ground truth classes, and ground truth box deltas.
        """

        # Check if there are no proposals to sample
        if max_proposals <= 0:
            return proposals, gt_classes, gt_box_deltas

        # Get indices of positive and negative samples
        positive_indices = torch.where(gt_classes > 0)[0]
        negative_indices = torch.where(gt_classes == 0)[0]

        # Calculate the number of positive and negative samples to select
        num_positive_proposals = len(positive_indices)
        num_negative_proposals = len(negative_indices)
        num_samples = min(max_proposals, len(gt_classes))
        num_positive_samples = min(
            round(num_samples * positive_fraction), num_positive_proposals
        )
        num_negative_samples = min(
            num_samples - num_positive_samples, num_negative_proposals
        )

        # Return empty tensors if there are no positive or negative samples
        if num_positive_samples <= 0 or num_negative_samples <= 0:
            return (
                proposals[:0],
                gt_classes[:0],
                gt_box_deltas[:0],
            )

        # Randomly sample positive and negative samples
        positive_sample_indices = positive_indices[
            torch.randperm(len(positive_indices))[:num_positive_samples]
        ]
        negative_sample_indices = negative_indices[
            torch.randperm(len(negative_indices))[:num_negative_samples]
        ]
        indices = torch.cat([positive_sample_indices, negative_sample_indices])

        # Return the sampled proposals, classes, and box deltas
        return proposals[indices], gt_classes[indices], gt_box_deltas[indices]