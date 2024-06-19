from typing import List, Tuple, Union
import torch
import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two sets of bounding boxes.

    @param box1: Array or Tensor of shape (m, 4) containing m bounding boxes in XYXY format.
    @param box2: Array or Tensor of shape (n, 4) containing n bounding boxes in XYXY format.

    :return: IoU values for each pair of boxes from box1 and box2 of shape (m, n).
    """
    if isinstance(box1, np.ndarray) and isinstance(box2, np.ndarray):
        # Numpy implementation
        box1 = np.expand_dims(box1, axis=1)  # (m, 1, 4)
        box2 = np.expand_dims(box2, axis=0)  # (1, n, 4)

        xA = np.maximum(box1[:, :, 0], box2[:, :, 0])
        yA = np.maximum(box1[:, :, 1], box2[:, :, 1])
        xB = np.minimum(box1[:, :, 2], box2[:, :, 2])
        yB = np.minimum(box1[:, :, 3], box2[:, :, 3])

        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

        box1Area = (box1[:, :, 2] - box1[:, :, 0]) * (
            box1[:, :, 3] - box1[:, :, 1]
        )
        box2Area = (box2[:, :, 2] - box2[:, :, 0]) * (
            box2[:, :, 3] - box2[:, :, 1]
        )

        iou = interArea / (box1Area + box2Area - interArea)
        return iou.squeeze()

    elif isinstance(box1, torch.Tensor) and isinstance(box2, torch.Tensor):
        # PyTorch implementation
        box1 = box1.unsqueeze(1)  # (m, 1, 4)
        box2 = box2.unsqueeze(0)  # (1, n, 4)

        xA = torch.maximum(box1[:, :, 0], box2[:, :, 0])
        yA = torch.maximum(box1[:, :, 1], box2[:, :, 1])
        xB = torch.minimum(box1[:, :, 2], box2[:, :, 2])
        yB = torch.minimum(box1[:, :, 3], box2[:, :, 3])

        interArea = torch.maximum(
            torch.tensor(0.0, device=box1.device), xB - xA
        ) * torch.maximum(torch.tensor(0.0, device=box1.device), yB - yA)

        box1Area = (box1[:, :, 2] - box1[:, :, 0]) * (
            box1[:, :, 3] - box1[:, :, 1]
        )
        box2Area = (box2[:, :, 2] - box2[:, :, 0]) * (
            box2[:, :, 3] - box2[:, :, 1]
        )

        iou = interArea / (box1Area + box2Area - interArea)
        return iou.squeeze()

    else:
        raise TypeError(
            "Unsupported type. Supported types are numpy.ndarray and torch.Tensor."
        )


def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on the bounding boxes and their corresponding scores.

    @param boxes: Array or Tensor of shape (1, N, 4) containing N bounding boxes in XYXY format.
    @param scores: Array or Tensor of shape (N,) containing the scores for each bounding box.
    @param iou_threshold: IoU threshold for NMS.

    :return: Indices of the selected boxes after NMS.
    """
    if isinstance(boxes, np.ndarray) and isinstance(scores, np.ndarray):
        # Numpy implementation
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = []

        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            selected_indices.append(current_idx)

            if len(sorted_indices) == 1:
                break

            ious = calculate_iou(
                np.expand_dims(boxes[0, current_idx], axis=0),
                boxes[0, sorted_indices[1:]],
            )

            keep_indices = np.where(ious <= iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]

        return selected_indices

    elif isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor):
        # PyTorch implementation
        sorted_indices = torch.argsort(scores, descending=True)
        selected_indices = []

        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            selected_indices.append(current_idx.item())

            if len(sorted_indices) == 1:
                break

            ious = calculate_iou(
                boxes[0, current_idx].unsqueeze(0),
                boxes[0, sorted_indices[1:]],
            )

            keep_indices = torch.nonzero(ious <= iou_threshold).squeeze()
            sorted_indices = sorted_indices[keep_indices + 1]
        selected_indices = np.array(selected_indices)
        return selected_indices

    else:
        raise TypeError(
            "Unsupported type. Supported types are numpy.ndarray and torch.Tensor."
        )


def generate_anchors(
    feature_map_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    ratios: List[float],
    scales: List[int],
) -> np.ndarray:
    """
    Generate anchor boxes for object detection given the feature
    map size, sub-sample size, aspect ratios, and anchor scales.
    Generate a grid of anchor boxes (bounding boxes) centered on each point of
    a feature map. Each center point has a set of anchors with different
    aspect ratios and scales.
    @param Tuple[int,int] feature_map_size: The size of the feature map
    (e.g., (50,50).
    @param Tuple[int, int] patch_size: The stride of the sliding window or
    feature map stride (e.g., (16, 16)).
    @param List[float] ratios: Aspect ratios of the anchors
    (e.g., [0.5, 1, 2]).
    @param List[int] scales: Anchor scales (e.g., [8, 16, 32]).

    :raises ValueError: If feature_map_size or patch_size is not
    a positive integer.
    :raises TypeError: If ratios or scales is not a list.
    :return np.ndarray: numpy array of shape (num_anchors, 4) containing the
    anchor boxes, where each anchor box is defined by
    [y_min, x_min, y_max, x_max].
    """
    feature_map_h, feature_map_w = feature_map_size
    patch_h, patch_w = patch_size

    if (
        not isinstance(feature_map_size, Union[Tuple, list])
        or not all(isinstance(i, int) for i in feature_map_size)
        or not all(i > 0 for i in feature_map_size)
    ):
        raise ValueError("feature_map_size must be a positive integer")
    if (
        not isinstance(patch_size, Union[tuple, List])
        or not all(isinstance(i, Union[int, np.int64]) for i in patch_size)
        or not all(i > 0 for i in patch_size)
    ):
        raise ValueError("patch_size must be a tuple of positive integers")
    if not isinstance(ratios, list) or not all(
        isinstance(r, (int, float)) for r in ratios
    ):
        raise TypeError("ratios must be a list of numbers")
    if not isinstance(scales, list) or not all(
        isinstance(s, (int, float)) for s in scales
    ):
        raise TypeError("scales must be a list of numbers")

    # Generate center points for the feature map
    center_x = np.arange(patch_w, (feature_map_w + 1) * patch_w, patch_w)
    center_y = np.arange(patch_h, (feature_map_h + 1) * patch_h, patch_h)

    # Create a grid of center points
    ctr = np.array(
        [
            (x - patch_w // 2, y - patch_h // 2)
            for y in center_y
            for x in center_x
        ]
    )

    # Calculate the total number of anchors
    num_anchors = feature_map_h * feature_map_w * len(ratios) * len(scales)

    # Initialize the anchors array
    anchors = np.zeros((num_anchors, 4))

    index = 0
    # Iterate over each center point
    for c in ctr:
        center_y, center_x = c
        # Iterate over each combination of ratio and scale
        for i in range(len(ratios)):
            for j in range(len(scales)):
                # Calculate the height and width of the anchor
                h = patch_h * scales[j] * np.sqrt(ratios[i])
                w = patch_w * scales[j] * np.sqrt(1.0 / ratios[i])

                # Calculate the coordinates of the anchor box
                anchors[index, 0] = center_y - h / 2.0  # y_min
                anchors[index, 1] = center_x - w / 2.0  # x_min
                anchors[index, 2] = center_y + h / 2.0  # y_max
                anchors[index, 3] = center_x + w / 2.0  # x_max

                # Move to the next anchor index
                index += 1

    return anchors


def get_anchors_for_coordinate(
    anchors: np.ndarray, coordinate: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Get anchors corresponding to a given coordinate.

    @param anchors: An array of anchor boxes.
    @param coordinate: A tuple containing (x, y) coordinate.
    :return: A list of anchor boxes that have their center at
    the given coordinate.
    """
    x, y = coordinate
    result_anchors = []

    for anchor in anchors:
        center_x = (anchor[1] + anchor[3]) / 2
        center_y = (anchor[0] + anchor[2]) / 2
        if center_x == x and center_y == y:
            result_anchors.append(anchor)

    return result_anchors


def generate_anchor_maps(
    image_size,
    feature_map_size,
    ratios,
    scales,
):
    image_h, image_w = image_size
    # 800/50 = 16 for Vgg16 and inp of (800, 800)
    patch_height = image_h // feature_map_size[-2]
    patch_width = image_w // feature_map_size[-1]
    anchor_bboxes = generate_anchors(
        feature_map_size=feature_map_size,
        patch_size=(patch_height, patch_width),
        ratios=ratios,
        scales=scales,
    )

    valid_anchor_bbox_indices = np.where(
        (anchor_bboxes[:, 0] >= 0)
        & (anchor_bboxes[:, 1] >= 0)
        & (anchor_bboxes[:, 2] <= 800)
        & (anchor_bboxes[:, 3] <= 800)
    )[0]

    # Return the anchor boxes in y1x1y2x2 format and with shape
    # (feature_map_height*feature_map_width*num_anchors, 4)
    return anchor_bboxes.astype(np.float32), valid_anchor_bbox_indices.astype(
        np.float32
    )


def generate_rpn_map(
    anchor_map,
    valid_anchor_indices,
    gt_bboxes,
    object_iou_threshold=0.7,
    background_iou_threshold=0.3,
):
   # Ensure inputs are tensors
    if not isinstance(anchor_map, torch.Tensor):
        anchor_map = torch.tensor(anchor_map, dtype=torch.float32)
    if not isinstance(valid_anchor_indices, torch.Tensor):
        valid_anchor_indices = torch.tensor(valid_anchor_indices, dtype=torch.long)
    if not isinstance(gt_bboxes, torch.Tensor):
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)

    valid_anchor_map = anchor_map[valid_anchor_indices]
    valid_labels = torch.full((len(valid_anchor_map),), -1, dtype=torch.int8).to("cuda")

    valid_bbox_ious = calculate_iou(valid_anchor_map, gt_bboxes)

    # Case-I: Finding All Anchors Having Highest Intersection with GT
    gt_argmax_ious = valid_bbox_ious.argmax(dim=0)
    gt_max_ious = valid_bbox_ious[gt_argmax_ious, torch.arange(valid_bbox_ious.size(1))]

    gt_max_iou_anchors_index = (valid_bbox_ious == gt_max_ious).nonzero(as_tuple=True)[0]

    # Case-II: Finding GT for which Anchor yields a high value
    anchor_argmax_ious = valid_bbox_ious.argmax(dim=1)
    anchor_max_ious = valid_bbox_ious[torch.arange(len(valid_anchor_map)), anchor_argmax_ious]

    # Assign negative label (0) to all the anchor bboxes which have max_iou less than negative threshold
    valid_labels[anchor_max_ious < background_iou_threshold] = 0

    # Assign positive label (1) to all the anchor boxes which have Highest IoU overlap with gt-bbox
    valid_labels[gt_max_iou_anchors_index] = 1
    valid_labels[anchor_max_ious >= object_iou_threshold] = 1

    max_iou_bboxes = gt_bboxes[anchor_argmax_ious]

    # Convert the bboxes to xywh
    valid_bbox_xywh = bbox_xyxy_to_cxcywh(valid_anchor_map)
    max_iou_bbox_xywh = bbox_xyxy_to_cxcywh(max_iou_bboxes)

    # Translate the valid bboxes
    valid_anchor_target_bboxes = t_convert_boxes_to_deltas_torch(
        gt_data=valid_bbox_xywh, anchor_data=max_iou_bbox_xywh
    )

    # Generate the labels first
    anchor_target_labels = torch.full((len(anchor_map),), -1, dtype=torch.int8).to("cuda")
    anchor_target_labels[valid_anchor_indices] = valid_labels

    # Final Anchor BBoxes that will act as a gt and we will sample from these
    anchor_target_bboxes = torch.zeros(anchor_map.shape, dtype=torch.float32).to("cuda")
    anchor_target_bboxes[valid_anchor_indices, :] = valid_anchor_target_bboxes

    return anchor_target_bboxes, anchor_target_labels


def t_convert_boxes_to_deltas(gt_data, anchor_data):
    height = gt_data["height"]
    width = gt_data["width"]
    ctr_x, ctr_y = gt_data["ctr_x"], gt_data["ctr_y"]

    base_height = anchor_data["height"]
    base_width = anchor_data["width"]
    base_ctr_x, base_ctr_y = anchor_data["ctr_x"], anchor_data["ctr_y"]

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()

    return anchor_locs

def t_convert_boxes_to_deltas_torch(gt_data, anchor_data):
    height = gt_data["height"]
    width = gt_data["width"]
    ctr_x, ctr_y = gt_data["ctr_x"], gt_data["ctr_y"]

    base_height = anchor_data["height"]
    base_width = anchor_data["width"]
    base_ctr_x, base_ctr_y = anchor_data["ctr_x"], anchor_data["ctr_y"]

    eps = torch.finfo(height.dtype).eps
    height = torch.maximum(height, torch.tensor(eps))
    width = torch.maximum(width, torch.tensor(eps))

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = torch.log(base_height / height)
    dw = torch.log(base_width / width)

    anchor_locs = torch.stack((dy, dx, dh, dw), dim=1)

    return anchor_locs


def t_convert_deltas_to_boxes(anchor_map, box_deltas_map):
    anc_height = anchor_map[:, 2] - anchor_map[:, 0]
    anc_width = anchor_map[:, 3] - anchor_map[:, 1]
    anc_ctr_y = anchor_map[:, 0] + 0.5 * anc_height
    anc_ctr_x = anchor_map[:, 1] + 0.5 * anc_width

    dy = box_deltas_map[:, 0::4]
    dx = box_deltas_map[:, 1::4]
    dh = box_deltas_map[:, 2::4]
    dw = box_deltas_map[:, 3::4]
    ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]

    cxcywh_bboxes = {
        "ctr_x": ctr_x,
        "ctr_y": ctr_y,
        "height": h,
        "width": w,
    }

    # (N, 4(y1,x1,y2,x2)) shaped bboxes will be returned
    yxyx_bboxes = bbox_xywh_to_xyxy(cxcywh_bboxes, shape=box_deltas_map.shape)
    return yxyx_bboxes


def bbox_xyxy_to_cxcywh(bboxes):
    height = bboxes[:, 2] - bboxes[:, 0]
    width = bboxes[:, 3] - bboxes[:, 1]
    ctr_y = bboxes[:, 0] + 0.5 * height
    ctr_x = bboxes[:, 1] + 0.5 * width

    bbox_data = {
        "ctr_x": ctr_x,
        "ctr_y": ctr_y,
        "width": width,
        "height": height,
    }
    return bbox_data


# def bbox_xywh_to_xyxy(bboxes, shape):
#     h, w = bboxes["height"], bboxes["width"]
#     # Converting the ctr_x, ctr_y, h, w to y1, x1, y2, x2
#     boxes = np.zeros(shape)
#     ctr_y, ctr_x = bboxes["ctr_y"], bboxes["ctr_x"]

#     boxes[:, 0::4] = ctr_y - 0.5 * h
#     boxes[:, 1::4] = ctr_x - 0.5 * w
#     boxes[:, 2::4] = ctr_y + 0.5 * h
#     boxes[:, 3::4] = ctr_x + 0.5 * w

#     return boxes


# def t_convert_deltas_to_boxes_torch(anchor_map, box_deltas_map):
#     anc_height = anchor_map[:, 2] - anchor_map[:, 0]
#     anc_width = anchor_map[:, 3] - anchor_map[:, 1]
#     anc_ctr_y = anchor_map[:, 0] + 0.5 * anc_height
#     anc_ctr_x = anchor_map[:, 1] + 0.5 * anc_width

#     # Reshape box_deltas_map to have the same number of anchors as anchor_map
#     # box_deltas_map = box_deltas_map.view(anchor_map.size(0), -1, 4)

#     dy = box_deltas_map[:, :, 0]
#     dx = box_deltas_map[:, :, 1]
#     dh = box_deltas_map[:, :, 2]
#     dw = box_deltas_map[:, :, 3]

#     ctr_y = dy * anc_height[:, None] + anc_ctr_y[:, None]
#     ctr_x = dx * anc_width[:, None] + anc_ctr_x[:, None]
#     h = torch.exp(dh) * anc_height[:, None]
#     w = torch.exp(dw) * anc_width[:, None]

#     cxcywh_bboxes = {
#         "ctr_x": ctr_x,
#         "ctr_y": ctr_y,
#         "height": h,
#         "width": w,
#     }

#     # (N, 4(y1,x1,y2,x2)) shaped bboxes will be returned
#     yxyx_bboxes = bbox_xywh_to_xyxy_torch(
#         cxcywh_bboxes, shape=box_deltas_map.shape
#     )
#     return yxyx_bboxes


# def bbox_xywh_to_xyxy_torch(bboxes, shape):
#     h, w = bboxes["height"], bboxes["width"]
#     ctr_y, ctr_x = bboxes["ctr_y"], bboxes["ctr_x"]

#     # Initialize the boxes tensor with the appropriate shape
#     boxes = torch.zeros(shape, device=h.device)

#     # Convert the center coordinates to (y1, x1, y2, x2)
#     boxes[:, 0::4] = ctr_y - 0.5 * h
#     boxes[:, 1::4] = ctr_x - 0.5 * w
#     boxes[:, 2::4] = ctr_y + 0.5 * h
#     boxes[:, 3::4] = ctr_x + 0.5 * w

#     return boxes


def t_convert_deltas_to_boxes_torch(anchor_map, box_deltas_map):
    """
    Convert anchor boxes to bounding boxes using the provided deltas.

    @param anchor_map: Tensor of shape (22500, 4) containing anchor boxes.
    @param box_deltas_map: Tensor of shape (1, 22500, 4) containing deltas.

    :return: Tensor of shape (1, 22500, 4) containing the converted bounding
    boxes.
    """
    # Calculate anchor dimensions and center points
    anc_height = anchor_map[:, :, 2] - anchor_map[:, :, 0]
    anc_width = anchor_map[:, :, 3] - anchor_map[:, :, 1]
    anc_ctr_y = anchor_map[:, :, 0] + 0.5 * anc_height
    anc_ctr_x = anchor_map[:, :, 1] + 0.5 * anc_width

    # Extract deltas
    dy = box_deltas_map[:, :, 0]
    dx = box_deltas_map[:, :, 1]
    dh = box_deltas_map[:, :, 2]
    dw = box_deltas_map[:, :, 3]

    # Calculate the center points and dimensions of the bounding boxes
    ctr_y = dy * anc_height.unsqueeze(0) + anc_ctr_y.unsqueeze(0)
    ctr_x = dx * anc_width.unsqueeze(0) + anc_ctr_x.unsqueeze(0)
    h = torch.exp(dh) * anc_height.unsqueeze(0)
    w = torch.exp(dw) * anc_width.unsqueeze(0)

    # Initialize the boxes tensor with the appropriate shape
    boxes = torch.zeros_like(box_deltas_map, device=anchor_map.device)

    # Convert the center coordinates to (y1, x1, y2, x2)
    boxes[:, :, 0] = ctr_y - 0.5 * h
    boxes[:, :, 1] = ctr_x - 0.5 * w
    boxes[:, :, 2] = ctr_y + 0.5 * h
    boxes[:, :, 3] = ctr_x + 0.5 * w

    return boxes
