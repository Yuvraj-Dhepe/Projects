import torch


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two
    sets of bounding boxes.

    @param box1: Tensor of shape (N, 4) containing N
    bounding boxes in XYXY format.
    @param box2: Tensor of shape (M, 4)
    containing M bounding boxes in XYXY format.

    :return Tensor: IoU values for each pair of
    boxes from box1 and box2 of shape (N, M).
    """
    # Calculate the (x, y)-coordinates of the intersection rectangles
    xA = torch.max(box1[:, None, 0], box2[:, 0])  # max of x1 of both boxes
    yA = torch.max(box1[:, None, 1], box2[:, 1])  # max of y1 of both boxes
    xB = torch.min(box1[:, None, 2], box2[:, 2])  # min of x2 of both boxes
    yB = torch.min(box1[:, None, 3], box2[:, 3])  # min of y2 of both boxes

    # Calculate the area of intersection rectangles
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

    # Calculate the area of both sets of bounding boxes
    box1Area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2Area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # Calculate the IoU
    iou = interArea / (box1Area[:, None] + box2Area - interArea)

    return iou
