import numpy as np


def generate_anchors(feature_map_size, patch_size, ratios, shapes):
    """
    Generate anchor boxes for object detection given the feature
    map size, sub-sample size, aspect ratios, and anchor scales.

    Generate a grid of anchor boxes (bounding boxes) centered on each point of
    a feature map. Each center point has a set of anchors with different
    aspect ratios and scales.

    @param int feature_map_size: The size of the feature map (e.g., 50).
    @param int patch_size: The stride of the sliding window or feature map
    stride (e.g., 16).
    @param list ratios: Aspect ratios of the anchors (e.g., [0.5, 1, 2]).
    @param list shapes: Anchor scales (e.g., [8, 16, 32]).
    :raises ValueError: If feature_map_size or patch_size is not a positive
    integer.
    :raises TypeError: If ratios or shapes is not a list.
    :return np.ndarray: A numpy array of shape (num_anchors, 4) containing the
    anchor boxes, where each anchor box is defined by [y_min, x_min, y_max,
    x_max].
    """

    if not isinstance(feature_map_size, int) or feature_map_size <= 0:
        raise ValueError("feature_map_size must be a positive integer")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if not isinstance(ratios, list) or not all(
        isinstance(r, (int, float)) for r in ratios
    ):
        raise TypeError("ratios must be a list of numbers")
    if not isinstance(shapes, list) or not all(
        isinstance(s, (int, float)) for s in shapes
    ):
        raise TypeError("shapes must be a list of numbers")

    # Getting the centers of 16 by 16 patches
    # We will generate the rightbottom coordinate of each patch, and
    # then subtract 8 from it to get the center
    # This center will be used to generate anchor bboxes
    # Generate center points for the feature map
    corner_x = np.arange(
        patch_size, (feature_map_size + 1) * patch_size, patch_size
    )
    corner_y = np.arange(
        patch_size, (feature_map_size + 1) * patch_size, patch_size
    )

    # Create a grid of center points
    ctr = np.array([(x - 8, y - 8) for x in corner_x for y in corner_y])

    # Calculate the total number of anchors
    num_anchors = (
        feature_map_size * feature_map_size * len(ratios) * len(shapes)
    )

    # Initialize the anchors array
    anchors = np.zeros((num_anchors, 4))

    index = 0
    # Iterate over each center point
    for c in ctr:
        corner_y, corner_x = c
        # Iterate over each combination of ratio and scale
        for i in range(len(ratios)):
            for j in range(len(shapes)):
                # Calculate the height and width of the anchor
                h = patch_size * shapes[j] * np.sqrt(ratios[i])
                w = patch_size * shapes[j] * np.sqrt(1.0 / ratios[i])

                # Calculate the coordinates of the anchor box
                anchors[index, 0] = corner_y - h / 2.0  # y_min
                anchors[index, 1] = corner_x - w / 2.0  # x_min
                anchors[index, 2] = corner_y + h / 2.0  # y_max
                anchors[index, 3] = corner_x + w / 2.0  # x_max

                # Move to the next anchor index
                index += 1

    return anchors


def get_anchors_for_coordinate(anchors, coordinate):
    """
    Retrieve all anchor boxes whose centers match a specific image coordinate.

    Given a list of anchor boxes and a specific image coordinate (x, y),
    retrieve all anchor boxes whose centers match that coordinate.

    :param list anchors: List of anchor boxes in the format
    [y_min, x_min, y_max, x_max].
    :param tuple coordinate: Tuple (x, y) specifying the image coordinate.
    :return: List of anchor boxes whose centers match the specified coordinate.
    """

    x, y = coordinate
    result_anchors = []

    for anchor in anchors:
        center_x = (anchor[1] + anchor[3]) / 2
        center_y = (anchor[0] + anchor[2]) / 2
        if center_x == x and center_y == y:
            result_anchors.append(anchor)

    return result_anchors
