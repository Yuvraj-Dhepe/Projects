import os
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms import v2


class Masks2BboxDs(torch.utils.data.Dataset):
    def __init__(self, root, image_dir, mask_dir, transforms=None):
        self.root = root
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        # load all image files in a sorted manner to ensure
        # they are aligned properly

        self.imgs = list(sorted(os.listdir(os.path.join(root, image_dir))))
        self.masks = list(sorted(os.listdir(os.path.join(root, mask_dir))))

    # Generate the target dictionary for an image of idx in the image_dir
    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root, self.image_dir, self.imgs[idx])
        mask_path = os.path.join(self.root, self.mask_dir, self.masks[idx])

        # Reading the image and mask
        img = read_image(img_path)
        mask = read_image(mask_path)

        # Instances are encoded as different colors
        obj_ids = torch.unique(mask)

        # First id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # Split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # Get bounding box cooridinates for each mask
        boxes = masks_to_boxes(masks)

        # There is only one class
        labels = [str(1) for _ in range(num_objs)]

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors
        img = tv_tensors.Image(img)

        #
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=v2.functional.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = is_crowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
