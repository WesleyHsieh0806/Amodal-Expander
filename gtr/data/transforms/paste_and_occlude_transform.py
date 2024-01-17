# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou
# File: transform.py
import os
import logging
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
from PIL import Image

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "PasteAndOccludeTransform",
]
logger = logging.getLogger(__name__)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    if len(img.shape) <= 3:
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return False

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha
        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    else:
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[-3], y + img_overlay.shape[-3])
        x1, x2 = max(0, x), min(img.shape[-2], x + img_overlay.shape[-2])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[-3], img.shape[-3] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[-2], img.shape[-2] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return False

        # Blend overlay within the determined ranges
        img_crop = img[..., y1:y2, x1:x2, :]
        img_overlay_crop = img_overlay[..., y1o:y2o, x1o:x2o, :]
        alpha = alpha_mask[..., y1o:y2o, x1o:x2o, :]
        alpha_inv = 1.0 - alpha
        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    return True

class PasteAndOccludeTransform(Transform):
    """
    """

    def __init__(self, scaled_h, scaled_w, offset_y, offset_x, img_scale, target_size, 
                 selected_segments,
                 segment_object_root,
                 interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            selected_segments: A list of dictionaries, where each dict contain the following keys:
                'image_path'
                'height'
                'width'
                'category_id'
                'x'
                'y'
            segment_object_root: Root path to segment object images.
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def load_segment_object_img(self, segment) -> np.ndarray:
        """Load the images of segment object, resize it, and convert it into a np array
        """
        img_path = os.path.join(self.segment_object_root, segment['image_path'])
        pil_img = Image.open(img_path)

        if segment['width'] < 0:
            logging.debug("Found a segment object with width < 0: {}".format(segment))
            segment['width'] = 100
        if segment['height'] < 0:
            logging.debug("Found a segment object with height < 0: {}".format(segment))
            segment['height'] = 100

        
        pil_img = pil_img.resize((segment['width'], segment['height']), self.interp)
        ret = np.asarray(pil_img)  # (H, W, 3)
        return ret

    def update_segment_position(self, width, height):
        """Update the segmented object position based on the width and height."""
        for selected_segment in self.selected_segments:
            img_w = width
            img_h = height

            # Randomly decide the position of the object among the range
            min_x = 0 - selected_segment['width'] * (3 / 4)
            max_x = img_w - selected_segment['width'] * (1 / 4)

            min_y = 0 - selected_segment['height'] * (3 / 4)
            max_y = img_h - selected_segment['height'] * (1 / 4)
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            selected_segment['x'] = int(x)
            selected_segment['y'] = int(y)
            
    def apply_image(self, img, interp=None):
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            pil_image = Image.fromarray(img)
            interp_method = interp if interp is not None else self.interp
            pil_image = pil_image.resize((self.scaled_w, self.scaled_h), interp_method)
            ret = np.array(pil_image)
            right = min(self.scaled_w, self.offset_x + self.target_size[1])
            lower = min(self.scaled_h, self.offset_y + self.target_size[0])
            if len(ret.shape) <= 3:
                ret = ret[self.offset_y: lower, self.offset_x: right]
            else:
                ret = ret[..., self.offset_y: lower, self.offset_x: right, :]

        else:
            # PIL only supports uint8
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {Image.BILINEAR: "bilinear", Image.BICUBIC: "bicubic"}
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            img = F.interpolate(img, (self.scaled_h, self.scaled_w), mode=mode, align_corners=False)
            shape[:2] = (self.scaled_h, self.scaled_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)
            right = min(self.scaled_w, self.offset_x + self.target_size[1])
            lower = min(self.scaled_h, self.offset_y + self.target_size[0])
            if len(ret.shape) <= 3:
                ret = ret[self.offset_y: lower, self.offset_x: right]
            else:
                ret = ret[..., self.offset_y: lower, self.offset_x: right, :]
        
        # Apply PasteAndOcclude
        for selected_segment in self.selected_segments:
            try:
                selected_segment['in_frame'] = False
                copied_selected_segment = deepcopy(selected_segment)
                copied_selected_segment['x'] = int(selected_segment['x'] * self.img_scale - self.offset_x)
                copied_selected_segment['y'] = int(selected_segment['y'] * self.img_scale - self.offset_y)
                copied_selected_segment['width'] = int(selected_segment['width'] * self.img_scale)
                copied_selected_segment['height'] = int(selected_segment['height'] * self.img_scale)

                object_img = self.load_segment_object_img(copied_selected_segment)
                object_mask = object_img != 0

                x = copied_selected_segment['x']
                y = copied_selected_segment['y']

        
                selected_segment['in_frame'] = overlay_image_alpha(ret, object_img, x, y, object_mask)
            except:
                logger.debug("PasteAndOccludeTransform: Failed to paste segmented object {}"
                             " onto the image of shape {}".format(selected_segment, ret.shape))
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * self.img_scale
        coords[:, 1] = coords[:, 1] * self.img_scale
        coords[:, 0] -= self.offset_x
        coords[:, 1] -= self.offset_y
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        raise NotImplementedError
        # return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)