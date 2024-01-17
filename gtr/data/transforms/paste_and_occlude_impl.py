# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou
"""
Implement many useful :class:`Augmentation`.
"""
import random
import logging
import numpy as np
import sys
import copy
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    VFlipTransform,
)
from PIL import Image

from detectron2.data.transforms.augmentation import Augmentation
from .paste_and_occlude_transform import PasteAndOccludeTransform

__all__ = [
    "PasteAndOcclude",
]


class PasteAndOcclude(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.

    Then, randomly select 0~num_segments segmented object from the list to apply paste and occlude.
    """

    def __init__(
        self, size, scale, segments, num_segments=2, interp=Image.BILINEAR, h=-1, w=-1,
        segment_object_scale=(0.2, 1.2),
        segment_object_size=256,
        segment_object_root='',
    ):
        """
        Args:
            size (int): e.g., 896
            scale (tuple): e.g., (0.8, 1.2)
            segments: A list of dictionaries
                Each dict contains information about the segmented object with the following keys:
                    - image_path
                    - height
                    - width
                    - category_id
                    - annotation_id
            num_segments(int):
                maximum number of segmentds pasted onto the image
        """
        super().__init__()
        if h < 0 and w < 0:
            self.target_size = (size, size)
        else:
            self.target_size = (h, w)
        self.scale = scale
        self.interp = interp
        self.segments = segments
        self.num_segments = num_segments

        self.segment_object_size = (segment_object_size, segment_object_size)
        self.segment_object_scale = segment_object_scale
        self.segment_object_root = segment_object_root
        logger = logging.getLogger(__name__)
        logger.info("PasteAndOcclude: Get {} segmented objects. "
                    "Maximum number of pasted objects: {}".format(len(segments), num_segments))

    def randomly_select_segments(self):
        """Randomly Select 1~num_segments objects from the segment set."""
        num_of_object = np.random.randint(1, self.num_segments + 1)
        self.selected_segments = np.random.choice(self.segments, size=num_of_object)
        self.selected_segments = copy.deepcopy(self.selected_segments)

    def get_transform(self, img):
        # Select a random scale factor.
        scale_factor = np.random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]
        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.shape[1], img.shape[0]
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * np.random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * np.random.uniform(0, 1))

        # TODO: Randomly decide the position and size of the segment object
        # Send segment objects as list of dicts into transform
        selected_segments = copy.deepcopy(self.selected_segments)  # Deep copy to avoid interference between two transforms
        for selected_segment in selected_segments:
            # Randomly decide the size of the object
            object_scale_factor_h, object_scale_factor_w = np.random.uniform(*self.segment_object_scale, 2)
            scaled_object_height = min(int(object_scale_factor_h * self.segment_object_size[0]), height)
            scaled_object_width = min(int(object_scale_factor_w * self.segment_object_size[1]), width)

            selected_segment['height'] = scaled_object_height
            selected_segment['width'] = scaled_object_width

            # TODO: The problem now is img_w here is not what we actually use in Gen Image Motion
            # The width, height is (640, 480) at first and then will changed to (932, 896) 
            # As we call aug_input.apply_augmentations twice
            # img_w = min(scaled_w, offset_x + self.target_size[1]) - offset_x
            # img_h = min(scaled_h, offset_y + self.target_size[0]) - offset_y
            img_w = width
            img_h = height

            # Randomly decide the position of the object among the range
            min_x = 0 - scaled_object_width * (3 / 4)
            max_x = img_w - scaled_object_width * (1 / 4)

            min_y = 0 - scaled_object_height * (3 / 4)
            max_y = img_h - scaled_object_height * (1 / 4)
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            selected_segment['x'] = int(x)
            selected_segment['y'] = int(y)

        return PasteAndOccludeTransform(
            scaled_h, scaled_w, offset_y, offset_x, img_scale, self.target_size, selected_segments,
            self.segment_object_root,
            self.interp)