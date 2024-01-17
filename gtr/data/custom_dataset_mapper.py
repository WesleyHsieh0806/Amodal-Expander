# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou: allow not clamp box
import copy
import logging
import os 

import json
import numpy as np
from typing import List, Optional, Union
import torch
import pycocotools.mask as mask_util

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import BoxMode

from .transforms.paste_and_occlude_impl import PasteAndOcclude
__all__ = ["CustomDatasetMapper"]

def custom_transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None, 
    not_clamp_box=False,
):
    """
    Not clamp box
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    if not not_clamp_box:
        bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
        annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    else:
        bbox = transforms.apply_box(np.array([bbox]))[0]
        annotation["bbox"] = bbox

    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation

class CustomDatasetMapper(DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        not_clamp_box: bool = False,
        use_paste_and_occlude: bool = False,
        segment_object_root: str = '',
        num_segments: int = 2,
        segment_object_scale: tuple = (0.5, 1.5),
        segment_object_size: int = 256,
        use_segment_loss: bool = True,
    ):
        """
        add instance_id # with_crowd, with_distill_score keys
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.not_clamp_box = not_clamp_box

        self.use_paste_and_occlude = use_paste_and_occlude
        self.segment_object_root = segment_object_root
        self.num_segments = num_segments
        self.segment_object_scale = segment_object_scale
        self.segment_object_size = segment_object_size
        self.use_segment_loss = use_segment_loss

        if self.use_paste_and_occlude and is_train:
            # Load segment_object.json
            with open(os.path.join(segment_object_root, 'segment_object.json'), 'r') as f:
                segment_set = json.load(f)
            
            logger.info("Use PasteNOcclude for amodal detection training...")
            logger.info('PasteAndOcclude Image Resize Scale: {}'.format(augmentations[0].scale))
            self.augmentations = [
                PasteAndOcclude(
                    augmentations[0].target_size[0], augmentations[0].scale, 
                    segment_set['segments'],
                    num_segments=num_segments,
                    segment_object_scale=segment_object_scale,
                    segment_object_size=segment_object_size,
                    segment_object_root=segment_object_root
                ),
                T.RandomFlip()
            ]


    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret['not_clamp_box'] = cfg.INPUT.NOT_CLAMP_BOX
        ret['use_paste_and_occlude'] = cfg.INPUT.USE_PASTE_AND_OCCLUDE
        ret['segment_object_root'] = cfg.INPUT.PASTE_AND_OCCLUDE.SEGMENT_OBJECT_ROOT
        ret['num_segments'] = cfg.INPUT.PASTE_AND_OCCLUDE.NUM_SEGMENTS
        ret['segment_object_scale'] = cfg.INPUT.PASTE_AND_OCCLUDE.OBJECT_SCALE
        ret['segment_object_size'] = cfg.INPUT.PASTE_AND_OCCLUDE.OBJECT_SIZE
        ret['use_segment_loss'] = cfg.INPUT.PASTE_AND_OCCLUDE.USE_SEGMENT_LOSS
        return ret

    def __call__(self, dataset_dict):
        """
        include is_crowd and instance_id
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            if self.use_paste_and_occlude:
                raise ValueError("PasteNOcclude is not tested with segmentation yet. Remove this line only if you still want to move on with PasteNOcclude.")
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)

        if self.use_paste_and_occlude:
            # Randomly select new segment objects
            self.augmentations[0].randomly_select_segments()
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # Add the annotations of the segment set.
            if self.use_paste_and_occlude and self.use_segment_loss:
                max_inst_id = max([ann['instance_id'] for ann in dataset_dict["annotations"]]) if dataset_dict["annotations"] else 0
                for inst_id, selected_segment in enumerate(transforms[0].selected_segments):
                    if not selected_segment['in_frame']:
                        continue
                    segment_ann = {
                        'bbox': [selected_segment['x'],
                                selected_segment['y'],
                                selected_segment['width'],
                                selected_segment['height']],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        'category_id': selected_segment['category_id'] - 1,  # Convert 1-indexed to 0-indexed
                        'score': 1,
                        'instance_id': max_inst_id + 1 + inst_id,
                        'pasted_segments': True,
                    }

                    dataset_dict["annotations"].append(segment_ann)

            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            all_annos = [
                (custom_transform_instance_annotations(
                    obj, transforms, image_shape, 
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                    not_clamp_box=self.not_clamp_box
                ),  obj.get("iscrowd", 0))
                for obj in dataset_dict.pop("annotations")
            ]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]
            # test_paste_and_occlude(image, all_annos, dataset_dict, 0)

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # gt_instance_ids are required for GTR w tracker.
            # Although GTR could be trained with GTR dataloader, if we only want to fine-tune the regression head, we can still use this dataloader to load more images in a batch
            instance_ids = [obj.get('instance_id', 0) for obj in annos]
            instances.gt_instance_ids = torch.tensor(instance_ids, dtype=torch.int64)
            
            del all_annos
            
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

# from PIL import Image
# import cv2
# _BLACK = (0, 0, 0)
# _RED = (255, 0, 0)
# _BLUE = (0, 0, 255)
# _GRAY = (218, 227, 218)
# _GREEN = (18, 127, 15)
# _WHITE = (255, 255, 255)

# _COLOR1 = tuple(255*x for x in (0.000, 0.447, 0.741))

# def test_paste_and_occlude(image, all_annos, dataset_dict, idx):
#     height, width = image.shape[:2]
#     new_image = np.ones([height * 2, width * 2, 3], dtype=np.uint8) * 255
#     startx = int(width / 2)
#     endx = startx + width
#     starty = int(height / 2) 
#     endy = starty + height
#     new_image[starty: endy, startx: endx, :] = image

#     with open('/home/chengyeh/TAO-Amodal-Root/TAO-GTR/datasets/lvis/lvis_v1_train+coco_box.json', 'r') as f:
#         lvis = json.load(f)
#         id_to_cat_name = {cat['id'] - 1: cat['name'] for cat in lvis['categories']}

#     for ann in all_annos[:2]:
#         if 'pasted_segments' in ann[0]:
#             break
#         oy, ox = new_image.shape[:2]
#         oy, ox = int(oy / 4), int(ox / 4)

#         box = ann[0]['bbox']
#         box = [box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy]
#         new_image = vis_bbox(new_image, box, fill_opacity=0.0, border_color=_BLUE, thickness=3)
#         new_image = vis_class(new_image, box[:2], id_to_cat_name[ann[0]['category_id']])
    
#     for ann in all_annos[-5:]:
#         if 'pasted_segments' not in ann[0]:
#             continue
#         oy, ox = new_image.shape[:2]
#         oy, ox = int(oy / 4), int(ox / 4)

#         box = ann[0]['bbox']
#         box = [box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy]
#         new_image = vis_bbox(new_image, box, fill_opacity=0.0, border_color=_RED, thickness=3)
#         new_image = vis_class(new_image, box[:2], id_to_cat_name[ann[0]['category_id']])

#     # Check bounding box, Check Category
#     pil_image = Image.fromarray(new_image)
#     pil_image.save('/data3/chengyeh/TAO-Amodal-experiments/GTR/LVIS_COCO_500/AModalDetector/{}'.format(dataset_dict["file_name"].split('/')[-1].replace('.jpg', '_{}.jpg'.format(idx))))

# def vis_class(image,
#               pos,
#               class_str,
#               font_scale=0.35,
#               bg_color=_BLACK,
#               text_color=_GRAY,
#               thickness=1):
#     """Visualizes the class."""
#     x, y = int(pos[0]), int(pos[1])
#     # Compute text size.
#     txt = class_str
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
#     # Place text background.
#     back_tl = x, y
#     back_br = x + txt_w, y + int(1.3 * txt_h)
#     # Show text.
#     txt_tl = x, y + int(1 * txt_h)
#     cv2.rectangle(image, back_tl, back_br, bg_color, -1)
#     cv2.putText(image,
#                 txt,
#                 txt_tl,
#                 font,
#                 font_scale,
#                 text_color,
#                 thickness=thickness,
#                 lineType=cv2.LINE_AA)
#     return image

# def vis_bbox(image,
#              box,
#              border_color=_BLACK,
#              fill_color=_COLOR1,
#              fill_opacity=0.65,
#              thickness=1):
#     """Visualizes a bounding box."""
#     x0, y0, x1, y1 = box
#     x1, y1 = int(x1), int(y1)
#     x0, y0 = int(x0), int(y0)
#     # Draw border
#     if fill_opacity > 0 and fill_color is not None:
#         with_fill = image.copy()
#         with_fill = cv2.rectangle(with_fill, (x0, y0), (x1, y1),
#                                   tuple(fill_color), cv2.FILLED)
#         image = cv2.addWeighted(with_fill, fill_opacity, image,
#                                 1 - fill_opacity, 0, image)
        
#     image = cv2.rectangle(image, (x0, y0), (x1, y1), tuple(border_color),
#                           thickness)
#     return image