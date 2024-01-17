import os
import copy
import logging
import json
import numpy as np
from typing import List, Optional, Union
import torch

import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.structures import Keypoints, PolygonMasks, BitMasks

from .transforms.custom_augmentation_impl import EfficientDetResizeCrop
from .transforms.paste_and_occlude_impl import PasteAndOcclude
__all__ = ["GTRDatasetMapper"]


def tao_amodal_transform_instance_annotations(
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

    # Prepare boxes for matching
    # bbox is 1d (per-instance bounding box)
    match_bbox = BoxMode.convert(annotation["match_bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    # As the match bbox should be modal, we clip them
    match_bbox = transforms.apply_box(np.array([match_bbox]))[0].clip(min=0)
    annotation["match_bbox"] = np.minimum(match_bbox, list(image_size + image_size)[::-1])


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

def tao_amodal_annotations_to_instances(annos, image_size, mask_format="polygon", \
    with_inst_id=False):
    """
    Add instance id
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    match_boxes = [BoxMode.convert(obj["match_bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target.match_boxes = Boxes(match_boxes)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            # TODO check type and provide better error
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    if with_inst_id:
        instance_ids = [obj.get('instance_id', 0) for obj in annos]
        target.gt_instance_ids = torch.tensor(instance_ids, dtype=torch.int64)

    # Check whether the target is a pasted segment
    target.gt_pasted_segment = torch.tensor([obj.get('pasted_segments', False) for obj in annos], 
                                            dtype=torch.bool)
    return target


class TAOAmodalDatasetModalMatchMapper(DatasetMapper):
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
        train_len: int = 8,
        not_clamp_box: bool = False,
        sample_range: float = 2.,
        dynamic_scale: bool = False,
        gen_image_motion: bool = False,
        segment_object_root: str = '',
        num_segments: int = 2,
        segment_object_scale: tuple = (0.5, 1.5),
        segment_object_size: int = 256,
        use_paste_and_occlude: bool = False,
    ):
        """
        add instance_id
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
        logger.info('Using Modal Boxes for Matching. Remember to ensure that your modal boxes provided in the annotation json are different from amodal boxes.')
        logger.info("Augmentations used in training: " + str(augmentations))
        self.train_len = train_len
        self.not_clamp_box = not_clamp_box
        self.sample_range = sample_range
        self.dynamic_scale = dynamic_scale
        self.gen_image_motion = gen_image_motion
        self.use_paste_and_occlude = use_paste_and_occlude
        if self.gen_image_motion and is_train:
            if not self.use_paste_and_occlude:
                logger.info('EfficientDetResizeCrop Resize Scale: {}'.format(augmentations[0].scale))
                self.motion_augmentations = [
                    EfficientDetResizeCrop(
                        augmentations[0].target_size[0], augmentations[0].scale)]
            else:
                # Load segment_object.json
                with open(os.path.join(segment_object_root, 'segment_object.json'), 'r') as f:
                    segment_set = json.load(f)
                logger.info("Motion Augmentations used in training: " + 'PasteAndOcclude')
                logger.info('PasteAndOcclude Image Resize Scale: {}'.format(augmentations[0].scale))
                logger.info('PasteAndOcclude Object Resize Scale: {}'.format(segment_object_scale))
                logger.info('PasteAndOcclude Object Size: {}'.format(segment_object_size))
                self.motion_augmentations = [
                    PasteAndOcclude(
                        augmentations[0].target_size[0], augmentations[0].scale, 
                        segment_set['segments'],
                        num_segments=num_segments,
                        segment_object_scale=segment_object_scale,
                        segment_object_size=segment_object_size,
                        segment_object_root=segment_object_root)]

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret['train_len'] = cfg.INPUT.VIDEO.TRAIN_LEN
        ret['not_clamp_box'] = cfg.INPUT.NOT_CLAMP_BOX
        ret['sample_range'] = cfg.INPUT.VIDEO.SAMPLE_RANGE
        ret['dynamic_scale'] = cfg.INPUT.VIDEO.DYNAMIC_SCALE
        ret['gen_image_motion'] = cfg.INPUT.VIDEO.GEN_IMAGE_MOTION
        ret['segment_object_root'] = cfg.INPUT.PASTE_AND_OCCLUDE.SEGMENT_OBJECT_ROOT
        ret['num_segments'] = cfg.INPUT.PASTE_AND_OCCLUDE.NUM_SEGMENTS
        ret['segment_object_scale'] = cfg.INPUT.PASTE_AND_OCCLUDE.OBJECT_SCALE
        ret['segment_object_size'] = cfg.INPUT.PASTE_AND_OCCLUDE.OBJECT_SIZE
        ret['use_paste_and_occlude'] = cfg.INPUT.USE_PASTE_AND_OCCLUDE
        return ret


    def __call__(self, video_dict):
        """
        video_dict: {'video_id': int, 'images': [{'image_id', 'annotations': []}]}
        """
        if self.is_train:
            num_frames = min(len(video_dict['images']), self.train_len)
        else:
            num_frames = len(video_dict['images'])
        st = np.random.randint(len(video_dict['images']) - num_frames + 1)
        gen_image_motion = self.gen_image_motion and self.is_train and \
            len(video_dict['images']) == 1

        if self.dynamic_scale and self.is_train and not gen_image_motion:
            image = utils.read_image(
                video_dict['images'][st]["file_name"], format=self.image_format)
            aug_input = T.StandardAugInput(image)
            transforms = aug_input.apply_augmentations(self.augmentations)
            auged_size = max(transforms[0].scaled_w, transforms[0].scaled_h)
            target_size = max(transforms[0].target_size)
            max_frames = int(num_frames * (target_size / auged_size) ** 2)
            if max_frames > self.train_len:
                num_frames = np.random.randint(
                    max_frames - self.train_len + 1) + self.train_len
                num_frames = min(self.train_len * 2, num_frames)
                num_frames = min(len(video_dict['images']), num_frames)
        else:
            transforms = None
        
        if gen_image_motion:
            num_frames = self.train_len
            images_dict = [copy.deepcopy(
                video_dict['images'][0]) for _ in range(num_frames)]
            image = utils.read_image(
                video_dict['images'][0]["file_name"], format=self.image_format)
            width, height = image.shape[1], image.shape[0]
            aug_input = T.StandardAugInput(image)

            # Randomly select new segment objects
            if self.use_paste_and_occlude:
                self.motion_augmentations[0].randomly_select_segments()


            transforms_st = aug_input.apply_augmentations(self.motion_augmentations)
            transforms_ed = aug_input.apply_augmentations(self.motion_augmentations)
            transforms_list = []

            # Update the position of segmented object in the last frame.
            if self.use_paste_and_occlude:
                transforms_ed[0].update_segment_position(width, height) 
            for x in range(num_frames):
                if num_frames == 1:
                    transforms_list.append(transforms_st)
                    break
                
                trans = copy.deepcopy(transforms_st)
                trans[0].offset_x += (transforms_ed[0].offset_x - \
                    transforms_st[0].offset_x) * x // (num_frames - 1)
                trans[0].offset_y += (transforms_ed[0].offset_y - \
                    transforms_st[0].offset_y) * x // (num_frames - 1)
                trans[0].img_scale += (transforms_ed[0].img_scale - \
                    transforms_st[0].img_scale) * x / (num_frames - 1)
                trans[0].scaled_h = int(height * trans[0].img_scale)
                trans[0].scaled_w = int(width * trans[0].img_scale)

                if self.use_paste_and_occlude:
                    for i, selected_segments in enumerate(trans[0].selected_segments):
                        selected_segments['x'] += (transforms_ed[0].selected_segments[i]['x'] - \
                        transforms_st[0].selected_segments[i]['x']) * x // (num_frames - 1)
                        selected_segments['y'] += (transforms_ed[0].selected_segments[i]['y'] - \
                        transforms_st[0].selected_segments[i]['y']) * x // (num_frames - 1)
                        
                        selected_segments['width'] += (transforms_ed[0].selected_segments[i]['width'] - \
                        transforms_st[0].selected_segments[i]['width']) * x // (num_frames - 1)
                        
                        selected_segments['height'] += (transforms_ed[0].selected_segments[i]['height'] - \
                        transforms_st[0].selected_segments[i]['height']) * x // (num_frames - 1)

                transforms_list.append(trans)
        elif self.sample_range > 1. and self.is_train:
            ed = min(st + int(self.sample_range * num_frames), len(video_dict['images']))
            num_frames = min(num_frames, ed - st)
            inds = sorted(
                np.random.choice(range(st, ed), size=num_frames, replace=False))
            images_dict = copy.deepcopy([video_dict['images'][x] for x in inds])
        else:
            images_dict = copy.deepcopy(video_dict['images'][st: st + num_frames])
        
        ret = []
        for i, dataset_dict in enumerate(images_dict):
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.StandardAugInput(image)
            if gen_image_motion:
                transforms = transforms_list[i]
                image = transforms.apply_image(image)

                if "annotations" in dataset_dict and self.use_paste_and_occlude:
                    max_inst_id = max([ann['instance_id'] for ann in dataset_dict["annotations"]]) if dataset_dict["annotations"] else 0
                    max_inst_id = max([ann['instance_id'] for ann in dataset_dict["annotations"]]) if dataset_dict["annotations"] else 0
                    for inst_id, selected_segment in enumerate(transforms[0].selected_segments):
                        if not selected_segment['in_frame']:
                            continue
                        segment_ann = {
                            'bbox': [selected_segment['x'],
                                    selected_segment['y'],
                                    selected_segment['width'],
                                    selected_segment['height']],
                            'match_bbox': [selected_segment['x'],
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
            elif transforms is None:
                transforms = aug_input.apply_augmentations(self.augmentations)
                image = aug_input.image
            else:
                image = transforms.apply_image(image)

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)

            if "annotations" in dataset_dict:
                for anno in dataset_dict["annotations"]:
                    if not self.use_instance_mask:
                        anno.pop("segmentation", None)
                    if not self.use_keypoint:
                        anno.pop("keypoints", None)

                all_annos = [
                    (tao_amodal_transform_instance_annotations(
                        obj, transforms, image_shape, 
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                        not_clamp_box=self.not_clamp_box,
                    ),  obj.get("iscrowd", 0))
                    for obj in dataset_dict.pop("annotations")
                ]
                annos = [ann[0] for ann in all_annos if ann[1] == 0]
                instances = tao_amodal_annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format,
                    with_inst_id=True
                )

                del all_annos
                dataset_dict["instances"] = utils.filter_empty_instances(instances)
            ret.append(dataset_dict)
        return ret
