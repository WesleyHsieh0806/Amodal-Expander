from typing import Dict, List, Optional, Tuple
import cv2
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from .custom_rcnn import CustomRCNN
from ..roi_heads.custom_fast_rcnn import custom_fast_rcnn_inference

@META_ARCH_REGISTRY.register()
class GTRRCNN(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        """
        """
        self.test_len = kwargs.pop('test_len')
        self.overlap_thresh = kwargs.pop('overlap_thresh')
        self.min_track_len = kwargs.pop('min_track_len')
        self.max_center_dist = kwargs.pop('max_center_dist')
        self.decay_time = kwargs.pop('decay_time')
        self.asso_thresh = kwargs.pop('asso_thresh')
        self.with_iou = kwargs.pop('with_iou')
        self.local_track = kwargs.pop('local_track')
        self.local_no_iou = kwargs.pop('local_no_iou')
        self.local_iou_only = kwargs.pop('local_iou_only')
        self.not_mult_thresh = kwargs.pop('not_mult_thresh')
        self.output_dir = kwargs.pop('output_dir')
        self.is_temporal_amodal_expander = kwargs.pop('is_temporal_amodal_expander')  # If True, we use temporal features from the transformer as the input to amodal expander
        super().__init__(**kwargs)


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['test_len'] = cfg.INPUT.VIDEO.TEST_LEN
        ret['overlap_thresh'] = cfg.VIDEO_TEST.OVERLAP_THRESH     
        ret['asso_thresh'] = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        ret['min_track_len'] = cfg.VIDEO_TEST.MIN_TRACK_LEN
        ret['max_center_dist'] = cfg.VIDEO_TEST.MAX_CENTER_DIST
        ret['decay_time'] = cfg.VIDEO_TEST.DECAY_TIME
        ret['with_iou'] = cfg.VIDEO_TEST.WITH_IOU
        ret['local_track'] = cfg.VIDEO_TEST.LOCAL_TRACK
        ret['local_no_iou'] = cfg.VIDEO_TEST.LOCAL_NO_IOU
        ret['local_iou_only'] = cfg.VIDEO_TEST.LOCAL_IOU_ONLY
        ret['not_mult_thresh'] = cfg.VIDEO_TEST.NOT_MULT_THRESH
        ret['output_dir'] = cfg.OUTPUT_DIR
        ret['is_temporal_amodal_expander'] = cfg.MODEL.AMODAL_EXPANDER.USE_TEMPORAL
        return ret


    def forward(self, batched_inputs):
        """
        All batched images are from the same video
        During testing, the current implementation requires all frames 
            in a video are loaded.
        TODO (Xingyi): one-the-fly testing
        """
        if not self.training:
            if self.local_track:
                return self.local_tracker_inference(batched_inputs)
            else:
                return self.sliding_inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances)
        
        if self.vis_period > 0:
            # Visualization for correctness check
            storage = get_event_storage()
            from detectron2.data.detection_utils import convert_image_to_rgb
            save_dir = self.output_dir
            for input, prop in zip(batched_inputs, proposals):
                img = input["image"]
                img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
                gt_boxes = input["instances"].gt_boxes.tensor
                gt_classes = input["instances"].gt_classes
                gt_instance_ids = input["instances"].gt_instance_ids
                gt_pasted_segment = input["instances"].gt_pasted_segment
                img = vis_image_and_annotations(img, gt_boxes, gt_classes, gt_instance_ids, gt_pasted_segment)
            
                img.save(os.path.join(save_dir, 
                        '{}_amodal.jpg'.format(len(os.listdir(save_dir)))))

            # for input, prop in zip(batched_inputs, proposals):
            #     img = input["image"]
            #     img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            #     gt_boxes = input["instances"].match_boxes.tensor
            #     gt_classes = input["instances"].gt_classes
            #     gt_instance_ids = input["instances"].gt_instance_ids
            #     img = vis_image_and_annotations(img, gt_boxes, gt_classes, gt_instance_ids)
            
            #     img.save(os.path.join(save_dir, 
            #             '{}_modal.jpg'.format(len(os.listdir(save_dir)))))

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _get_all_features(self, features_in_window):
        """
        Args:
            features_in_window: A list of length <= test_len
                each element should be a dictionary containing the keys of backbone features such as :
                    - p3
                    - p4
                    - p5
                    - p6
                    - p7
        Output:
            Dict[str->Tensor]:
        """
        keys = features_in_window[0].keys()
        all_features = {key: torch.cat([feature[key] for feature in features_in_window], dim=0) for key in keys}
        return all_features
    

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        start_frame: bool = False,
    ):
        """
        Allow not clamp box for MOT datasets
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)  # (1, 3, 800, 1088)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            if self.is_temporal_amodal_expander:
                if start_frame:
                    # We do not need to store the past features if temporal expander is not used
                    self.feature_in_window = [features]
                    self.proposal_in_window = proposals
                else:
                    self.feature_in_window.append(features)
                    self.proposal_in_window += proposals
                    
                    assert len(self.feature_in_window) == len(self.proposal_in_window)
                    if len(self.feature_in_window) > self.test_len:
                        self.feature_in_window = self.feature_in_window[1:]
                        self.proposal_in_window = self.proposal_in_window[1:]
                    
                all_features = self._get_all_features(self.feature_in_window)

                # Send in all features and proposals in the sliding window, but only get the results in the newest frame.
                # Images are not used in roi heads, so we ignore them for now.
                results, _ = self.roi_heads(images, all_features, self.proposal_in_window, None)
                results = results[-1:]
            else:
                results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes,
                not_clamp_box=self.not_clamp_box)
        else:
            return results

    def sliding_inference(self, batched_inputs):
        video_len = len(batched_inputs)
        instances = []
        id_count = 0
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False,
                start_frame=(frame_id == 0))
            instances.extend([x for x in instances_wo_id])

            if frame_id == 0: # first frame
                instances[0].track_ids = torch.arange(
                    1, len(instances[0]) + 1,
                    device=instances[0].reid_features.device)
                id_count = len(instances[0]) + 1
            else:
                win_st = max(0, frame_id + 1 - self.test_len)
                win_ed = frame_id + 1
                instances[win_st: win_ed], id_count = self.run_global_tracker(
                    batched_inputs[win_st: win_ed],
                    instances[win_st: win_ed],
                    k=min(self.test_len - 1, frame_id),
                    id_count=id_count) # n_k x N
            if frame_id - self.test_len >= 0:
                instances[frame_id - self.test_len].remove(
                    'reid_features')

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances


    def run_global_tracker(self, batched_inputs, instances, k, id_count):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)

        reid_features = torch.cat(
                [x.reid_features for x in instances], dim=0)[None]
        asso_output, pred_boxes, _, _ = self.roi_heads._forward_transformer(
            instances, reid_features, k) # [n_k x N], N x 4

        asso_output = asso_output[-1].split(n_t, dim=1) # T x [n_k x n_t]
        asso_output = self.roi_heads._activate_asso(asso_output) # T x [n_k x n_t]
        asso_output = torch.cat(asso_output, dim=1) # n_k x N

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np
        k_inds = [x for x in range(sum(n_t[:k]), sum(n_t[:k + 1]))]
        nonk_inds = [i for i in range(N) if not i in k_inds]
        asso_nonk = asso_output[:, nonk_inds] # n_k x Np
        k_boxes = pred_boxes[k_inds] # n_k x 4
        nonk_boxes = pred_boxes[nonk_inds] # Np x 4
        
        if self.roi_heads.delay_cls:
            # filter based on classification score similarity
            cls_scores = torch.cat(
                [x.cls_scores for x in instances], dim=0)[:, :-1] # N x (C + 1)
            cls_scores_k = cls_scores[k_inds] # n_k x (C + 1)
            cls_scores_nonk = cls_scores[nonk_inds] # Np x (C + 1)
            cls_similarity = torch.mm(
                cls_scores_k, cls_scores_nonk.permute(1, 0)) # n_k x Np
            asso_nonk[cls_similarity < 0.01] = 0

        unique_ids = torch.unique(ids) # M
        M = len(unique_ids) # number of existing tracks
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        # (n_k x Np) x (Np x M) --> n_k x M
        if self.decay_time > 0:
            # (n_k x Np) x (Np x M) --> n_k x M
            dts = torch.cat([x.reid_features.new_full((len(x),), T - t - 2) \
                for t, x in enumerate(instances) if t != k], dim=0) # Np
            asso_nonk = asso_nonk * (self.decay_time ** dts[None, :])

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M
        if id_inds.numel() > 0:
            last_inds = (id_inds * torch.arange(
                Np, device=id_inds.device)[:, None]).max(dim=0)[1] # M
            last_boxes = nonk_boxes[last_inds] # M x 4
            last_ious = pairwise_iou(
                Boxes(k_boxes), Boxes(last_boxes)) # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)
        
        if self.with_iou:
            traj_score = torch.max(traj_score, last_ious)
        
        if self.max_center_dist > 0.: # filter out too far-away trjactories
            # traj_score n_k x M
            k_boxes = pred_boxes[k_inds] # n_k x 4
            nonk_boxes = pred_boxes[nonk_inds] # Np x 4
            k_ct = (k_boxes[:, :2] + k_boxes[:, 2:]) / 2
            k_s = ((k_boxes[:, 2:] - k_boxes[:, :2]) ** 2).sum(dim=1) # n_k
            nonk_ct = (nonk_boxes[:, :2] + nonk_boxes[:, 2:]) / 2
            dist = ((k_ct[:, None] - nonk_ct[None, :]) ** 2).sum(dim=2) # n_k x Np
            norm_dist = dist / (k_s[:, None] + 1e-8) # n_k x Np
            # id_inds # Np x M
            valid = norm_dist < self.max_center_dist # n_k x Np
            valid_assn = torch.mm(
                valid.float(), id_inds).clamp_(max=1.).long().bool() # n_k x M
            traj_score[~valid_assn] = 0 # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.overlap_thresh * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.overlap_thresh
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        for i in range(n_k):
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
        instances[k].track_ids = track_ids

        assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        return instances, id_count


    def _remove_short_track(self, instances):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        # print("(# unique id, total # boxes):{}".format((unique_ids.shape[0], ids.shape[0])))
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N (could get CUDA OOM)
        num_insts_track = id_inds.sum(dim=1) # M
        remove_track_id = num_insts_track < self.min_track_len # M
        unique_ids[remove_track_id] = -1
        # ids = unique_ids[torch.where(id_inds.permute(1, 0))[1]]  # N (track id of each box)
        N = ids.shape[0]
        ids = torch.cat([unique_ids[torch.where(id_inds[:, int(i * N // 50): int((i+1) * N // 50)].permute(1, 0))[1]] for i in range(50)], dim=0)  # N (track id of each box)
        assert ids.shape == id_inds.shape[1:]

        ids = ids.split([len(x) for x in instances])  # [(N_1), (N_2), ..., (N_t)]
        for k in range(len(instances)):
            instances[k] = instances[k][ids[k] >= 0]
        return instances


    def _delay_cls(self, instances, video_id):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        M = len(unique_ids) # #existing tracks
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        # update scores
        cls_scores = torch.cat(
            [x.cls_scores for x in instances], dim=0) # N x (C + 1)
        traj_scores = torch.mm(id_inds, cls_scores) / \
            (id_inds.sum(dim=1)[:, None] + 1e-8) # M x (C + 1)
        _, traj_inds = torch.where(id_inds.permute(1, 0)) # N
        cls_scores = traj_scores[traj_inds] # N x (C + 1)

        n_t = [len(x) for x in instances]
        boxes = [x.pred_boxes.tensor for x in instances]
        track_ids = ids.split(n_t, dim=0)
        cls_scores = cls_scores.split(n_t, dim=0)
        instances, _ = custom_fast_rcnn_inference(
            boxes, cls_scores, track_ids, [None for _ in n_t],
            [x.image_size for x in instances],
            self.roi_heads.box_predictor[-1].test_score_thresh,
            self.roi_heads.box_predictor[-1].test_nms_thresh,
            self.roi_heads.box_predictor[-1].test_topk_per_image,
            self.not_clamp_box,
        )
        for inst in instances:
            inst.track_ids = inst.track_ids + inst.pred_classes * 10000 + \
                video_id * 100000000
        return instances

    def local_tracker_inference(self, batched_inputs):
        from ...tracking.local_tracker.fairmot import FairMOT
        local_tracker = FairMOT(
            no_iou=self.local_no_iou,
            iou_only=self.local_iou_only)

        video_len = len(batched_inputs)
        instances = []
        ret_instances = []
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])
            inst = instances[frame_id]
            dets = torch.cat([
                inst.pred_boxes.tensor, 
                inst.scores[:, None]], dim=1).cpu()
            id_feature = inst.reid_features.cpu()
            tracks = local_tracker.update(dets, id_feature)
            track_inds = [x.ind for x in tracks]
            ret_inst = inst[track_inds]
            track_ids = [x.track_id for x in tracks]
            ret_inst.track_ids = ret_inst.pred_classes.new_tensor(track_ids)
            ret_instances.append(ret_inst)
        instances = ret_instances

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances

import os
import cv2
import numpy as np
import json
from detectron2.data import detection_utils as utils
from IPython.display import display
from PIL import Image

_BLACK = (0, 0, 0)
_RED = (255, 0, 0)
_BLUE = (0, 0, 255)
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

_COLOR1 = tuple(255*x for x in (0.000, 0.447, 0.741))

def vis_image_and_annotations(image, gt_boxes, gt_classes, gt_instance_ids, gt_pasted_segments=None):
    all_annos = [{'category_id': gt_classes[i].item() + 1,
                  'bbox': np.array(gt_boxes[i]),
                 'track_id': gt_instance_ids[i].item() + 1,
                 'pasted_segments': False if gt_pasted_segments is None else gt_pasted_segments[i]} for i in range(gt_boxes.shape[0])]
    height, width = image.shape[:2]
    new_image = np.ones([height * 2, width * 2, 3], dtype=np.uint8) * 255
    startx = int(width / 2)
    endx = startx + width
    starty = int(height / 2) 
    endy = starty + height
    new_image[starty: endy, startx: endx, :] = image

    with open('/home/chengyeh/TAO-Amodal-Root/TAO-GTR/datasets/lvis/lvis_v1_train+coco_box.json', 'r') as f:
        lvis = json.load(f)
        id_to_cat_name = {cat['id']: cat['name'] for cat in lvis['categories']}

    for ann in all_annos:
        # if ann['category_id'] != 793 or ann['track_id'] in [1000001, 1000004, 1000007, 1000008]:
        #     continue
        oy, ox = new_image.shape[:2]
        oy, ox = int(oy / 4), int(ox / 4)

        box = ann['bbox']
        box = [box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy]
        new_image = vis_bbox(new_image, box, fill_opacity=-1, border_color=(213, 86, 144) \
                             if ann['pasted_segments'] else (47, 85, 151), thickness=3)
        # new_image = vis_class(new_image, box[:2], id_to_cat_name[ann['category_id']])
    

    # Check bounding box, Check Category
    pil_image = Image.fromarray(new_image)
    return pil_image

def vis_class(image,
              pos,
              class_str,
              font_scale=0.35,
              bg_color=_BLACK,
              text_color=_GRAY,
              thickness=1):
    """Visualizes the class."""
    x, y = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x, y
    back_br = x + txt_w, y + int(1.3 * txt_h)
    # Show text.
    txt_tl = x, y + int(1 * txt_h)
    cv2.rectangle(image, back_tl, back_br, bg_color, -1)
    cv2.putText(image,
                txt,
                txt_tl,
                font,
                font_scale,
                text_color,
                thickness=thickness,
                lineType=cv2.LINE_AA)
    return image

def vis_bbox(image,
             box,
             border_color=_BLACK,
             fill_color=_COLOR1,
             fill_opacity=0.65,
             thickness=1):
    """Visualizes a bounding box."""
    x0, y0, x1, y1 = box
    x1, y1 = int(x1), int(y1)
    x0, y0 = int(x0), int(y0)
    # Draw border
    if fill_opacity > 0 and fill_color is not None:
        with_fill = image.copy()
        with_fill = cv2.rectangle(with_fill, (x0, y0), (x1, y1),
                                  tuple(fill_color), cv2.FILLED)
        image = cv2.addWeighted(with_fill, fill_opacity, image,
                                1 - fill_opacity, 0, image)
        
    image = cv2.rectangle(image, (x0, y0), (x1, y1), tuple(border_color),
                          thickness)
    return image