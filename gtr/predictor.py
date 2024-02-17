import cv2
import numpy as np
import torch
import matplotlib.colors as mplc

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer, random_color
from detectron2.utils.video_visualizer import _create_text_labels
from detectron2.utils.visualizer import ColorMode, Visualizer

class CustomVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        height, width, _ = img_rgb.shape
        img = np.ones([height * 2, width * 2, 3], dtype=np.uint8) * 255
        img[int(0.5 * height): int(0.5 * height) + height, 
                int(0.5 * width): int(0.5 * width) + width] = img_rgb

        super().__init__(img_rgb=img, metadata=metadata, scale=scale, instance_mode=instance_mode)

class TrackingVisualizer(VideoVisualizer):
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode
        self._assigned_colors = {}
        self._max_num_instances = 10000
        self._num_colors = 74
        self._color_pool = [random_color(rgb=True, maximum=1) \
            for _ in range(self._num_colors)]
        self.color_idx = 0


    def draw_instance_predictions(self, frame, predictions):
        """
        """
        frame_visualizer = CustomVisualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        predictions.track_ids = predictions.track_ids % self._max_num_instances
        track_ids = predictions.track_ids.numpy() 
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        else:
            masks = None

        colors = self._assign_colors_by_track_id(predictions)
        labels = _create_text_labels(
            classes, scores, self.metadata.get("thing_classes", None))
        labels = ['({}) {}'.format(x, y[:y.rfind(' ')]) \
            for x, y in zip(track_ids, labels)]
        
        if predictions.has("pred_boxes"):
            height, width, _ = frame.shape
            boxes[:, 0] += int(0.5 * width)
            boxes[:, 2] += int(0.5 * width)
            boxes[:, 1] += int(0.5 * height)
            boxes[:, 3] += int(0.5 * height)

        if predictions.has("pred_masks"):
            raise ValueError('Amodal mask demo is not supported yet.')

        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes,  # boxes are a bit distracting
            masks=None if masks is None else masks,
            labels=labels,
            assigned_colors=colors,
            alpha=0.5,
        )

        return frame_visualizer.output

    def _assign_colors_by_track_id(self, instances):
        '''
        Allow duplicated colors
        '''
        colors = []
        for id_tensor in instances.track_ids:
            id = id_tensor.item()
            if id in self._assigned_colors:
                colors.append(self._color_pool[self._assigned_colors[id]])
            else:
                self.color_idx = (self.color_idx + 1) % self._num_colors
                color = self._color_pool[self.color_idx]
                self._assigned_colors[id] = self.color_idx
                colors.append(color)
        # print('self._assigned_colors', self._assigned_colors)
        return colors

class GTRPredictor(DefaultPredictor):
    @torch.no_grad()
    def __call__(self, original_frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        if self.input_format == "RGB":
            original_frames = \
                [x[:, :, ::-1] for x in original_frames]
        height, width = original_frames[0].shape[:2]
        frames = [self.aug.get_transform(x).apply_image(x) \
            for x in original_frames]
        frames = [torch.as_tensor(x.astype("float32").transpose(2, 0, 1))\
            for x in frames]
        inputs = [{"image": x, "height": height, "width": width, "video_id": 0} \
            for x in frames]
        predictions = self.model(inputs)
        return predictions


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.video_predictor = GTRPredictor(cfg)


    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


    def _process_predictions(self, tracker_visualizer, frame, predictions):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = predictions["instances"].to(self.cpu_device)
        vis_frame = tracker_visualizer.draw_instance_predictions(
            frame, predictions)
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        return vis_frame


    def run_on_video(self, video):
        """
        """
        tracker_visualizer = TrackingVisualizer(self.metadata, self.instance_mode)
        frames = [x for x in self._frame_from_video(video)]
        outputs = self.video_predictor(frames)
        for frame, instances in zip(frames, outputs):
            yield self._process_predictions(tracker_visualizer, frame, instances)


    def run_on_images(self, frames):
        """
        """
        tracker_visualizer = TrackingVisualizer(self.metadata, self.instance_mode)
        outputs = self.video_predictor(frames)
        for frame, instances in zip(frames, outputs):
            yield self._process_predictions(tracker_visualizer, frame, instances)