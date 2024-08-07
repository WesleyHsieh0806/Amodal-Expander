from detectron2.config import CfgNode as CN
def add_gtr_config(cfg):
    _C = cfg

    _C.MODEL.ROI_HEADS.NO_BOX_HEAD = False # One or two stage detector
    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False # classification loss
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.DELAY_CLS = False # classification after tracking
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/metadata/lvis_v1_train_cat_info.json' # LVIS metadata

    # association head
    _C.MODEL.ASSO_ON = False
    _C.MODEL.ASSO_HEAD = CN() # tracking transformer architecture parameters
    _C.MODEL.ASSO_HEAD.FC_DIM = 1024 
    _C.MODEL.ASSO_HEAD.NUM_FC = 2
    _C.MODEL.ASSO_HEAD.NUM_ENCODER_LAYERS = 1
    _C.MODEL.ASSO_HEAD.NUM_DECODER_LAYERS = 1
    _C.MODEL.ASSO_HEAD.NUM_WEIGHT_LAYERS = 2
    _C.MODEL.ASSO_HEAD.NUM_HEADS = 8
    _C.MODEL.ASSO_HEAD.DROPOUT = 0.1
    _C.MODEL.ASSO_HEAD.NORM = False
    _C.MODEL.ASSO_HEAD.ASSO_THRESH = 0.1
    _C.MODEL.ASSO_HEAD.ASSO_WEIGHT = 1.0
    _C.MODEL.ASSO_HEAD.NEG_UNMATCHED = False
    _C.MODEL.ASSO_HEAD.NO_DECODER_SELF_ATT = True
    _C.MODEL.ASSO_HEAD.NO_ENCODER_SELF_ATT = False
    _C.MODEL.ASSO_HEAD.WITH_TEMP_EMB = False
    _C.MODEL.ASSO_HEAD.NO_POS_EMB = False
    _C.MODEL.ASSO_HEAD.ASSO_THRESH_TEST = -1.0

    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.SIZE = 'B' # 'T', 'S', 'B'
    _C.MODEL.SWIN.USE_CHECKPOINT = False
    _C.MODEL.SWIN.OUT_FEATURES = (1, 2, 3) # (0, 1, 2, 3)

    # Amodal Expander
    _C.MODEL.AMODAL_EXPANDER = CN()
    _C.MODEL.AMODAL_EXPANDER.HIDDEN_DIM = 256
    _C.MODEL.AMODAL_EXPANDER.NUM_LAYER = 2
    _C.MODEL.AMODAL_EXPANDER.DROPOUT = 0.2
    _C.MODEL.AMODAL_EXPANDER.CASCADE_SHARE_WEIGHTS = True  # Whether or not to share the same weight between different cascade stages.
    _C.MODEL.AMODAL_EXPANDER.ONLY_LAST_STAGE = False  # Only Expand the modal predictions from the last stage 
    _C.MODEL.AMODAL_EXPANDER.ZERO_INIT = False  # If True, we use zero initialization for weights and biases in amodal expander
    _C.MODEL.AMODAL_EXPANDER.USE_TEMPORAL = False  # If True, we put proposal features across multiple frames into attention layer, which include temporal information from previous frames
    _C.MODEL.AMODAL_EXPANDER.USE_TEMPORAL_FROM_TRACKER = True  # If True, we use the features from transformer, which include temporal information from previous frames
    _C.MODEL.AMODAL_EXPANDER.USE_PROPOSAL_FEATURE = True  # If True, we use the region proposal feature as one part of the input to amodal expander
    _C.MODEL.AMODAL_EXPANDER.USE_MODAL_DELTA = True  # If True, we use the modal delta output by regression head as one part of the input to amodal expander
    
    _C.MODEL.FREEZE_TYPE = ''
    
    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1
    _C.SOLVER.USE_CUSTOM_SOLVER = False
    _C.SOLVER.OPTIMIZER = 'SGD'
    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0
    _C.SOLVER.CUSTOM_MULTIPLIER = 1.0
    _C.SOLVER.CUSTOM_MULTIPLIER_NAME = []

    _C.DATALOADER.SOURCE_AWARE = False
    _C.DATALOADER.DATASET_RATIO = [1, 1]

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TRAIN_H = -1
    _C.INPUT.TRAIN_W = -1
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.TEST_H = -1
    _C.INPUT.TEST_W = -1
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 
    _C.INPUT.NOT_CLAMP_BOX = False

    _C.INPUT.VIDEO = CN()
    _C.INPUT.VIDEO.TRAIN_LEN = 8 # number of frames in training
    _C.INPUT.VIDEO.TEST_LEN = 16 # number of frames for tracking in testing
    _C.INPUT.VIDEO.SAMPLE_RANGE = 2.0 # sampling frames with a random stride 
    _C.INPUT.VIDEO.DYNAMIC_SCALE = True # Increase video length for smaller resolution
    _C.INPUT.VIDEO.GEN_IMAGE_MOTION = True # Interpolate between two augmentations
    
    ## 
    _C.INPUT.USE_MODAL_MATCH = False  # If True, we use TAOAmodalDatasetMapper, which uses modal boxes for matching and amodal boxes for regression loss computation.

    ## Paste-And-Occlude Config
    _C.INPUT.USE_PASTE_AND_OCCLUDE = False  # Use Paste-And-Occlude Augmentation technique.
    
    _C.INPUT.PASTE_AND_OCCLUDE = CN()
    _C.INPUT.PASTE_AND_OCCLUDE.SEGMENT_OBJECT_ROOT = ''  # Path to the root of segment object dataset.
    _C.INPUT.PASTE_AND_OCCLUDE.NUM_SEGMENTS = 2  # Maximum number of objects pasted onto the image.
    _C.INPUT.PASTE_AND_OCCLUDE.OBJECT_SCALE = [0.2, 1.5]
    _C.INPUT.PASTE_AND_OCCLUDE.OBJECT_SIZE = 256
    _C.INPUT.PASTE_AND_OCCLUDE.USE_SEGMENT_LOSS = True
    
    _C.VIDEO_INPUT = False
    _C.VIDEO_TEST = CN()
    _C.VIDEO_TEST.OVERLAP_THRESH = 0.1 # threshold for continuing a tracking or starting a new track
    _C.VIDEO_TEST.NOT_MULT_THRESH = False # threshold for continuing a tracking or starting a new track
    _C.VIDEO_TEST.MIN_TRACK_LEN = 5 # post processing to filter out short tracks
    _C.VIDEO_TEST.MAX_CENTER_DIST = -1. # threshold for continuing a tracking or starting a new track
    _C.VIDEO_TEST.DECAY_TIME = -1. # reweighting hyper-parameters for association
    _C.VIDEO_TEST.WITH_IOU = False # combining with location in our tracker
    _C.VIDEO_TEST.LOCAL_TRACK = False # Run our baseline tracker
    _C.VIDEO_TEST.LOCAL_IOU_ONLY = False # IOU-only baseline
    _C.VIDEO_TEST.LOCAL_NO_IOU = False # ReID-only baseline

    _C.VIS_THRESH = 0.3
    _C.NOT_EVAL = False
    _C.FIND_UNUSED_PARAM = True