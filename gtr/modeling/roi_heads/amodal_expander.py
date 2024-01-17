import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Linear, ShapeSpec
from .association_head import MLP

logger = logging.getLogger(__name__)
class AmodalExpander(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_shape, hidden_dim, output_dim, num_layers, num_head=8,
        dropout=0.0, zero_init=False, use_temporal=False, use_proposal_feature=True, use_modal_delta=True):
        super().__init__()

        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        
        self.input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)  # 1024
        self.output_dim = output_dim
        # The total positiona encoding dim will be input size // 4, but we have four initial positions
        self.pos_encoding_dim = self.input_size // 16  # The encoding dim for each of x, y, w, h

        self.use_proposal_feature = use_proposal_feature  # If true, use regional proposal features as one part of the input
        self.use_modal_delta = use_modal_delta  # If true, use modal delta as one part of the input
        self.expand_modal_box = MLP(self.input_size * (1 + use_temporal) * int(use_proposal_feature) + self.pos_encoding_dim * 4 * int(use_modal_delta), 
                                        hidden_dim, output_dim, num_layers, dropout)
        if not (use_proposal_feature or use_modal_delta):
            raise ValueError("At least one of use_modal_delta {} and use_proposal_features {} should be True".format(self.use_modal_delta, self.use_proposal_feature))
        # self.pos_enc = MLP(4, hidden_dim, self.pos_encoding_dim * 4, 1)
        
        if use_temporal:
            logger.info("Use Temporal Amodal Expander with Multi-Head Attention {} output dim with {} heads.".format(self.input_size, num_head))
            self.temporal_attn = nn.MultiheadAttention(self.input_size, num_head)

        if zero_init:
            for name, p in self.named_parameters():
                logger.info("Zero initialize {}".format(name))
                nn.init.zeros_(p)


    def _box_pe(self, boxes):
        '''
        boxes: Tensor of Shape (N, 4)
        '''
        N = boxes.shape[0]
        out = torch.zeros([N, self.pos_encoding_dim * 4]).to(boxes.device)

        # x1
        x_pos_enc = torch.cat([boxes[:, 0:1] / np.power(10, 2 * (j // 2) / self.pos_encoding_dim) for j in range(self.pos_encoding_dim)],
                                dim=1)
        out[:, 0::8] = torch.sin(x_pos_enc[:, 0::2])  # sin x1
        out[:, 1::8] = torch.cos(x_pos_enc[:, 1::2])  # cos x1

        # y1
        y_pos_enc = torch.cat([boxes[:, 1:2] / np.power(10, 2 * (j // 2) / self.pos_encoding_dim) for j in range(self.pos_encoding_dim)],
                                dim=1)
        out[:, 2::8] = torch.sin(y_pos_enc[:, 0::2])  # sin y1
        out[:, 3::8] = torch.cos(y_pos_enc[:, 1::2])  # cos y1

        # x2
        x2_pos_enc = torch.cat([boxes[:, 2:3] / np.power(10, 2 * (j // 2) / self.pos_encoding_dim) for j in range(self.pos_encoding_dim)],
                                dim=1)
        out[:, 4::8] = torch.sin(x2_pos_enc[:, 0::2])  # sin x2
        out[:, 5::8] = torch.cos(x2_pos_enc[:, 1::2])  # cos x2

        # y2
        y2_pos_enc = torch.cat([boxes[:, 3:4] / np.power(10, 2 * (j // 2) / self.pos_encoding_dim) for j in range(self.pos_encoding_dim)],
                                dim=1)
        out[:, 6::8] = torch.sin(y2_pos_enc[:, 0::2])  # sin y2
        out[:, 7::8] = torch.cos(y2_pos_enc[:, 1::2])  # cos y2
        return out
    
    def get_temporal_feature(self, box_features):
        """
        Input:
            box_features: features of shape (N, F)
        Output:
            temporal features (concatenation)
        """
        src = box_features.unsqueeze(0).permute(1, 0, 2)  # (N, B=1, F)
        temporal_features = self.temporal_attn(src, src, src)[0]  # (N, B=1, F)
        return torch.cat([box_features, temporal_features.view(-1, temporal_features.shape[-1])], dim=-1)

    def forward(self, modal_delta, proposals, roi_features):
        """
        Args:
            modal_delta: tensor of shape (N_proposal, NumCategory * 4) or (N_proposal, 4) as GTR box prediction is class agnostic
            proposals:
                A list of instances object with the following field:
                    proposal boxes
                    objectness logits
            roi_features:
                (N_proposal, input_size)
        Output:
            amodal_delta:
                tensor of shape (N_proposal, NumCategory * 4)
        """

        if roi_features.dim() > 2:
            # (N, input_size)
            roi_features = torch.flatten(roi_features, start_dim=1)  # (N * N_C, 4)
        
        delta_shape = modal_delta.shape
        if modal_delta.shape[-1] != self.output_dim:
            modal_delta = modal_delta.reshape([-1, self.output_dim])  # 4096, 4
        
        # Apply positional encoding for modal delta
        # Proposals are XYXY here
        pos_enc_modal_delta = self._box_pe(modal_delta)


        # Concatenate roi features and modal delta
        if self.use_modal_delta and self.use_proposal_feature:
            features = torch.cat([roi_features, pos_enc_modal_delta], dim=1)
        elif self.use_modal_delta:
            features = pos_enc_modal_delta
        elif self.use_proposal_feature:
            features = roi_features
        
        # Obtain expanded delta for amodal boxes
        amodal_expanded_delta = self.expand_modal_box(features)

        if self.use_modal_delta:
            return (modal_delta + amodal_expanded_delta).reshape(delta_shape)
        return amodal_expanded_delta.reshape(delta_shape)