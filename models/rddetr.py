import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.backbone import CNN_layer


class RDDETR(nn.Module):
    def __init__(self, position_encoding, transformer, num_classes, num_queries, aux_loss=False, device='cuda'):
        super(RDDETR, self).__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.backbone = CNN_layer()
        self.box_head = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=6, num_layers=3)
        self.keypoint_dist_head = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                      output_dim=3 * 13, num_layers=3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.positional_encoding = position_encoding
        self.aux_loss = aux_loss
        self.device = device

    def forward(self, inputs):
        batch_size, tx_antenna_size, rx_antenna_size, sampling_size = inputs.shape
        flattened_inputs = inputs.view(batch_size, -1, sampling_size)
        channeled_inputs = flattened_inputs.unsqueeze(1)
        embedded_inputs = self.backbone(channeled_inputs)
        embedded_sequence = embedded_inputs.view(batch_size, self.backbone.out_channel, -1)
        embedded_sequence = embedded_sequence.permute(0, 2, 1)
        positional_encoding = self.positional_encoding(embedded_sequence)
        padding_mask = None
        hidden_state = self.transformer(embedded_sequence, padding_mask, self.query_embed.weight, positional_encoding)[0]
        output_3d_box = self.box_head(hidden_state).sigmoid()
        output_keypoint_dist = self.keypoint_dist_head(hidden_state).sigmoid()
        prediction = {'pred_keypoints': output_keypoint_dist[-1], 'pred_boxes': output_3d_box[-1]}
        if self.aux_loss:
            prediction['aux_outputs'] = self._set_aux_loss(output_keypoint_dist, output_3d_box)
        return prediction

    @torch.jit.unused
    def _set_aux_loss(self, output_keypoint, output_3d_box):
        return [{'pred_keypoints': k, 'pred_boxes': b} for k, b in zip(output_keypoint[:-1], output_3d_box[:-1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




