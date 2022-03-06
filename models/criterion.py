import torch
from torch import nn
import torch.nn.functional as F

from util import box_ops
from util.misc import accuracy


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, test):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.test = test
        self.src_keypoints = None
        self.target_keypoints = None

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices) # matched index

        src_preds = outputs['pred_keypoints'][idx] # [batch_size, points * 3], 3d displacement
        out_boxes = outputs['pred_boxes'][idx] # [batch_size, 6]

        # Calculate predicted keypoint and target keypoint for imaging
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        reshape_src_kepoints = src_preds.reshape(src_preds.shape[0], -1, 3)
        self.src_keypoints = (box_ops.box_cxcyczwhd_to_xyzxyz(out_boxes)[:, :3].unsqueeze(1) + reshape_src_kepoints) * \
                             out_boxes[:, 3:].unsqueeze(1)
        self.target_keypoints = target_keypoints
        target_3d_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Calculate target keypoint displacement
        target_3d_boxes_xyz = box_ops.box_cxcyczwhd_to_xyzxyz(target_3d_boxes)[:, :3].unsqueeze(1)
        reshape_target_keypoints = target_keypoints.reshape(target_keypoints.shape[0], -1, 3)
        dist_keypoints = (reshape_target_keypoints - target_3d_boxes_xyz) / target_3d_boxes[:, 3:].unsqueeze(1)
        target_dist_keypoints = dist_keypoints.reshape(dist_keypoints.shape[0], -1)

        loss_keypoints = F.l1_loss(src_preds, target_dist_keypoints, reduction='none')

        losses = {}
        losses['loss_keypoints'] = loss_keypoints.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_3d_box_iou(
            box_ops.box_cxcyczwhd_to_xyzxyz(out_boxes),
            box_ops.box_cxcyczwhd_to_xyzxyz(target_3d_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses