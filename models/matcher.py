import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, \
    generalized_3d_box_iou, box_cxcyczwhd_to_xyzxyz


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_keypoint: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_keypoint = cost_keypoint
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_keypoint != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_keypoints"].shape[:2]

        out_keypoints = outputs["pred_keypoints"].flatten(0, 1) # [batch_size * num_queries, num_points]
        out_boxes = outputs['pred_boxes'].flatten(0, 1) # [batch_size * num_queries, 6]

        tgt_3d_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_keypoints = torch.cat([v["keypoints"] for v in targets])

        # Calculate target keypoint displacement
        tgt_3d_boxes_xyz = box_cxcyczwhd_to_xyzxyz(tgt_3d_bbox)[:, :3].unsqueeze(1)
        reshape_tgt_keypoints = tgt_keypoints.reshape(tgt_keypoints.shape[0], -1, 3)
        dist_keypoints = (reshape_tgt_keypoints - tgt_3d_boxes_xyz) / tgt_3d_bbox[:, 3:].unsqueeze(1)
        tgt_dist_keypoints = dist_keypoints.reshape(dist_keypoints.shape[0], -1)

        # Compute the L1 cost between keypoint displacement
        cost_keypoint = torch.cdist(out_keypoints, tgt_dist_keypoints, p=1)

        # Compute the giou cost between 3d boxes
        cost_giou = -generalized_3d_box_iou(box_cxcyczwhd_to_xyzxyz(out_boxes), box_cxcyczwhd_to_xyzxyz(tgt_3d_bbox))

        # Final cost matrix
        C = self.cost_keypoint * cost_keypoint + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
