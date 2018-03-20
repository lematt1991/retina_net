# -*- coding: utf-8 -*-
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import log_sum_exp, jaccard, corner_form

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, neg_pos, use_gpu=True, variance=[0.1, 0.2]):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = variance

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_anchors,num_classes)
                loc shape: torch.size(batch_size,num_anchors,4)
                anchors shape: torch.size(num_anchors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_pred, conf_pred, anchors = predictions
        batch_size, num_anchors, _ = loc_pred.shape
        anchors = anchors[:loc_pred.size(1), :]

        # match anchors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_size, num_anchors, 4)
        conf_t = torch.LongTensor(batch_size, num_anchors)

        loc_loss, N, conf_loss = 0, 0, 0

        for idx in range(batch_size):
            target = targets[idx]

            # Get indices of anchors that overlap the most with ground truth data
            overlaps = jaccard(target[:, :-1].data, corner_form(anchors.data))
            _, match_idxs = overlaps.max(dim=1)

            # Get best overlap for each anchor with respect to ground truth
            iou, _ = overlaps.max(dim=0)

            loc_loss += F.smooth_l1_loss(loc_pred[idx][match_idxs], target[:, :-1], size_average=False)

            current_conf = conf_pred[idx]
            pos = current_conf[match_idxs]
            num_pos = pos.shape[0]

            # Hard negative mining
            neg = current_conf[(iou < 0.4).unsqueeze(1).expand_as(current_conf)].view(-1, 2)
            _, sorted_idxs = neg[:, 0].sort(descending=True)
            neg = neg[sorted_idxs[:self.negpos_ratio * num_pos]]

            labels = torch.cat([target[:, -1].data.long() + 1, torch.zeros(len(neg)).long()], dim=0)

            conf_loss += F.cross_entropy(torch.cat([pos, neg], dim=0), Variable(labels))

            N += len(target)

        loc_loss /= N
        conf_loss /= N
        return loc_loss, conf_loss
