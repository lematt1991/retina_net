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
        self.alpha = Variable(torch.Tensor([0.25] + [0.75] * (num_classes - 1)), requires_grad=False)

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

        try:
            loc_pred, conf_pred = predictions
            batch_size, num_anchors, num_classes = conf_pred.shape

            target_boxes = targets[:, :, :4].contiguous().view(-1, 4)
            target_labels = targets[:, :, 4].contiguous().view(-1).long()
            ious = targets[:, :, 5].contiguous().view(-1)

            loc_pred = loc_pred.view(-1, 4)

            mask = (ious >= 0.5).unsqueeze(1).expand_as(loc_pred)
            pos_pred_boxes = loc_pred[mask].view(-1, 4)
            pos_target_boxes = target_boxes[mask].view(-1, 4)

            loc_loss = F.smooth_l1_loss(pos_pred_boxes, pos_target_boxes)

            onehot = Variable(torch.eye(num_classes)[target_labels.data])

            # Ignore ambiguous cases where overlap is close, but not close enough...
            onehot[((ious > 0.3) & (ious < 0.5)).unsqueeze(1).expand_as(onehot)] = 0

            #Subtract max on each cell for numerical reasons
            conf_pred = conf_pred.view(-1, num_classes)
            conf_pred = conf_pred - conf_pred.max(dim=1)[0].unsqueeze(1).expand_as(conf_pred)

            pt = F.softmax(conf_pred, dim=1).clamp(min=0.000001, max=0.999999)
            gamma = 2

            conf_loss = -(torch.pow(1 - pt, gamma) * onehot * torch.log(pt)).sum()

            num_pos = max((ious > 0.5).sum().data[0], 1)
        except Exception as e:
            pdb.set_trace()

        return loc_loss / num_pos, conf_loss / num_pos
























