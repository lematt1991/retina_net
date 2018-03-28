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

    def cross_entropy(self, conf_pred, target_labels, num_classes):
        num_pos = (target_labels > 0).sum().data[0]
        num_neg = max(128, num_pos * self.negpos_ratio)

        # hard negative mining
        neg_conf = conf_pred[(target_labels == 0).unsqueeze(1).expand_as(conf_pred)].view(-1, num_classes)
        _, idxs = F.softmax(neg_conf, dim=1)[:, 0].sort()

        mined_neg_conf = neg_conf[idxs[:num_neg]]
        neg_labels = Variable(torch.zeros(mined_neg_conf.shape[0]).long(), requires_grad=False)

        # Class loss
        if num_pos > 0:
            pos_conf = conf_pred[(target_labels > 0).unsqueeze(1).expand_as(conf_pred)].view(-1, num_classes)
            pos_labels = target_labels[target_labels > 0]
            conf = torch.cat([pos_conf, mined_neg_conf], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
        else:
            conf = mined_neg_conf
            labels = neg_labels

        return F.cross_entropy(conf, labels)

    def focal_loss(self, conf_pred, target_labels, num_classes):
        td = target_labels.data
        onehot = torch.eye(num_classes)[td[td >= 0]]

        onehot = Variable(onehot, requires_grad = False)

        conf_pred = conf_pred[(td >= 0).unsqueeze(1).expand_as(conf_pred)].view(-1, num_classes)

        # Subtract max on each cell for numerical reasons
        # https://github.com/caffe2/caffe2/blob/master/modules/detectron/softmax_focal_loss_op.cu#L36-L41
        conf_pred = conf_pred - conf_pred.max(dim=1)[0].unsqueeze(1).expand_as(conf_pred)

        pt = F.softmax(conf_pred, dim=1)
        gamma = 2

        conf_loss = -(torch.pow(1 - pt, gamma) * onehot * torch.log(pt)).sum()

        return conf_loss / pt.shape[0]

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

        loc_pred, conf_pred = predictions
        batch_size, num_anchors, num_classes = conf_pred.shape

        target_boxes = targets[:, :, :4].contiguous().view(-1, 4)
        target_labels = targets[:, :, 4].contiguous().view(-1).long()

        conf_pred = conf_pred.view(-1, num_classes)
        loc_pred = loc_pred.view(-1, 4)

        # conf_loss = self.cross_entropy(conf_pred, target_labels, num_classes)
        conf_loss = self.focal_loss(conf_pred, target_labels, num_classes)

        # Location loss
        mask = (target_labels > 0).unsqueeze(1).expand_as(loc_pred)
        pos_pred_boxes = loc_pred[mask].view(-1, 4)
        pos_target_boxes = target_boxes[mask].view(-1, 4)

        if len(pos_pred_boxes) > 0:
            loc_loss = F.smooth_l1_loss(pos_pred_boxes, pos_target_boxes)
        else:
            loc_loss = 0

        return loc_loss, conf_loss
























