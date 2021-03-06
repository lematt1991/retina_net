# -*- coding: utf-8 -*-
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import log_sum_exp, jaccard, corner_form

class Loss(nn.Module):
    def __init__(self, num_classes, neg_pos, config):
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.negpos_ratio = neg_pos
        self.alpha = Variable(torch.Tensor(config.focal_loss_alpha), requires_grad=False)
        self.config = config

    def cross_entropy(self, conf_pred, target_labels):
        '''
        this is the normal hard negative mining cross entropy approach used
        in the SSD paper.
        '''
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

    def focal_loss(self, conf_pred, target_labels):
        td = target_labels.data
        onehot = torch.eye(self.num_classes)[td[td >= 0]]

        onehot = Variable(onehot, requires_grad = False) * self.alpha

        conf_pred = conf_pred[td >= 0, :]

        # Subtract max on each cell for numerical reasons
        # https://github.com/caffe2/caffe2/blob/master/modules/detectron/softmax_focal_loss_op.cu#L36-L41
        conf_pred = conf_pred - conf_pred.max(dim=1)[0].unsqueeze(1).expand_as(conf_pred)

        pt = F.softmax(conf_pred, dim=1)
        gamma = self.config.loss_gamma

        conf_loss = -(torch.pow(1 - pt, gamma) * onehot * torch.log(pt))

        if self.config.loss_baseline == 'total':
            baseline = pt.shape[0]
        elif self.config.loss_baseline == 'positive':
            baseline = max((target_labels > 0).long().sum().item(), 1)
        elif self.config.loss_baseline == 'pos_neg':
            return (conf_loss.sum(dim=0) / onehot.sum(dim=0).clamp_(min=1)).sum()
        else:
            raise ValueError('Invalid loss_baseline')

        return conf_loss / baseline

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

        # conf_loss = self.cross_entropy(conf_pred, target_labels)
        conf_loss = self.focal_loss(conf_pred, target_labels)

        # Location loss
        mask = (target_labels > 0).unsqueeze(1).expand_as(loc_pred)
        pos_pred_boxes = loc_pred[mask].view(-1, 4)
        pos_target_boxes = target_boxes[mask].view(-1, 4)

        if len(pos_pred_boxes) > 0:
            loc_loss = F.smooth_l1_loss(pos_pred_boxes, pos_target_boxes)
        else:
            loc_loss = Variable(torch.Tensor([0]))

        return loc_loss, conf_loss
























