from __future__ import division
import torch, pdb, math
from utils import corner_form, mesh, center_form, jaccard, encode, decode

def mk_anchors(config):
    areas = config.anchor_areas
    aspect_ratios = config.anchor_ratios
    scales = config.anchor_scales

    input_size = torch.Tensor([config.model_input_size, config.model_input_size])
    anchors_per_cell = len(aspect_ratios) * len(scales)

    anchor_hw = []
    for area in areas:
        for ar in aspect_ratios:
            h = math.sqrt(float(area*area) / ar)
            w = ar * h
            for scale in scales:
                anchor_hw.append([w * scale, h * scale])

    # Tensor: NFMs X 3 X 2
    hws = torch.Tensor(anchor_hw).view(len(areas), -1, 2)
    anchors = []
    for i, area in enumerate(areas):
        fm_size = (input_size / area * 4).ceil()

        width = int(fm_size[0])
        height = int(fm_size[1])

        grid_size = input_size / width

        xy = mesh(width, height) + 0.5 # center point

        # Create 3 xy points for each point in each grid cell
        xy = (xy * grid_size).view(height, width, 1, 2).expand(height, width, anchors_per_cell, 2)
        wh = hws[i].view(1,1,anchors_per_cell,2).expand(height, width, anchors_per_cell, 2)

        boxes = torch.cat([xy, wh], dim=3)
        anchors.append(boxes.view(-1, 4))

    result = torch.cat(anchors, dim=0)

    result = corner_form(result)
    result /= torch.cat([input_size, input_size], dim=0)
    result.clamp_(min=0, max=1)

    result = center_form(result)

    return result

class Anchors:
    def __init__(self, config):
        self.anchors = mk_anchors(config)
        self.encode = self.encode_argmax
        self.argmax_pos_thresh = config.argmax_pos_thresh
        self.argmax_neg_thresh = config.argmax_neg_thresh

    # Encode the target boxes according to:
    # https://arxiv.org/pdf/1611.10012.pdf
    # TLDR: 
    #   - [10 * xc/wa, 10 * yc/ha, 5*log w, 5*log h]
    #   - Use bipartite matching
    def encode_bipartite(self, target):
        if len(target) == 0:
            return torch.zeros(self.anchors.shape[0], 5)

        ious = jaccard(target[:, :4], corner_form(self.anchors))

        max_iou, iou_idxs = ious.max(dim=1)

        target_boxes = torch.zeros(self.anchors.shape[0], 5)

        anchors = self.anchors[iou_idxs]
        cf = center_form(target[:, :4])
        xy = 10 * (cf[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        wh = 5 * torch.log(cf[:, 2:] / anchors[:, 2:])
        encoded = torch.cat([xy, wh], dim=1)

        target_boxes[iou_idxs] = torch.cat([encoded, target[:, -1].unsqueeze(1) + 1], dim=1)
        return target_boxes

    # Encode the target boxes according to:
    # https://arxiv.org/pdf/1611.10012.pdf
    # TLDR: 
    #   - [10 * xc/wa, 10 * yc/ha, 5*log w, 5*log h]
    #   - Use argmax matching
    def encode_argmax(self, target):
        if len(target) == 0:
            return torch.zeros(self.anchors.shape[0], 5)

        ious = jaccard(target[:, :4], corner_form(self.anchors))
        max_iou, iou_idxs = ious.max(dim=0)

        if (max_iou >= self.argmax_pos_thresh).sum() == 0:
            return torch.zeros(self.anchors.shape[0], 5)

        boxes = center_form(target[:, :4])[iou_idxs]
        
        xy = 10 * (boxes[:, :2] - self.anchors[:, :2]) / self.anchors[:, 2:]
        wh = 5 * torch.log(boxes[:, 2:] / self.anchors[:, 2:])

        target_boxes = torch.cat([xy, wh], dim=1)

        labels = torch.zeros(target_boxes.shape[0], 1)


        labels[max_iou >= self.argmax_pos_thresh, 0] = target[:, -1][iou_idxs[max_iou >= self.argmax_pos_thresh]] + 1
        labels[(max_iou > self.argmax_neg_thresh) & (max_iou < self.argmax_pos_thresh)] = -1

        # If it doesn't have a high enough threshold, still give it a label if it is the nearest anchor
        _, idxs = ious.max(dim=1)
        labels[idxs, 0] = target[:, -1] + 1 


        return torch.cat([target_boxes, labels], dim=1)

    # Undo the anchor offset encoding done above...
    def decode(self, boxes):
        xy = boxes[:, :2] / 10 * self.anchors[:, 2:] + self.anchors[:, :2]
        wh = self.anchors[:, 2:] * torch.exp(boxes[:, 2:] / 5)

        boxes = torch.cat([xy, wh], dim=1)

        return corner_form(boxes)

if __name__ == '__main__':
    from config import config
    default_type = 'torch.DoubleTensor'
    torch.set_default_tensor_type(default_type)

    anchors = Anchors(config)
    boxes = torch.rand(20, 4) * 0.5

    boxes[:, 2:] += 0.5

    decoded = anchors.decode(anchors.encode(boxes)[:, :4])


    pdb.set_trace()


































