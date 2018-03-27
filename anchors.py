from __future__ import division
import torch, pdb, math
from utils import corner_form, mesh, center_form, jaccard, encode, decode

def mk_anchors(input_size):
    areas = [16*16, 32*32, 64*64, 128*128, 256*256, 512*512]
    aspect_ratios = [0.5, 1.0, 2.0]
    scales = [1.0, pow(2.0, 1.0/3.0), pow(2.0, 2.0/3.0)]
    if isinstance(input_size, int):
        input_size = torch.Tensor([input_size, input_size])
    else:
        input_size = torch.Tensor(input_size)

    anchor_hw = []
    for area in areas:
        for ar in aspect_ratios:
            h = math.sqrt(float(area) / ar)
            w = ar * h
            for scale in scales:
                anchor_hw.append([w * scale, h * scale])
    # Tensor: NFMs X 9 X 2
    hws = torch.Tensor(anchor_hw).view(len(areas), -1, 2)
    anchors = []
    for i, area in enumerate(areas):
        fm_size = (input_size / pow(2.0, i+2)).ceil()
        width = int(fm_size[0])
        height = int(fm_size[1])

        grid_size = input_size / width

        xy = mesh(width, height) + 0.5 # center point
        # Create 9 xy points for each point in each grid cell
        xy = (xy * grid_size).view(height, width, 1, 2).expand(height, width, 9, 2)

        wh = hws[i].view(1,1,9,2).expand(height, width, 9, 2)

        boxes = torch.cat([xy, wh], dim=3)
        anchors.append(boxes.view(-1, 4))

    result = torch.cat(anchors, dim=0)

    result = corner_form(result)
    result /= torch.cat([input_size, input_size], dim=0)
    result.clamp_(min=0, max=1)

    result = center_form(result)
    return result

class Anchors:
    def __init__(self, size):
        self.size = torch.Tensor([size, size])
        self.anchors = mk_anchors(size)

    # Encode the target boxes according to:
    # https://arxiv.org/pdf/1611.10012.pdf
    # TLDR: 
    #   - [10 * xc/wa, 10 * yc/ha, 5*log w, 5*log h]
    #   - Use argmax matching
    def encode(self, target):
        if len(target) == 0:
            return torch.zeros(self.anchors.shape[0], 6)

        ious = jaccard(target[:, :4], corner_form(self.anchors))
        max_iou, iou_idxs = ious.max(dim=0)

        if (max_iou >= 0.5).sum() == 0:
            return torch.zeros(self.anchors.shape[0], 6)

        boxes = center_form(target[:, :4])[iou_idxs]
        
        xy = 10 * (boxes[:, :2] - self.anchors[:, :2]) / self.anchors[:, 2:]
        wh = 5 * torch.log(boxes[:, 2:] / self.anchors[:, 2:])

        target_boxes = torch.cat([xy, wh], dim=1)

        labels = torch.zeros(target_boxes.shape[0], 1)
        labels[max_iou >= 0.5] = target[:, -1][iou_idxs[max_iou >= 0.5]] + 1
        return torch.cat([target_boxes, labels, max_iou.unsqueeze(1)], dim=1)

    # Undo the anchor offset encoding done above...
    def decode(self, boxes):
        xy = boxes[:, :2] / 10 * self.anchors[:, 2:] + self.anchors[:, :2]
        wh = self.anchors[:, 2:] * torch.exp(boxes[:, 2:] / 5)

        boxes = torch.cat([xy, wh], dim=1)

        return corner_form(boxes)

if __name__ == '__main__':
    default_type = 'torch.DoubleTensor'
    torch.set_default_tensor_type(default_type)

    anchors = Anchors(512, False)
    boxes = torch.rand(20, 4) * 0.5

    boxes[:, 2:] += 0.5

    decoded = anchors.decode(anchors.encode(boxes)[:, :4])


    pdb.set_trace()


































