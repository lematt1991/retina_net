from __future__ import division
import torch, pdb, math
from utils import corner_form, mesh, center_form

def anchors(input_size):
    areas = [16*16, 32*32]
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

    result = result / torch.cat([input_size, input_size], dim=0)
    result.clamp_(max=1, min=0)
    result = center_form(result)
    return result
