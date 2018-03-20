from __future__ import division
import torch, pdb, math
from utils import point_form

def mesh(x, y):
    '''
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    '''

    # Create linear tensor from `0 - x`, and repeat it `y` times
    xs = torch.arange(0, x).repeat(y)
    # `[0,0,0,0,1,1,1,1,2,2,2,2...]`
    ys = torch.arange(0, y).view(-1, 1).repeat(1, x).view(-1)
    # stack them side by side
    return torch.stack([xs, ys], dim=1)

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

    result = point_form(result)

    # result[:, (0, 2)] = result[:, (0, 2)].clamp(min=0, max=input_size[1])
    # result[:, (1, 3)] = result[:, (1, 3)].clamp(min=0, max=input_size[0])

    result = result / torch.cat([input_size, input_size], dim=0)
    result.clamp_(max=2, min=0)
    return result
