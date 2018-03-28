import pdb, os, glob, sys, torch, numpy as np, json
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

class SpaceNet(data.Dataset):
    name = 'SpaceNet'
    classes = ['building']
    def __init__(self, anno_file, transform):
        self.transform = transform
        self.root_dir = os.path.dirname(os.path.realpath(anno_file))

        self.annos = json.load(open(anno_file, 'r'))
        self.annos = list(filter(lambda x: len(x['rects']) > 0, self.annos))

        self.keys = ['x1', 'y1', 'x2', 'y2']

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        img = Image.open(os.path.join(self.root_dir, anno['image_path']))

        target = torch.Tensor([[r[k] for k in self.keys] + [0] for r in anno['rects']])

        mask = (target[:, 2] - target[:, 0] >= 3) & (target[:, 3] - target[:, 1] >= 3)

        target = target[mask.unsqueeze(1).expand_as(target)].view(-1, 5)
        
        img_data, target = self.transform(img, target)

        return img_data, target, img.height, img.width

class VOC(data.Dataset):
    name = 'VOC'
    classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor')
    
    def __init__(self, root_dir, transform, keep_difficult = False):
        self.transform = transform
        self.root_dir = root_dir
        anno_files = glob.glob(os.path.join(root_dir, 'Annotations/*.xml'))
        class_map = {k : i for i, k in enumerate(self.classes)}

        self.annos = []
        for file in anno_files:
            anno = ET.fromstring(open(file).read())
            imgfile = os.path.join(root_dir, 'JPEGImages', anno.find('filename').text)
            height = float(anno.find('size').find('height').text)
            width = float(anno.find('size').find('width').text)

            current = {'img' : imgfile, 'objects' : [], 'height': height, 'width': width}

            for obj in anno.findall('object'):
                bbox = obj.find('bndbox')
                if keep_difficult or obj.find('difficult').text == '0':
                    current['objects'].append((
                        float(bbox.find('xmin').text),
                        float(bbox.find('ymin').text),
                        float(bbox.find('xmax').text),
                        float(bbox.find('ymax').text),
                        float(class_map[obj.find('name').text])
                    ))
            if len(current['objects']) > 0:
                self.annos.append(current)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        img = Image.open(anno['img'])
        target = torch.Tensor(anno['objects'])
        img, target = self.transform(img, target)
        return img, target, anno['height'], anno['width']

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    imgs, targets, heights, widths = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(targets, 0)