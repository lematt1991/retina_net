import pdb, os, glob, sys, torch, numpy as np, json, re, random
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

class Transform:
    def __init__(self, config, anchors):
        self.model_input_size = config.model_input_size
        self.transform = transforms.Compose([
            transforms.Resize((self.model_input_size, self.model_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.anchors = anchors

    def __call__(self, img, target):
        '''
        img : PIL image
        target ndarray[N, 5] - minx, miny, maxx, maxy, class
        '''
        img_data = self.transform(img)
        if len(target) > 0:
            target[:, (0, 2)] /= float(img.width)
            target[:, (1, 3)] /= float(img.height)
        return img_data, self.anchors.encode(target)

class SpaceNet(data.Dataset):
    name = 'SpaceNet'
    classes = ['building']
    def __init__(self, anno_file, transform, root_dir = None):
        self.transform = transform
        self.root_dir = os.path.dirname(os.path.realpath(anno_file)) if root_dir is None else root_dir

        self.annos = json.load(open(anno_file, 'r'))

        self.keys = ['x1', 'y1', 'x2', 'y2']

        self.even()

    def __len__(self):
        return len(self.annos)

    def even(self):
        projs = {}
        for sample in self.annos:
            proj = re.search('(.*[^\d])(\d+)\.', os.path.basename(sample['image_path'])).group(1)
            if proj in projs:
                projs[proj].append(sample)
            else:
                projs[proj] = [sample]

        samples = []

        max_size = max([len(projs[k]) for k in projs.keys()])
        for proj in projs.keys():
            arr = projs[proj]
            count = 0
            while count + len(arr) <= max_size and count / 10 < len(arr):
                samples.extend(arr)
                count += len(arr)
            diff = (max_size - len(arr)) % len(arr)
            print('Added %d samples from %s' % (diff + count, proj))
            random.shuffle(arr)
            samples.extend(arr[:diff])
        self.annos = samples
        return self

    def __getitem__(self, idx):
        anno = self.annos[idx]
        img = Image.open(os.path.join(self.root_dir, anno['image_path']))

        target = torch.Tensor([[r[k] for k in self.keys] + [0] for r in anno['rects']])

        if len(target) > 0:
            mask = (target[:, 2] - target[:, 0] >= 3) & (target[:, 3] - target[:, 1] >= 3)
            target = target[mask.unsqueeze(1).expand_as(target)].view(-1, 5)
        
        img_data, target = self.transform(img, target)

        return img_data, target, torch.Tensor([img.width, img.height])

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
        return img, target, torch.Tensor([anno['width'], anno['height']])

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
    imgs, targets, width_heights = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(targets, 0)