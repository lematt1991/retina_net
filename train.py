#!/usr/bin/env python
import os, pdb, cv2, argparse, torch, numpy as np, time
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from Datasets import detection_collate, VOC, SpaceNet
from focal_loss import Loss
from subprocess import check_output
from datetime import datetime
from retina import Retina
from torchvision import transforms
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def save_checkpoint(net, args, iter, filename):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save({
        'iter': iter,
        'state_dict' : net.state_dict()
    }, filename)

def load_checkpoint(net, args):
    if args.resume:
        chkpnt = torch.load(args.resume)
        args.start_iter = chkpnt['iter'] + 1
        net.load_state_dict(chkpnt['state_dict'])
    else:
        net.init_weights()

class Transform:
    def __init__(self, size, anchors):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
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

def plot_training_data(args, inputs, targets, iter):
    r = np.random.randint(0, len(inputs))
    img = inputs[r].cpu().numpy().transpose((1,2,0))[:, :, (2,1,0)]
    target = targets[r].cpu().numpy()

    img = img - img.min(axis=(0, 1))
    img = img / img.max(axis=(0, 1)) * 255
    img = img.round().astype('uint8')

    target = target[:,:4] * img.shape[0]

    img = img.copy()

    for box in target.round().astype(int):
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (0,0,255))

    img = img[:, :, (2,1,0)].transpose((2, 0, 1))
    img = torch.from_numpy(img)

    args.writer.add_image('training_data', img, iter)

def train(net, dataset, args):
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = Loss(len(dataset.classes) + 1, 3)

    net.train()

    data_loader = data.DataLoader(dataset, args.batch_size, # num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate)
    N = len(data_loader)

    mk_var = lambda x: Variable(x.cuda() if args.cuda else x)
    lr = args.lr
    for epoch in range(args.start_iter, args.epochs):
        if epoch in args.stepvalues:
            lr = adjust_learning_rate(optimizer, lr)

        for i, (images, targets) in enumerate(data_loader):
            # plot_training_data(args, images, targets, N * epoch + i)

            images = mk_var(images)
            targets = mk_var(targets)

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            loc, conf = map(lambda x: x.view(-1, x.shape[-1]), out)
            target = targets.view(-1, targets.shape[-1])

            probs = F.softmax(conf[(target[:, -1] > 0).unsqueeze(1).expand_as(conf)].view(-1, conf.shape[-1]), dim=1)
            args.writer.add_histogram('data/building_conf', probs[:, 1], N * epoch + i)

            args.writer.add_scalar('data/loss', loss.data[0], N * epoch + i)
            args.writer.add_scalar('data/loss_l', loss_l.data[0], N * epoch + i)
            args.writer.add_scalar('data/loss_c', loss_c.data[0], N * epoch + i)

            print('%d: [%d/%d] || Loss: %.4f, loss_c: %f, loss_l: %f' % (epoch, i, N, loss.data[0], loss_c.data[0], loss_l.data[0]))

        save_checkpoint(net, args, epoch, os.path.join(args.checkpoint_dir, 'epoch_%d.pth' % epoch))

def adjust_learning_rate(optimizer, lr):
    new_lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Adjusting learning rate, new lr is %f' % new_lr)
    return new_lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum training epochs')
    parser.add_argument('--train_data', required=True, help='Path to training data')
    parser.add_argument('--ssd_size', default=512, type=int, help='Input dimensions for SSD')
    args = parser.parse_args()

    default_type = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'
    torch.set_default_tensor_type(default_type)

    DS_Class = VOC if 'VOC' in args.train_data else SpaceNet

    net = Retina(DS_Class.classes, args.ssd_size)
    dataset = DS_Class(args.train_data, Transform(args.ssd_size, net.anchors))

    args.checkpoint_dir = os.path.join(args.save_folder, 'ssd_%s' % datetime.now().isoformat())
    args.stepvalues = (1, 20, 50)
    args.start_iter = 0
    args.writer = SummaryWriter()

    if args.resume:
        args.checkpoint_dir = os.path.dirname(args.resume)

    os.makedirs(args.save_folder, exist_ok = True)

    if args.cuda:
        net = net.cuda()

    load_checkpoint(net, args)
    train(net, dataset, args)













