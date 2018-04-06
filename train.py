#!/usr/bin/env python
import os, pdb, cv2, argparse, torch, numpy as np, time, json
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from Datasets import detection_collate, VOC, SpaceNet, Transform
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

def save_checkpoint(net, args, iter, config, filename):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save({
        'iter': iter,
        'state_dict' : net.state_dict(),
        'config' : config
    }, filename)

def load_checkpoint(net, args):
    if args.resume:
        chkpnt = torch.load(args.resume)
        args.start_iter = chkpnt['iter'] + 1
        net.load_state_dict(chkpnt['state_dict'])
    else:
        net.init_weights()

def plot_training_data(args, batch, iter):
    r = np.random.randint(0, len(batch))
    img = np.array(batch[r]['img'])[:, :, (2, 1, 0)].copy()
    target = batch[r]['target'].cpu().numpy()

    for box in target.round().astype(int):
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (0,0,255))

    img = img[:, :, (2,1,0)].transpose((2, 0, 1))
    img = torch.from_numpy(img)

    args.writer.add_image('training_data', img, iter)

def train(net, dataset, args, config):
    lr = config.optim.lr
    optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=config.optim.momentum, weight_decay=config.optim.weight_decay)

    criterion = Loss(len(dataset.classes) + 1, 3)

    net.train()

    data_loader = data.DataLoader(dataset, config.batch_size, shuffle=True, collate_fn=detection_collate)
    N = len(data_loader)

    mk_var = lambda x: Variable(x.cuda() if args.cuda else x)
    for epoch in range(args.start_iter, args.epochs):
        if epoch in args.stepvalues:
            lr = adjust_learning_rate(optimizer, lr)

        for i, (images, targets) in enumerate(data_loader):
            # plot_training_data(args, orig, N * epoch + i)

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

            mask = target[:, -1] > 0
            if mask.sum().item() > 0:
                probs = F.softmax(conf[mask.unsqueeze(1).expand_as(conf)].view(-1, conf.shape[-1]), dim=1)
                args.writer.add_histogram('data/building_conf', probs[:, 1], N * epoch + i)

            probs = F.softmax(conf[(target[:, -1] == 0).unsqueeze(1).expand_as(conf)].view(-1, conf.shape[-1]), dim=1)
            args.writer.add_histogram('data/background_conf', probs[:, 0], N * epoch + i)

            args.writer.add_scalar('data/loss', loss.item(), N * epoch + i)
            args.writer.add_scalar('data/loss_l', loss_l.item(), N * epoch + i)
            args.writer.add_scalar('data/loss_c', loss_c.item(), N * epoch + i)

            print('%d: [%d/%d] || Loss: %.4f, loss_c: %f, loss_l: %f' % (epoch, i, N, loss.item(), loss_c.item(), loss_l.item()))

        save_checkpoint(net, args, epoch, config, os.path.join(args.checkpoint_dir, 'epoch_%d.pth' % epoch))

def adjust_learning_rate(optimizer, lr):
    new_lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Adjusting learning rate, new lr is %f' % new_lr)
    return new_lr

if __name__ == '__main__':
    from config import config

    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum training epochs')
    parser.add_argument('--train_data', required=True, help='Path to training data')
    parser.add_argument('--data_dir', default=None, help='Directory of training data')
    args = parser.parse_args()

    default_type = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'
    torch.set_default_tensor_type(default_type)

    DS_Class = VOC if 'VOC' in args.train_data else SpaceNet

    net = Retina(config)
    dataset = DS_Class(args.train_data, Transform(config, net.anchors), root_dir=args.data_dir)

    args.checkpoint_dir = os.path.join(args.save_folder, 'ssd_%s' % datetime.now().isoformat())
    args.stepvalues = (20, 50, 80)
    args.start_iter = 0
    args.writer = SummaryWriter()

    if args.resume:
        args.checkpoint_dir = os.path.dirname(args.resume)

    os.makedirs(args.save_folder, exist_ok = True)

    if args.cuda:
        net = net.cuda()

    load_checkpoint(net, args)
    train(net, dataset, args, config)













