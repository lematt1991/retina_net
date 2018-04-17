import os, pdb, numpy as np, time, json, pandas, glob, hashlib, torch, cv2
from shapely.geometry import MultiPolygon, box
from subprocess import check_output
from zipfile import ZipFile
from retina import Retina
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from Datasets import Transform, SpaceNet

pandas.options.mode.chained_assignment = None

NAME = 'RetinaNet'

class RetinaNet:
    name = NAME

    @classmethod
    def mk_hash(cls, path):
        '''
        Create an MD5 hash from a models weight file.
        Arguments:
            path : str - path to RetinaNet checkpoint
        '''
        dirs = path.split('/')
        if 'retina_net' in dirs:
            dirs = dirs[dirs.index('retina_net'):]
            path = '/'.join(dirs)
        else:
            path = os.path.join('retina_net', path)

        md5 = hashlib.md5()
        md5.update(path.encode('utf-8'))
        return md5.hexdigest()

    @classmethod
    def zip_weights(cls, path, base_dir='./'):
        if os.path.splitext(path)[1] != '.pth':
            raise ValueError('Invalid checkpoint')

        dirs = path.split('/')

        res = {
            'name' : 'RetinaNet',
            'instance' : '_'.join(dirs[-2:]),
            'id' : cls.mk_hash(path)
        }

        zipfile = os.path.join(base_dir, res['id'] + '.zip')

        if os.path.exists(zipfile):
            os.remove(zipfile)

        weight_dir = os.path.dirname(path)

        with ZipFile(zipfile, 'w') as z:
            z.write(path, os.path.join(res['id'], os.path.basename(path)))

        return zipfile

    def __init__(self, weights, classes=['building'], cuda = True):
        chkpnt = torch.load(weights)
        self.config = chkpnt['args']
        self.net = Retina(self.config).eval()
        self.net.load_state_dict(chkpnt['state_dict'])
        self.transform = transforms.Compose([
            transforms.Resize((self.config.model_input_size, self.config.model_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.net = self.net.cuda()
        self.net.anchors.anchors = self.net.anchors.anchors.cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.cuda = cuda

    def predict_image(self, image, eval_mode = False):
        """
        Infer buildings for a single image.
        Inputs:
            image :: n x m x 3 ndarray - Should be in RGB format
        """

        t0 = time.time()
        img = self.transform(image)
        if self.cuda:
            img = img.cuda()

        out = self.net(Variable(img.unsqueeze(0), requires_grad=False)).squeeze().data.cpu().numpy()
        total_time = time.time() - t0
        
        out = out[1] # ignore background class

        out[:, (1, 3)] = np.clip(out[:, (1, 3)] * image.width, a_min=0, a_max=image.width)
        out[:, (2, 4)] = np.clip(out[:, (2, 4)] * image.height, a_min=0, a_max=image.height)

        out = out[out[:, 0] > 0]

        return pandas.DataFrame(out, columns=['score', 'x1' ,'y1', 'x2', 'y2'])

    def predict_all(self, test_boxes_file, batch_size=8, data_dir = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(test_boxes_file))
        
        annos = json.load(open(test_boxes_file))

        total_time = 0.0

        for batch in range(0, len(annos), batch_size):
            images,  sizes = [], []
            for i in range(min(batch_size, len(annos) - batch)):
                img = Image.open(os.path.join(data_dir, annos[batch + i]['image_path']))
                images.append(self.transform(img))
                sizes.append(torch.Tensor([img.width, img.height]))

            images = torch.stack(images)
            sizes = torch.stack(sizes)

            if self.cuda:
                images = images.cuda()
                sizes = sizes.cuda()

            out = self.net(Variable(images, requires_grad=False)).data

            hws = torch.cat([sizes, sizes], dim=1).view(-1, 1, 1, 4).expand(-1, out.shape[1], out.shape[2], -1)

            out[:, :, :, 1:] *= hws
            out = out[:, 1, :, :].cpu().numpy()

            for i, detections in enumerate(out):
                anno = annos[batch + i]
                pred = cv2.imread('../data/' + anno['image_path'])

                detections = detections[detections[:, 0] > 0]
                df = pandas.DataFrame(detections, columns=['score', 'x1', 'y1', 'x2', 'y2'])
                df['image_id'] = anno['image_path']

                truth = pred.copy()

                for box in df[['x1', 'y1', 'x2', 'y2']].values.round().astype(int):
                    cv2.rectangle(pred, tuple(box[:2]), tuple(box[2:4]), (0,0,255))

                for r in anno['rects']:
                    box = list(map(lambda x: int(r[x]), ['x1', 'y1', 'x2', 'y2']))
                    cv2.rectangle(truth, tuple(box[:2]), tuple(box[2:]), (0, 0, 255))

                data = np.concatenate([pred, truth], axis=1)
                cv2.imwrite('samples/image_%d.jpg' % (batch + i), data)

                yield df

if __name__ == '__main__':
    import cv2, sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--img', required=True)
    args = parser.parse_args()
    
    img = Image.open(args.img)
    
    ssd = RetinaNet(args.weights, size=512)
    boxes = ssd.predict_image(img)

    img_data = np.array(img)[:, :, (2, 1, 0)].copy()

    for box in boxes[['x1', 'y1', 'x2', 'y2']].values[:10].round().astype(int):
        cv2.rectangle(img_data, tuple(box[:2]), tuple(box[2:]), (0,0,255))

    cv2.imwrite('out.jpg', img_data)







