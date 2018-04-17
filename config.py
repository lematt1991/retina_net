from types import SimpleNamespace
import argparse, pdb

class Dotable(dict):
    defaults = [
        ('model_input_size', 512),
        ('image_size', 512),
        ('argmax_pos_thresh', 0.5),
        ('argmax_neg_thresh', 0.3),
        ('step_values', (20, 50, 80)),
        ('loss_baseline', 'total')
    ]

    __getattr__= dict.__getitem__

    def __init__(self, d):
        self.update(**dict((k, self.parse(d[k])) for k in d.keys()))

    def __getstate__(self):
        return self.__dict__

    def _fill_defaults(self):
        '''
        If we add new parameters to the config that previously were not 
        modifiable, then fill what the hard coded option was.
        '''
        for k, v in self.defaults:
            if k not in self:
                setattr(self, k, v)

    def __setstate__(self, d):
        self.__init__(d)
        self._fill_defaults()

    @classmethod
    def parse(cls, v):
        if isinstance(v, dict):
            return cls(v)
        elif isinstance(v, list):
            return [cls.parse(i) for i in v]
        else:
            return v

parser = argparse.ArgumentParser()
parser.add_argument('--classes', default=['building'], type=str, nargs='+', help='Available classes')
parser.add_argument('--loss_baseline', default='positive', choices=['positive', 'total', 'hard_neg', 'pos_neg'],
    help='Baseline metric for class confidence loss function')
parser.add_argument( '--model_input_size', default=512, type=int, 
    help='Height and width of model input')
parser.add_argument('--step_values', default=(20, 50, 80), nargs='+', type=int,
    help='Learning rate reduction epoch steps')
parser.add_argument('--argmax_neg_thresh', default=0.3, type=float,
    help='If an anchor has max jaccard overlap less than this value, it is considered a negative')
parser.add_argument('--argmax_pos_thresh', default=0.7, type=float,
    help='If an anchor has max jaccard overlap greater than this value, it is considered a positive')
parser.add_argument('--image_size', default=512, type=int, help='Image size')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--focal_loss_alpha', default=[1, 95], nargs=2, type=float,
    help='Focal loss class balancing first value used for background class, second for foreground')
parser.add_argument('--s3', default='dl-training-data', type=str, 
    help='Name of S3 bucket containing the training images.')
parser.add_argument('--anchor_areas', default=[16, 32, 64, 128, 256, 512], type=int, 
    help='Number of square pixels anchors will cover in each level')
parser.add_argument('--anchor_ratios', default=[1, 2, 0.5], type=float, 
    help='Aspect ratio for each anchor')
parser.add_argument('--anchor_scales', default=[1.0, pow(2.0, 1.0/3.0), pow(2, 2.0/3.0)], type=float, 
    help='Scales for each anchor')
parser.add_argument('--anchor_encoding', default='bipartite', type=str, choices=['bipartite', 'argmax'], 
    help='Anchor box encoding type')
parser.add_argument('--loss_gamma', default=5, type=float, help='Focal loss gamma value')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Optimizer weight decay')
parser.add_argument('--learning_rate', default=0.000005, type=float, help='Learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Optimizer momentum')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.classes)


