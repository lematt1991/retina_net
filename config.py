from types import SimpleNamespace

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

config = Dotable.parse({
    'classes' : ['building'],
    'loss_baseline' : 'total', # either 'total' or 'positive'
    'model_input_size' : 512,
    'step_values' : (20, 50, 80), # learning rate reduction at each epoch
    'argmax_neg_thresh' : 0.3, # if overlap between anchor is less than this, consider negative example
    'argmax_pos_thresh' : 0.7, # if overlap between anchor is higher than this, consider it positive
    'image_size' : 512, # do we need to change the resolution of the image?
    'batch_size' : 8,
    'focal_loss_alpha' : [1, 95],
    's3' : 'dl-training-data',
    'anchors' : {
        'areas' : [16, 32, 64, 128, 256, 512],
        'aspect_ratios' : [1, 2, 0.5],
        'scales' : [1.0, pow(2.0, 1.0/3.0), pow(2, 2.0/3.0)],
        'encoding' : 'argmax' # bipartite or argmax
    },
    'loss_gamma' : 4,
    'optim' : {
        'weight_decay' : 5e-4,
        'lr' : 0.01,
        'momentum' : 0.9
    }    
})

