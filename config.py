from types import SimpleNamespace

class Dotable(dict):
    __getattr__= dict.__getitem__

    def __init__(self, d):
        self.update(**dict((k, self.parse(d[k])) for k in d.keys()))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__init__(d)

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
    'model_input_size' : 512,
    'image_size' : 512, # do we need to change the resolution of the image?
    'batch_size' : 8,
    'anchors' : {
        'areas' : [16, 32, 64, 128, 256, 512],
        'aspect_ratios' : [1],
        'scales' : [1.0, pow(2.0, 1.0/3.0), pow(2.0, 2.0/3.0)],
        'encoding' : 'argmax'
    },
    'optim' : {
        'weight_decay' : 5e-4,
        'lr' : 0.01,
        'momentum' : 0.9
    }    
})

