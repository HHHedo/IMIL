import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientBackBone(nn.Module):
    def __init__(self, name='b0'):
        super(EfficientBackBone, self).__init__()
        self.efficient_backbone = EfficientNet.from_name('efficientnet-' + name)
        self.feat_dim = self.efficient_backbone._bn1.num_features
    def forward(self, x):
        x = self.efficient_backbone.extract_features(x)
        return x