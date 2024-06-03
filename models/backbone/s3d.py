from torch import nn
import torch.nn.functional as F

from models.backbone.cnn_blocks import (
    SepConv3d, 
    BasicConv3d,
    Mixed_3b, 
    Mixed_3c, 
    Mixed_4b, 
    Mixed_4c, 
    Mixed_4d, 
    Mixed_4e, 
    Mixed_4f,
    Mixed_5b,
    Mixed_5c,
)

BLOCK2SIZE = {
    "block1": 64,
    "block2": 192,
    "block3": 480,
    "block4": 832,
    "block5": 1024,
}

class S3DModule(nn.Module):
    def __init__(self, in_channels, blocks=5):
        super(S3DModule, self).__init__()
        self.backbone = S3DBlock(in_channels=in_channels, blocks=blocks)
        self.out_channels = BLOCK2SIZE[f"block{blocks}"]
        self.stage_idx = [0, 3, 6, 12, 15][:blocks]
        self.blocks = blocks
        self.num_levels = 3
        
    def forward(self, videos):
        B, C, T, H, W = videos.shape
        features = []
        shortcuts = []
        for i, layer in enumerate(self.backbone.base):
            videos = layer(videos)

            for stage in self.stage_idx[self.blocks - self.num_levels:]:
                features.append(videos)
                
        return {'sgn_feature': features[-1], 'fea_lst': features}
    
class S3DBlock(nn.Module):
    def __init__(self, in_channels=3, blocks=5):
        super(S3DBlock, self).__init__()
        self.in_channels = in_channels
        self.base, self.base_num_layers = self.build(blocks)
        
    def build(self, blocks):
        # Define the blocks
        block1 = [
            SepConv3d(self.in_channels, 64, kernel_size=7, stride=2, padding=3)
        ]
        block2 = [
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        ]
        block3 = [
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            Mixed_3b(),
            Mixed_3c(),
        ]
        block4 = [
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
        ]
        block5 = [
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)),
            Mixed_5b(),
            Mixed_5c(),
        ]
        
        # Select n blocks to build the base model
        all_blocks = [block1, block2, block3, block4, block5]
        base = []
        for i in range(blocks):
            base += all_blocks[i]
        
        return nn.Sequential(*base), len(base)

    def forward(self, x):
        x = self.base(x)
        return x