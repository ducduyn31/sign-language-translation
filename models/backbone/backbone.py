from torch import nn

from models.backbone.fusion import LateralConnection
from models.backbone.two_stream import TwoStream

import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.ts_high = TwoStream(
            blocks=5, 
            pose_in_channels=63, 
            laterals=[True, True, True, True, True, True],
            fusion_features=['c1', 'c2', 'c3', 'c4'],
        )
        self.ts_low = TwoStream(
            blocks=5, 
            pose_in_channels=63, 
            laterals=[True, True, True, True, True, True],
            fusion_features=['c1', 'c2', 'c3', 'c4'],
        )
        
        inout_channels = [(64,64), (192,192), (480,480), (832,832)]
        self.laterals = [True, True, True, True, True, True]
        
        self.rgb_low2high = self._build_lateral(
            inout_channels=inout_channels, 
            ksize=(3, 1, 1), 
            ratio=(2, 1, 1), 
            direction='pose2rgb',
            variant=None,
            interpolate=False,
            adapt_first=False,
        )
        self.rgb_high2low = self._build_lateral(
            inout_channels=inout_channels, 
            ksize=(3, 1, 1), 
            ratio=(2, 1, 1), 
            direction='rgb2pose',
            variant=None,
            interpolate=False,
            adapt_first=False,
        )
        self.pose_low2high = self._build_lateral(
            inout_channels=inout_channels, 
            ksize=(3, 1, 1), 
            ratio=(2, 1, 1), 
            direction='pose2rgb',
            variant=None,
            interpolate=False,
            adapt_first=False,
        )
        self.pose_high2low = self._build_lateral(
            inout_channels=inout_channels, 
            ksize=(3, 1, 1), 
            ratio=(2, 1, 1), 
            direction='rgb2pose',
            variant=None,
            interpolate=False,
            adapt_first=False,
        )
        
        
    def _build_lateral(self, inout_channels, ksize, ratio, direction, variant, interpolate, adapt_first):
        return nn.ModuleList([
            LateralConnection(
                in_channels=in_channels, 
                out_channels=out_channels, 
                ksize=ksize, 
                ratio=ratio, 
                direction=direction, 
                variant=variant, 
                interpolate=interpolate, 
                adapt_first=adapt_first
            ) for in_channels, out_channels in inout_channels
        ])
        
    def forward(self, x_rgb_high, x_rgb_low, x_pose_high, x_pose_low):
        B, C, T, H, W = x_rgb_high.shape
        
        for i, (
            rgb_high_layer, 
            pose_high_layer, 
            rgb_low_layer, 
            pose_low_layer,
        ) in enumerate(zip(
            self.ts_high.rgb_stream.backbone.base, 
            self.ts_high.pose_stream.backbone.base, 
            self.ts_low.rgb_stream.backbone.base, 
            self.ts_low.pose_stream.backbone.base,
        )):
            # Advance
            x_rgb_high = rgb_high_layer(x_rgb_high)
            x_pose_high = pose_high_layer(x_pose_high)
            x_rgb_low = rgb_low_layer(x_rgb_low)
            x_pose_low = pose_low_layer(x_pose_low)
            
            # Fuse
            if i in self.ts_high.fuse_idx:
                x_rgb_high, x_pose_high = self._fuse_if_possible(i, 'high_intra', x_rgb_high, x_pose_high)
                x_rgb_low, x_pose_low = self._fuse_if_possible(i, 'low_intra', x_rgb_low, x_pose_low)
                x_rgb_low, x_rgb_high = self._fuse_if_possible(i, 'rgb_inter', x_rgb_high, x_rgb_low)
                x_pose_low, x_pose_high = self._fuse_if_possible(i, 'pose_inter', x_pose_high, x_pose_low)
                
        # Average
        x_rgb_high = self._global_pool(x_rgb_high)
        x_rgb_low = self._global_pool(x_rgb_low)
        x_pose_high = self._global_pool(x_pose_high)
        x_pose_low = self._global_pool(x_pose_low)
            
        return {'rgb-h': x_rgb_high, 'rgb-l': x_rgb_low, 'kp-h': x_pose_high, 'kp-l': x_pose_low}
            
    
    def _fuse_if_possible(self, layer_idx, stream, x1, x2):
        sources = {
            'high_intra': [0, 1], 
            'low_intra': [0, 1], 
            'rgb_inter': [2, 3], 
            'pose_inter': [4, 5],
        }
        layers = {
            'high_intra': [
                self.ts_high.rgb_stream_lateral[self.ts_high.fuse_idx.index(layer_idx)], 
                self.ts_high.pose_stream_lateral[self.ts_high.fuse_idx.index(layer_idx)],
            ], 
            'low_intra': [
                self.ts_low.rgb_stream_lateral[self.ts_low.fuse_idx.index(layer_idx)], 
                self.ts_low.pose_stream_lateral[self.ts_low.fuse_idx.index(layer_idx)],
            ], 
            'rgb_inter': [
                self.rgb_high2low[self.ts_high.fuse_idx.index(layer_idx)], 
                self.rgb_low2high[self.ts_high.fuse_idx.index(layer_idx)],
            ], 
            'pose_inter': [
                self.pose_high2low[self.ts_high.fuse_idx.index(layer_idx)], 
                self.pose_low2high[self.ts_high.fuse_idx.index(layer_idx)],
            ]
        }
        if not all([self.laterals[i] for i in sources[stream]]):
            return x1, x2
        
        _, x1_fused = layers[stream][0](x1, x2)
        _, x2_fused = layers[stream][1](x1, x2)
        
        return x1_fused, x2_fused
        
    def _global_pool(self, x):
        H, W = x.shape[-2:]
        
        try:
            x = F.avg_pool3d(x, kernel_size=(2, H, W), stride=1)
        except:
            x = F.avg_pool3d(x, kernel_size=(1, H, W), stride=1)
            
        B, C, T = x.shape[:3]
        return x.view(B, C, T).permute(0, 2, 1).mean(dim=1)