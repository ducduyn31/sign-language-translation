from torch import nn

from models.backbone.fusion import LateralConnection
from models.backbone.s3d import S3DModule

import torch.nn.functional as F

class TwoStream(nn.Module):
    def __init__(
        self, 
        blocks=5, pose_in_channels=17, 
        laterals=[False, False],
        lateral_variant = [None, None],
        lateral_ksize = (1, 3, 3),
        lateral_ratio = (1, 2, 2),
        lateral_interpolate = False,
        fusion_features=['c1', 'c2', 'c3'],
    ):
        super(TwoStream, self).__init__()
        self.rgb_stream = S3DModule(in_channels=3, blocks=blocks)
        self.pose_stream = S3DModule(in_channels=pose_in_channels, blocks=blocks)
        self.blocks = blocks
        self.laterals = laterals

        self.fusion_features = fusion_features
        
        self.init_levels = 4
        self.block_idx = [0, 3, 6, 12, 15]
        
        self.fuse_idx = [0, 3, 6, 12]
        inout_channels = [(64,64), (192,192), (480,480), (832,832)]
        
        if self.laterals[0]:
            # pose2rgb
            self.rgb_stream_lateral = self._build_lateral(
                inout_channels=inout_channels, 
                ksize=lateral_ksize, 
                ratio=lateral_ratio, 
                direction='pose2rgb',
                variant=lateral_variant[0],
                interpolate=lateral_interpolate
            )
            
        if self.laterals[1]:
            # rgb2pose
            self.pose_stream_lateral = self._build_lateral(
                inout_channels=inout_channels, 
                ksize=lateral_ksize, 
                ratio=lateral_ratio, 
                direction='rgb2pose',
                variant=lateral_variant[1],
                interpolate=lateral_interpolate
            )
            
    def _build_lateral(self, inout_channels, ksize, ratio, direction, variant, interpolate):
        def should_adapt_first(layer_idx):
            # Only apply to last layer
            if layer_idx < len(inout_channels) - 1:
                return False
            if 'c5' not in self.fusion_features:
                return False
            
            return True
            
        return nn.ModuleList([
            LateralConnection(
                in_channels=in_channels, 
                out_channels=out_channels, 
                ksize=ksize, 
                ratio=ratio, 
                direction=direction,
                variant=variant,
                interpolate=interpolate,
                adapt_first=should_adapt_first(i)
            )
            for i, (in_channels, out_channels) in enumerate(inout_channels)
        ])
        
    def forward(self, x_rgb, x_pose):
        B, C, T, H, W = x_rgb.shape
        rgb_features, pose_features = [], []
        
        def should_fuse(direction, layer_idx):
            allow_fuse = self.laterals[0] if direction == 'rgb2pose' else self.laterals[1]
            can_fuse = 'p' not in self.fusion_features[self.fuse_idx.index(layer_idx)]
            
            return allow_fuse and can_fuse
        
        for i, (rgb_layer, pose_layer) in enumerate(zip(self.rgb_stream.backbone.base, self.pose_stream.backbone.base)):
            # Advance to next layer
            x_rgb = rgb_layer(x_rgb)
            x_pose = pose_layer(x_pose)
            
            # Fuse betwen 2 streams
            if i in self.fuse_idx:
                x_rgb_fused = x_rgb
                x_pose_fused = x_pose
                
                if should_fuse('rgb2pose', i):
                    _, x_rgb_fused = self.rgb_stream_lateral[self.fuse_idx.index(i)](x_rgb, x_pose)
                    
                if should_fuse('pose2rgb', i):
                    _, x_pose_fused = self.pose_stream_lateral[self.fuse_idx.index(i)](x_rgb, x_pose)
                    
                x_rgb = x_rgb_fused
                x_pose = x_pose_fused
                
            if i in self.block_idx[:self.blocks]:
                rgb_features.append(x_rgb)
                pose_features.append(x_pose)
                
        rgb_fused = pose_fused = None
        diff = 1
        
        for i in range(len(rgb_features)):
            rgb_features[i] = self._global_pool(rgb_features[i])
            pose_features[i] = self._global_pool(pose_features[i])
            
        return {'rgb_fused': rgb_fused, 'pose_fused': pose_fused, 'rgb_fea_lst': rgb_features[diff:], 'pose_fea_lst': pose_features[diff:]}
        
    def _global_pool(self, feature):
        H, W = feature.shape[-2:]
        
        try:
            feature = F.avg_pool3d(feature, kernel_size=(2, H, W), stride=1)
        except:
            feature = F.avg_pool3d(feature, kernel_size=(1, H, W), stride=1, padding=1)
        
        B, C, T = feature.shape[:3]
        feature = feature.view(B, C, T).permute(0, 2, 1)
        return feature