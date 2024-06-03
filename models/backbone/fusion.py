import torch
from torch import nn

class LateralConnection(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, ratio, direction='rgb2pose', variant=None, interpolate=False, adapt_first=False):
        super(LateralConnection, self).__init__()
        assert direction in ['rgb2pose', 'pose2rgb']
        self.direction = direction
        
        self.variant = variant
        
        padding = tuple(x//2 for x in ksize)
        
        if self.direction == 'rgb2pose':
            self.build_rgb2pose(in_channels, out_channels, ksize, ratio, padding, interpolate, adapt_first)
        elif self.direction == 'pose2rgb':
            self.build_pose2rgb(in_channels, out_channels, ksize, ratio, padding, interpolate, adapt_first)
            
        self.init_weights()
            
                
    def build_rgb2pose(self, in_channels, out_channels, kernel_size, ratio, padding, interpolate, adapt_first):
        if interpolate:
            self.conv = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
            return
        
        if adapt_first:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=ratio, padding=0, bias=False)
            return
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=ratio, padding=padding, bias=False)
        
    def build_pose2rgb(self, in_channels, out_channels, kernel_size, ratio, padding, interpolate, adapt_first):
        if interpolate and adapt_first:
            self.conv = nn.Upsample(size=(8, 7, 7), mode='trilinear')
        elif interpolate:
            self.conv = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        elif adapt_first:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 3, 3), padding=padding, output_padding=0, bias=False)
        else:
            output_padding = (0, 1, 1) if ratio == (1, 2, 2) else (1, 0, 0)
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=ratio, padding=padding, output_padding=output_padding, bias=False)
            
    def init_weights(self):
        if isinstance(self.conv, nn.Conv3d) or isinstance(self.conv, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(self.conv.weight)
        
    def forward(self, x_rgb, x_pose):
        if self.direction == 'rgb2pose':
            x_rgb = self.conv(x_rgb)
        elif self.direction == 'pose2rgb':
            x_pose = self.conv(x_pose)
            
        return None, x_rgb + x_pose