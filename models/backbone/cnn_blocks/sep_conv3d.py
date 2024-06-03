from torch import nn
import torch

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, coord_setting=None):
        super(SepConv3d, self).__init__()
        self.coord_setting = coord_setting
        if coord_setting is None:
            self.conv_s = nn.Conv3d(
                in_planes, out_planes, 
                kernel_size=(1, kernel_size, kernel_size), 
                stride=(1, stride, stride), 
                padding=(0, padding, padding), 
                bias=False
            )
        else:
            self.conv_s = nn.Conv3d(
                in_planes+2, out_planes, 
                kernel_size=(1,kernel_size,kernel_size), 
                stride=(1,stride,stride), 
                padding=(0,padding,padding), 
                bias=False
            )
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()
        
        if coord_setting != 'all':
            self.conv_t = nn.Conv3d(
                out_planes, out_planes, 
                kernel_size=(kernel_size,1,1), 
                stride=(stride,1,1), 
                padding=(padding,0,0), 
                bias=False
            )
        else:
            self.conv_t = nn.Conv3d(
                out_planes+1, out_planes, 
                kernel_size=(kernel_size,1,1), 
                stride=(stride,1,1), 
                padding=(padding,0,0), 
                bias=False
            )
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()
        
    def forward(self, x):
        if self.coord_setting is not None:
            B,C,T,H,W = x.shape
            h_idx, w_idx = torch.arange(H).float().to(x.device)/H, torch.arange(W).float().to(x.device)/W
            h_idx, w_idx = ((h_idx-0.5)/0.5).reshape(1,1,1,H,1).expand(B,1,T,H,W), ((w_idx-0.5)/0.5).reshape(1,1,1,1,W).expand(B,1,T,H,W)
            x = torch.cat([x, h_idx, w_idx], dim=1)
        
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        if self.coord_setting == 'all':
            B,C,T,H,W = x.shape
            t_idx = torch.arange(T).float().to(x.device)
            t_idx = ((t_idx-0.5)/0.5).reshape(1,1,T,1,1).expand(B,1,T,H,W)
            x = torch.cat([x, t_idx], dim=1)
        
        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x