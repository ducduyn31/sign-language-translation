import math
import torch
import torchvision


# def gen_gaussian_hmap_op(coords):
#     map_h, map_w = (256, 256)
#     sigma = 8
    
#     T, hmap_num = coords.shape[:2]
#     factor_h, factor_w = 1, 1
#     coords_y = coords[..., 1] * factor_h
#     coords_x = coords[..., 0] * factor_w
#     confs = coords[..., 2]
    
#     y, x = torch.meshgrid(torch.arange(map_h), torch.arange(map_w))
#     coords = torch.stack([coords_y, coords_x], dim=0)
#     grid = torch.stack([y, x], dim=0).to(coords.device)
#     grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num, T, -1, -1, -1)
#     coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)
#     hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))
#     hmap = hmap.permute(1,0,2,3)
    
#     return hmap

# def generate_batch_heatmap(keypoints):
#     B, T, N, D = keypoints.shape
#     keypoints = keypoints.reshape(-1, N, D)
#     chunk_size = int(math.ceil(( B * T) / 16))
#     chunks = torch.split(keypoints, chunk_size, dim=0)
    
#     heatmaps = []
    
#     for chunk in chunks:
#         hm = gen_gaussian_hmap_op(chunk)
#         N, H, W = hm.shape[-3:]
#         heatmaps.append(hm)
        
#     heatmaps = torch.cat(heatmaps, dim=0)
#     return heatmaps.reshape(B, T, N, H, W)
        
# def generate_low_res(video, keypoints):
#     num_frames = 64
#     start = num_frames // 4
#     end = start + num_frames // 2
    
#     video_low = video[:, start:end, :, :, :]
#     keypoints_low = keypoints[:, start:end, :, :]
    
#     return video_low, keypoints_low

# def apply_spatial_ops(x, spatial_ops_func):
#     ndim = x.ndim
#     if ndim > 4:
#         B, T, C, H, W = x.shape
#         x = x.reshape(-1, C, H, W)
#     chunks = torch.split(x, 16, dim=0)
#     transformed_x = []
#     for chunk in chunks:
#         transformed_chunk = spatial_ops_func(chunk)
#         transformed_x.append(transformed_chunk)
#     _, C, H, W = transformed_x[-1].shape
#     transformed_x = torch.cat(transformed_x, dim=0)
#     if ndim > 4:
#         transformed_x = transformed_x.reshape(B, T, C, H, W)
#     return transformed_x

# def preprocess(sgn_video, sgn_keypoints):
#     sgn_video = sgn_video.float() / 255 # B x T x C x H x W
#     sgn_video_low, sgn_keypoints_low = generate_low_res(sgn_video, sgn_keypoints)
    
#     heatmap = generate_batch_heatmap(sgn_keypoints)
#     heatmap_low = generate_batch_heatmap(sgn_keypoints_low)
    
#     rgb_h, rgb_w = 224, 224
#     hm_h, hm_w = 112, 112
    
#     spatial_ops = []
#     spatial_ops.append(torchvision.transforms.Resize((rgb_h, rgb_w)))
#     spatial_ops = torchvision.transforms.Compose(spatial_ops)
#     sgn_video = apply_spatial_ops(sgn_video, spatial_ops)
#     sgn_video_low = apply_spatial_ops(sgn_video_low, spatial_ops)
    
#     spatial_ops = []
#     spatial_ops.append(torchvision.transforms.Resize((hm_h, hm_w)))
#     spatial_ops = torchvision.transforms.Compose(spatial_ops)
#     heatmap = apply_spatial_ops(heatmap, spatial_ops)
#     heatmap_low = apply_spatial_ops(heatmap_low, spatial_ops)
    
#     # Convert to BGR
#     sgn_video = sgn_video[:, :, [2,1,0], :, :]
#     sgn_video_low = sgn_video_low[:, :, [2,1,0], :, :]
    
#     # Normalize
#     sgn_video = ((sgn_video - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float() # B x C x T x H x W
#     sgn_video_low = ((sgn_video_low - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float() # B x C x T x H x W
#     heatmap = ((heatmap - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float()
#     heatmap_low = ((heatmap_low - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float()
    
#     return sgn_video, sgn_video_low, heatmap, heatmap_low

def gen_gaussian_hmap_op(coords):
    map_h, map_w = (256, 256)
    sigma = 8
    
    T, hmap_num = coords.shape[:2]
    factor_h, factor_w = 1, 1
    coords_y = coords[..., 1] * factor_h
    coords_x = coords[..., 0] * factor_w
    confs = coords[..., 2]
    
    y, x = torch.meshgrid(torch.arange(map_h), torch.arange(map_w))
    coords = torch.stack([coords_y, coords_x], dim=0)
    grid = torch.stack([y, x], dim=0).to(coords.device)
    grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num, T, -1, -1, -1)
    coords = coords.unsqueeze(0).unsqueeze(0).expand(map_h, map_w,-1,-1,-1).permute(4,3,2,0,1)
    hmap = torch.exp(-((grid-coords)**2).sum(dim=2) / (2*sigma**2))
    hmap = hmap.permute(1,0,2,3)
    
    return hmap

def generate_batch_heatmap(keypoints):
    B, T, N, D = keypoints.shape
    keypoints = keypoints.reshape(-1, N, D)
    chunk_size = int(math.ceil(( B * T) / 16))
    chunks = torch.split(keypoints, chunk_size, dim=0)
    
    heatmaps = []
    
    for chunk in chunks:
        hm = gen_gaussian_hmap_op(chunk)
        N, H, W = hm.shape[-3:]
        heatmaps.append(hm)
        
    heatmaps = torch.cat(heatmaps, dim=0)
    return heatmaps.reshape(B, T, N, H, W)
        
def generate_low_res(video, keypoints):
    num_frames = 64
    start = num_frames // 4
    end = start + num_frames // 2
    
    video_low = video[:, start:end, :, :, :]
    keypoints_low = keypoints[:, start:end, :, :]
    
    return video_low, keypoints_low

def apply_spatial_ops(x, spatial_ops_func):
    ndim = x.ndim
    if ndim > 4:
        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
    chunks = torch.split(x, 16, dim=0)
    transformed_x = []
    for chunk in chunks:
        transformed_chunk = spatial_ops_func(chunk)
        transformed_x.append(transformed_chunk)
    _, C, H, W = transformed_x[-1].shape
    transformed_x = torch.cat(transformed_x, dim=0)
    if ndim > 4:
        transformed_x = transformed_x.reshape(B, T, C, H, W)
    return transformed_x

def preprocess(sgn_video, sgn_keypoints):
    sgn_video = sgn_video.unsqueeze(0).permute(0, 1, 4, 2, 3).float() / 255 # B x T x C x H x W
    sgn_keypoints = sgn_keypoints.unsqueeze(0) # B x T x N x 3
    sgn_video_low, sgn_keypoints_low = generate_low_res(sgn_video, sgn_keypoints)
    
    heatmap = generate_batch_heatmap(sgn_keypoints)
    heatmap_low = generate_batch_heatmap(sgn_keypoints_low)
    
    rgb_h, rgb_w = 224, 224
    hm_h, hm_w = 112, 112
    
    spatial_ops = []
    spatial_ops.append(torchvision.transforms.Resize((rgb_h, rgb_w)))
    spatial_ops = torchvision.transforms.Compose(spatial_ops)
    sgn_video = apply_spatial_ops(sgn_video, spatial_ops)
    sgn_video_low = apply_spatial_ops(sgn_video_low, spatial_ops)
    
    spatial_ops = []
    spatial_ops.append(torchvision.transforms.Resize((hm_h, hm_w)))
    spatial_ops = torchvision.transforms.Compose(spatial_ops)
    heatmap = apply_spatial_ops(heatmap, spatial_ops)
    heatmap_low = apply_spatial_ops(heatmap_low, spatial_ops)
    
    # Convert to BGR
    sgn_video = sgn_video[:, :, [2,1,0], :, :]
    sgn_video_low = sgn_video_low[:, :, [2,1,0], :, :]
    
    # Normalize
    sgn_video = ((sgn_video - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float() # B x C x T x H x W
    sgn_video_low = ((sgn_video_low - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float() # B x C x T x H x W
    heatmap = ((heatmap - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float()
    heatmap_low = ((heatmap_low - 0.5) / 0.5).permute(0, 2, 1, 3, 4).float()
    
    return sgn_video, sgn_video_low, heatmap, heatmap_low
