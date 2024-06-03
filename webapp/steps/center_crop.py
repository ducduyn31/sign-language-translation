import torch
import torchvision


def center_crop(frames: torch.Tensor) -> torch.Tensor:
    # Resize and center crop to 256x256
    T, H, W, C = frames.shape
    target_w, target_h = 256, 256
    scale_h, scale_w = target_h / H, target_w / W
    
    # Select the smaller scale
    scale = min(scale_h, scale_w)
    
    # Apply resize and center crop
    spatial_ops = []
    spatial_ops.append(torchvision.transforms.Resize((int(H * scale), int(W * scale))))
    spatial_ops.append(torchvision.transforms.CenterCrop((target_h, target_w)))
    spatial_ops = torchvision.transforms.Compose(spatial_ops)
    
    # Permute for batch processing
    frames = frames.permute(0, 3, 1, 2)
    frames = spatial_ops(frames)
    
    # Permute back to original shape
    frames = frames.permute(0, 2, 3, 1)
    
    return frames

    
    