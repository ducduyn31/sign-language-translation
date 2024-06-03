import av
from fastapi import UploadFile
import numpy as np
import torch

def get_selected_indices(vlen, num_frames):
    pad = None
    if vlen >= num_frames:
        start = (vlen - num_frames) // 2
        selected_indices = np.arange(start, start + num_frames)
    else:
        remain = num_frames - vlen
        pad_left = remain // 2
        pad_right = remain - pad_left
        pad = (pad_left, pad_right)
        selected_indices = np.arange(0, vlen)
    return selected_indices, pad

# async def read_video(video: UploadFile):
#     # Load video
#     await video.seek(0)
#     format = video.filename.split('.')[-1]
#     container = av.open(video.file, mode="r", format=format)
    
#     # Get frame indices for our chunks
#     vlen = container.streams.video[0].frames
#     num_frames_per_chunk = 64
#     chunks = vlen // num_frames_per_chunk
#     selected_indices, pad = get_selected_indices(vlen, min(num_frames_per_chunk * chunks, num_frames_per_chunk))
    
#     # Get frames and split to chunks
#     frames = []
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > selected_indices.max():
#             break
#         if i in selected_indices:
#             frames.append(frame.to_ndarray(format="rgb24"))
#     frames = torch.from_numpy(np.stack(frames)) # T x H x W x C
    
#     # Pad if necessary
#     if pad is not None:
#         left, right = pad
#         if left > 0:
#             first_frame = frames[0]
#             repeat_dims = len(frames.shape) - 1
#             pad_left = first_frame.repeat(left, *([1] * repeat_dims))
#             frames = torch.cat([pad_left, frames])
#         if right > 0:
#             last_frame = frames[-1]
#             repeat_dims = len(frames.shape) - 1
#             pad_right = last_frame.repeat(right, *([1] * repeat_dims))
#             torch.cat([frames, pad_right])
            
#     # Convert to BGR
#     frames = frames[:, :, :, [2, 1, 0]]
    
#     # Convert to batch of chunks
#     frames = frames.reshape(chunks, num_frames_per_chunk, *frames.shape[1:])
    
#     # Convert to B x T x C x H x W
#     frames = frames.permute(0, 1, 4, 2, 3)
    
#     return frames

async def read_video(video: UploadFile):
    await video.seek(0)
    format = video.filename.split('.')[-1]
    container = av.open(video.file, mode='r', format=format)
    container.seek(0)
    vlen = container.streams.video[0].frames
    num_frames = 64
    
    frames = []
    
    # Select 64 frames
    selected_indices, pad = get_selected_indices(vlen, num_frames)
    for i, frame in enumerate(container.decode(video=0)):
        if i > selected_indices.max():
            break
        if i in selected_indices:
            frames.append(frame.to_ndarray(format='bgr24'))
    frames = torch.from_numpy(np.stack(frames))
    
    # Pad if necessary
    if pad is not None:
        left, right = pad
        if left > 0:
            first_frame = frames[0]
            repeat_dims = len(frames.shape) - 1
            pad = first_frame.unsqueeze(0).repeat(left, *[1]*repeat_dims)
            frames = torch.cat([pad, frames], dim=0)
        if right > 0:
            last_frame = frames[-1]
            repeat_dims = len(frames.shape) - 1
            pad = last_frame.unsqueeze(0).repeat(right, *[1]*repeat_dims)
            frames = torch.cat([frames, pad], dim=0)
            
    # Convert to BGR
    frames = frames[:, :, :, [2,1,0]]
            
    return frames
        