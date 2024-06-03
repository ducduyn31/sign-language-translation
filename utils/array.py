import av
import numpy as np
import torch

def pad_array_np(array, l_and_r):
    left, right = l_and_r
    if left > 0:
        pad_img = array[0]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([left]+[1]*(len(array.shape)-1)))
        array = np.concatenate([pad, array], axis=0)
        
    if right > 0:
        pad_img = array[-1]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([right]+[1]*(len(array.shape)-1)))
        array = np.concatenate([array, pad], axis=0)

    return array

def pad_array(array, l_and_r):
    left, right = l_and_r
    if left > 0:
        pad_img = array[0]
        # pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([left]+[1]*(len(array.shape)-1)))
        pad = pad_img.unsqueeze(0).repeat(left, *([1] * (len(array.shape) - 1)))
        # array = np.concatenate([pad, array], axis=0)
        array = torch.cat([pad, array], dim=0)
        
    if right > 0:
        pad_img = array[-1]
        # pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([right]+[1]*(len(array.shape)-1)))
        pad = pad_img.unsqueeze(0).repeat(right, *([1] * (len(array.shape) - 1)))
        array = torch.cat([array, pad], dim=0)

    return array

def load_frame_nums_to_4darray(video_path, frame_nums):
    container = av.open(video_path)
    
    frames = []
    container.seek(0)
    
    for i, frame in enumerate(container.decode(video=0)):
        if i > frame_nums.max():
            break
        if i in frame_nums:
            frames.append(frame.to_ndarray(format='rgb24'))
    
    return np.stack(frames)
        