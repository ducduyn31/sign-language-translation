from functools import lru_cache
import numpy as np
import torch
import torchvision
import onnxruntime as ort
import torch.nn.functional as F
from ultralytics import YOLO
from mmpose.apis import inference_topdown, init_model as init_pose_model

@lru_cache(maxsize=1)
def get_detection_model():
    ort_session = ort.InferenceSession("./yolov9c.onnx")
    return ort_session

@lru_cache(maxsize=1)
def get_pose_model():
    ort_session = ort.InferenceSession("./rtmpose_x.onnx")
    return ort_session

@lru_cache(maxsize=1)
def get_torch_detection_model():
    return YOLO('yolov9c.pt')

@lru_cache(maxsize=1)
def get_torch_pose_model():
    return init_pose_model("./configs/rtmpose.py", checkpoint="./ckpts/rtmpose.pth", device="cpu")

# def affine_resize(frames: torch.Tensor, target_size) -> torch.Tensor:
#     frames = frames.float()
#     B, T, C, H, W = frames.shape
#     target_h, target_w = target_size
    
#     scale_h, scale_w = target_h / H, target_w / W
    
#     theta = torch.tensor([
#         [scale_w, 0, 0],
#         [0, scale_h, 0]
#     ], dtype=torch.float32).unsqueeze(0).repeat(B * T, 1, 1)
    
#     grid = F.affine_grid(theta, torch.Size((B * T, C, target_h, target_w)), align_corners=False)
    
#     input_tensor_reshaped = frames.view(-1, C, H, W)
#     output_tensor_reshaped = F.grid_sample(input_tensor_reshaped, grid, align_corners=False)
#     output_tensor = output_tensor_reshaped.view(B, T, C, target_h, target_w)
#     return output_tensor

# def extract_keypoints(frames: torch.Tensor) -> torch.Tensor:
#     _, _, _, orig_h, orig_w = frames.shape
    
#     # Resize to 384x288
#     target_h, target_w = 384, 288
#     resized_frames = affine_resize(frames, (target_h, target_w))
    
#     # Merge for batch processing
#     B, T, C, H, W = resized_frames.shape
#     resized_frames = resized_frames.reshape(B * T, C, H, W)
#     pose_model = get_pose_model()
#     input_name = pose_model.get_inputs()[0].name
#     simcc_x_name = pose_model.get_outputs()[0].name
#     simcc_y_name = pose_model.get_outputs()[1].name
#     result = pose_model.run([simcc_x_name, simcc_y_name], {input_name: resized_frames.numpy()})
    
#     # Convert to x,y,visibility
#     simcc_x, simcc_y = result # T x 133 x 576, T x 133 x 768
#     output_size_x, output_size_y = simcc_x.shape[-1], simcc_y.shape[-1]
#     topk_x = torch.topk(torch.from_numpy(simcc_x).softmax(dim=-1), 1, dim=-1)
#     topk_y = torch.topk(torch.from_numpy(simcc_y).softmax(dim=-1), 1, dim=-1)
    
#     pred_x = topk_x.indices.squeeze(-1).float() # T x 133
#     pred_y = topk_y.indices.squeeze(-1).float() # T x 133
    
#     # Convert x,y to original scale
#     scale_x = orig_h / output_size_x
#     scale_y = orig_w / output_size_y
#     pred_x = pred_x.float() * scale_x
#     pred_y = pred_y.float() * scale_y
    
#     # Get confidence
#     max_prob_x = topk_x.values.squeeze(-1) # T x 133
#     max_prob_y = topk_y.values.squeeze(-1) # T x 133
#     confidence = torch.sqrt(max_prob_x * max_prob_y)
    
#     keypoints = torch.stack([pred_x, pred_y, confidence], dim=-1) # T x 133 x 3
    
#     # Revert to original shape
#     keypoints = keypoints.reshape(B, T, 133, 3)
    
    
#     # Select pose, hand and mouth
#     pose_idx = list(range(11))
#     hand_idx = list(range(91, 133))
#     mouth_half_idx = list(range(71, 91, 2))
    
#     pose = keypoints[:, :, pose_idx, :]
#     hand = keypoints[:, :, hand_idx, :]
#     mouth_half = keypoints[:, :, mouth_half_idx, :]
    
#     keypoints = torch.cat([hand, mouth_half, pose], axis=2)  # B x T x 63 x 3
    
#     return keypoints
    
def extract_keypoints(sgn_video):
    sgn_video = sgn_video.cpu().numpy()
    sgn_keypoints = []
    detection_model = get_torch_detection_model()
    pose_model = get_torch_pose_model()
    
    det_results = detection_model(list(sgn_video), conf=.75, classes=[0])

    for i, (frame, det_result) in enumerate(zip(sgn_video, det_results)):
        d = [x for x in det_result.boxes.xyxy.cpu().numpy()]
        result = inference_topdown(
            pose_model,
            frame,
            bboxes=d,
            bbox_format='xyxy',
        )
        # if len(result) == 0:
            # continue
        # sgn_keypoints.append(result[0].pred_instances.keypoints)
        all_keypoints = result[0].pred_instances.keypoints # 1x113x2
        visibles = result[0].pred_instances.keypoints_visible # 1x113
        pose_idx = list(range(11))
        hand_idx = list(range(91, 133))
        mouth_half_idx = list(range(71, 91, 2))
        pose = all_keypoints[:, pose_idx, :]
        pose_visible = np.expand_dims(visibles[:, pose_idx], axis=-1)
        pose_cat = np.concatenate([pose, pose_visible], axis=-1)
        
        hand = all_keypoints[:, hand_idx, :]
        hand_visible = np.expand_dims(visibles[:, hand_idx], axis=-1)
        hand_cat = np.concatenate([hand, hand_visible], axis=-1)
        
        mouth_half = all_keypoints[:, mouth_half_idx, :]
        mouth_half_visible = np.expand_dims(visibles[:, mouth_half_idx], axis=-1)
        mouth_half_cat = np.concatenate([mouth_half, mouth_half_visible], axis=-1)
        
        keypoints = np.concatenate([hand_cat, mouth_half_cat, pose_cat], axis=1) # 1x63x3
        sgn_keypoints.append(keypoints)
        
    return torch.from_numpy(np.stack(sgn_keypoints)).squeeze(dim=1)    
    