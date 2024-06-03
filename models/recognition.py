import torch
import math
import torchvision
import random
import os

import lightning as L
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

from models.backbone.backbone import Backbone
from models.head import SepConvHead
from models.utils.label_smooth import LabelSmoothCE
from utils.array import pad_array
from utils.gen_gaussian import gen_gaussian_hmap_op
from mmpose.apis import inference_topdown, init_model as init_pose_model

class SignRecognitionNetwork(L.LightningModule):
    def __init__(self, word_embedding, pose_mode_config):
        super().__init__()
        self.preprocess_chunksize = 16
        self.contras_loss_weight = 1.0
        self.embedding = word_embedding.clone()
        self.backbone = Backbone()
        self.visual_head_rgb_h = SepConvHead(word_embedding=word_embedding, topk=2000, input_size=1024)
        self.visual_head_rgb_l = SepConvHead(word_embedding=word_embedding, topk=2000, input_size=1024)
        self.visual_head_kp_h = SepConvHead(word_embedding=word_embedding, topk=2000, input_size=1024)
        self.visual_head_kp_l = SepConvHead(word_embedding=word_embedding, topk=2000, input_size=1024)
        self.visual_head_fuse = SepConvHead(word_embedding=word_embedding, topk=2000, input_size=4096)
        self.visual_head_fuse_h = SepConvHead(word_embedding=word_embedding, topk=2000, input_size=2048)
        self.visual_head_fuse_l = SepConvHead(word_embedding=word_embedding, topk=2000, input_size=2048)
        self.head_dict = {
            'rgb-h': self.visual_head_rgb_h, 
            'rgb-l': self.visual_head_rgb_l,
            'kp-h': self.visual_head_kp_h, 
            'kp-l': self.visual_head_kp_l,
            'fuse': self.visual_head_fuse, 
            'fuse-h': self.visual_head_fuse_h, 
            'fuse-l': self.visual_head_fuse_l, 
        }
        self.recognition_loss_func = LabelSmoothCE(
            epsilon=0.2, 
            reduction='mean',
            variant='word_sim',
            norm_type='softmax',
            temp=0.5,
            word_embedding=word_embedding
        )
        self.contras_loss_func = LabelSmoothCE(
            reduction='mean',
            variant='dual_ema_cosine',
        )
        self.pose_model = self.configure_pose_estimator(
            config_path=pose_mode_config["config_path"],
            weights_path=pose_mode_config["weights_path"],
        )
        self.pose_model_config = pose_mode_config
        
        self.train_losses = defaultdict(float)

        self.val_results = defaultdict(dict)
        self.val_losses = defaultdict(float)
        self.val_name_prob = {}
        self.val_logits_names = []
        
        self.test_results = defaultdict(dict)
        self.test_losses = defaultdict(float)
        self.test_name_prob = {}
        self.test_logits_names = []
        
    def configure_pose_estimator(self, config_path, weights_path):
        local_rank = os.getenv('LOCAL_RANK', 0)
        pose_model = init_pose_model(
            config=config_path,
            checkpoint=weights_path,
            device=f"cuda:{local_rank}",
        )
        for param in pose_model.parameters():
            param.requires_grad = False
        
        return pose_model
    
    def move_to_device(self, model, device):
        model = model.to(device)
        return model
        
    def load_from_pretrained(self, pretrained):
        model_dict = self.state_dict()
        tmp = {}
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
        for k, v in pretrained.items():
            if k in model_dict and model_dict[k].shape==v.shape:
                tmp[k] = v
            else:
                print(k)
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if 'pose_model' in k:
                continue
            if k not in pretrained:
                print(k)
            elif model_dict[k].shape != pretrained[k].shape:
                print(k, 'shape mis-matched, not loaded')
        print('===================================\n')

        del pretrained
        model_dict.update(tmp)
        del tmp
        self.load_state_dict(model_dict, strict=False)

    def forward(self, labels, sgn_videos, sgn_keypoints, sgn_videos_low, sgn_keypoints_low):
        # Implement your forward logic here
        s3d_outputs = []
        # self.embedding = self.embedding.to(sgn_videos.device)
        # sgn_keypoints = sgn_keypoints.to(sgn_videos.device)
        # sgn_keypoints_low = sgn_keypoints_low.to(sgn_videos.device)
        
        with torch.no_grad():   
            # Generate heatmaps
            sgn_heatmaps = self.generate_batch_heatmap(sgn_keypoints)
            sgn_heatmaps_low = self.generate_batch_heatmap(sgn_keypoints_low)
            
            # Preprocess inputs
            sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low = self.augment_preprocess_inputs(
                sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
            )
            
            # Mixup
            sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low, y_a, y_b, lam, index, do_mixup = self.mixup(
                a=sgn_videos, b=sgn_heatmaps, labels=labels, c=sgn_videos_low, d=sgn_heatmaps_low
            )
            
        # Backbone
        s3d_outputs = self.backbone(sgn_videos, sgn_videos_low, sgn_heatmaps, sgn_heatmaps_low)
        
        outputs = {}
        keys_of_int = ['gloss_logits', 'word_fused_gloss_logits', 'topk_idx']
        for head_name, features in s3d_outputs.items():
            head_ops = self.head_dict[head_name](x=features, labels=labels)
            for k in keys_of_int:
                outputs[f'{head_name}_{k}'] = head_ops[k]
                
        # Fuse heads
        effect_heads = ['rgb-h', 'kp-h', 'rgb-l', 'kp-l']
        for head_name, head in self.head_dict.items():
            if 'fuse' not in head_name:
                continue
            effect_heads.append(head_name)
            if head_name == 'fuse':
                fused_features = torch.cat([s3d_outputs['rgb-h'], s3d_outputs['rgb-l'], s3d_outputs['kp-h'], s3d_outputs['kp-l']], dim=-1)
                head_ops = head(x=fused_features, labels=labels)
            elif head_name == 'fuse-h':
                fused_features = torch.cat([s3d_outputs['rgb-h'], s3d_outputs['kp-h']], dim=-1)
                head_ops = head(x=fused_features, labels=labels)
            elif head_name == 'fuse-l':
                fused_features = torch.cat([s3d_outputs['rgb-l'], s3d_outputs['kp-l']], dim=-1)
                head_ops = head(x=fused_features, labels=labels)
                
            for k in keys_of_int:
                outputs[f'{head_name}_{k}'] = head_ops[k]
        del head_ops
        
        for head_name, head in self.head_dict.items():
            outputs['ensemble_all_gloss_logits'] = outputs.get('ensemble_all_gloss_logits', torch.zeros_like(outputs['rgb-h_gloss_logits']))\
                + outputs[f'{head_name}_gloss_logits'].softmax(dim=-1)
            if 'fuse' in head_name:
                outputs['ensemble_last_gloss_logits'] = outputs.get('ensemble_last_gloss_logits', torch.zeros_like(outputs['rgb-h_gloss_logits']))\
                    + outputs[f'{head_name}_gloss_logits'].softmax(dim=-1)
        
        outputs['ensemble_all_gloss_logits'] = outputs['ensemble_all_gloss_logits'].log()
        outputs['ensemble_last_gloss_logits'] = outputs['ensemble_last_gloss_logits'].log()
        
        # Compute losses
        keys = effect_heads
        head_outputs = outputs
        for k in keys:
            if f'{k}_gloss_logits' in head_outputs:
                outputs[f'recognition_loss_{k}'] = lam * self.recognition_loss_func(
                    logits=head_outputs[f'{k}_gloss_logits'],
                    label=y_a,
                    # head_name=k,
                ) + (1. - lam) * self.recognition_loss_func(
                    logits=head_outputs[f'{k}_gloss_logits'],
                    label=y_b,
                    # head_name=k,
                )
        for k in effect_heads:
            outputs['recognition_loss'] = outputs.get('recognition_loss', torch.tensor(0.0).to(sgn_videos.device)) + outputs[f'recognition_loss_{k}']
            
        outputs['total_loss'] = outputs['recognition_loss']
        
        keys = [f'{e}_' for e in effect_heads]
        for k in keys:
            contras_loss = torch.tensor(0.0).to(sgn_videos.device)
            word_fused_gloss_logits = outputs[f'{k}word_fused_gloss_logits']
            if word_fused_gloss_logits is None:
                continue
            topk_idx = outputs[f'{k}topk_idx']
            contras_loss = self.contras_loss_func(
                logits=word_fused_gloss_logits,
                label=labels,
                topk_idx=topk_idx,
                mixup_lam=lam,
                y_a=y_a,
                y_b=y_b
            )
            
            outputs[f'contras_loss_{k}'] = contras_loss
            epoch = self.current_epoch or 0
            cur_loss_weight = 0.5 * self.contras_loss_weight * (1.0 + math.cos(math.pi * epoch / 100))
            outputs['total_loss'] = outputs['total_loss'] + cur_loss_weight * contras_loss
            
        return outputs
            
    def generate_batch_heatmap(self, keypoints):
        B, T, N, D = keypoints.shape
        keypoints = keypoints.reshape(-1, N, D)
        chunk_size = int(math.ceil( B * T / self.preprocess_chunksize ))
        chunks = torch.split(keypoints, chunk_size, dim=0)
        
        heatmaps = []
        for chunk in chunks:
            hm = gen_gaussian_hmap_op(
                coords=chunk,
                sigma=8,
                raw_size=(256, 256),
                input_size=112,
            )
            N, H, W = hm.shape[-3:]
            heatmaps.append(hm)
            
        heatmaps = torch.cat(heatmaps, dim=0)
        return heatmaps.reshape(B, T, N, H, W)
    
    def augment_preprocess_inputs(self, sgn_videos=None, sgn_heatmaps=None, sgn_videos_low=None, sgn_heatmaps_low=None):
        if self.training:
            sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low = self.augment_for_training(
                sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
            )
        else:
            sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low = self.preprocess_for_inference(
                sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
            )
        
        # Convert to BGR
        sgn_videos = sgn_videos[:, :, [2, 1, 0], :, :]
        sgn_videos = sgn_videos.float()
        sgn_videos = (sgn_videos - 0.5) / 0.5
        sgn_videos = sgn_videos.permute(0, 2, 1, 3, 4).float() # B C T H W
        
        sgn_videos_low = sgn_videos_low[:, :, [2, 1, 0], :, :]
        sgn_videos_low = sgn_videos_low.float()
        sgn_videos_low = (sgn_videos_low - 0.5) / 0.5
        sgn_videos_low = sgn_videos_low.permute(0, 2, 1, 3, 4).float() # B C T H W
        
        sgn_heatmaps = (sgn_heatmaps - 0.5) / 0.5
        sgn_heatmaps = sgn_heatmaps.permute(0, 2, 1, 3, 4).float()
        
        sgn_heatmaps_low = (sgn_heatmaps_low - 0.5) / 0.5
        sgn_heatmaps_low = sgn_heatmaps_low.permute(0, 2, 1, 3, 4).float()
        
        # sgn_heatmaps = sgn_heatmaps.to(sgn_videos.device)
        # sgn_heatmaps_low = sgn_heatmaps_low.to(sgn_videos.device)
        
        return sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
          
    def augment_for_training(self, sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low):
        rgb_h, rgb_w = 224, 224
        hm_h, hm_w = 112, 112
        
        # Color Jitter
        collor_jitter_op = torchvision.transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        if random.random() < 0.3:
            sgn_videos = collor_jitter_op(sgn_videos)
            sgn_videos_low = collor_jitter_op(sgn_videos_low)
        
        # Random Resize Crop
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
            img=sgn_videos, scale=(0.7, 1.0), ratio=(.75, 1.3)
        )
        sgn_videos = self.apply_spatial_ops(
            sgn_videos,
            spatial_ops_func=lambda x: torchvision.transforms.functional.resized_crop(
                x, i, j, h, w, [rgb_h, rgb_w]
            )
        )
        sgn_videos_low = self.apply_spatial_ops(
            sgn_videos_low,
            spatial_ops_func=lambda x: torchvision.transforms.functional.resized_crop(
                x, i, j, h, w, [rgb_h, rgb_w]
            )
        )

        # Random Resize Crop for Heatmap
        i2, j2, h2, w2 = torchvision.transforms.RandomResizedCrop.get_params(
            img=sgn_heatmaps, scale=(0.7, 1.0), ratio=(.75, 1.3)
        ) 
        sgn_heatmaps = self.apply_spatial_ops(
            sgn_heatmaps,
            spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                x, i2, j2, h2, w2, [hm_h, hm_w]
            )
        )
        sgn_heatmaps_low = self.apply_spatial_ops(
            sgn_heatmaps_low,
            spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                x, i2, j2, h2, w2, [hm_h, hm_w]
            )
        )
        
        return sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
       
    def preprocess_for_inference(self, sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low):
        rgb_h, rgb_w = 224, 224
        hm_h, hm_w = 112, 112

        # Resize sign videos
        spatial_ops = []
        spatial_ops.append(torchvision.transforms.Resize([rgb_h, rgb_w]))
        spatial_ops = torchvision.transforms.Compose(spatial_ops)
        
        sgn_videos = self.apply_spatial_ops(sgn_videos, spatial_ops)
        sgn_videos_low = self.apply_spatial_ops(sgn_videos_low, spatial_ops)
        
        # Resize sign heatmaps
        spatial_ops = []
        spatial_ops.append(torchvision.transforms.Resize([hm_h, hm_w]))
        spatial_ops = torchvision.transforms.Compose(spatial_ops)
        
        sgn_heatmaps = self.apply_spatial_ops(sgn_heatmaps, spatial_ops)
        sgn_heatmaps_low = self.apply_spatial_ops(sgn_heatmaps_low, spatial_ops)
        
        return sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
          
    def apply_spatial_ops(self, x, spatial_ops_func):
        ndim = x.ndim
        if ndim > 4:
            B, T, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        chunks = torch.split(x, self.preprocess_chunksize, dim=0)
        transformed_x = []
        for chunk in chunks:
            transformed_x.append(spatial_ops_func(chunk))
        _, C, H, W = transformed_x[-1].shape
        transformed_x = torch.cat(transformed_x, dim=0)
        if ndim > 4:
            transformed_x = transformed_x.reshape(B, T, C, H, W)
        return transformed_x
            
    def mixup(self, a, b, labels, c, d):
        mix_a, mix_b, mix_c, mix_d = a, b, c, d
        y_a = y_b = labels
        lam = 0
        index = torch.arange(labels.shape[0])
        do_mixup = False
        prob, alpha = 0.75, 0.8
        
        if self.training and random.random() < prob:
            do_mixup = True
            lam = np.random.beta(alpha, alpha)
            batch_size = a.shape[0]
            index = torch.randperm(labels.shape[0])
            # index =index.to(a.device)
            
            mix_a = lam * a + (1. - lam) * a[index]
            mix_b = lam * b + (1. - lam) * b[index]
            mix_c = lam * c + (1. - lam) * c[index]
            mix_d = lam * d + (1. - lam) * d[index]
            
            y_a, y_b = labels, labels[index]
            
        return mix_a, mix_b, mix_c, mix_d, y_a, y_b, lam, index, do_mixup
        
    def extract_keypoints(self, sgn_video):
        # config_path, weights_path = self.pose_model_config["config_path"], self.pose_model_config["weights_path"]
        # det_model = YOLO("yolov9c.pt")
        # pose_model = init_pose_model(
        #     config=config_path,
        #     checkpoint=weights_path,
        # )
        sgn_keypoints = []
        # det_results = det_model(list(sgn_video), conf=.75, classes=[0])
        
        # for i, (frame, det_result) in enumerate(zip(sgn_video, det_results)):
        for frame in sgn_video:
            # d = [x for x in det_result.boxes.xyxy.cpu().numpy()]
            result = inference_topdown(
                self.pose_model,
                frame.cpu().numpy(),
                # bboxes=d,
                # bbox_format='xyxy',
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
            
        return torch.from_numpy(np.stack(sgn_keypoints)).squeeze(dim=1).to(sgn_video.device)
    
    def decode(self, logits, k=5):
        result = torch.argsort(logits, dim=-1, descending=True)
        result = result[:, :k]
        return result
    
    def _prepare_inputs(self, batched_sgn_videos, pads):
        batched_sgn_keypoints = []
        for i, sgn_video in enumerate(batched_sgn_videos):
            if not pads[i]:
                orig_video = sgn_video
            else:
                pad_left, pad_right = pads[i]
                orig_video = sgn_video[pad_left:-pad_right, :]
            sgn_keypoints = self.extract_keypoints(orig_video)
            if pads[i]:
                sgn_keypoints = pad_array(sgn_keypoints, pads[i])
            batched_sgn_keypoints.append(sgn_keypoints)
        batched_sgn_keypoints = torch.stack(batched_sgn_keypoints)
        
        batched_sgn_videos = batched_sgn_videos.float()
        batched_sgn_videos /= 255
        batched_sgn_videos = batched_sgn_videos.permute(0, 1, 4, 2, 3) # B T C H W
        
        # Generate low pairs
        number_output_frames = batched_sgn_videos.shape[1]
        if self.training:
            start = np.random.randint(0, number_output_frames // 2 + 1, 1)[0]
        else:
            start = number_output_frames // 4
        end = start + number_output_frames // 2
        
        batched_sgn_videos_low = batched_sgn_videos[:, start:end, :, :, :]
        batched_sgn_keypoints_low = batched_sgn_keypoints[:, start:end, :, :]
        
        # batched_sgn_keypoints = batched_sgn_keypoints.to(batched_sgn_videos.device)
        # batched_sgn_keypoints_low = batched_sgn_keypoints_low.to(batched_sgn_videos.device)
        
        return batched_sgn_videos, batched_sgn_keypoints, batched_sgn_videos_low, batched_sgn_keypoints_low
    
    def _calculate_results(self, outputs, labels, names):
        batch_results = defaultdict(dict)
        batch_name_prob = {}
        batch_logit_names = []
        batch_all_gloss_logits = {k.replace('gloss_logits', ''): v for k, v in outputs.items() if 'gloss_logits' in k and 'word_fused' not in k}
        
        for logits_name, gloss_logits in batch_all_gloss_logits.items():
            batch_logit_names.append(logits_name)
            
            decode_output = self.decode(logits=gloss_logits, k=10) # Top 10
            gloss_prob = outputs[f'{logits_name}gloss_logits'].softmax(dim=-1)
            gloss_prob = torch.sort(gloss_prob, dim=-1, descending=True)[0]
            gloss_prob = gloss_prob[:, :10]
            
            if logits_name in ['ensemble_last_', 'fuse_', 'ensemble_all_']:
                for i in range(gloss_prob.shape[0]):
                    name = logits_name + names[i]
                    batch_name_prob[name] = gloss_prob[i]
            
            for i in range(decode_output.shape[0]):
                name = names[i]
                h = [d.item() for d in decode_output[i]]
                prob = [d.item() for d in gloss_prob[i]]
                
                batch_results[name][f'{logits_name}hyp'] = h
                batch_results[name][f'{logits_name}prob'] = prob
                batch_results[name]['ref'] = labels[i]
                
        return batch_results, batch_name_prob, batch_logit_names
            
    def _calculate_metrics(self, results, logits_names):
        per_ins_stat_dict, per_cls_stat_dict = {}, {}
        
        for logits_name in logits_names:
            correct = correct_5 = correct_10 = num_samples = 0
            top1_t = np.zeros(2000, dtype=np.int32)
            top1_f = np.zeros(2000, dtype=np.int32)
            top5_t = np.zeros(2000, dtype=np.int32)
            top5_f = np.zeros(2000, dtype=np.int32)
            top10_t = np.zeros(2000, dtype=np.int32)
            top10_f = np.zeros(2000, dtype=np.int32)

            for name in results.keys():
                result = results[name]
                h = result[f'{logits_name}hyp']
                ref = result['ref']
                
                # Top 1
                if ref == h[0]:
                    correct += 1
                    top1_t[ref] += 1
                else:
                    top1_f[ref] += 1
                    
                # Top 5
                if ref in h[:5]:
                    correct_5 += 1
                    top5_t[ref] += 1
                else:
                    top5_f[ref] += 1
                    
                # Top 10
                if ref in h[:10]:
                    correct_10 += 1
                    top10_t[ref] += 1
                else:
                    top10_f[ref] += 1
                
                num_samples += 1
                
            per_ins_stat = torch.tensor([correct, correct_5, correct_10, num_samples], dtype=torch.float32)
            per_cls_stat = torch.stack([
                torch.from_numpy(top1_t), torch.from_numpy(top1_f),
                torch.from_numpy(top5_t), torch.from_numpy(top5_f),
                torch.from_numpy(top10_t), torch.from_numpy(top10_f)
            ], dim=0).float()
            
            per_ins_stat_dict[logits_name] = per_ins_stat
            per_cls_stat_dict[logits_name] = per_cls_stat
        
        return per_ins_stat_dict, per_cls_stat_dict
                
    def training_step(self, batch, batch_idx):
        batched_sgn_videos, batched_pads, batched_labels, _ = batch
        sgn_videos, sgn_keypoints, sgn_videos_low, sgn_keypoints_low = self._prepare_inputs(batched_sgn_videos, batched_pads)
        batched_labels = torch.tensor(batched_labels).long().to(sgn_videos.device)
        
        outputs = self(labels=batched_labels, sgn_videos=sgn_videos, sgn_keypoints=sgn_keypoints, sgn_videos_low=sgn_videos_low, sgn_keypoints_low=sgn_keypoints_low)
        
        # Get losses
        for k, v in outputs.items():
            if '_loss' in k:
                self.train_losses[k] += v

        return outputs['total_loss']
    
    def on_train_epoch_end(self):
        losses = {f"train_{k}": v.to("cuda") / self.trainer.num_training_batches for k, v in self.train_losses.items()}
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
    def validation_step(self, batch, batch_idx):
        batched_sgn_videos, batched_pads, batched_labels, batched_names = batch
        sgn_videos, sgn_keypoints, sgn_videos_low, sgn_keypoints_low = self._prepare_inputs(batched_sgn_videos, batched_pads)
        batched_labels = torch.tensor(batched_labels).long()
        
        outputs = self(labels=batched_labels, sgn_videos=sgn_videos, sgn_keypoints=sgn_keypoints, sgn_videos_low=sgn_videos_low, sgn_keypoints_low=sgn_keypoints_low)
        
        # Get losses
        for k, v in outputs.items():
            if '_loss' in k:
                self.val_losses[k] += v
        self.log('val_loss', outputs['total_loss'].to("cuda"), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Get results
        results, name_prob, logits_names = self._calculate_results(outputs, batched_labels, batched_names)
        
        # Update results for this batch
        for name, result in results.items():
            self.val_results[name].update(result)
        for name, prob in name_prob.items():
            self.val_name_prob[name] = prob
        self.val_logits_names = list(set(self.val_logits_names + logits_names))
     
    def on_validation_epoch_end(self):
        results = self.val_results
        logits_names = self.val_logits_names
        losses = {f"val_{k}": v.to("cuda") / sum(self.trainer.num_val_batches) for k, v in self.val_losses.items()}
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Get metrics
        per_ins_stat_dict, per_cls_stat_dict = self._calculate_metrics(results, logits_names)
        
        all_metrics = {}
        for logits_name in logits_names:
            per_ins_top1 = per_ins_stat_dict[logits_name][0]
            per_ins_top5 = per_ins_stat_dict[logits_name][1]
            per_ins_top10 = per_ins_stat_dict[logits_name][2]

            all_metrics[f"{logits_name}_per_ins_top1"] = per_ins_top1.to("cuda")
            all_metrics[f"{logits_name}_per_ins_top5"] = per_ins_top5.to("cuda")
            all_metrics[f"{logits_name}_per_ins_top10"] = per_ins_top10.to("cuda")
            
            top1_t, top1_f, top5_t, top5_f, top10_t, top10_f = per_cls_stat_dict[logits_name]
            per_cls_top1 = torch.nanmean((top1_t / (top1_t + top1_f)).float())
            per_cls_top5 = torch.nanmean((top5_t / (top5_t + top5_f)).float())
            per_cls_top10 = torch.nanmean((top10_t / (top10_t + top10_f)).float())
            
            all_metrics[f"{logits_name}_per_cls_top1"] = per_cls_top1.to("cuda")
            all_metrics[f"{logits_name}_per_cls_top5"] = per_cls_top5.to("cuda")
            all_metrics[f"{logits_name}_per_cls_top10"] = per_cls_top10.to("cuda")
            
        self.log_dict(all_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_results = defaultdict(dict)
        self.val_name_prob = {}
        self.val_logits_names = []
    
    def test_step(self, batch, batch_idx):
        batched_sgn_videos, batched_pads, batched_labels, batched_names = batch
        sgn_videos, sgn_keypoints, sgn_videos_low, sgn_keypoints_low = self._prepare_inputs(batched_sgn_videos, batched_pads)
        batched_labels = torch.tensor(batched_labels).long()
        
        outputs = self(labels=batched_labels, sgn_videos=sgn_videos, sgn_keypoints=sgn_keypoints, sgn_videos_low=sgn_videos_low, sgn_keypoints_low=sgn_keypoints_low)
        
        # Get losses
        for k, v in outputs.items():
            if '_loss' in k:
                self.test_losses[k] += v
        
        # Get results
        results, name_prob, logits_names = self._calculate_results(outputs, batched_labels, batched_names)
        
        # Update results for this batch
        for name, result in results.items():
            self.test_results[name].update(result)
        for name, prob in name_prob.items():
            self.test_name_prob[name] = prob
        self.test_logits_names = list(set(self.test_logits_names + logits_names))
        
    def on_test_epoch_end(self):
        results = self.test_results
        logits_names = self.test_logits_names
        losses = {f"test_{k}": v.to("cuda") / sum(self.trainer.num_test_batches) for k, v in self.test_losses.items()}
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Get metrics
        per_ins_stat_dict, per_cls_stat_dict = self._calculate_metrics(results, logits_names)
        
        all_metrics = {}
        for logits_name in logits_names:
            per_ins_top1 = per_ins_stat_dict[logits_name][0]
            per_ins_top5 = per_ins_stat_dict[logits_name][1]
            per_ins_top10 = per_ins_stat_dict[logits_name][2]

            all_metrics[f"{logits_name}_per_ins_top1"] = per_ins_top1.to("cuda")
            all_metrics[f"{logits_name}_per_ins_top5"] = per_ins_top5.to("cuda")
            all_metrics[f"{logits_name}_per_ins_top10"] = per_ins_top10.to("cuda")
            
            top1_t, top1_f, top5_t, top5_f, top10_t, top10_f = per_cls_stat_dict[logits_name]
            per_cls_top1 = torch.nanmean((top1_t / (top1_t + top1_f)).float())
            per_cls_top5 = torch.nanmean((top5_t / (top5_t + top5_f)).float())
            per_cls_top10 = torch.nanmean((top10_t / (top10_t + top10_f)).float())
            
            all_metrics[f"{logits_name}_per_cls_top1"] = per_cls_top1.to("cuda")
            all_metrics[f"{logits_name}_per_cls_top5"] = per_cls_top5.to("cuda")
            all_metrics[f"{logits_name}_per_cls_top10"] = per_cls_top10.to("cuda")
            
        self.log_dict(all_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_results = defaultdict(dict)
        self.test_name_prob = {}
        self.test_logits_names = []

    def configure_optimizers(self):
        optmizer = Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=0.001,
            betas=(0.9, 0.998),
        )
        scheduler = CosineAnnealingLR(
            optimizer=optmizer,
            T_max=100,
        )
        
        return {
            'optimizer': optmizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }