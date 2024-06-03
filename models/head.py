import torch

from torch import nn

import torch.nn.functional as F

class SepConvHead(nn.Module):
    def __init__(self, word_embedding, input_size=1024, hidden_size=512, ff_size=2048, pe=True, 
                 ff_kernelsize=[3, 3], word_emb_dim=300, topk=5, contras_setting='dual_ema_cosine'):
        super(SepConvHead, self).__init__()
        self.topk = topk
        
        self.gloss_output_layer = nn.Linear(input_size, 2000)
        
        self.word_embedding = word_embedding
        self.word_emb_dim = word_emb_dim
        self.word_embedding.requires_grad = False
        self.word_emb_sim = torch.matmul(F.normalize(self.word_embedding, dim=-1), F.normalize(self.word_embedding, dim=-1).T)
        self.word_emb_mapper = nn.Linear(self.word_emb_dim, input_size)
        self.word_fused_gloss_output_layer = nn.Linear(input_size, 2000)
        
    def forward(self, x, labels=None):
        if x.ndim > 3:
            x = F.avg_pool3d(x, kernel_size=(2, x.size(3), x.size(4)), stride=1)
            x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)
            
        if x.ndim > 2:
            x = x.mean(dim=1)

        self.word_embedding = self.word_embedding.to(x.device)
            
        B, C = x.shape[0], x.shape[-1]
        visual_feature = x
        
        k = gloss_probabilities = word_fused_gloss_logits = word_fused_gloss_probabilities = topk_idx = None
        
        logits = self.gloss_output_layer(x)
        if self.training:
            logits_data = logits.clone().detach()
            batch_idx = torch.arange(B)
            logits_data[batch_idx, labels] = -float('inf')
            idx = torch.argsort(logits_data, dim=1, descending=True)[:, :self.topk - 1]
            topk_idx = torch.cat([labels.unsqueeze(1), idx], dim=1)
        else:
            topk_idx = torch.argsort(logits, dim=1, descending=True)[:, :self.topk]
        topk_idx = topk_idx.reshape(-1)
            
        if k is None:
            k = self.word_emb_mapper(self.word_embedding)
        word_embs = k.index_select(0, topk_idx).reshape(B, -1, C)
        
        fused_feature = visual_feature.unsqueeze(1) + word_embs
        word_fused_gloss_logits = self.word_fused_gloss_output_layer(fused_feature)
        
        return {
            'gloss_logits': logits,
            'word_fused_gloss_logits': word_fused_gloss_logits,
            'topk_idx': topk_idx
        }