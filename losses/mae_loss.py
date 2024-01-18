import misc
import math
import torch
import torch.nn.functional as F
import util
from torch import nn
from lid import lid_mle, lid_mom_est

class MAELoss(nn.Module):
    def __init__(self, gather_distributed=False):
        super(MAELoss, self).__init__()
        self.epoch = 0 
        self.gather_distributed = gather_distributed

    def track_lid(self, features):
        # Track LID
        with torch.no_grad():
            if self.gather_distributed:
                full_rank_fe = torch.cat(misc.gather(features), dim=0)
            else:
                full_rank_fe = features
            lids_k32 = lid_mle(data=features.detach(), reference=full_rank_fe.detach(), k=32)
            lids_k512 = lid_mle(data=features.detach(), reference=full_rank_fe.detach(), k=512)
        return lids_k32, lids_k512

    def forward(self, model, images, online_labels=None):
        images = images.cuda(non_blocking=True)
        cls_token, loss, _, _, cls = model(images)
        
        # online prob
        online_prob_loss = F.cross_entropy(cls, online_labels)
        online_prob_loss.backward()
        online_prob_loss_acc = util.accuracy(cls, online_labels)[0]
        
        # Track LID and logits
        lids_k32, lids_k512 = self.track_lid(cls_token.detach())

        results = {
            "loss": loss,
            "lids32": lids_k32.detach(),
            "lids512": lids_k512.detach(),
            "main_loss": loss.item(),
            "online_acc": online_prob_loss_acc.item(),
        }
        return results
  


class MAELDRegLoss(MAELoss):
    def __init__(self, gather_distributed=False, reg_type='l1', 
                 lid_type='f', alpha=1.0, k=20, est_type='mom', warmup_epochs=0,
                 compute_mode='use_mm_for_euclid_dist_if_necessary'):
        super().__init__(gather_distributed)
        self.alpha = alpha
        self.k = k
        self.lid_type = lid_type
        self.reg_type = reg_type
        self.warmup_epochs = warmup_epochs
        self.epoch = 0 
        self.compute_mode = compute_mode
        if est_type == 'mle':
            self.lid_est_fn = lid_mle
        elif est_type == 'mom':
            self.lid_est_fn = lid_mom_est
        else:
            raise('Unknow Est')
        
    def forward(self, model, images, online_labels=None):
        images = images.cuda(non_blocking=True)
        cls_token, loss, _, _, cls = model(images)
        
        # online prob
        online_prob_loss = F.cross_entropy(cls, online_labels)
        online_prob_loss.backward()
        online_prob_loss_acc = util.accuracy(cls, online_labels)[0]
        
        # LID reg
        if self.epoch < self.warmup_epochs:
            lid_f = cls_token.detach()
        else:
            lid_f = cls_token
            
        if self.gather_distributed:
            full_rank_lid_f = torch.cat(misc.gather(lid_f), dim=0)
        else:
            full_rank_lid_f = lid_f

        lids = self.lid_est_fn(data=lid_f, reference=full_rank_lid_f.detach(), k=self.k, compute_mode=self.compute_mode)
        
        if self.reg_type == 'l1':
            lid_reg = - torch.abs(torch.log(lids))
        elif self.reg_type == 'l2':
            lid_reg = - torch.sqrt(torch.square(torch.log(lids)))
        
        if self.epoch < self.warmup_epochs:
            total_loss = loss
        else:
            total_loss = loss + self.alpha * lid_reg.mean()

        # Track LID and logits
        lids_k32, lids_k512 = self.track_lid(cls_token.detach())

        results = {
            "loss": total_loss,
            "lids32": lids_k32.detach(),
            "lids512": lids_k512.detach(),
            "reg_loss": lid_reg.detach().mean().item(),
            "main_loss": loss.detach().mean().item(),
            "online_acc": online_prob_loss_acc.item(),
        }
        return results