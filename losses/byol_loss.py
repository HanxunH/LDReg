import misc
import math
import torch
import torch.nn.functional as F
import util
from torch import nn
from lid import lid_mle, lid_mom_est

class BYOLLoss(nn.Module):
    def __init__(self, gather_distributed=False):
        super(BYOLLoss, self).__init__()
        self.gather_distributed = gather_distributed
        self.epoch = 0 

    def track_logits(self, out0, out1):
        batch_size, _ = out0.shape
        if self.gather_distributed and misc.world_size() > 1:
            # gather hidden representations from other processes
            out0_large = torch.cat(misc.gather(out0), 0)
            out1_large = torch.cat(misc.gather(out1), 0)
            diag_mask = misc.eye_rank(batch_size, device=out0.device)
        else:
            # single process
            out0_large = out0
            out1_large = out1
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        logits_00 = torch.einsum('nc,mc->nm', out0, out0_large) 
        logits_01 = torch.einsum('nc,mc->nm', out0, out1_large) 
        logits_10 = torch.einsum('nc,mc->nm', out1, out0_large) 
        logits_11 = torch.einsum('nc,mc->nm', out1, out1_large) 

        # remove simliarities between same views of the same image
        logits_00 = logits_00[~diag_mask].view(batch_size, -1)
        logits_11 = logits_11[~diag_mask].view(batch_size, -1)

        # concatenate logits
        # the logits tensor in the end has shape (2*n, 2*m-1)
        logits_0100 = torch.cat([logits_01, logits_00], dim=1)
        logits_1011 = torch.cat([logits_10, logits_11], dim=1)
        logits = torch.cat([logits_0100, logits_1011], dim=0)

        # create labels
        labels = torch.arange(batch_size, device=out0.device, dtype=torch.long)
        labels = labels + misc.rank() * batch_size
        labels = labels.repeat(2)
        return logits, labels

    def track_lid(self, fe0, fe1):
        # Track LID
        with torch.no_grad():
            fe = torch.cat([fe0, fe1], dim=0).detach()
            if self.gather_distributed:
                full_rank_fe = torch.cat(misc.gather(fe), dim=0)
            else:
                full_rank_fe = fe
            lids_k32 = lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=32)
            lids_k512 = lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=512)
        return lids_k32, lids_k512
    
    def loss(self, p, z):
        z = z.detach()
        preds_norm = F.normalize(p, dim=1)
        targets_norm = F.normalize(z, dim=1)
        loss = 2 - 2 * (preds_norm * targets_norm).sum(dim=1)
        return loss

    def forward(self, model, model_momentum, images, online_labels=None):
        (x0, x1) = images
        
        fe_0, _, p_0, cls_0 = model(x0)
        fe_1, _, p_1, cls_1 = model(x1)
        with torch.no_grad():
            _, z_0, _, _ = model_momentum(x0)
            _, z_1, _, _ = model_momentum(x1)
        
        # online prob
        cls = torch.cat([cls_0, cls_1], dim=0)
        online_labels = torch.cat([online_labels, online_labels], dim=0)
        online_prob_loss = F.cross_entropy(cls, online_labels)
        online_prob_loss.backward()
        online_prob_loss_acc = util.accuracy(cls, online_labels)[0]

        # BYOL Loss
        main_loss = self.loss(p_0, z_1.detach()) + self.loss(p_1, z_0.detach())
        main_loss = main_loss.mean() 
        
        # Track LID and logits
        lids_k32, lids_k512 = self.track_lid(fe_0, fe_1)
        logits, labels = self.track_logits(F.normalize(p_0.detach(), dim=1), F.normalize(p_1.detach(), dim=1))

        results = {
            "loss": main_loss,
            "logits": logits.detach(),
            "labels": labels,
            "lids32": lids_k32.detach(),
            "lids512": lids_k512.detach(),
            "main_loss": main_loss.item(),
            "online_acc": online_prob_loss_acc.item(),
        }
        return results
  

class BYOLLIDReg(BYOLLoss):
    def __init__(self, gather_distributed=False, reg_type='l1', 
                 lid_type='f', alpha=1.0, k=20, est_type='mom', warmup_epochs=0):
        super(BYOLLIDReg, self).__init__(gather_distributed=gather_distributed)
        self.alpha = alpha
        self.k = k
        self.lid_type = lid_type
        self.reg_type = reg_type
        self.warmup_epochs = warmup_epochs
        self.epoch = 0 
        if est_type == 'mle':
            self.lid_est_fn = lid_mle
        elif est_type == 'mom':
            self.lid_est_fn = lid_mom_est
        else:
            raise('Unknow Est')

    def forward(self, model, model_momentum, images, online_labels=None):
        (x0, x1) = images
        
        f_0, _, p_0, cls_0 = model(x0)
        f_1, _, p_1, cls_1 = model(x1)
        with torch.no_grad():
            f_0_m, z_0, _, _ = model_momentum(x0)
            f_1_m, z_1, _, _ = model_momentum(x1)
        
        # online prob
        cls = torch.cat([cls_0, cls_1], dim=0)
        online_labels = torch.cat([online_labels, online_labels], dim=0)
        online_prob_loss = F.cross_entropy(cls, online_labels)
        online_prob_loss.backward()
        online_prob_loss_acc = util.accuracy(cls, online_labels)[0]

        # LID reg
        lid_f_0, lid_f_1 = f_0, f_1
        
        if self.epoch < self.warmup_epochs:
            lid_f = torch.cat([lid_f_0.detach(), lid_f_1.detach()], dim=0)
        else:
            lid_f = torch.cat([lid_f_0, lid_f_1], dim=0)

        if self.gather_distributed:
            full_rank_lid_f = torch.cat(misc.gather(lid_f), dim=0)
        else:
            full_rank_lid_f = lid_f
        
        extend_f = torch.cat([f_0_m, f_1_m], dim=0)
        if self.gather_distributed:
            extend_f = torch.cat(misc.gather(extend_f), dim=0)
        full_rank_lid_f = torch.cat([full_rank_lid_f, extend_f], dim=0)

        lids_f_0 = self.lid_est_fn(data=lid_f_0, reference=full_rank_lid_f.detach(), k=self.k)
        lids_f_1 = self.lid_est_fn(data=lid_f_1, reference=full_rank_lid_f.detach(), k=self.k)
        if self.reg_type == 'l1':
            lid_reg_f_0 = - torch.abs(torch.log(lids_f_0))
            lid_reg_f_1 = - torch.abs(torch.log(lids_f_1))
        elif self.reg_type == 'l2':
            lid_reg_f_0 = - torch.sqrt(torch.square(torch.log(lids_f_0)))
            lid_reg_f_1 = - torch.sqrt(torch.square(torch.log(lids_f_1)))
        
        # BYOL Loss
        main_loss = self.loss(p_0, z_1.detach()) + self.loss(p_1, z_0.detach())
        lid_reg = lid_reg_f_0 + lid_reg_f_1

        if self.epoch < self.warmup_epochs:
            total_loss = main_loss.mean(dim=0)
        else:
            total_loss = main_loss + self.alpha * lid_reg
            total_loss = total_loss.mean(dim=0)
        
        # Track LID and logits
        lids_k32, lids_k512 = self.track_lid(f_0, f_1)
        logits, labels = self.track_logits(F.normalize(p_0.detach(), dim=1), F.normalize(p_1.detach(), dim=1))

        results = {
            "loss": total_loss,
            "logits": logits.detach(),
            "labels": labels,
            "lids32": lids_k32.detach(),
            "lids512": lids_k512.detach(),
            "reg_loss": lid_reg.detach().mean().item(),
            "main_loss": main_loss.detach().mean().item(),
            "online_acc": online_prob_loss_acc.item(),
        }
        return results
    
