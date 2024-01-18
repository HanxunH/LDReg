import misc
import torch
from torch import nn
from lid import lid_mle
import torch.nn.functional as F
import util
import numpy as np


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, gather_distributed: bool = False, dim_norm=False):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-8
        self.dim_norm = dim_norm
        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def track_lid(self, f_0, f_1):
        # Track LID
        with torch.no_grad():
            f = torch.cat([f_0, f_1], dim=0).detach()
            if self.gather_distributed:
                full_rank_f = torch.cat(misc.gather(f), dim=0)
            else:
                full_rank_f = f

            lids_k32 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=32)
            lids_k512 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=512)
        return lids_k32, lids_k512
    

    def forward(self, model, images, online_labels=None):
        
        (x0, x1) = images

        f_0, z_0, cls_0 = model(x0)
        f_1, z_1, cls_1 = model(x1)
        
        # online prob
        cls = torch.cat([cls_0, cls_1], dim=0)
        online_labels = torch.cat([online_labels, online_labels], dim=0)
        online_prob_loss = F.cross_entropy(cls, online_labels)
        online_prob_loss.backward()
        online_prob_loss_acc = util.accuracy(cls, online_labels)[0]

        batch_size = z_0.shape[0]
        if self.dim_norm:
            full_rank_z = torch.cat(misc.gather(torch.cat([z_0, z_1], dim=0)), dim=0)
            denom = full_rank_z.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12).expand_as(z_0)
            z_0 = z_0 / denom * np.sqrt(z_0.shape[0]/full_rank_z.shape[1])
            z_1 = z_1 / denom * np.sqrt(z_1.shape[0]/full_rank_z.shape[1])
        else:
            z_0 = F.normalize(z_0, dim=1)
            z_1 = F.normalize(z_1, dim=1)
        # user other samples from batch as negatives
        # and create diagonal mask that only selects similarities between
        # views of the same image
        if self.gather_distributed and misc.world_size() > 1:
            # gather hidden representations from other processes
            out0_large = torch.cat(misc.gather(z_0), 0)
            out1_large = torch.cat(misc.gather(z_1), 0)
            diag_mask = misc.eye_rank(batch_size, device=z_0.device)
        else:
            # single process
            out0_large = z_0
            out1_large = z_1
            diag_mask = torch.eye(batch_size, device=z_0.device, dtype=torch.bool)
        
        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        logits_00 = torch.einsum('nc,mc->nm', z_0, out0_large) / self.temperature
        logits_01 = torch.einsum('nc,mc->nm', z_0, out1_large) / self.temperature
        logits_10 = torch.einsum('nc,mc->nm', z_1, out0_large) / self.temperature
        logits_11 = torch.einsum('nc,mc->nm', z_1, out1_large) / self.temperature

        # remove simliarities between same views of the same image
        logits_00 = logits_00[~diag_mask].view(batch_size, -1)
        logits_11 = logits_11[~diag_mask].view(batch_size, -1)

        # concatenate logits
        # the logits tensor in the end has shape (2*n, 2*m-1)
        logits_0100 = torch.cat([logits_01, logits_00], dim=1)
        logits_1011 = torch.cat([logits_10, logits_11], dim=1)
        logits = torch.cat([logits_0100, logits_1011], dim=0)

        # create labels
        labels = torch.arange(batch_size, device=z_0.device, dtype=torch.long)
        labels = labels + misc.rank() * batch_size
        labels = labels.repeat(2)
        
        loss = self.cross_entropy(logits, labels)
        loss = loss.mean(dim=0)

        # Track LID
        lids_k32, lids_k512 = self.track_lid(f_0, f_1)
        
        results = {
            "loss": loss,
            "logits": logits.detach(),
            "labels": labels,
            "lids32": lids_k32.detach(),
            "lids512": lids_k512.detach(),
            "main_loss": loss.item(),
            "online_acc": online_prob_loss_acc.item(),
        }
        return results


