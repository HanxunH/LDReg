import mlconfig
import torch
from . import resnet
from . import mae
from . import vit
from . import lars

mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.AdamW)
mlconfig.register(torch.optim.LBFGS)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)

# Models
mlconfig.register(resnet.ResNet)
mlconfig.register(resnet.ResNetSimCLR)
mlconfig.register(resnet.ResNetSimCLRTuned)
mlconfig.register(resnet.ResNetBYOL)
mlconfig.register(mae.mae_vit_base_patch16_dec512d8b)
mlconfig.register(vit.vit_base_patch16)
mlconfig.register(vit.simclr_vit_base_patch16)
# 
mlconfig.register(lars.LARS)
