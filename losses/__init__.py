import mlconfig
import torch
from . import ntxent
from . import ntxent_lid_reg
from . import byol_loss
from . import mae_loss

mlconfig.register(torch.nn.CrossEntropyLoss)
mlconfig.register(ntxent.NTXentLoss)
mlconfig.register(ntxent_lid_reg.NTXentBase)
mlconfig.register(ntxent_lid_reg.NTXentLIDReg)
mlconfig.register(byol_loss.BYOLLoss)
mlconfig.register(byol_loss.BYOLLIDReg)
mlconfig.register(mae_loss.MAELoss)
mlconfig.register(mae_loss.MAELDRegLoss)