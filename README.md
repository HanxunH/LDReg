# LDReg: Local Dimensionality Regularized Self-Supervised Learning

Code for ICLR2024 paper ["LDReg: Local Dimensionality Regularized Self-Supervised Learning"](https://arxiv.org/abs/2401.10474)

---
## LDReg

```python
# Method of Moments estimation of LID
def lid_mom_est(data, reference, k, get_idx=False, 
                compute_mode='use_mm_for_euclid_dist_if_necessary'):
    b = data.shape[0]
    k = min(k, b-2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    m = torch.mean(a[:, 1:k], dim=1)
    lids = m / (a[:, k] - m)
    if get_idx:
        return idx, lids
    return lids

# features: representations that need LID to be estimated. 
# reference: reference representations, usually, the same batch of representations can be used. 
# k: locality parameter, the neighbourhood size. 
# NOTE: features and reference should be in the same dimension.

lids = lid_mom_est(data=features, reference=reference.detach(), k=k)
reg_loss = - torch.log(lids)  # Eq (1) of the paper. 
        
```

---
## Reproduce results from the paper
We provide configuration files in the *configs* folder. Details of all necessary hyperparameters are also in the Appendix of the paper. 

Pretrained models are available here in this [Google Drive](https://drive.google.com/drive/folders/1s70_QnFG_ZDBjxsdqe8Y0xGJIocLKom8?usp=share_link) folder. 

An example of how to run pretraing is the following:
```
srun python3 main_simclr.py --ddp --dist_eval                  \
                            --exp_path path/to/exp/folder      \
                            --exp_config path/to/config/folder \
                            --exp_name pretrain     
```


An example of how to run linear probing:
```
srun python3 main_linear_prob.py --ddp --dist_eval                  \
                                 --exp_path path/to/exp/folder      \
                                 --exp_config path/to/config/folder \
                                 --exp_name linear_prob          
```



## Citation
If you use this code in your work, please cite the accompanying paper:
```
@inproceedings{huang2024ldreg,
  title={LDReg: Local Dimensionality Regularized Self-Supervised Learning},
  author={Hanxun Huang and Ricardo J. G. B. Campello and Sarah Monazam Erfani and Xingjun Ma and Michael E. Houle and James Bailey},
  booktitle={ICLR},
  year={2024}
}
```

## Part of the code is based on the following repo:
  - PyTorch implementation of MAE:  https://github.com/facebookresearch/mae
  - VICReg official code base: https://github.com/facebookresearch/vicreg
  - Lightly: https://github.com/lightly-ai/lightly
  - PyTorch implementation of MoCo v3 : https://github.com/facebookresearch/moco-v3
  - Detectron2: https://github.com/facebookresearch/detectron2
 
         
