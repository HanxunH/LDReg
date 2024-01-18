import argparse
import torch
import mlconfig
import models
import datasets
import losses
import util
import misc
import os
import sys
import numpy as np
import time
import math
from lid import gmean
from exp_mgmt import ExperimentManager
from engine_simclr import train_epoch, evaluate
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='SSL-LID')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')


def save_model(model, optimizer, epoch=None):
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    if epoch is not None:
        exp.save_state(model, 'model_state_dict_epoch{:d}'.format(epoch))


def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config
    # Prepare Data
    data = config.dataset()
    if args.ddp:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if misc.get_rank() == 0:
            logger.info('World Size {}'.format(num_tasks))
        sampler_train = torch.utils.data.DistributedSampler(
            data.train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(data.test_set) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(data.test_set, num_replicas=num_tasks,
                                                              rank=global_rank, shuffle=True)
            # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(data.test_set)
    else:
        sampler_train = torch.utils.data.RandomSampler(data.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data.test_set)

    loader = data.get_loader(drop_last=True, train_shuffle=True, 
            train_sampler=sampler_train, test_sampler=sampler_val)
    train_loader, test_loader, _ = loader

    if 'blr' in exp.config:
        if exp.config.blr_scale == 'linear':
            # Linear scaling
            eff_batch_size = exp.config.dataset.train_bs * misc.get_world_size()
            exp.config.lr = exp.config.blr * eff_batch_size / 256
        else:
            # Square root scaling
            eff_batch_size = exp.config.dataset.train_bs * misc.get_world_size()
            exp.config.lr = exp.config.blr * math.sqrt(eff_batch_size)
        if misc.get_rank() == 0:
            logger.info('adjusted lr: {:.6f}'.format(exp.config.lr))
        
    # Prepare Model
    model = config.model().to(device)
    params = []
    online_params = []
    for item in model.named_parameters():
        if 'online' in item[0]:
            online_params.append(item[1])
        else:
            params.append(item[1])
    optimizer = config.optimizer(params)
    eff_batch_size = exp.config.dataset.train_bs * misc.get_world_size()
    online_lr = 0.1 * eff_batch_size / 256
    optimizer_online = models.lars.LARS(online_params, online_lr, weight_decay=0.0)

    if misc.get_rank() == 0:
        print(model)

    # Prepare Objective Loss function
    criterion = config.criterion()
    start_epoch = 0
    global_step = 0
    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.cuda.amp.GradScaler() 
    else:
        scaler = None

    # Resume: Load models
    if args.load_model:
        exp_stats = exp.load_epoch_stats()
        start_epoch = exp_stats['epoch'] + 1
        global_step = exp_stats['global_step'] + 1
        model = exp.load_state(model, 'model_state_dict')
        optimizer = exp.load_state(optimizer, 'optimizer_state_dict')

    if args.ddp:
        if misc.get_rank() == 0:
            logger.info('DDP')
        if 'sync_bn' in exp.config and exp.config.sync_bn:
            if misc.get_rank() == 0:
                logger.info('Sync Batch Norm')
            sync_bn_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            model = torch.nn.parallel.DistributedDataParallel(sync_bn_network, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Train Loops
    for epoch in range(start_epoch, exp.config.epochs):
        start_time = time.time()
        stats = {}
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(criterion, 'epoch'):
            criterion.epoch = epoch
        stats = train_epoch(exp, model, optimizer, optimizer_online, online_lr, 
                criterion, scaler, train_loader, global_step, epoch, logger, args)
        global_step = stats['global_step']
        # Epoch Eval Function
        if hasattr(config, 'eval_every_epoch') and epoch % config.eval_every_epoch == 0:
            if misc.get_rank() == 0:
                logger.info("="*20 + "Evaluations Epoch %d" % (epoch) + "="*20)
            model.eval()
            eval_rs = evaluate(model, test_loader, args, exp.config)
            test_set_lids32, test_set_lids512, online_acc = eval_rs
            if misc.get_rank() == 0:
                payload = 'Test set LID32 avg={:.4f} var={:.4f}'.format(
                    test_set_lids32.mean().item(), test_set_lids32.var().item())
                logger.info('\033[33m'+payload+'\033[0m')
                payload = 'Test set LID512 avg={:.4f} var={:.4f}'.format(
                    test_set_lids512.mean().item(), test_set_lids512.var().item())
                logger.info('\033[33m'+payload+'\033[0m')
                payload = 'Test set LID32 geometric avg={:.4f}'.format(
                    gmean(test_set_lids32).item())
                logger.info('\033[33m'+payload+'\033[0m')
                payload = 'Test set LID512 geometric avg={:.4f}'.format(
                    gmean(test_set_lids512).item())
                logger.info('\033[33m'+payload+'\033[0m')
                payload = 'Test set Online Acc avg={:.4f}'.format(online_acc)
                logger.info('\033[33m'+payload+'\033[0m')
                stats['test_lid32_avg'] = test_set_lids32.mean().item()
                stats['test_lid32_var'] = test_set_lids32.var().item()
                stats['test_lid512_avg'] = test_set_lids512.mean().item()
                stats['test_lid512_var'] = test_set_lids512.var().item()
                stats['test_lid32_gavg'] = gmean(test_set_lids32).item()
                stats['test_lid512_gavg'] = gmean(test_set_lids512).item()
                stats['test_online_acc'] = online_acc

        # Save Model
        if misc.get_rank() == 0:
            exp.save_epoch_stats(epoch=epoch, exp_stats=stats)
            save_model(model_without_ddp, optimizer)
            if epoch % config.snapshot_epoch == 0:
                save_model(model_without_ddp, optimizer, epoch=epoch)
        
        end_time = time.time()
        cost_per_epoch = (end_time - start_time) / 60
        esitmited_finish_cost = (end_time - start_time) / 3600 * (exp.config.epochs - epoch - 1)
        if misc.get_rank() == 0:
            payload = "Running Cost %.2f mins/epoch, finish in %.2f hours (esimitated)" % (cost_per_epoch, esitmited_finish_cost)
            logger.info('\033[33m'+payload+'\033[0m')
    return


if __name__ == '__main__':
    print(torch.cuda.nccl.version())
    global exp
    args = parser.parse_args()
    if args.ddp:
        misc.init_distributed_mode(args)
        if misc.get_rank() == 0:
            print('Distributed Training Enabled')
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)
    args.gpu = device

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    misc.destroy_process_group()