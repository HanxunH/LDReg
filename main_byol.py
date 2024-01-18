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
import copy
from lid import gmean
from exp_mgmt import ExperimentManager
from engine_byol import train_epoch, evaluate_full_set_lid, evaluate
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
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


def save_model(model, optimizer, epoch=None, key=None):
    # Save model
    if key is not None:
        exp.save_state(model, 'model_{}_state_dict'.format(key))
    else:
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

    loader = data.get_loader(drop_last=True, train_shuffle=True, train_sampler=sampler_train, test_sampler=sampler_val)
    train_loader, test_loader, eval_train_loader = loader

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
    model_momentum = config.model().to(device)
    model_momentum.momentum_branch = True
    
    for param in model_momentum.parameters():
        param.requires_grad = False
    if 'fix_pred_lr' in exp.config and exp.config.fix_pred_lr:
        optimizer = config.optimizer(model.get_parameter())
    else:
        optimizer = config.optimizer(model.parameters())

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
        try:
            exp_stats = exp.load_epoch_stats()
            start_epoch = exp_stats['epoch'] + 1
            global_step = exp_stats['global_step'] + 1
            model = exp.load_state(model, 'model_state_dict')
            optimizer = exp.load_state(optimizer, 'optimizer_state_dict')
            if 'run_id' in exp_stats:
                exp.run_id = exp_stats['run_id']
                exp._init_neptune()
        except:
            if misc.get_rank() ==0 :
                logger.info('Start New, load model failed')

    if args.ddp:
        if misc.get_rank() == 0:
            logger.info('DDP')
        if 'sync_bn' in exp.config and exp.config.sync_bn:
            if misc.get_rank() == 0:
                logger.info('Sync Batch Norm')
            sync_bn_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            model = torch.nn.parallel.DistributedDataParallel(sync_bn_network, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Train Loops
    for epoch in range(start_epoch, exp.config.epochs):
        stats = {}
        start_time = time.time()
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(criterion, 'epoch'):
            criterion.epoch = epoch
        stats = train_epoch(exp, model, model_momentum, optimizer, optimizer_online, online_lr,
                            criterion, scaler, train_loader, global_step, epoch, logger, args)
        global_step = stats['global_step']
        # Epoch Eval Function
        if hasattr(config, 'eval_every_epoch') and epoch % config.eval_every_epoch == 0:
            if misc.get_rank() == 0:
                logger.info("="*20 + "Evaluations Epoch %d" % (epoch) + "="*20)
            model.eval()
            test_set_lids32, test_set_lids512, online_acc = evaluate(model, test_loader, args, exp.config)
            if hasattr(config, 'full_set_lid_eval') and config.full_set_lid_eval:
                full_set_train_lid32, full_set_train_lid512 = evaluate_full_set_lid(model, eval_train_loader, args, exp.config)
                full_set_test_lid32, full_set_test_lid512 = evaluate_full_set_lid(model, test_loader, args, exp.config)
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
                if hasattr(config, 'full_set_lid_eval') and config.full_set_lid_eval:
                    stats['full_set_train_lid32_avg'] = full_set_train_lid32.mean().item()
                    stats['full_set_train_lid32_var'] = full_set_train_lid32.var().item()
                    stats['full_set_test_lid32_avg'] = full_set_test_lid32.mean().item()
                    stats['full_set_test_lid32_var'] = full_set_test_lid32.var().item()
                    stats['full_set_train_lid512_avg'] = full_set_train_lid512.mean().item()
                    stats['full_set_train_lid512_var'] = full_set_train_lid512.var().item()
                    stats['full_set_test_lid512_avg'] = full_set_test_lid512.mean().item()
                    stats['full_set_test_lid512_var'] = full_set_test_lid512.var().item()
                    payload = 'Full Train set LID32 avg={:.4f} var={:.4f}'.format(
                            full_set_train_lid32.mean().item(), full_set_train_lid32.var().item())
                    logger.info('\033[33m'+payload+'\033[0m')
                    payload = 'Full Test set LID32 avg={:.4f} var={:.4f}'.format(
                            full_set_test_lid32.mean().item(), full_set_test_lid32.var().item())
                    logger.info('\033[33m'+payload+'\033[0m')
                    payload = 'Full Train set LID512 avg={:.4f} var={:.4f}'.format(
                            full_set_train_lid512.mean().item(), full_set_train_lid512.var().item())
                    logger.info('\033[33m'+payload+'\033[0m')
                    payload = 'Full Test set LID512 avg={:.4f} var={:.4f}'.format(
                            full_set_test_lid512.mean().item(), full_set_test_lid512.var().item())
                    logger.info('\033[33m'+payload+'\033[0m')
        # Save Model
        if misc.get_rank() == 0:
            exp.save_epoch_stats(epoch=epoch, exp_stats=stats)
            save_model(model_without_ddp, optimizer)
            save_model(model_momentum, optimizer, key='momentum')
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
    global exp
    args = parser.parse_args()
    if args.ddp:
        misc.init_distributed_mode(args)
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