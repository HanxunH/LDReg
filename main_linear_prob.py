import argparse
import mlconfig
import torch
import time
import models
import datasets
import losses
import torch.nn.functional as F
import util
import os
import sys
import numpy as np
import misc
from exp_mgmt import ExperimentManager
from engine_linear_prob import train_epoch, evaluate
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
parser.add_argument('--dp', action='store_true', default=False)
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

def save_model(model, optimizer, is_best=False):
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    if is_best:
        exp.save_state(model, 'model_best_state_dict')
        exp.save_state(optimizer, 'optimizer_best_state_dict')

def train():
    best_acc = 0
    if 'blr' in exp.config:
        eff_batch_size = exp.config.dataset.train_bs * misc.get_world_size()
        exp.config.lr = exp.config.blr * eff_batch_size / 256
        if misc.get_rank() == 0:
            logger.info('adjusted lr: {:.6f}'.format(exp.config.lr))

    model = exp.config.model().to(device)
    scaler = None
    try:
        global_step = 0
        if hasattr(exp.config, 'pretrain_path'):
            dir = os.path.join(exp.config.pretrain_path, 'checkpoints')
        else:
            dir = os.path.join(exp.exp_path.replace(exp.exp_name, 'pretrain'), 'checkpoints')
        model = exp.load_state_with_dir(dir, model, 'model_state_dict', strict=False)
    except:
        if misc.get_rank() == 0:
            logger.info('Load model failed')
        raise('Load model failed')

    # Prepare Model
    linear_prob = model.add_linear_prob()
    optimizer = exp.config.optimizer(linear_prob.parameters())

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc, best_acc5 = 0, 0
    print(model)

    if args.ddp:
        if misc.get_rank() == 0:
            logger.info('DDP')
        if 'sync_bn' in exp.config and exp.config.sync_bn:
            if misc.get_rank() == 0:
                logger.info('Sync Batch Norm')
            sync_bn_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            model = torch.nn.parallel.DistributedDataParallel(sync_bn_network, device_ids=[args.gpu],
                                                              find_unused_parameters=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                              find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    util.adjust_learning_rate(optimizer, 0, exp.config)

    for epoch in range(exp.config.epochs):
        start_time = time.time()
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        stats = train_epoch(exp, model, optimizer, criterion, scaler, train_loader, global_step, epoch, args, logger)
        global_step = stats['global_step']
        # Epoch Eval Function
        if misc.get_rank() == 0:
            logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        model.eval()
        eval_loss, eval_acc, eval_acc5 = evaluate(model, test_loader, scaler, exp, args)
        is_best = False
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_acc5 = eval_acc5
            is_best = True
        payload = 'Eval Loss: %.4f Eval Acc: %.4f Best Acc: %.4f' % \
            (eval_loss, eval_acc, best_acc)
        payload += ' Eval Acc5: %.4f Best Acc5: %.4f' % (eval_acc5, best_acc5)
        if misc.get_rank() == 0:
            logger.info('\033[33m'+payload+'\033[0m')
            save_model(model_without_ddp, optimizer, is_best)
        end_time = time.time()
        cost_per_epoch = (end_time - start_time) / 60
        esitmited_finish_cost = (end_time - start_time) / 3600 * (exp.config.epochs - epoch - 1)
        if misc.get_rank() == 0:
            payload = "Running Cost %.2f mins/epoch, finish in %.2f hours (esimitated)" % (cost_per_epoch, esitmited_finish_cost)
            logger.info('\033[33m'+payload+'\033[0m')
    results = {
        'eval_loss': eval_loss,
        'eval_acc': eval_acc,
        'best_acc': best_acc,
        'eval_acc5': eval_acc5,
        'best_acc5': best_acc5,
    }
    return results


def main():
    # Set Global Vars
    global train_loader, test_loader, data
    global exp, logger

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
        sampler_val = torch.utils.data.SequentialSampler(data.test_set)
        # if args.dist_eval:
        #     if len(data.test_set) % num_tasks != 0:
        #         if misc.get_rank() == 0:
        #             logger.info(
        #                 'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #                 'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #                 'equal num of samples per-process.')
        #     sampler_val = torch.utils.data.DistributedSampler(data.test_set, num_replicas=num_tasks,
        #                                                       rank=global_rank, shuffle=True)
        #     # shuffle=True to reduce monitor bias
        # else:
        #     sampler_val = torch.utils.data.SequentialSampler(data.test_set)

    else:
        sampler_train = torch.utils.data.RandomSampler(data.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data.test_set)

    loader = data.get_loader(train_shuffle=True, train_sampler=sampler_train, test_sampler=sampler_val)
    train_loader, test_loader, _ = loader

    results = {}
    results = train()
    if misc.get_rank() == 0:
        exp.save_eval_stats(results, 'linear_eval')
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