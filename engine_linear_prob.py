import torch
import lid
import util
import misc
import time
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, loader, scaler, exp, args):
    model.eval()
    device = args.gpu
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = loss.mean().item()
        # Calculate acc
        if hasattr(exp.config, 'eval_metric') and exp.config.eval_metric == 'mean_cls_acc':
            acc = util.mean_cls_acc(logits, labels)
            acc5 = None
        else:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc = acc.item()
            acc5 = acc5.item()
        # Update Meters
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc, n=batch_size)
        if acc5 is not None:
            metric_logger.update(acc5=acc5, n=batch_size)
    metric_logger.synchronize_between_processes()
    if acc5 is not None:
        payload = metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, metric_logger.meters['acc5'].global_avg
    else:
        payload = metric_logger.meters['loss'].global_avg, metric_logger.meters['acc'].global_avg, None
    return payload

def train_epoch(exp, model, optimizer, criterion, scaler, train_loader, global_step, epoch, args, logger):
    epoch_stats = {}
    device = args.gpu
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    # Training
    for i, data in enumerate(train_loader):
        start = time.time()
        if args.ddp:
            model.module.adjust_train_mode()
        else:
            model.adjust_train_mode()

        if 'lr_schedule_level' in exp.config and exp.config['lr_schedule_level'] == 'epoch':
            util.adjust_learning_rate(optimizer, epoch, exp.config)
        else:
            util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)

        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        model.zero_grad(set_to_none=True)

        # Objective function
        logits = model(images)
        loss = criterion(logits, labels)

        # Optimize
        loss.backward()
        optimizer.step()  
        # Calculate acc
        loss = loss.item()
        if hasattr(exp.config, 'eval_metric') and exp.config.eval_metric == 'mean_cls_acc':
            acc = util.mean_cls_acc(logits, labels)
            acc5 = None
        else:
            acc, acc5 = util.accuracy(logits, labels, topk=(1,5))
            acc = acc.item()
            acc5 = acc5.item()

        # Update Meters
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc, n=batch_size)
        if acc5 is not None:
            metric_logger.update(acc5=acc5, n=batch_size)
        # Log results
        end = time.time()
        time_used = end - start
        if global_step % exp.config.log_frequency == 0:
            loss = misc.all_reduce_mean(loss)
            acc = misc.all_reduce_mean(acc)
            metric_logger.synchronize_between_processes()
            payload = {
                "acc": acc,
                "acc_avg": metric_logger.meters['acc'].avg,
                "acc5_avg": metric_logger.meters['acc5'].avg if acc5 is not None else None,
                "loss": loss,
                "loss_avg": metric_logger.meters['loss'].avg,
                "lr": optimizer.param_groups[0]['lr']
            }
            display = util.log_display(epoch=epoch,
                                       global_step=global_step,
                                       time_elapse=time_used,
                                       **payload)
            if misc.get_rank() == 0:
                logger.info(display)
        # Update Global Step
        global_step += 1
    metric_logger.synchronize_between_processes()
    epoch_stats['epoch'] = epoch
    epoch_stats['global_step'] = global_step
    epoch_stats['train_acc'] = metric_logger.meters['acc'].global_avg
    epoch_stats['train_loss'] = metric_logger.meters['loss'].global_avg
    return epoch_stats
