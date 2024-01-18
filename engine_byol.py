import torch
import lid
import util
import misc
import time
import math
from lid import gmean


@torch.no_grad()
def evaluate(model, test_loader, args, configs):
    model.eval()
    device = args.gpu
    # extract features
    lids_k32 = []
    lids_k512 = []
    metric_logger = misc.MetricLogger(delimiter="  ")
    for images, labels in test_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(images)
        fe = out[0]
        cls = out[-1]
        online_acc = util.accuracy(cls, labels)[0]
        metric_logger.update(online_acc=online_acc.item())
        if args.ddp:
            full_rank_fe = torch.cat(misc.gather(fe), dim=0)
        else:
            full_rank_fe = fe
        lids_k32.append(lid.lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=32))
        lids_k512.append(lid.lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=512))

    lids_k32 = torch.cat(lids_k32, dim=0)
    lids_k512 = torch.cat(lids_k512, dim=0)
    if args.ddp:
        lids_k32 = torch.cat(misc.gather(lids_k32.to(device)), dim=0)
        lids_k512 = torch.cat(misc.gather(lids_k512.to(device)), dim=0)
    metric_logger.synchronize_between_processes()
    lids_k32 = torch.nan_to_num(lids_k32, nan=0.0)
    lids_k512 = torch.nan_to_num(lids_k512, nan=0.0)
    return lids_k32.detach().cpu(), lids_k512.detach().cpu(), metric_logger.meters['online_acc'].global_avg


@torch.no_grad()
def evaluate_full_set_lid(model, loader, args, configs):
    model.eval()
    device = args.gpu
    # extract features
    features = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        fe = out[0]
        features.append(fe)
    
    features = torch.cat(features, dim=0)
    if args.ddp:
        full_rank_fe = torch.cat(misc.gather(features), dim=0)
    else:
        full_rank_fe = features
    lids_k32 = lid.lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=32)
    lids_k512 = lid.lid_mle(data=fe.detach(), reference=full_rank_fe.detach(), k=512)
    if args.ddp:
        lids_k32 = torch.cat(misc.gather(lids_k32.to(device)), dim=0)
        lids_k512 = torch.cat(misc.gather(lids_k512.to(device)), dim=0)
    return lids_k32.detach().cpu(), lids_k512.detach().cpu()


def train_epoch(exp, model, model_momentum, optimizer, optimizer_online, online_lr,
                criterion, scaler, train_loader, global_step, epoch, logger, args):
    epoch_stats = {}
    device = args.gpu
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # Training
    model.train()
    model_momentum.train()
    for param in model.parameters():
        param.grad = None
    for i, data in enumerate(train_loader):
        start = time.time()
        # Adjust LR
        util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        util.adjust_learning_rate_with_params(
            optimizer=optimizer_online, 
            epoch=i / len(train_loader) + epoch, 
            min_lr=0.0,
            lr=online_lr,
            warmup=0,
            epochs=exp.config.epochs)
        # Train step
        images, online_labels = data
        images = images[0].to(device, non_blocking=True), images[1].to(device, non_blocking=True)
        online_labels = online_labels.to(device, non_blocking=True)
        m = util.adjust_momentum(i / len(train_loader) + epoch, exp.config)
        util.update_momentum(model, model_momentum, m=m)
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = criterion(model, model_momentum, images, online_labels)
            loss = results['loss']
            logits = results['logits']
            labels = results['labels']
        # Optimize
        loss = results['loss']
        if torch.isnan(loss):
            if misc.get_rank() == 0:
                logger.info('Skip this batch, loss is nan!')
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            if hasattr(exp.config, 'grad_clip'):
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                if hasattr(exp.config, 'grad_clip'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), exp.config.grad_clip)
            scaler.step(optimizer)
            scaler.step(optimizer_online)
            scaler.update()
        else:
            loss.backward()
            if hasattr(exp.config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), exp.config.grad_clip)
            optimizer.step()
            optimizer_online.step()
        loss = loss.item()
        # Calculate acc
        acc, _ = util.accuracy(logits, labels, topk=(1, 5))
        # Update Meters
        batch_size = logits.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc.item(), n=batch_size)
        metric_logger.update(lid32_avg=results['lids32'].mean().item())
        metric_logger.update(lid32_var=results['lids32'].var().item())
        metric_logger.update(lid512_avg=results['lids512'].mean().item())
        metric_logger.update(lid512_var=results['lids512'].var().item())
        metric_logger.update(lid32_gavg=gmean(results['lids32']).item())
        metric_logger.update(lid512_gavg=gmean(results['lids512']).item())
        metric_logger.update(main_loss=results['main_loss'])
        metric_logger.update(online_acc=results['online_acc'])
        if 'reg_loss' in results:
            metric_logger.update(reg_loss=results['reg_loss'])
        # Log results
        end = time.time()
        time_used = end - start
        # track LR
        lr = optimizer.param_groups[0]['lr']
        if global_step % exp.config.log_frequency == 0:
            metric_logger.synchronize_between_processes()
            payload = {
                "lr": lr,
                "acc_avg": metric_logger.meters['acc'].avg,
                "online_acc": metric_logger.meters['online_acc'].avg,
                "lid32_gavg": metric_logger.meters['lid32_gavg'].avg,
                "lid512_gavg": metric_logger.meters['lid512_gavg'].avg,
                "loss_avg": metric_logger.meters['loss'].avg,
                "main_loss": metric_logger.meters['main_loss'].avg,
            }
            if 'reg_loss' in results:
                payload['reg_loss'] = metric_logger.meters['reg_loss'].avg
            if misc.get_rank() == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=global_step,
                                           time_elapse=time_used,
                                           **payload)
                logger.info(display)
        # Update Global Step
        global_step += 1

    metric_logger.synchronize_between_processes()
    epoch_stats['epoch'] = epoch
    epoch_stats['global_step'] = global_step
    epoch_stats['train_acc'] = metric_logger.meters['acc'].global_avg
    epoch_stats['train_online_acc'] = metric_logger.meters['online_acc'].global_avg
    epoch_stats['train_loss'] = metric_logger.meters['loss'].global_avg
    epoch_stats['train_lid32_avg'] = metric_logger.meters['lid32_avg'].global_avg
    epoch_stats['train_lid32_var'] = metric_logger.meters['lid32_var'].global_avg
    epoch_stats['train_lid512_avg'] = metric_logger.meters['lid512_avg'].global_avg
    epoch_stats['train_lid512_var'] = metric_logger.meters['lid512_var'].global_avg
    epoch_stats['train_lid32_gavg'] = metric_logger.meters['lid32_gavg'].global_avg
    epoch_stats['train_lid512_gavg'] = metric_logger.meters['lid512_gavg'].global_avg
    epoch_stats['main_loss'] = metric_logger.meters['main_loss'].global_avg
    if 'reg_loss' in results:
        epoch_stats['reg_loss'] = metric_logger.meters['reg_loss'].global_avg
    return epoch_stats
