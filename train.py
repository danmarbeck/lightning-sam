import argparse
import os
import time
from pathlib import Path

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from lightning.fabric.strategies import DDPStrategy

from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import MODELS, Model
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from utils import AverageMeter
from utils import calc_iou

torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, prompt_input, gt_masks = data
            num_images = images.size(0)
            pred_masks, _ = model(images, prompt_input)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                pred_mask = pred_mask.sigmoid()
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.state_dict()
    if fabric.global_rank == 0 and epoch % cfg.save_interval == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"{Path(fabric.logger.log_dir).name}_epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()

    return {"iou_val": ious.avg, "f1_avg": f1_scores.avg}


def train_sam(
        cfg: Box,
        fabric: L.Fabric,
        model: Model,
        optimizer: _FabricOptimizer,
        scheduler: _FabricOptimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in enumerate(train_dataloader):
            if (epoch == 1 or epoch % cfg.eval_interval == 0) and not validated:
                val_metrics = validate(fabric, model, val_dataloader, epoch)
                fabric.log_dict(val_metrics, step=(epoch - 1) * len(train_dataloader))
                validated = True

            data_time.update(time.time() - end)
            images, prompt_input, gt_masks = data
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, prompt_input)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask) / num_masks
                loss_dice += dice_loss(pred_mask, gt_mask) / num_masks
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.moving_avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.moving_avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.moving_avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.moving_avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.moving_avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.moving_avg:.4f})]')
            fabric.log_dict({"focal_loss": focal_losses.val,
                             "dice_loss": dice_losses.val,
                             "iou_loss": iou_losses.val,
                             "total_loss": total_losses.val,
                             "batch_time": batch_time.val,
                             "data_time": data_time.val,
                             }, step=(epoch - 1) * len(train_dataloader) + iter)
        fabric.log_dict({"lr": scheduler.get_last_lr()[0]}, step=epoch * len(train_dataloader))


def configure_opt(cfg: Box, model, num_steps_per_epoch):
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

    def lr_lambda_exp(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        else:
            return 0.95 ** (step // num_steps_per_epoch)

    optimizer = torch.optim.AdamW(model.get_parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_exp)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    Path(cfg.out_dir).mkdir(exist_ok=True, parents=True)
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy=DDPStrategy(start_method="popen", find_unused_parameters=True),
                      loggers=[TensorBoardLogger(cfg.out_dir, name=cfg.config_name)])
    cfg.out_dir = Path(cfg.out_dir, cfg.config_name)
    cfg.out_dir.mkdir(exist_ok=True, parents=True)
    fabric.launch()
    fabric.seed_everything((np.random.randint(1, 420) if args.seed else 1337) + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = MODELS[cfg.model.name](cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, model.get_img_size())
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model, num_steps_per_epoch=len(train_data))
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    val_metrics = validate(fabric, model, val_data, epoch=cfg.num_epochs)
    fabric.log_dict(val_metrics, step=(cfg.num_epochs - 1) * len(train_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune SAM using the corresponding training config")
    parser.add_argument("--config", default="configs/base_config.yaml", type=str,
                        help="Path to .yaml file containing the config.")
    parser.add_argument("--seed", action="store_true", help="if set, use random seed for init")

    args = parser.parse_args()
    if args.config is not None:
        try:
            cfg = Box.from_yaml(filename=str(Path(args.config).absolute()))
        except Exception as e:
            print("Failed to load config:")
            print(e)
            print("Using default config instead")
        cfg["config_name"] = Path(args.config).stem
    else:
        cfg["config_name"] = "internal_config"
    if "num_nodes" not in cfg.keys():
        cfg["num_nodes"] = 1
    main(cfg)
