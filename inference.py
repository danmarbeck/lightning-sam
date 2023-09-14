import argparse
import pickle as pkl
import segmentation_models_pytorch as smp
from pathlib import Path

import torch
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset import ResizeAndPad, DATASETS
from model import *
from utils import AverageMeter
import matplotlib.pyplot as plt
import numpy as np
from predictor import SamPredictor
import imageio

torch.set_float32_matmul_precision('high')


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if mask_image.max() > 1:
        mask_image = mask_image.astype(int)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=10):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def main(cfg: Box) -> None:
    device = torch.device("cuda:0")

    out_path = cfg.out_path
    if out_path is None:
        out_path = Path(Path(cfg.out_dir), cfg.config_name, "predictions")
    out_path.mkdir(parents=True, exist_ok=True)

    model = MODELS[cfg.model.name](cfg, inference=True)
    model.setup()
    model = model.to(device=torch.device("cuda:0"))

    transform = ResizeAndPad(model.get_img_size())
    dataset_cls = DATASETS[cfg.dataset.type]
    val_arg_dict = cfg.dataset.val.to_dict()
    val_arg_dict["inference"] = True
    val_arg_dict["transform"] = transform
    val_arg_dict["return_path"] = True
    val = dataset_cls(**val_arg_dict)

    ious = AverageMeter()
    f1_scores = AverageMeter()

    model.eval()

    with torch.no_grad():

        for iter, data in (pbar := tqdm(enumerate(val), total=len(val))):
            image, prompt_input, gt_masks, image_path, (H, W), (og_image, og_points, og_bboxes, og_mask_input, padding) = data
            image = image.to(device=device)[None, ...]
            gt_masks = gt_masks.to(device=device)

            point_coords = prompt_input["points"]
            point_labels = None
            if point_coords is not None:
                point_coords, point_labels = point_coords
                point_coords = point_coords.to(device=device)
                point_labels = point_labels.to(device=device)
            boxes = prompt_input["boxes"]
            if boxes is not None:
                boxes = boxes.to(device=device)
            mask_input = prompt_input["masks"]
            if mask_input is not None:
                mask_input = mask_input.to(device=device)

            prompt_input = {"points": (point_coords, point_labels) if point_coords is not None else None,
                            "boxes": boxes,
                            "masks": mask_input}

            pred_masks, iou_pred = model(image, (prompt_input,))

            # list with one tensor to tensor
            pred_masks = pred_masks[0]
            pred_masks = pred_masks.sigmoid()
            batch_stats = smp.metrics.get_stats(
                pred_masks.unsqueeze(0),
                gt_masks.int().unsqueeze(0),
                mode='binary',
                threshold=0.5,
            )

            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            ious.update(batch_iou)
            f1_scores.update(batch_f1)

            # unpad
            pred_masks = pred_masks[..., padding[1]:None if padding[3] == 0 else -padding[3],
                                       padding[0]:None if padding[2] == 0 else -padding[2]]

            masks = F.interpolate(
                pred_masks.unsqueeze(0),
                (H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            masks = masks > 0.5

            pbar.set_description(f'Val: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]', refresh=True)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.imshow(og_image)

            for pred_idx in range(len(masks)):
                show_mask(masks[pred_idx].detach().cpu().numpy(), ax=ax, random_color=True)
                if point_coords is not None:
                    show_points(og_points[pred_idx], point_labels[pred_idx].detach().cpu().numpy(), ax=ax,)
                if mask_input is not None:
                    show_mask(og_mask_input[pred_idx], ax=ax, random_color=True)
                if boxes is not None:
                    show_box(og_bboxes[pred_idx], ax=ax)

            fig.savefig(Path(out_path, Path(image_path).stem + ".png"))
            plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Infer the model on the validation dataset using the corresponding training config")
    parser.add_argument("--config", default="configs/base_config.yaml", type=str,
                        help="Path to .yaml file containing the config.")
    parser.add_argument("--out_path", default=None,
                        help="Path to output_folder for the predictions. If None, model location is used.")
    parser.add_argument("--model_path", "-m", type=str, help="Path to the model used for inference."
                                                             " Must match the provided config in terms of model type.")

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

    cfg.model.checkpoint = args.model_path  # override initial model path to fine-tuned model
    cfg["out_path"] = args.out_path
    main(cfg)
