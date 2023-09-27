import argparse
import pickle as pkl
import re

from PIL import Image, ImageDraw, ImageOps
import cv2
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

colors = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                  [145, 30, 180], [70, 240, 240],[240, 50, 230], [210, 245, 60], [250, 190, 212],
                  [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
                  [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
                  [200, 200, 25], [0, 0, 0], [0, 130, 255]]

color_scheme = {'background' : 0,
              'aeroplane' : 1,
              'bicycle' : 2,
              'bird' : 3,
              'boat' : 4,
              'bottle' : 5,
              'bus' : 6,
              'car' : 7,
              'cat' : 8,
              'chair' : 9,
              'cow' : 10,
              'diningtable' : 11,
              'dog' : 12,
              'horse' : 13,
              'motorbike' : 14,
              'person' : 15,
              'pottedplant' : 16,
              'sheep' : 17,
              'sofa' : 18,
              'train' : 19,
              'tvmonitor' : 20,
                'cells' : 22}


def crop_from_bbox(img, bbox, zero_pad=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        #assert (bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]

    return crop


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

    out_path = Path(cfg.out_path)
    if out_path is None:
        out_path = Path(Path(cfg.out_dir), cfg.config_name, "predictions")
    if args.object_based:
        out_path = Path(out_path, "object_based")
    else:
        out_path = Path(out_path, "image_based")
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

    rng = np.random.default_rng(4096)

    model.eval()

    with torch.no_grad():

        for iter, data in (pbar := tqdm(enumerate(val), total=len(val))):
            image, prompt_input, gt_masks, image_path, (H, W), (image_info, og_image, og_masks, og_points, og_bboxes, og_mask_input, classes,padding) = data
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

            if args.object_based:

                for pred_idx in range(len(masks)):
                    # calculate cropping coords etc
                    bbox = [int(v) for v in og_bboxes[pred_idx]]
                    bbox = [bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10]

                    # original image
                    cv_image = og_image.copy()
                    cropped_image = crop_from_bbox(cv_image, bbox)
                    imageio.v3.imwrite(Path(out_path, Path(image_path).stem + f"_{pred_idx}_0_orig.png"), cropped_image)

                    # image_with_prediction
                    cv_image = Image.fromarray(og_image.copy())
                    mask = masks[pred_idx].detach().cpu().numpy().astype(np.uint8)
                    color_cat = colors[color_scheme[classes[pred_idx]]]
                    mask_image = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    mask_image[mask.astype(bool)] = (color_cat[0], color_cat[1], color_cat[2], 200)

                    cv_image = Image.alpha_composite(cv_image.convert("RGBA"), Image.fromarray(mask_image)).convert("RGB")
                    cv_image_draw = ImageDraw.Draw(cv_image, mode="RGBA")

                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                    for idx, c in enumerate(contours):
                        coordinates = []
                        for cc in c:
                            coordinates.append((cc[0][0], cc[0][1]))

                        if len(coordinates) >= 2:
                            cv_image_draw.polygon(coordinates, fill=None, outline='black')
                    cropped_image = crop_from_bbox(np.asarray(cv_image, dtype=np.uint8), bbox)
                    imageio.v3.imwrite(Path(out_path, Path(image_path).stem + f"_{pred_idx}_3_pred.png"), cropped_image)

                    # image with gt
                    cv_image = Image.fromarray(og_image.copy())
                    mask = og_masks[pred_idx].astype(np.uint8)
                    color_cat = colors[color_scheme[classes[pred_idx]]]
                    mask_image = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    mask_image[mask.astype(bool)] = (color_cat[0], color_cat[1], color_cat[2], 200)

                    cv_image = Image.alpha_composite(cv_image.convert("RGBA"), Image.fromarray(mask_image)).convert(
                        "RGB")
                    cv_image_draw = ImageDraw.Draw(cv_image, mode="RGBA")

                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                    for idx, c in enumerate(contours):
                        coordinates = []
                        for cc in c:
                            coordinates.append((cc[0][0], cc[0][1]))

                        if len(coordinates) >= 2:
                            cv_image_draw.polygon(coordinates, fill=None, outline='black')
                    cropped_image = crop_from_bbox(np.asarray(cv_image, dtype=np.uint8), bbox)
                    imageio.v3.imwrite(Path(out_path, Path(image_path).stem + f"_{pred_idx}_1_gt.png"),
                                       cropped_image)

                    # image with input
                    cv_image = Image.fromarray(og_image.copy())

                    if "masks" in val.prompt_types and val.mask_type == "gaussian":
                        mask_image = np.stack([og_mask_input[pred_idx].astype(bool).astype(np.uint8)] * 3, axis=2)
                        mask_image[np.where((mask_image == [1, 1, 1, ]).all(axis=2))] = [255, 0, 0]
                        mask_image = np.append(mask_image, (og_mask_input[pred_idx] * 255).astype(np.uint8)[..., None], axis=2)
                        cv_image = cv_image.convert("RGBA")
                        cv_image.alpha_composite(Image.fromarray(mask_image))
                        cv_image = cv_image.convert("RGB")
                    elif "masks" in val.prompt_types:
                        mask_path = Path(image_info["heatmap_paths"][val.mask_type][pred_idx])
                        mask_image_path = Path(mask_path.parent.parent, re.sub('_mask', "", mask_path.parent.name),
                                               mask_path.stem)
                        mask_image = imageio.v3.imread(mask_image_path)
                        if mask_image.shape != cv_image.size[::-1]:
                            mask_image = np.array(ImageOps.pad(Image.fromarray(mask_image), cv_image.size))
                        cv_image = Image.fromarray(mask_image)
                        cv_image = cv_image.convert("RGB")
                    cv_image_draw = ImageDraw.Draw(cv_image, mode="RGBA")
                    if "points" in val.prompt_types or "fixations" in val.prompt_types:

                        points = og_points[pred_idx]
                        for point in points:
                            cv_image_draw.ellipse([int(point[0]) - 2, int(point[1]) - 2, int(point[0]) + 2, int(point[1]) + 2],
                                                  fill=(255, 0, 0, 255), outline="red")
                    if "boxes" in val.prompt_types:
                        og_bbox = [int(v) for v in og_bboxes[pred_idx]]

                        cv_image_draw.rectangle(og_bbox, fill=(0,0,0,0), outline="red", width=3)

                    cropped_image = crop_from_bbox(np.asarray(cv_image, dtype=np.uint8), bbox)
                    imageio.v3.imwrite(Path(out_path, Path(image_path).stem + f"_{pred_idx}_2_input.png"),
                                       cropped_image)
            else:
                #  plot whole images and all their objects
                cv_image = Image.fromarray(og_image.copy())

                for pred_idx in range(len(masks)):
                    # draw prediction
                    mask = masks[pred_idx].detach().cpu().numpy().astype(np.uint8)
                    color_cat = colors[color_scheme[classes[pred_idx]]]
                    mask_image = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    mask_image[mask.astype(bool)] = (color_cat[0], color_cat[1], color_cat[2], 200)

                    cv_image = Image.alpha_composite(cv_image.convert("RGBA"), Image.fromarray(mask_image)).convert(
                        "RGB")
                    cv_image_draw = ImageDraw.Draw(cv_image, mode="RGBA")

                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                    for idx, c in enumerate(contours):
                        coordinates = []
                        for cc in c:
                            coordinates.append((cc[0][0], cc[0][1]))

                        if len(coordinates) >= 2:
                            cv_image_draw.polygon(coordinates, fill=None, outline='black')

                    # draw input
                    if "masks" in val.prompt_types and val.mask_type == "gaussian":
                        mask_image = np.stack([og_mask_input[pred_idx].astype(bool).astype(np.uint8)] * 3, axis=2)
                        mask_image[np.where((mask_image == [1, 1, 1, ]).all(axis=2))] = [255, 0, 0]
                        mask_image = np.append(mask_image, (og_mask_input[pred_idx] * 255).astype(np.uint8)[..., None], axis=2)
                        cv_image = cv_image.convert("RGBA")
                        cv_image.alpha_composite(Image.fromarray(mask_image))
                        cv_image = cv_image.convert("RGB")
                    elif "masks" in val.prompt_types:
                        mask_path = Path(image_info["heatmap_paths"][val.mask_type][pred_idx])
                        mask_image_path = Path(mask_path.parent.parent, re.sub('_mask', "", mask_path.parent.name),
                                               mask_path.stem)
                        mask_image = imageio.v3.imread(mask_image_path)
                        if mask_image.shape != cv_image.size[::-1]:
                            mask_image = np.array(ImageOps.pad(Image.fromarray(mask_image), cv_image.size))
                        cv_image = Image.fromarray(mask_image)
                        cv_image = cv_image.convert("RGB")
                    cv_image_draw = ImageDraw.Draw(cv_image, mode="RGBA")
                    if "points" in val.prompt_types or "fixations" in val.prompt_types:

                        points = og_points[pred_idx]
                        for point in points:
                            cv_image_draw.ellipse([int(point[0]) - 4, int(point[1]) - 4, int(point[0]) + 4, int(point[1]) + 4],
                                                  fill=(*color_cat, 255), outline="white")
                    if "boxes" in val.prompt_types:
                        og_bbox = [int(v) for v in og_bboxes[pred_idx]]

                        cv_image_draw.rectangle(og_bbox, fill=(0,0,0,0), outline="red")
                imageio.v3.imwrite(Path(out_path, Path(image_path).stem + ".png"), np.asarray(cv_image))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Infer the model on the validation dataset using the corresponding training config")
    parser.add_argument("--config", default="configs/base_config.yaml", type=str,
                        help="Path to .yaml file containing the config.")
    parser.add_argument("--out_path", default=None,
                        help="Path to output_folder for the predictions. If None, model location is used.")
    parser.add_argument("--object_based", action="store_true",
                        help="if set, store images for each object.")
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
