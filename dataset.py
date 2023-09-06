import os
import re
from pathlib import Path

import imageio
import numpy as np
import pickle as pkl
import pycocotools
import torch
import torchvision.transforms as transforms
import yaml
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from typing import Tuple


class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None, return_path=False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = imageio.v3.imread(image_path)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        if self.return_path:
            return image, {"boxes": torch.tensor(bboxes)}, torch.tensor(masks).float(), image_path
        else:
            return image, {"boxes": torch.tensor(bboxes)}, torch.tensor(masks).float()


class PascalVOCDataset(Dataset):
    """
        PascalVOCDataset for image-based gaze version of the dataset, including the option of precomputed embeddings
        @param root_dir: Path to folder containing the gaze images, masks and original etc. as folders, per class
        @param prompt_types: Tuple containing the used prompt types. Choose out of: points, boxes, masks
    """

    def __init__(self, root_dir, inference=False, transform=None, return_path=False, use_embeddings=False,
                 prompt_types: Tuple = ("boxes",)):
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        self.split = Path(self.root_dir).name
        self.use_embeddings = use_embeddings
        self.inference = inference
        assert all(prompt_type in ["boxes", "points", "masks"] for prompt_type in prompt_types), "Invalid prompt type"
        self.prompt_types = prompt_types
        self.image_dict = self._get_image_dict()

        self.image_ids = list(self.image_dict.keys())

    def _get_image_dict(self):

        if Path(self.root_dir, "sam_dataset_info_dict.yaml").exists():
            return yaml.load(open(Path(self.root_dir, "sam_dataset_info_dict.yaml"), "r"), yaml.FullLoader)
        else:
            return self._construct_image_dict()

    def _construct_image_dict(self):
        """
            Constructs a dict, keys are image ids, dict contains list of bboxes and paths to additional info, and saves it to disk
        """
        image_dict = {}
        print("Constructing image dict...")

        bbox_regex = re.compile("\d{4}_\d{6}_x_min=(\d+)_x_max=(\d+)_y_min=(\d+)_y_max=(\d+)")

        for class_dir in Path(self.root_dir).iterdir():
            if not class_dir.is_dir():
                continue
            image_paths = list(Path(class_dir, "original").glob("*.png"))

            for image_path in image_paths:

                gaze_path = Path(image_path.parent.parent, "gaze_images", image_path.name).absolute()

                fileid = "_".join(image_path.stem.split("_")[:2])

                if fileid not in image_dict.keys():
                    r = {
                        "original": Path(Path(self.root_dir).parent.parent, "JPEGImages",
                                         fileid + ".jpg").absolute().__str__(),
                        "image_id": fileid,
                        "bboxes": [],
                        "masks": [],
                        "gaze_paths": [],
                    }
                    image_dict[fileid] = r

                bbox_wrong_order = list(map(float, bbox_regex.match(str(image_path.stem)).groups()))
                bbox = [bbox_wrong_order[2], bbox_wrong_order[0], bbox_wrong_order[3],
                        bbox_wrong_order[1]]  # x_min, y_min, x_max, y_max

                mask_array = imageio.v3.imread(Path(image_path.parent.parent, "masks", image_path.name))
                if len(mask_array.shape) == 3:
                    mask_array = np.sum(mask_array, axis=2, dtype=int)
                mask_array = mask_array.astype(bool).astype(np.uint8)
                segmentation_rle_dict = pycocotools.mask.encode(np.asarray(mask_array, order="F"))
                assert type(segmentation_rle_dict) == dict, type(segmentation_rle_dict)

                image_dict[fileid]["gaze_paths"].append(str(gaze_path))
                image_dict[fileid]["bboxes"].append(bbox)
                image_dict[fileid]["masks"].append(segmentation_rle_dict)

        yaml.dump(image_dict, open(Path(self.root_dir, "sam_dataset_info_dict.yaml"), "w"))
        return image_dict

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_dict[image_id]
        image_path = image_info["original"]
        image = imageio.v3.imread(image_path)
        og_h, og_w, _ = image.shape

        if self.use_embeddings:
            image_embedding = pkl.load(
                open(Path(Path(self.root_dir).parent.parent, "sam_embeddings", self.split, image_id + ".pkl"), "rb"))
            image_embedding = torch.tensor(image_embedding)
        else:
            image_embedding = None

        points = [] # TODO
        bboxes = image_info["bboxes"]
        masks_rle = image_info["masks"]
        masks = [pycocotools.mask.decode(rle) for rle in masks_rle]

        gaze_mask_paths = image_info["gaze_paths"]
        gaze_masks = [imageio.v3.imread(gaze_path) for gaze_path in gaze_mask_paths]

        if self.inference:
            original_data = (image.copy(), points.copy(), bboxes.copy(), gaze_masks.copy())

        if self.transform:
            image, masks, bboxes, gaze_masks = self.transform(image, masks, np.array(bboxes), gaze_masks)

        masks = np.stack(masks, axis=0)
        bboxes = np.stack(bboxes, axis=0)
        gaze_masks = np.stack(gaze_masks, axis=0)

        prompt_dict = {"points": None, "boxes": None, "masks": None}
        if "points" in self.prompt_types:
            raise NotImplementedError
        if "boxes" in self.prompt_types:
            prompt_dict["boxes"] = torch.tensor(bboxes)
        if "masks" in self.prompt_types:
            prompt_dict["masks"] = torch.tensor(gaze_masks).float()

        return_list = [prompt_dict, torch.tensor(masks).float()]
        if self.return_path:
            return_list.append(image_path)
        if self.inference:
            return_list.append((og_h, og_w))
            return_list.append(original_data)

        image_return = image_embedding if self.use_embeddings and not self.inference else image

        return image_return, *return_list


def collate_fn(batch):
    images, prompts, *rest = tuple(zip(*batch))
    images = torch.stack(images)
    return images, prompts, *rest


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.gaze_transform = transforms.GaussianBlur(kernel_size=7, sigma=5)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes, gaze_masks=None):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        if gaze_masks is not None:
            gaze_masks = [torch.tensor(self.transform.apply_image(gaze_mask)).bool().float() for gaze_mask in
                          gaze_masks]
            gaze_masks = [self.gaze_transform(gaze_mask[None, ...]) for gaze_mask in gaze_masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]
        if gaze_masks is not None:
            gaze_masks = [transforms.Pad(padding)(gaze_mask) for gaze_mask in gaze_masks]
            gaze_masks = [transforms.Resize(self.target_size // 4, antialias=True)(gaze_mask) for gaze_mask in
                          gaze_masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        if gaze_masks is not None:
            return image, masks, bboxes, gaze_masks
        else:
            return image, masks, bboxes


DATASETS = {"coco": COCODataset,
            "pascal": PascalVOCDataset,
            }


def load_datasets(cfg, img_size, return_path=False):
    transform = ResizeAndPad(img_size)
    dataset_cls = DATASETS[cfg.dataset.type]

    train_arg_dict = cfg.dataset.train.to_dict()
    train_arg_dict["transform"] = transform
    train_arg_dict["return_path"] = return_path
    train = dataset_cls(**train_arg_dict)

    val_arg_dict = cfg.dataset.val.to_dict()
    val_arg_dict["transform"] = transform
    val_arg_dict["return_path"] = return_path
    val = dataset_cls(**val_arg_dict)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader
