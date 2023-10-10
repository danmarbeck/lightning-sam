import itertools

import matplotlib.pyplot as plt
import numpy as np
import imageio
from pathlib import Path
import cv2

from tqdm import tqdm

from plot_lr_scan import METHOD_PRINT_NAMES

OBJECT_IDS = [
    "2007_000129_4",
    "2007_000452_1",
    "2007_000572_1",
    "2007_000661_0",
    "2007_000830_0",
    "2007_001239_0",
    "2007_002376_0",
    "2007_002565_0",
    "2007_004510_1",
    "2007_005705_0",
    "2007_006035_0",
    "2007_008260_1",
]

IMAGE_IDS = [
    "2007_003011",
    "2008_000391",
    "2008_006254",
    "2008_008362",
    "2009_000573",
    "2009_003378",
    "2010_001024",
    "2010_001767",
    "2007_005547",
    "2007_002260",
    "2009_001299",
    "2009_002415",
    "2008_000215",
    "2011_002548",
    "2011_001614",
    "2009_004455",
    "2009_005302",
    "2010_001851",
]


def plot_input_comparison(args):

    base_path = Path("/home/daniel/PycharmProjects/lightning-sam/out/training/pascal_final_models/")
    method_folders = list(base_path.glob("precomputed*"))
    out_path = Path("/home/daniel/PycharmProjects/lightning-sam/plotting/object_based/")
    out_path.mkdir(parents=True, exist_ok=True)

    methods = list(METHOD_PRINT_NAMES.keys())
    method_folder_names = {method: "precomputed_" + method + ("_" if len(method) > 0 else "") + "pascal" for method in methods}

    for object_id in tqdm(OBJECT_IDS, total=len(OBJECT_IDS)):

        og_image = imageio.v3.imread(Path(base_path, method_folder_names["fixations"], "object_based", object_id + "_0_orig.png"))
        #og_image = cv2.resize(og_image, (img_size, img_size))
        gt_image = imageio.v3.imread(Path(base_path, method_folder_names["fixations"], "object_based", object_id + "_1_gt.png"))
        #gt_image = cv2.resize(gt_image, (img_size, img_size))

        method_input_images = [
            imageio.v3.imread(Path(base_path, method_folder_names[method], "object_based", object_id + "_2_input.png"))
            for method in methods
        ]

        method_pred_images = [
            imageio.v3.imread(Path(base_path, method_folder_names[method], "object_based", object_id + "_3_pred.png"))
            for method in methods
        ]

        #method_input_images = [cv2.resize(img, (img_size, img_size)) for img in method_input_images]
        #method_pred_images = [cv2.resize(img, (img_size, img_size)) for img in method_pred_images]

        full_image_upper_row = np.concatenate([og_image] + method_input_images, axis=1)
        full_image_lower_row = np.concatenate([gt_image] + method_pred_images, axis=1)

        full_image = np.concatenate([full_image_upper_row, full_image_lower_row], axis=0)

        imageio.v3.imwrite(Path(out_path, object_id + ".png"), full_image)


def plot_full_images(args):

    base_path = Path("/home/daniel/PycharmProjects/lightning-sam/out/training/pascal_final_models/precomputed_fixations_pascal/image_based")
    out_path = Path("/home/daniel/PycharmProjects/lightning-sam/plotting/image_based/")
    out_path.mkdir(parents=True, exist_ok=True)

    for image_id in IMAGE_IDS:
        image = imageio.v3.imread(Path(base_path, image_id + ".png"))

        imageio.v3.imwrite(Path(out_path, image_id + ".png"), image)


if __name__ == '__main__':

    args = None

    plot_input_comparison(args)
    # plot_full_images(args)
