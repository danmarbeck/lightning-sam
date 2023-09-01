import argparse
import pickle as pkl
from pathlib import Path

import torch
from box import Box
from tqdm import tqdm

from config import cfg
from dataset import load_datasets
from model import PrecomputeModel

torch.set_float32_matmul_precision('high')


def main(cfg: Box) -> None:
    model = PrecomputeModel(cfg)
    model.setup()
    model = model.to(device=torch.device("cuda:0"))

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size, return_path=True)

    model.eval()

    with torch.no_grad():
        base_path = Path(cfg.out_path, "train")
        base_path.mkdir(exist_ok=True, parents=True)
        for iter, data in tqdm(enumerate(train_data), total=len(train_data)):
            images, bboxes, gt_masks, image_paths = data
            num_images = images.size(0)
            images = images.to(device=torch.device("cuda:0"))
            embeddings = model(images)
            embeddings = embeddings.detach().cpu().numpy()
            assert embeddings.shape[0] == num_images

            for idx, embedding in enumerate(embeddings):
                image_path = Path(image_paths[idx])

                pkl.dump(embedding, open(Path(base_path, image_path.stem + ".pkl"), "wb"))

        base_path = Path(cfg.out_path, "val")
        base_path.mkdir(exist_ok=True, parents=True)
        for iter, data in tqdm(enumerate(val_data), total=len(val_data)):
            images, bboxes, gt_masks, image_paths = data
            num_images = images.size(0)
            images = images.to(device=torch.device("cuda:0"))
            embeddings = model(images)
            embeddings = embeddings.detach().cpu().numpy()
            assert embeddings.shape[0] == num_images

            for idx, embedding in enumerate(embeddings):
                image_path = Path(image_paths[idx])

                pkl.dump(embedding, open(Path(base_path, image_path.stem + ".pkl"), "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Precompute embeddings for dataset using the corresponding training config")
    parser.add_argument("--config", default="configs/base_config.yaml", type=str,
                        help="Path to .yaml file containing the config.")
    parser.add_argument("--out_path", default="/data/coco/sam_embeddings/",
                        help="Path to base output folder for the embeddings")

    args = parser.parse_args()
    if args.config is not None:
        try:
            cfg = Box.from_yaml(filename=str(Path(args.config).absolute()))
        except Exception as e:
            print("Failed to load config:")
            print(e)
            print("Using default config instead")
    cfg["out_path"] = args.out_path
    main(cfg)
