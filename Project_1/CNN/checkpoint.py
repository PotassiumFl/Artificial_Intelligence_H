from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

import config
from model import ChineseCharCNN


def save_cnn_checkpoint(
    path: Union[str, Path],
    model: ChineseCharCNN,
    num_classes: int,
    random_state: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "random_state": random_state,
        "val_fraction": config.VAL_FRACTION,
        "class_image_size": tuple(config.CLASS_IMAGE_SIZE),
    }
    torch.save(payload, path)


def load_cnn_model(
    path: Union[str, Path], device: torch.device
) -> Tuple[ChineseCharCNN, Dict[str, Any]]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model = ChineseCharCNN(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
