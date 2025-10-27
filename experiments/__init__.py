from pathlib import Path


def cache_path(root: Path) -> Path:
    return root / "cache"


def checkpoints_path(root: Path) -> Path:
    return root / "checkpoints"


def wandb_path(root: Path) -> Path:
    return root / "wandb"
