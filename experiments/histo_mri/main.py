from .. import cache_path, checkpoints_path, wandb_path

from ssregpros.datasets.histo_mri.histo_mri import HistoMri
from ssregpros.loss.composite import CompositeLossConfig
from ssregpros.models.registration import RegistrationNetworkConfig
from ssregpros.models.segmentor import Segmentor
from ssregpros.regularisation.rigid_transform import (
    RigidTransformRegularisationLossConfig,
)
from ssregpros.training.data_augmentation import DataAugmentation
from ssregpros.training.dataloader import (
    MultiModalPersistentDataLoader as DataLoader,
)
from ssregpros.training.dataset import (
    MultiModalDataset as Dataset,
    patient_stratified_split,
)
from ssregpros.training.logging.wandb_logger import WandBConfig, WandBLogger
from ssregpros.training.trainer import TrainingConfig, train_model
from ssregpros.transforms.preprocessor import (
    Preprocessor,
    PreprocessorConfig,
)

from pathlib import Path

import argparse
import torch
import yaml


FILE_DIR_PATH = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Train on Histo-MRI",
        description="Train on the Histo-MRI dataset",
    )
    parser.add_argument(
        "--dataset_dir",
        help="Path to the Histo-MRI dataset",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        help="Path to directory under which to save preprocessed data.",
        type=Path,
        required=False,
        default=cache_path(FILE_DIR_PATH),
    )
    parser.add_argument(
        "--checkpoint_dir",
        help="Directory under which to save checkpoints.",
        type=Path,
        required=False,
        default=checkpoints_path(FILE_DIR_PATH),
    )
    parser.add_argument(
        "--use_data_augmentation",
        help="Whether to employ the training data augmentation specified in `data-augmentation.yaml`",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--device",
        help="Device to train on: 'cuda', 'cpu'",
        required=False,
        default="cuda",
    )
    args = parser.parse_args()

    device = args.device.lower()
    if device == "cuda":
        if not (torch.backends.cuda.is_built() and torch.cuda.is_available()):
            print("CUDA unavailable, falling back to CPU")
            device = "cpu"
    elif device == "mps":
        print("MPS is currently unsupported :( Falling back to CPU")
        device = "cpu"
    elif device != "cpu":
        print("Unknown device, falling back to CPU")
        device = "cpu"
    args.device = torch.device(device)

    return args


def main():
    # Parse args.
    args = parse_args()
    device = args.device

    # Discover correspondences.
    cd = HistoMri(root_dir=args.dataset_dir)
    # Read config.
    with open(FILE_DIR_PATH / "config.yaml", "rb") as handle:
        config = yaml.safe_load(handle)
    # Create model config.
    model_config = RegistrationNetworkConfig(
        seed=config["seed"],
        **config["registration_network"],
        device=device,
    )
    # Create loss config.
    reg_config = RigidTransformRegularisationLossConfig(
        **config["loss"].pop("regularisation", {}),
        regression_head_shrinkage_range=model_config.regression_head_shrinkage_range,
    )
    loss_config = CompositeLossConfig(
        **config["loss"],
        transformation_regularisation_config=reg_config,
    )
    # Create training config.
    training_config = TrainingConfig(
        seed=config["seed"],
        loss_config=loss_config,
        checkpointer_root=args.checkpoint_dir,
        **config["training"],
    )
    # Handle training data augmentation.
    if args.use_data_augmentation:
        aug = DataAugmentation.from_yaml(
            FILE_DIR_PATH / "data-augmentation.yaml"
        )
    else:
        aug = DataAugmentation()
    # Create dataset.
    preprocessor_config = PreprocessorConfig(**config["preprocessor"])
    dataset = Dataset(
        correspondence_discoverer=cd,
        segmentor=Segmentor(device=device),
        preprocessor=Preprocessor(preprocessor_config),
        cache_dir=args.cache_dir,
        device=device,
    )
    # Create dataloaders.
    train, val, _ = patient_stratified_split(
        dataset,
        train=config["dataset_split"][0],
        val=config["dataset_split"][1],
        test=config["dataset_split"][2],
        seed=config["seed"],
        train_transform=aug.transform(),
    )
    train_dl = DataLoader(
        train, visualisation=False, batch_size=config["batch_size"]
    )
    val_dl = DataLoader(val, visualisation=True, batch_size=len(val))
    # Create logger.
    wandb_config = WandBConfig(
        batch_size=config["batch_size"],
        training_config=training_config,
        preprocessor_signature=dataset.preprocessor_signature,
        dataset_id=cd.dataset_id,
        dataset_split=config["dataset_split"],
        data_augmentation=aug,
        dir=wandb_path(FILE_DIR_PATH),
        mode="online",
        group="histo-mri",
    )
    logger = WandBLogger(
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        **wandb_config.wandb_init_kwargs(),
    )
    # Train.
    train_model(
        training_config=training_config,
        model_config=model_config,
        logger=logger,
        training_dataloader=train_dl,
        validation_dataloader=val_dl,
        verbose=True,
    )


if __name__ == "__main__":
    main()
