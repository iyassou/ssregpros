from ssregpros.datasets.fake_histo_mri.fake_histo_mri import FakeHistoMri
from ssregpros.loss.composite import CompositeLossConfig
from ssregpros.models.registration import RegistrationNetworkConfig
from ssregpros.models.segmentor import Segmentor, SegmentorConfig
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
from ssregpros.training.logging.noop_logger import NoOpLogger
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
        prog="Example Experiment",
        description="Train on NCC for 5+5 epochs on fake correspondences",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file",
        required=False,
        type=Path,
        default=FILE_DIR_PATH / "config.yaml",
    )
    parser.add_argument(
        "--device",
        help="Device to train on: 'cuda', 'cpu'",
        required=False,
        default="cpu",
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


def split_dataset(
    seed: int,
    segmentor_config: SegmentorConfig,
    preprocessor_config: PreprocessorConfig,
    training_data_augmentation: DataAugmentation,
    split: tuple[float, float, float],
):
    # Create dataset.
    dataset = Dataset(
        correspondence_discoverer=FakeHistoMri(),
        segmentor=Segmentor(segmentor_config),
        preprocessor=Preprocessor(preprocessor_config),
        cache_dir=FILE_DIR_PATH / "cache",
        device=segmentor_config.device,
    )
    # Split dataset.
    train, val, test = patient_stratified_split(
        dataset,
        train=split[0],
        val=split[1],
        test=split[2],
        seed=seed,
        train_transform=training_data_augmentation.transform(),
    )
    return train, val, test


def main():
    # Parse args.
    args = parse_args()
    device = args.device

    # Read config.
    with open(args.config, "rb") as handle:
        config = yaml.safe_load(handle)
    # Create loss config.
    reg_config = RigidTransformRegularisationLossConfig(
        **config["loss"].pop("regularisation")
    )
    loss_config = CompositeLossConfig(
        **config["loss"],
        transformation_regularisation_config=reg_config,
    )
    # Create training config.
    training_config = TrainingConfig(
        seed=config["seed"],
        loss_config=loss_config,
        checkpointer_root=FILE_DIR_PATH / "checkpoints",
        **config["training"],
    )
    # Create model config.
    model_config = RegistrationNetworkConfig(
        seed=config["seed"], **config["registration_network"], device=device
    )
    # Create dataloaders.
    segmentor_config = SegmentorConfig(device=device)
    preprocessor_config = PreprocessorConfig(**config["preprocessor"])
    aug = DataAugmentation()
    train, val, _ = split_dataset(
        seed=config["seed"],
        segmentor_config=segmentor_config,
        preprocessor_config=preprocessor_config,
        training_data_augmentation=aug,
        split=config["dataset_split"],
    )
    train_dl = DataLoader(
        train,
        visualisation=False,
        batch_size=config["batch_size"],
    )
    val_dl = DataLoader(
        val,
        visualisation=True,
        batch_size=len(val),
    )
    # Create logger.
    logger = NoOpLogger()
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
