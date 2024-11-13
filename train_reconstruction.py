import argparse
import datetime
import os

import torch
import torchvision.transforms as transforms
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import create_balanced_sampler

# Import your dataset, models, and trainer
from src.data.reconstruction_dataset import ReconstructionDataset
from src.model.classification.classification_model import (
    AgeCEClassifier,
    GenderBCEClassifier,
)
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.reconstruction.style_gan import StyleGAN
from src.model.reconstruction.vgg import VGGReconstructionNetwork, get_configs
from src.trainer.trainer import Trainer
from src.utils.transformations import min_max_slice_normalization


def main():
    parser = argparse.ArgumentParser(description="Train a reconstruction model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameters from the configuration
    output_dir = config["output_dir"]
    output_name = config["output_name"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    num_train_samples = config.get("num_train_samples", None)
    num_val_samples = config.get("num_val_samples", None)
    network_type = config.get("network_type", "VGG")
    network_path = config.get("network_path", None)
    data_root = config["data_root"]
    seed = config.get("seed", 31415)
    save_interval = config.get("save_interval", 1)
    early_stopping_patience = config.get("early_stopping_patience", None)
    type = config.get("type", "T2")
    pathology = config.get("pathology", None)
    sampling_mask = config.get("sampling_mask", "radial")
    lower_slice = config.get("lower_slice", None)
    upper_slice = config.get("upper_slice", None)
    rebalancing = config.get("rebalancing", None)
    age_bins = config.get("age_bins", [0, 58, 100])

    # Append timestamp to output_name to make it unique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{output_name}_{timestamp}"
    output_dir = os.path.join(output_dir, output_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the configuration back into the output directory for tracking
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    transform = transforms.Compose(
        [
            min_max_slice_normalization,
        ]
    )

    # Datasets and DataLoaders
    train_dataset = ReconstructionDataset(
        data_root=data_root,
        transform=transform,
        split="train",
        number_of_samples=num_train_samples,
        seed=seed,
        type=type,
        pathology=pathology,
        sampling_mask=sampling_mask,
        lower_slice=lower_slice,
        upper_slice=upper_slice,
        age_bins=age_bins,
    )
    val_dataset = ReconstructionDataset(
        data_root=data_root,
        transform=transform,
        split="val",
        number_of_samples=num_val_samples,
        seed=seed,
        type=type,
        pathology=pathology,
        sampling_mask=sampling_mask,
        lower_slice=lower_slice,
        upper_slice=upper_slice,
        age_bins=age_bins,
    )

    val_sampler = None
    train_sampler = None
    shuffle = True

    if rebalancing is not None:
        shuffle = False
        if rebalancing == "Age":
            model = AgeCEClassifier(age_bins=age_bins)
            train_sampler = create_balanced_sampler(
                dataset=train_dataset, classifier=model
            )
            val_sampler = create_balanced_sampler(dataset=val_dataset, classifier=model)
        elif rebalancing == "Gender":
            model = GenderBCEClassifier()
            train_sampler = create_balanced_sampler(
                dataset=train_dataset, classifier=model
            )
            val_sampler = create_balanced_sampler(dataset=val_dataset, classifier=model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        sampler=train_sampler,
    )
    """val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        sampler=val_sampler,
    )"""

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classifier
    model = ReconstructionModel()
    model = model.to(device)

    # Model
    network = StyleGAN()

    network = network.to(device)

    # Add network to classifier
    model.set_network(network)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        samples=num_train_samples,
        train_loader=train_loader,
        train_dataset=train_dataset,
        val_loader=None,
        num_epochs=num_epochs,
        device=device,
        log_dir=os.path.join(output_dir, "logs"),
        output_dir=output_dir,
        output_name=output_name,
        save_interval=save_interval,
        early_stopping_patience=early_stopping_patience,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
