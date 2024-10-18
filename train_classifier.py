import argparse
import datetime
import os

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

# Import your dataset, models, and trainer
from src.data.classification_dataset import ClassificationDataset, KSpaceClassificationDataset
from src.data.dataset import create_balanced_sampler
from src.model.classification.classification_model import (AgeCEClassifier,
                                                           GenderBCEClassifier,
                                                           NLLSurvClassifier,
                                                           TGradeBCEClassifier,
                                                           TTypeBCEClassifier)
from src.model.classification.resnet_classification_network import \
    ResNetClassifierNetwork
from src.trainer.trainer import Trainer
from src.utils.transformations import min_max_slice_normalization


def main():
    parser = argparse.ArgumentParser(description="Train a classification model.")
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
    classifier_type = config["classifier_type"]
    network_type = config.get("network_type", "ResNet18")
    data_root = config["data_root"]
    seed = config.get("seed", 31415)
    save_interval = config.get("save_interval", 1)
    early_stopping_patience = config.get("early_stopping_patience", None)
    type = config.get("type", "T2")
    pathology = config.get("pathology", ["edema", "non_enhancing", "enhancing"])
    lower_slice = config.get("lower_slice", None)
    upper_slice = config.get("upper_slice", None)
    os_bins = config.get("os_bins", 4)
    age_bins = config.get("age_bins", [0, 3, 18, 42, 67, 96])
    balancing = config.get("balancing", False)

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
    train_dataset = KSpaceClassificationDataset(
        data_root=data_root,
        transform=transform,
        split="train",
        number_of_samples=num_train_samples,
        seed=seed,
        type=type,
        pathology=pathology,
        lower_slice=lower_slice,
        upper_slice=upper_slice,
        age_bins=age_bins,
        os_bins=os_bins,
    )
    val_dataset = KSpaceClassificationDataset(
        data_root=data_root,
        transform=transform,
        split="val",
        number_of_samples=num_val_samples,
        seed=seed,
        type=type,
        pathology=pathology,
        lower_slice=lower_slice,
        upper_slice=upper_slice,
        age_bins=age_bins,
        os_bins=os_bins,
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classifier
    if classifier_type == "TTypeBCEClassifier":
        model = TTypeBCEClassifier()
    elif classifier_type == "TGradeBCEClassifier":
        model = TGradeBCEClassifier()
    elif classifier_type == "NLLSurvClassifier":
        eps = config.get("eps", 1e-8)
        bin_size = train_dataset.os_bin_size
        model = NLLSurvClassifier(bins=os_bins, bin_size=bin_size, eps=eps)
    elif classifier_type == "AgeCEClassifier":
        model = AgeCEClassifier(age_bins=age_bins)
    elif classifier_type == "GenderBCEClassifier":
        model = GenderBCEClassifier()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    model = model.to(device)

    train_sampler = None
    val_sampler = None
    shuffle = True

    if balancing:
        print("Balancing the dataset.")
        train_sampler = create_balanced_sampler(train_dataset, classifier=model)
        val_sampler = create_balanced_sampler(val_dataset, classifier=model)
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=val_sampler
    )

    # Network
    if network_type == "ResNet18":
        network = ResNetClassifierNetwork(num_classes=model.num_classes)
    elif network_type == "ResNet50":
        network = ResNetClassifierNetwork(
            num_classes=model.num_classes, resnet_version="resnet50"
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    network = network.to(device)

    # Add network to classifier
    model.set_network(network)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
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
