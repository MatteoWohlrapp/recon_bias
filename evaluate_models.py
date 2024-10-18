import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml
from torch import nn

from src.data.classification_dataset import ClassificationDataset, KSpaceClassificationDataset
from src.data.reconstruction_dataset import ReconstructionDataset
from src.evaluation.classifier_prediction import classifier_predictions
from src.evaluation.evaluation import (classifier_evaluation,
                                       reconstruction_evaluation)
from src.evaluation.reconstruction_prediction import reconstruction_predictions
from src.model.classification.classification_model import (AgeCEClassifier,
                                                           ClassifierModel,
                                                           GenderBCEClassifier,
                                                           NLLSurvClassifier,
                                                           TGradeBCEClassifier,
                                                           TTypeBCEClassifier)
from src.model.classification.resnet_classification_network import \
    ResNetClassifierNetwork
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.reconstruction.vgg import VGGReconstructionNetwork, get_configs
from src.utils.mock_data import get_mock_data
from src.utils.transformations import min_max_slice_normalization


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # For reproducibility
    torch.backends.cudnn.benchmark = False


def load_metadata(metadata_path: str) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def load_classifier(
    classifier_type: str, network_type: str, model_path: str, device, config, dataset
) -> ClassifierModel:
    """Loads the appropriate classifier based on type and network."""
    # Classifier
    if classifier_type == "TTypeBCEClassifier":
        model = TTypeBCEClassifier()
    elif classifier_type == "TGradeBCEClassifier":
        model = TGradeBCEClassifier()
    elif classifier_type == "NLLSurvClassifier":
        eps = config.get("eps", 1e-8)
        bin_size = dataset.os_bin_size
        os_bins = dataset.os_bins
        model = NLLSurvClassifier(bins=os_bins, bin_size=bin_size, eps=eps)
    elif classifier_type == "AgeCEClassifier":
        age_bins = dataset.age_bins
        model = AgeCEClassifier(age_bins=age_bins)
    elif classifier_type == "GenderBCEClassifier":
        model = GenderBCEClassifier()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    model = model.to(device)

    # Model
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

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.network.eval()

    return model


def load_reconstruction_model(network_type, model_path, device) -> torch.nn.Module:
    model = ReconstructionModel()
    model = model.to(device)

    if network_type == "VGG":
        network = VGGReconstructionNetwork(get_configs("vgg16"))
    elif network_type == "UNet":
        network = UNet()
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    network = network.to(device)

    model.set_network(network)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.network.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification models.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract paths and classifier information from the config
    data_root = config["data_root"]
    output_dir = config["output_dir"]
    output_name = config["output_name"]

    # Create output directory for evaluation
    output_name = f"{output_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, output_name)
    os.makedirs(output_path, exist_ok=True)

    # Save config to output directory for reproducibility
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Load metadata
    metadata = load_metadata(data_root + "/metadata.csv")
    metadata = metadata[metadata["split"] == "test"]

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifiers_config = config["classifiers"]
    num_classifier_samples = classifiers_config.get("num_samples", None)
    lower_slice_classifier = classifiers_config.get("lower_slice", None)
    upper_slice_classifier = classifiers_config.get("upper_slice", None)
    pathology_classifier = classifiers_config.get("pathology", None)
    type_classifier = classifiers_config.get("type", "T2")
    classifier_results_path = classifiers_config.get("results_path", None)
    os_bins = config.get("os_bins", 4)
    age_bins = config.get("age_bins", [0, 3, 18, 42, 67, 96])

    transform = transforms.Compose(
        [
            min_max_slice_normalization,
        ]
    )

    seed = config.get("seed", 42)

    # Process classifiers
    classifier_dataset = KSpaceClassificationDataset(
        data_root=data_root,
        transform=transform,
        split="test",
        number_of_samples=num_classifier_samples,
        seed=seed,
        type=type_classifier,
        pathology=pathology_classifier,
        lower_slice=lower_slice_classifier,
        upper_slice=upper_slice_classifier,
        age_bins=age_bins,
        os_bins=os_bins,
        evaluation=True,
    )

    # Initialize classifiers
    classifiers = []
    for classifier_cfg in classifiers_config["models"]:
        classifier_type = classifier_cfg["type"]
        network_type = classifier_cfg["network"]
        model_path = classifier_cfg["model_path"]
        classifier = load_classifier(
            classifier_type,
            network_type,
            model_path,
            device,
            classifier_cfg,
            classifier_dataset,
        )
        classifiers.append({"model": classifier, "name": classifier_type})

        # Accessing reconstruction config
    reconstruction_config = config.get(
        "reconstruction", None
    )  # Get the reconstruction config, if it exists

    num_reconstruction_samples = None
    lower_slice_reconstruction = None
    upper_slice_reconstruction = None
    sampling_mask = None
    pathology_reconstruction = None
    type_reconstruction = None
    reconstruction_results_path = None
    reconstruction = None

    if (
        reconstruction_config is not None
        and "model" in reconstruction_config
        and reconstruction_config["model"] is not None
        and len(reconstruction_config["model"]) > 0
    ):
        num_reconstruction_samples = reconstruction_config.get("num_samples", None)
        lower_slice_reconstruction = reconstruction_config.get("lower_slice", None)
        upper_slice_reconstruction = reconstruction_config.get("upper_slice", None)
        pathology_reconstruction = reconstruction_config.get("pathology", None)
        sampling_mask = reconstruction_config.get("sampling_mask", "radial")
        type_reconstruction = reconstruction_config.get("type", "T2")
        reconstruction_results_path = reconstruction_config.get("results_path", None)

        reconstruction_cfg = reconstruction_config["model"][0]

        if "network" in reconstruction_cfg and "model_path" in reconstruction_cfg:
            network_type = reconstruction_cfg["network"]
            model_path = reconstruction_cfg["model_path"]

            reconstruction_model = load_reconstruction_model(
                network_type, model_path, device
            )
            reconstruction = {"model": reconstruction_model, "name": network_type}
        else:
            print(
                "Reconstruction model configuration is incomplete or missing required fields."
            )

    else:
        print("No reconstruction model specified.")

    reconstruction_dataset = ReconstructionDataset(
        data_root=data_root,
        transform=transform,
        split="test",
        number_of_samples=num_reconstruction_samples,
        seed=seed,
        type=type_reconstruction,
        pathology=pathology_reconstruction,
        lower_slice=lower_slice_reconstruction,
        upper_slice=upper_slice_reconstruction,
        evaluation=True,
        sampling_mask=sampling_mask,
    )

    if classifier_results_path is None:
        # Process and evaluate classification
        classifier_results = classifier_predictions(
            data_root,
            classifier_dataset,
            reconstruction_dataset,
            metadata,
            classifiers,
            reconstruction["model"],
            num_classifier_samples,
        )

        # Create DataFrame for results
        classifier_results_df = pd.DataFrame(classifier_results)

        # Save results to output directory
        classifier_results_df.to_csv(
            os.path.join(output_path, f"{output_name}_classifier_results.csv"),
            index=False,
        )
    else:
        classifier_results_df = pd.read_csv(classifier_results_path)

    # Evaluate predictions
    classifier_evaluation(classifier_results_df, classifiers, age_bins, output_path)

    if reconstruction is None:
        print("No reconstruction model specified. Skipping reconstruction evaluation.")
        return
    # Process reconstruction

    if reconstruction_results_path is None:
        # Process and evaluate reconstruction
        reconstruction_results = reconstruction_predictions(
            data_root,
            reconstruction_dataset,
            metadata,
            reconstruction,
            num_reconstruction_samples,
        )

        # Create DataFrame for results
        reconstruction_results_df = pd.DataFrame(reconstruction_results)

        # Save results to output directory
        reconstruction_results_df.to_csv(
            os.path.join(output_path, f"{output_name}_reconstruction_results.csv"),
            index=False,
        )
    else:
        reconstruction_results_df = pd.read_csv(reconstruction_results_path)

    # Evaluate predictions
    reconstruction_evaluation(
        reconstruction_results_df, reconstruction, age_bins, output_path
    )


if __name__ == "__main__":
    main()
