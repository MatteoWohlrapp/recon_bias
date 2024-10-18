import argparse
import datetime
import os
import random
import yaml

import torch
import torchvision.transforms as transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.data.classification_dataset import ClassificationDataset
from src.data.dataset import create_balanced_sampler
from src.model.classification.classification_model import (AgeCEClassifier,
                                                           GenderBCEClassifier,
                                                           NLLSurvClassifier,
                                                           TGradeBCEClassifier,
                                                           TTypeBCEClassifier, 
                                                           ClassifierModel)
from src.utils.transformations import min_max_slice_normalization
from src.model.classification.resnet_classification_network import \
    ResNetClassifierNetwork
from src.data.reconstruction_dataset import ReconstructionDataset
from src.evaluation.classifier_prediction import classifier_predictions
import pandas as pd
from src.model.reconstruction.reconstruction_model import ReconstructionModel
from src.model.reconstruction.unet import UNet
from src.model.reconstruction.vgg import VGGReconstructionNetwork, get_configs
from typing import List, Optional


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

def apply_gradcam(model, img_tensor, index):
    # Set the model to evaluation mode
    model.eval()

    # Access the internal ResNet model
    resnet_model = model.network.classifier

    # Initialize the GradCAM extractor, targeting the last convolutional block in ResNet18
    cam_extractor = GradCAM(resnet_model, target_layer='layer4')

    # Forward pass through the model
    output = model(img_tensor)

    cam_extractor(index, output, retain_graph=True)  # Ensure retain_graph is True to keep the computation graph

    # Generate the activation map
    activation_map = cam_extractor(index, output)[0].cpu().detach().numpy()

    min_val = activation_map.min()
    max_val = activation_map.max()

    # Avoid division by zero by checking if max and min are the same
    if max_val - min_val == 0:
        activation_map = np.full_like(activation_map, max_val, dtype=np.uint8)
    else:
        activation_map = (activation_map - min_val) / (max_val - min_val)
        activation_map = np.uint8(activation_map * 255)

    activation_map = activation_map.squeeze()

    # Convert the tensor image back to a PIL Image for visualization
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_tensor.squeeze().cpu())

    # Ensure the activation_map is 2D and of type uint8 before creating the PIL image
    if len(activation_map.shape) == 2:
        heatmap = Image.fromarray(activation_map)
    else:
        raise ValueError(f"Activation map is not 2D. Its shape is {activation_map.shape}")

    # Resize the heatmap to match the size of the original image
    heatmap = heatmap.resize(img_pil.size, resample=Image.BILINEAR)

    # Apply a colormap using Matplotlib
    colormap = plt.get_cmap("jet")
    heatmap_colored = colormap(np.array(heatmap) / 255.0)  # Normalize to [0, 1] for colormap

    # Convert to RGB and overlay on the original image
    heatmap_colored = Image.fromarray((heatmap_colored[:, :, :3] * 255).astype(np.uint8))  # Drop alpha channel and convert to RGB
    result = Image.blend(img_pil.convert("RGB"), heatmap_colored.convert("RGB"), alpha=0.5)  # Blend the two images

    return result

def save_combined_image(patient_id, sex, age, gt_img, classifier_img, recon_img, classfier_name, output_path, gt, classifier, recon):
    """Combines the ground truth, classifier Grad-CAM, and reconstruction Grad-CAM images side by side."""
    # Create a new figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Set the title including patient info
    fig.suptitle(f"Classifier: {classfier_name}, Patient ID: {patient_id}, Sex: {sex}, Age: {age}")

    # Display each image in the subplot
    axs[0].imshow(gt_img)
    axs[0].set_title(f"Ground Truth, Prediction {gt}")
    axs[0].axis('off')

    axs[1].imshow(classifier_img)
    axs[1].set_title(f"Classifier Grad-CAM, Prediction {classifier}")
    axs[1].axis('off')

    axs[2].imshow(recon_img)
    axs[2].set_title(f"Reconstruction Grad-CAM, Prediction {recon}")
    axs[2].axis('off')

    # Save the figure
    filename = f"{classifier_name}_patientID_{patient_id}_Sex_{sex}_Age_{age}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close(fig)
    print(f"Image saved: {os.path.join(output_path, filename)}")

def process_patient(
    patient_info,
    patient_classification_data,
    patient_reconstruction_data,
    classifier: dict,
    reconstruction_model: Optional[torch.nn.Module],
    output_path: str,
) -> dict:
    """Process each patient, returning a dictionary of results."""

    classifier_name = classifier["name"]
    classifier_model = classifier["model"]

    slice = len(patient_classification_data) // 2

    x, y = patient_classification_data[slice] 
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    y_transformed = classifier_model.target_transformation(y).item(),
    y_transformed = y_transformed[0]
    index = 0 if classifier_model.num_classes <= 2 else y_transformed
    y_class = classifier_model(x)
    x_recon, _ = patient_reconstruction_data[slice]
    x_recon = x_recon.unsqueeze(0)
    y_recon = classifier_model(x_recon)
    reconstructed_image = reconstruction_model(x_recon)

    gt_img = transforms.ToPILImage()(x.squeeze().cpu())
    result = apply_gradcam(classifier_model, x, index)
    result_recon = apply_gradcam(classifier_model, reconstructed_image, index)

    save_combined_image(
        patient_info['patient_id'],
        patient_info['sex'],
        patient_info['age'],
        gt_img,
        result,
        result_recon,
        classifier_name,
        output_path, 
        int(y_transformed),
        int(classifier_model.classification_criteria(y_class).item()),
        int(classifier_model.classification_criteria(y_recon).item())
    )


def apply_gradcam_to_models(
    data_root,
    classification_dataset,
    reconstruction_dataset,
    patients,
    classifier: dict,
    reconstruction_model: Optional[torch.nn.Module] = None,
) -> List[dict]:
    """Process all patients in the metadata file."""
    patient_predictions = []

    for _, row in patients.iterrows():
        patient_id  = row["ID"]
        age = row["Age at MRI"]
        sex = row["Sex"]
        print(f"Processing patient {patient_id}...")
        patient_info = {
            "patient_id": patient_id,
            "sex": sex,
            "age": age,
        }
        patient_classification_data = classification_dataset.get_patient_data(
            patient_id
        )
        patient_reconstruction_data = reconstruction_dataset.get_patient_data(
            patient_id
        )
        patient_prediction = process_patient(
            patient_info,
            patient_classification_data,
            patient_reconstruction_data,
            classifier,
            reconstruction_model,
            output_path=data_root
        )

    return patient_predictions


# Example usage inside your script (e.g., after defining and setting up your dataset and model)
if __name__ == "__main__":
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
    metadata = load_metadata(data_root + "/UCSF-PDGM-metadata.csv")

    patient_ids = config.get("patient_ids", [])
    if patient_ids:
        metadata = metadata[metadata["ID"].isin(patient_ids)]

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifiers_config = config["classifiers"]
    num_classifier_samples = classifiers_config.get("num_samples", None)
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
    classifier_dataset = ClassificationDataset(
        data_root=data_root,
        transform=transform,
        split="test",
        number_of_samples=num_classifier_samples,
        seed=seed,
        type=type_classifier,
        pathology=pathology_classifier,
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
        "reconstruction"
    )  # Get the reconstruction config, if it exists

    num_reconstruction_samples = None
    sampling_mask = None
    pathology_reconstruction = None
    type_reconstruction = None
    reconstruction_results_path = None
    reconstruction = None


    num_reconstruction_samples = reconstruction_config.get("num_samples", None)
    pathology_reconstruction = reconstruction_config.get("pathology", None)
    sampling_mask = reconstruction_config.get("sampling_mask", "radial")
    type_reconstruction = reconstruction_config.get("type", "T2")
    reconstruction_results_path = reconstruction_config.get("results_path", None)

    reconstruction_cfg = reconstruction_config["model"][0]

    network_type = reconstruction_cfg["network"]
    model_path = reconstruction_cfg["model_path"]

    reconstruction_model = load_reconstruction_model(
        network_type, model_path, device
    )
    reconstruction = {"model": reconstruction_model, "name": network_type}

    reconstruction_dataset = ReconstructionDataset(
        data_root=data_root,
        transform=transform,
        split="test",
        number_of_samples=num_reconstruction_samples,
        seed=seed,
        type=type_reconstruction,
        pathology=pathology_reconstruction,
        evaluation=True,
        sampling_mask=sampling_mask,
    )

    for classifier in classifiers:

        # Create directory for classifier
        classifier_name = classifier["name"]
        classifier_output_path = os.path.join(output_path, classifier_name)
        os.makedirs(classifier_output_path, exist_ok=True)

        # Apply Grad-CAM to the models
        apply_gradcam_to_models(data_root=classifier_output_path,
                                classification_dataset=classifier_dataset,
                                reconstruction_dataset=reconstruction_dataset,
                                patients=metadata,
                                classifier=classifier,
                                reconstruction_model=reconstruction_model)
