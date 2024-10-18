from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils.labels import extract_labels_from_row


def majority_voting(predictions: List[int]) -> int:
    """Aggregate predictions via majority voting."""
    return max(set(predictions), key=predictions.count)


def process_patient_data(
    patient_info,
    patient_classification_data,
    patient_reconstruction_data,
    classifiers: List[dict],
    reconstruction_model: Optional[torch.nn.Module] = None,
    classification_dataset=None,
) -> dict:
    """Process each patient, returning a dictionary of results."""

    for classifier_info in classifiers:
        classifier_name = classifier_info["name"]
        classifier = classifier_info["model"]

        patient_class_scores = []  # raw scores of the logits
        patient_class_predictions = []  # predictions based on the raw scores
        patient_recon_scores = []  # predictions based on the reconstruction
        patient_recon_predictions = []  # predictions based on the reconstruction
        patient_gt = None

        for i, (x, y) in enumerate(patient_classification_data):
            with torch.no_grad():
                # Prediction on original image
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                class_output = classifier(x)
                class_score = classifier.final_activation(class_output)
                patient_class_scores.append(class_score.item())
                class_pred = classifier.classification_criteria(class_output)
                patient_class_predictions.append(class_pred.item())

                # Ground truth
                if patient_gt is None:  # We only need to calculate GT once
                    patient_gt = classifier.target_transformation(y).item()

                # Prediction on reconstruction if reconstruction model exists
                x_recon, _ = patient_reconstruction_data[i]
                x_recon = x_recon.unsqueeze(0)
                recon_image = reconstruction_model(x_recon)
                recon_image = recon_image.squeeze()
                recon_image = classification_dataset.get_kspace(recon_image)
                recon_image = recon_image.unsqueeze(0)
                recon_output = classifier(recon_image)
                recon_score = classifier.final_activation(recon_output)
                patient_recon_scores.append(recon_score.item())
                recon_pred = classifier.classification_criteria(recon_output)
                patient_recon_predictions.append(recon_pred.item())

        # Aggregate predictions using majority voting
        majority_class_pred = majority_voting(patient_class_predictions)
        patient_info[f"{classifier_name}_gt"] = patient_gt
        patient_info[f"{classifier_name}_pred"] = majority_class_pred

        average_class_score = np.median(patient_class_scores)
        patient_gt_score = float(patient_gt)
        patient_info[f"{classifier_name}_gt_score"] = patient_gt_score
        patient_info[f"{classifier_name}_pred_score"] = average_class_score

        majority_recon_pred = majority_voting(patient_recon_predictions)
        patient_info[f"{classifier_name}_recon"] = majority_recon_pred

        average_recon_score = np.median(patient_recon_scores)
        patient_info[f"{classifier_name}_recon_score"] = average_recon_score

    return patient_info


def classifier_predictions(
    data_root,
    classification_dataset,
    reconstruction_dataset,
    metadata: pd.DataFrame,
    classifiers: List[dict],
    reconstruction_model: Optional[torch.nn.Module] = None,
    num_samples=None,
) -> List[dict]:
    """Process all patients in the metadata file."""
    patient_predictions = []

    index = 0
    for patient_id, patient_df in metadata.groupby("patient_id"):
        if num_samples is not None:
            if index >= num_samples:
                break
        index += 1
        print(f"Processing patient {patient_id}...")
        patient_info = {
            "patient_id": patient_id,
            "sex": patient_df["sex"].iloc[0],
            "age": patient_df["age_at_mri"].iloc[0],
        }
        patient_classification_data = classification_dataset.get_patient_data(
            patient_id
        )
        patient_reconstruction_data = reconstruction_dataset.get_patient_data(
            patient_id
        )
        patient_prediction = process_patient_data(
            patient_info,
            patient_classification_data,
            patient_reconstruction_data,
            classifiers,
            reconstruction_model,
            classification_dataset,
        )
        patient_predictions.append(patient_prediction)

    return patient_predictions
