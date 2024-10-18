from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_metric


def calculate_psnr(x: np.ndarray, y: np.ndarray, max_value: float = 1.0) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_value / np.sqrt(mse))


def calculate_nrmse(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Normalized RMSE between two images."""
    rmse = np.sqrt(np.mean((x - y) ** 2))
    norm_factor = np.max(y) - np.min(y)
    return rmse / norm_factor


def process_patient_data(
    patient_info,
    patient_reconstruction_data,
    reconstruction_model: Optional[torch.nn.Module] = None,
) -> dict:
    """Process each patient, returning a dictionary of results."""

    psnr_values = []
    ssim_values = []
    nrmse_values = []

    for i, (x, y) in enumerate(patient_reconstruction_data):
        with torch.no_grad():
            # Prediction on original image
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            # Run the reconstruction model to get the prediction
            prediction = reconstruction_model["model"](x)

            # Convert tensors to numpy arrays for metric calculations
            y_np = y.squeeze().squeeze().cpu().numpy()
            pred_np = prediction.squeeze().squeeze().cpu().numpy()

            # PSNR Calculation
            psnr_value = calculate_psnr(pred_np, y_np)
            psnr_values.append(psnr_value)

            # SSIM Calculation
            ssim_value = ssim_metric(
                y_np, pred_np, data_range=pred_np.max() - pred_np.min()
            )
            ssim_values.append(ssim_value)

            # NRMSE Calculation
            nrmse_value = calculate_nrmse(pred_np, y_np)
            nrmse_values.append(nrmse_value)

    # Aggregate predictions using majority voting
    aggregated_psnr = np.mean(psnr_values)
    aggregated_ssim = np.mean(ssim_values)
    aggregated_nrmse = np.mean(nrmse_values)

    # Store the aggregated results in the patient info
    patient_info[f"{reconstruction_model['model'].name}_psnr"] = aggregated_psnr
    patient_info[f"{reconstruction_model['model'].name}_ssim"] = aggregated_ssim
    patient_info[f"{reconstruction_model['model'].name}_nrmse"] = aggregated_nrmse

    return patient_info


def reconstruction_predictions(
    data_root,
    reconstruction_dataset,
    metadata: pd.DataFrame,
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
        patient_reconstruction_data = reconstruction_dataset.get_patient_data(
            patient_id
        )
        patient_prediction = process_patient_data(
            patient_info,
            patient_reconstruction_data,
            reconstruction_model,
        )
        patient_predictions.append(patient_prediction)

    return patient_predictions
