import pathlib
from typing import Callable, Optional

import fastmri
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import polars as pl
import torch
from fastmri import fft2c, ifft2c, tensor_to_complex_np
from fastmri.data.subsample import RandomMaskFunc
from torch.utils.data import Dataset

from src.data.dataset import BaseDataset


class ReconstructionDataset(BaseDataset):
    """Classification dataset to load MRI images"""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        number_of_samples: Optional[int] = 0,
        seed: Optional[int] = 31415,
        split: Optional[str] = "train",
        type: Optional[str] = "T2",
        pathology: Optional[list] = None,
        lower_slice=None,
        upper_slice=None,
        evaluation=False,
        age_bins=[0, 68, 100],
        sampling_mask: Optional[str] = "radial",
        age_skew: Optional[tuple] = None,  
        gender_skew: Optional[tuple] = None, 
        ttype_skew: Optional[tuple] = None
    ):
        """
        Initialize the MRIDataset.

        Args:
            data_root (pathlib.Path): The path to the data directory.
            transform (Optional[Callable]): The transform to apply to the data.
            number_of_samples (Optional[int]): The number of samples to use.
            seed (Optional[int]): The seed for reproducibility.
        """
        self.sampling_mask = sampling_mask
        super().__init__(
            data_root=data_root,
            transform=transform,
            number_of_samples=number_of_samples,
            seed=seed,
            split=split,
            type=type,
            pathology=pathology,
            lower_slice=lower_slice,
            upper_slice=upper_slice,
            evaluation=evaluation,
            age_bins=age_bins,
            age_skew=age_skew,
            gender_skew=gender_skew, 
            ttype_skew=ttype_skew
        )

    def convert_to_complex(self, image_slice):
        """
        Convert a real-valued 2D image slice to complex format.

        Parameters:
        - image_slice: 2D real-valued image tensor

        Returns:
        - complex_tensor: Complex-valued tensor with shape [H, W, 2] where the last dimension is (real, imaginary)
        """
        complex_tensor = torch.stack(
            (image_slice, torch.zeros_like(image_slice)), dim=-1
        )

        return complex_tensor

    def create_radial_mask(self, shape, num_rays=60):
        """
        Create a radial mask for undersampling k-space.

        Parameters:
        - shape: The shape of the mask (H, W)
        - num_rays: Number of radial rays in the mask

        Returns:
        - mask: A radial mask with values 0 (masked) and 1 (unmasked)
        """
        H, W = shape
        center = (H // 2, W // 2)
        Y, X = np.ogrid[:H, :W]
        mask = np.zeros((H, W), dtype=np.float32)

        # Define angles for rays
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        for angle in angles:
            line_x = np.cos(angle)
            line_y = np.sin(angle)
            for r in range(max(H, W) // 2):
                x = int(center[1] + r * line_x)
                y = int(center[0] + r * line_y)
                if 0 <= x < W and 0 <= y < H:
                    mask[y, x] = 1

        return mask

    def apply_radial_mask_to_kspace(self, kspace):
        """
        Apply a radial mask to the k-space data.

        Parameters:
        - kspace: The complex k-space data to mask

        Returns:
        - undersampled_kspace: The undersampled k-space data
        """
        H, W, _ = kspace.shape
        radial_mask = self.create_radial_mask((H, W))

        radial_mask = torch.from_numpy(radial_mask).to(kspace.device).unsqueeze(-1)

        undersampled_kspace = kspace * radial_mask

        return undersampled_kspace

    def apply_linear_mask_to_kspace(self, kspace):
        """
        Apply a linear mask to the k-space data.

        Parameters:
        - kspace: The complex k-space data to mask

        Returns:
        - undersampled_kspace: The undersampled k-space data
        """
        H, W, _ = kspace.shape
        mask_func = RandomMaskFunc(
            center_fractions=[0.1], accelerations=[6]
        )  # 4x undersampling
        mask = mask_func(kspace.shape, seed=None)[0]  # Extract the mask
        mask = mask.to(kspace.device).unsqueeze(
            -1
        )  # Ensure mask has an extra dimension for complex part

        # Apply mask to k-space (zero out k-space lines)
        undersampled_kspace = kspace * mask

        return undersampled_kspace

    def undersample_image_with_radial_mask(self, image_tensor):
        """
        Undersample an MRI image using a radial mask on the k-space.

        Parameters:
        - image_tensor: The 3D PyTorch tensor of image data
        - slice_index: The index of the slice to undersample

        Returns:
        - undersampled_image: The reconstructed image after undersampling
        """
        # Convert real slice to complex-valued tensor
        complex_slice = self.convert_to_complex(image_tensor)

        # Perform Fourier transform to go from image space to k-space
        kspace = fft2c(complex_slice)  # Apply center Fourier transform

        # Apply radial mask to k-space
        undersampled_kspace = self.apply_radial_mask_to_kspace(kspace)

        # Reconstruct the undersampled image by performing the inverse Fourier transform
        undersampled_image = ifft2c(undersampled_kspace)

        return undersampled_image

    def undersample_image_with_linear_mask(self, image_tensor):
        # Convert real slice to complex-valued tensor
        complex_slice = self.convert_to_complex(image_tensor)

        # Perform Fourier transform to go from image space to k-space
        kspace = fft2c(complex_slice)  # Apply center Fourier transform

        # Apply radial mask to k-space
        undersampled_kspace = self.apply_linear_mask_to_kspace(kspace)

        # Reconstruct the undersampled image by performing the inverse Fourier transform
        undersampled_image = ifft2c(undersampled_kspace)

        undersampled_image = undersampled_image.squeeze(0)

        return undersampled_image

    def undersample_slice(self, slice: torch.Tensor) -> torch.Tensor:
        if self.sampling_mask == "radial":
            undersampled_slice = self.undersample_image_with_radial_mask(slice)
        elif self.sampling_mask == "linear":
            undersampled_slice = self.undersample_image_with_linear_mask(slice)
        else:
            raise ValueError("Sampling mask not recognized")

        undersampled_slice = fastmri.complex_abs(undersampled_slice)

        return undersampled_slice

    def _get_item_from_row(self, row):
        # Load np array from file
        nifti_img = nib.load(self.data_root + "/" + row["file_path"])

        # Extract the image data as a numpy array
        scan = nifti_img.get_fdata()
        slice = scan[:, :, row["slice_id"]]
        slice_tensor = torch.from_numpy(slice).float()

        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        # undersample image
        undersampled_tensor = self.undersample_slice(slice_tensor)

        slice_tensor = slice_tensor.unsqueeze(0)
        undersampled_tensor = undersampled_tensor.unsqueeze(0)

        return undersampled_tensor, slice_tensor

    def get_random_sample(self):
        idx = np.random.randint(0, len(self.metadata))
        return self.__getitem__(idx)

    def get_patient_data(self, patient_id):
        patient_slices_metadata = self.metadata.filter(
            pl.col("patient_id") == patient_id
        )
        patient_slices_metadata = patient_slices_metadata.sort("slice_id")

        # If no slices found, raise an error or return empty
        if len(patient_slices_metadata) == 0:
            print(f"No slices found for patient_id={patient_id}")
            return []

        # Collect all slices for the patient
        slices = []
        for row_idx in range(len(patient_slices_metadata)):
            row = patient_slices_metadata.row(row_idx, named=True)

            # Load the slice for this row directly
            slice_tensor, labels = self._get_item_from_row(row)
            slices.append((slice_tensor, labels))

        return slices
