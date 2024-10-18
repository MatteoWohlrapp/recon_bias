import os
import pathlib
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import polars as pl
import torch
from fastmri import fft2c, ifft2c, tensor_to_complex_np
import fastmri
from src.utils.labels import diagnosis_map, extract_labels_from_row, sex_map

from src.data.dataset import BaseDataset


class ClassificationDataset(BaseDataset):
    """Classification dataset to load MRI images"""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        number_of_samples: Optional[int] = 0,
        seed: Optional[int] = 31415,
        split: Optional[str] = "train",
        type: Optional[str] = "T2",
        pathology: Optional[list] = ["edema", "non_enhancing", "enhancing"],
        lower_slice=None,
        upper_slice=None,
        evaluation=False,
        age_bins=[0, 68, 100],
        os_bins=4,
    ):
        """
        Initialize the MRIDataset.

        Args:
            data_root (pathlib.Path): The path to the data directory.
            transform (Optional[Callable]): The transform to apply to the data.
            number_of_samples (Optional[int]): The number of samples to use.
            seed (Optional[int]): The seed for reproducibility.
        """
        self.os_bins = os_bins
        super().__init__(data_root=data_root, transform=transform, number_of_samples=number_of_samples, seed=seed, split=split, type=type, pathology=pathology, lower_slice=lower_slice, upper_slice=upper_slice, evaluation=evaluation, age_bins=age_bins)
        self.os_bin_size = self._get_highest_dead_os() // self.os_bins

    def _get_item_from_row(self, row):
        nifti_img = nib.load(self.data_root + "/" + row["file_path"])

        # Extract the image data as a numpy array
        scan = nifti_img.get_fdata()
        slice = scan[:, :, row["slice_id"]]
        slice_tensor = torch.from_numpy(slice).float()
        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        labels = extract_labels_from_row(row, self.age_bins)

        slice_tensor = slice_tensor.unsqueeze(0)

        return slice_tensor, labels

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

    def _get_highest_dead_os(self):
        # Filter by split
        metadata = pl.scan_csv(self.data_root + "/metadata.csv")

        # Filter by pathology OR
        if (
            self.pathology and len(self.pathology) > 0
        ):  # Ensure pathology list is not empty
            pathology_filter = pl.col(self.pathology[0]) == True
            for path in self.pathology[1:]:
                pathology_filter |= pl.col(path) == True

            metadata = metadata.filter(pathology_filter)

        # Filter by diagnosis
        metadata = metadata.filter(pl.col("alive") == 1)

        metadata = metadata.collect()

        # Sort by age
        metadata = metadata.sort("os")

        # Get the first row
        row = metadata.row(len(metadata) - 1, named=True)

        os = row["os"]

        return os
    

class KSpaceClassificationDataset(BaseDataset):
    """Classification dataset to load MRI images"""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        number_of_samples: Optional[int] = 0,
        seed: Optional[int] = 31415,
        split: Optional[str] = "train",
        type: Optional[str] = "T2",
        pathology: Optional[list] = ["edema", "non_enhancing", "enhancing"],
        lower_slice=None,
        upper_slice=None,
        evaluation=False,
        age_bins=[0, 68, 100],
        os_bins=4,
    ):
        """
        Initialize the MRIDataset.

        Args:
            data_root (pathlib.Path): The path to the data directory.
            transform (Optional[Callable]): The transform to apply to the data.
            number_of_samples (Optional[int]): The number of samples to use.
            seed (Optional[int]): The seed for reproducibility.
        """
        self.os_bins = os_bins
        super().__init__(data_root=data_root, transform=transform, number_of_samples=number_of_samples, seed=seed, split=split, type=type, pathology=pathology, lower_slice=lower_slice, upper_slice=upper_slice, evaluation=evaluation, age_bins=age_bins)
        self.os_bin_size = self._get_highest_dead_os() // self.os_bins

    def _get_item_from_row(self, row):
        nifti_img = nib.load(self.data_root + "/" + row["file_path"])

        # Extract the image data as a numpy array
        scan = nifti_img.get_fdata()
        slice = scan[:, :, row["slice_id"]]
        slice_tensor = torch.from_numpy(slice).float()
        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        labels = extract_labels_from_row(row, self.age_bins)

        kspace = self.get_kspace(slice_tensor)

        kspace = kspace.permute(2, 0, 1)

        return kspace, labels

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

    def _get_highest_dead_os(self):
        # Filter by split
        metadata = pl.scan_csv(self.data_root + "/metadata.csv")

        # Filter by pathology OR
        if (
            self.pathology and len(self.pathology) > 0
        ):  # Ensure pathology list is not empty
            pathology_filter = pl.col(self.pathology[0]) == True
            for path in self.pathology[1:]:
                pathology_filter |= pl.col(path) == True

            metadata = metadata.filter(pathology_filter)

        # Filter by diagnosis
        metadata = metadata.filter(pl.col("alive") == 1)

        metadata = metadata.collect()

        # Sort by age
        metadata = metadata.sort("os")

        # Get the first row
        row = metadata.row(len(metadata) - 1, named=True)

        os = row["os"]

        return os
    
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

    def get_kspace(self, slice: torch.Tensor) -> torch.Tensor:
                
        complex_slice = self.convert_to_complex(slice)

        # Perform Fourier transform to go from image space to k-space
        kspace = fft2c(complex_slice)

        return kspace