import pathlib
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.utils.labels import diagnosis_map, extract_labels_from_row, sex_map


class BaseDataset(Dataset, ABC):
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
    ):
        self.data_root = data_root
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.split = split
        self.type = type
        self.pathology = pathology
        self.lower_slice = lower_slice
        self.upper_slice = upper_slice
        self.evaluation = evaluation
        self.age_bins = age_bins
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root + "/metadata.csv")
        self.data = self._prepare_metadata()
    """
    # Only smaller four
    def _prepare_metadata(self):
        # Filter metadata by split and type
        self.metadata = self.metadata.filter(pl.col("split") == self.split)
        self.metadata = self.metadata.filter(pl.col("type") == self.type)
        
        # Collect counts for tgrade four and smaller four
        four_count = self.metadata.filter(pl.col("who_cns_grade") == 4).collect().height
        sfour_count = self.metadata.filter(pl.col("who_cns_grade") < 4).collect().height

        # Determine the smallest class size
        smallest_count = min(four_count, sfour_count)

        # Sample only males with the size of the smallest class
        self.metadata = (
            self.metadata.filter(pl.col("who_cns_grade") < 4)
            .collect()
            .sample(n=smallest_count, seed=self.seed)
        )

        # Apply pathology filtering
        if self.pathology and len(self.pathology) > 0:
            pathology_filter = pl.col(self.pathology[0]) == True
            for path in self.pathology[1:]:
                pathology_filter |= pl.col(path) == True
            self.metadata = self.metadata.filter(pathology_filter)

        # Filter by slice range
        if self.lower_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") >= self.lower_slice)
        if self.upper_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") <= self.upper_slice)

        # If a specific number of samples is requested
        if self.number_of_samples and not self.evaluation:
            self.metadata = self.metadata.collect().sample(n=self.number_of_samples, seed=self.seed)

    """
    # Equal male and female
    def _prepare_metadata(self):
        # Filter metadata by split and type
        self.metadata = self.metadata.filter(pl.col("split") == self.split)
        self.metadata = self.metadata.filter(pl.col("type") == self.type)

        # Collect counts for tgrade four and smaller four
        four_count = self.metadata.filter(pl.col("who_cns_grade") == 4).collect().height
        sfour_count = self.metadata.filter(pl.col("who_cns_grade") < 4).collect().height

        # Determine the smallest class size
        smallest_count = min(four_count, sfour_count)

        # Sample equal male and female
        four_metadata = (
            self.metadata.filter(pl.col("who_cns_grade") == 4)
            .collect()
            .sample(n=smallest_count/2, seed=self.seed)
        )
        sfour_metadata = (
            self.metadata.filter(pl.col("who_cns_grade") < 4)
            .collect()
            .sample(n=smallest_count/2, seed=self.seed)
        )

        # Combine male and female metadata
        self.metadata = pl.concat([four_metadata, sfour_metadata])

        # Apply pathology filtering
        if self.pathology and len(self.pathology) > 0:
            pathology_filter = pl.col(self.pathology[0]) == True
            for path in self.pathology[1:]:
                pathology_filter |= pl.col(path) == True
            self.metadata = self.metadata.filter(pathology_filter)

        # Filter by slice range
        if self.lower_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") >= self.lower_slice)
        if self.upper_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") <= self.upper_slice)

        # If a specific number of samples is requested
        if self.number_of_samples and not self.evaluation:
            self.metadata = self.metadata.collect().sample(n=self.number_of_samples, seed=self.seed)

    """
    # Only four
    def _prepare_metadata(self):
        # Filter metadata by split and type
        self.metadata = self.metadata.filter(pl.col("split") == self.split)
        self.metadata = self.metadata.filter(pl.col("type") == self.type)
        
        # Collect counts for tgrade four and smaller four
        four_count = self.metadata.filter(pl.col("who_cns_grade") == 4).collect().height
        sfour_count = self.metadata.filter(pl.col("who_cns_grade") < 4).collect().height

        # Determine the smallest class size
        smallest_count = min(four_count, sfour_count)

        # Sample only males with the size of the smallest class
        self.metadata = (
            self.metadata.filter(pl.col("who_cns_grade") == 4)
            .collect()
            .sample(n=smallest_count, seed=self.seed)
        )

        # Apply pathology filtering
        if self.pathology and len(self.pathology) > 0:
            pathology_filter = pl.col(self.pathology[0]) == True
            for path in self.pathology[1:]:
                pathology_filter |= pl.col(path) == True
            self.metadata = self.metadata.filter(pathology_filter)

        # Filter by slice range
        if self.lower_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") >= self.lower_slice)
        if self.upper_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") <= self.upper_slice)

        # If a specific number of samples is requested
        if self.number_of_samples and not self.evaluation:
            self.metadata = self.metadata.collect().sample(n=self.number_of_samples, seed=self.seed)
    """
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.row(idx, named=True)

        return self._get_item_from_row(row)

    @abstractmethod
    def _get_item_from_row(self, row):
        pass

    @abstractmethod
    def get_random_sample(self):
        pass

    @abstractmethod
    def get_patient_data(self, patient_id):
        pass

    def get_class_labels(self):
        """Returns the class labels for each sample in the dataset."""
        class_labels = [
            extract_labels_from_row(row, self.age_bins)
            for row in self.metadata.iter_rows(named=True)
        ]
        class_labels_tensor = torch.stack(class_labels)
        return class_labels_tensor


def create_balanced_sampler(dataset, classifier):
    """Creates a WeightedRandomSampler for balanced class sampling."""
    dataset_class_labels = dataset.get_class_labels()

    class_labels = classifier.target_transformation(dataset_class_labels)

    # Count occurrences of each class (assuming binary or multi-class)
    class_counts = np.bincount(class_labels)
    class_weights = 1.0 / class_counts

    # Create weights for each sample based on its class
    sample_weights = np.array([class_weights[int(label)] for label in class_labels])

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,  # Sample with replacement to ensure balanced sampling
    )

    return sampler
