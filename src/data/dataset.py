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
        age_skew: Optional[tuple] = None,  # (young_percentage, old_percentage)
        gender_skew: Optional[tuple] = None,  # (male_percentage, female_percentage)
        ttype_skew: Optional[tuple] = None  # (ttype1_percentage, ttype2_percentage)
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
        self.age_skew = age_skew
        self.gender_skew = gender_skew
        self.ttype_skew = ttype_skew
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root + "/metadata.csv")
        self.data = self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare the metadata for the dataset with skew based on age or gender if specified."""

        # Filter data based on the split, type, and pathology
        self.metadata = self.metadata.filter(pl.col("split") == self.split)
        self.metadata = self.metadata.filter(pl.col("type") == self.type)
        
        if self.pathology and len(self.pathology) > 0:  # Ensure pathology list is not empty
            pathology_filter = pl.col(self.pathology[0]) == True
            for path in self.pathology[1:]:
                pathology_filter |= pl.col(path) == True
            self.metadata = self.metadata.filter(pathology_filter)

        if self.lower_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") >= self.lower_slice)
        if self.upper_slice:
            self.metadata = self.metadata.filter(pl.col("slice_id") <= self.upper_slice)

        if self.age_skew:
            print(f"Sampling based on age skew: {self.age_skew}")
            # Separate data by age
            young = self.metadata.filter(pl.col("age_at_mri") <= 58).collect()
            old = self.metadata.filter(pl.col("age_at_mri") > 58).collect()

            # Sample based on specified age skew
            young_sample_size = int(self.age_skew[0] * len(young))
            old_sample_size = int(self.age_skew[1] * len(old))

            young_sampled = young.sample(n=young_sample_size, seed=self.seed)
            old_sampled = old.sample(n=old_sample_size, seed=self.seed)

            self.metadata = pl.concat([young_sampled, old_sampled])

        elif self.gender_skew:
            print(f"Sampling based on gender skew: {self.gender_skew}")
            # Separate data by gender
            male = self.metadata.filter(pl.col("sex") == "M").collect()
            female = self.metadata.filter(pl.col("sex") == "F").collect()

            # Sample based on specified gender skew
            male_sample_size = int(self.gender_skew[0] * len(male))
            female_sample_size = int(self.gender_skew[1] * len(female))

            male_sampled = male.sample(n=male_sample_size, seed=self.seed)
            female_sampled = female.sample(n=female_sample_size, seed=self.seed)

            self.metadata = pl.concat([male_sampled, female_sampled])
        elif self.ttype_skew:
            print(f"Sampling based on tumor type skew: {self.ttype_skew}")
            # Separate data by tumor type
            ttype1 = self.metadata.filter(pl.col("final_diagnosis") == "Glioblastoma, IDH-wildtype").collect()
            ttype2 = self.metadata.filter(pl.col("final_diagnosis") != "Glioblastoma, IDH-wildtype").collect()

            # Sample based on specified tumor type skew
            ttype1_sample_size = int(self.ttype_skew[0] * len(ttype1))
            ttype2_sample_size = int(self.ttype_skew[1] * len(ttype2))

            ttype1_sampled = ttype1.sample(n=ttype1_sample_size, seed=self.seed)
            ttype2_sampled = ttype2.sample(n=ttype2_sample_size, seed=self.seed)

            self.metadata = pl.concat([ttype1_sampled, ttype2_sampled])
        else:
            self.metadata = self.metadata.collect()

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
        replacement=True  # Sample with replacement to ensure balanced sampling
    )

    return sampler
