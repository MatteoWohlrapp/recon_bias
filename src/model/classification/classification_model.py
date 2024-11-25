"""
Classifier wrappers for both training and evaluation.
"""

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from src.utils.bootstrap import bootstrap
from src.utils.delong import delong_roc_test
from src.utils.hypothesis_test import hypothesis_test

from ..model_wrapper import ModelWrapper


class ClassifierModel(ModelWrapper):
    """
    Classifier base class.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.network(x)

    @property
    @abstractmethod
    def num_classes(self):
        """
        Number of classes for the classification.
        """
        pass

    @abstractmethod
    def classification_criteria(self, logits):
        """
        Specifies how to convert logits to predictions.
        """
        pass

    @abstractmethod
    def final_activation(self, logits):
        """
        Specifies the final activation function for the model.
        """
        pass

    @abstractmethod
    def significance(self, gt, pred, recon):
        """
        Calculates the p value for the performance metric.

        Args:
            gt: ground truth
            pred: prediction
            recon: reconstruction
        Returns:
            p value
        """
        pass

    def save_snapshot(self, x, y, y_pred, path, device, epoch):
        y_transformed = self.target_transformation(y)
        y_transformed = y_transformed.squeeze()

        y_pred = self.classification_criteria(y_pred)
        y_pred = y_pred.squeeze()

        # save labels to text file
        with open(path + ".txt", "w") as f:
            f.write("Ground truth: " + str(y_transformed.cpu().numpy()) + "\n")
            f.write("Predictions: " + str(y_pred.cpu().numpy()) + "\n")


class TTypeBCEClassifier(ClassifierModel):
    """
    Classifier for tumor type prediction using binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def name(self):
        return "TType"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        logits = logits.squeeze(1)
        loss = self.bce_loss(logits, transformed_labels)
        return loss

    def target_transformation(self, y):
        target_labels = y[:, 3].clone()
        target_labels[target_labels < 3] = 0
        target_labels[target_labels == 3] = 1
        return target_labels

    def evaluation_performance_metric(self, x, y):
        # Check if both classes (0 and 1) are present
        if len(np.unique(y)) == 1:
            print(
                "Warning: Only one class present in y_true. ROC AUC score is not defined."
            )
            return 0  # or return a default value like 0.5 if preferred
        return roc_auc_score(y, x)

    def epoch_performance_metric(self, x, y):
        x = x.cpu().detach().numpy()
        target_transform = self.target_transformation(y)
        return self.evaluation_performance_metric(x, target_transform), 1

    @property
    def performance_metric_name(self):
        return "AUROC"

    @property
    def performance_metric_input_value(self):
        return "score"

    @property
    def evaluation_groups(self):
        return [
            (
                ["sex"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "sex",
            ),
            (
                ["age_bin"],
                {
                    "x": "age_bin",
                    "x_label": "Age Group",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "age",
            ),
            (
                ["sex", "age_bin"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": "age_bin",
                    "facet_col_label": "Age Group",
                },
                "sex_age",
            ),
        ]

    @property
    def num_classes(self):
        return 1

    def classification_criteria(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits) > 0.5

    def final_activation(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits)

    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value


class TGradeBCEClassifier(ClassifierModel):
    """
    Classifier for tumor grade prediction using binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def name(self):
        return "TGrade"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        logits = logits.squeeze(1)
        loss = self.bce_loss(logits, transformed_labels)
        return loss

    def target_transformation(self, y):
        target_labels = y[:, 2].clone()
        target_labels[target_labels < 4] = 0
        target_labels[target_labels == 4] = 1
        return target_labels

    def evaluation_performance_metric(self, x, y):
        # Check if both classes (0 and 1) are present
        if len(np.unique(y)) == 1:
            print(
                "Warning: Only one class present in y_true. ROC AUC score is not defined."
            )
            return 0  # or return a default value like 0.5 if preferred
        return roc_auc_score(y, x)

    def epoch_performance_metric(self, x, y):
        x = x.detach().numpy()
        target_transform = self.target_transformation(y)
        return self.evaluation_performance_metric(x, target_transform), 1

    @property
    def performance_metric_name(self):
        return "AUROC"

    @property
    def performance_metric_input_value(self):
        return "score"

    @property
    def evaluation_groups(self):
        return [
            (
                ["sex"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "sex",
            ),
            (
                ["age_bin"],
                {
                    "x": "age_bin",
                    "x_label": "Age Group",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "age",
            ),
            (
                ["sex", "age_bin"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": "age_bin",
                    "facet_col_label": "Age Group",
                },
                "sex_age",
            ),
        ]

    @property
    def num_classes(self):
        return 1

    def classification_criteria(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits) > 0.5

    def final_activation(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits)

    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value


class NLLSurvClassifier(ClassifierModel):
    """
    Classifier for survival prediction using negative log likelihood loss.
    """

    def __init__(self, bins, bin_size, eps=1e-8):
        super().__init__()
        self.bins = bins
        self.bin_size = bin_size
        self.eps = eps

    @property
    def name(self):
        return "Survival"

    def criterion(self, logits, labels):
        """
        Custom criterion function with NaN checks added for debugging.

        Args:
        logits: Model output logits.
        labels: Ground truth labels containing survival information.

        Returns:
        Computed loss (mean) with NaN checks at each step.
        """
        # Extract censoring information and ground truth survival times
        censor = labels[:, 4]
        censor = censor.unsqueeze(1)

        os = self.target_transformation(labels)
        os = os.unsqueeze(1)

        # Compute hazard probabilities and prevent log(0) downstream with a small epsilon
        haz = logits.sigmoid() + self.eps
        # Compute cumulative survival probabilities
        sur = torch.cumprod(1 - haz, dim=1)
        # Add padding for survival, prepending 1 to the cumulative product
        sur_pad = torch.cat([torch.ones_like(censor), sur], dim=1)
        # Gather values at ground truth bin (using transformed targets)
        sur_pre = sur_pad.gather(dim=1, index=os)
        sur_cur = sur_pad.gather(dim=1, index=os + 1)
        haz_cur = haz.gather(dim=1, index=os)

        sur_pre = torch.clamp(sur_pre, min=self.eps)
        sur_cur = torch.clamp(sur_cur, min=self.eps)
        haz_cur = torch.clamp(haz_cur, min=self.eps)

        # Compute Negative Log-Likelihood (NLL) loss
        loss = (
            -(1 - censor) * sur_pre.log()  # for uncensored data
            - (1 - censor) * haz_cur.log()  # for hazard at event time
            - censor * sur_cur.log()  # for censored data
        )
        # Return the mean of the loss
        return loss.mean()

    def target_transformation(self, y):
        # extend bins to tensor with same lenght as labels
        bins = torch.full((y.shape[0],), self.bins - 1)
        target_labels = (torch.min(y[:, 5].clone() // self.bin_size, bins)).long()
        return target_labels

    def evaluation_performance_metric(self, x, y):
        if len(np.unique(y)) == 1:
            print(
                "Warning: Only one class present in y_true. C-Index score is not defined."
            )
            return 0
        c_index = concordance_index(y, x)
        return c_index

    def epoch_performance_metric(self, x, y):
        x = x.clone()
        x = x.squeeze()
        y = y.squeeze()
        os = y[:, 5].clone()
        neg_risk = self._risk_score(x)
        neg_risk = neg_risk.detach()
        return self.evaluation_performance_metric(neg_risk, os), 1

    @property
    def performance_metric_name(self):
        return "C-Index"

    @property
    def performance_metric_input_value(self):
        return "score"

    @property
    def evaluation_groups(self):
        return [
            (
                ["sex"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "sex",
            ),
            (
                ["age_bin"],
                {
                    "x": "age_bin",
                    "x_label": "Age Group",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "age",
            ),
            (
                ["sex", "age_bin"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": "age_bin",
                    "facet_col_label": "Age Group",
                },
                "sex_age",
            ),
        ]

    @property
    def num_classes(self):
        return self.bins

    def classification_criteria(self, logits):
        _, preds = torch.max(logits, 1)
        return preds

    def final_activation(self, logits):
        return self._risk_score(logits)

    def significance(self, gt, pred, recon):
        try:
            return bootstrap(gt, pred, recon, concordance_index)
        except ZeroDivisionError:
            print("Warning: No admissible pairs in the data.")
            return 1

    def _risk_score(self, logits):
        haz = logits.sigmoid() + self.eps
        sur = torch.cumprod(1 - haz, dim=1)
        return torch.sum(sur, dim=1)


class AgeCEClassifier(ClassifierModel):
    """
    Classifier for age prediction using cross entropy loss.
    """

    def __init__(self, age_bins):
        super().__init__()
        self.loss = (
            nn.CrossEntropyLoss() if len(age_bins) > 3 else nn.BCEWithLogitsLoss()
        )
        self.age_bins = age_bins
        self.age_labels = list(range(0, len(self.age_bins) - 1))

    @property
    def name(self):
        return "Age"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        transformed_labels = transformed_labels
        if len(self.age_bins) > 3:
            transformed_labels = transformed_labels.long()
        logits = logits.squeeze(1)
        loss = self.loss(logits, transformed_labels)
        return loss

    def target_transformation(self, y):
        target_labels = y[:, 6].clone()
        return target_labels

    def evaluation_performance_metric(self, x, y):
        if self.performance_metric_name == "Accuracy":
            correct = (x == y).sum().item()
            return correct / len(y)
        if len(np.unique(y)) == 1:
            print(
                "Warning: Only one class present in y_true. AUROC score is not defined."
            )
            return 0
        return roc_auc_score(y, x)

    def epoch_performance_metric(self, x, y):
        if self.performance_metric_name == "Accuracy":
            _, preds = torch.max(x, 1)
            target_transform = self.target_transformation(y)
            return self.evaluation_performance_metric(preds, target_transform), 1
        x = x.cpu().detach().numpy()
        target_transform = self.target_transformation(y)
        return self.evaluation_performance_metric(x, target_transform), 1

    @property
    def performance_metric_name(self):
        if len(self.age_bins) > 3:
            return "Accuracy"
        return "AUROC"

    @property
    def performance_metric_input_value(self):
        if len(self.age_bins) > 3:
            return "prediction"
        return "score"

    @property
    def evaluation_groups(self):
        return [
            (
                ["sex"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "sex",
            ),
            (
                ["age_bin"],
                {
                    "x": "age_bin",
                    "x_label": "Age Group",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "age",
            ),
            (
                ["sex", "age_bin"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": "age_bin",
                    "facet_col_label": "Age Group",
                },
                "sex_age",
            ),
        ]

    @property
    def num_classes(self):
        if len(self.age_bins) > 3:
            return len(self.age_bins) - 1
        return 1

    def classification_criteria(self, logits):
        if len(self.age_bins) > 3:
            _, preds = torch.max(logits, 1)
            return preds
        return torch.sigmoid(logits) > 0.5

    def final_activation(self, logits):
        if len(self.age_bins) > 3:
            logits = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            return preds
        return torch.sigmoid(logits)

    def significance(self, gt, pred, recon):
        if self.performance_metric_name == "Accuracy":
            p_value = hypothesis_test(pred, recon)
            return p_value
        p_value = delong_roc_test(gt, pred, recon)
        return p_value


class GenderBCEClassifier(ClassifierModel):
    """
    Classifier for tumor type prediction using binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    @property
    def name(self):
        return "Gender"

    def criterion(self, logits, labels):
        transformed_labels = self.target_transformation(labels)
        logits = logits.squeeze(1)
        loss = self.bce_loss(logits, transformed_labels)
        return loss

    def target_transformation(self, y):
        target_labels = y[:, 0].clone()
        target_labels[target_labels == "M"] = 1
        target_labels[target_labels == "F"] = 0
        return target_labels

    def evaluation_performance_metric(self, x, y):
        # Check if both classes (0 and 1) are present
        if len(np.unique(y)) == 1:
            print(
                "Warning: Only one class present in y_true. ROC AUC score is not defined."
            )
            return 0  # or return a default value like 0.5 if preferred
        return roc_auc_score(y, x)

    def epoch_performance_metric(self, x, y):
        x = x.detach().numpy()
        target_transform = self.target_transformation(y)
        return self.evaluation_performance_metric(x, target_transform), 1

    @property
    def performance_metric_name(self):
        return "AUROC"

    @property
    def performance_metric_input_value(self):
        return "score"

    @property
    def evaluation_groups(self):
        return [
            (
                ["sex"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "sex",
            ),
            (
                ["age_bin"],
                {
                    "x": "age_bin",
                    "x_label": "Age Group",
                    "facet_col": None,
                    "facet_col_label": None,
                },
                "age",
            ),
            (
                ["sex", "age_bin"],
                {
                    "x": "sex",
                    "x_label": "Sex",
                    "facet_col": "age_bin",
                    "facet_col_label": "Age Group",
                },
                "sex_age",
            ),
        ]

    @property
    def num_classes(self):
        return 1

    def classification_criteria(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits) > 0.5

    def final_activation(self, logits):
        logits = logits.squeeze()
        return torch.sigmoid(logits)

    def significance(self, gt, pred, recon):
        p_value = delong_roc_test(gt, pred, recon)
        return p_value
