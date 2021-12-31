from typing import Any, Dict, List
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix


class TextClassificationMetrics:
    def __init__(self, cfg: Dict, num_labels: int, dataset_type: str):
        self.cfg = cfg
        if dataset_type == "train":
            self.metrics_summary = {
                "accuracy": Accuracy(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.train_config.compute_on_step,
                ),
                "precision": Precision(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.train_config.compute_on_step,
                ),
                "recall": Recall(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.train_config.compute_on_step,
                ),
                "f1": F1(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.train_config.compute_on_step,
                ),
                "confusion_matrix": ConfusionMatrix(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.train_config.compute_on_step,
                ),
            }
        else:
            self.metrics_summary = {
                "accuracy": Accuracy(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.test_config.compute_on_step,
                ),
                "precision": Precision(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.test_config.compute_on_step,
                ),
                "recall": Recall(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.test_config.compute_on_step,
                ),
                "f1": F1(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.test_config.compute_on_step,
                ),
                "confusion_matrix": ConfusionMatrix(
                    num_classes=num_labels,
                    threshold=self.cfg.metrics_config.threshold,
                    compute_on_step=self.cfg.metrics_config.test_config.compute_on_step,
                ),
            }

    def __call__(self, predictions: Any, ground_truth: List[int]):
        # TODO: Change the type of predictions later.
        # Compute the minibatch metric performance here.
        per_step_metric = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": None,
        }
        for current_metric in self.metrics_summary:
            metric_value = self.metrics_summary[current_metric](
                predictions, ground_truth
            )
            per_step_metric[current_metric] = metric_value
        return per_step_metric

    def compute_global_metric_performance(self):
        global_metrics_summary = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": None,
        }
        for current_metric in self.metrics_summary:
            curr_global_value = self.metrics_summary[current_metric].compute()
            global_metrics_summary[current_metric] = curr_global_value
            # Reset each metric to be used for a new epoch calculation.
            self.metrics_summary[current_metric].reset()
        return global_metrics_summary
