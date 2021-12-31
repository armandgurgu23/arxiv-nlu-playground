from typing import Dict
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

    def __call__(self):
        # Compute the minibatch metric performance here.
        pass

    def compute_global_metric_performance(self):
        pass
