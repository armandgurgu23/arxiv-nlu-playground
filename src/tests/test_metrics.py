from hydra import initialize, compose
from metrics.text_classification_metrics import TextClassificationMetrics
import numpy as np


class TestTextClassificationMetrics:
    def test_text_classification_metrics_initialization(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=experiments",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/processedPaperDataset_2021-12-23_12-48-04",
                    "data.batch_size=2",
                    "models=text_classification_sklearn",
                    "models.sklearn_model_config.model_type=sgd-classifier",
                    "models.sklearn_model_config.sgd_config.learning_rate=constant",
                    "metrics=text_classification_metrics",
                    "metrics.metrics_config.threshold=0.5",
                    "metrics.metrics_config.train_config.compute_on_step=true",
                    "metrics.metrics_config.test_config.compute_on_step=false",
                ],
            )
        tc_train_metrics = TextClassificationMetrics(
            cfg.metrics, num_labels=4, dataset_type="train"
        )
        tc_valid_metrics = TextClassificationMetrics(
            cfg.metrics, num_labels=4, dataset_type="test"
        )
        assert list(tc_train_metrics.metrics_summary.keys()) == [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "confusion_matrix",
        ]
        assert (
            tc_train_metrics.metrics_summary["accuracy"].__dict__["compute_on_step"]
            == True
        )
        assert (
            tc_valid_metrics.metrics_summary["accuracy"].__dict__["compute_on_step"]
            == False
        )

    def test_text_classification_metrics_global_calculation(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=experiments",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/processedPaperDataset_2021-12-23_12-48-04",
                    "data.batch_size=2",
                    "models=text_classification_sklearn",
                    "models.sklearn_model_config.model_type=sgd-classifier",
                    "models.sklearn_model_config.sgd_config.learning_rate=constant",
                    "metrics=text_classification_metrics",
                    "metrics.metrics_config.threshold=0.5",
                    "metrics.metrics_config.mdmc_average=samplewise",
                    "metrics.metrics_config.train_config.compute_on_step=true",
                    "metrics.metrics_config.test_config.compute_on_step=false",
                ],
            )
        mock_preds_b1 = [[2, 0, 1, 3], [2, 0, 1, 2]]
        ground_truths_b1 = [[2, 0, 1, 2], [2, 0, 1, 1]]
        mock_preds_b2 = [[1, 0, 1, 3], [1, 0, 1, 2]]
        ground_truths_b2 = [[1, 2, 1, 2], [1, 3, 1, 1]]
        all_mock_predictions = np.array([mock_preds_b1, mock_preds_b2])
        all_mock_gt = np.array([ground_truths_b1, ground_truths_b2], dtype=np.int32)
        tc_train_metrics = TextClassificationMetrics(
            cfg.metrics, num_labels=4, dataset_type="train"
        )
        samplewise_metric_values = []
        for batch_index in range(all_mock_predictions.shape[0]):
            batch_preds = all_mock_predictions[batch_index, ...]
            batch_gt = all_mock_gt[batch_index, ...]
            curr_metric_info = tc_train_metrics(batch_preds, batch_gt)
            samplewise_metric_values.append(curr_metric_info)
        global_metric_info = tc_train_metrics.compute_global_metric_performance()
        assert samplewise_metric_values[0]["precision"].item() == 0.7500
        assert samplewise_metric_values[1]["precision"].item() == 0.5000
        assert global_metric_info["precision"].item() == 0.625
