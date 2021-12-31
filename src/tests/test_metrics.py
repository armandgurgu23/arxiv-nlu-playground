from hydra import initialize, compose
from metrics.text_classification_metrics import TextClassificationMetrics


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
        # TODO: Implement this test.
        pass
