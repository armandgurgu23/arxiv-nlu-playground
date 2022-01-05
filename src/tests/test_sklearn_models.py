from hydra import initialize, compose
from models.sklearn_text_classifier import SklearnTextClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np


class TestSklearnTextClassifier:
    def test_sklearn_text_classifier_initialization(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=experiments",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/processedPaperDataset_2021-12-23_12-48-04",
                    "models=text_classification_sklearn",
                    "models.sklearn_model_config.model_type=sgd-classifier",
                    "models.sklearn_model_config.sgd_config.learning_rate=constant",
                ],
            )
        mock_class_ids = [0, 1, 2, 3]
        sklearn_classifier = SklearnTextClassifier(cfg.models, mock_class_ids)
        assert isinstance(sklearn_classifier.model, SGDClassifier)
        assert sklearn_classifier.model.learning_rate == "constant"

    def test_sklearn_text_classifier_forward_pass(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=experiments",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/processedPaperDataset_2021-12-23_12-48-04",
                    "models=text_classification_sklearn",
                    "models.sklearn_model_config.model_type=sgd-classifier",
                    "models.sklearn_model_config.sgd_config.learning_rate=constant",
                ],
            )
        mock_class_ids = [0, 1, 2, 3]
        # Input data should mock a featurized input with dims [batch_size, num_text_features].
        mock_input_data = np.eye(N=4, M=6)
        mock_ground_truth = np.array([0, 0, 1, 3])
        mock_input_data_featurized = np.array(
            [
                [
                    2.30940108,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    2.30940108,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    2.30940108,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    2.30940108,
                    0,
                    0,
                ],
            ]
        )
        sklearn_classifier = SklearnTextClassifier(cfg.models, mock_class_ids)
        mock_prediction_logits = sklearn_classifier(mock_input_data)
        loss_value = sklearn_classifier.compute_loss(
            mock_prediction_logits, mock_ground_truth
        )
        # AFAIK you cannot access the loss value of scikit learn models. Hence
        # for sklearn models we return a null loss value.
        assert loss_value == None
        np.testing.assert_allclose(mock_prediction_logits, mock_input_data_featurized)
