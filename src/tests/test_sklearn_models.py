from hydra import initialize, compose
from models.sklearn_text_classifier import SklearnTextClassifier
from sklearn.linear_model import SGDClassifier


class TestSklearnTextClassifier:
    def test_sklearn_text_classifier_initialization(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=experiments",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/processedPaperDataset_2021-12-23_12-48-04",
                    "models=text_classification",
                    "models.sklearn_model_config.model_type=sgd-classifier",
                    "models.sklearn_model_config.sgd_config.learning_rate=constant",
                ],
            )
        sklearn_classifier = SklearnTextClassifier(cfg.models)
        assert isinstance(sklearn_classifier.model, SGDClassifier)
        assert sklearn_classifier.model.learning_rate == "constant"
