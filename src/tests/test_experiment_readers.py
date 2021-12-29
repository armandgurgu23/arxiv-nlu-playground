from hydra import initialize, compose
from data.experiment_corpus_readers import SklearnTextClassificationReader


class TestSklearnTextClassificationReader:
    def test_sklearn_text_classification_reader_initialization(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=experiments",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/processedPaperDataset_2021-12-23_12-48-04",
                    "data.batch_size=2",
                    "models=text_classification",
                    "models.sklearn_model_config.model_type=sgd-classifier",
                    "models.sklearn_model_config.sgd_config.learning_rate=constant",
                ],
            )
        sklearn_experiment_reader = SklearnTextClassificationReader(
            cfg.data, "train", cfg.experiment_seed
        )
        for current_sample in sklearn_experiment_reader:
            pass
        assert 2 == 2
