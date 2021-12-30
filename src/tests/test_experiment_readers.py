from hydra import initialize, compose
from data.experiment_corpus_readers import SklearnTextClassificationReader
from scipy.sparse.csr import csr_matrix


class TestSklearnTextClassificationReader:
    def test_sklearn_text_classification_reader_iteration(self):
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
                ],
            )
        sklearn_experiment_reader = SklearnTextClassificationReader(
            cfg.data, "train", cfg.experiment_seed
        )
        reader_sample = None
        for current_sample in sklearn_experiment_reader:
            reader_sample = current_sample
            break
        assert reader_sample[2] == ["robotics_30", "astrophysics_70"]
        assert len(reader_sample[1]) == 408
        assert isinstance(reader_sample[0], csr_matrix)
