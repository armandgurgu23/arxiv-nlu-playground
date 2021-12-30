from typing import List
from omegaconf import DictConfig
from models.sklearn_text_classifier import SklearnTextClassifier
from data.experiment_corpus_readers import SklearnTextClassificationReader


class TextClassificationTrainer:
    def __init__(self, cfg: DictConfig):
        self.full_config = cfg
        self.train_reader, self.valid_reader = self.initialize_dataset_readers(
            self.full_config.trainer.model_framework, self.full_config
        )
        self.model_class = self.initialize_model_class(
            self.full_config.trainer.model_framework,
            self.full_config,
            self.train_reader.labels,
        )

    def __call__(self):
        self.run_training_loop(self.full_config)
        return

    def run_training_loop(self, cfg: DictConfig):
        if self.full_config.trainer.model_framework == "sklearn":
            self.run_sklearn_training_loop(cfg)
        elif self.full_config.trainer.model_framework == "tensorflow":
            self.run_tensorflow_training_loop(cfg)
        else:
            raise NotImplementedError(
                f"Trainer does not support training for framework {self.full_config.trainer.model_framework}!"
            )

    def run_sklearn_training_loop(self, cfg: DictConfig):
        raise NotImplementedError("Insert full trainer logic for sklearn here!")

    def run_tensorflow_training_loop(self, cfg: DictConfig):
        raise NotImplementedError("Insert full trainer logic for tensorflow here!")

    def initialize_dataset_readers(self, model_framework: str, cfg: DictConfig):
        if model_framework == "sklearn":
            train_reader = SklearnTextClassificationReader(
                cfg.data, "train", cfg.experiment_seed
            )
            valid_reader = SklearnTextClassificationReader(
                cfg.data, "valid", cfg.experiment_seed
            )
        else:
            raise NotImplementedError(
                f"Reader framework {model_framework} for text classification not implemented!"
            )
        return train_reader, valid_reader

    def initialize_model_class(
        self, model_framework: str, cfg: DictConfig, mock_class_ids: List[int]
    ):
        if model_framework == "sklearn":
            model_class = SklearnTextClassifier(cfg.models, mock_class_ids)
        else:
            raise NotImplementedError(
                f"Model class framework {model_framework} for text classification not implemented!"
            )
        return model_class
