from typing import List, Dict
from omegaconf import DictConfig
from models.sklearn_text_classifier import SklearnTextClassifier
from data.experiment_corpus_readers import SklearnTextClassificationReader
from metrics.text_classification_metrics import TextClassificationMetrics
from logging import getLogger
import numpy as np
import os
import mlflow

hydra_logger = getLogger(__name__)


class TextClassificationTrainer:
    def __init__(self, cfg: DictConfig):
        self.full_config = cfg
        self.train_reader, self.valid_reader = self.initialize_dataset_readers(
            self.full_config.trainer.model_framework, self.full_config
        )
        self.ids_to_labels = self.create_label_name_to_label_id_mapping(
            self.train_reader
        )
        self.model_class = self.initialize_model_class(
            self.full_config.trainer.model_framework,
            self.full_config,
            list(self.ids_to_labels.keys()),
        )
        self.train_metrics = TextClassificationMetrics(
            self.full_config.metrics,
            num_labels=len(self.train_reader.labels),
            dataset_type="train",
        )
        self.valid_metrics = TextClassificationMetrics(
            self.full_config.metrics,
            num_labels=len(self.valid_reader.labels),
            dataset_type="test",
        )

    def create_label_name_to_label_id_mapping(
        self, train_reader: SklearnTextClassificationReader
    ) -> Dict[int, str]:
        labels_array = train_reader.labels
        id_to_labels_dict = {}
        for current_id, current_label in enumerate(labels_array):
            id_to_labels_dict[current_id] = current_label
        return id_to_labels_dict

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

    def get_mlflow_tracking_uri_and_experiment_name(self):
        curr_path = os.getcwd()
        if "temp_save" in curr_path:
            return (
                curr_path,
                f"{self.full_config.trainer.experiment_task}_{curr_path.split('/')[-1]}",
            )
        else:
            raise NotImplementedError(
                "Apply processing on path to be used for MLflow when running actual experiments!"
            )

    def run_sklearn_training_loop(self, cfg: DictConfig):
        hydra_logger.info(
            f"Trainer initialized sklearn model! Training model for {cfg.trainer.train_epochs} epochs!"
        )
        (
            mlflow_experiments_uri,
            mlflow_experiment_name,
        ) = self.get_mlflow_tracking_uri_and_experiment_name()
        # Need to add /mlruns at the end of the URI, otherwise mlflow will
        # try to select a subfolder at random.
        mlflow.set_tracking_uri(f"file://{mlflow_experiments_uri}/mlruns")
        mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
        with mlflow.start_run(experiment_id=mlflow_experiment_id):
            mlflow.log_artifacts(os.path.join(os.getcwd(), ".hydra"))
            for current_epoch in range(cfg.trainer.train_epochs):
                hydra_logger.info(f"Starting training epoch {current_epoch}")
                self.train_reader, self.valid_reader = self.initialize_dataset_readers(
                    self.full_config.trainer.model_framework, self.full_config
                )
                self.run_sklearn_training_epoch(self.train_reader)
                self.run_sklearn_validation_epoch(self.valid_reader)
                if (
                    current_epoch % self.full_config.trainer.save_after_num_epochs == 0
                    and current_epoch != 0
                ):
                    checkpoint_suffix = f"model_epoch_{current_epoch}"
                    self.model_class.save_model(
                        os.path.join(os.getcwd(), checkpoint_suffix)
                    )
            hydra_logger.info(
                f"Trainer finished training sklearn model for {cfg.trainer.train_epochs} epochs!"
            )
            self.model_class.save_model(os.path.join(os.getcwd(), "final_model"))
        return

    def run_sklearn_validation_epoch(
        self, valid_reader: SklearnTextClassificationReader
    ):
        for valid_minibatch in valid_reader:
            paper_features, paper_labels, _ = valid_minibatch
            predicted_labels = self.model_class(
                paper_features, return_predicted_labels=True
            )
            paper_labels = np.array(paper_labels, dtype=np.int32)
            self.valid_metrics(predicted_labels, paper_labels)
        epoch_summary_valid_metrics = (
            self.valid_metrics.compute_global_metric_performance()
        )
        hydra_logger.info(
            f"Validation metrics summary: P = {epoch_summary_valid_metrics['precision'].item()} R = {epoch_summary_valid_metrics['recall'].item()} F1 = {epoch_summary_valid_metrics['f1'].item()} Acc = {epoch_summary_valid_metrics['accuracy'].item()}"
        )
        return

    def run_sklearn_training_epoch(self, train_reader: SklearnTextClassificationReader):
        for train_minibatch in train_reader:
            paper_features, paper_labels, _ = train_minibatch
            # Run a forward pass on the sklearn model.
            prediction_logits = self.model_class(paper_features)
            self.model_class.compute_loss(prediction_logits, paper_labels)
        return

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
