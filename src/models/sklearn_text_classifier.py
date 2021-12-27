from typing import List
from models.ml_model_base import MLModel, Dict, Any
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from logging import getLogger

hydra_logger = getLogger(__name__)


class SklearnModel(MLModel):
    def __init__(self, model_config: Dict):
        super().__init__(model_config)

    def __call__(self, input_data: Any):
        # TODO: Change the type above later!
        return super().__call__(input_data)

    def save_model(self, model_path: str):
        mlflow.sklearn.save_model(
            self.model,
            model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        hydra_logger.info(f"Finished saving sklearn model at: {model_path}!")
        return

    def load_model(self, model_path: str):
        sklearn_model = mlflow.sklearn.load_model(model_path)
        # TODO: Figure out how to best reassign the class model attribute
        # once starting on the inference mode.
        return


class SklearnTextClassifier(SklearnModel):
    def __init__(self, model_config: Dict, class_ids: List[int]):
        super().__init__(model_config)
        self.class_ids = class_ids

    def initialize_model_architecture(self, model_config: Dict):
        if model_config.sklearn_model_config.feature_preprocessing:
            self.feature_preprocessors = self.setup_feature_preprocessor(
                model_config.sklearn_model_config.feature_preprocessing_config
            )
        return self.setup_sklearn_classifier(model_config.sklearn_model_config)

    def compute_loss(self, prediction_logits: Any, ground_truth: Any):
        self.model.partial_fit(prediction_logits, ground_truth, classes=self.class_ids)
        return

    def forward_pass(self, input_data: Any):
        if hasattr(self, "feature_preprocessors"):
            input_data = self.feature_preprocessors.fit_transform(input_data)
        return input_data

    def setup_feature_preprocessor(self, feature_preprocessor_config: Dict):
        pipeline_steps = []
        if feature_preprocessor_config.use_standard_scaler:
            pipeline_steps.append(("standard-scaler", StandardScaler()))
        return Pipeline(pipeline_steps)

    def setup_sklearn_classifier(self, sklearn_model_config: Dict):
        if sklearn_model_config.model_type == "sgd-classifier":
            return SGDClassifier(**sklearn_model_config.sgd_config)
        else:
            raise NotImplementedError(
                f"For sklearn engine, training text classifier {sklearn_model_config.model_type} is not supported!"
            )
