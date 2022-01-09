from typing import Any, Dict


class MLModel:
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.model = self.initialize_model_architecture(self.model_config)

    def __call__(self, input_data: Any, return_predicted_labels: bool = False):
        return self.forward_pass(input_data, return_predicted_labels)

    def compute_loss(self, prediction_logits: Any, ground_truth: Any):
        raise NotImplementedError(
            "Overwrite this method for implementing loss function!"
        )

    def initialize_model_architecture(self, model_config: Dict):
        raise NotImplementedError(
            "Overwrite this method for initializing model architecture!"
        )

    def forward_pass(self, input_data: Any, return_predicted_labels: bool):
        raise NotImplementedError("Overwrite this method for the forward pass logic!")

    def load_model(self, model_path: str):
        raise NotImplementedError("Overwrite this method for the model loading logic!")

    def save_model(self, model_path: str):
        raise NotImplementedError("Overwrite this method for the model saving logic!")
