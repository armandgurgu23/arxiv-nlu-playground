from typing import Dict


class MLModel:
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.initialize_model_architecture(self.model_config)

    def __call__(self, input_data):
        return self.forward_pass(input_data)

    def initialize_model_architecture(self, model_config: Dict):
        raise NotImplementedError(
            "Overwrite this method for initializing model architecture!"
        )

    def forward_pass(self, input_data):
        raise NotImplementedError("Overwrite this method for the forward pass logic!")

    def load_model(self, model_path: str):
        raise NotImplementedError("Overwrite this method for the model loading logic!")

    def save_model(self, model_path: str):
        raise NotImplementedError("Overwrite this method for the model saving logic!")
