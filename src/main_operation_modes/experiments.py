from omegaconf import DictConfig
from trainer.trainer import TextClassificationTrainer


def main_text_classification_experiment_mode(cfg: DictConfig):
    tc_experiment_trainer = TextClassificationTrainer(cfg)
    tc_experiment_trainer()
    return


def main_experiment_mode(cfg: DictConfig):
    if cfg.trainer.experiment_task == "text_classification":
        main_text_classification_experiment_mode(cfg)
    else:
        raise NotImplementedError(
            f"Experiment type {cfg.trainer.experiment_task} not supported!"
        )
