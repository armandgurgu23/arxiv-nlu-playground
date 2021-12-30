from omegaconf import DictConfig


def main_text_classification_experiment_mode(cfg: DictConfig):
    print(cfg)
    raise NotImplementedError(
        "Please insert logic for running an experiment-text classification!"
    )


def main_experiment_mode(cfg: DictConfig):
    if cfg.trainer.experiment_task == "text_classification":
        main_text_classification_experiment_mode(cfg)
    else:
        raise NotImplementedError(
            f"Experiment type {cfg.trainer.experiment_task} not supported!"
        )
