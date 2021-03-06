import hydra
from main_operation_modes.exploration import main_exploration_mode
from main_operation_modes.paper_cleanup import main_paper_cleanup_mode
from main_operation_modes.experiments import main_experiment_mode
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.main_operation_mode == "exploration":
        main_exploration_mode(cfg)
    elif cfg.main_operation_mode == "paper-cleanup":
        main_paper_cleanup_mode(cfg)
    elif cfg.main_operation_mode == "experiment":
        main_experiment_mode(cfg)
    else:
        raise NotImplementedError(
            f"Operation mode {cfg.main_operation_mode} not supported!"
        )
    return


if __name__ == "__main__":
    main()
