from omegaconf import DictConfig
from data.corpus_reader import Corpus_Reader
from main_operation_modes.static_text_analysis import StaticTextAnalyzer


def main_exploration_mode(cfg: DictConfig):
    corpus_reader = Corpus_Reader(**cfg.data)
    corpus_data_iter = corpus_reader()
    if cfg.data.exploration_mode == "static-text-analysis":
        static_text_analyzer = StaticTextAnalyzer(cfg)
        static_text_analyzer(corpus_data_iter, corpus_reader)
    else:
        raise NotImplementedError(
            f"Exploration mode {cfg.data.exploration_mode} not supported!"
        )
    return
