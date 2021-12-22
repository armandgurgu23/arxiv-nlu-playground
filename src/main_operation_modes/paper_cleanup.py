from omegaconf import DictConfig
from data.corpus_reader import Corpus_Reader
from main_operation_modes.generate_cleaned_papers import CleanedPapersGenerator


def main_paper_cleanup_mode(cfg: DictConfig):
    print(cfg)
    corpus_reader = Corpus_Reader(**cfg.data)
    corpus_data_iter = corpus_reader()
    if cfg.data.paper_cleanup_mode == "generate-cleaned-papers":
        cleaned_papers_generator = CleanedPapersGenerator(cfg)
        cleaned_papers_generator(corpus_data_iter, corpus_reader)
    else:
        raise NotImplementedError(
            f"Exploration mode {cfg.data.exploration_mode} not supported!"
        )
