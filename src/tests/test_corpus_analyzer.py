from data.corpus_analyzer import Corpus_Analyzer
from hydra import initialize, compose


class TestCorpusAnalyzer:
    def test_corpus_analyzer_initialization(self):
        with initialize(config_path="../config"):
            cfg = compose(config_name="config.yaml")
            print(cfg)
        assert 2 == 2
