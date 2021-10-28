from data.corpus_analyzer import Corpus_Analyzer
from hydra import initialize, compose
from pprint import pprint


class TestCorpusAnalyzer:
    def test_corpus_analyzer_newline_removal(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=exploration",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/partitionedDataset/train/economics_20/paper.txt",
                ],
            )
        corpus_analyzer = Corpus_Analyzer(**cfg.data.textdistance_config)
        test_paper_contents = corpus_analyzer(data_path=cfg.data.data_path)
        pprint(test_paper_contents)
        assert "\n" not in test_paper_contents
