from types import GeneratorType
from data.corpus_reader import Corpus_Reader
from hydra import initialize, compose


class TestCorpusReader:
    def test_corpus_reader_newline_removal(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=exploration",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/partitionedDataset/train/economics_20/paper.txt",
                ],
            )
        corpus_reader = Corpus_Reader(**cfg.data)
        test_paper_contents = corpus_reader()
        assert "\n" not in test_paper_contents and corpus_reader.labels == "economics"

    def test_corpus_reader_detect_intro_section(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=exploration",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/partitionedDataset/train/economics_20/paper.txt",
                ],
            )
        corpus_reader = Corpus_Reader(**cfg.data)
        test_paper_contents = corpus_reader()
        intro_section_tuples = corpus_reader.find_semantic_section_in_paper(
            test_paper_contents, "intro"
        )
        assert (
            intro_section_tuples[0][1] == "introduction"
            and corpus_reader.labels == "economics"
        )

    def test_corpus_reader_detect_conclusion_section(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=exploration",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/partitionedDataset/train/economics_21/paper.txt",
                ],
            )
        corpus_reader = Corpus_Reader(**cfg.data)
        test_paper_contents = corpus_reader()
        conclusion_section_tuples = corpus_reader.find_semantic_section_in_paper(
            test_paper_contents, "conc"
        )
        assert (
            conclusion_section_tuples[0][1] == "conclusion"
            and corpus_reader.labels == "economics"
        )

    def test_corpus_reader_dataset_iterator(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=exploration",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/partitionedDataset/train/",
                ],
            )
        corpus_reader = Corpus_Reader(**cfg.data)
        corpus_data_iter = corpus_reader()
        sample_paper_contents, paper_category = next(corpus_data_iter)
        assert (
            "economics" in paper_category
            and type(corpus_data_iter) == GeneratorType
            and type(sample_paper_contents) == list
        )

    def test_corpus_reader_unique_labels(self):
        with initialize(config_path="../config"):
            cfg = compose(
                config_name="config.yaml",
                overrides=[
                    "data=exploration",
                    "data.data_path=/Users/armandgurgu/Documents/datasets_side_projects/researchPapersDatasets/partitionedDataset/train/",
                ],
            )
        corpus_reader = Corpus_Reader(**cfg.data)
        assert set(corpus_reader.labels) == set(
            [
                "economics",
                "astrophysics",
                "robotics",
                "biology",
            ]
        )
