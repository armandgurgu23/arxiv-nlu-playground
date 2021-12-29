import logging
from os import path, listdir
from typing import Dict, List, Optional

from utils.dataset_utils import get_all_classes_for_text_classification
from logging import getLogger
import re
import random

hydra_logger = getLogger(__name__)


class SklearnTextClassificationReader:
    def __init__(
        self, reader_cfg: Dict, dataset_type: str, experiment_seed: Optional[int] = None
    ):
        if not path.exists(path.join(reader_cfg.data_path, dataset_type)):
            raise RuntimeError(
                f"Dataset type {dataset_type} does not exist in dataset at path {reader_cfg.data_path}!"
            )
        self.reader_cfg = reader_cfg
        self.experiment_seed = experiment_seed
        self.dataset_type = dataset_type
        self.dataset_path = path.join(self.reader_cfg.data_path, self.dataset_type)
        self.paper_categories = get_all_classes_for_text_classification(
            self.dataset_path
        )
        self.punctuation_regex = re.compile("[.!?]")

    @property
    def labels(self):
        return self.paper_categories

    def __iter__(self):
        dataset_paper_ids = listdir(self.dataset_path)
        if self.experiment_seed:
            self.shuffle_papers(self.experiment_seed, dataset_paper_ids)
        for paper_id in dataset_paper_ids:
            if paper_id.split("_")[0] not in self.paper_categories:
                logging.warning(
                    f"Read in {paper_id} id which is not part of the paper category set! Unexpected behaviour!"
                )
                continue
            raw_paper_contents = self.open_paper_id_contents(
                paper_id, self.dataset_path
            )
            paper_sentences = self.split_paper_contents_into_sentences(
                raw_paper_contents
            )
            paper_label_vector = self.create_paper_contents_label_vector(
                paper_id, len(paper_sentences)
            )
            print(paper_label_vector)
            raise NotImplementedError()

    def shuffle_papers(self, seed: int, paper_ids: List[str]) -> None:
        random.seed(seed)
        random.shuffle(paper_ids)
        return

    def create_paper_contents_label_vector(
        self, paper_id: str, num_sentences_in_paper: int
    ) -> List[int]:
        class_number = self.paper_categories.index(paper_id.split("_")[0])
        return [class_number] * num_sentences_in_paper

    def open_paper_id_contents(self, paper_id: str, dataset_path: str) -> str:
        with open(
            path.join(dataset_path, paper_id, "processed_paper.txt"), "r"
        ) as file_object:
            paper_text_contents = file_object.read()
        return paper_text_contents

    def split_paper_contents_into_sentences(self, raw_paper_contents: str) -> List[str]:
        paper_sentences = re.split(self.punctuation_regex, raw_paper_contents)
        return [
            sentence for sentence in paper_sentences if sentence.replace(" ", "") != ""
        ]
