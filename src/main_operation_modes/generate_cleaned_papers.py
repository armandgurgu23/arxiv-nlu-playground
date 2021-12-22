from logging import getLogger
from omegaconf import DictConfig
from typing import Generator, Tuple, List, Any, Type
from data.corpus_reader import Corpus_Reader, ReferenceInfo

hydra_logger = getLogger(__name__)


class CleanedPapersGenerator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(
        self,
        dataset_reader: Generator[Tuple[List[str], str], Any, Any],
        dataset_reader_handler: Type[Corpus_Reader],
    ):
        for current_paper_contents, paper_id in dataset_reader:
            (
                reference_tuple,
                current_paper_contents,
            ) = self.find_start_of_references_in_paper(
                current_paper_contents, dataset_reader_handler
            )
            cleaned_paper_contents = self.extract_non_reference_paper_contents(
                current_paper_contents, reference_tuple
            )
        return

    def extract_non_reference_paper_contents(
        self, current_paper_contents: List[str], reference_tuple: ReferenceInfo
    ):
        return current_paper_contents[0 : reference_tuple[0][-1]]

    def find_start_of_references_in_paper(
        self,
        current_paper_contents: List[str],
        dataset_reader_handler: Type[Corpus_Reader],
    ) -> Tuple[ReferenceInfo, List[str]]:
        current_paper_contents = dataset_reader_handler.paper_contents_processor.remove_paper_contents_by_token_count(
            current_paper_contents,
            self.cfg.data.token_processing.token_threshold,
            dataset_reader_handler.paper_semantic_keywords["refer"],
        )
        start_of_reference_tuple = (
            dataset_reader_handler.find_beginning_of_references_in_paper(
                current_paper_contents, "refer"
            )
        )
        return start_of_reference_tuple, current_paper_contents
