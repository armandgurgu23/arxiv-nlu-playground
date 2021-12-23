from logging import getLogger, log
import logging
from omegaconf import DictConfig
from typing import Generator, Tuple, List, Any, Type
from data.corpus_reader import Corpus_Reader, ReferenceInfo
from os import getcwd, path, makedirs

hydra_logger = getLogger(__name__)


class CleanedPapersGenerator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(
        self,
        dataset_reader: Generator[Tuple[List[str], str], Any, Any],
        dataset_reader_handler: Type[Corpus_Reader],
    ):
        output_dataset_path, dataset_type = self.create_dataset_type_folder_structure()
        for current_paper_contents, paper_id in dataset_reader:
            logging.info(
                f"Starting to clean up paper {paper_id} for dataset type {dataset_type}!"
            )
            (
                reference_tuple,
                current_paper_contents,
            ) = self.find_start_of_references_in_paper(
                current_paper_contents, dataset_reader_handler
            )
            # In case we cannot find an indicator for a reference section, we keep the entire
            # paper and document these ids.
            if reference_tuple:
                cleaned_paper_contents = self.extract_non_reference_paper_contents(
                    current_paper_contents, reference_tuple
                )
            else:
                logging.error(
                    f"Could not detect the start of the references section for paper {paper_id} in dataset type {dataset_type}"
                )
                cleaned_paper_contents = current_paper_contents
            paper_contents_str = self.join_paper_contents_together(
                cleaned_paper_contents
            )
            CleanedPapersGenerator.write_paper_contents_to_disk(
                output_dataset_path, paper_id, paper_contents_str
            )
            logging.info(
                f"Finished cleaning up paper {paper_id} for dataset type {dataset_type}!"
            )
        logging.info(
            f"Finished processing paper dataset! Output path: {output_dataset_path}"
        )
        return

    def create_dataset_type_folder_structure(self):
        dataset_type = self.cfg.data.data_path.split("/")[-1]
        dataset_output_path = path.join(getcwd(), dataset_type)
        CleanedPapersGenerator.create_directory(dataset_output_path)
        return dataset_output_path, dataset_type

    @staticmethod
    def write_paper_contents_to_disk(
        folder_path: str, paper_id: str, paper_contents: str
    ):
        output_path = path.join(folder_path, paper_id)
        output_filename = path.join(output_path, "processed_paper.txt")
        CleanedPapersGenerator.create_directory(output_path)
        with open(output_filename, "w") as file_object:
            file_object.write(paper_contents)
        return

    @staticmethod
    def create_directory(dir_path: str):
        if not path.exists(dir_path):
            makedirs(dir_path)
        return

    def join_paper_contents_together(self, cleaned_paper_contents: List[str]):
        output_contents = []
        for current_line in cleaned_paper_contents:
            current_line = self.remove_newline_characters_and_handle_spacing(
                current_line
            )
            output_contents.append(current_line)
        return "".join(output_contents)

    def remove_newline_characters_and_handle_spacing(self, paper_line_string: str):
        if paper_line_string.endswith("\n") and paper_line_string[-2] == "-":
            paper_line_string = paper_line_string.replace("-\n", "")
        else:
            paper_line_string = paper_line_string.replace("\n", " ")
        return paper_line_string

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
