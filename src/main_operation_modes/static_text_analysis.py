from omegaconf import DictConfig
from typing import Dict, Generator, Tuple, List, Any, Type, Union
from data.corpus_reader import Corpus_Reader
from json import dumps, load
from os import path

from logging import getLogger

hydra_logger = getLogger(__name__)


class StaticTextAnalyzer(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(
        self,
        dataset_reader: Generator[Tuple[List[str], str], Any, Any],
        dataset_reader_handler: Type[Corpus_Reader],
    ):
        if (
            self.cfg.data.static_text_analysis.static_text_analysis_mode
            == "detect-keywords"
        ):
            self.handle_paper_semantic_section_detection(
                dataset_reader, dataset_reader_handler
            )
        else:
            path_to_analysis_file = self.get_section_keyword_analysis_file_path()
            jsonl_keyword_sums = StaticTextAnalyzer.get_jsonl_reader_iterator(
                path_to_analysis_file
            )
            for current_sum in jsonl_keyword_sums:
                print(current_sum)
            raise NotImplementedError("Support more static text analysis modes here!")
        return

    @staticmethod
    def get_jsonl_reader_iterator(jsonl_file_path: str):
        with open(jsonl_file_path, "r") as file_object:
            for current_paper_summary in file_object:
                print(current_paper_summary)
                yield load(current_paper_summary)

    def get_section_keyword_analysis_file_path(self):
        return path.join(
            self.cfg.data.static_text_analysis.section_keywords_analysis_path,
            self.cfg.data.static_text_analysis.keywords_dump_path,
        )

    def handle_paper_semantic_section_detection(
        self,
        dataset_reader: Generator[Tuple[List[str], str], Any, Any],
        dataset_reader_handler: Type[Corpus_Reader],
    ):
        query_keywords = self.get_section_keywords_to_query_in_papers(
            dataset_reader_handler
        )
        hydra_logger.info("Starting paper section keyword search across dataset!")
        for current_paper, paper_id in dataset_reader:
            current_paper_keyword_sum = self.find_keywords_in_paper(
                current_paper, query_keywords, dataset_reader_handler, paper_id
            )
            self.write_keyword_summary_to_file(
                current_paper_keyword_sum,
                self.cfg.data.static_text_analysis.keywords_dump_path,
            )
        hydra_logger.info(
            "Finished running static analysis on semantic section detection on research papers!"
        )
        return

    def write_keyword_summary_to_file(
        self, keyword_sum: Dict[str, Union[str, Dict]], output_filename: str
    ):
        if ".jsonl" in output_filename:
            with open(output_filename, "a") as file_object:
                file_object.write(dumps(keyword_sum))
                file_object.write("\n")
        else:
            raise NotImplementedError(f"Output of analysis only in jsonl so far!")

    def find_keywords_in_paper(
        self,
        paper_content: List[str],
        query_keywords: List[str],
        dataset_reader_handler: Type[Corpus_Reader],
        paper_id: str,
    ):
        paper_content_info_dict = {"id": paper_id, "section_keyword_detection": {}}
        for current_section_keyword in query_keywords:
            try:
                section_keyword_tuples = (
                    dataset_reader_handler.find_semantic_section_in_paper(
                        paper_content, current_section_keyword
                    )
                )
            except AssertionError:
                hydra_logger.info(
                    f"Paper {paper_id} does not contain section type: {current_section_keyword}"
                )
                section_keyword_tuples = ["Not found!"]
            paper_content_info_dict["section_keyword_detection"][
                current_section_keyword
            ] = section_keyword_tuples
        return paper_content_info_dict

    def get_section_keywords_to_query_in_papers(
        self, dataset_reader_handler: Type[Corpus_Reader]
    ):
        if self.cfg.data.static_text_analysis.semantic_section_keywords == "all":
            keywords_to_query = (
                dataset_reader_handler.paper_semantic_keywords_dict.keys()
            )
            keywords_to_query = list(keywords_to_query)
        return keywords_to_query
