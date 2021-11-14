from omegaconf import DictConfig
from typing import Dict, Generator, Tuple, List, Any, Type, Union
from data.corpus_reader import Corpus_Reader
from json import dumps
from os import path
from utils.serialization_utils import get_jsonl_reader_iterator, write_dict_to_json_file

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
        elif (
            self.cfg.data.static_text_analysis.static_text_analysis_mode
            == "detect-first-reference"
        ):
            self.handle_paper_first_reference_detection(
                dataset_reader, dataset_reader_handler
            )
        elif (
            self.cfg.data.static_text_analysis.static_text_analysis_mode
            == "first-reference-analysis"
        ):
            self.handle_first_reference_analysis_mode()
        elif (
            self.cfg.data.static_text_analysis.static_text_analysis_mode
            == "section-keywords-analysis"
        ):
            path_to_analysis_file = self.get_section_keyword_analysis_file_path()
            jsonl_keyword_sums = StaticTextAnalyzer.get_jsonl_keywords_iterator(
                path_to_analysis_file
            )
            summary_dict = self.compute_semantic_section_full_dataset_metrics(
                jsonl_keyword_sums
            )
            StaticTextAnalyzer.write_keywords_json_to_disk(
                "keywords_analysis.json", summary_dict
            )
        else:
            raise NotImplementedError("Support more static text analysis modes here!")
        return

    def handle_first_reference_analysis_mode(self):
        path_to_analysis_file = self.get_section_keyword_analysis_file_path()
        jsonl_keyword_sums = StaticTextAnalyzer.get_jsonl_keywords_iterator(
            path_to_analysis_file
        )
        (
            single_count_ref_json,
            non_single_count_ref_json,
        ) = self.compute_first_reference_detection_breakdown(jsonl_keyword_sums)
        StaticTextAnalyzer.write_keywords_json_to_disk(
            "first_ref_single_counts_analysis.json", single_count_ref_json
        )
        StaticTextAnalyzer.write_keywords_json_to_disk(
            "first_ref_non_single_counts_analysis.json", non_single_count_ref_json
        )
        hydra_logger.info("Finished analyzing first reference detection results!")
        return

    def compute_first_reference_detection_breakdown(
        self, jsonl_data_iter: Generator[Dict, None, None]
    ) -> Tuple[Dict, Dict]:
        single_count_first_refs_json = {
            "total_count": 0,
            "dataset_size_diff": 0,
            "samples_info": {},
        }
        non_single_count_refs_json = {
            "total_count": 0,
            "dataset_size_diff": 0,
            "samples_info": {},
        }
        dataset_size = 0
        for current_jsonl_sample in jsonl_data_iter:
            dataset_size += 1
            current_paper_id = current_jsonl_sample["id"]
            if current_jsonl_sample["first_ref_det_info"]["count"] == 1:
                single_count_first_refs_json["total_count"] += 1
                single_count_first_refs_json["samples_info"][
                    current_paper_id
                ] = current_jsonl_sample["first_ref_det_info"]["ref_tuples"]
            else:
                non_single_count_refs_json["total_count"] += 1
                non_single_count_refs_json["samples_info"][
                    current_paper_id
                ] = current_jsonl_sample["first_ref_det_info"]["ref_tuples"]
        single_count_first_refs_json["dataset_size_diff"] = (
            dataset_size - single_count_first_refs_json["total_count"]
        )
        non_single_count_refs_json["dataset_size_diff"] = (
            dataset_size - non_single_count_refs_json["total_count"]
        )
        return single_count_first_refs_json, non_single_count_refs_json

    def handle_paper_first_reference_detection(
        self,
        dataset_reader: Generator[Tuple[List[str], str], Any, Any],
        dataset_reader_handler: Type[Corpus_Reader],
    ):
        hydra_logger.info("Searching dataset for presence of first reference!")
        for current_paper_contents, paper_id in dataset_reader:
            current_paper_contents = dataset_reader_handler.paper_contents_processor.remove_paper_contents_by_token_count(
                current_paper_contents, self.cfg.data.token_processing.token_threshold
            )
            sample_first_ref_det = self.find_first_reference_matches_in_paper(
                current_paper_contents, dataset_reader_handler, paper_id
            )
            self.write_keyword_summary_to_file(
                sample_first_ref_det,
                self.cfg.data.static_text_analysis.keywords_dump_path,
            )
        hydra_logger.info("Finished searching dataset for first reference!")
        return

    def find_first_reference_matches_in_paper(
        self,
        current_paper_contents: List[str],
        dataset_reader_handler: Type[Corpus_Reader],
        paper_id: str,
    ):
        detection_info_dict = {"id": paper_id, "first_ref_det_info": {}}
        first_ref_matches = dataset_reader_handler.find_first_reference_in_paper(
            current_paper_contents
        )
        if len(first_ref_matches) != 1:
            hydra_logger.info(
                f"Searching for first reference returned {len(first_ref_matches)} matches for paper {paper_id}! Inspect further!"
            )
        detection_info_dict["first_ref_det_info"]["count"] = len(first_ref_matches)
        detection_info_dict["first_ref_det_info"]["ref_tuples"] = first_ref_matches
        return detection_info_dict

    def compute_semantic_section_full_dataset_metrics(
        self, jsonl_data_iter: Generator[Dict, None, None]
    ):

        dataset_size = 0
        keywords_breakdown = {
            "intro": {"count": 0, "paper_ids_missing": []},
            "ending": {"count": 0, "paper_ids_missing": []},
        }
        endings_sample_assignment = {}
        for current_sample in jsonl_data_iter:
            dataset_size += 1
            endings_sample_assignment[current_sample["id"]] = False
            for current_section_type in current_sample["section_keyword_detection"]:
                detection_section_info = current_sample["section_keyword_detection"][
                    current_section_type
                ]
                if not isinstance(detection_section_info[0], str):
                    if current_section_type == "intro":
                        keywords_breakdown["intro"]["count"] += 1
                    else:
                        # Visiting an ending section.
                        if not endings_sample_assignment[current_sample["id"]]:
                            keywords_breakdown["ending"]["count"] += 1
                            endings_sample_assignment[current_sample["id"]] = True
                else:
                    # This keyword was not detected so we track paper id that didn't have it.
                    if current_section_type == "intro":
                        keywords_breakdown[current_section_type][
                            "paper_ids_missing"
                        ].append(current_sample["id"])
        keywords_breakdown["dataset_size"] = dataset_size
        keywords_breakdown["ending"][
            "paper_ids_missing"
        ] = self.find_missing_paper_ids_for_ending_section(endings_sample_assignment)
        return keywords_breakdown

    def find_missing_paper_ids_for_ending_section(
        self, endings_assignment_dict: Dict[str, bool]
    ):
        missing_paper_ids = []
        for current_paper_id in endings_assignment_dict:
            paper_id_status = endings_assignment_dict[current_paper_id]
            if not paper_id_status:
                missing_paper_ids.append(current_paper_id)
        return missing_paper_ids

    @staticmethod
    def write_keywords_json_to_disk(
        json_output_path: str, keywords_dict: Dict[str, Dict]
    ):
        return write_dict_to_json_file(json_output_path, keywords_dict)

    @staticmethod
    def get_jsonl_keywords_iterator(jsonl_file_path: str):
        return get_jsonl_reader_iterator(jsonl_file_path)

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
