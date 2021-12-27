from os import listdir
from posixpath import join
from typing import Any, Dict, Generator, List, Tuple, Union
from os.path import isfile, isdir
from data.paper_preprocessor import Paper_Preprocessor
from utils.dataset_utils import get_all_classes_for_text_classification
import textdistance
import re


ReferenceInfo = Union[List[Tuple[str, str, int]], List[Tuple[str, int]]]


class Corpus_Reader(object):
    def __init__(self, **analyzer_cfg: Dict):
        self.cfg = analyzer_cfg
        if self.cfg["apply_paper_processor_in_reader"]:
            self.paper_contents_processor = Paper_Preprocessor()
        self.paper_semantic_keywords = {
            "intro": [
                "introduction",
                "background",
                "overview",
                "main text",
                "Summary of the presentations",
            ],
            "conc": ["conclusion", "discussion and results", "discussion"],
            "refer": ["references", "literature", "bibliography"],
            "acknow": ["acknowledgements"],
        }
        self.keyword_similarity_criterion = (
            textdistance.levenshtein.normalized_similarity
        )
        if isdir(self.cfg["data_path"]):
            # Precompute all the corpus labels.
            self.paper_categories = self.find_all_paper_categories(
                self.cfg["data_path"]
            )
        else:
            # Input is a sample paper. Simple parsing based on dataset schema.
            self.paper_categories = self.cfg["data_path"].split("/")[-2].split("_")[0]

        self.first_ref_year_pattern = re.compile("\(([1-9][0-9]{3})\)")
        self.multi_authors_keyword = "et al"
        self.second_char_non_digit_re = re.compile("1\D")
        self.section_numbering_regex = re.compile("[0-9]")

    @property
    def paper_semantic_keywords_dict(self):
        return self.paper_semantic_keywords

    @property
    def labels(self):
        return self.paper_categories

    def find_all_paper_categories(self, data_path: str):
        return get_all_classes_for_text_classification(data_path)

    def __call__(self):
        raw_contents = self.open_contents_from_data_path(self.cfg["data_path"])
        if type(raw_contents) == list:
            if hasattr(self, "paper_contents_processor"):
                filtered_contents = self.paper_contents_processor(raw_contents)
                return filtered_contents
            else:
                return raw_contents
        else:
            return raw_contents

    def find_beginning_of_references_in_paper(
        self, current_paper_contents: List[str], semantic_section_type: str
    ) -> ReferenceInfo:
        try:
            beginning_of_ref = self.find_semantic_section_in_paper(
                current_paper_contents, semantic_section_type, True
            )
        except AssertionError:
            beginning_of_ref = self.find_first_reference_in_paper(
                current_paper_contents
            )
        return beginning_of_ref

    def find_first_reference_in_paper(
        self, paper_contents: List[str]
    ) -> List[Tuple[str, int]]:
        first_ref_matches = []
        for line_index, current_line in enumerate(paper_contents):
            if current_line.startswith("[1]") and (
                line_index / len(paper_contents)
                >= self.cfg["textdistance_config"][
                    "selection_threshold_paper_len_pointer"
                ]
            ):
                return [(current_line, line_index)]
            if current_line.startswith("1") and (
                line_index / len(paper_contents)
                >= self.cfg["textdistance_config"][
                    "selection_threshold_paper_len_pointer"
                ]
            ):
                first_ref_matches.append((current_line, line_index))
        if len(first_ref_matches) == 1:
            return first_ref_matches
        else:
            filtered_matches = self.select_first_reference_from_multiple_matches(
                first_ref_matches
            )
            if filtered_matches:
                return filtered_matches
            else:
                return self.select_first_reference_simpler_criterion(first_ref_matches)

    def select_first_reference_simpler_criterion(self, matches: List[Tuple[str, int]]):
        candidate_matches = []
        for current_candidate, current_index in matches:
            if (
                self.second_char_non_digit_re.search(current_candidate)
                and self.second_char_non_digit_re.search(current_candidate).span()[0]
                == 0
            ):
                candidate_matches.append((current_candidate, current_index))
        return candidate_matches

    def select_first_reference_from_multiple_matches(
        self, matches: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        candidate_matches = []
        for current_candidate, current_index in matches:
            if self.first_ref_year_pattern.search(current_candidate) and (
                self.second_char_non_digit_re.search(current_candidate)
                and self.second_char_non_digit_re.search(current_candidate).span()[0]
                == 0
            ):
                candidate_matches.append((current_candidate, current_index))
        return candidate_matches

    def find_semantic_section_in_paper(
        self,
        paper_contents: List[str],
        section_type: str,
        attempt_resolution: bool = False,
    ) -> List[Tuple[str, str, int]]:
        keyword_tuples = []
        for index_paper, current_line in enumerate(paper_contents):
            normalized_line = current_line.strip().lower()
            normalized_line = self.section_numbering_regex.sub("", normalized_line)
            semantic_section_keywords = self.paper_semantic_keywords[section_type]
            for section_keyword in semantic_section_keywords:
                if (
                    self.keyword_similarity_criterion(normalized_line, section_keyword)
                    >= self.cfg["textdistance_config"]["similarity_threshold"]
                ):
                    if (index_paper / len(paper_contents)) < self.cfg[
                        "textdistance_config"
                    ]["selection_threshold_max_bound_pointer"]:
                        keyword_tuples.append(
                            (current_line, section_keyword, index_paper)
                        )
        assert (
            len(keyword_tuples) != 0
        ), f"This lookup returned no semantic section for section type: {section_type}!"
        # For multiple entries, try out idea of returning the earliest index for that
        # section type.
        if attempt_resolution and len(keyword_tuples) > 1:
            first_attempt_tuples = self.select_semantic_section_tuple_from_choices(
                keyword_tuples,
                len(paper_contents),
                self.cfg["textdistance_config"][
                    "selection_threshold_paper_len_pointer"
                ],
            )
            if len(first_attempt_tuples) == 1:
                return first_attempt_tuples
            else:
                return self.select_semantic_section_tuple_from_choices(
                    keyword_tuples,
                    len(paper_contents),
                    self.cfg["textdistance_config"][
                        "selection_threshold_paper_len_secondary"
                    ],
                )
        else:
            return keyword_tuples

    def select_semantic_section_tuple_from_choices(
        self,
        semantic_section_tuples: List[Tuple[str, str, int]],
        paper_num_lines: int,
        selection_threshold: float,
    ) -> List[Tuple[str, str, int]]:
        candidate_ratios = []
        candidate_indexes = []
        for current_index, current_candidate in enumerate(semantic_section_tuples):
            section_line_nr = current_candidate[-1]
            ratio_pointer = section_line_nr / paper_num_lines
            if ratio_pointer >= selection_threshold:
                candidate_ratios.append(ratio_pointer)
                candidate_indexes.append(current_index)
        if candidate_ratios:
            index_min = candidate_ratios.index(min(candidate_ratios))
            return [semantic_section_tuples[candidate_indexes[index_min]]]
        else:
            # Greedy selection has failed, returning all the choices for
            # further inspection.
            return semantic_section_tuples

    def open_contents_from_data_path(self, data_path: str):
        if isfile(data_path):
            return self.open_file_contents(data_path)
        elif isdir(data_path):
            return self.get_data_directory_generator(data_path)
        else:
            raise RuntimeError(
                f"Attempting to handle data_path which is not file or directory! It is {data_path}!"
            )

    def open_file_contents(
        self, data_path: str, file_encoding: str = "utf-8"
    ) -> List[str]:
        raw_file_contents = []
        with open(data_path, "r", encoding=file_encoding) as file_object:
            for current_line in file_object:
                raw_file_contents.append(current_line)
        return raw_file_contents

    def get_data_directory_generator(
        self, data_path: str
    ) -> Generator[Tuple[List[str], str], Any, Any]:
        # Yield the contents of one research paper at a time.
        paper_folders = listdir(data_path)
        for current_paper_folder in paper_folders:
            if current_paper_folder.startswith("."):
                continue
            current_paper_filepath = join(data_path, current_paper_folder, "paper.txt")
            raw_paper_contents = self.open_file_contents(current_paper_filepath)
            if hasattr(self, "paper_contents_processor"):
                yield self.paper_contents_processor(
                    raw_paper_contents
                ), current_paper_folder
            else:
                yield raw_paper_contents, current_paper_folder
