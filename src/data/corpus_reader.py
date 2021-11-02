from os import listdir
from posixpath import join
from typing import Any, Dict, Generator, List, Tuple
from os.path import isfile, isdir
from data.paper_preprocessor import Paper_Preprocessor
import textdistance


class Corpus_Reader(object):
    def __init__(self, **analyzer_cfg: Dict):
        self.cfg = analyzer_cfg
        self.paper_contents_processor = Paper_Preprocessor()
        self.paper_semantic_keywords = {
            "intro": ["introduction", "background", "overview"],
            "conc": ["conclusion", "discussion and results", "discussion"],
            "refer": ["references"],
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

    @property
    def labels(self):
        return self.paper_categories

    def find_all_paper_categories(self, data_path):
        # Yield the contents of one research paper at a time.
        paper_folders = listdir(data_path)
        paper_categories = set()
        for current_paper_folder in paper_folders:
            if current_paper_folder.startswith("."):
                continue
            category_paper = current_paper_folder.split("_")[0]
            if category_paper not in paper_categories:
                paper_categories.add(category_paper)
        return list(paper_categories)

    def __call__(self):
        raw_contents = self.open_contents_from_data_path(self.cfg["data_path"])
        if type(raw_contents) == list:
            filtered_contents = self.paper_contents_processor(raw_contents)
            return filtered_contents
        else:
            return raw_contents

    def find_semantic_section_in_paper(
        self, paper_contents: List[str], section_type: str
    ) -> List[Tuple[str, str, int]]:
        keyword_tuples = []
        for index_paper, current_line in enumerate(paper_contents):
            normalized_line = current_line.strip().lower()
            semantic_section_keywords = self.paper_semantic_keywords[section_type]
            for section_keyword in semantic_section_keywords:
                if (
                    self.keyword_similarity_criterion(normalized_line, section_keyword)
                    >= self.cfg["textdistance_config"]["similarity_threshold"]
                ):
                    keyword_tuples.append((current_line, section_keyword, index_paper))
        assert (
            len(keyword_tuples) != 0
        ), f"This lookup returned no semantic section for section type: {section_type}!"
        # For multiple entries, try out idea of returning the earliest index for that
        # section type.
        return keyword_tuples

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
            yield self.paper_contents_processor(
                raw_paper_contents
            ), current_paper_folder
