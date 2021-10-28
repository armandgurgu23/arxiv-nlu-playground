from typing import Dict, List
from os.path import isfile, isdir
from data.paper_preprocessor import Paper_Preprocessor


class Corpus_Analyzer(object):
    def __init__(self, **analyzer_cfg: Dict):
        self.cfg = analyzer_cfg
        self.paper_contents_processor = Paper_Preprocessor()

    def __call__(self, **analyzer_params: Dict):
        raw_contents = self.open_contents_from_data_path(analyzer_params["data_path"])
        filtered_contents = (
            self.paper_contents_processor.remove_newline_characters_from_contents(
                raw_contents
            )
        )
        return filtered_contents

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

    def get_data_directory_generator(self, data_path: str):
        raise NotImplementedError("To be added later!")
