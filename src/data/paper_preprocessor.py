from typing import List


class Paper_Preprocessor(object):
    def __init__(self):
        self.paper_processing_methods = [self.remove_newline_characters_from_contents]

    def __call__(self, paper_contents: List[str]) -> List[str]:
        for current_processor in self.paper_processing_methods:
            paper_contents = current_processor(paper_contents)
        return paper_contents

    def remove_newline_characters_from_contents(
        self, paper_contents: List[str]
    ) -> List[str]:
        filtered_contents = []
        for current_line in paper_contents:
            if current_line in {"\n", " \n", "\n "}:
                continue
            filtered_contents.append(current_line)
        return filtered_contents
