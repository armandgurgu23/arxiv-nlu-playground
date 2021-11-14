from typing import List, Optional


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

    def remove_paper_contents_by_token_count(
        self,
        paper_contents: List[str],
        token_thresh: int,
        keep_semantic_line: Optional[str] = None,
    ) -> List[str]:
        filtered_contents = []
        for current_line in paper_contents:
            # TODO: Update this line here to not exclude references tokens.
            current_line_tokens = current_line.split(" ")
            if len(current_line_tokens) <= token_thresh:
                continue
            filtered_contents.append(current_line)
        return filtered_contents
