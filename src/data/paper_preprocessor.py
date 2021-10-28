from typing import List


class Paper_Preprocessor(object):
    def __init__(self):
        pass

    def remove_newline_characters_from_contents(self, paper_contents: List[str]):
        filtered_contents = []
        for current_line in paper_contents:
            if current_line in {"\n", " \n", "\n "}:
                continue
            filtered_contents.append(current_line)
        return filtered_contents
