from os import listdir
from typing import List


def get_all_classes_for_text_classification(data_path: str) -> List[str]:
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
