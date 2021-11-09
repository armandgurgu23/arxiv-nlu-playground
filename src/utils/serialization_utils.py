from json import loads, dump
from typing import Any, Dict


def get_jsonl_reader_iterator(jsonl_file_path: str):
    with open(jsonl_file_path, "r") as file_object:
        for current_paper_summary in file_object:
            yield loads(current_paper_summary)


def write_dict_to_json_file(json_output_path: str, input_dict: Dict[str, Any]):
    with open(json_output_path, "w") as file_object:
        dump(input_dict, file_object)
    return
