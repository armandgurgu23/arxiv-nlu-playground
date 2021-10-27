from typing import Dict


class Corpus_Analyzer(object):
    def __init__(self, **analyzer_cfg: Dict):
        self.cfg = analyzer_cfg

    def __call__(self, **analyzer_params: Dict):
        print(analyzer_params)
        raise NotImplementedError("Is this working!?")
