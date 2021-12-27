class TextClassificationReader:
    def __init__(self) -> None:
        pass

    def __iter__(self):
        raise NotImplementedError(
            "Implement logic to create minibatches for a dataset reader."
        )


class SklearnTextClassificationReader(TextClassificationReader):
    def __init__(self) -> None:
        pass

    def __iter__(self):
        pass
