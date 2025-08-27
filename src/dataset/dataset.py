from dataclasses import dataclass
from typing import Union, List
import torch

@dataclass
class DataSample:
    id: int
    query: str
    instruction_positive:str
    instruction_negative:str
    x_positive: str
    x_negative: str
    passage_positive: str
    passage_negative: str
    similarity_pos:float
    similarity_neg:float

class TrainSample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(
        self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0, similarity_score: List[float] = None
    ):
        """
        Creates one TrainSample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label
        self.similarity_score = similarity_score

    def __str__(self):
        return "<TrainSample> label: {}, texts: {}".format(
            str(self.label), "; ".join(self.texts)
        )


class Dataset(torch.utils.data.Dataset):
    def load_data(self, file_path: str = None):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()