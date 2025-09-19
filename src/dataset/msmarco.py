from typing import List
from .dataset import DataSample, TrainSample, Dataset
from datasets import load_dataset
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
import json 
from datasets import load_from_disk

class MSMARCO(Dataset):
    """
    Dataset class for MSMARCO data, inheriting from torch.utils.data.Dataset.
    """

    def __init__(
        self,
        mode: str = "train",
    ):
        self.data: List = []
        self.mode: str = mode
        self.load_data()

    def __len__(self) -> int:
        return len(self.data)

    def load_data(self) -> None:
        datasets = load_from_disk("./dataset/msmarco_with_scores")[self.mode]

        for i, dataset in enumerate(datasets):
            self.data.append(
                DataSample(
                    id=i,
                    query=dataset["query_positive"],
                    instruction=dataset["instruction_positive"],
                    x=f'{dataset["instruction_positive"]}\n{dataset["query_positive"]}',
                    passage_positive=dataset["document_positive"],
                    passage_negative=dataset["hard_negative_document_1"],
                )
            )
    
    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(
            texts=[sample.query, sample.instruction, sample.x, sample.passage_positive, sample.passage_negative], 
            label=1.0,
        )
    
