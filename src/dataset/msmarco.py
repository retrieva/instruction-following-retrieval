from typing import List
from dataset import DataSample, TrainSample, Dataset
from datasets import load_dataset
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

class MSMARCO(Dataset):
    """
    Dataset class for MSMARCO data, inheriting from torch.utils.data.Dataset.
    """

    def __init__(
        self,
        dataset_name: str = "msmarco",
        file_path: str = "InF-IR/InF-IR",
    ):
        self.dataset_name: str = dataset_name
        self.data: List = []
        self.file_path: str = file_path
        self.load_data()

    def __len__(self) -> int:
        return len(self.data)

    def load_data(self) -> None:
        """
        Loads MSMARCO dataset from the given file path.
        """
        #logger.info(f"Loading MSMARCO data from {self.file_path} ...")
        datasets = load_dataset(self.file_path)

        for i, dataset in enumerate(datasets["msmarco"]):
            self.data.append(
                DataSample(
                    id = i,
                    query = dataset["query_positive"],
                    # instruction_positive = dataset["instruction_positive"],
                    # instruction_negative = dataset["instruction_negative"],
                    passage_positive = dataset["document_positive"],
                    passage_negative = dataset["hard_negative_document_1"]
                )
            )
    
    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(
            texts=[sample.query, sample.passage_positive, sample.passage_negative], label=1.0
        )