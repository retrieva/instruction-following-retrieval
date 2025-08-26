from typing import List
from .dataset import DataSample, TrainSample, Dataset
from datasets import load_dataset
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
import json

class UAESIMILARITY(Dataset):
    def __init__(
        self,
        file_path: str = "/home/seiji_sugiyama/works/instructir/dataset/similarity/similarity.json",
    ):
        self.data: List = []
        self.file_path: str = file_path
        self.load_data()

    def __len__(self) -> int:
        return len(self.data)

    def load_data(self) -> None:
        """
        Loads UAESIMILARITY dataset from the given file path.
        """
        with open(self.file_path, "r") as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample