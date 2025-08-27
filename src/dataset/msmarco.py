from typing import List
from .dataset import DataSample, TrainSample, Dataset
from datasets import load_dataset
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
from .similarity import UAESIMILARITY
import json 

class MSMARCO(Dataset):
    """
    Dataset class for MSMARCO data, inheriting from torch.utils.data.Dataset.
    """

    def __init__(
        self,
        dataset_name: str = "msmarco",
        file_path: str = "InF-IR/InF-IR",
        similarity_file_path: str = "./dataset/similarity/similarity.json"
    ):
        self.dataset_name: str = dataset_name
        self.data: List = []
        self.similarity_data: List = []
        self.file_path: str = file_path
        self.similarity_file_path: str = similarity_file_path
        self.load_data()

    def __len__(self) -> int:
        return len(self.data)

    def load_data(self) -> None:
        """
        Loads MSMARCO dataset from the given file path.
        """
        #logger.info(f"Loading MSMARCO data from {self.file_path} ...")
        datasets = load_dataset(self.file_path)
        with open(self.similarity_file_path, "r") as f:
            self.similarity_data = json.load(f)

        for i, dataset in enumerate(datasets["msmarco"]):
            self.data.append(
                DataSample(
                    id=i,
                    query=dataset["query_positive"],
                    instruction_positive=dataset["instruction_positive"],
                    instruction_negative=dataset["instruction_negative"],
                    x_positive=f'{dataset["instruction_positive"]}{dataset["query_positive"]}',
                    x_negative=f'{dataset["instruction_negative"]}{dataset["query_positive"]}',
                    passage_positive=dataset["document_positive"],
                    passage_negative=dataset["hard_negative_document_1"],
                    **{score_key: self.similarity_data[i][score_key] 
                    for score_key in [
                            "q_and_p_pos_score", "q_and_p_neg_score",
                            "inst_pos_and_p_pos_score", "inst_pos_and_p_neg_score",
                            "inst_neg_and_p_pos_score", "inst_neg_and_p_neg_score",
                            "x_pos_and_p_pos_score", "x_pos_and_p_neg_score",
                            "x_neg_and_p_pos_score", "x_neg_and_p_neg_score"
                        ]}
                )
            )
    
    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(
            texts=[sample.query, sample.instruction_positive, sample.instruction_negative, sample.x_positive, sample.x_negative, sample.passage_positive, sample.passage_negative], 
            label=1.0,
            similarity_score=[sample.q_and_p_pos_score, sample.q_and_p_neg_score, sample.inst_pos_and_p_pos_score, sample.inst_pos_and_p_neg_score, sample.inst_neg_and_p_pos_score, sample.inst_neg_and_p_neg_score, sample.x_pos_and_p_pos_score, sample.x_pos_and_p_neg_score, sample.x_neg_and_p_pos_score, sample.x_neg_and_p_neg_score]
        )