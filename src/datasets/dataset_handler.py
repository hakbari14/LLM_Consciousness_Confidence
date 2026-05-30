from src.utils.enums_class import dataset_element_type_enum, llm_pipeline_type_enum
from src.datasets.dataset_config import dataset_config
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from datasets import Dataset
import configparser

class dataset_handler(ABC): 

    def __init__(self, config : dataset_config):
        self.config: dataset_config = config
        self.config.validate()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get_model_name())
        self.prompt_config = configparser.ConfigParser()
        self.prompt_config.read('src/datasets/dataset_prompt.cfg')

    def preprocess_dataset(self) -> tuple[Dataset, Dataset]:
        train_dataset = self.train_dataset.map(lambda x: self.generate_model_prompt(x))
        train_dataset = train_dataset.filter(lambda x: self.filter_by_required_criteria(x, dataset_element_type_enum.TRAIN))

        test_dataset = self.test_dataset.map(lambda x: self.generate_model_prompt(x))
        test_dataset = test_dataset.filter(lambda x: self.filter_by_required_criteria(x, dataset_element_type_enum.EVAL))
       
        train_dataset = train_dataset.add_column("split", [dataset_element_type_enum.TRAIN] * len(train_dataset))
        train_dataset = train_dataset.add_column("sample_id", list(range(len(train_dataset))))

        test_dataset_size = len(test_dataset)
        if self.config.get_max_test_dataset_size() is not None: 
            test_dataset_size = self.config.get_max_test_dataset_size()

        eval_dataset = test_dataset.select(range(test_dataset_size))
        eval_dataset = eval_dataset.add_column("split", [dataset_element_type_enum.EVAL] * len(eval_dataset))
        eval_dataset = eval_dataset.add_column("sample_id", list(range(len(eval_dataset))))

        return train_dataset, eval_dataset

    def filter_by_required_criteria(self, x: dict, dataset_type: dataset_element_type_enum) -> bool:
        return True

    def extract_and_verify_final_answer(self, prompt: str, completion: str, target: str) -> tuple[str, bool, str]:
        final_answer = self.final_answer_extraction(prompt, completion, target)
        if final_answer is None:
            return final_answer, False, final_answer

        target_answer_equal, comapred_final_answer = self.verify_final_answer(target, final_answer)
        return final_answer, target_answer_equal, comapred_final_answer
        
    def verify_final_answer(self, target: str, final_answer: str) -> tuple[bool, str]:
        return final_answer == target, final_answer

    def get_config(self) -> dataset_config: 
        return self.config

    def set_config(self, value: dataset_config) -> None: 
        self.config = value
    
    @abstractmethod
    def final_answer_extraction(self, prompt, completion, target):
        pass

    @abstractmethod
    def generate_model_prompt(self, x):
        pass



