from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.utils.enums_class import llm_pipeline_type_enum
from src.datasets.dataset_config import dataset_config
from datasets import Dataset
from datasets import load_dataset
import re


class gsm8k_dataset(math_dataset_handler): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)
        self.dataset_id = "openai/gsm8k"
        # self.dataset_id = "/home/hr_akbari/.cache/huggingface/datasets/openai___gsm8k/main"
        self.dataset: Dataset = load_dataset(self.dataset_id, "")
        self.train_dataset: Dataset = self.dataset["train"]
        self.test_dataset: Dataset = self.dataset["test"]
        
        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']
        
        self.instruction = self.prompt_config.get('Math', 'gsm8k')        
        self.force_generate_answer_text = '####'

    def final_answer_extraction(self, prompt: str, solution: str, target: str) -> str :
        _SOLUTION_CLIP_CHARS = 300
        if len(solution) > _SOLUTION_CLIP_CHARS:
            solution = solution[-_SOLUTION_CLIP_CHARS:]

        patterns = [
            r'(?i)####\s*(-?[0-9.,]+)',
            r'(?i)\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}',            
            r'(?i)\*[^*]*?(\d+(?:\.\d+)?)[^*]*?\*',            
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, solution, re.IGNORECASE))
            if not matches: continue

            last_match = matches[-1]
            final_answer = self.extract_number(last_match.group(1))
            return final_answer

        return None

    def generate_model_prompt(self, x):
        question = x['question']
        solution = x['answer']

        question = question + " " + self.instruction
        final_answer = self.final_answer_extraction('', solution, '')
        r1_prefix = [
            {"role": "user",
                "content":question
                },
        ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer,
                "problem_id": None
                }


    def extract_number(self, text: str) -> float:
        chars_to_remove = "\\!@#$%^&*(),/"
        table = str.maketrans('', '', chars_to_remove)

        result = text.translate(table)        
        pattern = r'-?\d+\.\d+|-?\d+'
        match = re.search(pattern, result)

        if match:
            return float(match.group())
        return None
    
# config = dataset_config('Qwen/Qwen2.5-1.5B')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = gsm8k_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
