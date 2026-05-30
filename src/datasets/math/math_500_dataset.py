from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.datasets.dataset_config import dataset_config
from src.datasets.math.utils.evaluate_utils import normalize_answer
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
import re


class math_500_dataset(math_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        #self.dataset_id = "HuggingFaceH4/MATH-500"
        self.dataset_id = "/home/hr_akbari/.cache/huggingface/datasets/datasets--HuggingFaceH4--MATH-500" 
        self.dataset = load_dataset(self.dataset_id)
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset['test']

        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']
        
        self.instruction = self.prompt_config.get('Math', 'math_500')        
        self.force_generate_answer_text = '\\boxed{'

    
    def final_answer_extraction(self, prompt, solution, target):
        pattern = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
    
        matches = pattern.findall(solution)
        if not matches:
            return None

        final = matches[-1].strip()
        final = re.sub(r"\s+", " ", final).strip()
        return normalize_answer(final)

        
    def generate_model_prompt(self, x):
        question = x["problem"]
        final_answer = x["answer"]
        problem_id = x["unique_id"]

        question = question + " " + self.instruction
        r1_prefix = [
            {"role": "user",
                "content": question
                },
        ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer,
                "problem_id": problem_id
                }



# config = dataset_config('Qwen/Qwen2.5-1.5B')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = math_500_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
