from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.datasets.math.utils.evaluate_utils import normalize_answer
from src.utils.enums_class import llm_pipeline_type_enum
from src.datasets.dataset_config import dataset_config
from datasets import Dataset
from datasets import load_dataset
import re


class math_hard_dataset(math_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        self.dataset_id = "lighteval/MATH-Hard" # another option with identical format (DigitalLearningGmbH/MATH-lighteval)
        self.dataset = load_dataset(self.dataset_id)
        self.train_dataset = self.dataset["train"]
        self.test_dataset = self.dataset["test"]
        self.instruction = self.prompt_config.get('Math', 'math_hard')        
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
        question = x['problem']
        solution = x['solution']

        question = question + " " + self.instruction
        final_answer = self.final_answer_extraction('', solution, '')

        r1_prefix = [
            {"role": "user",
                "content":question
                },
        ]
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer
                }


# config = dataset_config('Qwen/Qwen2.5-1.5B')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = math_hard_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
