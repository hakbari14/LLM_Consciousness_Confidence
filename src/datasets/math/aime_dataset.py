from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
import re


class aime_dataset(math_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        self.dataset_id = "di-zhang-fdu/AIME_1983_2024" 
        # self.dataset_id = '/home/hr_akbari/.cache/huggingface/datasets/di-zhang-fdu___aime_1983_2024/default/0.0.0/3e2cc86390666c5c756622afc0eeb9e6194496bc'
        self.dataset = load_dataset(self.dataset_id)
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset['train']

        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']

        self.instruction = self.prompt_config.get('Math', 'aime')
        self.force_generate_answer_text = '\\boxed{'

    def generate_model_prompt(self, x):
        question = x["Question"]
        final_answer = x["Answer"]
        problem_id = x["Problem Number"]

        r1_prefix = [{"role": "system", "content": self.instruction},
                     {"role": "user", "content": question}
                    ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer,
                "problem_id": problem_id
                }

    def final_answer_extraction(self, prompt, solution_str, target):
        _SOLUTION_CLIP_CHARS = 600
        if len(solution_str) > _SOLUTION_CLIP_CHARS:
            solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

        patterns = [
            r'(?i)\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}',            
            r'(?i)Answer[^\d]*(\d+)\**',            
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, solution_str, re.IGNORECASE | re.DOTALL))
            if not matches: continue

            last_match = matches[-1]
            final_answer = last_match.group(1)
            return final_answer
        
        return None
    


# config = dataset_config('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = aime_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
