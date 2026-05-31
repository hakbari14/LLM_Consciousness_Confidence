from src.datasets.math.countdown_dataset_handler import countdown_dataset_handler
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
import re



class countdown_dataset(countdown_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        # self.dataset_id = "/home/hr_akbari/.cache/huggingface/datasets/datasets--Jiayi-Pan--Countdown-Tasks-3to4"
        self.dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
        self.dataset = load_dataset(self.dataset_id)["train"].train_test_split(test_size=0.001, seed=42)
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset["test"]

        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']

        self.instruction = self.prompt_config.get('Math', 'countdown') 

    def final_answer_extraction(self, prompt, solution, target):
        patterns = [
            r'(?i)<answer>(.*?)</answer>',
            r'(?i)\\?oxed\s*\{(.*?)\}',            
        ]

        for pattern in patterns:
            pattern_regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            matches = pattern_regex.findall(solution)
            if not matches or len(matches) == 0: continue
            
            equation = matches[-1].strip()
            equation = equation.replace('\\times', '*')
            equation = equation.replace('\\div', '/')
            if "=" in equation:
                equation = equation.split("=", 1)[0]
            
            return equation
    
        return None

    def generate_model_prompt(self, x):
        numbers = x['nums']
        target = x['target']
        problem_id = None
        
        r1_prefix = [{ 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
        },]

        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": target, 
                "nums": numbers,
                "problem_id": problem_id
                }


# config = dataset_config('Qwen/Qwen2.5-1.5B')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = countdown_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
