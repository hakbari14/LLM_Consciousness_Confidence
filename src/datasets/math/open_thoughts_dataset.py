from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
from src.datasets.math.utils.evaluate_utils import extract_boxed_answer, use_math_verify

class open_thoughts_dataset(math_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        # self.dataset_id = "anonym-submit-paper/Orig-R1-Thoughts-correct"  # "open-r1/OpenThoughts-114k-math" 
        self.dataset_id = "/opt/huggingface/hub/datasets--anonym-submit-paper--Orig-R1-Thoughts-correct/snapshots/2bc7ec5a5cfcc6a230af76d24bea4922c0d25e25/math" 
        self.dataset = load_dataset(self.dataset_id)
        correct_dataset = self.dataset.filter(lambda x: self.filter_dataset(x))
        
        dataset = correct_dataset["train"].train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        self.instruction = self.prompt_config.get('Math', 'open_thought')        
        self.force_generate_answer_text = '\\boxed{'
    
    def generate_model_prompt(self, x):
        question = x['problem']
        solution = x['solution']
        problem_id = x['ProblemIdx']

        final_answer = self.final_answer_extraction('', solution, '')
        r1_prefix = [{"role": "system", "content": self.instruction},
                     {"role": "user", "content": question}
                    ]

        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer,
                "problem_id": problem_id
                }

    def final_answer_extraction(self, prompt, completion, target):
        return extract_boxed_answer(completion)

    def filter_dataset(self, x):
        if x['correct'] != True:
            return False
        if self.config.get_max_completion_length() is not None and x['generated_token_count'] > self.config.get_max_completion_length():
            return False
        return True
    
    

# config = dataset_config('/opt/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = open_thoughts_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
