from datasets import Dataset
from src.datasets.dataset_handler import dataset_handler
from src.datasets.dataset_config import dataset_config
import re
import pandas as pd 

class metacognitive_dataset(dataset_handler): 

    def __init__(self, config: dataset_config):
        super().__init__(config)

        df = pd.read_csv('data/metacognitive_dataset.csv')
        if config.get_ratio_test_dataset_size() is not None: 
            df = df.sample(frac=config.get_ratio_test_dataset_size(), random_state=42)
            
        self.dataset = Dataset.from_pandas(df)
        self.train_dataset = Dataset.from_dict({"correct_prompt": [], "correct_target": [], "incorrect_prompt": [], "incorrect_target": [], "problem_id" : []})
        self.test_dataset = self.dataset
        

    def final_answer_extraction(self, prompt, solution, target):
        patterns = [
            r"Answer\s*:\s*[*\s]*(yes|no)[*\s]*",
            r"boxed\{[^}]*\b(yes|no)\b[^}]*\}",
            r"</think>\s*[*\s]*(yes|no)[*\s]*",
            r"\b(yes|no)\b"
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue
            answer = match.group(1).lower().capitalize()
            if answer not in ['Yes', 'No']: continue
            return answer
        
        return None

        
    def generate_model_prompt(self, x):
        unique_id: str = x['unique_id']
        question: str = x['question']
        correct_answer = x['correct_answer']
        incorrect_answer = x['incorrect_answer']
        
        correct_prompt = f'Question: {question}\n\n'
        correct_prompt += f'Proposed Answer: {correct_answer}\n'
        correct_prompt += 'Evaluate whether the Proposed Answer is correct. Do not generate a new answer.\n'
        correct_prompt += 'Answer: Yes/No'
        correct_target = 'Yes'

        incorrect_prompt = f'Question: {question}\n\n'
        incorrect_prompt += f'Proposed Answer: {incorrect_answer}\n'
        incorrect_prompt += 'Evaluate whether the Proposed Answer is correct. Do not generate a new answer.\n'
        incorrect_prompt += 'Answer: Yes/No'
        incorrect_target = 'No'
        
        correct_prefix = [
            {"role": "user",
                "content": correct_prompt
                },
        ]

        incorrect_prefix = [
            {"role": "user",
                "content": incorrect_prompt
                },
        ]
        
        return {
                "question": question, 
                "correct_answer": correct_answer, 
                "incorrect_answer": incorrect_answer, 
                "correct_prompt": self.tokenizer.apply_chat_template(correct_prefix, tokenize=False, continue_final_message=True), 
                "correct_target": correct_target,
                "incorrect_prompt": self.tokenizer.apply_chat_template(incorrect_prefix, tokenize=False, continue_final_message=True), 
                "incorrect_target": incorrect_target,
                "problem_id": unique_id
                }




# config: dataset_config = dataset_config('Qwen/Qwen2.5-1.5B')
# d = metacognitive_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
