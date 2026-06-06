from src.datasets.dataset_handler import dataset_handler
from src.datasets.dataset_config import dataset_config
from datasets import Dataset
import re
from datasets import Dataset
from datasets import load_dataset

class mmlu_dataset(dataset_handler): 

    def __init__(self, config: dataset_config):
        super().__init__(config)
        
        self.dataset_id = 'lighteval/mmlu'
        self.dataset: Dataset = load_dataset(self.dataset_id, "all")
        self.test_dataset: Dataset = self.dataset['test']
        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']
        
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.test_dataset.add_column('unique_id', range(len(self.test_dataset)))

    
    def final_answer_extraction(self, prompt, solution, target):
        patterns = [
            r'(?i)\bthe\s+correct\s+answer\s+is\s*[:\-\(]?\s*([ABCD])\b',
            r'(?i)\\?oxed\s*\{\s*\**\s*\(?\s*([ABCD])\s*\)?\s*\**\s*\}',            
            r'(?i)answer[\s:*()\[\]\-_=+\n\r\t]*(?:is[\s:*()\[\]\-_=+\n\r\t]*)?(?:option|choice)?[\s:*()\[\]\-_=+\n\r\t]*([ABCD])',
            r'(?i)</think>\s+([ABCD])',
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue
            answer = match.group(1).upper()
            if answer not in ['A', 'B', 'C', 'D']: continue
            return answer
        
        return None

        
    def generate_model_prompt(self, x):
        unique_id: str = x['unique_id']
        question: str = x['question']
        choices: list[str] = x['choices']
        label_index = x['answer']
        subject = x['subject']
        return self.generate_model_prompt_item(unique_id, subject, question, choices, label_index)

    def generate_model_prompt_permutation(self, x, num_choice_permutations: int) -> list[dict]:
        unique_id: str = x['unique_id']
        question: str = x['question']
        choices: list[str] = x['choices']
        label_index = x['answer']
        subject = x['subject']

        prompt_list: list[dict] = [] 
        result: list[tuple[list[str], int]] = self.permute_options(choices, label_index, num_choice_permutations)
        for item in result:
            permute_choices = item[0]
            permute_label_index = item[1]
            prompt_list.append(self.generate_model_prompt_item(unique_id, subject, question, permute_choices, permute_label_index))
            
        return prompt_list

    def generate_model_prompt_item(self, unique_id: str, subject: str, question: str, choices: list[str], label_index: int) -> str:
        labels: list[str] = ['A', 'B', 'C', 'D']
        label: str = labels[label_index]

        prompt = f'Question: {question}\n\n'
        prompt += 'Options:\n'
        for idx, choice in enumerate(choices):
            prompt += f'{labels[idx]}. {choice}\n'
        prompt += 'Reason step by step internally and then choose the most truthful answer.\n'
        prompt += 'Respond with only one letter (A, B, C, or D).\n'
        prompt += 'Answer:'
        
        r1_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": label,
                "problem_id": unique_id
                }


# config: dataset_config = dataset_config('Qwen/Qwen2.5-1.5B')
# d = mmlu_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
