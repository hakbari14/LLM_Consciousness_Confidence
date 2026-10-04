from src.datasets.dataset_handler import dataset_handler
from src.datasets.dataset_config import dataset_config
from datasets import Dataset
from datasets import load_dataset
import re

class prontoqa_dataset(dataset_handler): 

    def __init__(self, config: dataset_config):
        super().__init__(config)
        
        self.dataset_id = 'renma/ProntoQA'
        self.dataset: Dataset = load_dataset(self.dataset_id)
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset['validation']

        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']

    
    def final_answer_extraction(self, prompt, solution, target):
        patterns = [
            r'(?i)final\s+answer\s*:?\s*([ab])\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue
            answer = match.group(1).upper()
            if answer not in ['A', 'B']: continue
            return answer
        
        return None

        
    def generate_model_prompt(self, x):
        unique_id: str = x['id']
        question: str = x['question']
        context: str = x['context']
        choices_text: str = x['options']
        target = x['answer']
        
        prompt = (
            "You are given a logical reasoning problem.\n\n"
            "Use only the facts and rules provided in the context below. "
            "Determine whether the statement in the question is True or False.\n\n"
            "Reason step by step by applying the rules logically. "
            "Do not use any external knowledge or assumptions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{choices_text}\n\n"
            "Think step by step and derive the answer from the given facts and rules.\n"
            "At the end of your response, output the final answer exactly as:\n"
            "Final Answer: A\n"
            "or\n"
            "Final Answer: B"
        )

        r1_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True), 
                "target": target,
                "question": question,
                "problem_id": unique_id
                }

    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        question: str = x['question']
        context: str = x['context']
        choices_text: str = x['options']
        
        prompt = (
            "You are given a logical reasoning problem.\n\n"
            "Use only the facts and rules provided in the context below. "
            "Determine whether the statement in the question is True or False.\n\n"
            "Reason step by step by applying the rules logically. "
            "Do not use any external knowledge or assumptions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{choices_text}\n\n"
            "Think step by step and derive the answer from the given facts and rules.\n"
            "At the end of your response, output the final answer exactly as:\n"
            "Final Answer: A\n"
            "or\n"
            "Final Answer: B"
        )

        partial_cot = partial_cot.strip()
        partial_cot = partial_cot.replace("<think>", "")
        partial_cot = partial_cot.replace("</think>", "")

        prefix = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": partial_cot
            },
        ]

        if self.get_enable_thinking():
            return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True, enable_thinking=True)
        else: 
            return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)


    def generate_wrong_answer(self, latex_expr:str) -> str:
        return None

    def final_answer_confidence_extraction(self, prompt, completion, target):
        return None

    def generate_model_prompt_confidence(self, x):
        return None

    def generate_another_prompt_confidence(self, question: str, answer: str) -> str:
        return None

    def extract_another_confidence(self, solution: str) -> float:
        return None

