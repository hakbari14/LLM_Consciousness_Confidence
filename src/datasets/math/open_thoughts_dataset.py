from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
from src.datasets.math.utils.evaluate_utils import extract_last_boxed, use_math_verify
import re

class open_thoughts_dataset(math_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        self.dataset_id = "anonym-submit-paper/Orig-R1-Thoughts-correct"
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
                "problem_id": problem_id,
                "question": question,
                }

    def generate_model_prompt_confidence(self, x):
        question = x['problem']
        solution = x['solution']
        problem_id = x['ProblemIdx']

        final_answer = self.final_answer_extraction('', solution, '')
        
        prompt = f"Solve the following math problem step by step.\n\n"
        prompt += f"Question:\n{question}\n\n"
        prompt += "After obtaining your final answer, carefully review your own reasoning and estimate the probability that your Final Answer is actually correct.\n\n"
        
        prompt += (
            "Respond in exactly the following format:\n\n"
            "Solution:\n<your step-by-step solution>\n\n"
            "Final Answer:\n<answer>\n\n"
            "Confidence:\n<integer between 0 and 100>"
        )
        
        r1_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]

        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer,
                "problem_id": problem_id,
                "question": question,
                }

    def generate_another_prompt_confidence(self, question, answer) -> str:
        prompt = f'[Question]: {question}\n\n'
        prompt += f'[Answer]: {answer}\n\n'
        prompt += (
            f'Task: Evaluate how likely the provided Answer is correct based only on the Question and Answer.\n\n'
            'You must follow this output protocol exactly:\n\n'

            'PHASE 1 — REASONING\n'
            'Start your response immediately with a detailed step-by-step analysis of whether the Answer is correct.\n'

            'PHASE 2 — FINAL CONFIDENCE\n'
            'Only after completing the reasoning, output exactly one final line:\n'
            'Confidence:<integer between 0 and 100>\n\n'

            'Begin now with PHASE 1 reasoning.'
        )
        return prompt

    def final_answer_extraction(self, prompt, completion, target):
        return extract_last_boxed(completion)

    def final_answer_confidence_extraction(self, prompt, completion, target) -> tuple[str, str]:
        final_answer = self.extract_final_answer(completion)
        confidence = self.extract_confidence(completion)
        return final_answer, confidence
    
    def extract_final_answer(self, completion):
        patterns = [
            r"Final\s*Answer\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE)
            if not match: continue
            return match.group(1)

        answer_to_confidence = self.extract_between_final_answer_to_confidence(completion)
        final_answer = extract_last_boxed(answer_to_confidence)
        if final_answer is not None:
            return final_answer
        
        return answer_to_confidence
    
    def extract_confidence(self, solution: str) -> float:
        patterns = [
            r"Confidence\s*:\s*<\s*(\d{1,3})\s*>",
            r"\[\s*Confidence\s*\]\s*:\s*(\d+)",
            r"Confidence[\s*]*:[\s*]*(\d+(?:\.\d+)?)",
            r"Confidence\s*Score[\s*\n:]*([0-9]+(?:\.[0-9]+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue
            answer = float(match.group(1))
            if answer > 100 or answer < 0: continue
            return answer
        
        return None

    def extract_another_confidence(self, solution: str) -> float:
        return self.extract_confidence(solution)

    def extract_between_final_answer_to_confidence(self, text: str):
        final_pattern = re.compile(r'final\s*answer\s*:?', re.IGNORECASE)
        matches = list(final_pattern.finditer(text))
        if not matches:
            return None

        last_final = matches[-1]

        confidence_pattern = re.compile(r'confidence\s*:?', re.IGNORECASE)
        confidence_match = confidence_pattern.search(text, last_final.end())

        if not confidence_match:
            return text[last_final.end():].strip()

        return text[last_final.end():confidence_match.start()].strip()

    def filter_dataset(self, x):
        if x['correct'] != True:
            return False
        if self.config.get_max_completion_length() is not None and x['generated_token_count'] > self.config.get_max_completion_length():
            return False
        return True
    
    

# config = dataset_config('Qwen/Qwen3-8B')
# d = open_thoughts_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset_with_confidence()
# print(len(train_dataset))
# print(len(test_dataset))
