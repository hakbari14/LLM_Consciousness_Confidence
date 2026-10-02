from src.datasets.math.gsm8k.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config
import re

class gsm8k_dataset_meta_llama_Llama_3_1_8B_Instruct(gsm8k_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt(self, x):
        question = x['question']
        solution = x['answer']

        instruction = (
            "Solve the problem step by step, showing the calculations concisely.\n"
            "End your response with a separate line in exactly this format:\n"
            "The final answer is: <number>\n"
            "Replace <number> with the final numerical answer.\n"
            "Do not include units, commas, or additional text on this line.\n"
            "Do not write anything after the final answer."
        )

        final_answer = self.final_answer_extraction('', solution, '')
        r1_prefix = [
            {"role": "user",
                "content":f"{question}\n\n{instruction}"
                },
        ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True), 
                "target": final_answer,
                "problem_id": None
                }

    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        question = x['question'] + " " + self.instruction

        partial_cot = partial_cot.strip()

        prefix = [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": partial_cot
            },
        ]
        
        return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)

    def gsm8k_answer_extraction(self, solution: str) -> str :
        _SOLUTION_CLIP_CHARS = 500
        if len(solution) > _SOLUTION_CLIP_CHARS:
            solution = solution[-_SOLUTION_CLIP_CHARS:]

        patterns = [
            r'####.*?([0-9]+(?:[.,][0-9]+)?)',            
            r'(?i)\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}',            
            r'\b(?:The\s+)?final\s+answer\s+is\s*:?\s*([-+]?\d+(?:\.\d+)?)',
            r'(?i)\*[^*]*?(\d+(?:\.\d+)?)[^*]*?\*',            
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, solution, re.DOTALL | re.IGNORECASE))
            if not matches: continue

            last_match = matches[-1]
            x = last_match.group(1)
            try:
                return float(x.strip())
            except ValueError:
                return gsm8k_dataset.extract_number(x)

        return None

