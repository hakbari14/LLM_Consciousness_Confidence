from src.datasets.math.aime.aime_dataset import aime_dataset
from src.datasets.dataset_config import dataset_config
import re


class aime_dataset_mistral_7B_Instruct(aime_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt(self, x):
        question = x["Question"]
        final_answer = x["Answer"]
        problem_id = x["Problem Number"]

        content = """
                Solve the following math problem carefully.

                Provide a step-by-step derivation.
                At the end, put the final answer inside \\boxed{}.

                Problem:\n
                """
        content = content + question
        r1_prefix = [
            {"role": "user",
                "content": content
                },
        ]

        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True), 
                "target": final_answer,
                "question": question,
                "problem_id": problem_id
                }
    
    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        question = x['question']

        partial_cot = partial_cot.strip()

        prefix = [
            {
                "role": "system", 
                "content": self.instruction
            },
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


