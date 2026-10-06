from src.datasets.math.math_500.math_500_dataset import math_500_dataset
from src.datasets.dataset_config import dataset_config


class math_500_dataset_mistral_7B_Instruct(math_500_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt(self, x):
        question = x["problem"]
        final_answer = x["answer"]
        problem_id = x["unique_id"]

        content = ("Solve the following math problem carefully.\n\n"
                   "Provide a step-by-step derivation.\n"
                   "At the end, put the final answer inside \\boxed{}.\n\n"
                   "Problem:\n")
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
        question = x['problem'] + " " + self.instruction

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


