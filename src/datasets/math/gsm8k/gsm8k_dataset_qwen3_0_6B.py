from src.datasets.math.gsm8k.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config


class gsm8k_dataset_qwen3_0_6B(gsm8k_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt(self, x):
        question = x['question']
        solution = x['answer']

        question = question + " " + self.instruction
        final_answer = self.final_answer_extraction('', solution, '')
        r1_prefix = [
            {"role": "user",
                "content":question
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
        partial_cot = partial_cot.replace("<think>", "")
        partial_cot = partial_cot.replace("</think>", "")

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
        
        return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True, enable_thinking=True)


