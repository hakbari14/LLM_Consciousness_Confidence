from src.datasets.math.math_500.math_500_dataset import math_500_dataset
from src.datasets.dataset_config import dataset_config


class math_500_dataset_deepSeek_r1_distill_qwen_7B(math_500_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        question = x['problem'] + " " + self.instruction

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


