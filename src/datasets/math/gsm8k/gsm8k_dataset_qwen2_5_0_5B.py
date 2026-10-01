from src.datasets.math.gsm8k.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config


class gsm8k_dataset_qwen2_5_0_5B(gsm8k_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt(self, x):
        question = x["question"]
        solution = x["answer"]

        question = question + " " + self.instruction
        final_answer = self.final_answer_extraction("", solution, "")
        prompt = question

        return {
            "prompt": prompt,
            "target": final_answer,
            "problem_id": None,
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


