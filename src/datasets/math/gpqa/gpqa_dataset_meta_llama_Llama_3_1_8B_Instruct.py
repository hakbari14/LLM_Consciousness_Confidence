from src.datasets.math.gpqa.gpqa_dataset import gpqa_dataset
from src.datasets.dataset_config import dataset_config
import re


class gpqa_dataset_meta_llama_Llama_3_1_8B_Instruct(gpqa_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        question = x["problem"]

        q = question.split("Choices:")[0]
        choices_part = question.split("Choices:")[1]
        pattern = r'^([A-D])\.\s*(.+)$'
        matches = re.findall(pattern, choices_part, re.MULTILINE)
        
        prompt = f"What is the correct answer to this question: {q}"
        prompt += f"\n\nChoices:\n(A) {matches[0][1]}\n(B) {matches[1][1]}\n(C) {matches[2][1]}\n(D) {matches[3][1]}"
        prompt += f"\n\nFormat your response as follows: \"The correct answer is (insert answer here)\""

        partial_cot = partial_cot.strip()
        
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

        
        return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)


