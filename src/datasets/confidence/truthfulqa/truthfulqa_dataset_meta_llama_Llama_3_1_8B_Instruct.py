from src.datasets.confidence.truthfulqa.truthfulqa_dataset import truthfulqa_dataset
from src.datasets.dataset_config import dataset_config


class truthfulqa_dataset_meta_llama_Llama_3_1_8B_Instruct(truthfulqa_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        unique_id: str = x['unique_id']
        question: str = x['question']
        choices: list[str] = x['choices']
        label_index = x['label']
        
        prompt, label = self.generate_model_prompt_item(unique_id, question, choices, label_index)

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
        
        return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)


