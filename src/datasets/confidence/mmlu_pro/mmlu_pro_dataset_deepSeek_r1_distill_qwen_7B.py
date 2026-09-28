from src.datasets.confidence.mmlu_pro.mmlu_pro_dataset import mmlu_pro_dataset
from src.datasets.dataset_config import dataset_config


class mmlu_pro_dataset_deepSeek_r1_distill_qwen_7B(mmlu_pro_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        unique_id: str = x['unique_id']
        question: str = x['question']
        choices: list[str] = x['options']
        answer_index = x['answer_index']
        answer = x['answer']
        problem_id = x['question_id']
        prompt, label = self.generate_model_prompt_item(unique_id, problem_id, question, choices, answer_index, answer)

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
        
        return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True, enable_thinking=True)


