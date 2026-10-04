from src.datasets.confidence.prontoqa.prontoqa_dataset import prontoqa_dataset
from src.datasets.dataset_config import dataset_config


class prontoqa_dataset_qwen2_5_0_5B(prontoqa_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

    def generate_model_prompt_22(self, x):
        unique_id: str = x['id']
        question: str = x['question']
        context: str = x['context']
        choices_text: str = x['options']
        target = x['answer']
        
        prompt = (
            "You are given a logical reasoning problem.\n\n"
            "Use only the facts and rules provided in the context below. "
            "Determine whether the statement in the question is True or False.\n\n"
            "Reason step by step by applying the rules logically. "
            "Do not use any external knowledge or assumptions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{choices_text}\n\n"
            "Think step by step and derive the answer from the given facts and rules.\n"
            "At the end of your response, output the final answer exactly as:\n"
            "Final Answer: A\n"
            "or\n"
            "Final Answer: B"
        )

        return {
            "prompt": prompt,
            "target": target,
            "problem_id": unique_id,
        }


    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        question: str = x['question']
        context: str = x['context']
        choices_text: str = x['options']
        
        prompt = (
            "You are given a logical reasoning problem.\n\n"
            "Use only the facts and rules provided in the context below. "
            "Determine whether the statement in the question is True or False.\n\n"
            "Reason step by step by applying the rules logically. "
            "Do not use any external knowledge or assumptions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{choices_text}\n\n"
            "Think step by step and derive the answer from the given facts and rules.\n"
            "At the end of your response, output the final answer exactly as:\n"
            "Final Answer: A\n"
            "or\n"
            "Final Answer: B"
        )

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


