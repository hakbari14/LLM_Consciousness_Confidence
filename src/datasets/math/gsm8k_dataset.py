from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.utils.enums_class import llm_pipeline_type_enum
from src.datasets.dataset_config import dataset_config
from datasets import Dataset
from datasets import load_dataset
import re


class gsm8k_dataset(math_dataset_handler): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)
        self.dataset_id = "openai/gsm8k"
        self.dataset: Dataset = load_dataset(self.dataset_id, "main")
        self.train_dataset: Dataset = self.dataset["train"]
        self.test_dataset: Dataset = self.dataset["test"]
        
        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']
        
        self.instruction = self.prompt_config.get('Math', 'gsm8k')        
        self.force_generate_answer_text = '####'

    def final_answer_extraction(self, prompt: str, solution: str, target: str) -> str :
        return gsm8k_dataset.gsm8k_answer_extraction(solution)

    def get_min_chain_of_thought_length(self) -> int:
        return 200

    def chain_of_thought_extraction(self, question: str, solution: str) -> str :
        chain_of_thought = solution
        pos = chain_of_thought.find(question)
        if pos != -1:
            chain_of_thought = chain_of_thought[pos + len(question):]

        # Locate the region holding the reasoning. A substantive <think> block IS
        # the reasoning, and everything after </think> is the answer presentation.
        # When that block is trivial or absent the model reasoned in the
        # presentation instead, so drop the block along with the prompt
        # instruction echoed before it.
        think_end = chain_of_thought.find('</think>')
        if think_end >= self.get_min_chain_of_thought_length():
            chain_of_thought = chain_of_thought[:think_end]
        elif think_end != -1:
            chain_of_thought = chain_of_thought[think_end + len('</think>'):]

        # Then cut at the FIRST answer statement inside that region: the model
        # states its answer inside the thinking block too, and one that repeats
        # '#### 45' many times would leave earlier copies behind if we cut at the
        # last. 'boxed{}' with empty braces is the instruction the prompt asks the
        # model to follow, echoed back before any reasoning happens, so it must
        # not count as an answer -- requiring a non-empty brace separates the
        # echoed '\boxed{}' from a real '\boxed{145}'.
        answer_statement = re.compile(
            re.escape(self.force_generate_answer_text) + r'|Final Answer|boxed\{[^}\s]'
        )
        match = answer_statement.search(chain_of_thought)
        if match is None:
            return chain_of_thought

        return chain_of_thought[:match.start()]

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
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer,
                "problem_id": None
                }

    def generate_model_prompt_chain_of_thought(self, question_list: list[str], partial_cot_list: list[str]) -> list[str]:
        prompt_list : list[str] = []
        for question, partial_cot in zip(question_list, partial_cot_list):        
            prompt = "You are continuing an unfinished reasoning process.\n\n"
            prompt += (
                "The reasoning below represents the current reasoning state reached while "
                "solving the question.\n"
                "Assume that every reasoning step in the provided partial reasoning is "
                "correct and should be preserved.\n\n"
            )
            prompt += (
                "Follow these instructions carefully:\n"
                "- Do NOT restart the solution from the beginning.\n"
                "- Do NOT repeat, summarize, or rewrite the provided reasoning.\n"
                "- Treat the partial reasoning as the current reasoning state.\n"
                "- Continue reasoning directly from the final step of the provided partial reasoning.\n"
                "- Your first generated sentence must logically follow the final sentence of the provided reasoning.\n"
                "- If multiple valid continuations exist, choose one plausible continuation and follow it consistently until reaching a final answer.\n"
                "- Do NOT revise or question earlier reasoning unless the last step is explicitly incomplete.\n"
                "- Continue reasoning until the problem is completely solved.\n"
                "- Output only the continuation of the reasoning followed by the final answer.\n\n"
            )

            prompt += f"Question:\n{question}\n\n"
            prompt += f"Partial Reasoning:\n{partial_cot}\n\n"
            prompt += "Continue the reasoning from this point and output the final answer after ####"
            
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    @staticmethod
    def gsm8k_answer_extraction(solution: str) -> str :
        _SOLUTION_CLIP_CHARS = 300
        if len(solution) > _SOLUTION_CLIP_CHARS:
            solution = solution[-_SOLUTION_CLIP_CHARS:]

        patterns = [
            r'####.*?([0-9]+(?:[.,][0-9]+)?)',            
            r'(?i)\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}',            
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

    @staticmethod
    def extract_number(text: str) -> float:
        chars_to_remove = "\\!@#$%^&*(),/"
        table = str.maketrans('', '', chars_to_remove)

        result = text.translate(table)        
        pattern = r'-?\d+\.\d+|-?\d+'
        match = re.search(pattern, result)

        if match:
            return float(match.group())
        return None
    
# config = dataset_config('Qwen/Qwen2.5-1.5B')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = gsm8k_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))

