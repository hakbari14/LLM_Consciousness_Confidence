from src.utils.enums_class import dataset_element_type_enum, llm_pipeline_type_enum
from src.datasets.dataset_config import dataset_config
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from datasets import Dataset
import configparser
import random

class dataset_handler(ABC): 

    def __init__(self, config : dataset_config):
        self.config: dataset_config = config
        self.config.validate()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get_model_name())
        self.prompt_config = configparser.ConfigParser()
        self.prompt_config.read('src/datasets/dataset_prompt.cfg')

    def preprocess_dataset(self) -> tuple[Dataset, Dataset]:
        train_dataset = self.train_dataset.map(lambda x: self.generate_model_prompt(x))
        train_dataset = train_dataset.filter(lambda x: self.filter_by_required_criteria(x, dataset_element_type_enum.TRAIN))

        test_dataset = self.test_dataset.map(lambda x: self.generate_model_prompt(x))
        test_dataset = test_dataset.filter(lambda x: self.filter_by_required_criteria(x, dataset_element_type_enum.EVAL))
       
        train_dataset = train_dataset.add_column("split", [dataset_element_type_enum.TRAIN] * len(train_dataset))
        train_dataset = train_dataset.add_column("sample_id", list(range(len(train_dataset))))

        if self.config.get_max_test_dataset_size() is not None: 
            rng = random.Random(42)            
            indices = rng.sample(range(len(test_dataset)), self.config.get_max_test_dataset_size())
            test_dataset = test_dataset.select(indices)

        eval_dataset = test_dataset
        eval_dataset = eval_dataset.add_column("split", [dataset_element_type_enum.EVAL] * len(eval_dataset))
        eval_dataset = eval_dataset.add_column("sample_id", list(range(len(eval_dataset))))

        return train_dataset, eval_dataset

    def preprocess_dataset_with_confidence(self) -> tuple[Dataset, Dataset]:
        train_dataset = self.train_dataset.map(lambda x: self.generate_model_prompt_confidence(x))
        train_dataset = train_dataset.filter(lambda x: self.filter_by_required_criteria(x, dataset_element_type_enum.TRAIN))

        test_dataset = self.test_dataset.map(lambda x: self.generate_model_prompt_confidence(x))
        test_dataset = test_dataset.filter(lambda x: self.filter_by_required_criteria(x, dataset_element_type_enum.EVAL))
       
        train_dataset = train_dataset.add_column("split", [dataset_element_type_enum.TRAIN] * len(train_dataset))
        train_dataset = train_dataset.add_column("sample_id", list(range(len(train_dataset))))

        if self.config.get_max_test_dataset_size() is not None: 
            rng = random.Random(42)            
            indices = rng.sample(range(len(test_dataset)), self.config.get_max_test_dataset_size())
            test_dataset = test_dataset.select(indices)

        eval_dataset = test_dataset
        eval_dataset = eval_dataset.add_column("split", [dataset_element_type_enum.EVAL] * len(eval_dataset))
        eval_dataset = eval_dataset.add_column("sample_id", list(range(len(eval_dataset))))

        return train_dataset, eval_dataset

    def generate_model_prompt_permutation(self, x, num_choice_permutations: int) -> list[dict]:
        data_list = []
        for i in range(0, num_choice_permutations):
            data_list.append(self.generate_model_prompt(x))
            
        return data_list

    def filter_by_required_criteria(self, x: dict, dataset_type: dataset_element_type_enum) -> bool:
        return True

    def extract_and_verify_final_answer(self, prompt: str, completion: str, target: str) -> tuple[str, bool, str]:
        final_answer = self.final_answer_extraction(prompt, completion, target)
        if final_answer is None:
            return final_answer, False, final_answer

        target_answer_equal, comapred_final_answer = self.verify_final_answer(target, final_answer)
        return final_answer, target_answer_equal, comapred_final_answer

    def extract_and_verify_final_answer_confidence(self, prompt: str, completion: str, target: str) -> tuple[str, str, bool, str]:
        final_answer, confidence = self.final_answer_confidence_extraction(prompt, completion, target)
        if final_answer is None:
            return final_answer, confidence, False, final_answer

        target_answer_equal, comapred_final_answer = self.verify_final_answer(target, final_answer)
        return final_answer, confidence, target_answer_equal, comapred_final_answer
        
    def verify_final_answer(self, target: str, final_answer: str) -> tuple[bool, str]:
        return final_answer == target, final_answer

    def permute_options(self, options: list[str], correct_index: int, num_permutations: int = 10) -> list[tuple[list[str], int]]:
        results = []
        for _ in range(num_permutations):
            indexed_options = list(enumerate(options))

            random.shuffle(indexed_options)
            new_options = []
            new_correct_index = None

            for new_idx, (old_idx, opt) in enumerate(indexed_options):
                new_options.append(opt)
                if old_idx == correct_index:
                    new_correct_index = new_idx

            results.append((new_options, new_correct_index))
        return results

    def get_config(self) -> dataset_config: 
        return self.config

    def set_config(self, value: dataset_config) -> None: 
        self.config = value
    
    @abstractmethod
    def final_answer_extraction(self, prompt, completion, target):
        pass

    def get_final_answer_marker(self) -> str:
        return getattr(self, 'force_generate_answer_text', '####')

    def strip_answer_statement(self, text: str) -> str:
        patterns = [self.get_final_answer_marker(), 'Final Answer', 'boxed{']
        min_pos = len(text)
        for pattern in patterns:
            pos = text.find(pattern)
            if pos == -1: continue

            min_pos = min(min_pos, pos)

        return text[:min_pos]

    def chain_of_thought_extraction(self, question: str, solution: str) -> str:
        """Return the reasoning with the final answer statement removed.

        The diffusion decision model chunks this into growing prefixes, so any
        answer left behind would let a continuation agree without reasoning.
        """
        chain_of_thought = solution
        pos = chain_of_thought.find(question)
        if pos != -1:
            chain_of_thought = chain_of_thought[pos + len(question):]

        # The prompt asks for the answer within \boxed{}, and the model echoes that
        # instruction back before it reasons. Those empty braces are not an answer,
        # so drop them or they cut the chain of thought off at its first characters.
        chain_of_thought = chain_of_thought.replace('\\boxed{}', '')

        think_end = chain_of_thought.find('</think>')
        if think_end == -1:
            return self.strip_answer_statement(chain_of_thought)

        # The model reasons either inside the <think> block or, when it closes
        # that block immediately after echoing the prompt instruction, in the
        # presentation that follows. Strip the answer from both candidates and
        # keep whichever retains more reasoning: comparing the stripped results
        # cannot discard the larger body of reasoning, so a genuinely short chain
        # of thought survives instead of being traded for an empty presentation.
        thinking = self.strip_answer_statement(chain_of_thought[:think_end])
        presentation = self.strip_answer_statement(chain_of_thought[think_end + len('</think>'):])

        return thinking if len(thinking) >= len(presentation) else presentation

    @abstractmethod
    def final_answer_confidence_extraction(self, prompt, completion, target):
        pass

    @abstractmethod
    def generate_model_prompt(self, x):
        pass

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
            prompt += f"Continue the reasoning from this point and output the final answer after {self.get_final_answer_marker()}"

            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))

        return prompt_list

    @abstractmethod
    def generate_model_prompt_confidence(self, x):
        pass

    @abstractmethod
    def generate_another_prompt_confidence(self, question: str, answer: str) -> str:
        pass

    @abstractmethod
    def extract_another_confidence(self, solution: str) -> float:
        pass
    
    @abstractmethod
    def generate_wrong_answer(self, x):
        pass


