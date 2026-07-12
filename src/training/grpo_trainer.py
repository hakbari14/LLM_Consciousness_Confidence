from trl import GRPOTrainer, get_peft_config
from abc import ABC, abstractmethod
from src.logger.training.training_log_entity import training_log_entity
from src.utils.llm_representation import llm_representation
from src.utils.enums_class import training_type_enum, confidence_type_enum
import torch
import re

class grpo_trainer(ABC): 

    def __init__(self, model_name: str, training_type: training_type_enum, confidence_type : confidence_type_enum) -> None:
        self.model_name = model_name
        self.training_type  = training_type
        self.confidence_type  = confidence_type
        if self.model_name is None:
            raise Exception('model name is required')

        self.representation = llm_representation()
        self.dataset = None
        self.model_config = None
        self.training_args = None
        self.trainer = None
        self.logger = None

    def train(self):
        trainer = self.get_trainer()
        trainer.model.config.use_cache = False
        trainer.train()

    def get_trainer(self):
        if self.trainer is None:
            train_dataset, eval_dataset = self.get_dataset().preprocess_dataset()
            model_config = self.get_model_config()
            
            self.trainer = GRPOTrainer(
                model = self.model_name,
                reward_funcs = self.get_reward_funcs(),
                args = self.get_training_args(),
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                peft_config = get_peft_config(model_config),
            )
            
        self.trainer.model.print_trainable_parameters() 
        return self.trainer

    def get_reward_funcs(self):
        if training_type_enum.ACCURACY_REWARD == self.training_type: 
            return [self.accuracy_reward]

        if training_type_enum.CONFIDENCE == self.training_type or training_type_enum.CONFIDENCE_WITH_CRITERAI == self.training_type: 
            return [self.calculate_confidence_reward]

        if training_type_enum.ACCURACY_REWARD_CONFIDENCE == self.training_type or training_type_enum.ACCURACY_REWARD_CONFIDENCE_WITH_CRITERAI == self.training_type: 
            return [self.accuracy_reward, self.calculate_confidence_reward]
        
        return []

    @torch.inference_mode()
    def calculate_confidence_reward(self, completions, target=None, tokenizer=None, **kwargs):
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        questions = kwargs.get("question")     
        trainer_state = kwargs.get("trainer_state", None)
        trainer_global_step = trainer_state.global_step
        
        log_list: list[training_log_entity] = self.get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
        if training_type_enum.CONFIDENCE == self.training_type or training_type_enum.ACCURACY_REWARD_CONFIDENCE == self.training_type:
            log_list = self.generate_confidence(log_list)
        if training_type_enum.CONFIDENCE_WITH_CRITERAI == self.training_type or training_type_enum.ACCURACY_REWARD_CONFIDENCE_WITH_CRITERAI == self.training_type:
            log_list = self.generate_self_criteria(log_list)
            log_list = self.generate_confidence_in_criteria_mode(log_list)

        rewards = []
        for log in log_list:
            if log.confidence is None: 
                log.confidence_reward = 0.0
                rewards.append(log.confidence_reward)
                continue
            
            confidence_reward = self.calculate_confidence_reward_on_log(log)
            log.confidence_reward = confidence_reward
            rewards.append(confidence_reward)

        self.get_logger().write_to_log_file()
        return rewards

    def accuracy_reward(self, completions, target, **kwargs):
        rewards = []
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        questions = kwargs.get("question")     
        trainer_state = kwargs.get("trainer_state", None)
        trainer_global_step = trainer_state.global_step

        log_list: list[training_log_entity] = self.get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
        for log in log_list: 
            rewards.append(log.accuracy_reward)

        if training_type_enum.ACCURACY_REWARD == self.training_type:
            self.get_logger().write_to_log_file()

        return rewards

    def generate_confidence(self, log_list: list[training_log_entity]) -> list[training_log_entity]:
        log_list, prompt_ID_list = self.generate_prompt_confidence(log_list)
    
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class

        _, completion_ids, _, _ = trainer.vllm_generation.generate(prompts=prompt_ID_list, images=[], num_generations=1)
        completion_confidence_list = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for completion_confidence, log in zip(completion_confidence_list, log_list):
            log.completion_confidence = completion_confidence
        
        for log in log_list:
            log.confidence = self.extract_confidence(log.completion_confidence)
        
        return log_list

    def generate_self_criteria(self, log_list: list[training_log_entity]) -> list[training_log_entity]:
        log_list, prompt_ID_list = self.generate_prompt_self_criteria(log_list)
    
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class

        _, completion_ids, _, _ = trainer.vllm_generation.generate(prompts=prompt_ID_list, images=[], num_generations=1)
        completion_sc_list = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for completion_sc, log in zip(completion_sc_list, log_list):
            log.completion_self_criteria = completion_sc
        
        for log in log_list:
            log.self_criteria = self.extract_self_criteria(log.completion_self_criteria)
        
        return log_list

    def generate_confidence_in_criteria_mode(self, log_list: list[training_log_entity]) -> list[training_log_entity]:
        log_list, prompt_ID_list = self.generate_prompt_confidence_in_criteria_mode(log_list)
    
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class

        _, completion_ids, _, _ = trainer.vllm_generation.generate(prompts=prompt_ID_list, images=[], num_generations=1)
        completion_confidence_list = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for completion_confidence, log in zip(completion_confidence_list, log_list):
            log.completion_confidence = completion_confidence
        
        for log in log_list:
            log.confidence = self.extract_confidence(log.completion_confidence)
        
        return log_list
        
    def generate_prompt_confidence(self, log_list: list[training_log_entity]) -> tuple[list[training_log_entity], list[list[int]]]:
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class
        prompt_list : list[list[int]] = []
        for log in log_list:        
        
            if confidence_type_enum.PROBABILITY == self.confidence_type: 
                prompt = f'[Question]: {log.question}\n\n'
                prompt += f'[Answer]: {log.final_answer}\n\n'

                prompt += 'Your task is only to evaluate the likelihood that the given answer is correct based on the question and the answer.\n'
                prompt += 'Do not revise, improve, replace, or reinterpret the answer. Evaluate the answer exactly as provided.\n'
                prompt += 'Base your confidence estimate on the consistency between the question and the answer.\n'
                prompt += 'Interpret the confidence score as the estimated probability that the given answer is correct.\n'
                prompt += 'Reserve confidence values near the extremes (0 or 100) for exceptional cases where the available evidence overwhelmingly supports such certainty.\n'
                prompt += 'Return only:\n'
                prompt += 'Confidence:<integer between 0 and 100>\n'
                
            elif confidence_type_enum.LEVEL == self.confidence_type:
                prompt = f'[Question]: {log.question}\n\n'
                prompt += f'[Answer]: {log.final_answer}\n\n'

                prompt += 'Your task is only to evaluate the likelihood that the given answer is correct based on the question and the answer.\n'
                prompt += 'Do not revise, improve, replace, or reinterpret the answer. Evaluate the answer exactly as provided.\n'
                prompt += 'Base your confidence estimate on the consistency between the question and the answer.\n'
                prompt += 'Select exactly one confidence level that best reflects how likely the given answer is to be correct.\n'
                prompt += 'Use the extreme confidence levels (Very Low and Very High) only when the available evidence overwhelmingly supports such certainty.\n'
                prompt += 'Return only in the following format:\n'
                prompt += 'Confidence:<Very Low | Low | Medium | High | Very High>\n'

            log.prompt_confidence = prompt
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)['input_ids'])
        
        return log_list, prompt_list        

    def generate_prompt_self_criteria(self, log_list: list[training_log_entity]) -> tuple[list[training_log_entity], list[list[int]]]:
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class
        prompt_list : list[list[int]] = []
        for log in log_list:        
        
            prompt = f'[Question]: {log.question}\n\n'
            prompt += f'[Answer]: {log.final_answer}\n\n'
            prompt += f'[Reasoning Process]: {log.completion}\n\n'
            
            prompt += 'A question, its answer, and the reasoning process used to reach the answer are provided.\n'
            prompt += 'Based on the question, answer, and reasoning process, generate a list of up to five criteria for assessing confidence in the correctness of the answer.\n'
            prompt += 'Output requirements:\n'
            prompt += 'Return only the criteria and nothing else.\n'
            prompt += 'Format the output strictly as a list using either number or - (e.g., "1. criterion" or "- criterion" or "[1] criterion").\n'
            prompt += 'Each criterion must appear on a separate line.\n'
            
            log.prompt_self_criteria = prompt
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)['input_ids'])

        
        return log_list, prompt_list        


    def generate_prompt_confidence_in_criteria_mode(self, log_list: list[training_log_entity]) -> tuple[list[training_log_entity], list[list[int]]]:
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class
        prompt_list : list[list[int]] = []
        for log in log_list:        
        
            if confidence_type_enum.PROBABILITY == self.confidence_type: 
                prompt = f'[Question]: {log.question}\n\n'
                prompt += f'[Answer]: {log.final_answer}\n\n'
                prompt += f'[Reasoning Process]: {log.completion}\n\n'
                prompt += f'[Evaluation Criteria]:\n{log.self_criteria}\n\n'

                prompt += 'Your task is only to evaluate the likelihood that the given answer is correct based on the question, the answer, the reasoning process, and the evaluation criteria.\n'
                prompt += 'Do not revise, improve, replace, or reinterpret the answer. Evaluate the answer exactly as provided.\n'
                prompt += 'Consider all available information before estimating the confidence score.\n'
                prompt += 'Interpret the confidence score as the estimated probability that the given answer is correct.\n'
                prompt += 'Reserve confidence values near the extremes (0 or 100) for exceptional cases where the available evidence overwhelmingly supports such certainty.\n'
                prompt += 'Return only:\n'
                prompt += 'Confidence:<integer between 0 and 100>\n'
                
            elif confidence_type_enum.LEVEL == self.confidence_type:
                prompt = f'[Question]: {log.question}\n\n'
                prompt += f'[Answer]: {log.final_answer}\n\n'
                prompt += f'[Reasoning Process]: {log.completion}\n\n'
                prompt += f'[Evaluation Criteria]:\n{log.self_criteria}\n\n'

                prompt += 'Your task is only to evaluate the likelihood that the given answer is correct based on the question, the answer, the reasoning process, and the evaluation criteria.\n'
                prompt += 'Do not revise, improve, replace, or reinterpret the answer. Evaluate the answer exactly as provided.\n'
                prompt += 'Base your confidence assessment on the consistency between the question, the answer, the reasoning process, and the evaluation criteria.\n'
                prompt += 'Select exactly one confidence level that best reflects how likely the given answer is to be correct.\n'
                prompt += 'Use the extreme confidence levels (Very Low and Very High) only when the available evidence overwhelmingly supports such certainty.\n'
                prompt += 'Return only in the following format:\n'
                prompt += 'Confidence:<Very Low | Low | Medium | High | Very High>\n'


            log.prompt_confidence = prompt
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)['input_ids'])
        
        return log_list, prompt_list        


    def extract_confidence(self, solution) -> str:
        if confidence_type_enum.PROBABILITY == self.confidence_type: 
            return self.extract_confidence_prbability(solution)
        elif confidence_type_enum.LEVEL == self.confidence_type:
            return self.extract_confidence_level(solution)
        
        return None

    def extract_confidence_prbability(self, solution):
        patterns = [
            r"Confidence[\s*]*:[\s*]*(\d+(?:\.\d+)?)",
            r"Confidence\s*Score[\s*\n:]*([0-9]+(?:\.[0-9]+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue
            answer = float(match.group(1))
            if answer > 100 or answer < 0: continue
            return str(answer)
        
        return None

    def extract_confidence_level(self, solution):
        levels = [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
        lookup = {item.lower(): item for item in levels}

        patterns = [
            r"confidence\s*[:=\-]?\s*(very\s+low|low|medium|high|very\s+high)\b",
            r"\b(very\s+low|low|medium|high|very\s+high)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue

            confidence_level = match.group(1)
            if not confidence_level: continue
            
            return lookup.get(confidence_level.lower())            
        
        return None

    def calculate_confidence_reward_on_log(self, log):
        if confidence_type_enum.PROBABILITY == self.confidence_type: 
            if log.accuracy: 
                return float(log.confidence) / 100
            else:
                return 1.0 - float(log.confidence) / 100
            
        elif confidence_type_enum.LEVEL == self.confidence_type:

            confidence_values = {
                "very low": 0.1,
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very high": 0.9,
            }

            confidence_reward = confidence_values.get(log.confidence.strip().casefold(), None)            
            if confidence_reward is None:
                return None
             
            if log.accuracy: 
                return confidence_reward
            else:
                return 1.0 - confidence_reward
        
        return None

    def extract_self_criteria(self, self_criteria_completion):
        parts = self_criteria_completion.split("</think>", 1)
        if len(parts) >= 2:
            content = parts[1]
        else:
            content = self_criteria_completion
            
        patterns = [
                r"^\s*\**\s*Step\s+\d+\s*[:\.\-]?\s*(.+)$", 
                r"^\s*(?:\[\s*\d+\s*\]|\d+\s*[\.:]?|-[\.:]?)\s*(.+)$",
                r"^\s*(?:\d+\s*[\.:]?|-[\.:]?)\s*(.+)$",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, flags=re.MULTILINE | re.IGNORECASE)
            if not matches or len(matches) == 0: continue
            result = "\n".join(
                f"{i}. {item.strip()}"
                for i, item in enumerate(matches, start=1)
            )
            return result

        return ""

    def get_log_list(self, trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target) -> list[training_log_entity]:
        log_list = self.get_logger().get_log_list(trainer_global_step)
        
        if len(log_list) > 0:
            return log_list

        log_list: list[training_log_entity] = []
        for i, (completion, gt, question) in enumerate(zip(completions, target, questions)):
            split = split_list[i]
            sample_ID = sample_ids[i]
            problem_id = problem_ids[i] if problem_ids is not None else None
            prompt = prompts[i]

            log = training_log_entity()
            log.ID = f'{sample_ID}_{i}' 
            log.sample_ID = sample_ID
            log.problem_id = problem_id
            log.split = split
            log.trainer_global_step = trainer_global_step
            log.question = question
            log.prompt = prompt
            log.target = gt
            log.completion = completion
            try:
                answer, target_answer_equal, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, completion, gt)
                if answer is None:
                    acc_reward = 0.0
                else:
                    acc_reward = 1.0 if target_answer_equal else 0.0

                log.final_answer = answer
                log.compared_final_answer = compared_final_answer
                log.accuracy_reward = acc_reward
                log.accuracy = acc_reward == 1.0
            except Exception:
                log.accuracy = False
                log.accuracy_reward = 0.0
            
            log_list.append(log)
            self.get_logger().add_to_buffer(log)
        
        return log_list
    
    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def get_training_args(self):
        pass

    @abstractmethod
    def get_logger(self):
        pass




    