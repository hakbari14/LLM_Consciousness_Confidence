from trl import GRPOTrainer, get_peft_config
from abc import ABC, abstractmethod
from src.logger.training.training_log_entity import training_log_entity
from src.utils.llm_representation import llm_representation
from src.utils.enums_class import training_type_enum, confidence_type_enum, confidence_reward_calculation_type_enum
from src.training.training_config import training_config
import torch
import re
import gc
import math

class grpo_trainer(ABC): 

    def __init__(self, config: training_config) -> None:
        self.config = config
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
                model = self.config.model_name,
                reward_funcs = self.get_reward_funcs(),
                args = self.get_training_args(),
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                peft_config = get_peft_config(model_config),
            )
            
        self.trainer.model.print_trainable_parameters() 
        return self.trainer

    def get_reward_funcs(self):
        if training_type_enum.ACCURACY_REWARD == self.config.training_type: 
            return [self.accuracy_reward]

        if training_type_enum.CONFIDENCE == self.config.training_type: 
            return [self.calculate_confidence_reward, self.calculate_entropy_confidence_reward]

        if training_type_enum.ACCURACY_REWARD_CONFIDENCE == self.config.training_type: 
            return [self.accuracy_reward, self.calculate_confidence_reward, self.calculate_entropy_confidence_reward]
        
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
        log_list, prompt_ID_list = self.generate_prompt_confidence(log_list)
    
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class

        _, completion_ids, _, _ = trainer.vllm_generation.generate(prompts=prompt_ID_list, images=[], num_generations=1)
        completion_confidence_list = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for completion_confidence, log in zip(completion_confidence_list, log_list):
            log.completion_confidence = completion_confidence
        
        rewards = []
        for log in log_list:
            log.verbal_confidence = self.extract_confidence(log.completion_confidence)
            if log.verbal_confidence is None: 
                log.verbal_confidence_reward = 0.0
                rewards.append(0.0)
                continue
                
            verbal_confidence_reward = self.calculate_verbal_confidence(log.verbal_confidence)
            
            if confidence_reward_calculation_type_enum.linear == self.config.confidence_reward_type:
                verbal_confidence_reward = verbal_confidence_reward if log.accuracy else 1.0 - verbal_confidence_reward
            elif confidence_reward_calculation_type_enum.brier_score == self.config.confidence_reward_type:
                y = 1 if log.accuracy else 0 
                verbal_confidence_reward = 1 - pow(verbal_confidence_reward - y, 2)
            
            log.verbal_confidence_reward = verbal_confidence_reward
            reward = self.config.confidence_reward_coefficient * log.verbal_confidence_reward
            rewards.append(reward)
            
        return rewards

    @torch.inference_mode()
    def calculate_entropy_confidence_reward(self, completions, target=None, tokenizer=None, **kwargs):
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        questions = kwargs.get("question")     
        trainer_state = kwargs.get("trainer_state", None)
        trainer_global_step = trainer_state.global_step
        
        log_list: list[training_log_entity] = self.get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class

        rewards = []
        for log in log_list:
            if log.completion is None or log.verbal_confidence is None: 
                log.entropy_reward = 0.0
                rewards.append(0.0)
                continue
            
            try:
                entropy, _, _ = self.representation.calculate_entropy(log.completion, model, tokenizer)
                log.entropy = entropy
                model_confidence = math.exp(-1.0 * log.entropy)
                verbal_confidence = self.calculate_verbal_confidence(log.verbal_confidence)
                
                if confidence_reward_calculation_type_enum.linear == self.config.entropy_reward_type:
                    log.entropy_reward = 1.0 - abs(verbal_confidence - model_confidence)
                elif confidence_reward_calculation_type_enum.brier_score == self.config.entropy_reward_type:
                    log.entropy_reward = 1.0 - pow(abs(verbal_confidence - model_confidence) ,2)

                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                log.entropy_reward = 0.0
                print(f"[WARN] Calculate Entropy: {e}")
                
            reward = self.config.entropy_confidence_reward_coefficient * log.entropy_reward
            rewards.append(reward)
        
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

        if training_type_enum.ACCURACY_REWARD == self.config.training_type:
            self.get_logger().write_to_log_file()

        return rewards

    def generate_prompt_confidence(self, log_list: list[training_log_entity]) -> tuple[list[training_log_entity], list[list[int]]]:
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class
        prompt_list : list[list[int]] = []
        for log in log_list:        
        
            if confidence_type_enum.PROBABILITY == self.config.confidence_type: 
                prompt = f'[Question]: {log.question}\n\n'
                prompt += f'[Reasoning Process]: {log.completion}\n\n'
                prompt += f'[Answer]: {log.final_answer}\n\n'

                prompt += (
                    'Your only task is to estimate how likely it is that the provided Answer is correct, '
                    'using only the Question, the Reasoning Process, and the Answer. Critically evaluate whether the reasoning process logically supports the answer.\n\n'

                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence score or number in this section.\n'
                    '2. After you have completed your reasoning, on a new line, output your final confidence in the exact format: Confidence:<integer between 0 and 100>\n'
                    '3. Do not output anything else after the confidence value. Do not output the confidence before the reasoning.\n'
                )
                
            elif confidence_type_enum.LEVEL == self.config.confidence_type:
                prompt = f'[Question]: {log.question}\n\n'
                prompt += f'[Reasoning Process]: {log.completion}\n\n'
                prompt += f'[Answer]: {log.final_answer}\n\n'

                prompt += (
                    'Your only task is to estimate how likely it is that the provided Answer is correct, '
                    'using only the Question, the Reasoning Process, and the Answer. Critically evaluate whether the reasoning process logically supports the answer.\n\n'

                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence level or category during your reasoning.\n'
                    '2. After all reasoning is complete, output exactly one final line in the following format:\n'
                    '   Confidence:<Very Low | Low | Medium | High | Very High>\n'
                    '3. The confidence level must be exactly one of: Very Low, Low, Medium, High, or Very High.\n'
                    '4. Do not output anything before the reasoning or after the confidence line.\n'
                )

            log.prompt_confidence = prompt
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)['input_ids'])
        
        return log_list, prompt_list        

    def extract_confidence(self, solution) -> str:
        if confidence_type_enum.PROBABILITY == self.config.confidence_type: 
            return self.extract_confidence_prbability(solution)
        elif confidence_type_enum.LEVEL == self.config.confidence_type:
            return self.extract_confidence_level(solution)
        
        return None

    def extract_confidence_prbability(self, solution):
        patterns = [
            r"Confidence\s*:\s*<\s*(\d{1,3})\s*>",
            r"\[\s*Confidence\s*\]\s*:\s*(\d+)",
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

    def calculate_verbal_confidence(self, verbal_confidence):
        if confidence_type_enum.PROBABILITY == self.config.confidence_type: 
            confidence = float(verbal_confidence) / 100
        elif confidence_type_enum.LEVEL == self.config.confidence_type:

            confidence_values = {
                "very low": 0.1,
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very high": 0.9,
            }
            confidence = confidence_values.get(verbal_confidence.strip().casefold(), None)            
        
        return confidence

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
                    log.accuracy = False
                else:
                    acc_reward = 1.0 if target_answer_equal else 0.0
                    log.accuracy = target_answer_equal

                log.final_answer = answer
                log.compared_final_answer = compared_final_answer
                log.accuracy_reward = self.config.acurray_reward_coefficient * acc_reward
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




    