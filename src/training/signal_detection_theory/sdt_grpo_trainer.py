from trl import GRPOTrainer, get_peft_config
from abc import ABC, abstractmethod
from src.logger.training.signal_detection_theory.training_sdt_log_entity import training_sdt_log_entity
from src.utils.llm_representation import llm_representation
from src.utils.enums_class import training_type_enum, confidence_type_enum, confidence_reward_calculation_type_enum
from src.training.training_config import training_config
from scipy.stats import norm
import torch
import re
import numpy as np

class sdt_grpo_trainer(ABC): 

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
            train_dataset, eval_dataset = self.get_dataset().preprocess_dataset_with_confidence()
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

        if training_type_enum.SIGNAL_DETECTION_THEORY == self.config.training_type: 
            return [self.accuracy_reward, self.calculate_sdt_reward]
        
        return []

    @torch.inference_mode()
    def calculate_sdt_reward(self, completions, target=None, tokenizer=None, **kwargs):
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        questions = kwargs.get("question")     
        trainer_state = kwargs.get("trainer_state", None)
        trainer_global_step = trainer_state.global_step
        
        trainer = self.get_trainer()
        tokenizer = trainer.processing_class

        log_list: list[training_sdt_log_entity] = self.get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
        another_prompt_list : list[list[int]] = []
        for log in log_list:
            question = log.question
            answer = self.get_dataset().generate_wrong_answer(log.target) if log.accuracy else log.target
            another_prompt = self.get_dataset().generate_another_prompt_confidence(question, answer)
            log.another_prompt = another_prompt
            log.another_target = answer
            prefix = [
                {"role": "user",
                    "content": another_prompt
                    },
            ]
            another_prompt_list.append(tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)['input_ids'])

        _, completion_ids, _, _ = trainer.vllm_generation.generate(prompts=another_prompt_list, images=[], num_generations=1)
        another_completion_confidence_list = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for another_completion_confidence, log in zip(another_completion_confidence_list, log_list):
            log.another_completion = another_completion_confidence

        rewards = []
        for log in log_list:
            log.another_confidence = self.get_dataset().extract_another_confidence(log.another_completion)
            if log.confidence is None or log.another_confidence is None: 
                log.sdt_reward = 0.0
                rewards.append(0.0)
                continue
            
            eps=1e-3
            hit = log.get_correct_confidence() / 100
            hit = np.clip(hit, eps, 1 - eps)
            
            false_alarm = log.get_incorrect_confidence() / 100
            false_alarm = np.clip(false_alarm, eps, 1 - eps)
            
            d_prime = norm.ppf(hit) - norm.ppf(false_alarm)
            log.sdt_d_prime = d_prime
            log.sdt_reward = (np.tanh(d_prime / 3.0) + 1.0) / 2.0
            
            reward = self.config.confidence_reward_coefficient * log.sdt_reward
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

        log_list: list[training_sdt_log_entity] = self.get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
        for log in log_list: 
            rewards.append(log.accuracy_reward)

        if training_type_enum.ACCURACY_REWARD == self.config.training_type:
            self.get_logger().write_to_log_file()

        return rewards

    def get_log_list(self, trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target) -> list[training_sdt_log_entity]:
        log_list = self.get_logger().get_log_list(trainer_global_step)
        
        if len(log_list) > 0:
            return log_list

        log_list: list[training_sdt_log_entity] = []
        for i, (completion, gt, question) in enumerate(zip(completions, target, questions)):
            split = split_list[i]
            sample_ID = sample_ids[i]
            problem_id = problem_ids[i] if problem_ids is not None else None
            prompt = prompts[i]

            log = training_sdt_log_entity()
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
                answer, confidence, target_answer_equal, compared_final_answer = self.get_dataset().extract_and_verify_final_answer_confidence(prompt, completion, gt)
                if answer is None:
                    acc_reward = 0.0
                    log.accuracy = False
                else:
                    acc_reward = 1.0 if target_answer_equal else 0.0
                    log.accuracy = target_answer_equal

                log.final_answer = answer
                log.compared_final_answer = compared_final_answer
                log.confidence = confidence
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




    