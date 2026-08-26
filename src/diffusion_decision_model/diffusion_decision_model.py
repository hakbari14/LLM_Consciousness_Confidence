from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity
from src.datasets.dataset_handler import dataset_handler
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger
from src.utils.utility import my_utils
from math import ceil
import torch
import numpy as np 
import numpy as np
import pandas as pd
import traceback
from collections import Counter
import logging



logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)        


class diffusion_decision_model(ABC): 

    def __init__(self, modelname: str, number_of_evidence: int) -> None:
        self.modelname = modelname
        self.number_of_evidence = number_of_evidence

        if self.modelname is None:
            raise Exception('modelname is required')
        if self.number_of_evidence is None:
            raise Exception('number of evidence is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)
        

    def run(self, from_run_number: int , to_run_number: int) -> None:
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            log_list: list[diffusion_decision_model_log_entity] = self.generate_response()
            log_list = self.generate_self_consistency(log_list)
            
            logger = self.create_logger(run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()
            
            print(f"{'*' * 210}")

    @torch.inference_mode()
    def generate_response(self, batch_size = 128) -> list[diffusion_decision_model_log_entity]: 
        print(f"{'*' * 100}  Generate Response {'*' * 100}")
        _, test_dataset = self.get_dataset().preprocess_dataset()

        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.2, 
                n = 1, 
                top_p= 0.9, 
                top_k=50,
                logprobs=1
            )
        
        log_list: list[diffusion_decision_model_log_entity] = []
        idx: int = 0
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing Batches", unit="step"):
            batch_dict = test_dataset[i : i + batch_size]

            batch = [
                {key: values[j] for key, values in batch_dict.items()}
                for j in range(len(batch_dict[next(iter(batch_dict))]))
            ]

            sample_ID_list = batch_dict['sample_id']
            problem_id_list = batch_dict['problem_id']
            split_list = batch_dict['split']
            question_list = batch_dict['question']
            prompt_list = batch_dict['prompt']
            target_list = batch_dict['target']
            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    question = question_list[j]
                    x = batch[j]
                    
                    log = diffusion_decision_model_log_entity()
                    log.ID = idx
                    log.x = x
                    log.sample_ID = sample_ID
                    log.problem_id = problem_id
                    log.split = split
                    log.question = question
                    log.prompt = prompt
                    log.target = target
                    
                    if output.outputs is None: continue
                    response = output.outputs[0]
                    completion = response.text
                    
                    log.completion = completion
                    log.token_count = len(response.token_ids)
                    log.completion_loss = my_utils.get_loss_from_vllm_output(response)

                    try:
                        final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                        if final_answer is None or compared_final_answer is None: continue
                        log.final_answer = final_answer
                        log.compared_final_answer = compared_final_answer
                        log.accuracy = accuracy
                        log = self.add_evidence_log_list(log, response)
                    except Exception as e:
                        logging.exception("An exception occurred")                        
                        print(f"[WARN]: {e}")
                        traceback.print_exc()                        

                    idx += 1
                    log_list.append(log)    
            except Exception as e:
                logging.exception("An exception occurred")                        
                print(f"[WARN]: {e}")
                traceback.print_exc()                        

        return log_list

    @torch.inference_mode()
    def generate_self_consistency(self, log_list: list[diffusion_decision_model_log_entity]) -> list[diffusion_decision_model_log_entity]: 
        print(f"{'*' * 100}  Generate Self Consistency {'*' * 100}")
        sc_sample_count = 10
        batch_size = ceil(120 / sc_sample_count)
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=1.0, 
                n = sc_sample_count, 
                top_p= 0.9, 
                top_k=50, 
                logprobs=1
            )

        evidence_log_list: list[diffusion_decision_model_evidence_log_entity] = []
        x_list: list[dict] = []
        final_answer_list: list[str] = []
        for log in log_list: 
            for evidence_log in log.evidence_list:
                evidence_log_list.append(evidence_log)
                x_list.append(log.x)
                final_answer_list.append(log.compared_final_answer)
        
        for i in tqdm(range(0, len(evidence_log_list), batch_size), desc="Processing Batches", unit="step"):
            batch: list[diffusion_decision_model_evidence_log_entity] = evidence_log_list[i : i + batch_size]        
            batch_partial_cot_list = list(map(lambda x: x.partial_cot, batch))
            
            batch_x_list: list[str] = x_list[i : i + batch_size]        
            batch_final_answer_list: list[str] = final_answer_list[i : i + batch_size]        

            try:
                prompt_list : list[str] = []
                for x, partial_cot in zip(batch_x_list, batch_partial_cot_list):
                    prompt_list.append(self.get_dataset().generate_model_prompt_chain_of_thought(x, partial_cot))
                
                outputs = self.model.generate(prompt_list, sampling_params, use_tqdm=False)
                for j, output in enumerate(outputs):
                    if output.outputs is None: continue
                    idx = i + j
                    evidence_log = batch[j]
                    prompt = prompt_list[j]
                    original_final_answer = batch_final_answer_list[j]
                    
                    for index in range(sc_sample_count):
                        response = output.outputs[index]
                        completion = response.text

                        log_detail = diffusion_decision_model_log_detail_entity()
                        log_detail.index = f'{idx}_{index}'
                        log_detail.prompt = prompt
                        log_detail.completion = completion
                        log_detail.token_count = len(response.token_ids)
                        log_detail.original_final_answer = original_final_answer
                        log_detail.loss = my_utils.get_loss_from_vllm_output(response)
                        
                        try:
                            final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), original_final_answer)
                            log_detail.final_answer = final_answer
                            log_detail.compared_final_answer = compared_final_answer
                            log_detail.accuracy = accuracy
                        except Exception as e:
                            logging.exception("An exception occurred")                        
                            print(f"[WARN]: {e}")
                            traceback.print_exc()                        
                            
                        evidence_log.add_consistency_list(log_detail)
                        
                    true_count = sum(x.accuracy for x in evidence_log.consistency_list)
                    evidence_log.evidence_accumulation_self_consistency = (true_count + 1.0) / (len(evidence_log.consistency_list) + 1.0)

                    losses = [
                        x.loss
                        for x in evidence_log.consistency_list
                        if x.accuracy and x.loss is not None
                    ]
                    losses.append(evidence_log.partial_cot_loss)
                    evidence_log.evidence_accumulation_loss = float(np.mean(losses))

            except Exception as e:
                logging.exception("An exception occurred")                        
                print(f"[WARN]: {e}")
                traceback.print_exc()                        

        
        for log in log_list: 
            for i in range(1, len(log.evidence_list)):
                current = log.evidence_list[i]
                previous = log.evidence_list[i - 1]
                current.delta_evidence_self_consistency = current.evidence_accumulation_self_consistency - previous.evidence_accumulation_self_consistency
                current.delta_evidence_loss = previous.evidence_accumulation_loss - current.evidence_accumulation_loss
            
            self.calculate_self_consistency_log(log)

        return log_list

    def calculate_self_consistency_log(self, log: diffusion_decision_model_log_entity) -> diffusion_decision_model_log_entity:
        evidence_filtered_list = list(filter(lambda x: x.index == 0 , log.evidence_list))
        if len(evidence_filtered_list) == 0:
            return log
        
        evidence_0: diffusion_decision_model_evidence_log_entity = evidence_filtered_list[0]
        answers = [
            log_detail.compared_final_answer
            for log_detail in evidence_0.consistency_list
            if log_detail.compared_final_answer is not None
        ]

        if not answers: 
            return log
        
        answer_counts = Counter(answers)
        mv_compare_final_answer, mv_count = answer_counts.most_common(1)[0]
        log.self_consistency_confidence = mv_count / len(evidence_0.consistency_list)
        try:
            mv_accuracy, compared_final_answer = self.get_dataset().verify_final_answer(log.target, str(mv_compare_final_answer))
            log.self_consistency_accuracy = mv_accuracy
            log.self_consistency_final_answer = compared_final_answer
        except Exception as e:
            logging.exception("An exception occurred")                        
            print(f"[WARN]: {e}")
            traceback.print_exc()                        
            
        return log
    
    def add_evidence_log_list(self, log: diffusion_decision_model_log_entity, response) -> diffusion_decision_model_log_entity:
        token_count = len(response.logprobs)
        base = token_count // (self.number_of_evidence + 1)
        remainder = token_count % (self.number_of_evidence + 1)
        
        token_ids = response.token_ids 
        start = 0

        evidence_log: diffusion_decision_model_evidence_log_entity = self.create_evidence_log(index = 0, evidence = '', partial_cot = '', partial_completion = log.completion, partial_cot_loss = log.completion_loss)
        log.add_evidence_list(evidence_log)

        for i in range(1, self.number_of_evidence):
            group_size = base + (1 if i < remainder else 0)
            evidence = self.tokenizer.decode(token_ids[start:start + group_size], skip_special_tokens=True)
            partial_cot = self.tokenizer.decode(token_ids[0:start + group_size], skip_special_tokens=True)
            partial_completion = self.tokenizer.decode(token_ids[start + group_size:], skip_special_tokens=True)
            partial_cot_loss: float = my_utils.get_loss_from_vllm_output(response, token_start = start + group_size)

            evidence_log: diffusion_decision_model_evidence_log_entity = self.create_evidence_log(index = i, evidence = evidence, partial_cot = partial_cot, partial_completion = partial_completion, partial_cot_loss = partial_cot_loss)
            log.add_evidence_list(evidence_log)

            start += group_size
           
                
        return log

    def create_evidence_log(self, index: int, evidence: str , partial_cot: str, partial_completion: str, partial_cot_loss: float) -> int:
        evidence_log = diffusion_decision_model_evidence_log_entity()
        evidence_log.index = index
        evidence_log.evidence = evidence
        evidence_log.partial_cot = partial_cot
        evidence_log.partial_completion = partial_completion
        evidence_log.partial_cot_loss = partial_cot_loss
        return evidence_log

    @staticmethod
    def load_logs_list(df_logs, df_evidences, df_samples) -> list[diffusion_decision_model_log_entity]:
        log_list: list[diffusion_decision_model_log_entity] = []
        for _, a_row in df_logs.iterrows():
            log = diffusion_decision_model_log_entity()
            log.ID = a_row["ID"]
            log.sample_ID = a_row["Sample_ID"]
            log.problem_id = a_row["problem_id"]
            log.split = a_row["Split"]
            log.question = a_row["Question"]
            log.prompt = a_row["Prompt"]
            log.target = a_row["Target"]
            log.completion = a_row["Completion"]
            log.completion_loss = a_row["Completion_Loss"]
            log.final_answer = a_row["Final_Answer"]
            log.compared_final_answer = a_row["Compared_Final_Answer"]
            log.accuracy = a_row["Accuracy"]
            log.token_count = a_row["Token_Count"]
            log.evidence_accumulation_avg = a_row["Evidence_Accumulation_Avg"]
            log.driff_rate = a_row["Drift_Rate"]

            b_subset = df_evidences[df_evidences["Sample_ID"] == log.sample_ID]
            for _, b_row in b_subset.iterrows():
                log_evidence = diffusion_decision_model_evidence_log_entity()
                log_evidence.index = b_row["Evidence_Index"]
                log_evidence.evidence = b_row["Evidence"]
                log_evidence.partial_cot = b_row["Partial_COT"]
                log_evidence.partial_cot_loss = b_row["Partial_COT_Loss"]
                log_evidence.partial_completion = b_row["Partial_Completion"]
                log_evidence.evidence_accumulation_self_consistency = b_row["Evidence_Accumulation_Self_Consistency"]
                log_evidence.delta_evidence_self_consistency = b_row["Delta_Evidence_Self_Consistency"]
                log_evidence.evidence_accumulation_loss = b_row["Evidence_Accumulation_Loss"]
                log_evidence.delta_evidence_loss = b_row["Delta_Evidence_Loss"]

                s_subset = df_samples[(df_samples["Sample_ID"] == log.sample_ID) & (df_samples["Evidence_Index"] == log_evidence.index)]
                for _, s_row in s_subset.iterrows():
                    log_detail = diffusion_decision_model_log_detail_entity()
                    log_detail.index = s_row["Index"]
                    log_detail.prompt = s_row["Prompt"]
                    log_detail.completion = s_row["Completion"]
                    log_detail.token_count = s_row["Token_Count"]
                    log_detail.original_final_answer = s_row["Original_Final_Answer"]
                    log_detail.final_answer = s_row["Final_Answer"]
                    log_detail.compared_final_answer = s_row["Compared_Final_Answer"]
                    log_detail.accuracy = s_row["Accuracy"]
                    log_detail.loss = s_row["Loss"]
                
                    log_evidence.add_consistency_list(log_detail)

                log.add_evidence_list(log_evidence)
            
            log_list.append(log)

        return log_list

    def get_max_new_tokens(self) -> int:
        return 15000

    def get_modelname_dir(self) -> str:
        return self.modelname.replace('/', '-').lower()

    def get_number_of_evidence_dir(self) -> str:
        return f'_nv_{self.number_of_evidence}'

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_logger(self, run_number) -> diffusion_decision_model_logger:
        pass

