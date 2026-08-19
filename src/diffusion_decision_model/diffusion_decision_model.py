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
import re
import numpy as np 

class diffusion_decision_model(ABC): 

    def __init__(self, modelname: str, number_of_evidence: int | None = None) -> None:
        self.modelname = modelname
        self.number_of_evidence = number_of_evidence
        
        if self.modelname is None:
            raise Exception('modelname is required')
        
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
            batch = test_dataset[i : i + batch_size]

            sample_ID_list = batch['sample_id']
            problem_id_list = batch['problem_id']
            split_list = batch['split']
            question_list = batch['question']
            prompt_list = batch['prompt']
            target_list = batch['target']
            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    question = question_list[j]
                    
                    log = diffusion_decision_model_log_entity()
                    log.ID = idx
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
                        log.final_answer = final_answer
                        log.compared_final_answer = compared_final_answer
                        log.accuracy = accuracy
                        log = self.add_evidence_log_list(log)
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")

                    idx += 1
                    log_list.append(log)    
            except Exception as e:
                print(f"[WARN] generate failed: {e}")

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
        question_list: list[str] = []
        final_answer_list: list[str] = []
        for log in log_list: 
            if log.final_answer is None: continue
            
            for evidence_log in log.evidence_list:
                evidence_log_list.append(evidence_log)
                question_list.append(log.question)
                final_answer_list.append(log.final_answer)
        
        for i in tqdm(range(0, len(evidence_log_list), batch_size), desc="Processing Batches", unit="step"):
            batch: list[diffusion_decision_model_evidence_log_entity] = evidence_log_list[i : i + batch_size]        
            batch_partial_cot_list = list(map(lambda x: x.partial_cot, batch))
            
            batch_question_list: list[str] = question_list[i : i + batch_size]        
            batch_final_answer_list: list[str] = final_answer_list[i : i + batch_size]        

            try:
                prompt_list = self.get_dataset().generate_model_prompt_chain_of_thought(batch_question_list, batch_partial_cot_list)
                outputs = self.model.generate(prompt_list, sampling_params)
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
                            print(f"[WARN] generate failed: {e}")
                            
                        evidence_log.add_consistency_list(log_detail)
                        
                    true_count = sum(x.accuracy for x in evidence_log.consistency_list)
                    # Laplace smoothing: (k + 1) / (K + 2) keeps the estimate strictly
                    # inside (0, 1). (K + 1) would return exactly 1.0 when every
                    # continuation agrees, and its logit -- the evidence axis a drift
                    # rate is fitted on -- would be infinite.
                    evidence_log.evidence_accumulation_self_consistency = (true_count + 1.0) / (len(evidence_log.consistency_list) + 2.0)

                    # Per-token NLL, not the raw sum: x.loss is -sum(logprobs) and so
                    # grows with completion length. Averaging raw sums would compare
                    # a long continuation against a short one on different scales.
                    # completion_loss is deliberately NOT averaged in here: it is a
                    # per-sample constant covering the whole phase-1 trajectory, so it
                    # carries no information about this particular prefix. It is kept
                    # on the log entity as a per-sample baseline instead.
                    losses = [
                        x.loss
                        for x in evidence_log.consistency_list
                        if x.accuracy and x.loss is not None
                    ]
                    evidence_log.evidence_accumulation_loss = float(np.mean(losses)) if losses else None

            except Exception as e:
                print(f"[WARN] generate failed: {e}")

        
        for log in log_list: 
            for i in range(1, len(log.evidence_list)):
                current = log.evidence_list[i]
                previous = log.evidence_list[i - 1]
                current.delta_evidence_self_consistency = current.evidence_accumulation_self_consistency - previous.evidence_accumulation_self_consistency

                # evidence_accumulation_loss is None when no continuation agreed, so
                # there is no per-token NLL to average. Leave the delta undefined
                # rather than crashing this loop, which runs outside any try/except.
                if previous.evidence_accumulation_loss is None or current.evidence_accumulation_loss is None:
                    current.delta_evidence_loss = None
                else:
                    current.delta_evidence_loss = previous.evidence_accumulation_loss - current.evidence_accumulation_loss

        return log_list

    def add_evidence_log_list(self, log: diffusion_decision_model_log_entity) -> diffusion_decision_model_log_entity:
        chain_of_thought = self.get_dataset().chain_of_thought_extraction(log.question, log.completion)
        sentences = re.split(r'(?<=[.!?])\s+', chain_of_thought)

        chunk_size = self.get_chunk_size(sentences)
        for idx, i in enumerate(range(0, len(sentences), chunk_size)):        
            evidence = " ".join(sentences[i:i + chunk_size])
            partial_cot = " ".join(sentences[0:i + chunk_size])
            
            evidence_log = diffusion_decision_model_evidence_log_entity()
            evidence_log.index = idx
            evidence_log.evidence = evidence
            evidence_log.partial_cot = partial_cot
            log.add_evidence_list(evidence_log)
                
        return log

    def get_max_new_tokens(self) -> int:
        return 15000

    def get_chunk_size(self, sentences) -> int:
        if self.number_of_evidence is None: 
            return 1.0
        
        return ceil(len(sentences) / self.number_of_evidence)

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_logger(self, run_number) -> diffusion_decision_model_logger:
        pass

