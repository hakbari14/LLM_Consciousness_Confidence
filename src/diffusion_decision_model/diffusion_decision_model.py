from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity
from src.datasets.dataset_handler import dataset_handler
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger
import torch
import re
from math import ceil

class diffusion_decision_model(ABC): 

    def __init__(self, modelname) -> None:
        self.modelname = modelname
        
        if self.modelname is None:
            raise Exception('modelname is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)

    def run(self, from_run_number: int , to_run_number: int) -> None:
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            log_list: list[diffusion_decision_model_log_entity] = self.generate_response(run_number = run_number)
            log_list = self.generate_self_consistency(log_list)
            
            logger = self.create_logger(run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()
            
            print(f"{'*' * 210}")

    @torch.inference_mode()
    def generate_response(self, batch_size = 128, run_number = 0) -> list[diffusion_decision_model_log_entity]: 
        print(f"{'*' * 100}  Generate Response {'*' * 100}")

        _, test_dataset = self.get_dataset().preprocess_dataset()
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.2, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
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
                top_k=50
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

            prompt_list = self.generate_model_prompt_chain_of_thought(batch_question_list, batch_partial_cot_list)
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
                    log_detail.completion = completion
                    log_detail.token_count = len(response.token_ids)
                    log_detail.original_final_answer = original_final_answer
                    
                    try:
                        final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), original_final_answer)
                        log_detail.final_answer = final_answer
                        log_detail.compared_final_answer = compared_final_answer
                        log_detail.accuracy = accuracy
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")
                        
                    evidence_log.add_consistency_list(log_detail)
                    
                true_count = sum(x.accuracy for x in evidence_log.consistency_list)
                evidence_log.evidence_accumulation = true_count / len(evidence_log.consistency_list)
        
        for log in log_list: 
            for i in range(1, len(log.evidence_list)):
                current = log.evidence_list[i]
                previous = log.evidence_list[i - 1]
                current.delta_evidence = current.evidence_accumulation - previous.evidence_accumulation
        
        return log_list

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

    def add_evidence_log_list(self, log: diffusion_decision_model_log_entity) -> diffusion_decision_model_log_entity:
        chain_of_thought = log.completion
        pos = chain_of_thought.find(log.question)
        if pos != -1:
            chain_of_thought = chain_of_thought[pos + len(log.question):]
            
        sentences = re.split(r'(?<=[.!?])\s+', chain_of_thought)
        chunk_size = ceil(len(sentences) / self.get_number_of_evidence())
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

    def get_number_of_evidence(self) -> int:
        return 20

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_logger(self, run_number) -> diffusion_decision_model_logger:
        pass

