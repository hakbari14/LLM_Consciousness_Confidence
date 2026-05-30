from abc import ABC, abstractmethod
from src.logger.inference.self_consistency.self_consistency_log_entity import self_consistency_log_entity
from src.logger.inference.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from src.utils.enums_class import llm_pipeline_type_enum, iit_layer_type_enum
from confidence.services.iit_calculation_services_entity import self_consistency_log_api_entity, self_consistency_log_res_api_entity, self_consistency_log_detail_api_entity
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import requests
from typing import List
from pydantic import TypeAdapter


class confidence_inference_client(ABC): 

    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name is None:
            raise Exception('model name is required')
        
        self.pipeline_type = llm_pipeline_type_enum.INFERENCE
        self.dataset = None
        config = self.get_dataset().get_config()
        config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iit_calculator_list = None
        self.logger = None

    @torch.inference_mode()
    def calculate_accuracy_confidence(self, batch_size = 25, num_sequences = 4, run_number = 0): 
        _, test_dataset = self.get_dataset().preprocess_dataset()
        log_dict = {}

        print(f'{'*' * 100}  Calculate IIT  {'*' * 100}')
        model = LLM(model=self.model_name, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = num_sequences, 
                top_p= 0.9, 
                top_k=50
            )

        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batches"):
            batch = test_dataset[i : i + batch_size]
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']

            try:
                outputs = model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    log = self_consistency_log_entity(idx, sample_ID, problem_id, split, prompt, target)
                    
                    if output.outputs is None: continue
                    for index in range(num_sequences):
                        response = output.outputs[index]
                        completion = response.text
                        
                        log_detail = self_consistency_log_detail_entity(f'{idx}_{index}')
                        log_detail.set_completion(completion)
                        log_detail.set_token_count(len(response.token_ids))

                        try:
                            final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                            log_detail.set_final_answer(final_answer)
                            log_detail.set_compared_final_answer(compared_final_answer)
                            log_detail.set_accuracy(accuracy)
                        except Exception as e:
                            print(f"[WARN] generate failed: {e}")
                            
                        log.add_consistency_list(log_detail)
                        
                    self.call_calculate_iit_api(log_dict, log)
            except Exception as e:
                print(f"[WARN] generate failed: {e}")
        
        for settings, log_list in log_dict.items():
            logger = self.create_logger(settings, run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()
        
        
    def call_calculate_iit_api(self, log_dict, log : self_consistency_log_entity)-> self_consistency_log_entity:
        url = "http://127.0.0.1:8000/calculate_iit"

        payload = self_consistency_log_api_entity()
        payload.sample_ID = str(log.get_sample_ID())
        payload.split = log.get_split()
        payload.prompt = log.get_prompt()
        payload.target = str(log.get_target())
        payload.problem_id = log.get_problem_id()
        for log_detail in log.get_consistency_list():
            i_log_detail = self_consistency_log_detail_api_entity()
            i_log_detail.index = str(log_detail.get_index())
            i_log_detail.completion = log_detail.get_completion()
            i_log_detail.final_answer = str(log_detail.get_final_answer())
            i_log_detail.compared_final_answer = str(log_detail.get_compared_final_answer())
            i_log_detail.accuracy = log_detail.get_accuracy()
            payload.add_consistency_list(i_log_detail)

        response = requests.post(url, json=payload.model_dump())

        if response.status_code == 200:
            data = response.json()
            i_entity_list = TypeAdapter(list[self_consistency_log_res_api_entity]).validate_python(data)            
            for i_entity in i_entity_list: 
                log = self_consistency_log_entity(i_entity.sample_ID, i_entity.sample_ID, i_entity.problem_id, i_entity.split, i_entity.prompt, i_entity.target)
                log.set_completion(i_entity.completion)
                log.set_token_count(i_entity.token_count)
                log.set_final_answer(i_entity.final_answer)
                log.set_accuracy(i_entity.accuracy)
                log.set_completion_embedding_shape(i_entity.completion_embedding_shape)
                log.set_completion_loss(i_entity.completion_loss)
                log.set_perplexity(i_entity.perplexity)
                log.set_entropy(i_entity.entropy)
                log.set_token_count_for_reduced_dim(i_entity.token_count_for_reduced_dim)
                log.set_reduced_dim(i_entity.reduced_dim)
                log.set_phi_reward(i_entity.phi_reward)
                log.set_phi_reward_raw(i_entity.phi_reward_raw)
                log.set_phi_reward_raw_actual(i_entity.phi_reward_raw_actual)
                log.set_tpm_loss(i_entity.tpm_loss)
                log.set_tpm_entropy(i_entity.tpm_entropy)

                for i_log_detail in i_entity.consistency_list:
                    log_detail = self_consistency_log_detail_entity(i_log_detail.index)
                    log_detail.set_completion(i_log_detail.completion)
                    log_detail.set_final_answer(i_log_detail.final_answer)
                    log_detail.set_compared_final_answer(i_log_detail.compared_final_answer)
                    log_detail.set_token_count(i_log_detail.token_count)
                    log_detail.set_accuracy(i_log_detail.accuracy)
                    log.add_consistency_list(log_detail)
                    
                log_list = log_dict.setdefault(i_entity.ii_calculator_name, [])
                log_list.append(log)
        else:
            print("Error:", response.status_code, response.text)

    def get_layer_type(self):
        return iit_layer_type_enum.SOME

    def get_max_new_tokens(self):
        return 5000

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def create_logger(self, settings, run_number):
        pass
