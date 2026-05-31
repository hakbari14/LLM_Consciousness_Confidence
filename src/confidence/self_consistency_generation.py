from abc import ABC, abstractmethod
from src.logger.self_consistency.self_consistency_log_entity import self_consistency_log_entity
from src.logger.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from src.logger.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch


class self_consistency_generation(ABC): 

    def __init__(self, modelname):
        self.modelname = modelname
        
        if self.modelname is None:
            raise Exception('modelname is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None


    @torch.inference_mode()
    def generate_self_consistency(self, batch_size = 25, num_sequences = 5, run_number = 0): 
        _, test_dataset = self.get_dataset().preprocess_dataset()

        print(f'{'*' * 90}  Generate Self Consistency Run Number {run_number} {'*' * 90}')
        model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = num_sequences, 
                top_p= 0.9, 
                top_k=50
            )
        
        log_list: list[self_consistency_log_entity] = []
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing atches", unit="step"):
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
                    
                    log = self_consistency_log_entity()
                    log.ID = idx
                    log.sample_ID = sample_ID
                    log.problem_id = problem_id
                    log.split = split
                    log.prompt = prompt
                    log.target = target
                    
                    if output.outputs is None: continue
                    for index in range(num_sequences):
                        response = output.outputs[index]
                        completion = response.text
                        
                        log_detail = self_consistency_log_detail_entity()
                        log_detail.index = f'{idx}_{index}'
                        log_detail.completion = completion
                        log_detail.token_count = len(response.token_ids)

                        try:
                            final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                            log_detail.final_answer = final_answer
                            log_detail.compared_final_answer = compared_final_answer
                            log_detail.accuracy = accuracy
                        except Exception as e:
                            print(f"[WARN] generate failed: {e}")
                            
                        log.add_consistency_list(log_detail)
                        
                    log_list.append(log)    
            except Exception as e:
                print(f"[WARN] generate failed: {e}")

        logger = self.create_self_consistency_logger(run_number)
        logger.add_to_buffer_list(log_list)
        logger.write_to_log_file()


    def get_max_new_tokens(self):
        return 5000

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def create_self_consistency_logger(self, run_number) -> self_consistency_inference_logger:
        pass

