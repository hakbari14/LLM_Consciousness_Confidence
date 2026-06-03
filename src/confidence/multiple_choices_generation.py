from abc import ABC, abstractmethod
from src.logger.multiple_choices.multiple_choices_inference_logger import multiple_choices_inference_logger
from src.logger.multiple_choices.multiple_choices_log_entity import multiple_choices_log_entity
from src.logger.multiple_choices.multiple_choices_log_detail_entity import multiple_choices_log_detail_entity
from src.datasets.dataset_handler import dataset_handler
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch


class multiple_choices_generation(ABC): 

    def __init__(self, modelname):
        self.modelname = modelname
        
        if self.modelname is None:
            raise Exception('modelname is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None


    @torch.inference_mode()
    def generate_response(self, batch_size = 12, num_choice_permutations = 10, run_number = 0): 
        _, test_dataset = self.get_dataset().preprocess_dataset()

        print(f'{'*' * 90}  Generate Self Consistency Run Number {run_number} {'*' * 90}')
        model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        log_list: list[multiple_choices_log_entity] = []
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing Batches", unit="step"):
            prompt_list: list[str] = []
            target_list: list[str] = []
            for j in range(i, min((i + batch_size), len(test_dataset))):
                x = test_dataset[j]
                permutation_list: list[dict] = self.get_dataset().generate_model_prompt_permutation(x, num_choice_permutations)
                for permutation in permutation_list:
                    prompt_list.append(permutation['prompt'])
                    target_list.append(permutation['target'])
                    
            outputs = model.generate(prompt_list, sampling_params)
            idx: int = 0
            for j in range(i, min((i + batch_size), len(test_dataset))):
                x = test_dataset[j]
                sample_ID = x['sample_id']
                split = x['split']
                prompt = x['prompt']
                target = x['target']
                problem_id = x['problem_id']

                log = multiple_choices_log_entity()
                log.ID = j
                log.sample_ID = sample_ID
                log.split = split
                log.prompt = prompt
                log.target = target
                log.problem_id = problem_id
                
                for k in range(0, num_choice_permutations):
                    response = outputs[idx].outputs[0]
                    completion = response.text

                    prompt_detail = prompt_list[idx]
                    target_detail = target_list[idx]
                    
                    log_detail = multiple_choices_log_detail_entity()
                    log_detail.index = f'{j}_{k}'
                    log_detail.prompt = prompt_detail
                    log_detail.target = target_detail
                    log_detail.completion = completion
                    log_detail.token_count = len(response.token_ids)

                    try:
                        final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt_detail, str(completion), target_detail)
                        log_detail.final_answer = final_answer
                        log_detail.compared_final_answer = compared_final_answer
                        log_detail.accuracy = accuracy
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")
                        
                    log.add_permutation_list(log_detail)
                    idx += 1
                
                log.calculate_confidence()
                log_list.append(log)    

        logger = self.create_multiple_choices_logger(run_number)
        logger.add_to_buffer_list(log_list)
        logger.write_to_log_file()


    def get_max_new_tokens(self) -> int:
        return 5000

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_multiple_choices_logger(self, run_number) -> multiple_choices_inference_logger:
        pass

