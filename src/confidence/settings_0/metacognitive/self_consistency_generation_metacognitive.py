from src.logger.self_consistency.self_consistency_log_entity import self_consistency_log_entity
from src.logger.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
from src.confidence.self_consistency_generation import self_consistency_generation
from src.datasets.confidence.metacognitive_dataset import metacognitive_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger


class self_consistency_generation_metacognitive(self_consistency_generation): 

    def __init__(self, modelname):
        super().__init__(modelname)

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
        idx: int = 0
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing Batches", unit="step"):
            batch = test_dataset[i : i + batch_size]

            sample_ID_list = batch['sample_id']
            problem_id_list = batch['problem_id']
            split_list = batch['split']

            correct_prompt_list = batch['correct_prompt']
            correct_target_list = batch['correct_target']
            log_list.extend(self.generate(model, sampling_params, num_sequences, sample_ID_list, problem_id_list, split_list, correct_prompt_list, correct_target_list, idx))
            idx += len(log_list)
            
            incorrect_prompt_list = batch['incorrect_prompt']
            incorrect_target_list = batch['incorrect_target']
            log_list.extend(self.generate(model, sampling_params, num_sequences, sample_ID_list, problem_id_list, split_list, incorrect_prompt_list, incorrect_target_list, idx))
            idx += len(log_list)

        logger = self.create_self_consistency_logger(run_number)
        logger.add_to_buffer_list(log_list)
        logger.write_to_log_file()

    def generate(self, model, sampling_params, num_sequences, sample_ID_list, problem_id_list, split_list, prompt_list, target_list, idx) -> list[self_consistency_log_entity]: 
        log_list: list[self_consistency_log_entity] = []
        try:
            outputs = model.generate(prompt_list, sampling_params)
            for j, output in enumerate(outputs):
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

                idx += 1
                log_list.append(log)    
        except Exception as e:
            print(f"[WARN] generate failed: {e}")
            
        return log_list


    def get_max_new_tokens(self) -> int:
        return 15000

    def get_dataset(self) -> metacognitive_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            self.dataset = metacognitive_dataset(config)
        return self.dataset

    def create_self_consistency_logger(self, run_number) -> self_consistency_inference_logger:
        return self_consistency_inference_logger(log_file_name = f'src/confidence/settings_0/metacognitive/run_{run_number}/self_consistency_metacognitive.csv')

for run_number in range(5,6):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = self_consistency_generation_metacognitive(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    t.generate_self_consistency(run_number = run_number)
    print(f'{'*' * 210}')

