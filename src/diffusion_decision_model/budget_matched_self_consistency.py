from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_log_entity import budget_matched_self_consistency_log_entity
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_log_detail_entity import budget_matched_self_consistency_log_detail_entity
from src.datasets.dataset_handler import dataset_handler
import torch
import traceback
import logging

logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)        

class budget_matched_self_consistency(ABC): 

    def __init__(self, modelname: str) -> None:
        self.modelname = modelname

        if self.modelname is None:
            raise Exception('modelname is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None

    def run(self, from_run_number: int , to_run_number: int, number_of_outputs: int = 125) -> None:
        self.model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        
        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")

            _, test_dataset = self.get_dataset().preprocess_dataset()
            sampling_params = SamplingParams(
                    max_tokens=self.get_max_new_tokens(), 
                    temperature=1.0, 
                    n = number_of_outputs, 
                    top_p= 0.9, 
                    top_k=50,
                )
            
            log_list: list[budget_matched_self_consistency_log_entity] = []
            for i in tqdm(range(0, len(test_dataset)), desc="Processing", unit="step"):
                x = test_dataset[i]
                log = budget_matched_self_consistency_log_entity()
                log.ID = i
                log.x = x
                log.sample_ID = x['sample_id']
                log.problem_id = x['problem_id']
                log.split = x['split']
                log.question = x['question']
                log.prompt = x['prompt']
                log.target = x['target']
                
                try:
                    outputs = self.model.generate([].append(log.prompt), sampling_params)
                    for j, output in enumerate(outputs):
                        idx = i + j

                        if output.outputs is None: continue
                        response = output.outputs[0]
                        log_detail = budget_matched_self_consistency_log_detail_entity()
                        log_detail = idx
                        log_detail.completion = response.text
                        
                        try:
                            final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(log.prompt, str(log_detail.completion), log.target)
                            if final_answer is None or compared_final_answer is None: continue
                            log_detail.final_answer = final_answer
                            log_detail.compared_final_answer = compared_final_answer
                            log_detail.accuracy = accuracy
                        except Exception as e:
                            logging.exception("An exception occurred")                        
                            print(f"[WARN]: {e}")
                            traceback.print_exc()          
                                          
                        log.add_consistency_list(log_detail)
                except Exception as e:
                    logging.exception("An exception occurred")                        
                    print(f"[WARN]: {e}")
                    traceback.print_exc()                        
                

                log.compared_final_answer_nv5, log.accuracy_nv5, log.self_consistency_nv5 = self.calculate_mjority_vote(log.consistency_list[:25])                     
                log.compared_final_answer_nv10, log.accuracy_nv10, log.self_consistency_nv10 = self.calculate_mjority_vote(log.consistency_list[:50])                     
                log.compared_final_answer_nv15, log.accuracy_nv15, log.self_consistency_nv15 = self.calculate_mjority_vote(log.consistency_list[:75])                     
                log.compared_final_answer_nv20, log.accuracy_nv20, log.self_consistency_nv20 = self.calculate_mjority_vote(log.consistency_list[:100])                     
                log.compared_final_answer_nv25, log.accuracy_nv25, log.self_consistency_nv25 = self.calculate_mjority_vote(log.consistency_list[:125])                     
                log_list.append(log)    
            
            logger = self.create_logger(run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()
            
            print(f"{'*' * 210}")

    def calculate_mjority_vote(consistency_list: list[budget_matched_self_consistency_log_detail_entity]) -> tuple[str, bool, float]: 
        vote_count = {}
        for log_detail in consistency_list:
            if log_detail.compared_final_answer == None: continue
            if type(log_detail.compared_final_answer) == str and len(log_detail.compared_final_answer) == 0: continue
            vote_count[log_detail.compared_final_answer] = vote_count.get(log_detail.compared_final_answer, 0) + 1

        if len(vote_count.items()) > 0: 
            max_vote = max(vote_count.items(), key=lambda x: x[1])
            compared_final_answer = max_vote[0]
            log_detail_list = list(filter(lambda x: x.compared_final_answer == compared_final_answer, consistency_list))
            if len(log_detail_list) >= 1: 
                log_detail = log_detail_list[0]
                return log_detail.compared_final_answer, log_detail.accuracy, log_detail_list / len(consistency_list)
        else : 
                if len(consistency_list) > 0:
                    log_detail = consistency_list[0]
                    return log_detail.compared_final_answer, log_detail.accuracy, 1.0 / len(consistency_list)
                
        return None, None, None

    def get_max_new_tokens(self) -> int:
        return 15000

    def get_modelname_dir(self) -> str:
        return self.modelname.replace('/', '-').lower()

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        pass
