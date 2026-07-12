from abc import ABC, abstractmethod
from src.logger.llm_response.llm_response_log_entity import llm_response_log_entity
from src.logger.llm_response.llm_response_inference_logger import llm_response_inference_logger
from src.datasets.dataset_handler import dataset_handler
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.utils.enums_class import confidence_type_enum
import torch
import pandas as pd
import gc

class llm_generation(ABC): 

    def __init__(self, modelname):
        self.modelname = modelname
        
        if self.modelname is None:
            raise Exception('modelname is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)
        

    def run(self, from_run_number: int , to_run_number: int, confidence_type : confidence_type_enum) -> None:
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            self.generate_response(run_number = run_number)
            self.generate_confidence(confidence_type=confidence_type, run_number = run_number)
            self.generate_self_criteria(run_number = run_number)
            self.generate_confidence_self_criteria(confidence_type=confidence_type, run_number = run_number)
            self.generate_self_criteria_with_solution(run_number = run_number)
            self.generate_confidence_self_criteria_with_solution(confidence_type=confidence_type, run_number = run_number)
            
            print(f"{'*' * 210}")

    @torch.inference_mode()
    def generate_response(self, batch_size = 128, run_number = 0): 
        _, test_dataset = self.get_dataset().preprocess_dataset()

        print(f"{'*' * 90}  Generate Response Run Number {run_number} {'*' * 90}")
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        log_list: list[llm_response_log_entity] = []
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing Batches", unit="step"):
            batch = test_dataset[i : i + batch_size]
            question_list = batch['question']
            answer_list = batch['answer']
            prompt_list = batch['prompt']
            sample_ID_list = batch['sample_id']
            split_list = batch['split']
            target_list = batch['target']
            problem_id_list = batch['problem_id']

            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    idx = i + j
                    question = question_list[j]
                    answer = answer_list[j]
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    
                    log = llm_response_log_entity()
                    log.ID = idx
                    log.sample_ID = sample_ID
                    log.problem_id = problem_id
                    log.split = split
                    log.question = question
                    log.answer = answer
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
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")
                        
                    log_list.append(log)    
            except Exception as e:
                print(f"[WARN] generate failed: {e}")

        logger = self.create_llm_response_logger(run_number)
        logger.add_to_buffer_list(log_list)
        logger.write_to_log_file()
        

    @torch.inference_mode()
    def generate_confidence(self, confidence_type: confidence_type_enum, batch_size = 128, run_number = 0): 
        logger = self.create_llm_response_logger(run_number)
        df = pd.read_csv(logger.get_log_file_name())
        
        if 'Confidence_Prompt' not in df.columns:
            df['Confidence_Prompt'] = pd.Series(dtype="string")
        if 'Confidence_Completion' not in df.columns:
            df['Confidence_Completion'] = pd.Series(dtype="string")
        if 'Confidence_Level' not in df.columns:
            df['Confidence_Level'] = pd.Series(dtype="float64")
        
        print(f"{'*' * 90}  Generate Confidence Run Number {run_number} {'*' * 90}")
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="step"):
            batch = df[i : i + batch_size]
            question_list = batch['Question']
            answer_list = batch['Answer']
            prompt_list = self.generate_model_prompt_confidence(confidence_type, question_list, answer_list)

            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    df.at[idx, "Confidence_Prompt"] = prompt
                    
                    if output.outputs is None: continue
                    
                    response = output.outputs[0]
                    confidence_completion = response.text
                    df.at[idx, "Confidence_Completion"] = confidence_completion
                    
                    try:
                        confidence_level = self.extract_confidence(confidence_type, confidence_completion)
                        df.at[idx, "Confidence_Level"] = confidence_level
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")

            except Exception as e:
                print(f"[WARN] generate failed: {e}")
                
        df.to_csv(logger.get_log_file_name(), index=False)

    @torch.inference_mode()
    def generate_self_criteria(self, batch_size = 128, run_number = 0): 
        logger = self.create_llm_response_logger(run_number)
        df = pd.read_csv(logger.get_log_file_name())
        
        if 'Prompt_Self_Criteria' not in df.columns:
            df['Prompt_Self_Criteria'] = pd.Series(dtype="string")
        if 'Self_Criteria_Completion' not in df.columns:
            df['Self_Criteria_Completion'] = pd.Series(dtype="string")
        if 'Self_Criteria' not in df.columns:
            df['Self_Criteria'] = pd.Series(dtype="string")
        
        print(f"{'*' * 90}  Generate Self Criteria Run Number {run_number} {'*' * 90}")
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="step"):
            batch = df[i : i + batch_size]
            question_list = batch['Question']
            answer_list = batch['Answer']
            accuracy_list = batch['Accuracy']
            prompt_list = self.generate_model_prompt_self_criteria(question_list, answer_list, accuracy_list)

            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    df.at[idx, "Prompt_Self_Criteria"] = prompt
                    
                    if output.outputs is None: continue
                    
                    response = output.outputs[0]
                    self_criteria_completion = response.text
                    df.at[idx, "Self_Criteria_Completion"] = self_criteria_completion
                    
                    try:
                        self_criteria = self.extract_self_criteria(self_criteria_completion)
                        df.at[idx, "Self_Criteria"] = self_criteria
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")

            except Exception as e:
                print(f"[WARN] generate failed: {e}")
                
        df.to_csv(logger.get_log_file_name(), index=False)            


    @torch.inference_mode()
    def generate_confidence_self_criteria(self, confidence_type: confidence_type_enum, batch_size = 128, run_number = 0): 
        logger = self.create_llm_response_logger(run_number)
        df = pd.read_csv(logger.get_log_file_name())
        
        if 'Confidence_Prompt_Self_Criteria' not in df.columns:
            df['Confidence_Prompt_Self_Criteria'] = pd.Series(dtype="string")
        if 'Confidence_Completion_Self_Criteria' not in df.columns:
            df['Confidence_Completion_Self_Criteria'] = pd.Series(dtype="string")
        if 'Confidence_Level_Self_Criteria' not in df.columns:
            df['Confidence_Level_Self_Criteria'] = pd.Series(dtype="float64")
        
        print(f"{'*' * 90}  Generate Confidence with Self_Criteria Run Number {run_number} {'*' * 90}")
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="step"):
            batch = df[i : i + batch_size]
            question_list = batch['Question']
            answer_list = batch['Answer']
            self_criteria_list = batch['Self_Criteria']
            prompt_list = self.generate_model_prompt_self_criteria_confidence(confidence_type, question_list, answer_list, self_criteria_list)

            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    df.at[idx, "Confidence_Prompt_Self_Criteria"] = prompt
                    
                    if output.outputs is None: continue
                    
                    response = output.outputs[0]
                    confidence_completion = response.text
                    df.at[idx, "Confidence_Completion_Self_Criteria"] = confidence_completion
                    
                    try:
                        confidence_level = self.extract_confidence(confidence_type, confidence_completion)
                        df.at[idx, "Confidence_Level_Self_Criteria"] = confidence_level
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")

            except Exception as e:
                print(f"[WARN] generate failed: {e}")
                
        df.to_csv(logger.get_log_file_name(), index=False)            

    @torch.inference_mode()
    def generate_self_criteria_with_solution(self, batch_size = 128, run_number = 0): 
        logger = self.create_llm_response_logger(run_number)
        df = pd.read_csv(logger.get_log_file_name())
        
        if 'Prompt_Self_Criteria_With_Solution' not in df.columns:
            df['Prompt_Self_Criteria_With_Solution'] = pd.Series(dtype="string")
        if 'Self_Criteria_Completion_With_Solution' not in df.columns:
            df['Self_Criteria_Completion_With_Solution'] = pd.Series(dtype="string")
        if 'Self_Criteria_With_Solution' not in df.columns:
            df['Self_Criteria_With_Solution'] = pd.Series(dtype="string")
        
        print(f"{'*' * 90}  Generate Self Criteria With Solution Run Number {run_number} {'*' * 90}")
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="step"):
            batch = df[i : i + batch_size]
            question_list = batch['Question']
            answer_list = batch['Answer']
            completion_list = batch['Completion']
            accuracy_list = batch['Accuracy']
            prompt_list = self.generate_model_prompt_self_criteria_with_solution(question_list, answer_list, completion_list, accuracy_list)

            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    df.at[idx, "Prompt_Self_Criteria_With_Solution"] = prompt
                    
                    if output.outputs is None: continue
                    
                    response = output.outputs[0]
                    self_criteria_completion = response.text
                    df.at[idx, "Self_Criteria_Completion_With_Solution"] = self_criteria_completion
                    
                    try:
                        self_criteria = self.extract_self_criteria(self_criteria_completion)
                        df.at[idx, "Self_Criteria_With_Solution"] = self_criteria
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")

            except Exception as e:
                print(f"[WARN] generate failed: {e}")
                
        df.to_csv(logger.get_log_file_name(), index=False)            

    @torch.inference_mode()
    def generate_confidence_self_criteria_with_solution(self, confidence_type: confidence_type_enum, batch_size = 128, run_number = 0): 
        logger = self.create_llm_response_logger(run_number)
        df = pd.read_csv(logger.get_log_file_name())
        
        if 'Confidence_Prompt_Self_Criteria_With_Solution' not in df.columns:
            df['Confidence_Prompt_Self_Criteria_With_Solution'] = pd.Series(dtype="string")
        if 'Confidence_Completion_Self_Criteria_With_Solution' not in df.columns:
            df['Confidence_Completion_Self_Criteria_With_Solution'] = pd.Series(dtype="string")
        if 'Confidence_Level_Self_Criteria_With_Solution' not in df.columns:
            df['Confidence_Level_Self_Criteria_With_Solution'] = pd.Series(dtype="float64")
        
        print(f"{'*' * 90}  Generate Confidence with Self_Criteria With Solution Run Number {run_number} {'*' * 90}")
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="step"):
            batch = df[i : i + batch_size]
            question_list = batch['Question']
            answer_list = batch['Answer']
            completion_list = batch['Completion']
            self_criteria_list = batch['Self_Criteria']
            prompt_list = self.generate_model_prompt_self_criteria_with_solution_confidence(confidence_type, question_list, answer_list, completion_list, self_criteria_list)

            try:
                outputs = self.model.generate(prompt_list, sampling_params)
                for j, output in enumerate(outputs):
                    idx = i + j
                    prompt = prompt_list[j]
                    df.at[idx, "Confidence_Prompt_Self_Criteria_With_Solution"] = prompt
                    
                    if output.outputs is None: continue
                    
                    response = output.outputs[0]
                    confidence_completion = response.text
                    df.at[idx, "Confidence_Completion_Self_Criteria_With_Solution"] = confidence_completion
                    
                    try:
                        confidence_level = self.extract_confidence(confidence_type, confidence_completion)
                        df.at[idx, "Confidence_Level_Self_Criteria_With_Solution"] = confidence_level
                    except Exception as e:
                        print(f"[WARN] generate failed: {e}")

            except Exception as e:
                print(f"[WARN] generate failed: {e}")
                
        df.to_csv(logger.get_log_file_name(), index=False)            

    @abstractmethod
    def generate_model_prompt_confidence(self, confidence_type: confidence_type_enum, question_list, answer_list) -> list[str]:
        pass

    @abstractmethod
    def generate_model_prompt_self_criteria(self, question_list, answer_list, accuracy_list) -> list[str]:
        pass

    @abstractmethod
    def generate_model_prompt_self_criteria_confidence(self, confidence_type: confidence_type_enum, question_list, answer_list, self_criteria_list) -> list[str]:
        pass

    @abstractmethod
    def generate_model_prompt_self_criteria_with_solution(self, question_list, answer_list, accuracy_list) -> list[str]:
        pass

    @abstractmethod
    def generate_model_prompt_self_criteria_with_solution_confidence(self, confidence_type: confidence_type_enum, question_list, answer_list, completion_list, self_criteria_list) -> list[str]:
        pass

    @abstractmethod
    def extract_confidence(self, confidence_type: confidence_type_enum, solution) -> float:
        pass

    @abstractmethod
    def extract_self_criteria(self, self_criteria_completion) -> str:
        pass

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_llm_response_logger(self, run_number) -> llm_response_inference_logger:
        pass

    def clean_gpu(self, model) -> None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def get_max_new_tokens(self) -> int:
        return 5000
