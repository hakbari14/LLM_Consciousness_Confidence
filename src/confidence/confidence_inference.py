from abc import ABC, abstractmethod
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from src.integrated_information_theory.integrated_information_theory import integrated_information_theory
from src.logger.self_consistency.self_consistency_log_entity import self_consistency_log_entity
from src.logger.confidence.confidence_log_entity import confidence_log_entity
from src.logger.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from src.integrated_information_theory.entity.iit_entity import iit_entity
from src.logger.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from src.logger.confidence.confidence_logger import confidence_logger
from src.utils.llm_representation import llm_representation
from src.utils.enums_class import iit_layer_type_enum
from tqdm import tqdm
from src.integrated_information_theory.intrinsic_information import intrinsic_information
from src.integrated_information_theory.integrated_information_theory import integrated_information_theory
from src.integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from src.integrated_information_theory.integrated_information import integrated_information
from src.integrated_information_theory.config.integrated_information_config import integrated_information_config
from src.utils.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum,ii_phi_type_enum
import pandas as pd
import torch
import gc


class confidence_inference(ABC): 

    def __init__(self, whitebox_modelname, blackbox_modelname):
        self.whitebox_modelname = whitebox_modelname
        self.blackbox_modelname = blackbox_modelname
        
        if self.whitebox_modelname is None:
            raise Exception('whitebox modelname is required')
        if self.blackbox_modelname is None:
            raise Exception('blackbox modelname is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.representation = llm_representation()
        self.iit_calculator_list = None

        self.whitebox_model = None
        self.whitebox_tokenizer = None
        self.blackbox_model = None
        self.blackbox_tokenizer = None

    @torch.inference_mode()
    def calculate_confidence_blackbox_model(self, run_number = 0): 
        print(f'{'*' * 90}  Calculate BlackBox Confidence Run Number {run_number} {'*' * 90}')
        log_list: list[self_consistency_log_entity] = self.load_logs_list(run_number)
        log_dict = {}
        model, tokenizer = self.get_blackbox_model()
        for log in tqdm(log_list, desc="Integrated Information Processing", unit="step"): 
            try:
                log, origin_entity_list = self.load_embedding(log, model, tokenizer)
                for iit_calculator in self.get_iit_calculator_list():
                    entity_list = iit_entity.clone_list(origin_entity_list)
                    entity = iit_calculator.calculate_prompt(entity_list)
                    if entity is None: continue
                        
                    new_log = confidence_log_entity()
                    new_log.ID = log.ID
                    new_log.sample_ID = log.sample_ID
                    new_log.problem_id = log.problem_id
                    new_log.split = log.split
                    new_log.prompt = log.prompt
                    new_log.target = log.target

                    new_log.iit_calculator_name = iit_calculator.get_config().name

                    new_log.completion_embedding_shape = entity.completion_embedding_shape
                    new_log.completion = entity.completion

                    new_log.phi_reward = entity.iit_reward
                    new_log.phi_reward_raw = entity.iit_reward_raw
                    new_log.phi_reward_raw_actual = entity.iit_reward_raw_actual
                    new_log.tpm_loss = entity.tpm_loss
                    new_log.tpm_entropy = entity.tpm_entropy

                    log_detail_list = list(filter(lambda x: x.index == entity.key, log.consistency_list))
                    new_log_detail = log_detail_list[0]
                    new_log.token_count = new_log_detail.token_count
                    new_log.final_answer = new_log_detail.final_answer
                    new_log.accuracy = new_log_detail.accuracy
                    
                    log_list = log_dict.setdefault(iit_calculator.get_config().name, [])
                    log_list.append(new_log)
                
                    del entity_list, entity
                    gc.collect()
                    torch.cuda.empty_cache()

                del origin_entity_list
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[WARN] generate failed: {e}")
            

        for settings, log_list in log_dict.items():
            logger = self.create_confidence_logger(settings, run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()


    @torch.inference_mode()
    def calculate_confidence_whitebox_model(self, run_number = 0): 
        print(f'{'*' * 90}  Calculate whiteBox Confidence Run Number {run_number} {'*' * 90}')
        model, tokenizer = self.get_whitebox_model()
        for iit_calculator in self.get_iit_calculator_list():
            print(f'{'*' * 90}  Calculate WhiteBox Confidence {iit_calculator.get_config().name} {'*' * 90}')

            try:
                logger = self.create_confidence_logger(iit_calculator.get_config().name, run_number)
                df = pd.read_csv(logger.get_log_file_name())
                for index, row in enumerate(tqdm(df.iterrows(), desc="BlackBox Confidence Processing", unit="step")): 
                    try:
                        completion = df.loc[index, "Completion"]
                        loss, sum_probs, avg_probs, avg_entropy = self.representation.calculate_whitebox_confidence(completion, model, tokenizer)

                        df.at[index, "Completion_Loss"] = loss
                        df.at[index, "Sequence_Probability"] = sum_probs
                        df.at[index, "Length_Normalized_Sequence_Probability"] = avg_probs
                        df.at[index, "Entropy"] = avg_entropy
                    except Exception as e:
                        print(f"Exception : {e}")

                df.to_csv(logger.get_log_file_name(), index=False)            
            except Exception as e:
                print(f"Exception : {e}")

    @torch.inference_mode()
    def load_embedding(self, log: self_consistency_log_entity, model, tokenizer) -> tuple[self_consistency_log_entity, list[iit_entity]]: 
        entity_list = []
        refine_prompt = self.representation.clean_prompt_for_phi(log.prompt)
        prompt_emb, _, _ = self.representation.extract_representation(refine_prompt, model, tokenizer, self.get_layer_type())
        for log_detail in log.consistency_list:     
            try:
                entity = iit_entity()
                entity.key = log_detail.index
                entity.prompt = log.prompt
                entity.prompt_embedding = prompt_emb
                entity.completion = log_detail.completion
                if log_detail.completion is not None:
                    completion_emb, _, _ = self.representation.extract_representation(entity.completion, model, tokenizer, self.get_layer_type())
                    entity.set_completion_embedding_and_shape(completion_emb)
                    entity.token_count = completion_emb.shape[1]
                
                if entity.is_calcutable():
                    entity_list.append(entity)

            except Exception as e:
                entity_list.append(entity)
                print(f"[WARN] Load Embedding: {e}")
       
        return log, entity_list

    def load_logs_list(self, run_number = 0) -> list[self_consistency_log_entity]:
        logger = self.create_self_consistency_logger(run_number)

        df_logs = pd.read_csv(logger.get_log_file_name())
        df_samples = pd.read_csv(logger.get_samples_log_file_name())

        log_list: list[self_consistency_log_entity] = []
        for _, a_row in df_logs.iterrows():
            log = self_consistency_log_entity()
            log.ID = a_row["ID"]
            log.sample_ID = a_row["Sample_ID"]
            log.problem_id = a_row["problem_id"]
            log.split = a_row["Split"]
            log.prompt = a_row["Prompt"]
            log.target = a_row["Target"]

            b_subset = df_samples[df_samples["Sample_ID"] == log.sample_ID]
            for _, b_row in b_subset.iterrows():
                log_detail = self_consistency_log_detail_entity()
                log_detail.index = b_row["Index"]
                log_detail.completion = b_row["Completion"]
                log_detail.final_answer = b_row["Final_Answer"]
                log_detail.compared_final_answer = b_row["Compared_Final_Answer"]
                log_detail.token_count = b_row["Token_Count"]
                log_detail.accuracy = b_row["Accuracy"]
               
                log.add_consistency_list(log_detail)
            
            log_list.append(log)

        return log_list

    def get_whitebox_model(self):
        if self.whitebox_model == None: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
                bnb_4bit_use_double_quant = False,
            )
            model = AutoModelForCausalLM.from_pretrained(self.whitebox_modelname, quantization_config = bnb_config)
            model.config.use_cache = True
            model.config.pretraining_tp = 1        
            self.whitebox_model = model
            self.whitebox_tokenizer = AutoTokenizer.from_pretrained(self.whitebox_modelname)

        return self.whitebox_model, self.whitebox_tokenizer 

    def get_blackbox_model(self):
        if self.blackbox_model == None: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
                bnb_4bit_use_double_quant = False,
            )
            model = AutoModelForCausalLM.from_pretrained(self.blackbox_modelname, quantization_config = bnb_config)
            model.config.use_cache = True
            model.config.pretraining_tp = 1        
            self.blackbox_model = model
            self.blackbox_tokenizer = AutoTokenizer.from_pretrained(self.blackbox_modelname)

        return self.blackbox_model, self.blackbox_tokenizer 

    def get_iit_calculator_list(self) -> list[integrated_information_theory]:
        if self.iit_calculator_list is None: 
            iit_calculator_list: list[integrated_information_theory] = []

            config = intrinsic_information_config()
            config.name = 'Settings_46'
            config.calculation_type = ii_calculation_type_enum.SUM
            config.reduced_dimension = 5
            config.tpm_creation_type = tpm_creation_type_enum.PROMPT
            config.layer_type = iit_layer_type_enum.SOME
            config.threashold_type = iit_threashold_type_enum.AVERAGE
            config.last_layer_computation_type = last_layer_computation_type_enum.EXP
            config.last_layer_computation_param = 0.09
            iit_calculator_list.append(intrinsic_information(config)) 

            config = integrated_information_config()
            config.name = 'Settings_64' 
            config.phi_type = ii_phi_type_enum.SYSTEM_PHI
            config.reduced_dimension = 4
            config.layer_type = iit_layer_type_enum.SOME
            config.threashold_type = iit_threashold_type_enum.AVERAGE
            config.tpm_creation_type = tpm_creation_type_enum.PROMPT
            config.last_layer_computation_type = last_layer_computation_type_enum.EXP
            config.last_layer_computation_param = 0.09
            iit_calculator_list.append(integrated_information(config)) 

            config = integrated_information_config()
            config.name = 'Settings_65'
            config.phi_type = ii_phi_type_enum.BIG_PHI
            config.reduced_dimension = 4
            config.layer_type = iit_layer_type_enum.SOME
            config.threashold_type = iit_threashold_type_enum.AVERAGE
            config.tpm_creation_type = tpm_creation_type_enum.PROMPT
            config.last_layer_computation_type = last_layer_computation_type_enum.EXP
            config.last_layer_computation_param = 0.09
            iit_calculator_list.append(integrated_information(config)) 
            
            self.iit_calculator_list = iit_calculator_list
        
        return self.iit_calculator_list

    def get_layer_type(self):
        return iit_layer_type_enum.SOME

    @abstractmethod
    def create_self_consistency_logger(self, run_number) -> self_consistency_inference_logger:
        pass

    @abstractmethod
    def create_confidence_logger(self, settings, run_number) -> confidence_logger:
        pass
