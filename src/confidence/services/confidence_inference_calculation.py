from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from confidence.services.iit_calculation_services_entity import self_consistency_log_api_entity, self_consistency_log_detail_api_entity, self_consistency_log_res_api_entity
from src.logger.inference.self_consistency.self_consistency_log_entity import self_consistency_log_entity
from src.logger.inference.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from src.integrated_information_theory.entity.iit_entity import iit_entity
from src.utils.llm_representation import llm_representation
from src.utils.enums_class import llm_pipeline_type_enum, iit_layer_type_enum
from src.utils.utility import my_utils
from src.integrated_information_theory.intrinsic_information import intrinsic_information
from src.integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from src.integrated_information_theory.integrated_information import integrated_information
from src.integrated_information_theory.config.integrated_information_config import integrated_information_config
from src.utils.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum,ii_phi_type_enum
from typing import List
import torch
import gc


class confidence_inference_calculation(): 

    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name is None:
            raise Exception('model name is required')
        
        self.representation = llm_representation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.iit_calculator_list = None

    @torch.inference_mode()
    def calculate_iit(self, log: self_consistency_log_api_entity) -> List[self_consistency_log_res_api_entity]: 
        log, origin_entity_list = self.load_embedding_and_loss(log)
        response_list = []
        for iit_calculator in self.get_iit_calculator_list():
            entity_list = iit_entity.clone_list(origin_entity_list)
            entity = iit_calculator.calculate_prompt(entity_list)
            if entity is None: continue
                
            new_log = self_consistency_log_res_api_entity()
            new_log.ii_calculator_name = iit_calculator.get_config().get_name()
            new_log.sample_ID = log.sample_ID
            new_log.problem_id = log.problem_id
            new_log.split = log.split
            new_log.prompt = log.prompt
            new_log.target = log.target
            for log_detail in log.consistency_list:
                new_log_detail = self_consistency_log_detail_api_entity()
                new_log_detail.index = log_detail.index
                new_log_detail.completion = log_detail.completion
                new_log_detail.final_answer = log_detail.final_answer
                new_log_detail.compared_final_answer = log_detail.compared_final_answer
                new_log_detail.accuracy = log_detail.accuracy
                new_log.add_consistency_list(new_log_detail)

            new_log.token_count = entity.get_token_count()
            new_log.reduced_dim = entity.get_reduced_dim()
            new_log.phi_reward = entity.get_iit_reward()
            new_log.phi_reward_raw = entity.get_iit_reward_raw()
            new_log.phi_reward_raw_actual = entity.get_iit_reward_raw_actual()
            new_log.tpm_loss = entity.get_tpm_loss()
            new_log.tpm_entropy = entity.get_tpm_entropy()
            new_log.completion_loss = my_utils.tensor_tostring(entity.get_completion_loss())
            new_log.perplexity = my_utils.calculate_perplexity(entity.get_completion_loss())
            new_log.entropy = entity.get_completion_entropy()
            new_log.completion_embedding_shape = entity.get_completion_embedding_shape()
            new_log.completion = entity.get_completion()
            new_log.token_count_for_reduced_dim = entity.get_token_count_for_reduced_dim()

            log_detail_list = list(filter(lambda x: x.index == entity.get_key(), new_log.consistency_list))
            new_log_detail = log_detail_list[0]
            new_log.final_answer = new_log_detail.final_answer
            new_log.accuracy = new_log_detail.accuracy
            response_list.append(new_log)
            del entity_list, entity
            gc.collect()
            torch.cuda.empty_cache()

        del origin_entity_list
        gc.collect()
        torch.cuda.empty_cache()
        return response_list

    @torch.inference_mode()
    def load_embedding_and_loss(self, log: self_consistency_log_api_entity) -> tuple[self_consistency_log_api_entity, list[iit_entity]]: 
        entity_list: list[iit_entity] = []
        refine_prompt = self.representation.clean_prompt_for_phi(log.prompt)
        prompt_emb, _, _ = self.representation.extract_representation(refine_prompt, self.get_model(), self.tokenizer, iit_layer_type_enum.SOME)
        for log_detail in log.consistency_list:     
            try:
                entity = iit_entity(key = log_detail.index)
                entity.set_promptID(log.sample_ID)
                entity.set_prompt(log.prompt)
                entity.set_prompt_embedding(prompt_emb)
                entity.set_completion(log_detail.completion)
                if log_detail.completion is not None:
                    completion_emb, completion_loss, entropy = self.representation.extract_representation(entity.get_completion(), self.get_model(), self.tokenizer, iit_layer_type_enum.SOME)
                    entity.set_completion_loss(completion_loss)
                    entity.set_completion_embedding_and_shape(completion_emb)
                    entity.set_completion_entropy(entropy)
                    entity.set_token_count(completion_emb.shape[1])
                
                if entity.is_calcutable():
                    entity_list.append(entity)

            except Exception as e:
                entity_list.append(entity)
                print(f"[WARN] Load Embedding: {e}")
       
        return log, entity_list

    def get_model(self):
        if self.model == None: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
                bnb_4bit_use_double_quant = False,
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = bnb_config)
            model.config.use_cache = True
            model.config.pretraining_tp = 1        
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return self.model

    def generate(self, prompt, temperature = 1.0, num_return_sequences = 1, num_beams = 1.0, top_p = 1.0, top_k = 50, do_sample = False ):
        model = self.get_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens = self.get_max_new_tokens(),
            temperature=temperature, 
            top_p=top_p,
            top_k=top_k,            
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            repetition_penalty = 1.1,
            early_stopping = True, #Can stop before reach the max_length
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            do_sample = do_sample
        )

        return outputs
    
    def get_max_new_tokens(self):
        return 5000

    def get_iit_calculator_list(self):
        if self.iit_calculator_list is None: 
            iit_calculator_list = []

            config = intrinsic_information_config()
            config.set_name('Settings_46')
            config.set_calculation_type(ii_calculation_type_enum.SUM)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(5)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            iit_calculator_list.append(intrinsic_information(config)) 

            config = integrated_information_config()
            config.set_name('Settings_64')
            config.set_phi_type(ii_phi_type_enum.SYSTEM_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            iit_calculator_list.append(integrated_information(config)) 

            config = integrated_information_config()
            config.set_name('Settings_65')
            config.set_phi_type(ii_phi_type_enum.BIG_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            iit_calculator_list.append(integrated_information(config)) 
            
            self.iit_calculator_list = iit_calculator_list
        
        return self.iit_calculator_list

