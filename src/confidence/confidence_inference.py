from abc import ABC, abstractmethod
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from src.integrated_information_theory.integrated_information_theory import integrated_information_theory
from src.logger.inference.self_consistency.self_consistency_log_entity import self_consistency_log_entity
from src.logger.inference.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from src.integrated_information_theory.entity.iit_entity import iit_entity
from src.utils.llm_representation import llm_representation
from src.utils.enums_class import llm_pipeline_type_enum, iit_layer_type_enum
from src.utils.utility import my_utils
from tqdm import tqdm
import torch
import gc


class confidence_inference(ABC): 

    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name is None:
            raise Exception('model name is required')
        
        self.representation = llm_representation()
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.iit_calculator_list = None
        self.logger = None

    @torch.inference_mode()
    def calculate_accuracy_confidence(self, num_sequences = 4, run_number = 0): 
        _, test_dataset = self.get_dataset().preprocess_dataset()
        log_dict = {}
        print(f'{'*' * 100}  Calculate IIT  {'*' * 100}')
        
        for idx, x in enumerate(tqdm(test_dataset, desc="Integrated Information Processing", unit="step")):        
            prompt = x['prompt']
            sample_ID = x['sample_id']
            split = x['split']
            target = x['target']
            log = self_consistency_log_entity(idx, sample_ID, sample_ID, split, prompt, target)
            
            try:
                foundational_outputs_sentence = self.generate(prompt, temperature=0.7, num_return_sequences=num_sequences, top_p=0.9, top_k = 50, do_sample=True)
                output = self.tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True)
                for index in range(num_sequences):
                    completion = output[index]
                    final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                    log_detail = self_consistency_log_detail_entity(f'{idx}_{index}')
                    log_detail.set_completion(completion)
                    log_detail.set_final_answer(final_answer)
                    log_detail.set_compared_final_answer(compared_final_answer)
                    log_detail.set_accuracy(accuracy)
                    log.add_consistency_list(log_detail)
                    
                log, origin_entity_list = self.load_embedding_and_loss(log)
                for iit_calculator in self.get_iit_calculator_list():
                    entity_list = iit_entity.clone_list(origin_entity_list)
                    entity = iit_calculator.calculate_prompt(entity_list)
                    if entity is None: continue
                        
                    new_log = self_consistency_log_entity(idx, sample_ID, sample_ID, split, prompt, target)
                    for log_detail in log.get_consistency_list():
                        new_log_detail = self_consistency_log_detail_entity(log_detail.get_index())
                        new_log_detail.set_completion(log_detail.get_completion())
                        new_log_detail.set_final_answer(log_detail.get_final_answer())
                        new_log_detail.set_compared_final_answer(log_detail.get_compared_final_answer())
                        new_log_detail.set_accuracy(log_detail.get_accuracy())
                        new_log.add_consistency_list(new_log_detail)

                    new_log.set_token_count(entity.token_count)
                    new_log.set_reduced_dim(iit_calculator.get_config().reduced_dimension)
                    new_log.set_phi_reward(entity.iit_reward)
                    new_log.set_phi_reward_raw(entity.iit_reward_raw)
                    new_log.set_phi_reward_raw_actual(entity.iit_reward_raw_actual)
                    new_log.set_tpm_loss(entity.tpm_loss)
                    new_log.set_tpm_entropy(entity.tpm_entropy)
                    new_log.set_completion_loss(entity.completion_loss)
                    new_log.set_perplexity(my_utils.calculate_perplexity(entity.completion_loss))
                    new_log.set_entropy(entity.completion_entropy)
                    new_log.set_completion_embedding_shape(entity.completion_embedding_shape)
                    new_log.set_completion(entity.completion)

                    log_detail_list = list(filter(lambda x: x.get_index() == entity.key, new_log.get_consistency_list()))
                    new_log_detail = log_detail_list[0]
                    new_log.set_final_answer(new_log_detail.get_final_answer())
                    new_log.set_accuracy(new_log_detail.get_accuracy())
                    
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
            logger = self.create_logger(settings, run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()


    @torch.inference_mode()
    def load_embedding_and_loss(self, log): 
        entity_list = []
        refine_prompt = self.representation.clean_prompt_for_phi(log.get_prompt())
        prompt_emb, _, _ = self.representation.extract_representation(refine_prompt, self.get_model(), self.tokenizer, self.get_layer_type())
        for log_detail in log.get_consistency_list():     
            try:
                entity = iit_entity()
                entity.key = log_detail.get_index()
                entity.prompt = log.get_prompt()
                entity.prompt_embedding = prompt_emb
                entity.completion = log_detail.get_completion()
                if log_detail.get_completion() is not None:
                    completion_emb, completion_loss, entropy = self.representation.extract_representation(entity.completion, self.get_model(), self.tokenizer, self.get_layer_type())
                    entity.set_completion_embedding_and_shape(completion_emb)
                    entity.token_count = completion_emb.shape[1]
                    entity.completion_loss = completion_loss
                    entity.completion_entropy = entropy
                
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
    
    def get_layer_type(self):
        return iit_layer_type_enum.SOME

    def get_max_new_tokens(self):
        return 5000

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_iit_calculator_list(self) -> list[integrated_information_theory]:
        pass

    @abstractmethod
    def create_logger(self, settings, run_number):
        pass
