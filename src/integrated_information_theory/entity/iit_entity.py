from dataclasses import dataclass, field
from typing import Optional, List
from src.integrated_information_theory.entity.iit_token_entity import iit_token_entity
from src.utils.utility import my_utils
import numpy as np
import pyphi
import gc

@dataclass
class iit_entity:
    key : Optional[str] = None
    prompt_ID : Optional[str] = None
    prompt : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    
    iit_reward : Optional[float] = 0.0
    iit_reward_raw : Optional[float] = 0.0
    iit_reward_raw_actual : Optional[float] = 0.0
    tpm_loss : Optional[float] = 0.0
    tpm_entropy : Optional[float] = 0.0
    completion_loss : Optional[float] = 0.0
    completion_entropy : Optional[float] = 0.0
    
    prompt_embedding : Optional[np.ndarray] = None
    completion_embedding : Optional[np.ndarray] = None
    completion_embedding_for_pca : Optional[np.ndarray] = None
    completion_concatenated_embedding : Optional[np.ndarray] = None
    completion_embedding_shape : Optional[str] = None
    markov_chain : Optional[list] = None

    iit_token_list: List[iit_token_entity] = field(default_factory=list)

    def add_iit_token_list(self, iit_token: iit_token_entity):
        self.iit_token_list.append(iit_token)

    def set_completion_embedding_and_shape(self, completion_embedding: np.ndarray) -> None:
        self.completion_embedding = completion_embedding
        if completion_embedding is not None: 
            self.completion_embedding_shape = completion_embedding.shape

    def add_token_list(self, tokenizer, completion: str, completion_emb: np.ndarray) -> None:
        tokens = tokenizer.tokenize(completion)
        if my_utils.has_add_bos_token(tokenizer):
            tokens.insert(0, 'BOS')

        if len(tokens) != completion_emb.shape[1]:
            raise Exception(f'The number of tokens is not the same as the representation dimensions, completion_shape ={completion_emb.shape}, token count = {len(tokens)}')

        self.token_count = completion_emb.shape[1]
        for idx, t in enumerate(tokens): 
            t_entity = iit_token_entity(idx, t)
            self.add_iit_token_list(t_entity)

    def aggregate_token_list(self, start: int, length: int) -> str:
        result = []
        end = start + length
        for t in self.iit_token_list: 
            if t.token_number() >= start and t.token_number() < end:  
                result.append(t)

        if len(result) == 0:
            return '', ''
        
        token_text_list = list(map(lambda x: x.token(), result))
        token_emb_list = list(map(lambda x: x.attended_embedding, result))
        return " ".join(token_text_list), np.concatenate(tuple(token_emb_list), axis=0,)

    def set_all_markov_chain(self, value) -> None:
        self.markov_chain = value
        for idx, m in enumerate(self.markov_chain):
            t_entity_list = list(filter(lambda x: x.token_number == idx , self.iit_token_list))
            if len(t_entity_list) == 0: continue
            if len(t_entity_list) != 1:
                raise Exception(f'token Entity not found by key {idx} and size list {len(t_entity_list)}')
            
            t_entity = t_entity_list[0]
            t_entity.token_markov_chain_0_1_emb = m
            t_entity.state_index = pyphi.convert.state2le_index(m)

    def set_integrated_information_value(self, state: int, value: float) -> None:
        for t in self.iit_token_list: 
            if t.state_index != state:  
                continue
            t.iit_value = value

    def set_intrinsic_information_value(self, state : int, ii_value: float, effect_value: float, cause_value: float, effect_state_index: int, cause_state_index: int) -> None:
        for t in self.iit_token_list: 
            if t.state_index != state:  
                continue

            t.iit_effect_value = effect_value
            t.iit_cause_value = cause_value
            t.iit_value = ii_value
            t.iit_effect_state_index = effect_state_index
            t.iit_cause_state_index = cause_state_index

    def is_calcutable(self) -> bool:
        if self.completion is None or len(self.completion) == 0:
            return False
        if self.prompt_embedding is None or self.prompt_embedding.shape is None:
            return False
        if self.completion_embedding is None or self.completion_embedding.shape is None:
            return False
        
        return True

    def has_completion_embedding_for_pca(self) -> bool : 
        return self.completion_embedding_for_pca is not None and self.completion_embedding_for_pca.shape is not None

    def has_concatenated_embedding(self) -> bool: 
        return self.completion_concatenated_embedding is not None and self.completion_concatenated_embedding.shape is not None

    @staticmethod
    def clone_list(entity_list: list[iit_token_entity]) -> list[iit_token_entity]: 
        new_entity_list: list[iit_token_entity] = []
        for entity in entity_list: 
            new_entity_list.append(iit_entity.clone(entity))
            
        return new_entity_list

    @staticmethod
    def clone(entity) -> "iit_entity": 
        new_entity = iit_entity()
        new_entity.key = entity.key
        new_entity.prompt_ID = entity.prompt_ID
        new_entity.prompt = entity.prompt
        new_entity.completion = entity.completion
        new_entity.token_count = entity.token_count
        
        if entity.prompt_embedding is not None: 
            new_entity.prompt_embedding = entity.prompt_embedding.copy()
        if entity.completion_embedding is not None: 
            new_entity.completion_embedding = entity.completion_embedding.copy()
        if entity.completion_embedding_for_pca is not None: 
            new_entity.completion_embedding_for_pca = entity.completion_embedding_for_pca.copy()
        if entity.completion_concatenated_embedding is not None: 
            new_entity.completion_concatenated_embedding = entity.completion_concatenated_embedding.copy()
        new_entity.markov_chain = entity.markov_chain

        new_entity.completion_loss = entity.completion_loss
        new_entity.completion_entropy = entity.completion_entropy
        new_entity.iit_reward = entity.iit_reward
        new_entity.iit_reward_raw = entity.iit_reward_raw
        new_entity.iit_reward_raw_actual = entity.iit_reward_raw_actual

        for token_entity in entity.iit_token_list:
            new_token_entity = iit_token_entity.clone(token_entity)
            new_entity.add_iit_token_list(new_token_entity)
    
        return new_entity

    @staticmethod
    def release_memory(entity_list: list["iit_entity"]) -> None: 
        for entity in entity_list: 
            entity.prompt_embedding = None
            entity.completion_embedding = None
            entity.completion_embedding_for_pca = None
            entity.completion_concatenated_embedding = None
            entity.markov_chain = None
            
            for token_entity in entity.iit_token_list:
                token_entity.attended_embedding = None
                token_entity.token_markov_chain_0_1_emb = None
                
        gc.collect()

