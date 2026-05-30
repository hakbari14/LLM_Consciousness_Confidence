from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class iit_token_entity:

    token_number : Optional[int] = None
    token : Optional[str] = None
    attended_embedding : Optional[np.ndarray] = None
    token_markov_chain_0_1_emb : Optional[list] = None
    
    state_index : Optional[int] = None
    iit_cause_state_index: Optional[int] = None
    iit_cause_value: Optional[float] = None
    iit_effect_state_index: Optional[int] = None
    iit_effect_value: Optional[float] = None
    iit_value: Optional[float] = None
    

    @staticmethod
    def clone(token_entity: "iit_token_entity") -> "iit_token_entity": 
        new_token_entity = iit_token_entity()
        
        new_token_entity.token_number = token_entity.token_number
        new_token_entity.token = token_entity.token

        new_token_entity.attended_embedding = token_entity.attended_embedding
        new_token_entity.token_markov_chain_0_1_emb = token_entity.token_markov_chain_0_1_emb

        new_token_entity.iit_cause_state_index = token_entity.iit_cause_state_index
        new_token_entity.iit_cause_value = token_entity.iit_cause_value

        new_token_entity.iit_effect_state_index = token_entity.iit_effect_state_index
        new_token_entity.iit_effect_value = token_entity.iit_effect_value

        new_token_entity.state_index = token_entity.state_index
        new_token_entity.iit_value = token_entity.iit_value
   
        return new_token_entity
