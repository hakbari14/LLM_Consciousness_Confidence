from src.logger.log_entity import log_entity
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class confidence_log_entity(log_entity):

    iit_calculator_name : Optional[str] = None
    completion : Optional[str] = None
    final_answer : Optional[str] = None
    accuracy : Optional[bool] = None
    token_count : Optional[int] = None
    completion_embedding_shape : Optional[str] = None

    completion_loss : Optional[float] = 0.0
    sequence_probability : Optional[float] = 0.0
    length_normalized_sequence_probability : Optional[float] = 0.0
    entropy : Optional[float] = 0.0

    phi_reward : Optional[float] = 0.0
    phi_reward_raw : Optional[float] = 0.0
    phi_reward_raw_actual : Optional[float] = 0.0
    tpm_loss : Optional[float] = 0.0
    tpm_entropy : Optional[float] = 0.0
