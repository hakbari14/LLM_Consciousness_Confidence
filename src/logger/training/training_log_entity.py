from src.logger.log_entity import log_entity
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class training_log_entity(log_entity):

    trainer_global_step : Optional[str] = None
    completion : Optional[str] = None
    accuracy_reward : Optional[float] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None

    prompt_confidence : Optional[str] = None
    completion_confidence : Optional[str] = None
    verbal_confidence : Optional[str] = None
    verbal_confidence_reward : Optional[float] = None

    entropy : Optional[float] = None
    entropy_reward : Optional[float] = None

    def validate(self): 
        super().validate()


