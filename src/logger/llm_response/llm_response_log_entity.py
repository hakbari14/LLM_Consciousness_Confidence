from src.logger.log_entity import log_entity
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class llm_response_log_entity(log_entity):

    proposed_answer : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None

    def validate(self): 
        super().validate()
        if self.completion is None:
            raise Exception('completion is required')
        if self.token_count is None:
            raise Exception('token count is required')
        if self.accuracy is None:
            raise Exception('accuracy is required')


