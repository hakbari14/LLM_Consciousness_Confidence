from dataclasses import dataclass
from typing import Optional

@dataclass
class diffusion_decision_model_log_detail_entity:

    index : Optional[str] = None
    prompt : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    original_final_answer : Optional[str] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None
    loss : Optional[float] = None

    def validate(self): 
        if self.index is None:
            raise Exception('index required')

        if self.original_final_answer is None:
            raise Exception('original final answer required')

        if self.token_count is None:
            raise Exception('token count required')

        if self.accuracy is None:
            raise Exception('accuracy required')

