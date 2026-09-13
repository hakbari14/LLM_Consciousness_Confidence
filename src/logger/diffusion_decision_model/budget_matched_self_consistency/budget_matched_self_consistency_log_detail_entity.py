from dataclasses import dataclass
from typing import Optional

@dataclass
class budget_matched_self_consistency_log_detail_entity:

    index : Optional[str] = None
    prompt : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None

    def validate(self): 
        if self.index is None:
            raise Exception('index required')

        if self.token_count is None:
            raise Exception('token count required')

        if self.accuracy is None:
            raise Exception('accuracy required')

