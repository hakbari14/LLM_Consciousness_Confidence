from dataclasses import dataclass
from typing import Optional

@dataclass
class multiple_choices_log_detail_entity:

    index : Optional[str] = None
    prompt : Optional[str] = None
    target : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None

