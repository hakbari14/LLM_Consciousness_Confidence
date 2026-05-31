from dataclasses import dataclass
from typing import Optional

@dataclass
class self_consistency_log_detail_entity:

    index : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None

