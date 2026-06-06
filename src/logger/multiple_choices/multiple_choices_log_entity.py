from src.logger.log_entity import log_entity
from src.logger.multiple_choices.multiple_choices_log_detail_entity import multiple_choices_log_detail_entity
from dataclasses import dataclass, field
from typing import List
from typing import Optional
from collections import Counter

@dataclass
class multiple_choices_log_entity(log_entity):

    confidence : Optional[float] = None
    permutation_list: List[multiple_choices_log_detail_entity] = field(default_factory=list)

    def add_permutation_list(self, log_detail: multiple_choices_log_detail_entity):
        self.permutation_list.append(log_detail)

    def validate(self): 
        super().validate()
        if self.permutation_list is None or len(self.permutation_list) == 0:
            raise Exception('permutation list empty')

    def calculate_confidence(self) -> float: 
        compared_final_answer_list: list[str] = set(map(lambda x: x.compared_final_answer, self.permutation_list))
        counter = 0
        if len(compared_final_answer_list) != 0:
            counts = Counter(compared_final_answer_list)
            counter = max(counts.values())            
                
        self.confidence = counter / len(self.permutation_list)
        return self.confidence


