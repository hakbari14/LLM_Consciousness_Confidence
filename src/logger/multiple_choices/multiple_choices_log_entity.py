from src.logger.log_entity import log_entity
from src.logger.multiple_choices.multiple_choices_log_detail_entity import multiple_choices_log_detail_entity
from dataclasses import dataclass, field
from typing import List
from typing import Optional

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
        correct = 0
        for log_detail in self.permutation_list:
            if log_detail.accuracy == True:
                correct += 1
                
        self.confidence = correct / len(self.permutation_list)
        return self.confidence


