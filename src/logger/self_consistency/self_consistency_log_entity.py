from src.logger.log_entity import log_entity
from src.logger.self_consistency.self_consistency_log_detail_entity import self_consistency_log_detail_entity
from dataclasses import dataclass, field
from typing import List

@dataclass
class self_consistency_log_entity(log_entity):

    consistency_list: List[self_consistency_log_detail_entity] = field(default_factory=list)

    def add_consistency_list(self, log_detail: self_consistency_log_detail_entity):
        self.consistency_list.append(log_detail)

    def validate(self): 
        super().validate()
        if self.consistency_list is None or len(self.consistency_list) == 0:
            raise Exception('self consistency list empty')


