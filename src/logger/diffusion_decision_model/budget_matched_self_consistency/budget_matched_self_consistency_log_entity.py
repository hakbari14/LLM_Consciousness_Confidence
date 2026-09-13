from src.logger.log_entity import log_entity
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_log_detail_entity import  budget_matched_self_consistency_log_detail_entity
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class budget_matched_self_consistency_log_entity(log_entity):

    x: Optional[dict] = None

    compared_final_answer_nv5 : Optional[str] = None
    self_consistency_nv5 : Optional[str] = None
    accuracy_nv5 : Optional[bool] = None

    compared_final_answer_nv10 : Optional[str] = None
    self_consistency_nv10 : Optional[str] = None
    accuracy_nv10 : Optional[bool] = None

    compared_final_answer_nv15 : Optional[str] = None
    self_consistency_nv15 : Optional[str] = None
    accuracy_nv15 : Optional[bool] = None

    compared_final_answer_nv20 : Optional[str] = None
    self_consistency_nv20 : Optional[str] = None
    accuracy_nv20 : Optional[bool] = None

    compared_final_answer_nv25 : Optional[str] = None
    self_consistency_nv25 : Optional[str] = None
    accuracy_nv25 : Optional[bool] = None

    consistency_list: List[budget_matched_self_consistency_log_detail_entity] = field(default_factory=list)

    def add_consistency_list(self, log_detail: budget_matched_self_consistency_log_detail_entity):
        self.consistency_list.append(log_detail)

    def validate(self): 
        super().validate()

        if self.x is None:
            raise Exception('data x is required')
        
        if self.final_answer is None:
            raise Exception('final answer is required')

        if self.consistency_list is None or len(self.consistency_list) == 0:
            raise Exception('self consistency list empty')
        
        for sc in self.consistency_list: 
            sc.validate()
        
