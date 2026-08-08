from typing import Optional, List
from dataclasses import dataclass, field
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity

@dataclass
class diffusion_decision_model_evidence_log_entity:

    index : Optional[str] = None
    evidence : Optional[str] = None
    partial_cot : Optional[str] = None
    evidence_accumulation : Optional[float] = 0.0
    delta_evidence : Optional[float] = 0.0

    consistency_list: List[diffusion_decision_model_log_detail_entity] = field(default_factory=list)

    def add_consistency_list(self, log_detail: diffusion_decision_model_log_detail_entity):
        self.consistency_list.append(log_detail)

    def validate(self): 
        if self.consistency_list is None or len(self.consistency_list) == 0:
            raise Exception('self consistency list empty')
        for sc in self.consistency_list: 
            sc.validate()
