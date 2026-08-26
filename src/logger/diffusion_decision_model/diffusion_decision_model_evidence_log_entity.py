from typing import Optional, List
from dataclasses import dataclass, field
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity

@dataclass
class diffusion_decision_model_evidence_log_entity:

    index : Optional[str] = None
    evidence : Optional[str] = None
    partial_cot : Optional[str] = None
    partial_completion : Optional[str] = None
    partial_cot_loss : Optional[float] = 0.0
    evidence_accumulation_self_consistency : Optional[float] = 0.0
    delta_evidence_self_consistency : Optional[float] = 0.0
    evidence_accumulation_loss : Optional[float] = 0.0
    delta_evidence_loss : Optional[float] = 0.0

    consistency_list: List[diffusion_decision_model_log_detail_entity] = field(default_factory=list)

    def add_consistency_list(self, log_detail: diffusion_decision_model_log_detail_entity):
        self.consistency_list.append(log_detail)

    def validate(self): 
        if self.index is None:
            raise Exception('index required')
        if self.evidence is None:
            raise Exception('evidence required')
        if self.index != 0 and self.partial_cot is None:
            raise Exception('partial cot required')
        if self.partial_cot_loss is None:
            raise Exception('partial cot loss required')

        if self.consistency_list is None or len(self.consistency_list) == 0:
            raise Exception('self consistency list empty')
        
        for sc in self.consistency_list: 
            sc.validate()
