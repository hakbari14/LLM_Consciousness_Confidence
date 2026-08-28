from src.logger.log_entity import log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class diffusion_decision_model_log_entity(log_entity):

    x: Optional[dict] = None
    completion : Optional[str] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None
    token_count : Optional[int] = None
    completion_embedding_shape : Optional[str] = None

    completion_loss : Optional[float] = 0.0
    sequence_probability : Optional[float] = 0.0
    length_normalized_sequence_probability : Optional[float] = 0.0
    entropy : Optional[float] = 0.0
    
    self_consistency_confidence : Optional[float] = 0.0
    self_consistency_final_answer : Optional[str] = None
    self_consistency_accuracy : Optional[bool] = None

    self_consistency_completion_confidence : Optional[float] = 0.0
    self_consistency_completion_final_answer : Optional[str] = None
    self_consistency_completion_accuracy : Optional[bool] = None

    driff_rate : Optional[float] = 0.0
    evidence_accumulation_avg : Optional[float] = 0.0
    evidence_list: List[diffusion_decision_model_evidence_log_entity] = field(default_factory=list)

    def add_evidence_list(self, evidence_log: diffusion_decision_model_evidence_log_entity):
        self.evidence_list.append(evidence_log)

    def validate(self): 
        super().validate()

        # if self.x is None:
        #     raise Exception('data x is required')
        
        if self.final_answer is None:
            raise Exception('final answer is required')
        
        if self.evidence_list is None or len(self.evidence_list) == 0:
            raise Exception('evidence list empty')
            
        for e in self.evidence_list: 
            e.validate()

