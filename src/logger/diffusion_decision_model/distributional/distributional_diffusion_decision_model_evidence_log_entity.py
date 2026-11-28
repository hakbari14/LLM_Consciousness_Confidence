from typing import Optional
from dataclasses import dataclass

@dataclass
class distributional_diffusion_decision_model_evidence_log_entity:

    index : Optional[int] = None
    log_index : Optional[int] = None
    number_of_samples : Optional[int] = None

    evidence_accumulation_self_consistency : Optional[float] = 0.0
    delta_evidence_self_consistency : Optional[float] = 0.0
    evidence_accumulation_loss : Optional[float] = 0.0
    delta_evidence_loss : Optional[float] = 0.0
    evidence_accumulation_avg_prob : Optional[float] = 0.0
    delta_evidence_avg_prob : Optional[float] = 0.0

    def validate(self): 
        if self.index is None:
            raise Exception('index required')
        if self.log_index is None:
            raise Exception('log index required')
        if self.number_of_samples is None:
            raise Exception('number of samples required')
