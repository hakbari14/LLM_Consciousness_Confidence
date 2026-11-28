from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_log_entity import distributional_diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_evidence_log_entity import distributional_diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_log_detail_entity import distributional_diffusion_decision_model_log_detail_entity
from dataclasses import dataclass
from typing import Optional

@dataclass
class distributional_ddm_prompt_entity:

    log: Optional[distributional_diffusion_decision_model_log_entity] = None
    evidence_log : Optional[distributional_diffusion_decision_model_evidence_log_entity] = None
    log_detail : Optional[distributional_diffusion_decision_model_log_detail_entity] = None
    
    partial_prompt : Optional[str] = None
    partial_prompt_tokens : Optional[list] = None

    def validate(self): 
        super().validate()

        if self.log is None:
            raise Exception('log is required')
        if self.evidence is None:
            raise Exception('evidence log is required')
        if self.log_detail is None:
            raise Exception('log_detail is required')

        if self.partial_prompt is None:
            raise Exception('partial prompt is required')
        if self.partial_prompt_tokens is None or len(self.partial_prompt_tokens) == 0:
            raise Exception('partial prompt tokens is required')

    def get_next_log_detail_index(self) -> int: 
        if len(self.log.self_consistency_list) == 0:
            return 0.0
        
        return max(x.index for x in self.log.self_consistency_list) + 1
        
