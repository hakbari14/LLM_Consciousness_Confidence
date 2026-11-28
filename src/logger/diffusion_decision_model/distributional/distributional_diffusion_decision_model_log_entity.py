from src.logger.log_entity import log_entity
from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_evidence_log_entity import distributional_diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_log_detail_entity import distributional_diffusion_decision_model_log_detail_entity
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import json 
import random

@dataclass
class distributional_diffusion_decision_model_log_entity(log_entity):

    index : Optional[int] = None
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
    mean_entropy : Optional[float] = 0.0
    entropy : Optional[float] = 0.0
    last_layer_representations : Optional[str] = None
    
    self_consistency_confidence_evidence_0 : Optional[float] = 0.0
    self_consistency_final_answer_evidence_0 : Optional[str] = None
    self_consistency_accuracy_evidence_0 : Optional[bool] = None

    self_consistency_MV_confidence_evidence_0 : Optional[float] = 0.0
    self_consistency_MV_final_answer_evidence_0 : Optional[str] = None
    self_consistency_MV_accuracy_evidence_0 : Optional[bool] = None

    self_consistency_confidence_evidence_last : Optional[float] = 0.0
    self_consistency_final_answer_evidence_last : Optional[str] = None
    self_consistency_accuracy_evidence_last : Optional[bool] = None

    self_consistency_MV_confidence_evidence_last : Optional[float] = 0.0
    self_consistency_MV_final_answer_evidence_last : Optional[str] = None
    self_consistency_MV_accuracy_evidence_last : Optional[bool] = None

    driff_rate : Optional[float] = 0.0
    evidence_accumulation_avg : Optional[float] = 0.0
    
    evidence_list: List[distributional_diffusion_decision_model_evidence_log_entity] = field(default_factory=list)
    self_consistency_list: List[distributional_diffusion_decision_model_log_detail_entity] = field(default_factory=list)

    def add_evidence_list(self, evidence_log: distributional_diffusion_decision_model_evidence_log_entity):
        self.evidence_list.append(evidence_log)

    def add_self_consistency_list(self, log_detail: distributional_diffusion_decision_model_log_detail_entity):
        self.self_consistency_list.append(log_detail)

    def random_sample(self, number_of_samples: int) -> list[distributional_diffusion_decision_model_log_detail_entity]:
        if number_of_samples < 0:
            raise ValueError("n must be non-negative")

        true_list = list(filter(lambda x: x.compared_final_answer == self.compared_final_answer , self.self_consistency_list))
        if not true_list:
            return []

        if len(true_list) >= number_of_samples:
            return random.sample(true_list, k = number_of_samples)

        result = random.sample(true_list, k=len(true_list))
        result.extend(random.choices(true_list, k = number_of_samples - len(true_list)))
        random.shuffle(result)
        return result

    def validate(self): 
        super().validate()

        if self.x is None:
            raise Exception('data x is required')
        
        if self.final_answer is None:
            raise Exception('final answer is required')
        
        if self.evidence_list is None or len(self.evidence_list) == 0:
            raise Exception('evidence list empty')

        if self.self_consistency_list is None or len(self.self_consistency_list) == 0:
            raise Exception('self_consistency list empty')
            
        for e in self.evidence_list: 
            e.validate()

        for e in self.self_consistency_list: 
            e.validate()

    def get_last_layer_representations_numpy(self) -> np.ndarray: 
        return np.array(json.loads(self.last_layer_representations),dtype=np.float64)
