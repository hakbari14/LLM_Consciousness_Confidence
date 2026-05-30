from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

class self_consistency_log_detail_api_entity(BaseModel):
    index: Optional[str] = None
    completion: Optional[str] = None
    final_answer: Optional[str] = None
    compared_final_answer: Optional[str] = None
    token_count: Optional[int] = None
    accuracy: Optional[bool] = None
        

class self_consistency_log_api_entity(BaseModel):
    sample_ID: Optional[str] = None
    problem_id: Optional[str] = None
    split: Optional[str] = None
    prompt: Optional[str] = None
    target: Optional[str] = None
    completion: Optional[str] = None

    token_count: Optional[int] = None
    final_answer: Optional[str] = None
    accuracy: Optional[bool] = None

    completion_embedding_shape: Optional[str] = None
    completion_loss: Optional[float] = None
    perplexity: Optional[float] = None
    entropy: Optional[float] = None

    token_count_for_reduced_dim: Optional[int] = None
    reduced_dim: Optional[int] = None

    phi_reward: Optional[float] = None
    phi_reward_raw: Optional[float] = None
    phi_reward_raw_actual: Optional[float] = None

    tpm_loss: Optional[float] = None
    tpm_entropy: Optional[float] = None

    consistency_list: List = Field(default_factory=list)

    def add_consistency_list(self, item: self_consistency_log_detail_api_entity) -> None:
        self.consistency_list.append(item)

    @field_validator("consistency_list", mode="before")
    @classmethod
    def normalize_children(cls, v):
        if not v:
            return []

        normalized = []
        for item in v:
            if isinstance(item, dict):
                normalized.append(self_consistency_log_detail_api_entity(**item))
            elif isinstance(item, self_consistency_log_detail_api_entity):
                normalized.append(item)
            else:
                raise ValueError(f"Invalid type in consistency_list: {type(item)}")

        return normalized
    
class self_consistency_log_res_api_entity(self_consistency_log_api_entity):
    ii_calculator_name: Optional[str] = None


