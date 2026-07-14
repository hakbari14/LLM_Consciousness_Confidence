from dataclasses import dataclass
from src.utils.enums_class import training_type_enum, confidence_type_enum, confidence_reward_calculation_type_enum


@dataclass
class training_config:
    model_name: str
    
    training_type: training_type_enum 
    confidence_type : confidence_type_enum

    acurray_reward_coefficient: float 
    confidence_reward_coefficient: float 
    confidence_reward_type: confidence_reward_calculation_type_enum
    
    def to_dict(self):
        return self.__dict__