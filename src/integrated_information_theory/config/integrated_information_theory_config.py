from src.utils.enums_class import tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum
from dataclasses import dataclass
from abc import ABC
from typing import Optional


@dataclass
class integrated_information_theory_config(ABC): 

    name: Optional[str] = None
    reduced_dimension : Optional[int] = None
    tpm_creation_type: Optional[tpm_creation_type_enum] = None
    layer_type: Optional[iit_layer_type_enum] = None 
    threashold_type: Optional[iit_threashold_type_enum] = None 

    last_layer_computation_type: Optional[last_layer_computation_type_enum] = None 
    last_layer_computation_param : Optional[float] = None

    def validate(self): 
        if self.name is None:
            raise Exception('name is required')
        if self.reduced_dimension is None:
            raise Exception('reduced dimension is required')

        if self.tpm_creation_type is None:
            raise Exception('TPM creation type is required')
        if tpm_creation_type_enum.TRAJECTORY != self.tpm_creation_type and tpm_creation_type_enum.PROMPT != self.tpm_creation_type and tpm_creation_type_enum.BATCH != self.tpm_creation_type:
            raise Exception('TPM creation type has not been correctly determined')

        if self.layer_type is None:
            raise Exception('Layer type is required')
        if iit_layer_type_enum.ALL != self.layer_type and iit_layer_type_enum.LAST != self.layer_type and iit_layer_type_enum.SOME != self.layer_type:
            raise Exception('Layer type has not been correctly determined')

        if self.threashold_type is None:
            raise Exception('Threashold type is required')
        if iit_threashold_type_enum.AVERAGE != self.threashold_type and iit_threashold_type_enum.MEDIAN != self.threashold_type:
            raise Exception('Threashold type has not been correctly determined')

        if self.last_layer_computation_type is None:
            raise Exception('last layer computation type is required')
        if last_layer_computation_type_enum.TANH != self.last_layer_computation_type and last_layer_computation_type_enum.EXP != self.last_layer_computation_type and last_layer_computation_type_enum.IDENTITY != self.last_layer_computation_type:
            raise Exception('last layer computation type has not been correctly determined')
        if self.last_layer_computation_param is None:
            raise Exception('last layer computation param is required')
        

    
