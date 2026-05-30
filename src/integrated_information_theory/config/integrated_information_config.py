from src.integrated_information_theory.config.integrated_information_theory_config import integrated_information_theory_config
from src.utils.enums_class import ii_phi_type_enum
from dataclasses import dataclass
from typing import Optional


@dataclass
class integrated_information_config(integrated_information_theory_config): 

    phi_type: Optional[ii_phi_type_enum] = None
    
    def vaidate(self):
        super().vaidate()

        if self.phi_type is None:
            raise Exception('phi type is required')
        if ii_phi_type_enum.SYSTEM_PHI != self.phi_type and ii_phi_type_enum.BIG_PHI != self.phi_type:
            raise Exception('phi type has not been correctly determined')

    
