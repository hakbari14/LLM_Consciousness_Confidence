from src.integrated_information_theory.config.integrated_information_theory_config import integrated_information_theory_config
from src.utils.enums_class import ii_calculation_type_enum
from dataclasses import dataclass
from typing import Optional

@dataclass
class intrinsic_information_config(integrated_information_theory_config): 

    calculation_type: Optional[ii_calculation_type_enum] = None
    
    def vaidate(self):
        super().vaidate()

        if self.calculation_type is None:
            raise Exception('calculation type is required')
        if ii_calculation_type_enum.SUM != self.calculation_type and ii_calculation_type_enum.MAX != self.calculation_type:
            raise Exception('calculation type has not been correctly determined')

    
