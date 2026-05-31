from dataclasses import dataclass
from typing import Optional

@dataclass
class log_entity:

    ID : Optional[int] = None
    sample_ID : Optional[str] = None
    problem_id : Optional[str] = None
    split : Optional[str] = None
    prompt : Optional[str] = None
    target : Optional[str] = None

    def equal(self, x):
        return self.ID == x.ID

    def validate(self): 
        if self.ID is None:
            raise Exception('ID is required')
        if self.sample_ID is None:
            raise Exception('Sample ID is required')
        if self.prompt is None or len(self.prompt) == 0:
            raise Exception('prompt is required')
        if self.split is None:
            raise Exception('split is required')
        if self.target is None:
            raise Exception('target is required')

