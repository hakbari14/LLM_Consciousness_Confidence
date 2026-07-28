from src.logger.log_entity import log_entity
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class training_sdt_log_entity(log_entity):

    trainer_global_step : Optional[str] = None
    completion : Optional[str] = None
    accuracy_reward : Optional[float] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None
    confidence : Optional[float] = None

    another_prompt : Optional[str] = None
    another_target : Optional[str] = None
    another_completion : Optional[str] = None
    another_confidence : Optional[float] = None
    sdt_d_prime : Optional[float] = None
    sdt_reward : Optional[float] = None

    def validate(self): 
        super().validate()

    def get_correct_prompt(self) -> Optional[float]:
        return self.prompt if self.accuracy else self.another_prompt

    def get_incorrect_prompt(self) -> Optional[float]:
        return self.another_prompt if self.accuracy else self.prompt

    def get_correct_completion(self) -> Optional[float]:
        return self.completion if self.accuracy else self.another_completion

    def get_incorrect_completion(self) -> Optional[float]:
        return self.another_completion if self.accuracy else self.completion

    def get_correct_confidence(self) -> Optional[float]:
        return self.confidence if self.accuracy else self.another_confidence

    def get_incorrect_confidence(self) -> Optional[float]:
        return self.another_confidence if self.accuracy else self.confidence


