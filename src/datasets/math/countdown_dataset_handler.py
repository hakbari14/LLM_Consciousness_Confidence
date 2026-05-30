from src.datasets.dataset_handler import dataset_handler
import re

class countdown_dataset_handler(dataset_handler): 

    def __init__(self, config):
        super().__init__(config)

    def verify_final_answer(self, target, equation):
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           return False, None
        result = eval(equation, {"__builtins__": None}, {})
        return abs(float(result) - float(target)) < 1e-5, result

        