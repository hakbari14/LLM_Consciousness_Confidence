from src.datasets.dataset_handler import dataset_handler
from src.datasets.math.utils.evaluate_utils import use_math_verify


class math_dataset_handler(dataset_handler): 

    def __init__(self, config):
        super().__init__(config)

    def verify_final_answer(self, target, final_answer):
        if target == final_answer: 
            return True, final_answer
        else: 
            return use_math_verify(target, final_answer), final_answer
