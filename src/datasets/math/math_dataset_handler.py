from src.datasets.dataset_handler import dataset_handler
from src.datasets.math.utils.evaluate_utils import use_math_verify
import re
import random


class math_dataset_handler(dataset_handler): 

    def __init__(self, config):
        super().__init__(config)

    def verify_final_answer(self, target, final_answer):
        if target == final_answer: 
            return True, final_answer
        else: 
            return use_math_verify(target, final_answer), final_answer

    def generate_wrong_answer(self, latex_expr:str) -> str:
        origin_expr = str(latex_expr)
        numbers = re.findall(r'-?\d+\.?\d*', latex_expr)

        if not numbers:
            replacements = [
                (r'\+', '-'),
                (r'-', '+'),
                (r'\\times', r'\\div'),
                (r'\\div', r'\\times')
            ]

            for old, new in replacements:
                if re.search(old, latex_expr):
                    return re.sub(old, new, latex_expr, count=1)

            return latex_expr if latex_expr != origin_expr else None

        selected = random.choice(numbers)
        value = float(selected)
        delta = random.choice([-2, -1, 1, 2])
        new_value = value + delta

        if selected.isdigit():
            new_value = int(new_value)

        new_expr = latex_expr.replace(
            selected,
            str(new_value),
            1
        )
        return new_expr if new_expr != origin_expr else None

