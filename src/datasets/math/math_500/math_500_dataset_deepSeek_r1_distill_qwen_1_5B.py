from src.datasets.math.math_500.math_500_dataset import math_500_dataset
from src.datasets.dataset_config import dataset_config


class math_500_dataset_deepSeek_r1_distill_qwen_1_5B(math_500_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)

