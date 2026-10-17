from src.datasets.math.gpqa.gpqa_dataset import gpqa_dataset
from src.datasets.dataset_config import dataset_config

class gpqa_dataset_deepSeek_r1_distill_qwen_7B(gpqa_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)



