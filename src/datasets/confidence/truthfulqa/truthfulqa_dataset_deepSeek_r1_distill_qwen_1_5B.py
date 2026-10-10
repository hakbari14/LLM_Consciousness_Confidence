from src.datasets.confidence.truthfulqa.truthfulqa_dataset import truthfulqa_dataset
from src.datasets.dataset_config import dataset_config


class truthfulqa_dataset_deepSeek_r1_distill_qwen_1_5B(truthfulqa_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)


