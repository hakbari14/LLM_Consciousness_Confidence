from src.datasets.confidence.mmlu.mmlu_dataset import mmlu_dataset
from src.datasets.dataset_config import dataset_config


class mmlu_dataset_deepSeek_r1_distill_qwen_1_5B(mmlu_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)




