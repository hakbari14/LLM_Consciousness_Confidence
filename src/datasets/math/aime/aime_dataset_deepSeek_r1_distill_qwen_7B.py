from src.datasets.math.aime.aime_dataset import aime_dataset
from src.datasets.dataset_config import dataset_config
import re

class aime_dataset_deepSeek_r1_distill_qwen_7B(aime_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)



