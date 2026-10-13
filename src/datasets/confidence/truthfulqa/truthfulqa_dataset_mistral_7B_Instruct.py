from src.datasets.confidence.truthfulqa.truthfulqa_dataset import truthfulqa_dataset
from src.datasets.dataset_config import dataset_config


class truthfulqa_dataset_mistral_7B_Instruct(truthfulqa_dataset): 

    def __init__(self, config: dataset_config) -> None:
        super().__init__(config)


