from src.confidence.self_consistency_generation import self_consistency_generation
from src.datasets.confidence.truthfulqa_dataset import truthfulqa_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger

class self_consistency_generation_truthfulqa(self_consistency_generation): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def get_dataset(self) -> truthfulqa_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(100)
            self.dataset = truthfulqa_dataset(config)
        return self.dataset

    def create_self_consistency_logger(self, run_number) -> self_consistency_inference_logger:
        return self_consistency_inference_logger(log_file_name = f'src/confidence/settings_0/truthfulqa/run_{run_number}/self_consistency_truthfulqa.csv')


for run_number in range(1,2):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = self_consistency_generation_truthfulqa(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    t.generate_self_consistency(run_number = run_number)
    print(f'{'*' * 210}')

