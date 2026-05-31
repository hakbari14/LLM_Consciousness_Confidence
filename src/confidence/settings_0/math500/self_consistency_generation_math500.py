from src.confidence.self_consistency_generation import self_consistency_generation
from src.datasets.math.math_500_dataset import math_500_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger

class self_consistency_generation_math500(self_consistency_generation): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_ratio_test_dataset_size(0.2)            
            self.dataset = math_500_dataset(config)
        return self.dataset

    def create_self_consistency_logger(self, run_number):
        return self_consistency_inference_logger(log_file_name = f'src/confidence/settings_0/math500/run_{run_number}/self_consistency_math500.csv')

for run_number in range(1,6):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = self_consistency_generation_math500(modelname='/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60')
    t.generate_self_consistency(run_number = run_number)
    print(f'{'*' * 210}')
