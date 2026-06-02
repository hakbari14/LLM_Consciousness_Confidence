from src.confidence.self_consistency_generation import self_consistency_generation
from src.datasets.math.aime_dataset import aime_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger

class self_consistency_generation_aime(self_consistency_generation): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.modelname)
            self.dataset = aime_dataset(config)
        return self.dataset

    def get_max_new_tokens(self):
        return 15000

    def create_self_consistency_logger(self, run_number):
        return self_consistency_inference_logger(log_file_name = f'src/confidence/settings_0/aime/run_{run_number}/self_consistency_aime.csv')


for run_number in range(1,6):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = self_consistency_generation_aime(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    t.generate_self_consistency(run_number = run_number)
    print(f'{'*' * 210}')

