from src.confidence.multiple_choices_generation import multiple_choices_generation
from src.datasets.confidence.mmlu_dataset import mmlu_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.multiple_choices.multiple_choices_inference_logger import multiple_choices_inference_logger

class multiple_choices_generation_mmlu(multiple_choices_generation): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def get_dataset(self) -> mmlu_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_ratio_test_dataset_size(0.01)
            self.dataset = mmlu_dataset(config)
        return self.dataset

    def create_multiple_choices_logger(self, run_number) -> multiple_choices_inference_logger:
        return multiple_choices_inference_logger(log_file_name = f'src/confidence/settings_0/mmlu/run_{run_number}/multiple_choices_mmlu.csv')


for run_number in range(1,6):
    print(f'{'*' * 100}  MMLU : Run Number {run_number}  {'*' * 100}')
    t = multiple_choices_generation_mmlu(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    t.generate_response(run_number = run_number)
    print(f'{'*' * 210}')

