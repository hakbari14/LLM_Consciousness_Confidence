from src.confidence.multiple_choices_generation import multiple_choices_generation
from src.datasets.confidence.truthfulqa_dataset import truthfulqa_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.multiple_choices.multiple_choices_inference_logger import multiple_choices_inference_logger

class multiple_choices_generation(multiple_choices_generation): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def get_dataset(self) -> truthfulqa_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_ratio_test_dataset_size(0.2)
            self.dataset = truthfulqa_dataset(config)
        return self.dataset

    def create_multiple_choices_logger(self, run_number) -> multiple_choices_inference_logger:
        return multiple_choices_inference_logger(log_file_name = f'src/confidence/settings_0/truthfulqa/run_{run_number}/multiple_choices_truthfulqa.csv')


for run_number in range(1,2):
    print(f'{'*' * 100}  TruthfulQA : Run Number {run_number}  {'*' * 100}')
    t = multiple_choices_generation(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    t.generate_response(run_number = run_number)
    print(f'{'*' * 210}')

