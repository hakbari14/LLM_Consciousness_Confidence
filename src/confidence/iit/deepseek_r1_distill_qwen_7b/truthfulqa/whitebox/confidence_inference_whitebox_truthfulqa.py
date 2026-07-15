from src.confidence.multiple_choices_inference import multiple_choices_inference
from src.logger.multiple_choices.multiple_choices_inference_logger import multiple_choices_inference_logger
from src.logger.confidence.confidence_logger import confidence_logger

class confidence_inference_whitebox_truthfulqa(multiple_choices_inference): 

    def __init__(self, whitebox_modelname, blackbox_modelname):
        super().__init__(whitebox_modelname, blackbox_modelname)

    def create_multiple_choices_logger(self, run_number):
        return multiple_choices_inference_logger(log_file_name = f'src/confidence/settings_0/truthfulqa/run_{run_number}/multiple_choices_truthfulqa.csv')

    def create_confidence_logger(self, settings, run_number):
        return confidence_logger(log_file_name = f'src/confidence/settings_0/truthfulqa/whitebox/run_{run_number}/confidence_whitebox_truthfulqa_{settings}.csv')


for run_number in range(6,11):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = confidence_inference_whitebox_truthfulqa(
                                            whitebox_modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                                            blackbox_modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
                                            )
    t.calculate_confidence_blackbox_model(run_number = run_number)
    t.calculate_confidence_whitebox_model(run_number = run_number)
    print(f'{'*' * 210}')
