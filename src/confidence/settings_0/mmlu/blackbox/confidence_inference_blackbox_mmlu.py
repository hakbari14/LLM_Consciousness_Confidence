from src.confidence.confidence_inference import confidence_inference
from src.logger.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from src.logger.confidence.confidence_logger import confidence_logger

class confidence_inference_blackbox_mmlu(confidence_inference): 

    def __init__(self, whitebox_modelname, blackbox_modelname):
        super().__init__(whitebox_modelname, blackbox_modelname)

    def create_self_consistency_logger(self, run_number):
        return self_consistency_inference_logger(log_file_name = f'src/confidence/settings_0/mmlu/run_{run_number}/multiple_choices_mmlu.csv')

    def create_confidence_logger(self, settings, run_number):
        return confidence_logger(log_file_name = f'src/confidence/settings_0/mmlu/blackbox/run_{run_number}/confidence_blackbox_mmlu_{settings}.csv')


for run_number in range(6,7):
    print(f'{'*' * 100}  Run Number {run_number}  {'*' * 100}')
    t = confidence_inference_blackbox_mmlu(
                                            whitebox_modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                                            blackbox_modelname='Qwen/Qwen2.5-3B-Instruct'
                                            )
    t.calculate_confidence_blackbox_model(run_number = run_number)
    t.calculate_confidence_whitebox_model(run_number = run_number)
    print(f'{'*' * 210}')

