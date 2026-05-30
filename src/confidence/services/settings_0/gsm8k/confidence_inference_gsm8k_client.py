import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from confidence.services.confidence_inference_client import confidence_inference_client
from src.datasets.math.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger

class confidence_inference_gsm8k_client(confidence_inference_client): 

    def __init__(self, model_name):
        super().__init__(model_name)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_max_test_dataset_size(1)
            # config.set_ratio_test_dataset_size(0.2)            
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    def create_logger(self, settings, run_number):
        return self_consistency_inference_logger(log_file_name = f'integrated_information_theory/inference/confidence/services/settings_0/gsm8k/run_{run_number}/confidence_gsm8k_{settings}.csv')

t = confidence_inference_gsm8k_client('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60')
t.calculate_accuracy_confidence(run_number = 3)
