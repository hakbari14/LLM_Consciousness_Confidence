from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.math.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_gsm8k(budget_matched_self_consistency): 

    def __init__(self, modelname) -> None:
        super().__init__(modelname)
        

    def get_dataset(self) -> gsm8k_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(1)
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/gsm8k/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_gsm8k.csv')

t = budget_matched_self_consistency_gsm8k(modelname='Qwen/Qwen3-8B')
t.run(from_run_number=1, to_run_number=2)
