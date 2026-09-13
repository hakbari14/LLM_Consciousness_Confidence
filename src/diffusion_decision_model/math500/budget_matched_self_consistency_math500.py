from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.math.math_500_dataset import math_500_dataset
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_math500(budget_matched_self_consistency): 

    def __init__(self, modelname) -> None:
        super().__init__(modelname)
        

    def get_dataset(self) -> math_500_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(150)
            self.dataset = math_500_dataset(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/math500/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_math500.csv')

t = budget_matched_self_consistency_math500(modelname='Qwen/Qwen3-8B')
t.run(from_run_number=1, to_run_number=2)
