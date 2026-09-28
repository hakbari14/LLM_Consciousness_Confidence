from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.math.gpqa.gpqa_dataset_qwen3_8B import gpqa_dataset_qwen3_8B
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_gpqa_qwen3_8B(budget_matched_self_consistency): 

    def __init__(self, modelname) -> None:
        super().__init__(modelname)
        

    def get_dataset(self) -> gpqa_dataset_qwen3_8B:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            self.dataset = gpqa_dataset_qwen3_8B(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 15000

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/gpqa/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_gpqa.csv')

t = budget_matched_self_consistency_gpqa_qwen3_8B(modelname='Qwen/Qwen3-8B')
t.run(from_run_number=1, to_run_number=2)

