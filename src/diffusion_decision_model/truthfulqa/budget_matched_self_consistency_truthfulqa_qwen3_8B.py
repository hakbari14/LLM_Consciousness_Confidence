from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.confidence.truthfulqa.truthfulqa_dataset_qwen3_8B import truthfulqa_dataset_qwen3_8B
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_truthfulqa_qwen3_8B(budget_matched_self_consistency): 

    def __init__(self) -> None:
        super().__init__()
        

    def get_dataset(self) -> truthfulqa_dataset_qwen3_8B:
        if self.dataset is None:
            config = dataset_config(self.get_model_name())
            self.dataset = truthfulqa_dataset_qwen3_8B(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def get_model_name(self) -> str:
        return 'Qwen/Qwen3-8B'

    def get_benchmark_name(self) -> str:
        return 'truthfulqa'

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/truthfulqa/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_truthfulqa.csv')

if __name__ == "__main__":
    t = budget_matched_self_consistency_truthfulqa_qwen3_8B()
    t.run(from_run_number=2, to_run_number=3)
    t.calculate_accracy(run_number=2)

