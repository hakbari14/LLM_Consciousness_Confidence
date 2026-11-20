from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.math.gpqa.gpqa_dataset_deepSeek_r1_distill_qwen_7B import gpqa_dataset_deepSeek_r1_distill_qwen_7B
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_gpqa_deepSeek_r1_distill_qwen_7B(budget_matched_self_consistency): 

    def __init__(self) -> None:
        super().__init__()

    def get_dataset(self) -> gpqa_dataset_deepSeek_r1_distill_qwen_7B:
        if self.dataset is None:
            config = dataset_config(self.get_model_name())
            self.dataset = gpqa_dataset_deepSeek_r1_distill_qwen_7B(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 9000
    
    def get_model_name(self) -> str:
        return 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

    def get_benchmark_name(self) -> str:
        return 'gpqa'

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/gpqa/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_gpqa.csv')

if __name__ == "__main__":
    t = budget_matched_self_consistency_gpqa_deepSeek_r1_distill_qwen_7B()
    t.run(from_run_number=2, to_run_number=3)
    t.calculate_accracy(run_number=2)
