from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.confidence.truthfulqa.truthfulqa_dataset_deepSeek_r1_distill_qwen_7B import truthfulqa_dataset_deepSeek_r1_distill_qwen_1_5B 
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_truthfulqa_deepSeek_r1_distill_qwen_1_5B(budget_matched_self_consistency): 

    def __init__(self, modelname) -> None:
        super().__init__(modelname)
        

    def get_dataset(self) -> truthfulqa_dataset_deepSeek_r1_distill_qwen_1_5B:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(500)
            self.dataset = truthfulqa_dataset_deepSeek_r1_distill_qwen_1_5B(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/truthfulqa/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_truthfulqa.csv')

t = budget_matched_self_consistency_truthfulqa_deepSeek_r1_distill_qwen_1_5B(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
t.run(from_run_number=2, to_run_number=3)
