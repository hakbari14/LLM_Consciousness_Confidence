from src.diffusion_decision_model.diffusion_decision_model import diffusion_decision_model
from src.datasets.confidence.mmlu.mmlu_dataset_qwen3_8B import mmlu_dataset_qwen3_8B
from src.datasets.dataset_config import dataset_config
from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger

class diffusion_decision_model_mmlu_qwen3_8B(diffusion_decision_model): 

    def __init__(self, modelname, number_of_evidence: int) -> None:
        super().__init__(modelname, number_of_evidence)
        

    def get_dataset(self) -> mmlu_dataset_qwen3_8B:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size_per_category(5)
            self.dataset = mmlu_dataset_qwen3_8B(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> diffusion_decision_model_logger:
        return diffusion_decision_model_logger(log_file_name = f'logs/diffusion_decision_model/mmlu/{self.get_modelname_dir()}/run_{run_number}/diffusion_decision_model_mmlu{self.get_number_of_evidence_dir()}.csv')


for nv in [5, 10, 15, 20, 25]:
    print(f"{'*' * 100}  Number Of Evidence {nv}  {'*' * 100}")
    t = diffusion_decision_model_mmlu_qwen3_8B(modelname='Qwen/Qwen3-8B', number_of_evidence=nv)
    t.run(from_run_number=1, to_run_number=2)
    t.baseline_features_extractor(from_run_number=1, to_run_number=2)
    print(f"{'*' * 210}")

for nv in [5, 10, 15, 20, 25]:
    t = diffusion_decision_model_mmlu_qwen3_8B(modelname='Qwen/Qwen3-8B', number_of_evidence=nv)
    t.calculate_accracy(run_number=1)

