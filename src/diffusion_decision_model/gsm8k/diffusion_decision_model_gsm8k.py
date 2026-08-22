from src.diffusion_decision_model.diffusion_decision_model import diffusion_decision_model
from src.datasets.math.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger

class diffusion_decision_model_gsm8k(diffusion_decision_model): 

    def __init__(self, modelname, number_of_evidence: int | None = None) -> None:
        super().__init__(modelname, number_of_evidence)
        

    def get_dataset(self) -> gsm8k_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(150)
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> diffusion_decision_model_logger:
        return diffusion_decision_model_logger(log_file_name = f'src/diffusion_decision_model/gsm8k/{self.get_modelname_dir()}/run_{run_number}/diffusion_decision_model_gsm8k{self.get_number_of_evidence_dir()}.csv')


t = diffusion_decision_model_gsm8k(modelname='Qwen/Qwen3-8B', number_of_evidence=10)
t.run(from_run_number=1, to_run_number=2)
