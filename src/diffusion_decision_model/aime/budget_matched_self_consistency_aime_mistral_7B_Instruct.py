from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.math.aime.aime_dataset_mistral_7B_Instruct import aime_dataset_mistral_7B_Instruct
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_aime_mistral_7B_Instruct(budget_matched_self_consistency): 

    def __init__(self, modelname) -> None:
        super().__init__(modelname)
        

    def get_dataset(self) -> aime_dataset_mistral_7B_Instruct:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(150)
            self.dataset = aime_dataset_mistral_7B_Instruct(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 15000

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/aime/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_aime.csv')

t = budget_matched_self_consistency_aime_mistral_7B_Instruct(modelname='mistralai/Mistral-7B-Instruct-v0.3')
t.run(from_run_number=1, to_run_number=2)

