from src.diffusion_decision_model.budget_matched_self_consistency import budget_matched_self_consistency
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_logger import budget_matched_self_consistency_logger
from src.datasets.math.gpqa.gpqa_dataset_meta_llama_Llama_3_1_8B_Instruct import gpqa_dataset_meta_llama_Llama_3_1_8B_Instruct
from src.datasets.dataset_config import dataset_config


class budget_matched_self_consistency_gpqa_meta_llama_Llama_3_1_8B_Instruct(budget_matched_self_consistency): 

    def __init__(self, modelname) -> None:
        super().__init__(modelname)
        

    def get_dataset(self) -> gpqa_dataset_meta_llama_Llama_3_1_8B_Instruct:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            self.dataset = gpqa_dataset_meta_llama_Llama_3_1_8B_Instruct(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 15000

    def create_logger(self, run_number) -> budget_matched_self_consistency_logger:
        return budget_matched_self_consistency_logger(log_file_name = f'logs/diffusion_decision_model/gpqa/{self.get_modelname_dir()}/run_{run_number}/budget_matched_self_consistency_gpqa.csv')

t = budget_matched_self_consistency_gpqa_meta_llama_Llama_3_1_8B_Instruct(modelname='/home/hr_akbari/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659')
t.run(from_run_number=1, to_run_number=2)

