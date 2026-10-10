from src.diffusion_decision_model.diffusion_decision_model import diffusion_decision_model
from src.datasets.confidence.truthfulqa.truthfulqa_dataset_meta_llama_Llama_3_1_8B_Instruct import truthfulqa_dataset_meta_llama_Llama_3_1_8B_Instruct
from src.datasets.dataset_config import dataset_config
from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger

class diffusion_decision_model_truthfulqa_meta_llama_Llama_3_1_8B_Instruct(diffusion_decision_model): 

    def __init__(self, modelname, number_of_evidence: int) -> None:
        super().__init__(modelname, number_of_evidence)
        

    def get_dataset(self) -> truthfulqa_dataset_meta_llama_Llama_3_1_8B_Instruct:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(500)
            self.dataset = truthfulqa_dataset_meta_llama_Llama_3_1_8B_Instruct(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 5000

    def create_logger(self, run_number) -> diffusion_decision_model_logger:
        return diffusion_decision_model_logger(log_file_name = f'logs/diffusion_decision_model/truthfulqa/{self.get_modelname_dir()}/run_{run_number}/diffusion_decision_model_truthfulqa{self.get_number_of_evidence_dir()}.csv')


# for nv in [5, 10, 15, 20, 25]:
for nv in [5]:
    print(f"{'*' * 100}  Number Of Evidence {nv}  {'*' * 100}")
    t = diffusion_decision_model_truthfulqa_meta_llama_Llama_3_1_8B_Instruct(modelname='/home/hr_akbari/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659', number_of_evidence=nv)
    t.run(from_run_number=2, to_run_number=3)
    t.baseline_features_extractor(from_run_number=2, to_run_number=3)
    print(f"{'*' * 210}")

for nv in [5, 10, 15, 20, 25]:
    t = diffusion_decision_model_truthfulqa_meta_llama_Llama_3_1_8B_Instruct(modelname='/home/hr_akbari/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659', number_of_evidence=nv)
    t.calculate_accracy(run_number=2)

