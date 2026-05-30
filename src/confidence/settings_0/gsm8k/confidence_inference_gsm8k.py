from src.confidence.confidence_inference import confidence_inference
from src.datasets.math.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from src.integrated_information_theory.intrinsic_information import intrinsic_information
from src.integrated_information_theory.integrated_information_theory import integrated_information_theory
from src.integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from src.integrated_information_theory.integrated_information import integrated_information
from src.integrated_information_theory.config.integrated_information_config import integrated_information_config
from src.utils.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum,ii_phi_type_enum

class confidence_inference_gsm8k(confidence_inference): 

    def __init__(self, model_name):
        super().__init__(model_name)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_max_test_dataset_size(1)
            # config.set_ratio_test_dataset_size(0.2)            
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    
    def get_iit_calculator_list(self) -> list[integrated_information_theory]:
        if self.iit_calculator_list is None: 
            iit_calculator_list: list[integrated_information_theory] = []

            config = intrinsic_information_config()
            config.name = 'Settings_46'
            config.calculation_type = ii_calculation_type_enum.SUM
            config.reduced_dimension = 5
            config.tpm_creation_type = tpm_creation_type_enum.PROMPT
            config.layer_type = iit_layer_type_enum.SOME
            config.threashold_type = iit_threashold_type_enum.AVERAGE
            config.last_layer_computation_type = last_layer_computation_type_enum.EXP
            config.last_layer_computation_param = 0.09
            iit_calculator_list.append(intrinsic_information(config)) 

            config = integrated_information_config()
            config.name = 'Settings_64' 
            config.phi_type = ii_phi_type_enum.SYSTEM_PHI
            config.reduced_dimension = 4
            config.layer_type = iit_layer_type_enum.SOME
            config.threashold_type = iit_threashold_type_enum.AVERAGE
            config.tpm_creation_type = tpm_creation_type_enum.PROMPT
            config.last_layer_computation_type = last_layer_computation_type_enum.EXP
            config.last_layer_computation_param = 0.09
            iit_calculator_list.append(integrated_information(config)) 

            config = integrated_information_config()
            config.name = 'Settings_65'
            config.phi_type = ii_phi_type_enum.BIG_PHI
            config.reduced_dimension = 4
            config.layer_type = iit_layer_type_enum.SOME
            config.threashold_type = iit_threashold_type_enum.AVERAGE
            config.tpm_creation_type = tpm_creation_type_enum.PROMPT
            config.last_layer_computation_type = last_layer_computation_type_enum.EXP
            config.last_layer_computation_param = 0.09
            iit_calculator_list.append(integrated_information(config)) 
            
            self.iit_calculator_list = iit_calculator_list
        
        return self.iit_calculator_list

    def create_logger(self, settings, run_number):
        return self_consistency_inference_logger(log_file_name = f'src/confidence/settings_0/gsm8k/run_{run_number}/confidence_gsm8k_{settings}.csv')

t = confidence_inference_gsm8k('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60')
t.calculate_accuracy_confidence(run_number = 6)
