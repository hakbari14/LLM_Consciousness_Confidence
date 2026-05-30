from confidence.confidence_inference import confidence_inference
from src.datasets.math.aime_dataset import aime_dataset
from src.datasets.dataset_config import dataset_config
from src.logger.inference.self_consistency.self_consistency_inference_logger import self_consistency_inference_logger
from src.integrated_information_theory.intrinsic_information import intrinsic_information
from src.integrated_information_theory.config.intrinsic_information_config import intrinsic_information_config
from src.integrated_information_theory.integrated_information import integrated_information
from src.integrated_information_theory.config.integrated_information_config import integrated_information_config
from src.utils.enums_class import ii_calculation_type_enum, tpm_creation_type_enum, last_layer_computation_type_enum, granularity_enum, iit_layer_type_enum, iit_threashold_type_enum,ii_phi_type_enum

class confidence_inference_aime(confidence_inference): 

    def __init__(self, model_name):
        super().__init__(model_name)

    def get_dataset(self):
        if self.dataset is None:
            config = dataset_config(self.model_name)
            config.set_ratio_test_dataset_size(0.1)            
            self.dataset = aime_dataset(config)
        return self.dataset

    
    def get_iit_calculator_list(self):
        if self.iit_calculator_list is None: 
            iit_calculator_list = []

            config = intrinsic_information_config()
            config.set_name('Settings_46')
            config.set_calculation_type(ii_calculation_type_enum.SUM)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(5)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            iit_calculator_list.append(intrinsic_information(config)) 

            config = integrated_information_config()
            config.set_name('Settings_64')
            config.set_phi_type(ii_phi_type_enum.SYSTEM_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            iit_calculator_list.append(integrated_information(config)) 

            config = integrated_information_config()
            config.set_name('Settings_65')
            config.set_phi_type(ii_phi_type_enum.BIG_PHI)
            config.set_adaptive_dim(False)
            config.set_reduced_dim(4)
            config.set_layer_type(iit_layer_type_enum.SOME)
            config.set_threashold_type(iit_threashold_type_enum.AVERAGE)
            config.set_tpm_creation_type(tpm_creation_type_enum.PROMPT)
            config.set_last_layer_computation_type(last_layer_computation_type_enum.EXP)
            config.set_last_layer_computation_param(0.09)
            config.set_granularity(granularity_enum.TOKEN)
            iit_calculator_list.append(integrated_information(config)) 
            
            self.iit_calculator_list = iit_calculator_list
        
        return self.iit_calculator_list

    def get_max_new_tokens(self):
        return 15000

    def create_logger(self, settings, run_number):
        return self_consistency_inference_logger(log_file_name = f'integrated_information_theory/inference/confidence/settings_0/aime/run_{run_number}/confidence_aime_{settings}.csv')

t = confidence_inference_aime('/home/hr_akbari/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60',)
t.calculate_accuracy_confidence(run_number = 2)
