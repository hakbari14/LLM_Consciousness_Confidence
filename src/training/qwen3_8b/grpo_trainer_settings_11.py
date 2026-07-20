from src.training.grpo_trainer import grpo_trainer
from src.logger.training.training_logger import training_logger
from trl import GRPOConfig, ModelConfig
from src.datasets.math.open_thoughts_dataset import open_thoughts_dataset
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import training_type_enum, confidence_type_enum, confidence_reward_calculation_type_enum
from src.training.training_config import training_config

class grpo_trainer_settings_11(grpo_trainer): 

    def __init__(self):
        config = training_config(
            
            model_name="Qwen/Qwen3-8B",
            training_type = training_type_enum.ACCURACY_REWARD_CONFIDENCE, 
            confidence_type = confidence_type_enum.LEVEL,
            confidence_reward_type = confidence_reward_calculation_type_enum.brier_score,
            entropy_reward_type = confidence_reward_calculation_type_enum.linear,
            acurray_reward_coefficient = 1.0,
            confidence_reward_coefficient = 1.0,
            entropy_confidence_reward_coefficient = 1.0,
        )
        
        super().__init__(config)

    def get_dataset(self) -> open_thoughts_dataset:
        if self.dataset is None:
            config = dataset_config(self.config.model_name)
            config.set_max_test_dataset_size(160)
            self.dataset = open_thoughts_dataset(config)
        return self.dataset

    def get_model_config(self):
        if self.model_config is None:
            self.model_config = ModelConfig(
                model_name_or_path = self.config.model_name,
                attn_implementation="flash_attention_2",
                use_peft=True,
                lora_r=2048,
                lora_alpha=1024,
                load_in_4bit=True,
            )
        return self.model_config

    def get_training_args(self):
        
        if self.training_args is None:
            self.training_args = GRPOConfig(
                output_dir="live_logs/settings_11",
                learning_rate=3e-6,
                lr_scheduler_type="cosine",
                logging_steps=10,
                max_steps=1200,
                per_device_train_batch_size=2,      
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=2,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                bf16=True,
                use_vllm=True,
                vllm_mode="server",
                vllm_server_host="localhost",
                vllm_server_port=8000,                
                max_completion_length=5000, 
                num_generations=2,                      
                num_generations_eval=1,                      
                beta=0.001,
                warmup_ratio=0.0,

                report_to=['tensorboard'],
                logging_dir='live_logs/settings_11/tb_logs',  
                eval_strategy="steps",  
                eval_steps=50,
                save_steps=50
            )

        return self.training_args
    
    def get_logger(self):
        if self.logger is None:
            self.logger = training_logger(log_file_name = 'live_logs/settings_11/settings_11.csv')

        return self.logger


t = grpo_trainer_settings_11()
t.train()
