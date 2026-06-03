from src.logger.logger import logger
from typing import cast
from src.logger.confidence.confidence_log_entity import confidence_log_entity

class confidence_logger(logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)


    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            log = cast(confidence_log_entity, log)
            b = { 
                'ID': log.ID, 
                'Settings': log.iit_calculator_name, 
                'Split': log.split, 
                'Sample_ID': log.sample_ID, 
                'problem_id': log.problem_id, 
                'Prompt': log.prompt, 
                'Target': log.target, 
                'Completion': log.completion, 
                'Final_Answer': log.final_answer,
                'Accuracy': log.accuracy,
                'Token_Count': log.token_count, 
                'Completion_Loss': log.completion_loss, 
                'Sequence_Probability': log.sequence_probability,
                'Length_Normalized_Sequence_Probability': log.length_normalized_sequence_probability,
                'Entropy': log.entropy,
                'Confidence_MultipleChoices': log.confidence_mc,
                'Phi_Reward': log.phi_reward,
                'Phi_Reward_Raw': log.phi_reward_raw,
                'Phi_Reward_Raw_Actual': log.phi_reward_raw_actual,
                'Tpm_Loss': log.tpm_loss,
                'Tpm_Entropy': log.tpm_entropy,
                'Completion_Embedding_Shape': log.completion_embedding_shape, 
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return [ 
                'ID', 
                'Settings', 
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Prompt', 
                'Target', 
                'Completion', 
                'Final_Answer',
                'Accuracy',
                'Token_Count', 
                'Completion_Loss', 
                'Sequence_Probability',
                'Length_Normalized_Sequence_Probability',
                'Entropy',
                'Confidence_MultipleChoices',
                'Phi_Reward',
                'Phi_Reward_Raw',
                'Phi_Reward_Raw_Actual',
                'Tpm_Loss',
                'Tpm_Entropy',
                'Completion_Embedding_Shape', 
                ]

        
