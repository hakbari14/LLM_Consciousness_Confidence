from src.logger.logger import logger
from typing import cast
from src.logger.training.training_log_entity import training_log_entity

class training_sdt_logger(logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)

    def get_log_list(self, trainer_global_step): 
        return list(filter(lambda x: x.trainer_global_step == trainer_global_step , self.buffer))

    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            log = cast(training_log_entity, log)
            b = { 
                'ID': log.ID, 
                'Trainer_Global_Step': log.trainer_global_step, 
                'Split': log.split, 
                'Sample_ID': log.sample_ID, 
                'problem_id': log.problem_id, 
                'Question': log.question, 
                'Prompt': log.prompt, 
                'Target': log.target, 
                'Completion': log.completion, 
                'Token_Count': log.token_count, 
                'Final_Answer': log.final_answer,
                'Compared_Final_Answer': log.compared_final_answer,
                'Confidence': log.confidence,
                'Accuracy_Reward': log.accuracy_reward,
                'Accuracy': log.accuracy,
                'Another_Prompt': log.another_prompt,
                'Another_Target': log.another_target,
                'Another_Completion': log.another_completion,
                'Another_Confidence': log.another_confidence,
                'SDT_D_Prime': log.sdt_d_prime,
                'SDT_Reward': log.sdt_reward,
                }
            list.append(b)            
        return list


    def get_fieldnames(self): 
        return [ 
                'ID', 
                'Trainer_Global_Step', 
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Question', 
                'Prompt', 
                'Target', 
                'Completion', 
                'Token_Count', 
                'Final_Answer',
                'Compared_Final_Answer',
                'Confidence',
                'Accuracy_Reward',
                'Accuracy',
                'Another_Prompt',
                'Another_Target',
                'Another_Completion',
                'Another_Confidence',
                'SDT_D_Prime',
                'SDT_Reward',
                ]

        
