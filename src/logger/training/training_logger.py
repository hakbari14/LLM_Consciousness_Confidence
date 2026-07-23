from src.logger.logger import logger
from typing import cast
from src.logger.training.training_log_entity import training_log_entity

class training_logger(logger): 

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
                'Accuracy_Reward': log.accuracy_reward,
                'Accuracy': log.accuracy,

                'Prompt_SDT_Correct': log.prompt_sdt_correct,
                'Completion_SDT_Correct': log.completion_sdt_correct,
                'Confidence_SDT_Correct': log.confidence_sdt_correct,
                'Prompt_SDT_Incorrect': log.prompt_sdt_incorrect,
                'Completion_SDT_Incorrect': log.completion_sdt_incorrect,
                'Confidence_SDT_Incorrect': log.confidence_sdt_incorrect,
                'SDT_Reward': log.sdt_reward,

                'Prompt_Confidence': log.prompt_confidence,
                'Completion_Confidence': log.completion_confidence,
                'Verbal_Confidence': log.verbal_confidence,
                'Verbal_Confidence_Reward': log.verbal_confidence_reward,
                'Entropy': log.entropy,
                'Entropy_Reward': log.entropy_reward,
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
                'Accuracy_Reward',
                'Accuracy',

                'Prompt_SDT_Correct',
                'Completion_SDT_Correct',
                'Confidence_SDT_Correct',
                'Prompt_SDT_Incorrect',
                'Completion_SDT_Incorrect',
                'Confidence_SDT_Incorrect',
                'SDT_Reward',

                'Prompt_Confidence',
                'Completion_Confidence',
                'Verbal_Confidence',
                'Verbal_Confidence_Reward',
                'Entropy',
                'Entropy_Reward',
                ]

        
