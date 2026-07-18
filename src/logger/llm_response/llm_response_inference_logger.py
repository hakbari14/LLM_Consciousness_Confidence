from src.logger.logger import logger
import csv


class llm_response_inference_logger(logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)

    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            b = { 
                'ID': log.ID, 
                'Split': log.split, 
                'Sample_ID': log.sample_ID, 
                'problem_id': log.problem_id, 
                'Question': log.question, 
                'Proposed_Answer': log.proposed_answer, 
                'Prompt': log.prompt, 
                'Target': log.target, 
                'Completion': log.completion, 
                'Token_Count': log.token_count, 
                'Final_Answer': log.final_answer, 
                'Accuracy': log.accuracy, 
                'Compared_Final_Answer': log.compared_final_answer, 
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return [ 
                'ID', 
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Question', 
                'Proposed_Answer', 
                'Prompt', 
                'Target', 
                'Completion', 
                'Token_Count', 
                'Final_Answer', 
                'Accuracy', 
                'Compared_Final_Answer', 
                ]

