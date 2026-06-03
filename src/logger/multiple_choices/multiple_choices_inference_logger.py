from src.logger.logger import logger
import csv


class multiple_choices_inference_logger(logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)
        self.samples_log_file_name = log_file_name.replace('.csv', '_samples.csv')
        self.create_and_prepare(self.samples_log_file_name, self.get_samples_fieldnames())

    def write_attachments(self): 
        super().write_attachments()
        self.write_samples_to_log_file()

    def write_samples_to_log_file(self): 
        if len(self.buffer) == 0:
            return 
        
        try:
            with open(self.samples_log_file_name, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = self.get_samples_fieldnames())
                writer.writerows(self.convert_samples_buffer())
                csvfile.close()
        except Exception as e:
            print(f"[WARN] Could not logs to CSV: {e}")

    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            b = { 
                'ID': log.ID, 
                'Split': log.split, 
                'Sample_ID': log.sample_ID, 
                'problem_id': log.problem_id, 
                'Prompt': log.prompt, 
                'Target': log.target, 
                'Confidence': log.confidence, 
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return [ 
                'ID', 
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Prompt', 
                'Target', 
                'Confidence', 
                ]


    def convert_samples_buffer(self): 
        list = []
        for log in self.buffer:
            for sample_log in log.permutation_list:
                b = { 
                    'Index': sample_log.index, 
                    'Sample_ID': log.sample_ID, 
                    'Prompt': sample_log.prompt, 
                    'Target': sample_log.target, 
                    'Completion': sample_log.completion, 
                    'Token_Count': sample_log.token_count, 
                    'Final_Answer': sample_log.final_answer, 
                    'Accuracy': sample_log.accuracy, 
                    'Compared_Final_Answer': sample_log.compared_final_answer, 
                    }
                list.append(b)            
        return list

    def get_samples_fieldnames(self): 
        return [ 
                'Index', 
                'Sample_ID', 
                'Prompt', 
                'Target', 
                'Completion', 
                'Token_Count', 
                'Final_Answer', 
                'Accuracy', 
                'Compared_Final_Answer', 
                ]

    def get_samples_log_file_name(self) -> str:
        return self.samples_log_file_name

    def set_samples_log_file_name(self, value : str) -> None:
        self.samples_log_file_name = value
