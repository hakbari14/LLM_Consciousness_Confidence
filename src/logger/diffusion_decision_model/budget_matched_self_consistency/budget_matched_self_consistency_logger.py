from src.logger.logger import logger
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_log_entity import budget_matched_self_consistency_log_entity
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_log_detail_entity import budget_matched_self_consistency_log_detail_entity
import csv
import pandas as pd

class budget_matched_self_consistency_logger(logger):

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)
        self.samples_log_file_name = log_file_name.replace('.csv', '_samples.csv')

    def validate_log(self, log):
        # A sample whose final answer could not be parsed never gets self-consistency
        # continuations, so strict validation would raise and abort the whole write.
        # Keep the row and report it instead of losing the run.
        try:
            super().validate_log(log)
        except Exception as e:
            print(f"[WARN] validation failed for log ID={getattr(log, 'ID', None)}: {e}")

    def write_attachments(self):
        super().write_attachments()
        self.write_samples_to_log_file()

    def write_samples_to_log_file(self):
        if len(self.buffer) == 0:
            return

        self.create_and_prepare(self.samples_log_file_name, self.get_samples_fieldnames())
        try:
            with open(self.samples_log_file_name, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = self.get_samples_fieldnames())
                writer.writerows(self.convert_samples_buffer())
                csvfile.close()
        except Exception as e:
            print(f"[WARN] Could not logs to CSV: {e}")

    def load_logs_list(self) -> list[budget_matched_self_consistency_log_entity]:
        df_logs = pd.read_csv(self.get_log_file_name())
        df_samples = pd.read_csv(self.get_samples_log_file_name())

        log_list: list[budget_matched_self_consistency_log_entity] = []
        for _, a_row in df_logs.iterrows():
            log = budget_matched_self_consistency_log_entity()
            log.ID = a_row["ID"]
            log.sample_ID = a_row["Sample_ID"]
            log.problem_id = a_row["problem_id"]
            log.split = a_row["Split"]
            log.question = a_row["Question"]
            log.prompt = a_row["Prompt"]
            log.target = a_row["Target"]
            log.compared_final_answer = a_row["Compared_Final_Answer"]
            log.accuracy = a_row["Accuracy"]

            log.compared_final_answer_nv5 = a_row["Compared_Final_Answer_nv5"]
            log.self_consistency_nv5 = a_row["Self_Consistency_nv5"]
            log.accuracy_nv5 = a_row["Accuracy_nv5"]

            log.compared_final_answer_nv10 = a_row["Compared_Final_Answer_nv10"]
            log.self_consistency_nv10 = a_row["Self_Consistency_nv10"]
            log.accuracy_nv10 = a_row["Accuracy_nv10"]

            log.compared_final_answer_nv15 = a_row["Compared_Final_Answer_nv15"]
            log.self_consistency_nv15 = a_row["Self_Consistency_nv15"]
            log.accuracy_nv15 = a_row["Accuracy_nv15"]

            log.compared_final_answer_nv20 = a_row["Compared_Final_Answer_nv20"]
            log.self_consistency_nv20 = a_row["Self_Consistency_nv20"]
            log.accuracy_nv20 = a_row["Accuracy_nv20"]

            log.compared_final_answer_nv25 = a_row["Compared_Final_Answer_nv25"]
            log.self_consistency_nv25 = a_row["Self_Consistency_nv25"]
            log.accuracy_nv25 = a_row["Accuracy_nv25"]

            s_subset = df_samples[df_samples["Sample_ID"] == log.sample_ID]
            for _, s_row in s_subset.iterrows():
                log_detail = budget_matched_self_consistency_log_detail_entity()
                log_detail.index = s_row["Index"]
                log_detail.prompt = s_row["Prompt"]
                log_detail.completion = s_row["Completion"]
                log_detail.token_count = s_row["Token_Count"]
                log_detail.original_final_answer = s_row["Original_Final_Answer"]
                log_detail.final_answer = s_row["Final_Answer"]
                log_detail.compared_final_answer = s_row["Compared_Final_Answer"]
                log_detail.accuracy = s_row["Accuracy"]
            
                log.add_consistency_list(log_detail)
            
            log_list.append(log)

        return log_list

    def convert_buffer(self):
        list = []
        for log in self.buffer:
            b = {
                'ID': log.ID,
                'Split': log.split,
                'Sample_ID': log.sample_ID,
                'problem_id': log.problem_id,
                'Question': log.question,
                'Prompt': log.prompt,
                'Target': log.target,

                'Compared_Final_Answer_nv5': log.compared_final_answer_nv5,
                'Self_Consistency_nv5': log.self_consistency_nv5,
                'Accuracy_nv5': log.accuracy_nv5,

                'Compared_Final_Answer_nv10': log.compared_final_answer_nv10,
                'Self_Consistency_nv10': log.self_consistency_nv10,
                'Accuracy_nv10': log.accuracy_nv10,

                'Compared_Final_Answer_nv15': log.compared_final_answer_nv15,
                'Self_Consistency_nv15': log.self_consistency_nv15,
                'Accuracy_nv15': log.accuracy_nv15,

                'Compared_Final_Answer_nv20': log.compared_final_answer_nv20,
                'Self_Consistency_nv20': log.self_consistency_nv20,
                'Accuracy_nv20': log.accuracy_nv20,
               
                'Compared_Final_Answer_nv25': log.compared_final_answer_nv25,
                'Self_Consistency_nv25': log.self_consistency_nv25,
                'Accuracy_nv25': log.accuracy_nv25,
               
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
                'Prompt',
                'Target',

                'Compared_Final_Answer_nv5',
                'Self_Consistency_nv5',
                'Accuracy_nv5',

                'Compared_Final_Answer_nv10',
                'Self_Consistency_nv10',
                'Accuracy_nv10',

                'Compared_Final_Answer_nv15',
                'Self_Consistency_nv15',
                'Accuracy_nv15',

                'Compared_Final_Answer_nv20',
                'Self_Consistency_nv20',
                'Accuracy_nv20',
               
                'Compared_Final_Answer_nv25',
                'Self_Consistency_nv25',
                'Accuracy_nv25',
                ]


    def convert_samples_buffer(self):
        list = []
        for log in self.buffer:
            for evidence_log in log.evidence_list:
                for sample_log in evidence_log.consistency_list:
                    b = {
                        'Index': sample_log.index,
                        'Sample_ID': log.sample_ID,
                        'Parent_ID': log.ID,
                        'Prompt': sample_log.prompt,
                        'Completion': sample_log.completion,
                        'Token_Count': sample_log.token_count,
                        'Target': log.target,
                        'Final_Answer': sample_log.final_answer,
                        'Compared_Final_Answer': sample_log.compared_final_answer,
                        'Accuracy': sample_log.accuracy,
                        }
                    list.append(b)
        return list

    def get_samples_fieldnames(self):
        return [
                'Index',
                'Sample_ID',
                'Parent_ID',
                'Prompt',
                'Completion',
                'Token_Count',
                'Target',
                'Final_Answer',
                'Compared_Final_Answer',
                'Accuracy',
                ]

    def get_samples_log_file_name(self) -> str:
        return self.samples_log_file_name

    def set_samples_log_file_name(self, value : str) -> None:
        self.samples_log_file_name = value


