from src.logger.logger import logger
import csv


class diffusion_decision_model_logger(logger):

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)
        self.samples_log_file_name = log_file_name.replace('.csv', '_samples.csv')
        self.create_and_prepare(self.samples_log_file_name, self.get_samples_fieldnames())
        self.evidence_log_file_name = log_file_name.replace('.csv', '_evidence.csv')
        self.create_and_prepare(self.evidence_log_file_name, self.get_evidence_fieldnames())

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
        self.write_evidence_to_log_file()
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

    def write_evidence_to_log_file(self):
        if len(self.buffer) == 0:
            return

        try:
            with open(self.evidence_log_file_name, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = self.get_evidence_fieldnames())
                writer.writerows(self.convert_evidence_buffer())
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
                'Question': log.question,
                'Prompt': log.prompt,
                'Target': log.target,
                'Completion': log.completion,
                'Token_Count': log.token_count,
                'Completion_Loss': log.completion_loss,
                'Final_Answer': log.final_answer,
                'Compared_Final_Answer': getattr(log, 'compared_final_answer', None),
                'Accuracy': log.accuracy,
                'Evidence_Count': len(log.evidence_list),
                'Evidence_Accumulation_Avg': log.evidence_accumulation_avg,
                'Drift_Rate': log.driff_rate,
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
                'Completion',
                'Token_Count',
                'Completion_Loss',
                'Final_Answer',
                'Compared_Final_Answer',
                'Accuracy',
                'Evidence_Count',
                'Evidence_Accumulation_Avg',
                'Drift_Rate',
                ]

    def convert_evidence_buffer(self):
        list = []
        for log in self.buffer:
            for evidence_log in log.evidence_list:
                b = {
                    'Evidence_Index': evidence_log.index,
                    'Sample_ID': log.sample_ID,
                    'Parent_ID': log.ID,
                    'Evidence': evidence_log.evidence,
                    'Partial_COT': evidence_log.partial_cot,
                    'Evidence_Accumulation_Self_Consistency': evidence_log.evidence_accumulation_self_consistency,
                    'Delta_Evidence_Self_Consistency': evidence_log.delta_evidence_self_consistency,
                    'Evidence_Accumulation_Loss': evidence_log.evidence_accumulation_loss,
                    'Delta_Evidence_Loss': evidence_log.delta_evidence_loss,
                    'Consistency_Count': len(evidence_log.consistency_list),
                    'Original_Final_Answer': log.final_answer,
                    }
                list.append(b)
        return list

    def get_evidence_fieldnames(self):
        return [
                'Evidence_Index',
                'Sample_ID',
                'Parent_ID',
                'Evidence',
                'Partial_COT',
                'Evidence_Accumulation_Self_Consistency',
                'Delta_Evidence_Self_Consistency',
                'Evidence_Accumulation_Loss',
                'Delta_Evidence_Loss',
                'Consistency_Count',
                'Original_Final_Answer',
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
                        'Evidence_Index': evidence_log.index,
                        'Prompt': sample_log.prompt,
                        'Completion': sample_log.completion,
                        'Token_Count': sample_log.token_count,
                        'Original_Final_Answer': sample_log.original_final_answer,
                        'Final_Answer': sample_log.final_answer,
                        'Compared_Final_Answer': getattr(sample_log, 'compared_final_answer', None),
                        'Accuracy': sample_log.accuracy,
                        'Loss': sample_log.loss,
                        }
                    list.append(b)
        return list

    def get_samples_fieldnames(self):
        return [
                'Index',
                'Sample_ID',
                'Parent_ID',
                'Evidence_Index',
                'Prompt',
                'Completion',
                'Token_Count',
                'Original_Final_Answer',
                'Final_Answer',
                'Compared_Final_Answer',
                'Accuracy',
                'Loss',
                ]

    def get_samples_log_file_name(self) -> str:
        return self.samples_log_file_name

    def set_samples_log_file_name(self, value : str) -> None:
        self.samples_log_file_name = value

    def get_evidence_log_file_name(self) -> str:
        return self.evidence_log_file_name

    def set_evidence_log_file_name(self, value : str) -> None:
        self.evidence_log_file_name = value
