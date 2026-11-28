from src.logger.logger import logger
from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_log_entity import distributional_diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_evidence_log_entity import distributional_diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.distributional.distributional_diffusion_decision_model_log_detail_entity import distributional_diffusion_decision_model_log_detail_entity
import csv
import pandas as pd

class distributional_diffusion_decision_model_logger(logger):

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)
        self.samples_log_file_name = log_file_name.replace('.csv', '_samples.csv')
        self.evidence_log_file_name = log_file_name.replace('.csv', '_evidence.csv')

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

        self.create_and_prepare(self.samples_log_file_name, self.get_samples_fieldnames())
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

        self.create_and_prepare(self.evidence_log_file_name, self.get_evidence_fieldnames())
        try:
            with open(self.evidence_log_file_name, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = self.get_evidence_fieldnames())
                writer.writerows(self.convert_evidence_buffer())
                csvfile.close()
        except Exception as e:
            print(f"[WARN] Could not logs to CSV: {e}")

    def load_logs_list(self) -> list[distributional_diffusion_decision_model_log_entity]:
        df_logs = pd.read_csv(self.get_log_file_name())
        df_evidences = pd.read_csv(self.get_evidence_log_file_name())
        df_samples = pd.read_csv(self.get_samples_log_file_name())

        log_list: list[distributional_diffusion_decision_model_log_entity] = []
        for _, a_row in df_logs.iterrows():
            log = distributional_diffusion_decision_model_log_entity()
            log.ID = a_row["ID"]
            log.sample_ID = a_row["Sample_ID"]
            log.problem_id = a_row["problem_id"]
            log.split = a_row["Split"]
            log.question = a_row["Question"]
            log.prompt = a_row["Prompt"]
            log.target = a_row["Target"]
            log.completion = a_row["Completion"]
            log.completion_loss = a_row["Completion_Loss"]
            log.final_answer = a_row["Final_Answer"]
            log.compared_final_answer = a_row["Compared_Final_Answer"]
            log.accuracy = a_row["Accuracy"]
            log.token_count = a_row["Token_Count"]
            log.evidence_accumulation_avg = a_row["Evidence_Accumulation_Avg"]
            log.driff_rate = a_row["Drift_Rate"]

            log.self_consistency_confidence_evidence_0 = a_row["Self_Consistency_Confidence_Evidence_0"]
            log.self_consistency_final_answer_evidence_0 = a_row["Self_Consistency_Final_Answer_Evidence_0"]
            log.self_consistency_accuracy_evidence_0 = a_row["Self_Consistency_Accuracy_Evidence_0"]

            log.self_consistency_MV_confidence_evidence_0 = a_row["Self_Consistency_MV_Confidence_Evidence_0"]
            log.self_consistency_MV_final_answer_evidence_0 = a_row["Self_Consistency_MV_Final_Answer_Evidence_0"]
            log.self_consistency_MV_accuracy_evidence_0 = a_row["Self_Consistency_MV_Accuracy_Evidence_0"]

            log.self_consistency_MV_confidence_evidence_last = a_row["Self_Consistency_Confidence_Evidence_Last"]
            log.self_consistency_final_answer_evidence_last = a_row["Self_Consistency_Final_Answer_Evidence_Last"]
            log.self_consistency_accuracy_evidence_last = a_row["Self_Consistency_Accuracy_Evidence_Last"]

            log.self_consistency_MV_confidence_evidence_last = a_row["Self_Consistency_MV_Confidence_Evidence_Last"]
            log.self_consistency_MV_final_answer_evidence_last = a_row["Self_Consistency_MV_Final_Answer_Evidence_Last"]
            log.self_consistency_MV_accuracy_evidence_last = a_row["Self_Consistency_MV_Accuracy_Evidence_Last"]

            log.sequence_probability = a_row["Sequence_Probability"]
            log.length_normalized_sequence_probability = a_row["Length_Normalized_Sequence_Probability"]
            log.mean_entropy = a_row["Mean_Entropy"]
            log.entropy = a_row["Entropy"]
            log.last_layer_representations = a_row["Last_Layer_Representations"]
            log.get_last_layer_representations_numpy()

            b_subset = df_evidences[df_evidences["Log_Index"] == log.index]
            for _, b_row in b_subset.iterrows():
                log_evidence = distributional_diffusion_decision_model_evidence_log_entity()
                log_evidence.index = b_row["Index"]
                log_evidence.log_index = b_row["Log_Index"]
                log_evidence.number_of_samples = b_row["Number_Of_Samples"]

                log_evidence.evidence_accumulation_self_consistency = b_row["Evidence_Accumulation_Self_Consistency"]
                log_evidence.delta_evidence_self_consistency = b_row["Delta_Evidence_Self_Consistency"]
                log_evidence.evidence_accumulation_loss = b_row["Evidence_Accumulation_Loss"]
                log_evidence.delta_evidence_loss = b_row["Delta_Evidence_Loss"]
                log_evidence.evidence_accumulation_avg_prob = b_row["Evidence_Accumulation_Avg_Prob"]
                log_evidence.delta_evidence_avg_prob = b_row["Delta_Evidence_Avg_Prob"]

                log.add_evidence_list(log_evidence)
            
            s_subset = df_samples[df_samples["Log_Index"] == log.index]
            for _, s_row in s_subset.iterrows():
                log_detail = distributional_diffusion_decision_model_log_detail_entity()
                log_detail.index = s_row["Index"]
                log_detail.log_index = s_row["Log_Index"]
                log_detail.evidence_index = s_row["Evidence_Index"]
                
                log_detail.prompt = s_row["Prompt"]
                log_detail.completion = s_row["Completion"]
                log_detail.token_count = s_row["Token_Count"]
                log_detail.original_final_answer = s_row["Original_Final_Answer"]
                log_detail.final_answer = s_row["Final_Answer"]
                log_detail.compared_final_answer = s_row["Compared_Final_Answer"]
                log_detail.accuracy = s_row["Accuracy"]
                log_detail.loss = s_row["Loss"]
            
                log.add_self_consistency_list(log_detail)
            
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
                'Completion': log.completion,
                'Token_Count': log.token_count,
                'Completion_Loss': log.completion_loss,
                'Final_Answer': log.final_answer,
                'Compared_Final_Answer': log.compared_final_answer,
                'Accuracy': log.accuracy,
                'Evidence_Count': len(log.evidence_list),
                'Evidence_Accumulation_Avg': log.evidence_accumulation_avg,
                'Drift_Rate': log.driff_rate,
                
                'Self_Consistency_Confidence_Evidence_0': log.self_consistency_confidence_evidence_0,
                'Self_Consistency_Final_Answer_Evidence_0': log.self_consistency_final_answer_evidence_0,
                'Self_Consistency_Accuracy_Evidence_0': log.self_consistency_accuracy_evidence_0,
                'Self_Consistency_MV_Confidence_Evidence_0': log.self_consistency_MV_confidence_evidence_0,
                'Self_Consistency_MV_Final_Answer_Evidence_0': log.self_consistency_MV_final_answer_evidence_0,
                'Self_Consistency_MV_Accuracy_Evidence_0': log.self_consistency_MV_accuracy_evidence_0,
                
                'Self_Consistency_Confidence_Evidence_Last': log.self_consistency_confidence_evidence_last,
                'Self_Consistency_Final_Answer_Evidence_Last': log.self_consistency_final_answer_evidence_last,
                'Self_Consistency_Accuracy_Evidence_Last': log.self_consistency_accuracy_evidence_last,
                'Self_Consistency_MV_Confidence_Evidence_Last': log.self_consistency_MV_confidence_evidence_last,
                'Self_Consistency_MV_Final_Answer_Evidence_Last': log.self_consistency_MV_final_answer_evidence_last,
                'Self_Consistency_MV_Accuracy_Evidence_Last': log.self_consistency_MV_accuracy_evidence_last,
                
                'Sequence_Probability': log.sequence_probability,
                'Length_Normalized_Sequence_Probability': log.length_normalized_sequence_probability,
                'Mean_Entropy': log.mean_entropy,
                'Entropy': log.entropy,
                'Last_Layer_Representations': log.last_layer_representations,
               
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

                'Self_Consistency_Confidence_Evidence_0',
                'Self_Consistency_Final_Answer_Evidence_0',
                'Self_Consistency_Accuracy_Evidence_0',
                'Self_Consistency_MV_Confidence_Evidence_0',
                'Self_Consistency_MV_Final_Answer_Evidence_0',
                'Self_Consistency_MV_Accuracy_Evidence_0',
                
                'Self_Consistency_Confidence_Evidence_Last',
                'Self_Consistency_Final_Answer_Evidence_Last',
                'Self_Consistency_Accuracy_Evidence_Last',
                'Self_Consistency_MV_Confidence_Evidence_Last',
                'Self_Consistency_MV_Final_Answer_Evidence_Last',
                'Self_Consistency_MV_Accuracy_Evidence_Last',

                'Sequence_Probability',
                'Length_Normalized_Sequence_Probability',
                'Mean_Entropy',
                'Entropy',
                'Last_Layer_Representations',
                ]

    def convert_evidence_buffer(self):
        list = []
        for log in self.buffer:
            for evidence_log in log.evidence_list:
                b = {
                    'Index': evidence_log.index,
                    'Log_Index': evidence_log.log_index,
                    'Number_Of_Samples': evidence_log.number_of_samples,
                    'Sample_ID': log.sample_ID,
                    'Evidence_Accumulation_Self_Consistency': evidence_log.evidence_accumulation_self_consistency,
                    'Delta_Evidence_Self_Consistency': evidence_log.delta_evidence_self_consistency,
                    'Evidence_Accumulation_Loss': evidence_log.evidence_accumulation_loss,
                    'Delta_Evidence_Loss': evidence_log.delta_evidence_loss,
                    'Original_Final_Answer': log.final_answer,
                    'Evidence_Accumulation_Avg_Prob': evidence_log.evidence_accumulation_avg_prob,
                    'Delta_Evidence_Avg_Prob': evidence_log.delta_evidence_avg_prob,
                    }
                list.append(b)
        return list

    def get_evidence_fieldnames(self):
        return [
                'Index',
                'Log_Index',
                'Number_Of_Samples',
                'Sample_ID',
                'Evidence_Accumulation_Self_Consistency',
                'Delta_Evidence_Self_Consistency',
                'Evidence_Accumulation_Loss',
                'Delta_Evidence_Loss',
                'Original_Final_Answer',
                'Evidence_Accumulation_Avg_Prob',
                'Delta_Evidence_Avg_Prob',
                ]

    def convert_samples_buffer(self):
        list = []
        for log in self.buffer:
            for sample_log in log.self_consistency_list:
                b = {
                    'Index': sample_log.index,
                    'Log_Index': sample_log.log_index,
                    'Evidence_Index': sample_log.evidence_index,
                    'Sample_ID': log.sample_ID,
                    'Prompt': sample_log.prompt,
                    'Completion': sample_log.completion,
                    'Token_Count': sample_log.token_count,
                    'Original_Final_Answer': log.final_answer,
                    'Final_Answer': sample_log.final_answer,
                    'Compared_Final_Answer': sample_log.compared_final_answer,
                    'Accuracy': sample_log.accuracy,
                    'Loss': sample_log.loss,
                    }
                list.append(b)
        return list

    def get_samples_fieldnames(self):
        return [
                'Index',
                'Log_Index',
                'Evidence_Index',
                'Sample_ID',
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

