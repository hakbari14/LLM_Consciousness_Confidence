from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger
import torch
import numpy as np 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import roc_curve, auc

class diffusion_decision_model_training: 

    def __init__(self, number_of_evidence: int) -> None:
        self.number_of_evidence = number_of_evidence
        if self.number_of_evidence is None:
            raise Exception('number of evidence is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train_logistic_regression(self, from_run_number = 1, to_run_number = 2) -> None:
        log_list: list[diffusion_decision_model_log_entity] = []
        datasets = ['gpqa', 'countdown', 'aime', 'gsm8k']
        for dataset in datasets:
            for run_number in range(from_run_number,to_run_number):
                logger = diffusion_decision_model_logger(log_file_name = f'src/diffusion_decision_model/{dataset}/qwen-qwen3-8b/run_{run_number}/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv')
                df_logs = pd.read_csv(logger.get_log_file_name())
                df_evidences = pd.read_csv(logger.get_evidence_log_file_name())
                df_samples = pd.read_csv(logger.get_samples_log_file_name())
                log_list.extend(self.load_logs_list(df_logs, df_evidences, df_samples))

        X = np.array([
            [e.evidence_accumulation_loss for e in log.evidence_list]
            for log in log_list
        ], dtype=float)
        y = np.array([1 if log.accuracy else 0 for log in log_list], dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples:     {len(X_test)}")

        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        
        print("\n===== Classification Report =====")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\n===== Confusion Matrix =====")
        print(confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        ece, _ = self.calculate_ECE_MCE(y_test, y_prob)        
        print(f"ROC : {roc_auc:.4f}")
        print(f"ECE : {ece:.4f}")
        

    def load_logs_list(self, df_logs, df_evidences, df_samples) -> list[diffusion_decision_model_log_entity]:
        log_list: list[diffusion_decision_model_log_entity] = []
        for _, a_row in df_logs.iterrows():
            log = diffusion_decision_model_log_entity()
            log.ID = a_row["ID"]
            log.sample_ID = a_row["Sample_ID"]
            log.problem_id = a_row["problem_id"]
            log.split = a_row["Split"]
            log.prompt = a_row["Prompt"]
            log.target = a_row["Target"]
            log.completion = a_row["Completion"]
            log.final_answer = a_row["Final_Answer"]
            log.accuracy = a_row["Accuracy"]
            log.token_count = a_row["Token_Count"]

            b_subset = df_evidences[df_evidences["Sample_ID"] == log.sample_ID]
            for _, b_row in b_subset.iterrows():
                log_evidence = diffusion_decision_model_evidence_log_entity()
                log_evidence.index = b_row["Evidence_Index"]
                log_evidence.evidence = b_row["Evidence"]
                log_evidence.partial_cot = b_row["Partial_COT"]
                log_evidence.evidence_accumulation_self_consistency = b_row["Evidence_Accumulation_Self_Consistency"]
                log_evidence.delta_evidence_self_consistency = b_row["Delta_Evidence_Self_Consistency"]
                log_evidence.evidence_accumulation_loss = b_row["Evidence_Accumulation_Loss"]
                log_evidence.delta_evidence_loss = b_row["Delta_Evidence_Loss"]
               
                s_subset = df_samples[(df_samples["Sample_ID"] == log.sample_ID) & (df_samples["Evidence_Index"] == log_evidence.index)]
                for _, s_row in s_subset.iterrows():
                    log_detail = diffusion_decision_model_log_detail_entity()
                    log_detail.index = s_row["Index"]
                    log_detail.prompt = s_row["Prompt"]
                    log_detail.completion = s_row["Completion"]
                    log_detail.token_count = s_row["Token_Count"]
                    log_detail.original_final_answer = s_row["Original_Final_Answer"]
                    log_detail.final_answer = s_row["Final_Answer"]
                    log_detail.accuracy = s_row["Accuracy"]
                    log_detail.loss = s_row["Loss"]
                
                    log_evidence.add_consistency_list(log_detail)

                log.add_evidence_list(log_evidence)
            
            log_list.append(log)

        return log_list

    def calculate_ECE_MCE(df, y_list, confidence_list, n_bins = 10):
        df = pd.DataFrame({
                "confidence": confidence_list,
                "accuracy_reward": y_list
            })
             
        df['binned_confidence'] = pd.qcut(df['confidence'], q=n_bins, duplicates='drop')
        agg_perplexity = df.groupby('binned_confidence', observed=False)['confidence'].agg(['mean'])
        agg_accuracy = df.groupby('binned_confidence', observed=False)['accuracy_reward'].agg(['mean'])

        expected_calibration_error = 0
        maximum_calibration_error = 0
        for idx, row in enumerate(agg_perplexity.iterrows()):
            confidence = row[1]['mean']
            accuracy = agg_accuracy.iloc[idx]['mean']
            expected_calibration_error += abs(confidence - accuracy)
            maximum_calibration_error = max(abs(confidence - accuracy), maximum_calibration_error)

        expected_calibration_error = expected_calibration_error / (idx + 1)
        return expected_calibration_error, maximum_calibration_error


training = diffusion_decision_model_training(number_of_evidence=20)
training.train_logistic_regression()