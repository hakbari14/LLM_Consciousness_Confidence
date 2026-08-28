from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger
from src.diffusion_decision_model.diffusion_decision_model import diffusion_decision_model

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
        self.datasets = ['gpqa', 'countdown', 'math500', 'gsm8k', 'mmlu', 'truthfulqa', 'mmlu_pro', 'aime']

    def train_logistic_regression(self, from_run_number, to_run_number) -> None:
        log_list: list[diffusion_decision_model_log_entity] = []
        
        X = np.empty((0, 4 * self.number_of_evidence))
        y = np.empty(0)
        for dataset in self.datasets:
            for run_number in range(from_run_number,to_run_number):
                logger = diffusion_decision_model_logger(log_file_name = f'src/diffusion_decision_model/{dataset}/qwen-qwen3-8b/run_{run_number}/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv')
                log_list = logger.load_logs_list()

                X_b = np.array([
                    [
                        value
                        for e in log.evidence_list
                        for value in [
                            e.evidence_accumulation_loss,
                            e.evidence_accumulation_self_consistency,
                            e.delta_evidence_loss,
                            e.delta_evidence_self_consistency
                        ]
                    ]
                    for log in log_list
                ], dtype=float)

                y_b = np.array([1 if log.accuracy else 0 for log in log_list], dtype=int)
                
                X = np.vstack((X, X_b))
                y = np.concatenate((y, y_b))                


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

    def self_consistency_confidence_completion(self, from_run_number, to_run_number) -> None:
        log_list: list[diffusion_decision_model_log_entity] = []
        
        X = np.empty(0)
        y = np.empty(0)
        for dataset in self.datasets:
            for run_number in range(from_run_number,to_run_number):
                logger = diffusion_decision_model_logger(log_file_name = f'src/diffusion_decision_model/{dataset}/qwen-qwen3-8b/run_{run_number}/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv')
                log_list = logger.load_logs_list()
                
                X_b = np.array([log.self_consistency_completion_confidence for log in log_list], dtype=float)
                y_b = np.array([1 if log.self_consistency_completion_accuracy else 0 for log in log_list], dtype=int)
                
                X = np.concatenate((X, X_b))                
                y = np.concatenate((y, y_b))                
        
        fpr, tpr, _ = roc_curve(y, X)
        roc_auc = auc(fpr, tpr)

        ece, _ = self.calculate_ECE_MCE(y, X)        
        print(f"Accuracy Completion : {np.mean(y):.4f}")
        print(f"ROC Completion : {roc_auc:.4f}")
        print(f"ECE Completion : {ece:.4f}")

    def self_consistency_confidence(self, from_run_number, to_run_number) -> None:
        log_list: list[diffusion_decision_model_log_entity] = []
        
        X = np.empty(0)
        y = np.empty(0)
        for dataset in self.datasets:
            for run_number in range(from_run_number,to_run_number):
                logger = diffusion_decision_model_logger(log_file_name = f'src/diffusion_decision_model/{dataset}/qwen-qwen3-8b/run_{run_number}/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv')
                log_list = logger.load_logs_list()
                
                X_b = np.array([log.self_consistency_confidence for log in log_list], dtype=float)
                y_b = np.array([1 if log.self_consistency_accuracy else 0 for log in log_list], dtype=int)
                
                X = np.concatenate((X, X_b))                
                y = np.concatenate((y, y_b))                
        
        fpr, tpr, _ = roc_curve(y, X)
        roc_auc = auc(fpr, tpr)

        ece, _ = self.calculate_ECE_MCE(y, X)        
        print(f"Accuracy : {np.mean(y):.4f}")
        print(f"ROC : {roc_auc:.4f}")
        print(f"ECE : {ece:.4f}")
        

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
# training.train_logistic_regression(from_run_number = 1, to_run_number = 2)
training.self_consistency_confidence_completion(from_run_number = 1, to_run_number = 2)
training.self_consistency_confidence(from_run_number = 1, to_run_number = 2)