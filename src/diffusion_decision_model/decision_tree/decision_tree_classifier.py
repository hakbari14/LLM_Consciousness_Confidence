from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger

import sys
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

class decision_tree_classifier:

    def __init__(self, number_of_evidence: int, modelname_dir: str = 'qwen-qwen3-8b') -> None:
        if number_of_evidence is None:
            raise Exception('number of evidence is required')

        self.number_of_evidence = number_of_evidence
        self.modelname_dir = modelname_dir
        self.datasets = ['gpqa', 'countdown', 'math500', 'gsm8k', 'mmlu', 'truthfulqa', 'mmlu_pro', 'aime']
        self.log_directory = './logs/diffusion_decision_model'
        self.log_cache = {}


    def train_decision_tree(self, from_run_number, to_run_number, test_size=0.2, random_state=42, max_depth=None):
        X_list = []
        y_list = []
        for dataset in self.datasets:
            for run_number in range(from_run_number, to_run_number):
                for log in self.load_logs(dataset, run_number):
                    if len(log.evidence_list) != self.number_of_evidence:
                        continue
                    losses = np.array(
                            [[evidence.evidence_accumulation_loss, evidence.evidence_accumulation_self_consistency, evidence.delta_evidence_loss, evidence.delta_evidence_self_consistency] for evidence in log.evidence_list],
                            dtype=float
                        )                    
                    X_list.append(losses)
                    y_list.append(1.0 if log.accuracy else 0.0)

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        X = np.asarray(X)
        y = np.asarray(y)

        print("Original X shape:", X.shape)
        print("y shape:", y.shape)

        X_flat = X.reshape(X.shape[0], -1)
        print("Flattened X shape:", X_flat.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X_flat,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("\n===== Results =====")
        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\n===== Tree Information =====")
        print("Number of nodes:", clf.tree_.node_count)
        print("Tree depth:", clf.tree_.max_depth)
        print("Number of leaves:", clf.get_n_leaves())

        feature_importance = clf.feature_importances_

        print("\n===== Feature Importance =====")
        for i, importance in enumerate(feature_importance):
            if importance > 0:
                row = i // X.shape[2]
                col = i % X.shape[2]

                print(
                    f"X[{row},{col}] -> "
                    f"{importance:.6f}"
                )

        plt.figure(figsize=(20, 10))

        plot_tree(
            clf,
            filled=True,
            feature_names=[
                f"x[{i // X.shape[2]},{i % X.shape[2]}]"
                for i in range(X_flat.shape[1])
            ],
            class_names=[
                str(c) for c in clf.classes_
            ],
            rounded=True,
            fontsize=8
        )

        plt.title("Decision Tree")
        plt.tight_layout()
        plt.savefig("./src/diffusion_decision_model/decision_tree/decision_tree.png", dpi=300, bbox_inches="tight")
        plt.close()
         
        self.extract_tree_rules(clf)
        return clf

    def log_file_name(self, dataset: str, run_number: int) -> str:
        return (f'{self.log_directory}/{dataset}/{self.modelname_dir}/run_{run_number}'
                f'/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv')

    def load_logs(self, dataset: str, run_number: int) -> list[diffusion_decision_model_log_entity]:
        """Read one run once.  A rollout log is hundreds of megabytes."""
        key = (dataset, run_number)
        if key not in self.log_cache:
            logger = diffusion_decision_model_logger(log_file_name = self.log_file_name(dataset, run_number))
            self.log_cache[key] = logger.load_logs_list()
            print(f'loaded {dataset} run {run_number}: {len(self.log_cache[key])} samples', file = sys.stderr)

        return self.log_cache[key]

trainer = decision_tree_classifier(number_of_evidence=5)
trainer.train_decision_tree(from_run_number=1, to_run_number=2)