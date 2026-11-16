
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_log_entity import budget_matched_self_consistency_log_entity
from src.logger.diffusion_decision_model.budget_matched_self_consistency.budget_matched_self_consistency_log_detail_entity import  budget_matched_self_consistency_log_detail_entity

from typing import (
    List,
    Dict,
    Tuple,
)
from collections import Counter
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os

class budget_matched_selfconsistency_evaluator:

    @staticmethod
    def filter_entities(entities: List[budget_matched_self_consistency_log_entity], min_samples: int = 128) -> List[Tuple[budget_matched_self_consistency_log_entity, List[budget_matched_self_consistency_log_detail_entity]]]:
        filtered: List[
            Tuple[
                budget_matched_self_consistency_log_entity,
                List[budget_matched_self_consistency_log_detail_entity]
            ]
        ] = []

        for entity in entities:
            if not entity.target:
                continue

            valid_list: List[budget_matched_self_consistency_log_detail_entity] = [
                item for item in entity.consistency_list
                if item.compared_final_answer is not None
                and str(item.compared_final_answer).strip() != ""
            ]

            if len(valid_list) >= min_samples:
                filtered.append((entity, valid_list))

        return filtered

    @staticmethod
    def calculate_ece(
        y_true: NDArray[np.int_],
        confidence: NDArray[np.float64],
        n_bins: int = 10
    ) -> float:

        bin_boundaries: NDArray[np.float64] = np.linspace(
            0.0, 1.0, n_bins + 1
        )

        ece: float = 0.0

        for i in range(n_bins):

            if i == n_bins - 1:
                mask: NDArray[np.bool_] = (
                    (confidence >= bin_boundaries[i]) &
                    (confidence <= bin_boundaries[i + 1])
                )
            else:
                mask = (
                    (confidence >= bin_boundaries[i]) &
                    (confidence < bin_boundaries[i + 1])
                )

            if np.any(mask):

                bin_accuracy: float = float(np.mean(y_true[mask]))

                bin_confidence: float = float(
                    np.mean(confidence[mask])
                )

                ece += (
                    np.sum(mask) / len(y_true)
                ) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    @staticmethod
    def calculate_metrics(
        entities: List[
            Tuple[
                budget_matched_self_consistency_log_entity,
                List[budget_matched_self_consistency_log_detail_entity]
            ]
        ],
        n: int,
        rng: np.random.Generator,
        n_bins: int = 10
    ) -> Tuple[float, float]:

        y_true: List[int] = []
        confidence: List[float] = []

        for entity, valid_list in entities:
            indices: NDArray[np.int_] = rng.choice(
                len(valid_list),
                size=n,
                replace=False
            )

            selected: List[budget_matched_self_consistency_log_detail_entity] = [ valid_list[int(i)] for i in indices]
            answers: List[str] = [
                str(item.compared_final_answer).strip()
                for item in selected
            ]

            answer_counts: Counter[str] = Counter(answers)
            max_votes: int = max(answer_counts.values())

            majority_answers: List[str] = [
                answer
                for answer, count in answer_counts.items()
                if count == max_votes
            ]

            predicted_answer: str = str(rng.choice(majority_answers))
            conf: float = max_votes / n

            is_correct: int = int(predicted_answer == str(entity.target).strip())

            confidence.append(conf)
            y_true.append(is_correct)

        y_true_array: NDArray[np.int_] = np.asarray(y_true, dtype=int)
        confidence_array: NDArray[np.float64] = np.asarray(confidence, dtype=float)

        if len(np.unique(y_true_array)) < 2:
            auroc: float = float("nan")
        else:
            auroc = float(
                roc_auc_score(y_true_array, confidence_array)
            )

        ece: float = budget_matched_selfconsistency_evaluator.calculate_ece(
            y_true=y_true_array,
            confidence=confidence_array,
            n_bins=n_bins
        )

        return auroc, ece

    @staticmethod
    def run(
        entities: List[budget_matched_self_consistency_log_entity],
        min_n: int = 5,
        max_n: int = 128,
        num_runs: int = 10,
        n_bins: int = 10,
        seed: int = 42
    ) -> Dict[str, List[float]]:

        filtered_entities = (budget_matched_selfconsistency_evaluator.filter_entities(entities=entities, min_samples = max_n))

        print(f"Total entities: {len(entities)}")
        print(f"Filtered entities: {len(filtered_entities)}")
        if len(filtered_entities) < 2:
            raise ValueError(
                "Not enough valid entities for evaluation."
            )

        n_values: List[int] = list(range(min_n, max_n + 1))
        results: Dict[str, List[float]] = {
            "N": [],
            "AUROC_mean": [],
            "AUROC_std": [],
            "ECE_mean": [],
            "ECE_std": []
        }

        for n in n_values:
            auroc_runs: List[float] = []
            ece_runs: List[float] = []

            for run in range(num_runs):
                rng: np.random.Generator = np.random.default_rng(seed + run)

                auroc, ece = (
                    budget_matched_selfconsistency_evaluator.calculate_metrics(
                        entities=filtered_entities,
                        n=n,
                        rng=rng,
                        n_bins=n_bins
                    )
                )

                if not np.isnan(auroc):
                    auroc_runs.append(auroc)

                ece_runs.append(ece)

            auroc_mean: float = (
                float(np.mean(auroc_runs))
                if auroc_runs else float("nan")
            )

            auroc_std: float = (
                float(np.std(auroc_runs))
                if auroc_runs else float("nan")
            )

            ece_mean: float = float(np.mean(ece_runs))
            ece_std: float = float(np.std(ece_runs))

            results["N"].append(n)
            results["AUROC_mean"].append(auroc_mean)
            results["AUROC_std"].append(auroc_std)
            results["ECE_mean"].append(ece_mean)
            results["ECE_std"].append(ece_std)

            print(
                f"N={n:3d} | "
                f"AUROC={auroc_mean:.4f} ± {auroc_std:.4f} | "
                f"ECE={ece_mean:.4f} ± {ece_std:.4f}"
            )

        return results

    @staticmethod
    def plot_results(results: Dict[str, List[float]], log_file_name: str, dpi: int = 300) -> None:
        x: NDArray[np.int_] = np.asarray(results["N"], dtype=int)
        
        auroc: NDArray[np.float64] = np.asarray(results["AUROC_mean"], dtype=float)
        auroc_std: NDArray[np.float64] = np.asarray(results["AUROC_std"], dtype=float)
        ece: NDArray[np.float64] = np.asarray(results["ECE_mean"], dtype=float)
        ece_std: NDArray[np.float64] = np.asarray(results["ECE_std"], dtype=float)

        # ----------------------------------
        # Figure 1: AUROC
        # ----------------------------------

        plt.figure(figsize=(9, 5))
        plt.plot(
            x, auroc,
            marker="o",
            markersize=3,
            label="Self-Consistency"
        )
        plt.fill_between(
            x,
            auroc - auroc_std,
            auroc + auroc_std,
            alpha=0.2
        )

        plt.xlabel("Number of Samples (N)")
        plt.ylabel("AUROC")
        plt.title("AUROC vs. Self-Consistency Sample Size")

        plt.xticks([5, 16, 32, 64, 96, 128])
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            log_file_name.replace('.csv', '_auroc.png'),
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close()

        # ----------------------------------
        # Figure 2: ECE
        # ----------------------------------

        plt.figure(figsize=(9, 5))

        plt.plot(
            x, ece,
            marker="o",
            markersize=3,
            label="Self-Consistency"
        )

        plt.fill_between(
            x,
            ece - ece_std,
            ece + ece_std,
            alpha=0.2
        )

        plt.xlabel("Number of Samples (N)")
        plt.ylabel("ECE")
        plt.title("ECE vs. Self-Consistency Sample Size")

        plt.xticks([5, 16, 32, 64, 96, 128])
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            log_file_name.replace('.csv', '_ece.png'),
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close()

