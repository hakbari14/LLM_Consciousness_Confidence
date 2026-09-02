from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger
from src.diffusion_decision_model.diffusion_decision_model import diffusion_decision_model

import torch
import math
import warnings
import os
import contextlib
import numpy as np 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.exceptions import ConvergenceWarning


class diffusion_decision_model_training: 

    # How the two loss channels of the feature are built.
    #
    #   LOSS_MODE_TOTAL     the Evidence_Accumulation_Loss the run wrote, which is
    #                       a sum of negative log probabilities over every token it
    #                       scored, so it grows with the length of that text.
    #   LOSS_MODE_PER_TOKEN the same average taken again with every loss first
    #                       divided by the number of tokens it was summed over, so
    #                       the channel is an average per token loss and no longer
    #                       carries the length of the text behind it.
    LOSS_MODE_TOTAL = 'total'
    LOSS_MODE_PER_TOKEN = 'per_token'

    # Which of the two answers a row is graded against.
    #
    #   TARGET_RUN   whether the answer the run itself gave was right. The evidence
    #                features describe that completion, so this is what they were
    #                built to predict.
    #   TARGET_VOTE  whether the answer the rollouts voted for was right. This is
    #                the usual way self consistency is scored, and it is a genuinely
    #                different question: the two answers differ on one sample in
    #                eight, so a number measured against one says nothing about the
    #                other. Report both rather than picking the flattering one.
    TARGET_RUN = 'run'
    TARGET_VOTE = 'vote'

    # How long the solver is allowed to search. The unscaled total loss needs
    # about 4800 rounds, because raw summed losses run into the thousands and
    # leave a long narrow valley to walk, so the usual thousand cut it off part
    # way and left its coefficients wherever they happened to be. The normalized
    # loss reaches the same place in about fifty.
    MAX_ITER = 10000

    # Which numbers describe a sample. Each is one group of channels per evidence
    # step, so with twenty steps the widths are 80, 20, 40, 20 and 20.
    #
    #   FULL                 the two accumulations and their two steps, which is
    #                        the feature the pipeline has always built.
    #   ROLLOUT_LENGTH       how long the ten rollouts ran, averaged. No loss and
    #                        no agreement, only length, which is the control every
    #                        loss channel has to beat before it can claim to carry
    #                        anything beyond how much text was scored.
    #   ROLLOUT_LENGTH_STD   the same average with the spread beside it, since a
    #                        run that wanders may say more than one that is long.
    #   AGREEING_LENGTH      the same average over only the rollouts that reached
    #                        the answer the run itself gave.
    #   SELF_CONSISTENCY     the agreement alone, with no length and no loss.
    #   BASELINE_SCALAR   5  how likely the model held its own answer to be and how
    #                        uncertain it was writing it, each as a total and per
    #                        token, with the loss of the completion beside them.
    #   BASELINE_HIDDEN      the mean pooled last hidden state, which asks whether
    #                        correctness can simply be read off the representation.
    FEATURE_SET_FULL = 'full'
    FEATURE_SET_ROLLOUT_LENGTH = 'rollout_length'
    FEATURE_SET_ROLLOUT_LENGTH_STD = 'rollout_length_std'
    FEATURE_SET_AGREEING_LENGTH = 'agreeing_length'
    FEATURE_SET_SELF_CONSISTENCY = 'self_consistency'
    FEATURE_SET_BASELINE_SCALAR = 'baseline_scalar'
    FEATURE_SET_BASELINE_HIDDEN = 'baseline_hidden'

    FEATURE_SETS = [FEATURE_SET_FULL, FEATURE_SET_ROLLOUT_LENGTH, FEATURE_SET_ROLLOUT_LENGTH_STD,
                    FEATURE_SET_AGREEING_LENGTH, FEATURE_SET_SELF_CONSISTENCY,
                    FEATURE_SET_BASELINE_SCALAR, FEATURE_SET_BASELINE_HIDDEN]

    # The last two describe the whole completion rather than any evidence step, and
    # they are the published baselines rather than anything of ours. They come from
    # one forward pass over the answer the run already wrote, where everything above
    # costs two hundred rollouts a sample, so they are the bar that decides whether
    # any of this is worth its compute.
    BASELINE_FEATURE_SETS = [FEATURE_SET_BASELINE_SCALAR, FEATURE_SET_BASELINE_HIDDEN]

    # Ten seeds rather than the single 42. They change the split of the random
    # split table, and for a held out dataset, where the split is fixed and the
    # solver is deterministic so the seed can change nothing at all, they redraw
    # the held out samples instead. Either way the tables report the average and
    # the spread, so a number that rests on one lucky draw shows itself.
    SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)

    # A held out set needs enough of the rarer class for the area under the curve
    # to mean anything, since that is what the curve rests on. math500 carries one
    # such sample and countdown three, so they are left out of the second average.
    MINORITY_FLOOR = 30

    # Benchmarks that ask the same kind of question, held out together so that none
    # of them can lean on a sibling. Holding mmlu out on its own leaves mmlu_pro in
    # the training half, which is very nearly the same task, so the number says less
    # about reaching a new domain than it appears to. Held out as a group there is
    # no sibling left, and the question becomes whether any of this survives a
    # change of domain rather than a change of benchmark.
    DATASET_GROUPS = {
        'multiple choice knowledge': ['mmlu', 'mmlu_pro'],
        'mathematics': ['gsm8k', 'math500', 'aime'],
        }

    def __init__(self, number_of_evidence: int, modelname_dir: str = 'qwen-qwen3-8b') -> None:
        self.number_of_evidence = number_of_evidence
        self.modelname_dir = modelname_dir
        if self.number_of_evidence is None:
            raise Exception('number of evidence is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datasets = ['gpqa', 'countdown', 'math500', 'gsm8k', 'mmlu', 'truthfulqa', 'mmlu_pro', 'aime']
        self.log_directory = '/home/hr_akbari/research/LLM_Consciousness_Confidence/logs/diffusion_decision_model'
        self.log_cache = {}

    def to_float(self, value) -> float:
        """Read a logged number, with anything missing coming back as a not a number.

        A gap in the csv reads back as None or as a not a number, and both of them
        print as a word, so one test on the text covers every way a field can be
        empty. A real zero still reads as '0' and is kept.
        """
        return float(value) if str(value).strip().lower() not in ('', 'nan', 'none') else float('nan')

    def get_log_file_name(self, dataset: str, run_number: int) -> str:
        return f'{self.log_directory}/{dataset}/{self.modelname_dir}/run_{run_number}/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv'

    def build_matrix(self, datasets: list[str], from_run_number: int, to_run_number: int, loss_mode: str = LOSS_MODE_TOTAL, target: str = TARGET_RUN, feature_set: str = FEATURE_SET_FULL):
        """Build the features and the labels for the given datasets.

        Every sample becomes one group of numbers per evidence step, and the
        feature set decides which group. The loss mode decides how the two loss
        numbers of the full set are built; the other sets carry no loss and ignore
        it.
        """
        row_list = []
        label_list = []
        baseline_list = []

        for dataset in datasets:
            for run_number in range(from_run_number, to_run_number):
                # A rollout log runs to hundreds of megabytes, and the ablation
                # reads every dataset once per configuration, so read each once.
                cache_key = (dataset, run_number)
                if cache_key not in self.log_cache:
                    logger = diffusion_decision_model_logger(log_file_name = self.get_log_file_name(dataset, run_number))
                    self.log_cache[cache_key] = logger.load_logs_list()
                    print(f'loaded {dataset} run {run_number}: {len(self.log_cache[cache_key])} samples')

                for log in self.log_cache[cache_key]:
                    evidence_list = sorted(log.evidence_list, key = lambda evidence_log: int(evidence_log.index))
                    if len(evidence_list) != self.number_of_evidence:
                        continue

                    accumulation_loss_list = []
                    accumulation_self_consistency_list = []
                    rollout_length_list = []
                    rollout_length_spread_list = []
                    agreeing_length_list = []
                    for evidence_log in evidence_list:
                        accumulation_self_consistency_list.append(self.to_float(evidence_log.evidence_accumulation_self_consistency))

                        # How long the rollouts of this step ran, over all ten and
                        # over only those that landed on the answer the run gave.
                        # A gap reads back as a not a number and counts as true, so
                        # ask for the word when testing whether one agreed.
                        length_list = [self.to_float(log_detail.token_count) for log_detail in evidence_log.consistency_list]
                        length_list = [value for value in length_list if not math.isnan(value)]
                        agreeing = [self.to_float(log_detail.token_count) for log_detail in evidence_log.consistency_list
                                    if str(log_detail.accuracy).strip().lower() == 'true']
                        agreeing = [value for value in agreeing if not math.isnan(value)]

                        rollout_length_list.append(float(np.mean(length_list)) if length_list else float('nan'))
                        rollout_length_spread_list.append(float(np.std(length_list)) if len(length_list) > 1 else float('nan'))
                        agreeing_length_list.append(float(np.mean(agreeing)) if agreeing else float('nan'))

                        if loss_mode == self.LOSS_MODE_TOTAL:
                            accumulation_loss_list.append(self.to_float(evidence_log.evidence_accumulation_loss))
                            continue

                        if loss_mode != self.LOSS_MODE_PER_TOKEN:
                            raise Exception(f'unknown loss mode {loss_mode}')

                        # Per token. The run averaged the loss of the rollouts that
                        # reached the same answer as the whole chain of thought,
                        # together with the loss of what was left of the original
                        # completion. Take that same average with every loss first
                        # divided by the number of tokens it was summed over, so the
                        # channel stops growing with the length of the text behind it.
                        #
                        # That token count is not in the log, but the arithmetic that
                        # produced it is fixed. Evidence zero has an empty prefix and
                        # carries the loss of the whole completion. After that the run
                        # cuts the completion into number_of_evidence + 1 groups of
                        # near equal size and scores whatever follows the first i of
                        # them, so what is left is the completion minus that prefix.
                        completion_token_count = self.to_float(log.token_count)
                        evidence_index = int(evidence_log.index)
                        if evidence_index == 0 or math.isnan(completion_token_count):
                            scored_token_count = completion_token_count
                        else:
                            group_count = self.number_of_evidence + 1
                            base = int(completion_token_count) // group_count
                            remainder = int(completion_token_count) % group_count
                            scored_token_count = completion_token_count - (evidence_index * base + max(0, min(evidence_index, remainder - 1)))

                        per_token_list = []
                        for log_detail in evidence_log.consistency_list:
                            # A gap reads back as a not a number, and that counts as
                            # true, so ask for the word rather than trust the value.
                            if str(log_detail.accuracy).strip().lower() != 'true':
                                continue

                            per_token_list.append(self.to_float(log_detail.loss) / self.to_float(log_detail.token_count)
                                                  if self.to_float(log_detail.token_count) > 0 else float('nan'))

                        per_token_list.append(self.to_float(evidence_log.partial_cot_loss) / scored_token_count
                                              if scored_token_count > 0 else float('nan'))

                        per_token_list = [value for value in per_token_list if not math.isnan(value)]
                        accumulation_loss_list.append(float(np.mean(per_token_list)) if per_token_list else float('nan'))

                    if feature_set == self.FEATURE_SET_BASELINE_SCALAR:
                        # Read from the completion, not from any evidence step.
                        row = [
                            self.to_float(log.completion_loss),
                            self.to_float(log.sequence_probability),
                            self.to_float(log.length_normalized_sequence_probability),
                            self.to_float(log.entropy),
                            self.to_float(log.mean_entropy),
                            ]
                        row_list.append(row)
                        self.append_labels(log, label_list, baseline_list)
                        continue

                    if feature_set == self.FEATURE_SET_BASELINE_HIDDEN:
                        # Thousands of numbers against about fourteen hundred
                        # samples, so this is the one set that can fit the training
                        # half far better than it can carry to anything else. A
                        # sample whose vector was never written is left out.
                        try:
                            row = list(log.get_last_layer_representations_numpy())
                        except Exception:
                            continue

                        row_list.append(row)
                        self.append_labels(log, label_list, baseline_list)
                        continue

                    row = []
                    for index, evidence_log in enumerate(evidence_list):
                        # The logged steps belong to the totals, so they are read as
                        # they are under the total mode, which keeps this feature
                        # exactly the one the pipeline already builds. Under any other
                        # mode they no longer describe the channel and are taken again
                        # from the accumulations. The signs follow the run: self
                        # consistency is measured as it rises, loss as it falls.
                        if loss_mode == self.LOSS_MODE_TOTAL:
                            delta_loss = self.to_float(evidence_log.delta_evidence_loss)
                            delta_self_consistency = self.to_float(evidence_log.delta_evidence_self_consistency)
                        elif index == 0:
                            delta_loss = 0.0
                            delta_self_consistency = 0.0
                        else:
                            delta_loss = accumulation_loss_list[index - 1] - accumulation_loss_list[index]
                            delta_self_consistency = accumulation_self_consistency_list[index] - accumulation_self_consistency_list[index - 1]

                        if feature_set == self.FEATURE_SET_FULL:
                            row.extend([
                                accumulation_loss_list[index],
                                accumulation_self_consistency_list[index],
                                delta_loss,
                                delta_self_consistency,
                                ])
                        elif feature_set == self.FEATURE_SET_ROLLOUT_LENGTH:
                            row.append(rollout_length_list[index])
                        elif feature_set == self.FEATURE_SET_ROLLOUT_LENGTH_STD:
                            row.extend([rollout_length_list[index], rollout_length_spread_list[index]])
                        elif feature_set == self.FEATURE_SET_AGREEING_LENGTH:
                            row.append(agreeing_length_list[index])
                        elif feature_set == self.FEATURE_SET_SELF_CONSISTENCY:
                            row.append(accumulation_self_consistency_list[index])
                        else:
                            raise Exception(f'unknown feature set {feature_set}')

                    row_list.append(row)
                    self.append_labels(log, label_list, baseline_list)


        if target not in (self.TARGET_RUN, self.TARGET_VOTE):
            raise Exception(f'unknown target {target}')

        X = np.array(row_list, dtype=float)
        labels = np.array(label_list, dtype=float)
        baseline = np.array(baseline_list, dtype=float)

        target_index = 0 if target == self.TARGET_RUN else 1
        keep = ~np.isnan(labels[:, target_index])
        return X[keep], labels[keep, target_index].astype(int), baseline[keep]

    def append_labels(self, log, label_list, baseline_list) -> None:
        """The two labels and the two untrained confidences of one sample.

        Both labels are kept and one is chosen at the end, because there are two
        answers that can be graded: the one the run gave, and the one the rollouts
        voted for. A sample with no vote has no second label and is dropped when
        that is the target, rather than counted as a wrong one. The vote share is
        carried alongside so a table can show what the same samples give with
        nothing fitted; a sample whose rollouts reached no readable answer has no
        vote to report, and its answer field is where that shows.
        """
        vote_accuracy = str(log.self_consistency_accuracy).strip().lower()
        label_list.append([
            1.0 if str(log.accuracy).strip().lower() == 'true' else 0.0,
            float('nan') if vote_accuracy in ('', 'nan', 'none') else (1.0 if vote_accuracy == 'true' else 0.0),
            ])

        baseline_row = []
        for confidence, answer in [(log.self_consistency_confidence, log.self_consistency_final_answer),
                                   (log.self_consistency_completion_confidence, log.self_consistency_completion_final_answer)]:
            has_answer = str(answer).strip().lower() not in ('', 'nan', 'none')
            baseline_row.append(self.to_float(confidence) if has_answer else float('nan'))

        # The published measures, taken the way their papers take them: as the score
        # itself, with nothing fitted. Malinin and Gales score a sequence by its log
        # likelihood and by that divided by its length; LM-Polygraph scores it by the
        # mean entropy of its tokens. Every one is turned so that larger means more
        # likely to be right, which for a loss or an entropy means negating it.
        # Completion_Loss is already the summed negative log likelihood of the
        # completion, so the first two are exact without needing anything rebuilt.
        completion_loss = self.to_float(log.completion_loss)
        token_count = self.to_float(log.token_count)
        baseline_row.extend([
            -completion_loss,
            -completion_loss / token_count if token_count > 0 else float('nan'),
            -self.to_float(log.entropy),
            -self.to_float(log.mean_entropy),
            self.to_float(log.length_normalized_sequence_probability),
            ])

        baseline_list.append(baseline_row)

    def evaluate(self, test_datasets: list[str] = None, from_run_number: int = 1, to_run_number: int = 2,
                 loss_mode: str = LOSS_MODE_TOTAL, standardize: bool = True, class_weight = None,
                 target: str = TARGET_RUN, feature_set: str = FEATURE_SET_FULL, seeds = SEEDS,
                 verbose: bool = False) -> dict:
        """Train on one set of datasets and score on another, averaged over seeds.

        Name the datasets to hold out and they are kept out of training entirely,
        which asks whether the method carries to a benchmark it has never seen.
        Leave them out and it falls back to a random split over everything, the
        easier question, and the reference the held out numbers are read against.

        Everything the model learns about scale comes from the training half only,
        the column means that fill the gaps as much as the standardiser. Fitting
        either of them on the held out set would hand it the answer sheet.

        The seeds mean different things on the two paths, because the randomness
        does. On the random split they redraw the split, so each seed is a fresh
        model. On a held out dataset there is nothing for a seed to change: the
        split is decided by which dataset was named and the solver is
        deterministic, so every seed returns the identical fit. There the model is
        built once and the seeds redraw the held out samples instead, which
        answers the question actually worth asking, how much the number rests on
        which problems the benchmark happens to contain.
        """
        test_datasets = list(test_datasets or [])
        held_out_run = bool(test_datasets)
        if held_out_run:
            train_datasets = [dataset for dataset in self.datasets if dataset not in test_datasets]
            if not train_datasets:
                raise Exception('every dataset was held out, nothing is left to train on')

            X_train, y_train, baseline_train = self.build_matrix(train_datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            X_test, y_test, baseline_test = self.build_matrix(test_datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            held_out = ','.join(test_datasets)
        else:
            X, y, baseline = self.build_matrix(self.datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            held_out = 'random split'

        def fit_and_predict(X_fit, y_fit, X_score):
            """Fill, scale and fit on the training half, then score the other half."""
            column_mean = np.nanmean(X_fit, axis=0)
            column_mean = np.where(np.isnan(column_mean), 0.0, column_mean)
            X_fit, X_score = np.array(X_fit, dtype=float), np.array(X_score, dtype=float)
            for matrix in (X_fit, X_score):
                missing_position = np.isnan(matrix)
                if missing_position.any():
                    matrix[missing_position] = np.take(column_mean, np.where(missing_position)[1])

            if standardize:
                scaler = StandardScaler().fit(X_fit)
                X_fit, X_score = scaler.transform(X_fit), scaler.transform(X_score)

            # The solver prints a warning for every fit it cannot finish, which
            # would bury the table, so it is silenced and answered by the column
            # instead: a row that stopped early is worth less than one that did not.
            model = LogisticRegression(max_iter = self.MAX_ITER, random_state = 42, class_weight = class_weight)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category = ConvergenceWarning)
                model.fit(X_fit, y_fit)

            return model.predict(X_score), model.predict_proba(X_score)[:, 1], bool(np.all(np.asarray(model.n_iter_) < self.MAX_ITER))

        measurement_list = []
        converged = True
        if held_out_run:
            y_pred, y_prob, converged = fit_and_predict(X_train, y_train, X_test)
            for seed in seeds:
                draw = np.arange(len(y_test)) if seed == seeds[0] else np.random.default_rng(seed).integers(0, len(y_test), len(y_test))
                measurement_list.append(self.measure(y_test[draw], y_pred[draw], y_prob[draw]))

            train_count, test_count = len(y_train), len(y_test)
        else:
            for seed in seeds:
                X_train, X_test, y_train, y_test, baseline_train, baseline_test = train_test_split(X, y, baseline, test_size=0.2, random_state=seed, stratify=y)
                y_pred, y_prob, fit_ok = fit_and_predict(X_train, y_train, X_test)
                converged = converged and fit_ok
                measurement_list.append(self.measure(y_test, y_pred, y_prob))

            train_count, test_count = len(y_train), len(y_test)

        if verbose:
            print("\n===== Classification Report =====")
            print(classification_report(y_test, y_pred, zero_division=0))
            print("===== Confusion Matrix =====")
            print(confusion_matrix(y_test, y_pred))

        # The last draw of the loop is the one whose samples the untrained rows and
        # the class counts describe, so they are read from it.
        # The untrained rows. The two vote shares are proportions, so a threshold and
        # a calibration curve mean something for them. The published measures are log
        # likelihoods and entropies, which are not probabilities of anything, so only
        # the ranking is reported for those and the rest is left empty rather than
        # filled with a number that would read as if it meant something.
        baseline_rows = []
        for position, name, is_probability in [
                (0, 'self cons 0', True),
                (1, 'self cons last', True),
                (2, 'seq logprob', False),
                (3, 'seq logprob/tok', False),
                (4, 'entropy total', False),
                (5, 'mean token ent', False),
                (6, 'mean token prob', True),
                ]:
            confidence = baseline_test[:, position]
            keep = ~np.isnan(confidence)
            calibration = (baseline_train[:, position], y_train)
            baseline_rows.append(self.score_confidence(held_out, name, confidence[keep], y_test[keep], target, is_probability, calibration))

        result = {
            'baseline_rows': baseline_rows,
            'held_out': held_out,
            'target': target,
            'feature_set': feature_set,
            'loss_mode': loss_mode if feature_set == self.FEATURE_SET_FULL else '-',
            'standardize': standardize,
            'class_weight': class_weight if class_weight else 'none',
            'train_count': train_count,
            'test_count': test_count,
            'minority_count': int(min(np.sum(y_test == 1), np.sum(y_test == 0))),
            'majority_rate': float(max(np.mean(y_test), 1.0 - np.mean(y_test))),
            'converged': converged,
            }

        for name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ece']:
            values = [measurement[name] for measurement in measurement_list]
            result[name] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float('nan')
            result[name + '_sd'] = float(np.nanstd(values)) if not np.all(np.isnan(values)) else float('nan')

        return result

    def measure(self, y_true, y_pred, y_prob) -> dict:
        """Every score of one draw. The curve needs both classes, so it can be absent."""
        try:
            ece, _ = self.calculate_ECE_MCE(y_true, y_prob)
        except Exception:
            ece = float('nan')

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan'),
            'ece': ece,
            }

    def score_confidence(self, held_out: str, name: str, confidence, y_true, target: str, is_probability: bool = True, calibration = None) -> dict:
        """Score a confidence that needed no training, in the shape of a trained row.

        A log likelihood or an entropy ranks samples perfectly well but is not a
        probability of anything, so a calibration error cannot be read off it as it
        stands, and cutting it at one half would mean nothing. Such a score is first
        put through Platt scaling: one logistic curve from that single number to
        correctness, fitted on the training half, the same half every trained row
        learns from, and applied here. The curve only ever rises, so the ranking and
        the area under it are untouched to the last decimal; all it does is put the
        score on the probability scale where a threshold and a calibration error
        mean something. A score that is already a proportion is left alone.
        """
        roc_auc = float(roc_auc_score(y_true, confidence)) if len(np.unique(y_true)) > 1 and len(y_true) else float('nan')

        def unscorable():
            return {
                'baseline_rows': [], 'held_out': held_out, 'target': target, 'feature_set': '-',
                'loss_mode': name, 'standardize': '-', 'class_weight': '-', 'train_count': 0,
                'test_count': len(y_true),
                'minority_count': int(min(np.sum(y_true == 1), np.sum(y_true == 0))) if len(y_true) else 0,
                'majority_rate': float(max(np.mean(y_true), 1.0 - np.mean(y_true))) if len(y_true) else float('nan'),
                'accuracy': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan'),
                'roc_auc': roc_auc, 'ece': float('nan'), 'roc_auc_sd': 0.0, 'ece_sd': 0.0, 'converged': True,
                }

        if not is_probability:
            if calibration is None or not len(y_true):
                return unscorable()

            train_confidence, train_label = calibration
            train_confidence = np.asarray(train_confidence, dtype=float)
            keep = ~np.isnan(train_confidence)
            if keep.sum() < 2 or len(np.unique(np.asarray(train_label)[keep])) < 2:
                return unscorable()

            # These scores run to the thousands, so the curve is fitted on a
            # standardised copy to keep the solver on comfortable ground.
            scaler = StandardScaler().fit(train_confidence[keep].reshape(-1, 1))
            calibrator = LogisticRegression(max_iter = self.MAX_ITER, random_state = 42)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category = ConvergenceWarning)
                calibrator.fit(scaler.transform(train_confidence[keep].reshape(-1, 1)), np.asarray(train_label)[keep])

            confidence = calibrator.predict_proba(scaler.transform(np.asarray(confidence, dtype=float).reshape(-1, 1)))[:, 1]

        try:
            ece, _ = self.calculate_ECE_MCE(y_true, confidence)
        except Exception:
            ece = float('nan')

        y_pred = (np.asarray(confidence) >= 0.5).astype(int)
        return {
            'baseline_rows': [],
            'held_out': held_out,
            'target': target,
            'feature_set': '-',
            'loss_mode': name,
            'standardize': '-',
            'class_weight': '-',
            'train_count': 0,
            'test_count': len(y_true),
            'minority_count': int(min(np.sum(y_true == 1), np.sum(y_true == 0))) if len(y_true) else 0,
            'majority_rate': float(max(np.mean(y_true), 1.0 - np.mean(y_true))) if len(y_true) else float('nan'),
            'accuracy': accuracy_score(y_true, y_pred) if len(y_true) else float('nan'),
            'precision': precision_score(y_true, y_pred, zero_division=0) if len(y_true) else float('nan'),
            'recall': recall_score(y_true, y_pred, zero_division=0) if len(y_true) else float('nan'),
            'f1': f1_score(y_true, y_pred, zero_division=0) if len(y_true) else float('nan'),
            'roc_auc': roc_auc,
            'ece': ece,
            'roc_auc_sd': 0.0,
            'ece_sd': 0.0,
            'converged': True,
            }

    def ablation(self, test_datasets: list[str] = None, from_run_number: int = 1, to_run_number: int = 2,
                 feature_set: str = FEATURE_SET_FULL, seeds = SEEDS) -> list[dict]:
        """Every combination of the switches, on the same held out set, for both targets.

        Per target: the loss taken as a total or per token, the features
        standardised or left alone, the classes weighted evenly or by their size,
        then the two untrained rows. Within one target every row sees the same
        samples, so the difference between any two of them is the switch that
        changed and nothing else. Across the two targets the samples differ, and
        so does the question, so read those blocks separately.
        """
        # Only the full set carries a loss, so there is nothing for the other sets
        # to sweep and repeating them under both modes would print the same row twice.
        loss_modes = [self.LOSS_MODE_TOTAL, self.LOSS_MODE_PER_TOKEN] if feature_set == self.FEATURE_SET_FULL else [self.LOSS_MODE_TOTAL]

        results = []
        for target in [self.TARGET_RUN, self.TARGET_VOTE]:
            trained = []
            for loss_mode in loss_modes:
                for standardize in [False, True]:
                    for class_weight in [None, 'balanced']:
                        trained.append(self.evaluate(test_datasets, from_run_number, to_run_number, loss_mode, standardize, class_weight, target, feature_set, seeds))

            # The untrained rows do not depend on the switches, so take them from
            # the first fit and close each target's block with them.
            results.extend(trained)
            results.extend(trained[0]['baseline_rows'])

        held = ','.join(test_datasets) if test_datasets else 'random split'
        self.print_results(results, f'ablation, features: {feature_set}, held out: {held}')
        return results

    def train_logistic_regression(self, from_run_number, to_run_number, loss_mode: str = LOSS_MODE_TOTAL, standardize: bool = False, class_weight = None, target: str = TARGET_RUN) -> dict:
        """The random split over every dataset, which is what this has always run."""
        result = self.evaluate(None, from_run_number, to_run_number, loss_mode, standardize, class_weight, target, verbose = True)
        self.print_results([result], 'random split over every dataset')
        return result

    def print_reading_notes(self) -> None:
        """What someone opening the results needs to know before reading a number."""
        print('=' * 94)
        print('== how to read this')
        print('=' * 94)
        print("""
Produced by:
    .venv/bin/python -m src.diffusion_decision_model.diffusion_decision_model_training

One table per held out setting. The first is a random 80/20 split over every
dataset, which is the reference. The rest hold one dataset out completely: it is
kept out of training and is the only thing scored, which asks whether any of this
carries to a benchmark the model has never seen.

Each table has two blocks, one per graded answer, and each block has the
configurations followed by two rows that fit nothing at all. There is one file
per feature set, named after it, since the feature sets answer separate
questions and are not rows of one table.

Held out one benchmark at a time, a benchmark that has a close relative can lean
on it: mmlu held out alone still leaves mmlu_pro in the training half, and the
three mathematics sets cover for each other the same way. The tables at the foot
of each file hold whole domains out instead, mmlu with mmlu_pro and gsm8k with
math500 and aime, which is the harder and more honest question. Read the drop
between a benchmark held out alone and its domain held out whole as the part of
the earlier number that was a sibling doing the work. Holding the mathematics
sets out together also makes them readable for the first time: alone they carry
1, 13 and 17 of the rarer class, and together they carry 31.

The last average is the one to quote. It holds every benchmark out exactly once,
the two domains together and gpqa, truthfulqa and countdown on their own, so each
of the eight counts once and none of them sits on both sides of the split. The
average over the eight single hold outs above it cannot say either of those
things: it counts math500, whose rarer class holds one sample, as heavily as
mmlu_pro, whose rarer class holds a hundred and five, which lifts it by about a
tenth of a point of area under the curve for no reason anyone should trust.

features  which numbers describe a sample, one group per evidence step, so with
          twenty steps the widths are:
            full                80  the two accumulations and their two steps,
                                    the feature this pipeline has always built
            rollout_length      20  how long the ten rollouts ran, averaged.
                                    No loss and no agreement, only length. This
                                    is the control: a loss channel that cannot
                                    beat it is not carrying anything beyond how
                                    much text it happened to score
            rollout_length_std  40  that average with its spread beside it
            agreeing_length     20  the same average over only the rollouts that
                                    reached the answer the run itself gave
            self_consistency    20  the agreement alone, no length, no loss
            baseline_scalar      5  the published confidence signals, read from the
                                    completion the run already wrote: the loss of
                                    it, how likely the model held its own answer to
                                    be, and how uncertain it was writing it, each as
                                    a total and per token
            baseline_hidden   4096  the mean pooled last hidden state, asking
                                    whether correctness can be read straight off the
                                    representation
          Only full carries a loss, so only its tables sweep the loss column; the
          others show a dash there and have four configurations per block.

          The last two are the ones that decide whether any of this is worth its
          price. They need one forward pass over an answer the model has already
          written. Everything above them needs two hundred rollouts a sample. A
          feature set of ours that does not clearly beat them is not a result.

          Note that baseline_scalar trains a classifier over those five numbers,
          which is not how the papers that introduced them use them. There they are
          the score itself, with nothing fitted, and that is how they appear in the
          untrained rows at the foot of every block. Read those rows as the real
          baseline. The trained set is kept beside them because it answers a
          different and narrower question, whether a classifier can do better with
          the same five numbers, and on a held out benchmark it can do considerably
          worse: a direction learned on one set of benchmarks can point the wrong
          way on another, which a raw score cannot do.
          baseline_hidden carries four thousand numbers against about fourteen
          hundred training samples, so it can fit the training half far better than
          it can carry to anything else; read its held out rows, not its fit.

target    which of the two answers the row was graded against. The run produces
          two: the one it wrote in its single completion, and the one its ten
          rollouts voted for. They disagree on one sample in eight, so the two
          blocks answer different questions and their numbers are not comparable
          with each other. Use run if the system returns the completion, vote if
          it returns the majority answer.

loss      total is the Evidence_Accumulation_Loss the run wrote, a sum of
          negative log probabilities over every token it scored, so it grows with
          the length of that text. per_token divides each loss by the number of
          tokens behind it first. The self cons rows fit nothing at all: they take
          the vote share itself as the confidence. Those two are the bar. A
          trained row that does not clear them is not worth the compute it costs.

scaled    whether the features were standardised. The standardiser and the column
          means that fill gaps are fitted on the training half only, never on the
          held out set.

weight    balanced makes the classifier weigh the rarer class as heavily as the
          common one. It buys a little ranking and costs a lot of calibration, so
          read its ECE before believing its ROC.

fit       whether the solver finished. ok means it stopped because it had found
          the answer, STOP means it used its whole budget and was cut off with the
          coefficients wherever they had got to. The budget is max_iter=10000,
          raised from the usual thousand because the unscaled total loss needs
          about 4800 rounds: raw summed losses run into the thousands and the four
          channels sit on very different scales, which leaves a long narrow valley
          to walk. The normalized loss reaches the same place in about fifty. At
          the old limit those rows were cut off and scored 0.7444, slightly above
          the 0.7360 they settle at once allowed to finish, so stopping early had
          been flattering them rather than holding them back.

train     how many samples were fitted on, and how many were scored. They differ
test      between the two blocks because a sample with no vote has no second
          label and is dropped rather than counted as a wrong one.

minority  how many of the rarer class the scored set holds. READ THIS FIRST. The
          area under the curve rests on those samples alone. math500 holds 1 and
          countdown holds 3, so their ROC means nothing whatever, including the
          row that reports 1.0000. Only gpqa (91), mmlu_pro (105), truthfulqa (85)
          and mmlu (50) carry enough of both classes to be worth reading.

majority  the share of the commoner class, which is the accuracy of answering the
          same thing every time. It is identical down a whole block because it
          describes the data, not the model. An accuracy below it means the row
          lost to doing nothing.

accuracy  the CLASSIFIER's hit rate, not the language model's answer accuracy. A
          sample whose answer was wrong and which the classifier correctly flagged
          as wrong counts as a hit here.

ROC       ranking and calibration. ROC is unmoved by a constant shift in the
ECE       predicted probabilities and ECE is not, which is why balanced can look
          fine on one and bad on the other.

ROCsd     how far the score moved over ten seeds, and what a seed does depends on
ECEsd     the row. On the random split it redraws the split, so every seed is a
          fresh model and the spread is the spread over models. On a held out
          dataset a seed can change nothing at all: the split is decided by which
          dataset was named and the solver is deterministic, so all ten seeds
          return the identical fit. There the seeds redraw the held out samples
          instead, and the spread answers the question actually worth asking,
          how much the number rests on which problems the benchmark happens to
          contain. Two rows whose difference is smaller than their spread are not
          different. The untrained rows read 0.0000 because they are a fixed
          confidence over a fixed set with nothing to redraw.

what the tables say, averaged over the four held out sets with enough of the
rarer class to trust:

  per_token beats total under both targets and takes the top rows of each block
  the trained rows beat the vote share by +0.16 on target run, +0.09 on vote
  under target vote, plain self consistency beats every total configuration
  self cons last is near chance on both targets, far below self cons 0

worth knowing before trusting any of it:

  the last evidence step covers about 92 percent of the reasoning rather than all
  of it, because the chunk loop stops two groups short of the end. changing that
  needs the generation run done again, it cannot be repaired from the logs
  these come from logs repaired after the vote counting fix. before it, the
  rollouts that reached no readable answer grouped together and won the vote,
  which was written out as a confidence of 1.0 for an answer of nan on 131 samples
  every table is one generation run of one model. the spread columns cover the
  seed and the sample draw, nothing else: a second generation run of the same
  model would move these further than any of the switches do
""")

    def print_results(self, results: list[dict], caption: str) -> None:
        width = 158
        print('\n' + '=' * width)
        print(f'== {caption}')
        print('=' * width)
        print(f"{'held out':<16}{'target':>6} {'loss':<16}{'scaled':>7}{'weight':>10}{'fit':>5}{'train':>7}{'test':>6}{'minority':>9}"
              f"{'majority':>9}{'accuracy':>9}{'precision':>10}{'recall':>8}{'F1':>8}{'ROC':>8}{'ROCsd':>8}{'ECE':>8}{'ECEsd':>8}")
        print('-' * width)
        for result in results:
            print(f"{result['held_out']:<16}{result['target']:>6} {result['loss_mode']:<16}{str(result['standardize']):>7}{str(result['class_weight']):>10}"
                  f"{('ok' if result['converged'] else 'STOP'):>5}"
                  f"{result['train_count']:>7}{result['test_count']:>6}{result['minority_count']:>9}{result['majority_rate']:>9.3f}"
                  f"{result['accuracy']:>9.4f}{result['precision']:>10.4f}{result['recall']:>8.4f}{result['f1']:>8.4f}"
                  f"{result['roc_auc']:>8.4f}{result['roc_auc_sd']:>8.4f}{result['ece']:>8.4f}{result['ece_sd']:>8.4f}")
        print('-' * width)
        print('minority is how many of the rarer class the held out set holds. The area under the curve')
        print('rests on those alone, so a handful of them means the number is mostly noise.')
        print('fit says whether the solver settled. STOP means it ran out of iterations and stopped')
        print('wherever it had got to, so read that row as a weaker result, not a comparable one.')
        print('the last rows of each block fit nothing at all. They are the bar the trained rows')
        print('have to clear to be worth the compute they cost. self cons 0 and self cons last are')
        print('the vote share at the first and last evidence. The rest are the published measures,')
        print('taken as their papers take them, as the score itself rather than as something to')
        print('train on: seq logprob and seq logprob/tok are the sequence log likelihood and the')
        print('same per token (Malinin and Gales), entropy total and mean token ent are the summed')
        print('and averaged predictive entropy (mean token entropy is the LM-Polygraph measure),')
        print('mean token prob is the average probability the model gave its own tokens. All are')
        print('turned so larger means more likely correct. Four of them are log likelihoods or')
        print('entropies rather than probabilities, so a calibration error cannot be read off')
        print('them as they stand. Those four are put through Platt scaling first: one logistic')
        print('curve from the single number to correctness, fitted on the training half, the same')
        print('half every trained row learns from, and applied to the held out half. The curve')
        print('only rises, so the ranking and the area under it are unchanged to the last decimal')
        print('and only the accuracy and calibration columns become readable. This is why their')
        print('ROC is the same whether the scaling is applied or not, and why their ECE should be')
        print('read as the calibration of the scaled score rather than of the raw measure.')
        print('target says which answer was graded: run is the answer the model actually gave,')
        print('vote is the answer the rollouts settled on. They differ on one sample in eight, so')
        print('the two blocks answer different questions and their numbers are not interchangeable.')

    def build_confidence_arrays(self, log_list: list[diffusion_decision_model_log_entity], confidence_attribute: str, accuracy_attribute: str, answer_attribute: str):
        confidence_list = []
        label_list = []
        skipped_count = 0

        for log in log_list:
            # A sample with no vote has to be left out. The answer is checked as
            # well as the confidence, because runs written before the vote filter
            # was fixed let the rollouts that reached no readable answer group
            # together and win, which was logged as a confidence of one for an
            # answer of 'nan'. Those carry a confidence that looks measured, so
            # the answer is the only field that gives them away. A gap reads back
            # as a not a number, which counts as true, so ask for the word.
            confidence = getattr(log, confidence_attribute)
            accuracy = getattr(log, accuracy_attribute)
            answer = getattr(log, answer_attribute)
            if any(str(value).strip().lower() in ('', 'nan', 'none') for value in (confidence, accuracy, answer)):
                skipped_count += 1
                continue

            confidence_list.append(float(confidence))
            label_list.append(1 if str(accuracy).strip().lower() == 'true' else 0)

        if skipped_count:
            print(f'[WARN] {skipped_count} samples skipped, they have no self consistency vote to score')

        return np.array(confidence_list, dtype=float), np.array(label_list, dtype=int)

    def self_consistency_confidence_completion(self, from_run_number, to_run_number) -> None:
        log_list: list[diffusion_decision_model_log_entity] = []
        
        X = np.empty(0)
        y = np.empty(0)
        for dataset in self.datasets:
            for run_number in range(from_run_number,to_run_number):
                logger = diffusion_decision_model_logger(log_file_name = self.get_log_file_name(dataset, run_number))
                log_list = logger.load_logs_list()
                
                X_b, y_b = self.build_confidence_arrays(log_list, 'self_consistency_completion_confidence', 'self_consistency_completion_accuracy', 'self_consistency_completion_final_answer')
                
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
                logger = diffusion_decision_model_logger(log_file_name = self.get_log_file_name(dataset, run_number))
                log_list = logger.load_logs_list()
                
                X_b, y_b = self.build_confidence_arrays(log_list, 'self_consistency_confidence', 'self_consistency_accuracy', 'self_consistency_final_answer')
                
                X = np.concatenate((X, X_b))                
                y = np.concatenate((y, y_b))                
        
        fpr, tpr, _ = roc_curve(y, X)
        roc_auc = auc(fpr, tpr)

        ece, _ = self.calculate_ECE_MCE(y, X)        
        print(f"Accuracy : {np.mean(y):.4f}")
        print(f"ROC : {roc_auc:.4f}")
        print(f"ECE : {ece:.4f}")
        

    def calculate_grouped_averages(self, data: list[list[dict]]) -> None:
        parameter_keys = ["target", "loss_mode", "standardize", "class_weight"]
        calculation_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "ece"]

        records = [
            item
            for inner_list in data
            for item in inner_list
        ]

        df = pd.DataFrame(records)
        result = (
            df.groupby(parameter_keys, dropna=False)[calculation_keys]
            .mean()
            .round(3)            
            .reset_index()
        )
        
        df_summary = pd.DataFrame(result)
        print()
        print(df_summary.to_string(index=False))

    def calculate_ECE_MCE(df, y_list, confidence_list, n_bins = 10):
        df = pd.DataFrame({
                "confidence": confidence_list,
                "accuracy_reward": y_list
            })
             
        # A coarse confidence, such as a vote share out of ten where most samples
        # land on one value, leaves qcut with a bin that no sample falls into. Kept
        # as a category, that empty bin has a mean of not a number, and it poisons
        # the sum so the whole error comes back empty. Counting only the bins that
        # have samples in them is the same calculation everywhere else and gives an
        # answer here too.
        df['binned_confidence'] = pd.qcut(df['confidence'], q=n_bins, duplicates='drop')
        agg_perplexity = df.groupby('binned_confidence', observed=True)['confidence'].agg(['mean'])
        agg_accuracy = df.groupby('binned_confidence', observed=True)['accuracy_reward'].agg(['mean'])

        expected_calibration_error = 0
        maximum_calibration_error = 0
        for idx, row in enumerate(agg_perplexity.iterrows()):
            confidence = row[1]['mean']
            accuracy = agg_accuracy.iloc[idx]['mean']
            expected_calibration_error += abs(confidence - accuracy)
            maximum_calibration_error = max(abs(confidence - accuracy), maximum_calibration_error)

        expected_calibration_error = expected_calibration_error / (idx + 1)
        return expected_calibration_error, maximum_calibration_error


if __name__ == '__main__':
    training = diffusion_decision_model_training(number_of_evidence=20)

    # -- how to train ---------------------------------------------------------
    # Every call reads run 1 only. Widen the run range to pool several runs.
    #
    # the random split over every dataset, the way this has always been run
    # training.train_logistic_regression(from_run_number = 1, to_run_number = 2)
    #
    # hold gpqa out completely, train on the other seven, score only on gpqa
    # training.evaluate(test_datasets = ['gpqa'])
    #
    # the same with the loss taken per token rather than as a total
    # training.evaluate(test_datasets = ['gpqa'], loss_mode = training.LOSS_MODE_PER_TOKEN)
    #
    # grade the answer the rollouts voted for instead of the one the run gave
    # training.evaluate(test_datasets = ['gpqa'], target = training.TARGET_VOTE)
    #
    # describe the samples by rollout length alone, the control for the loss
    # training.evaluate(test_datasets = ['gpqa'], feature_set = training.FEATURE_SET_ROLLOUT_LENGTH)
    #
    # all configurations on one held out set, with the untrained rows below them
    # training.ablation(test_datasets = ['gpqa'])
    #
    # the self consistency vote share on its own, scored over every dataset
    # training.self_consistency_confidence(from_run_number = 1, to_run_number = 2)
    # training.self_consistency_confidence_completion(from_run_number = 1, to_run_number = 2)
    # -------------------------------------------------------------------------

    # The whole sweep, one file per feature set, since the feature sets answer
    # separate questions and do not belong in one table. Within a file: the random
    # split first as the reference, then every dataset held out in turn, each
    # under every configuration, each block closed by the vote share with nothing
    # fitted, and the averages over the held out datasets at the foot.
    output_directory = 'src/diffusion_decision_model/ablations'
    os.makedirs(output_directory, exist_ok=True)

    for feature_set in diffusion_decision_model_training.FEATURE_SETS:
        output_file_name = f'{output_directory}/ablation_{feature_set}.txt'
        with open(output_file_name, 'w') as handle, contextlib.redirect_stdout(handle):
            training.print_reading_notes()
            training.ablation(feature_set = feature_set)

            total_result = []
            for dataset in training.datasets:
                total_result.append(training.ablation(test_datasets = [dataset], feature_set = feature_set))

            print('\n\naveraged over every held out dataset:')
            training.calculate_grouped_averages(total_result)

            # The average above counts math500 and countdown, whose rarer class
            # holds one sample and three, so their area under the curve is noise
            # folded in with the rest. This one keeps only the held out sets that
            # carry enough of both classes for the curve to mean anything.
            enough = [result for result in total_result if result[0]['minority_count'] >= training.MINORITY_FLOOR]
            names = [dataset for dataset, result in zip(training.datasets, total_result) if result[0]['minority_count'] >= training.MINORITY_FLOOR]
            print(f"\n\naveraged over the held out datasets carrying at least {training.MINORITY_FLOOR} of the rarer class ({', '.join(names)}):")
            training.calculate_grouped_averages(enough)

            # The same again with whole domains held out rather than single
            # benchmarks, which is the harder question: one benchmark held out on
            # its own can still lean on a sibling left in the training half.
            group_result = []
            for group_name, group in training.DATASET_GROUPS.items():
                print(f'\n\nheld out as a domain: {group_name}')
                group_result.append(training.ablation(test_datasets = group, feature_set = feature_set))

            # The partition. Every benchmark is held out exactly once: those with a
            # close relative leave together with it, the rest leave alone. Because
            # no benchmark appears twice, this average counts each of them once and
            # no problem sits on both sides of it, which the average over the eight
            # single hold outs cannot say. The tables it draws on are all printed
            # above, so nothing is fitted a second time to build it.
            partition = list(group_result)
            partition_names = list(training.DATASET_GROUPS)
            for dataset, result in zip(training.datasets, total_result):
                if any(dataset in group for group in training.DATASET_GROUPS.values()):
                    continue

                partition.append(result)
                partition_names.append(dataset)

            print(f"\n\naveraged over every benchmark held out exactly once ({', '.join(partition_names)}):")
            training.calculate_grouped_averages(partition)

        print(f'wrote {output_file_name}')
