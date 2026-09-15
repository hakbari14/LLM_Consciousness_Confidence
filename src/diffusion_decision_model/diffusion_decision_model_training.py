from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger

import os
import math
import contextlib
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning


MISSING = ('', 'nan', 'none')


def to_float(value) -> float:
    """A logged number, or nan when the field was empty."""
    return float(value) if str(value).strip().lower() not in MISSING else float('nan')


def is_present(value) -> bool:
    return str(value).strip().lower() not in MISSING


def is_true(value) -> bool:
    return str(value).strip().lower() == 'true'


def mean_or_nan(values) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(np.mean(values)) if values else float('nan')


class diffusion_decision_model_training:

    # The two ways of reading the evidence loss.  Total is what the run logged;
    # per token divides it by the tokens behind it so it stops growing with length.
    LOSS_TOTAL = 'evidence_total_loss'
    LOSS_PER_TOKEN = 'evidence_per_token_loss'

    # Which answer is graded.  run: the one the model wrote.  vote: the one the ten
    # rollouts agreed on.  They differ on one sample in eight.
    TARGET_RUN = 'run'
    TARGET_VOTE = 'vote'

    # Which numbers describe a sample.  Widths with twenty evidence steps:
    #   full 80, rollout_length 20, rollout_length_std 40, agreeing_length 20,
    #   evidence_loss 20, self_consistency 20, baseline_scalar 5, baseline_hidden 4096
    #
    # evidence_loss and self_consistency are the two halves of full, each one
    # channel of twenty steps, so putting them side by side says which channel the
    # full set is living on.
    FULL = 'full'
    ROLLOUT_LENGTH = 'rollout_length'
    ROLLOUT_LENGTH_STD = 'rollout_length_std'
    AGREEING_LENGTH = 'agreeing_length'
    EVIDENCE_LOSS = 'evidence_loss'
    SELF_CONSISTENCY = 'self_consistency'
    BASELINE_SCALAR = 'baseline_scalar'
    BASELINE_HIDDEN = 'baseline_hidden'

    FEATURE_SETS = [FULL, ROLLOUT_LENGTH, ROLLOUT_LENGTH_STD, AGREEING_LENGTH,
                    EVIDENCE_LOSS, SELF_CONSISTENCY, BASELINE_SCALAR, BASELINE_HIDDEN]

    # The sets whose features are read from the evidence loss, so the loss column
    # means something and both ways of reading it are worth sweeping.
    LOSS_BEARING_SETS = [FULL, EVIDENCE_LOSS]

    # The two baselines read the whole completion, not the evidence steps.
    WHOLE_COMPLETION_SETS = [BASELINE_SCALAR, BASELINE_HIDDEN]

    # The untrained confidences carried beside every sample, in column order.
    BASELINE_NAMES = ['baseline self cons 0', 'baseline self cons last',
                      'baseline cot loss', 'baseline cot loss/tok',
                      'baseline entropy total', 'baseline mean token ent',
                      'baseline arith mean prob', 'baseline budget self cons']

    METRICS = ['roc_auc', 'ece', 'ece_minmax']

    # The unscaled total loss needs about 4800 rounds; per_token needs about 50.
    MAX_ITER = 10000

    # On a random split a seed redraws the split.  On a held out dataset nothing is
    # random, so a seed redraws the held out samples instead.
    SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)

    # A held out set with fewer of the rarer class than this cannot support a ROC.
    MINORITY_FLOOR = 30

    # Benchmarks of the same kind, held out together so none can lean on a sibling.
    DATASET_GROUPS = {
        'multiple choice knowledge': ['mmlu', 'mmlu_pro'],
        'mathematics': ['gsm8k', 'math500', 'aime'],
        }

    def __init__(self, number_of_evidence: int, modelname_dir: str = 'qwen-qwen3-8b') -> None:
        if number_of_evidence is None:
            raise Exception('number of evidence is required')

        self.number_of_evidence = number_of_evidence
        self.modelname_dir = modelname_dir
        self.datasets = ['gpqa', 'countdown', 'math500', 'gsm8k', 'mmlu', 'truthfulqa', 'mmlu_pro', 'aime']
        self.log_directory = '/home/hr_akbari/research/LLM_Consciousness_Confidence/logs/diffusion_decision_model'
        self.log_cache = {}
        self.budget_cache = {}

    # ------------------------------------------------------------------ loading

    def log_file_name(self, dataset: str, run_number: int) -> str:
        return (f'{self.log_directory}/{dataset}/{self.modelname_dir}/run_{run_number}'
                f'/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv')

    def load_logs(self, dataset: str, run_number: int) -> list:
        """Read one run once.  A rollout log is hundreds of megabytes."""
        key = (dataset, run_number)
        if key not in self.log_cache:
            logger = diffusion_decision_model_logger(log_file_name = self.log_file_name(dataset, run_number))
            self.log_cache[key] = logger.load_logs_list()
            print(f'loaded {dataset} run {run_number}: {len(self.log_cache[key])} samples', file = sys.stderr)

        return self.log_cache[key]

    def load_budget_confidence(self, dataset: str, run_number: int) -> dict:
        """{question: the budget run's vote share for this evidence count}.  Empty when not run yet.

        The budget run samples answers from the bare question, five per evidence step,
        and logs the share of them that gave the most common answer.  Matched on the
        question text, because on mmlu and mmlu_pro the two runs gave the same question
        different Sample_IDs.
        """
        key = (dataset, run_number)
        if key not in self.budget_cache:
            path = (f'{self.log_directory}/{dataset}/{self.modelname_dir}/run_{run_number}'
                    f'/budget_matched_self_consistency_{dataset}.csv')
            column = f'Self_Consistency_nv{self.number_of_evidence}'
            self.budget_cache[key] = {}
            if os.path.exists(path):
                budget = pd.read_csv(path, dtype=str, usecols=['Question', column])
                self.budget_cache[key] = {question.strip(): to_float(confidence)
                                          for question, confidence in zip(budget['Question'], budget[column])}

        return self.budget_cache[key]

    # ----------------------------------------------------------------- features

    def scored_token_count(self, completion_token_count: float, evidence_index: int) -> float:
        """How many tokens Partial_COT_Loss was summed over.  Not logged, so rebuilt.

        Evidence zero carries the loss of the whole completion.  After that the run
        cuts the completion into number_of_evidence + 1 near equal groups and scores
        whatever follows the first i of them.
        """
        if math.isnan(completion_token_count) or evidence_index == 0:
            return completion_token_count

        groups = self.number_of_evidence + 1
        base = int(completion_token_count) // groups
        remainder = int(completion_token_count) % groups
        prefix = evidence_index * base + max(0, min(evidence_index, remainder - 1))
        return completion_token_count - prefix

    def per_token_loss(self, log, evidence_log) -> float:
        """The accumulation loss with every loss divided by its own token count."""
        values = []
        for rollout in evidence_log.consistency_list:
            if is_true(rollout.accuracy) and to_float(rollout.token_count) > 0:
                values.append(to_float(rollout.loss) / to_float(rollout.token_count))

        scored = self.scored_token_count(to_float(log.token_count), int(evidence_log.index))
        if scored > 0:
            values.append(to_float(evidence_log.partial_cot_loss) / scored)

        return mean_or_nan(values)

    def evidence_channels(self, log, loss_mode: str) -> dict:
        """One list per channel, one entry per evidence step."""
        evidence_list = sorted(log.evidence_list, key = lambda evidence: int(evidence.index))
        channels = {name: [] for name in ['loss', 'self_consistency', 'length', 'length_spread',
                                          'agreeing_length', 'delta_loss', 'delta_self_consistency']}

        for evidence_log in evidence_list:
            lengths = [to_float(rollout.token_count) for rollout in evidence_log.consistency_list]
            lengths = [value for value in lengths if not math.isnan(value)]
            agreeing = [to_float(rollout.token_count) for rollout in evidence_log.consistency_list
                        if is_true(rollout.accuracy)]

            channels['length'].append(mean_or_nan(lengths))
            channels['length_spread'].append(float(np.std(lengths)) if len(lengths) > 1 else float('nan'))
            channels['agreeing_length'].append(mean_or_nan(agreeing))
            channels['self_consistency'].append(to_float(evidence_log.evidence_accumulation_self_consistency))

            if loss_mode == self.LOSS_TOTAL:
                channels['loss'].append(to_float(evidence_log.evidence_accumulation_loss))
            elif loss_mode == self.LOSS_PER_TOKEN:
                channels['loss'].append(self.per_token_loss(log, evidence_log))
            else:
                raise Exception(f'unknown loss mode {loss_mode}')

        # Under total the logged steps still describe the channel, so they are read
        # as they are and this reproduces the existing pipeline exactly.
        for index, evidence_log in enumerate(evidence_list):
            if loss_mode == self.LOSS_TOTAL:
                channels['delta_loss'].append(to_float(evidence_log.delta_evidence_loss))
                channels['delta_self_consistency'].append(to_float(evidence_log.delta_evidence_self_consistency))
            elif index == 0:
                channels['delta_loss'].append(0.0)
                channels['delta_self_consistency'].append(0.0)
            else:
                channels['delta_loss'].append(channels['loss'][index - 1] - channels['loss'][index])
                channels['delta_self_consistency'].append(channels['self_consistency'][index] - channels['self_consistency'][index - 1])

        return channels

    def sample_features(self, log, loss_mode: str, feature_set: str) -> list:
        """The feature row of one sample, or None when it cannot be built."""
        if feature_set == self.BASELINE_SCALAR:
            return [to_float(log.completion_loss),
                    to_float(log.sequence_probability),
                    to_float(log.length_normalized_sequence_probability),
                    to_float(log.entropy),
                    to_float(log.mean_entropy)]

        if feature_set == self.BASELINE_HIDDEN:
            try:
                return list(log.get_last_layer_representations_numpy())
            except Exception:
                return None

        channels = self.evidence_channels(log, loss_mode)
        wanted = {
            self.FULL: ['loss', 'self_consistency', 'delta_loss', 'delta_self_consistency'],
            self.ROLLOUT_LENGTH: ['length'],
            self.ROLLOUT_LENGTH_STD: ['length', 'length_spread'],
            self.AGREEING_LENGTH: ['agreeing_length'],
            self.EVIDENCE_LOSS: ['loss'],
            self.SELF_CONSISTENCY: ['self_consistency'],
            }
        if feature_set not in wanted:
            raise Exception(f'unknown feature set {feature_set}')

        row = []
        for index in range(self.number_of_evidence):
            row.extend(channels[name][index] for name in wanted[feature_set])

        return row

    def sample_labels(self, log) -> list:
        """[was the run's answer right, was the vote's answer right].  nan when no vote."""
        vote = str(log.self_consistency_accuracy).strip().lower()
        return [1.0 if is_true(log.accuracy) else 0.0,
                float('nan') if vote in MISSING else (1.0 if vote == 'true' else 0.0)]

    def sample_baselines(self, log, budget_confidence: dict) -> list:
        """The confidences that need no training, in BASELINE_NAMES order.

        The published measures are turned so larger means more likely correct, which
        for a loss or an entropy means negating it.  Completion_Loss is already the
        summed negative log likelihood, so the first two are exact.
        """
        vote_shares = []
        for confidence, answer in [(log.self_consistency_confidence, log.self_consistency_final_answer),
                                   (log.self_consistency_completion_confidence, log.self_consistency_completion_final_answer)]:
            vote_shares.append(to_float(confidence) if is_present(answer) else float('nan'))

        loss = to_float(log.completion_loss)
        tokens = to_float(log.token_count)
        return vote_shares + [
            -loss,
            -loss / tokens if tokens > 0 else float('nan'),
            -to_float(log.entropy),
            -to_float(log.mean_entropy),
            to_float(log.length_normalized_sequence_probability),
            budget_confidence.get(str(log.question).strip(), float('nan')),
            ]

    def build_matrix(self, datasets: list, from_run_number: int, to_run_number: int,
                     loss_mode: str = LOSS_TOTAL, target: str = TARGET_RUN, feature_set: str = FULL):
        """Features, labels and untrained confidences for the given datasets."""
        if target not in (self.TARGET_RUN, self.TARGET_VOTE):
            raise Exception(f'unknown target {target}')

        rows, labels, baselines = [], [], []
        for dataset in datasets:
            for run_number in range(from_run_number, to_run_number):
                budget_confidence = self.load_budget_confidence(dataset, run_number)
                for log in self.load_logs(dataset, run_number):
                    if len(log.evidence_list) != self.number_of_evidence:
                        continue

                    row = self.sample_features(log, loss_mode, feature_set)
                    if row is None:
                        continue

                    rows.append(row)
                    labels.append(self.sample_labels(log))
                    baselines.append(self.sample_baselines(log, budget_confidence))

        X = np.array(rows, dtype=float)
        labels = np.array(labels, dtype=float)
        baselines = np.array(baselines, dtype=float)

        # A sample with no vote has no second label and is dropped, not counted wrong.
        label_column = 0 if target == self.TARGET_RUN else 1
        graded = ~np.isnan(labels[:, label_column])
        return X[graded], labels[graded, label_column].astype(int), baselines[graded]

    # ------------------------------------------------------------------ fitting

    def fill_and_scale(self, X_train, X_test, standardize: bool):
        """Fill the gaps and set the scale, both measured on X_train alone.

        Nothing is ever measured on X_test.  It is only transformed by the numbers
        X_train gave, which is what keeps the held out set unseen during training.
        """
        X_train = np.array(X_train, dtype=float)
        X_test = np.array(X_test, dtype=float)

        # A missing feature becomes that column's training mean.
        column_mean = np.nanmean(X_train, axis=0)
        column_mean = np.where(np.isnan(column_mean), 0.0, column_mean)
        for matrix in (X_train, X_test):
            missing = np.isnan(matrix)
            if missing.any():
                matrix[missing] = np.take(column_mean, np.where(missing)[1])

        # Every column to training mean zero and training standard deviation one.
        if standardize:
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        return X_train, X_test

    def fit_logistic(self, X_train, y_train, class_weight):
        """Fit, silencing the convergence warning and reporting it instead."""
        model = LogisticRegression(max_iter = self.MAX_ITER, random_state = 42, class_weight = class_weight)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category = ConvergenceWarning)
            model.fit(X_train, y_train)

        return model, bool(np.all(np.asarray(model.n_iter_) < self.MAX_ITER))

    # ------------------------------------------------------------------ metrics

    def measure(self, y_test, confidence) -> dict:
        """ROC from the raw score, and the calibration error on the two scales.

        Nothing here is fitted, so every column is a property of the score itself.
        """
        y_test = np.asarray(y_test)
        confidence = np.asarray(confidence, dtype=float)
        row = {metric: float('nan') for metric in self.METRICS}
        if not len(y_test):
            return row

        if len(np.unique(y_test)) > 1:
            row['roc_auc'] = float(roc_auc_score(y_test, confidence))

        # The rescaling used elsewhere in this repo, (x - min) / (max - min), with no
        # labels.  It only rises, so it leaves the order ROC read untouched.
        low, high = confidence.min(), confidence.max()
        minmax = (confidence - low) / (high - low) if high > low else confidence

        # ECE compares a confidence against how often answers at that confidence are
        # right, so it needs a probability.  A log likelihood or an entropy is not
        # one and gets no ECE, only the rescaled column.
        scales = [('ece_minmax', minmax)]
        if low >= 0.0 and high <= 1.0:
            scales.append(('ece', confidence))

        # calculate_ECE_MCE throws when qcut cannot bin the scores, which is a
        # missing number, not a failed run.
        for metric, scaled in scales:
            try:
                row[metric] = float(self.calculate_ECE_MCE(y_test, scaled)[0])
            except Exception:
                pass

        return row

    def result_row(self, held_out, target, method, y_test) -> dict:
        """The label and size columns of one table row.  The caller adds the metrics."""
        y_test = np.asarray(y_test)
        return {'held_out': held_out, 'target': target, 'method': method,
                'test_count': len(y_test),
                'minority_count': int(min(np.sum(y_test == 1), np.sum(y_test == 0))) if len(y_test) else 0}

    def score_untrained(self, held_out, target, method, test_confidence, y_test) -> dict:
        """A published confidence used exactly as it is, with nothing fitted to it."""
        row = self.result_row(held_out, target, method, y_test)
        row.update(self.measure(y_test, np.asarray(test_confidence, dtype=float)))
        row.update({'baseline_rows': [], 'feature_set': '-', 'standardize': '-',
                    'class_weight': '-', 'train_count': 0, 'converged': True})
        for metric in self.METRICS:
            row[metric + '_sd'] = 0.0

        return row

    # --------------------------------------------------------------- experiment

    def evaluate(self, test_datasets: list = None, from_run_number: int = 1, to_run_number: int = 2,
                 loss_mode: str = LOSS_TOTAL, standardize: bool = True, class_weight = None,
                 target: str = TARGET_RUN, feature_set: str = FULL, seeds = SEEDS) -> dict:
        """Train on everything except the held out datasets, and score only on those.

        With no datasets named it falls back to a random split over all of them.
        Gaps, scale and calibration are all measured on the training rows only.
        """
        test_datasets = list(test_datasets or [])
        if test_datasets:
            train_datasets = [dataset for dataset in self.datasets if dataset not in test_datasets]
            if not train_datasets:
                raise Exception('every dataset was held out, nothing is left to train on')

            X_train, y_train, _ = self.build_matrix(train_datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            X_test, y_test, baselines_test = self.build_matrix(test_datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            held_out = ','.join(test_datasets)

            # Which dataset is held out fixes the split and the solver is
            # deterministic, so a seed cannot change the fit.  Fit once, then redraw
            # the held out samples to see how much the number rests on which
            # problems the benchmark happens to contain.
            X_train, X_test = self.fill_and_scale(X_train, X_test, standardize)
            model, converged = self.fit_logistic(X_train, y_train, class_weight)
            probability = model.predict_proba(X_test)[:, 1]

            measurements = []
            for seed in seeds:
                draw = (np.arange(len(y_test)) if seed == seeds[0]
                        else np.random.default_rng(seed).integers(0, len(y_test), len(y_test)))
                measurements.append(self.measure(y_test[draw], probability[draw]))
        else:
            X, y, baselines = self.build_matrix(self.datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            held_out = 'random split'
            measurements, converged = [], True
            for seed in seeds:
                X_train, X_test, y_train, y_test, _, baselines_test = train_test_split(
                    X, y, baselines, test_size=0.2, random_state=seed, stratify=y)
                X_train, X_test = self.fill_and_scale(X_train, X_test, standardize)
                model, ok = self.fit_logistic(X_train, y_train, class_weight)
                converged = converged and ok
                measurements.append(self.measure(y_test, model.predict_proba(X_test)[:, 1]))

        method = loss_mode if feature_set in self.LOSS_BEARING_SETS else '-'
        row = self.result_row(held_out, target, method, y_test)
        row.update({'feature_set': feature_set, 'standardize': standardize,
                    'class_weight': class_weight if class_weight else 'none',
                    'train_count': len(y_train), 'converged': converged})

        # The mean and the spread of each metric over the seeds.
        for metric in self.METRICS:
            values = [measurement[metric] for measurement in measurements]
            row[metric] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float('nan')
            row[metric + '_sd'] = float(np.nanstd(values)) if not np.all(np.isnan(values)) else float('nan')

        # The same held out samples scored by each published confidence, for comparison.
        row['baseline_rows'] = []
        for column, name in enumerate(self.BASELINE_NAMES):
            scored = ~np.isnan(baselines_test[:, column])
            row['baseline_rows'].append(
                self.score_untrained(held_out, target, name, baselines_test[scored, column],
                                     y_test[scored]))

        return row

    # ------------------------------------------------------------------- sweeps

    def ablation(self, test_datasets: list = None, from_run_number: int = 1, to_run_number: int = 2,
                 feature_set: str = FULL, seeds = SEEDS) -> list:
        """Every switch combination on one held out set, for both graded answers."""
        # Only a set built from the loss sweeps the loss column; for the rest the
        # mode changes nothing, so one pass is the whole sweep.
        loss_modes = ([self.LOSS_TOTAL, self.LOSS_PER_TOKEN] if feature_set in self.LOSS_BEARING_SETS
                      else [self.LOSS_TOTAL])

        results = []
        for target in [self.TARGET_RUN, self.TARGET_VOTE]:
            trained = []
            for loss_mode in loss_modes:
                for standardize in [False, True]:
                    for class_weight in [None, 'balanced']:
                        trained.append(self.evaluate(test_datasets, from_run_number, to_run_number,
                                                     loss_mode, standardize, class_weight, target, feature_set, seeds))

            # The untrained rows do not depend on the switches; take them from the first.
            results.extend(trained)
            results.extend(trained[0]['baseline_rows'])

        held_out = ','.join(test_datasets) if test_datasets else 'random split'
        self.print_results(results, f'ablation, features: {feature_set}, held out: {held_out}')
        return results

    def train_logistic_regression(self, from_run_number, to_run_number, loss_mode: str = LOSS_TOTAL,
                                  standardize: bool = False, class_weight = None, target: str = TARGET_RUN) -> dict:
        """The random split over every dataset, which is what this has always run."""
        result = self.evaluate(None, from_run_number, to_run_number, loss_mode, standardize, class_weight, target)
        self.print_results([result], 'random split over every dataset')
        return result

    # ---------------------------------------------------------------- reporting

    def print_results(self, results: list, caption: str) -> None:
        width = 134
        print('\n' + '=' * width)
        print(f'== {caption}')
        print('=' * width)
        print(f"{'held out':<20}{'target':>6} {'method':<26}{'scaled':>7}{'weight':>10}{'fit':>5}"
              f"{'train':>7}{'test':>6}{'minority':>9}{'ROC':>8}{'ROCsd':>8}{'ECE':>10}{'ECEminmax':>11}")
        print('-' * width)
        for row in results:
            print(f"{row['held_out']:<20}{row['target']:>6} {row['method']:<26}"
                  f"{str(row['standardize']):>7}{str(row['class_weight']):>10}"
                  f"{('ok' if row['converged'] else 'STOP'):>5}"
                  f"{row['train_count']:>7}{row['test_count']:>6}{row['minority_count']:>9}"
                  f"{row['roc_auc']:>8.3f}{row['roc_auc_sd']:>8.3f}"
                  f"{row['ece']:>10.3f}{row['ece_minmax']:>11.3f}")
        print('-' * width)

    # ------------------------------- kept from the earlier version of this file
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
            print(f'[WARN] {skipped_count} samples skipped, they have no self consistency vote to score', file = sys.stderr)

        return np.array(confidence_list, dtype=float), np.array(label_list, dtype=int)

    def self_consistency_confidence_completion(self, from_run_number, to_run_number) -> None:
        log_list: list[diffusion_decision_model_log_entity] = []
        
        X = np.empty(0)
        y = np.empty(0)
        for dataset in self.datasets:
            for run_number in range(from_run_number,to_run_number):
                log_list = self.load_logs(dataset, run_number)
                
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
                log_list = self.load_logs(dataset, run_number)
                
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
        parameter_keys = ["target", "method", "standardize", "class_weight"]
        calculation_keys = ["roc_auc", "ece", "ece_minmax"]

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

    def calculate_ECE_MCE(self, y_list, confidence_list, n_bins = 10):
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
    # -- how to run one thing by hand ---------------------------------------
    # training = diffusion_decision_model_training(number_of_evidence = 20)
    # training.train_logistic_regression(from_run_number = 1, to_run_number = 2)
    # training.evaluate(test_datasets = ['gpqa'])
    # training.evaluate(test_datasets = ['gpqa'], loss_mode = training.LOSS_PER_TOKEN)
    # training.evaluate(test_datasets = ['gpqa'], target = training.TARGET_VOTE)
    # training.evaluate(test_datasets = ['gpqa'], feature_set = training.ROLLOUT_LENGTH)
    # training.ablation(test_datasets = ['gpqa'])
    # -----------------------------------------------------------------------

    # Every output the generation produced: one model and evidence count per row.
    # Each gets its own folder so the feature set files never mix runs.
    # Every feature set, or only the ones named on the command line, which is how a
    # newly added set is filled in without rewriting the files the others own.
    FEATURE_SETS_TO_RUN = sys.argv[1:] or diffusion_decision_model_training.FEATURE_SETS
    for name in FEATURE_SETS_TO_RUN:
        if name not in diffusion_decision_model_training.FEATURE_SETS:
            raise Exception(f'unknown feature set {name}')

    RUNS = [('qwen-qwen3-8b', 5),
            ('qwen-qwen3-8b', 10),
            ('qwen-qwen3-8b', 15),
            ('qwen-qwen3-8b', 20),
            ('qwen-qwen3-8b', 25),
            ('deepseek-ai-deepseek-r1-distill-qwen-7b', 20)]

    for modelname_dir, number_of_evidence in RUNS:
        print(f'\n===== {modelname_dir}, {number_of_evidence} evidence steps =====')
        training = diffusion_decision_model_training(number_of_evidence = number_of_evidence,
                                                     modelname_dir = modelname_dir)

        # One file per feature set: the random split, then every dataset held out,
        # then the two domains held out whole, then the averages.
        output_directory = f'src/diffusion_decision_model/ablations/{modelname_dir}/nv_{number_of_evidence}'
        os.makedirs(output_directory, exist_ok=True)

        for feature_set in FEATURE_SETS_TO_RUN:
            output_file_name = f'{output_directory}/ablation_{feature_set}.txt'
            with open(output_file_name, 'w') as output_file, contextlib.redirect_stdout(output_file):
                # The free bar: the vote share used as the confidence, over every dataset
                # pooled with nothing held out.  The same numbers in all seven files.
                print('\nself consistency confidence, every dataset pooled, nothing held out')
                print('-' * 134)
                training.self_consistency_confidence(from_run_number = 1, to_run_number = 2)
                training.self_consistency_confidence_completion(from_run_number = 1, to_run_number = 2)
                print('-' * 134)

                training.ablation(feature_set = feature_set)

                per_dataset = [training.ablation(test_datasets = [dataset], feature_set = feature_set)
                               for dataset in training.datasets]

                print('\n\naveraged over every held out dataset:')
                training.calculate_grouped_averages(per_dataset)

                # math500 and countdown hold 1 and 3 of the rarer class, so their
                # numbers are noise being folded in with the rest.
                big_enough = [result for result in per_dataset if result[0]['minority_count'] >= training.MINORITY_FLOOR]
                big_enough_names = [dataset for dataset, result in zip(training.datasets, per_dataset)
                                    if result[0]['minority_count'] >= training.MINORITY_FLOOR]
                print(f"\n\naveraged over the held out datasets carrying at least {training.MINORITY_FLOOR} of the rarer class ({', '.join(big_enough_names)}):")
                training.calculate_grouped_averages(big_enough)

                # Whole domains held out, so no benchmark can lean on a sibling.
                by_domain = []
                for group_name, group in training.DATASET_GROUPS.items():
                    print(f'\n\nheld out as a domain: {group_name}')
                    by_domain.append(training.ablation(test_datasets = group, feature_set = feature_set))

                # Every benchmark held out exactly once: the domains together, the rest
                # alone, and only where the rarer class can carry a ROC.  countdown has 3
                # and scores 1.000 or 0.000 on nearly every measure, so a fifth of each
                # average was a coin flip.  math500 has 1 but sits inside the mathematics
                # group, which has 31 between its three benchmarks and stays.
                partition = list(by_domain)
                partition_names = list(training.DATASET_GROUPS)
                for dataset, result in zip(training.datasets, per_dataset):
                    if any(dataset in group for group in training.DATASET_GROUPS.values()):
                        continue

                    if result[0]['minority_count'] < training.MINORITY_FLOOR:
                        continue

                    partition.append(result)
                    partition_names.append(dataset)

                print(f"\n\nMAIN AVERAGE, over these four held out groups: {', '.join(partition_names)}")
                print('every benchmark counted exactly once, the two domains held out together,')
                print(f'and only groups carrying at least {training.MINORITY_FLOOR} of the rarer class')
                training.calculate_grouped_averages(partition)

            print(f'wrote {output_file_name}')
