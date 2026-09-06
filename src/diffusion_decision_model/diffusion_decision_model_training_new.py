from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger

import os
import math
import contextlib
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

    # total: the summed loss the run logged.  per_token: the same divided by the
    # tokens it was summed over, so it stops growing with the length of the text.
    LOSS_TOTAL = 'total'
    LOSS_PER_TOKEN = 'per_token'

    # Which answer is graded.  run: the one the model wrote.  vote: the one the ten
    # rollouts agreed on.  They differ on one sample in eight.
    TARGET_RUN = 'run'
    TARGET_VOTE = 'vote'

    # Which numbers describe a sample.  Widths with twenty evidence steps:
    #   full 80, rollout_length 20, rollout_length_std 40, agreeing_length 20,
    #   self_consistency 20, baseline_scalar 5, baseline_hidden 4096
    FULL = 'full'
    ROLLOUT_LENGTH = 'rollout_length'
    ROLLOUT_LENGTH_STD = 'rollout_length_std'
    AGREEING_LENGTH = 'agreeing_length'
    SELF_CONSISTENCY = 'self_consistency'
    BASELINE_SCALAR = 'baseline_scalar'
    BASELINE_HIDDEN = 'baseline_hidden'

    FEATURE_SETS = [FULL, ROLLOUT_LENGTH, ROLLOUT_LENGTH_STD, AGREEING_LENGTH,
                    SELF_CONSISTENCY, BASELINE_SCALAR, BASELINE_HIDDEN]

    # The two baselines read the whole completion, not the evidence steps.
    WHOLE_COMPLETION_SETS = [BASELINE_SCALAR, BASELINE_HIDDEN]

    # The untrained confidences carried beside every sample, in column order.
    BASELINE_NAMES = ['self cons 0', 'self cons last', 'seq logprob',
                      'seq logprob/tok', 'entropy total', 'mean token ent',
                      'mean token prob']

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
            print(f'loaded {dataset} run {run_number}: {len(self.log_cache[key])} samples')

        return self.log_cache[key]

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

    def sample_baselines(self, log) -> list:
        """The seven confidences that need no training, in BASELINE_NAMES order.

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
            ]

    def build_matrix(self, datasets: list, from_run_number: int, to_run_number: int,
                     loss_mode: str = LOSS_TOTAL, target: str = TARGET_RUN, feature_set: str = FULL):
        """Features, labels and untrained confidences for the given datasets."""
        if target not in (self.TARGET_RUN, self.TARGET_VOTE):
            raise Exception(f'unknown target {target}')

        rows, labels, baselines = [], [], []
        for dataset in datasets:
            for run_number in range(from_run_number, to_run_number):
                for log in self.load_logs(dataset, run_number):
                    if len(log.evidence_list) != self.number_of_evidence:
                        continue

                    row = self.sample_features(log, loss_mode, feature_set)
                    if row is None:
                        continue

                    rows.append(row)
                    labels.append(self.sample_labels(log))
                    baselines.append(self.sample_baselines(log))

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

    def platt_scale(self, test_confidence, train_confidence, y_train):
        """Turn a raw score into a probability with one logistic curve.

        The curve is fitted on the training rows, so it has seen how often answers
        there are correct.  Its slope can come out negative, which reverses the
        order, so the result feeds ECE only and never ROC.
        """
        train_confidence = np.asarray(train_confidence, dtype=float)
        usable = ~np.isnan(train_confidence)
        if usable.sum() < 2 or len(np.unique(np.asarray(y_train)[usable])) < 2:
            return test_confidence

        scaler = StandardScaler().fit(train_confidence[usable].reshape(-1, 1))
        curve, _ = self.fit_logistic(scaler.transform(train_confidence[usable].reshape(-1, 1)),
                                     np.asarray(y_train)[usable], None)
        return curve.predict_proba(scaler.transform(test_confidence.reshape(-1, 1)))[:, 1]

    # ------------------------------------------------------------------ metrics

    def expected_calibration_error(self, y_test, probability) -> float:
        try:
            error, _ = self.calculate_ECE_MCE(y_test, probability)
            return float(error)
        except Exception:
            return float('nan')

    def minmax_scale(self, confidence):
        """The rescaling used elsewhere in this repo: x / (max - min), no labels.

        It knows only the spread of the scores, so it cannot know how often the
        held out benchmark is answered correctly, and the level it lands on is
        whatever the spread happens to give.
        """
        confidence = np.asarray(confidence, dtype=float)
        spread = confidence.max() - confidence.min()
        return confidence / spread if spread > 0 else confidence

    def measure(self, y_test, confidence, probability = None) -> dict:
        """ROC from the ordering, and the calibration error under both rescalings.

        probability is the same confidence already on the zero to one scale, which a
        trained row is and a raw score is not.  Without it the confidence stands as is.
        """
        y_test = np.asarray(y_test)
        confidence = np.asarray(confidence, dtype=float)
        if not len(y_test):
            return {metric: float('nan') for metric in self.METRICS}

        return {
            'roc_auc': float(roc_auc_score(y_test, confidence)) if len(np.unique(y_test)) > 1 else float('nan'),
            'ece': self.expected_calibration_error(y_test, confidence if probability is None else probability),
            'ece_minmax': self.expected_calibration_error(y_test, self.minmax_scale(confidence)),
            }

    def result_row(self, held_out, target, method, y_test, metrics) -> dict:
        """One table row: what was run, how big the held out set was, and the metrics."""
        y_test = np.asarray(y_test)
        row = {'held_out': held_out, 'target': target, 'loss_mode': method,
               'test_count': len(y_test),
               'minority_count': int(min(np.sum(y_test == 1), np.sum(y_test == 0))) if len(y_test) else 0}
        row.update(metrics)
        return row

    def score_untrained(self, held_out, target, method, test_confidence, y_test,
                        train_confidence, y_train) -> dict:
        """A published confidence used as it stands.  ROC from the raw score, ECE after scaling."""
        test_confidence = np.asarray(test_confidence, dtype=float)

        # A log likelihood or an entropy is not a probability, so ECE needs one made
        # for it.  Platt can flip the order, so the flipped copy feeds ECE only.
        probability = None
        if len(test_confidence) and (test_confidence.min() < 0.0 or test_confidence.max() > 1.0):
            probability = self.platt_scale(test_confidence, train_confidence, y_train)

        row = self.result_row(held_out, target, method, y_test,
                              self.measure(y_test, test_confidence, probability))
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

            X_train, y_train, baselines_train = self.build_matrix(train_datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            X_test, y_test, baselines_test = self.build_matrix(test_datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            held_out = ','.join(test_datasets)
            measurements, converged = self.measure_over_draws(X_train, y_train, X_test, y_test, standardize, class_weight, seeds)
        else:
            X, y, baselines = self.build_matrix(self.datasets, from_run_number, to_run_number, loss_mode, target, feature_set)
            held_out = 'random split'
            measurements, converged = [], True
            for seed in seeds:
                X_train, X_test, y_train, y_test, baselines_train, baselines_test = train_test_split(
                    X, y, baselines, test_size=0.2, random_state=seed, stratify=y)
                X_train, X_test = self.fill_and_scale(X_train, X_test, standardize)
                model, ok = self.fit_logistic(X_train, y_train, class_weight)
                converged = converged and ok
                measurements.append(self.measure(y_test, model.predict_proba(X_test)[:, 1]))

        row = self.result_row(held_out, target, loss_mode if feature_set == self.FULL else '-', y_test, {})
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
                                     y_test[scored], baselines_train[:, column], y_train))

        return row

    def measure_over_draws(self, X_train, y_train, X_test, y_test, standardize, class_weight, seeds):
        """Fit once, then redraw the held out samples per seed to get a spread.

        The split is fixed by which dataset was held out and the solver is
        deterministic, so a seed cannot change the fit; it can only ask how much the
        number rests on which problems the benchmark happens to contain.
        """
        X_train, X_test = self.fill_and_scale(X_train, X_test, standardize)
        model, converged = self.fit_logistic(X_train, y_train, class_weight)
        probability = model.predict_proba(X_test)[:, 1]

        measurements = []
        for seed in seeds:
            draw = np.arange(len(y_test)) if seed == seeds[0] else np.random.default_rng(seed).integers(0, len(y_test), len(y_test))
            measurements.append(self.measure(y_test[draw], probability[draw]))

        return measurements, converged

    # ------------------------------------------------------------------- sweeps

    def ablation(self, test_datasets: list = None, from_run_number: int = 1, to_run_number: int = 2,
                 feature_set: str = FULL, seeds = SEEDS) -> list:
        """Every switch combination on one held out set, for both graded answers."""
        # Only the full set carries a loss, so only it sweeps the loss column.
        loss_modes = [self.LOSS_TOTAL, self.LOSS_PER_TOKEN] if feature_set == self.FULL else [self.LOSS_TOTAL]

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
        width = 122
        print('\n' + '=' * width)
        print(f'== {caption}')
        print('=' * width)
        print(f"{'held out':<16}{'target':>6} {'loss':<16}{'scaled':>7}{'weight':>10}{'fit':>5}"
              f"{'train':>7}{'test':>6}{'minority':>9}{'ROC':>8}{'ROCsd':>8}{'ECE':>10}{'ECEminmax':>11}")
        print('-' * width)
        for row in results:
            print(f"{row['held_out']:<16}{row['target']:>6} {row['loss_mode']:<16}"
                  f"{str(row['standardize']):>7}{str(row['class_weight']):>10}"
                  f"{('ok' if row['converged'] else 'STOP'):>5}"
                  f"{row['train_count']:>7}{row['test_count']:>6}{row['minority_count']:>9}"
                  f"{row['roc_auc']:>8.3f}{row['roc_auc_sd']:>8.3f}"
                  f"{row['ece']:>10.3f}{row['ece_minmax']:>11.3f}")
        print('-' * width)

    def print_reading_notes(self) -> None:
        print("""\
================================================================================
how to read this
================================================================================
One table per held out setting, then the averages.  Each table has two blocks,
one per graded answer, each ending with the rows that fit nothing at all.

COLUMNS
  target     which answer was graded.  run = the one the model wrote,
             vote = the one the ten rollouts agreed on.  They disagree on one
             sample in eight, so the two blocks are not comparable with each other.
  loss       total = the summed loss the run logged, per_token = the same divided
             by the tokens behind it.  A dash means the feature set carries no loss.
  scaled     were the features standardised.  Fitted on the training half only.
  weight     balanced weighs the rarer class as heavily as the common one.
  fit        ok = the solver finished, STOP = it hit max_iter and stopped early.
  minority   how many of the rarer class the held out set holds.  READ THIS FIRST.
             Everything right of it rests on those samples alone.  math500 has 1
             and countdown has 3, so their numbers are noise.
  ROC        over every pair of one right and one wrong answer, how often the right
             one scored higher.  Taken from the raw score, so nothing is fitted for
             it.  ROCsd is its spread over ten seeds.
  ECE        calibration under Platt scaling: one logistic curve from the score to
             correctness, fitted on the training half.  Uses the labels.
  ECEminmax  calibration under the rescaling used elsewhere in this repo,
             x / (max - min), which uses no labels at all.

FEATURE SETS (one file each)
  full 80             the two accumulations and their two steps
  rollout_length 20   how long the ten rollouts ran, averaged.  No loss, no
                      agreement, only length.  The control the loss must beat.
  rollout_length_std 40   that average with its spread
  agreeing_length 20  the same over only rollouts that matched the run's answer
  self_consistency 20 the agreement alone
  baseline_scalar 5   published: the completion's loss, how likely the model held
                      its own answer, and its entropy, each total and per token
  baseline_hidden 4096  published: the mean pooled last hidden state

THE UNTRAINED ROWS
  self cons 0 / last  the vote share at the first and last evidence step
  seq logprob         the sequence log likelihood (Malinin and Gales)
  seq logprob/tok     the same per token, their length normalised score
  entropy total       summed predictive entropy
  mean token ent      averaged predictive entropy (LM-Polygraph's measure)
  mean token prob     the average probability the model gave its own tokens
  All are turned so larger means more likely correct, and all are used as the
  score itself, which is how their papers use them.  These are the bar: a trained
  row that does not clear them is not worth its compute.

THE TWO ECE COLUMNS
  A log likelihood or an entropy is not a probability, so a calibration error
  cannot be read off it directly.  Something has to put it on the probability
  scale first, and the two columns are the two ways of doing that.

  ECE uses Platt scaling: a logistic curve fitted on the training half, so it has
  seen how often answers there are correct.  Its slope can come out negative,
  which reverses the ranking, so it is used for this column only and never for ROC.

  ECEminmax uses x / (max - min), which sees only the spread of the scores and no
  labels at all.  It therefore cannot know how often the held out benchmark is
  answered correctly, and the level it lands on is whatever the spread gives.

  Neither is the truth.  On the same scores in the same order a min max map, a
  quantile map and a fitted logistic map gave 0.42, 0.17 and 0.09, so the number
  depends heavily on which map was chosen.  And most of what is left after fitting
  is not about the measure: the curve learns its level where the model answers
  about eighty per cent correctly and is applied where it may answer half, which
  on gpqa was roughly two thirds of an ECE of 0.30.  Read ROC as the statement
  about the confidence and the two ECE columns as a statement about the rescaling.
================================================================================
""")

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
            print(f'[WARN] {skipped_count} samples skipped, they have no self consistency vote to score')

        return np.array(confidence_list, dtype=float), np.array(label_list, dtype=int)

    def self_consistency_confidence_completion(self, from_run_number, to_run_number) -> None:
        log_list: list[diffusion_decision_model_log_entity] = []
        
        X = np.empty(0)
        y = np.empty(0)
        for dataset in self.datasets:
            for run_number in range(from_run_number,to_run_number):
                logger = diffusion_decision_model_logger(log_file_name = self.log_file_name(dataset, run_number))
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
                logger = diffusion_decision_model_logger(log_file_name = self.log_file_name(dataset, run_number))
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
    training = diffusion_decision_model_training(number_of_evidence=20)

    # -- how to run ---------------------------------------------------------
    # training.train_logistic_regression(from_run_number = 1, to_run_number = 2)
    # training.evaluate(test_datasets = ['gpqa'])
    # training.evaluate(test_datasets = ['gpqa'], loss_mode = training.LOSS_PER_TOKEN)
    # training.evaluate(test_datasets = ['gpqa'], target = training.TARGET_VOTE)
    # training.evaluate(test_datasets = ['gpqa'], feature_set = training.ROLLOUT_LENGTH)
    # training.ablation(test_datasets = ['gpqa'])
    # -----------------------------------------------------------------------

    # One file per feature set: the random split, then every dataset held out, then
    # the two domains held out whole, then the averages.
    output_directory = 'src/diffusion_decision_model/ablations'
    os.makedirs(output_directory, exist_ok=True)

    for feature_set in diffusion_decision_model_training.FEATURE_SETS:
        output_file_name = f'{output_directory}/ablation_{feature_set}.txt'
        with open(output_file_name, 'w') as output_file, contextlib.redirect_stdout(output_file):
            training.print_reading_notes()
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

            # Every benchmark held out exactly once: the domains together, the rest alone.
            partition = list(by_domain)
            partition_names = list(training.DATASET_GROUPS)
            for dataset, result in zip(training.datasets, per_dataset):
                if any(dataset in group for group in training.DATASET_GROUPS.values()):
                    continue

                partition.append(result)
                partition_names.append(dataset)

            print(f"\n\naveraged over every benchmark held out exactly once ({', '.join(partition_names)}):")
            training.calculate_grouped_averages(partition)

        print(f'wrote {output_file_name}')
