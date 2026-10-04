"""Five fold cross validation inside one benchmark.  An ad hoc test, kept apart from the ablations.

The ablations train on the other benchmarks and test on the held out one.  This asks
what the same regression reaches when it may learn from the benchmark itself: its
questions are cut into five folds, the regression is fitted on four and scored on the
fifth, each fold in turn, and the five scores are averaged.  The untrained baselines
are scored on the same folds, so every row answers for the same questions.

Nothing in the training code is changed; its features, fit and metrics are reused.
The tables go to ablations_adhoc/kfold_within_benchmark/, not to ablations/.

    python -m src.diffusion_decision_model.kfold_within_benchmark              # gsm8k, nv 5, every model that has a log
    python -m src.diffusion_decision_model.kfold_within_benchmark gsm8k 5 10 15 20 25
    python -m src.diffusion_decision_model.kfold_within_benchmark gsm8k 5 qwen-qwen3-0.6b
"""

import os
import sys
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.diffusion_decision_model.diffusion_decision_model_training import diffusion_decision_model_training

OUT_DIRECTORY = 'src/diffusion_decision_model/ablations_adhoc/kfold_within_benchmark'
FOLDS = 5

def measure(trainer, y, confidence) -> dict:
    """ROC and ECE over the samples that have this confidence at all."""
    present = ~np.isnan(confidence)
    return trainer.measure(y[present], confidence[present])


def cross_validate(trainer, features: dict, y, baselines, seed: int):
    """({method: [one measurement per fold]}, {method: the measurement of all folds' predictions together}).

    The folds keep the benchmark's share of wrong answers, so every fold has some.
    The regression is the paper's: unscaled features, no class weight.
    """
    methods = list(features) + trainer.BASELINE_NAMES
    per_fold = {method: [] for method in methods}
    out_of_fold = {method: np.full(len(y), np.nan) for method in methods}

    for train, test in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(y, y):
        for method, X in features.items():
            X_train, X_test = trainer.fill_and_scale(X[train], X[test], standardize=False)
            model, _ = trainer.fit_logistic(X_train, y[train], class_weight=None)
            out_of_fold[method][test] = model.predict_proba(X_test)[:, 1]
        for column, method in enumerate(trainer.BASELINE_NAMES):
            out_of_fold[method][test] = baselines[test, column]

        for method in methods:
            per_fold[method].append(measure(trainer, y[test], out_of_fold[method][test]))

    return per_fold, {method: measure(trainer, y, out_of_fold[method]) for method in methods}


def report(trainer, dataset: str) -> str:
    # CoT-EIG, then its two halves: the agreement alone and the token probability alone.
    features = {}
    for feature_set in (trainer.FULL, trainer.SELF_CONSISTENCY, trainer.EVIDENCE_PROBABILITY):
        features[feature_set], y, baselines = trainer.build_matrix([dataset], 1, 2, feature_set=feature_set)

    seeds = trainer.SEEDS
    runs = [cross_validate(trainer, features, y, baselines, seed) for seed in seeds]
    methods = list(runs[0][0])

    # The trainer leaves out a response stopped by the token limit: its answer was
    # never finished, so its grade says nothing about the model.
    logs = [log for log in trainer.load_logs(dataset, 1) if len(log.evidence_list) == trainer.number_of_evidence]
    cut_off = sum(trainer.is_cut_off(log) for log in logs)

    wrong = int(np.sum(y == 0))
    lines = [f'{FOLDS} fold cross validation inside {dataset}, {trainer.modelname_dir}, nv {trainer.number_of_evidence}',
             f'{len(y)} samples, {wrong} answered wrong, so about {wrong / FOLDS:.1f} wrong answers in each fold',
             f'{cut_off} more were stopped by the token limit and are left out',
             *([f'FEWER THAN {trainer.MINORITY_FLOOR} WRONG ANSWERS: a fold may hold none, and the ROC below is noise']
               if wrong < trainer.MINORITY_FLOOR else []),
             'regression: unscaled features, no class weight, as in the paper\'s tables',
             'pooled: the five folds\' predictions put together and measured once',
             'CoT-EIG: each evidence\'s agreement, the change in agreement, its mean token probability, and that',
             'mean over the whole completion\'s.  EIG-SC: the agreement alone.  EIG-Prob: the two probability ones alone.']

    # The folds drawn with the first seed, one column per fold.
    per_fold, pooled = runs[0]
    for metric, title in (('roc_auc', 'ROC'), ('ece', 'ECE')):
        lines += ['', f'== {title}, folds drawn with seed {seeds[0]}',
                  f"{'method':<14}" + ''.join(f'{f"fold {fold + 1}":>8}' for fold in range(FOLDS))
                  + f"{'mean':>9}{'sd':>7}{'pooled':>8}"]
        for method in methods:
            values = [measurement[metric] for measurement in per_fold[method]]
            lines.append(f'{method:<14}' + ''.join(f'{value:>8.3f}' for value in values)
                         + f'{np.nanmean(values):>9.3f}{np.nanstd(values):>7.3f}{pooled[method][metric]:>8.3f}')

    # How much the answer rests on which questions fell in which fold.
    lines += ['', f'== ROC when the folds are drawn again, {len(seeds)} seeds {seeds}',
              f"{'method':<14}{'mean of folds':>15}{'lowest':>8}{'highest':>9}{'pooled':>10}{'lowest':>8}{'highest':>9}"]
    for method in methods:
        fold_means = [np.nanmean([measurement['roc_auc'] for measurement in per_fold[method]]) for per_fold, _ in runs]
        pooled_values = [pooled[method]['roc_auc'] for _, pooled in runs]
        lines.append(f'{method:<14}{np.mean(fold_means):>15.3f}{np.min(fold_means):>8.3f}{np.max(fold_means):>9.3f}'
                     f'{np.mean(pooled_values):>10.3f}{np.min(pooled_values):>8.3f}{np.max(pooled_values):>9.3f}')

    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    evidence_counts = [int(argument) for argument in sys.argv[1:] if argument.isdigit()] or [5]
    names = [argument for argument in sys.argv[1:] if not argument.isdigit()]

    # A name is a benchmark if the generation has a log folder for it, a model if that
    # benchmark has logs for it, and a mistake otherwise.
    log_directory = diffusion_decision_model_training(evidence_counts[0]).log_directory
    dataset = next((name for name in names if name in os.listdir(log_directory)), 'gsm8k')
    logged_models = sorted(os.listdir(f'{log_directory}/{dataset}'))
    for name in names:
        if name != dataset and name not in logged_models:
            raise Exception(f'{name} is neither a benchmark nor a model with a {dataset} log')
    models = [name for name in names if name in logged_models] or logged_models

    for model in models:
        for evidence_count in evidence_counts:
            trainer = diffusion_decision_model_training(evidence_count, model)
            if not os.path.exists(trainer.log_file_name(dataset, 1)):
                print(f'no log for {model}, {dataset}, nv {evidence_count}', file=sys.stderr)
                continue

            # With too few wrong answers a fold can hold none and has no ROC.  The table
            # says so once; the libraries would say it for every fold and every mean.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                table = report(trainer, dataset)
            os.makedirs(f'{OUT_DIRECTORY}/{model}/{dataset}', exist_ok=True)
            with open(f'{OUT_DIRECTORY}/{model}/{dataset}/nv_{evidence_count}.txt', 'w') as output:
                output.write(table)
            print(table)
