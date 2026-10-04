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
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.diffusion_decision_model.diffusion_decision_model_training import diffusion_decision_model_training, to_float
from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity

OUT_DIRECTORY = 'src/diffusion_decision_model/ablations_adhoc/kfold_within_benchmark'
FOLDS = 5

# The logger parses every sample's last layer representation as it loads, and stops on
# a log that was written without one.  Nothing here reads the representation, so an
# empty one is let through instead of ending the run.
parse_representation = diffusion_decision_model_log_entity.get_last_layer_representations_numpy
diffusion_decision_model_log_entity.get_last_layer_representations_numpy = (
    lambda log: parse_representation(log) if isinstance(log.last_layer_representations, str) else None)


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
    features = {}
    for method, loss_mode in (('CoT-EIG-Mean', trainer.LOSS_PER_TOKEN), ('CoT-EIG-Sum', trainer.LOSS_TOTAL)):
        features[method], y, baselines = trainer.build_matrix([dataset], 1, 2, loss_mode)

    seeds = trainer.SEEDS
    runs = [cross_validate(trainer, features, y, baselines, seed) for seed in seeds]
    methods = list(runs[0][0])

    # A generation stopped by the token limit can have its answer cut mid number and so
    # be graded wrong although the model was right.  Those are trivially told apart, so
    # a benchmark whose wrong answers are mostly cut off cannot be trusted here.
    tokens = np.array([to_float(log.token_count) for log in trainer.load_logs(dataset, 1)
                       if len(log.evidence_list) == trainer.number_of_evidence])
    cut_off = (tokens == tokens.max()) if np.sum(tokens == tokens.max()) > 1 else np.zeros(len(tokens), dtype=bool)

    wrong = int(np.sum(y == 0))
    lines = [f'{FOLDS} fold cross validation inside {dataset}, {trainer.modelname_dir}, nv {trainer.number_of_evidence}',
             f'{len(y)} samples, {wrong} answered wrong, so about {wrong / FOLDS:.1f} wrong answers in each fold',
             f'{int(cut_off.sum())} generations stopped by the token limit, {int(np.sum(cut_off & (y == 0)))} of them graded wrong',
             'regression: unscaled features, no class weight, as in the paper\'s tables',
             'pooled: the five folds\' predictions put together and measured once']

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

    reference = diffusion_decision_model_training(evidence_counts[0])
    dataset = next((name for name in names if name in reference.datasets), 'gsm8k')
    models = ([name for name in names if name not in reference.datasets]
              or sorted(os.listdir(f'{reference.log_directory}/{dataset}')))

    for model in models:
        for evidence_count in evidence_counts:
            trainer = diffusion_decision_model_training(evidence_count, model)
            if not os.path.exists(trainer.log_file_name(dataset, 1)):
                print(f'no log for {model}, {dataset}, nv {evidence_count}', file=sys.stderr)
                continue

            table = report(trainer, dataset)
            os.makedirs(f'{OUT_DIRECTORY}/{model}/{dataset}', exist_ok=True)
            with open(f'{OUT_DIRECTORY}/{model}/{dataset}/nv_{evidence_count}.txt', 'w') as output:
                output.write(table)
            print(table)
