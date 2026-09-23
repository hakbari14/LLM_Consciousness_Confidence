"""The fitted head for one held out benchmark, and what it predicts there.

Trains exactly as the ablation does: the full feature set with the per token loss,
unscaled and unweighted, on every benchmark except the held out one.  Prints the
weights and writes one row per held out question.

    python -m src.diffusion_decision_model.holdout_inference
"""

import numpy as np
import pandas as pd

from src.diffusion_decision_model.diffusion_decision_model_training import (
    diffusion_decision_model_training, is_true, to_float)

HELD_OUT = 'gsm8k'
EVIDENCE_COUNT = 5
MODEL = 'qwen-qwen3-8b'
CHANNELS = ['loss', 'agreement', 'delta_loss', 'delta_agreement']
OUT_DIRECTORY = f'src/diffusion_decision_model/ablations/{MODEL}/nv_{EVIDENCE_COUNT}'


def feature_names(evidence_count):
    """The columns of one feature row, in the order sample_features builds them."""
    return [f'step{step}_{channel}' for step in range(evidence_count) for channel in CHANNELS]


if __name__ == '__main__':
    training = diffusion_decision_model_training(number_of_evidence = EVIDENCE_COUNT, modelname_dir = MODEL)
    train_datasets = [dataset for dataset in training.datasets if dataset != HELD_OUT]

    X_train, y_train, _ = training.build_matrix(train_datasets, 1, 2, training.LOSS_PER_TOKEN,
                                                training.TARGET_RUN, training.FULL)
    X_test, y_test, _ = training.build_matrix([HELD_OUT], 1, 2, training.LOSS_PER_TOKEN,
                                              training.TARGET_RUN, training.FULL)
    X_train_filled, X_test_filled = training.fill_and_scale(X_train, X_test, False)
    model, converged = training.fit_logistic(X_train_filled, y_train, None)
    probability = model.predict_proba(X_test_filled)[:, 1]

    names = feature_names(EVIDENCE_COUNT)
    weights = model.coef_[0]

    print(f'held out {HELD_OUT}, {EVIDENCE_COUNT} evidence steps, model {MODEL}')
    print(f'trained on {", ".join(train_datasets)}: {len(y_train)} questions, {int(y_train.sum())} correct')
    print(f'converged: {converged}')
    print(f'\nintercept {model.intercept_[0]:+.4f}')
    print(f"\n{'feature':<24}{'weight':>10}   the same weights grouped by channel")
    print('-' * 66)
    for name, weight in zip(names, weights):
        print(f'{name:<24}{weight:>+10.4f}')

    print(f"\n{'channel':<18}{'sum of weights':>16}{'largest step':>16}")
    print('-' * 50)
    for index, channel in enumerate(CHANNELS):
        column = weights[index::len(CHANNELS)]
        print(f'{channel:<18}{column.sum():>+16.4f}{f"step{int(np.argmax(abs(column)))}":>16}')

    # One row per held out question: what went in, what came out.
    logs = [log for log in training.load_logs(HELD_OUT, 1) if len(log.evidence_list) == EVIDENCE_COUNT]
    rows = []
    for log, features, label, confidence in zip(logs, X_test, y_test, probability):
        row = {'sample_ID': log.sample_ID,
               'question': str(log.question)[:120],
               'answer': log.compared_final_answer,
               'target': log.target,
               'correct': bool(label),
               'confidence': round(float(confidence), 4)}
        row.update({name: value for name, value in zip(names, features)})
        rows.append(row)

    inference = pd.DataFrame(rows)
    path = f'{OUT_DIRECTORY}/inference_{HELD_OUT}.csv'
    inference.to_csv(path, index=False)

    correct, wrong = inference[inference.correct], inference[~inference.correct]
    print(f'\n{len(inference)} held out questions, {len(correct)} correct')
    print(f'mean confidence: correct {correct.confidence.mean():.3f}, wrong {wrong.confidence.mean():.3f}')
    print(f'ROC {training.measure(y_test, probability)["roc_auc"]:.3f}')
    print(f'wrote {path}')
