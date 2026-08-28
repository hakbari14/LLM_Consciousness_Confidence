from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger
from src.diffusion_decision_model.diffusion_decision_model import diffusion_decision_model

import torch
import math
import warnings
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

    def __init__(self, number_of_evidence: int) -> None:
        self.number_of_evidence = number_of_evidence
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
        return f'{self.log_directory}/{dataset}/qwen-qwen3-8b/run_{run_number}/diffusion_decision_model_{dataset}_nv_{self.number_of_evidence}.csv'

    def build_matrix(self, datasets: list[str], from_run_number: int, to_run_number: int, loss_mode: str = LOSS_MODE_TOTAL, target: str = TARGET_RUN):
        """Build the features and the labels for the given datasets.

        Every sample becomes four numbers per evidence step: the two accumulations
        the run recorded, and the step each of them took since the evidence before
        it. The loss mode decides how the two loss numbers are built.
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
                    for evidence_log in evidence_list:
                        accumulation_self_consistency_list.append(self.to_float(evidence_log.evidence_accumulation_self_consistency))

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

                        row.extend([
                            accumulation_loss_list[index],
                            accumulation_self_consistency_list[index],
                            delta_loss,
                            delta_self_consistency,
                            ])

                    row_list.append(row)

                    # Both labels are kept and one is chosen at the end, because
                    # there are two answers that can be graded: the one the run
                    # gave, and the one the rollouts voted for. A sample with no
                    # vote has no second label, and is dropped when that is the
                    # target rather than counted as a wrong one.
                    vote_accuracy = str(log.self_consistency_accuracy).strip().lower()
                    label_list.append([
                        1.0 if str(log.accuracy).strip().lower() == 'true' else 0.0,
                        float('nan') if vote_accuracy in ('', 'nan', 'none') else (1.0 if vote_accuracy == 'true' else 0.0),
                        ])

                    # The vote share on its own, at the first evidence and at the
                    # last, carried alongside so a table can show what the same
                    # samples give with no model fitted at all. A sample whose
                    # rollouts never reached a readable answer has no vote to
                    # report, and the answer field is where that shows, so it is
                    # left out rather than scored as a confidence of nothing.
                    baseline_row = []
                    for confidence, answer in [(log.self_consistency_confidence, log.self_consistency_final_answer),
                                               (log.self_consistency_completion_confidence, log.self_consistency_completion_final_answer)]:
                        has_answer = str(answer).strip().lower() not in ('', 'nan', 'none')
                        baseline_row.append(self.to_float(confidence) if has_answer else float('nan'))

                    baseline_list.append(baseline_row)

        if target not in (self.TARGET_RUN, self.TARGET_VOTE):
            raise Exception(f'unknown target {target}')

        X = np.array(row_list, dtype=float)
        labels = np.array(label_list, dtype=float)
        baseline = np.array(baseline_list, dtype=float)

        target_index = 0 if target == self.TARGET_RUN else 1
        keep = ~np.isnan(labels[:, target_index])
        return X[keep], labels[keep, target_index].astype(int), baseline[keep]

    def evaluate(self, test_datasets: list[str] = None, from_run_number: int = 1, to_run_number: int = 2,
                 loss_mode: str = LOSS_MODE_TOTAL, standardize: bool = True, class_weight = None,
                 target: str = TARGET_RUN, verbose: bool = False) -> dict:
        """Train on one set of datasets and score on another.

        Name the datasets to hold out and they are kept out of training entirely,
        which asks whether the method carries to a benchmark it has never seen.
        Leave them out and it falls back to a random split over everything, the
        easier question, and the reference the held out numbers are read against.

        Everything the model learns about scale comes from the training half only,
        the column means that fill the gaps as much as the standardiser. Fitting
        either of them on the held out set would hand it the answer sheet.
        """
        test_datasets = list(test_datasets or [])
        if test_datasets:
            train_datasets = [dataset for dataset in self.datasets if dataset not in test_datasets]
            if not train_datasets:
                raise Exception('every dataset was held out, nothing is left to train on')

            X_train, y_train, _ = self.build_matrix(train_datasets, from_run_number, to_run_number, loss_mode, target)
            X_test, y_test, baseline_test = self.build_matrix(test_datasets, from_run_number, to_run_number, loss_mode, target)
            held_out = ','.join(test_datasets)
        else:
            X, y, baseline = self.build_matrix(self.datasets, from_run_number, to_run_number, loss_mode, target)
            X_train, X_test, y_train, y_test, _, baseline_test = train_test_split(X, y, baseline, test_size=0.2, random_state=42, stratify=y)
            held_out = 'random split'

        # Fill the gaps with the training columns, never the held out ones.
        column_mean = np.nanmean(X_train, axis=0)
        column_mean = np.where(np.isnan(column_mean), 0.0, column_mean)
        for matrix in (X_train, X_test):
            missing_position = np.isnan(matrix)
            if missing_position.any():
                matrix[missing_position] = np.take(column_mean, np.where(missing_position)[1])

        if standardize:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # The solver does not always settle within its budget, which it says by
        # printing a warning for every fit. That would bury the table, so the
        # warning is silenced and the answer to it is carried in the result
        # instead: a row that did not converge stopped wherever the solver ran
        # out, and its numbers are worth less than the ones that did.
        model = LogisticRegression(max_iter = 1000, random_state = 42, class_weight = class_weight)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category = ConvergenceWarning)
            model.fit(X_train, y_train)

        converged = bool(np.all(np.asarray(model.n_iter_) < 1000))
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # The area under the curve rests on the rarer class alone, so it says
        # nothing at all when the held out set carries only one of the two.
        roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else float('nan')
        try:
            ece, _ = self.calculate_ECE_MCE(y_test, y_prob)
        except Exception:
            ece = float('nan')

        if verbose:
            print("\n===== Classification Report =====")
            print(classification_report(y_test, y_pred, zero_division=0))
            print("===== Confusion Matrix =====")
            print(confusion_matrix(y_test, y_pred))

        # The same held out samples scored by the vote share alone, so the tables
        # can show what the features are worth over taking self consistency as it
        # is. The label is the one every other row uses, whether the answer the
        # run actually gave was right, so the rows can be read against each other.
        baseline_rows = []
        for position, name in [(0, 'self cons 0'), (1, 'self cons last')]:
            confidence = baseline_test[:, position]
            keep = ~np.isnan(confidence)
            baseline_rows.append(self.score_confidence(held_out, name, confidence[keep], y_test[keep], target))

        return {
            'baseline_rows': baseline_rows,
            'held_out': held_out,
            'target': target,
            'loss_mode': loss_mode,
            'standardize': standardize,
            'class_weight': class_weight if class_weight else 'none',
            'train_count': len(X_train),
            'test_count': len(X_test),
            'minority_count': int(min(np.sum(y_test == 1), np.sum(y_test == 0))),
            'majority_rate': float(max(np.mean(y_test), 1.0 - np.mean(y_test))),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc,
            'ece': ece,
            'converged': converged,
            }

    def score_confidence(self, held_out: str, name: str, confidence, y_true, target: str) -> dict:
        """Score a confidence that needed no training, in the shape of a trained row."""
        roc_auc = float(roc_auc_score(y_true, confidence)) if len(np.unique(y_true)) > 1 and len(y_true) else float('nan')
        try:
            ece, _ = self.calculate_ECE_MCE(y_true, confidence)
        except Exception:
            ece = float('nan')

        y_pred = (np.asarray(confidence) >= 0.5).astype(int)
        return {
            'baseline_rows': [],
            'held_out': held_out,
            'target': target,
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
            'converged': True,
            }

    def ablation(self, test_datasets: list[str] = None, from_run_number: int = 1, to_run_number: int = 2) -> list[dict]:
        """Every combination of the switches, on the same held out set, for both targets.

        Per target: the loss taken as a total or per token, the features
        standardised or left alone, the classes weighted evenly or by their size,
        then the two untrained rows. Within one target every row sees the same
        samples, so the difference between any two of them is the switch that
        changed and nothing else. Across the two targets the samples differ, and
        so does the question, so read those blocks separately.
        """
        results = []
        for target in [self.TARGET_RUN, self.TARGET_VOTE]:
            trained = []
            for loss_mode in [self.LOSS_MODE_TOTAL, self.LOSS_MODE_PER_TOKEN]:
                for standardize in [False, True]:
                    for class_weight in [None, 'balanced']:
                        trained.append(self.evaluate(test_datasets, from_run_number, to_run_number, loss_mode, standardize, class_weight, target))

            # The untrained rows do not depend on the three switches, so take them
            # from the first fit and close each target's block with them.
            results.extend(trained)
            results.extend(trained[0]['baseline_rows'])

        self.print_results(results, f"ablation, held out: {','.join(test_datasets) if test_datasets else 'random split'}")
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

Each table has two blocks, one per graded answer, and each block has the eight
configurations followed by two rows that fit nothing at all.

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
          the answer. STOP means it reached max_iter=1000 and was cut off, so the
          coefficients are wherever it had got to. Only the unscaled total rows do
          this: raw summed losses run into the thousands and the four channels sit
          on very different scales, which leaves a long narrow valley the solver
          needs about 4800 rounds to walk. per_token solves the same problem in
          about 50. Raising max_iter to 10000 makes those rows converge and moves
          their mean ROC from 0.7444 down to 0.7360, so stopping early is
          currently flattering them rather than penalising them.

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
  every table is one run of one model, with no error bars. a single split of this
  size moves by roughly 0.06 ROC on the random seed alone, so small differences
  between neighbouring rows are not differences
""")

    def print_results(self, results: list[dict], caption: str) -> None:
        width = 142
        print('\n' + '=' * width)
        print(f'== {caption}')
        print('=' * width)
        print(f"{'held out':<16}{'target':>6} {'loss':<16}{'scaled':>7}{'weight':>10}{'fit':>5}{'train':>7}{'test':>6}{'minority':>9}"
              f"{'majority':>9}{'accuracy':>9}{'precision':>10}{'recall':>8}{'F1':>8}{'ROC':>8}{'ECE':>8}")
        print('-' * width)
        for result in results:
            print(f"{result['held_out']:<16}{result['target']:>6} {result['loss_mode']:<16}{str(result['standardize']):>7}{str(result['class_weight']):>10}"
                  f"{('ok' if result['converged'] else 'STOP'):>5}"
                  f"{result['train_count']:>7}{result['test_count']:>6}{result['minority_count']:>9}{result['majority_rate']:>9.3f}"
                  f"{result['accuracy']:>9.4f}{result['precision']:>10.4f}{result['recall']:>8.4f}{result['f1']:>8.4f}"
                  f"{result['roc_auc']:>8.4f}{result['ece']:>8.4f}")
        print('-' * width)
        print('minority is how many of the rarer class the held out set holds. The area under the curve')
        print('rests on those alone, so a handful of them means the number is mostly noise.')
        print('fit says whether the solver settled. STOP means it ran out of iterations and stopped')
        print('wherever it had got to, so read that row as a weaker result, not a comparable one.')
        print('the last rows of each block take the vote share as the confidence with nothing fitted,')
        print('and are what the trained rows have to beat to be worth the compute they cost.')
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
    # hold out more than one dataset at a time
    # training.evaluate(test_datasets = ['gpqa', 'mmlu'])
    #
    # all eight configurations on one held out set, with the untrained rows below them
    # training.ablation(test_datasets = ['gpqa'])
    #
    # the self consistency vote share on its own, scored over every dataset
    # training.self_consistency_confidence(from_run_number = 1, to_run_number = 2)
    # training.self_consistency_confidence_completion(from_run_number = 1, to_run_number = 2)
    # -------------------------------------------------------------------------

    # The whole sweep. The random split first as the reference, then every dataset
    # held out in turn. Each table holds two blocks, one per graded answer, and each
    # block holds the eight configurations followed by the two untrained rows that
    # take the vote share as it is, which is the bar the trained rows have to beat.
    training.print_reading_notes()
    training.ablation()
    for dataset in training.datasets:
        training.ablation(test_datasets = [dataset])
