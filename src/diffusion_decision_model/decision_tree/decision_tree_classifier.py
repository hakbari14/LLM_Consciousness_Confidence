from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

class decision_tree_classifier:

    # The two questions the paper walks through: (dataset, Sample_ID, label, colour).
    PAPER_SAMPLES = [('gsm8k', 138, 'Sample A: GSM8K #138, answered correctly', 'green'),
                     ('gpqa', 122, 'Sample B: GPQA #122, answered incorrectly', 'red')]

    def __init__(self, number_of_evidence: int, modelname_dir: str = 'qwen-qwen3-8b') -> None:
        if number_of_evidence is None:
            raise Exception('number of evidence is required')

        self.number_of_evidence = number_of_evidence
        self.modelname_dir = modelname_dir
        self.datasets = ['gpqa', 'countdown', 'math500', 'gsm8k', 'mmlu', 'truthfulqa', 'mmlu_pro', 'aime']
        self.log_directory = './logs/diffusion_decision_model'
        self.log_cache = {}


    def train_decision_tree(self, from_run_number, to_run_number, test_size=0.2, random_state=42, max_depth=None, min_samples_leaf=10):
        X_list = []
        y_list = []
        keys = []
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
                    keys.append((dataset, int(log.sample_ID)))

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        X = np.asarray(X)
        y = np.asarray(y)

        print("Original X shape:", X.shape)
        print("y shape:", y.shape)

        X_flat = X.reshape(X.shape[0], -1)
        print("Flattened X shape:", X_flat.shape)

        X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
            X_flat,
            y,
            np.arange(len(y)),
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # The tree explains the logistic regression, so its target is the regression's
        # decision, 1 when its output is above 0.5, not the label.  The regression
        # itself is fitted on the label.
        regression = LogisticRegression(max_iter=10000, random_state=random_state)
        regression.fit(X_train, y_train)
        probability_test = regression.predict_proba(X_test)[:, 1]
        decision_train = regression.predict_proba(X_train)[:, 1] > 0.5
        decision_test = probability_test > 0.5

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

        clf.fit(X_train, decision_train)

        decision_pred = clf.predict(X_test)

        print("\n===== Results =====")
        print(f"Logistic regression ROC on test: {roc_auc_score(y_test, probability_test):.4f}")
        print(f"Regression output above 0.5: {decision_train.mean():.1%} of train, {decision_test.mean():.1%} of test")
        print(f"Tree agrees with the regression on test: {(decision_pred == decision_test).mean():.1%}, "
              f"balanced {balanced_accuracy_score(decision_test, decision_pred):.1%}")
        print(f"Tree ROC on test: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.4f}")

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

        channels = ['loss', 'agreement', 'delta_loss', 'delta_agreement']
        feature_names = [f"step{i // X.shape[2]}_{channels[i % X.shape[2]]}" for i in range(X_flat.shape[1])]

        # Only the tree with every split that ends in the same class on both sides
        # folded back into its parent is drawn: those splits change no decision.
        merged = self.merge_same_class_leaves(clf)

        # Wide enough that every leaf keeps its own box however deep the tree grows.
        figsize = (max(20, 1.8 * clf.get_n_leaves()), 4 + 2.2 * clf.get_depth())
        title = f"Decision Tree, max depth {max_depth}, at least {min_samples_leaf} questions per leaf, same-class leaves merged"
        directory = f"./src/diffusion_decision_model/decision_tree/{self.modelname_dir}"
        file_name = f"decision_tree{self.get_number_of_evidence_dir()}_depth_{max_depth}_leaf_{min_samples_leaf}.png"

        print("\n===== Rules, same-class leaves merged =====")
        print(export_text(merged, feature_names=feature_names, decimals=3))

        self.draw_paper_paths(merged, regression, X_flat, keys, set(index_test), feature_names,
                              figsize, title, f"{directory}/merged_paths", file_name)
        return clf

    def draw_tree(self, tree, feature_names, figsize, title):
        """The tree on a new figure; returns the figure and one annotation per drawn node."""
        figure = plt.figure(figsize=figsize)
        annotations = plot_tree(
            tree,
            filled=True,
            feature_names=feature_names,
            class_names=['0: regression <= 0.5', '1: regression > 0.5'],
            rounded=True,
            fontsize=8
        )
        plt.title(title)
        plt.tight_layout()
        return figure, annotations

    def held_out_confidence(self, dataset, sample_id):
        """The confidence the paper reports: the per token regression trained with this
        benchmark held out, as written by holdout_inference.py."""
        path = (f"./src/diffusion_decision_model/ablations/{self.modelname_dir}"
                f"/nv_{self.number_of_evidence}/inference_{dataset}.csv")
        inference = pd.read_csv(path)
        return float(inference.loc[inference['sample_ID'] == sample_id, 'confidence'].iloc[0])

    def decision_path(self, tree, x):
        """Node ids from the root to the leaf one question lands in."""
        x = x.astype(np.float32)  # the tree compares in float32, as predict does
        node, path = 0, [0]
        while tree.children_left[node] != -1:
            node = tree.children_left[node] if x[tree.feature[node]] <= tree.threshold[node] else tree.children_right[node]
            path.append(node)
        return path

    def drawn_nodes(self, tree):
        """Node ids in the order plot_tree draws them: root first, left subtree before right."""
        order, stack = [], [0]
        while stack:
            node = stack.pop()
            order.append(node)
            if tree.children_left[node] != -1:
                stack += [tree.children_right[node], tree.children_left[node]]
        return order

    def draw_paper_paths(self, merged, regression, X_flat, keys, test_indices, feature_names, figsize, title, folder, file_name):
        """The merged tree with each paper sample's route from root to leaf in its colour."""
        tree = merged.tree_
        figure, annotations = self.draw_tree(merged, feature_names, figsize, title + ", paper samples")
        axis = figure.axes[0]

        # Tie every drawn box to its node, and make sure the order really matches.  The
        # True / False labels under the root are annotations too, but not boxes.
        annotations = [annotation for annotation in annotations if annotation.get_text().strip() not in ('True', 'False')]
        nodes = self.drawn_nodes(tree)
        if len(nodes) != len(annotations):
            raise Exception('drawn boxes do not match the nodes of the tree')
        for node, annotation in zip(nodes, annotations):
            expected = feature_names[tree.feature[node]] if tree.children_left[node] != -1 else 'gini'
            if not annotation.get_text().startswith(expected):
                raise Exception(f'box of node {node} does not show {expected}')
        box = dict(zip(nodes, annotations))

        print("\n===== Paper samples =====")
        paths, handles = [], []
        for dataset, sample_id, label, colour in self.PAPER_SAMPLES:
            index = keys.index((dataset, sample_id))
            x = X_flat[index]
            path = self.decision_path(tree, x)
            if path[-1] != merged.apply(x[None].astype(np.float32))[0]:
                raise Exception(f'{label} path does not end where the tree puts it')
            leaf = path[-1]
            leaf_class = int(np.argmax(tree.value[leaf]))
            confidence = self.held_out_confidence(dataset, sample_id)
            paths.append((path, colour))

            print(f"{label}: in the {'test' if index in test_indices else 'train'} split, "
                  f"held out confidence {confidence:.3f}, tree class {leaf_class}")
            for node in path[:-1]:
                feature, threshold = tree.feature[node], tree.threshold[node]
                side = '<=' if np.float32(x[feature]) <= threshold else '> '
                print(f"    {feature_names[feature]:<22} = {x[feature]:>9.3f}  {side} {threshold:.3f}")
            print(f"    leaf: {tree.n_node_samples[leaf]} questions, class {leaf_class}")
            handles.append(Line2D([0], [0], color=colour, linewidth=4,
                                  label=f"{label}  →  class {leaf_class} (held out confidence {confidence:.3f})"))

        # Edges: a thick line under the boxes, offset side by side where both samples pass.
        edges = [set(zip(path, path[1:])) for path, _ in paths]
        shared_edges = edges[0] & edges[1]
        for (path, colour), offset in zip(paths, (-0.003, 0.003)):
            for parent, child in zip(path, path[1:]):
                shift = offset if (parent, child) in shared_edges else 0.0
                (x0, y0), (x1, y1) = box[parent].xyann, box[child].xyann
                axis.plot([x0 + shift, x1 + shift], [y0, y1], color=colour, linewidth=4,
                          alpha=0.85, solid_capstyle='round', transform=axis.transAxes, zorder=1)

        # Boxes: the sample's colour as the outline; a box both pass through gets both.
        shared_nodes = set(paths[0][0]) & set(paths[1][0])
        for path, colour in paths:
            for node in path:
                if node not in shared_nodes:
                    box[node].get_bbox_patch().set_edgecolor(colour)
                    box[node].get_bbox_patch().set_linewidth(3.5)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for node in shared_nodes:
            extent = box[node].get_bbox_patch().get_window_extent(figure.canvas.get_renderer())
            for colour, padding in [(paths[0][1], 2), (paths[1][1], 7)]:
                (x0, y0), (x1, y1) = to_figure.transform([(extent.x0 - padding, extent.y0 - padding),
                                                          (extent.x1 + padding, extent.y1 + padding)])
                figure.patches.append(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0',
                                                     fill=False, edgecolor=colour, linewidth=3.5,
                                                     transform=figure.transFigure, figure=figure, zorder=200))

        axis.legend(handles=handles, loc='upper left', fontsize=11, frameon=False)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/{file_name}", dpi=300, bbox_inches="tight")
        plt.close()

    def merge_same_class_leaves(self, clf):
        """A copy of the tree where a split whose two leaves give the same class is undone.

        Works from the bottom up, so a parent that becomes a leaf can be merged with its
        own sibling in turn.  The class of every question is unchanged; only splits that
        change nothing in the decision disappear.  The fitted tree is left as it is.
        """
        merged = copy.deepcopy(clf)
        tree = merged.tree_
        leaf = -1

        def merge(node):
            left, right = tree.children_left[node], tree.children_right[node]
            if left == leaf:
                return
            merge(left)
            merge(right)
            both_leaves = tree.children_left[left] == leaf and tree.children_left[right] == leaf
            if both_leaves and np.argmax(tree.value[left]) == np.argmax(tree.value[right]):
                # A leaf has no children and no split; export_text reads the second.
                tree.children_left[node] = leaf
                tree.children_right[node] = leaf
                tree.feature[node] = -2
                tree.threshold[node] = -2.0

        merge(0)
        return merged

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

    def get_modelname_dir(self) -> str:
        return self.modelname.replace('/', '-').lower()

    def get_number_of_evidence_dir(self) -> str:
        return f'_nv_{self.number_of_evidence}'


trainer = decision_tree_classifier(number_of_evidence=5)
for depth in range(3, 11):
    print(f"\n{'#' * 40} max depth {depth} {'#' * 40}")
    trainer.train_decision_tree(from_run_number=1, to_run_number=2, max_depth=depth)