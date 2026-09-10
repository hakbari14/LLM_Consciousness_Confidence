"""Plots over the ablation tables, so the numbers can be read at a glance.

Reads the text files the training run writes and puts PNGs next to them.  Nothing
here touches the training code, the logs, or the tables themselves.

    python -m src.diffusion_decision_model.ablation_plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ABLATIONS = 'src/diffusion_decision_model/ablations'
MODEL = 'qwen-qwen3-8b'
EVIDENCE_COUNTS = [5, 10, 15, 20, 25]
FEATURE_SETS = ['full', 'self_consistency', 'agreeing_length', 'rollout_length',
                'rollout_length_std', 'baseline_scalar', 'baseline_hidden']

# The four groups the main average covers, in the order they read best.
GROUPS = ['mmlu,mmlu_pro', 'gsm8k,math500,aime', 'gpqa', 'truthfulqa']
GROUP_LABELS = ['multiple choice\n(mmlu, mmlu_pro)', 'mathematics\n(gsm8k, math500, aime)', 'gpqa', 'truthfulqa']

OURS = 'evidence_per_token_loss'
OURS_TOTAL = 'evidence_total_loss'
BASELINES = ['baseline self cons 0', 'baseline self cons last', 'baseline cot loss',
             'baseline cot loss/tok', 'baseline entropy total', 'baseline mean token ent',
             'baseline arith mean prob']

BLUE, DARKBLUE, GREY, ORANGE, GREEN = '#1f6fb4', '#0d3c61', '#8a8a8a', '#d1701c', '#2e7d5b'

# Every plot names and colours a method the same way, using the table's own name.
# Baselines are grouped by family: votes orange, losses grey, entropy and
# probability green to purple.
NAMES = {OURS: 'ours, per token loss', OURS_TOTAL: 'ours, total loss'}
NAMES.update({name: name.replace('baseline ', '') for name in BASELINES})
COLOURS = {OURS: DARKBLUE, OURS_TOTAL: BLUE,
           'baseline self cons 0': '#d1701c', 'baseline self cons last': '#e8a86a',
           'baseline cot loss': '#a3a3a3', 'baseline cot loss/tok': '#5c5c5c',
           'baseline entropy total': '#2e7d5b', 'baseline mean token ent': '#5fae8f',
           'baseline arith mean prob': '#8e5fa8'}


def table_path(evidence_count, feature_set):
    return f'{ABLATIONS}/{MODEL}/nv_{evidence_count}/ablation_{feature_set}.txt'


def read_main_average(path, target='run'):
    """The MAIN AVERAGE block: {(method, scaled, weight): (roc, ece, ece_minmax)}."""
    rows, inside = {}, False
    for line in open(path):
        if line.startswith('MAIN AVERAGE'):
            inside = True
            continue
        if not inside:
            continue

        fields = line.split()
        if len(fields) < 7 or fields[0] != target:
            continue

        try:
            roc, ece, minmax = float(fields[-3]), fields[-2], fields[-1]
        except ValueError:
            continue

        method = ' '.join(fields[1:-5])
        rows[(method, fields[-5], fields[-4])] = (roc, ece, minmax)

    return rows


def read_per_benchmark(path, target='run'):
    """Every per held out row: {(held_out, method, scaled, weight): (roc, minority)}."""
    rows = {}
    for line in open(path):
        fields = line.split()
        if len(fields) < 13 or fields[1] != target or fields[-8] not in ('ok', 'STOP'):
            continue

        try:
            roc, minority = float(fields[-4]), int(fields[-5])
        except ValueError:
            continue

        method = ' '.join(fields[2:-10])
        rows[(fields[0], method, fields[-10], fields[-9])] = (roc, minority)

    return rows


def method_key(method):
    """How a method appears in the MAIN AVERAGE: trained rows unscaled, no weight."""
    return (method, 'False', 'none') if method in (OURS, OURS_TOTAL) else (method, '-', '-')


def spread_labels(values, gap):
    """Nudge label heights apart so none sit closer than gap, keeping their order."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for position in range(1, len(order)):
        below, here = order[position - 1], order[position]
        placed[here] = max(placed[here], placed[below] + gap)
    return placed


def plain(axis, title, xlabel, ylabel):
    """The look every plot here shares."""
    axis.set_title(title, fontsize=11, loc='left', pad=10)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis='y', color='#e6e6e6', linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)


def plot_roc_against_evidence_count(out_directory):
    """Does giving the model more evidence steps buy anything?"""
    averages = {nv: read_main_average(table_path(nv, 'full')) for nv in EVIDENCE_COUNTS}
    methods = [OURS, OURS_TOTAL] + BASELINES

    figure, axis = plt.subplots(figsize=(9, 5.4))
    last_values = []
    for method in methods:
        values = [averages[nv].get(method_key(method), (np.nan,))[0] for nv in EVIDENCE_COUNTS]
        ours = method in (OURS, OURS_TOTAL)
        axis.plot(EVIDENCE_COUNTS, values, '-' if ours else '--', color=COLOURS[method],
                  linewidth=2.4 if ours else 1.4, marker='o', markersize=5 if ours else 3.5)
        last_values.append(values[-1])

    # Names at the right end of each line instead of a nine entry legend.
    for method, value, height in zip(methods, last_values, spread_labels(last_values, 0.014)):
        axis.annotate(NAMES[method], xy=(EVIDENCE_COUNTS[-1], value), xytext=(26.2, height),
                      fontsize=8, color=COLOURS[method], va='center',
                      fontweight='bold' if method in (OURS, OURS_TOTAL) else 'normal',
                      arrowprops={'arrowstyle': '-', 'color': '#d0d0d0', 'linewidth': 0.6})

    axis.axhline(0.5, color='#7a7a7a', linewidth=1, linestyle=(0, (4, 3)))
    axis.text(4.6, 0.505, 'chance', fontsize=8, color='#7a7a7a')
    axis.set_xticks(EVIDENCE_COUNTS)
    axis.set_xlim(4, 31)
    axis.set_ylim(0.45, 0.9)
    plain(axis, 'More evidence steps do not help, for ours or for any baseline',
          'number of evidence steps', 'ROC, main average over four held out groups')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_by_evidence_count.png', dpi=160)
    plt.close(figure)


def plot_feature_sets(out_directory):
    """Which numbers actually carry the signal, and which are just length."""
    labels = ['full\n(loss + agreement)', 'agreement\nonly', 'agreeing rollout\nlength',
              'rollout\nlength', 'rollout length\n+ spread', 'published\nscalars', 'hidden state\n(4096 dim)']
    means, spreads = [], []
    for feature_set in FEATURE_SETS:
        values = []
        for nv in EVIDENCE_COUNTS:
            rows = read_main_average(table_path(nv, feature_set))
            key = (OURS, 'False', 'none') if feature_set == 'full' else ('-', 'False', 'none')
            if key in rows:
                values.append(rows[key][0])
        means.append(np.mean(values) if values else np.nan)
        spreads.append((np.max(values) - np.min(values)) / 2 if len(values) > 1 else 0.0)

    colours = [DARKBLUE, BLUE] + [GREY] * 3 + [ORANGE, ORANGE]
    figure, axis = plt.subplots(figsize=(9, 4.6))
    bars = axis.bar(range(len(labels)), means, yerr=spreads, capsize=3, zorder=2,
                    color=colours, width=0.66, error_kw={'elinewidth': 1, 'ecolor': '#666666'})
    for bar, value in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f'{value:.3f}',
                  ha='center', fontsize=8.5)

    axis.axhline(0.5, color='#7a7a7a', linewidth=1, linestyle=(0, (4, 3)), zorder=3)
    axis.text(len(labels) - 0.55, 0.507, 'chance', fontsize=8, color='#7a7a7a', ha='right', zorder=4)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylim(0.45, 0.92)
    plain(axis, 'The signal is in the loss and the agreement, not in how long the rollouts ran',
          '', 'ROC, averaged over the five evidence counts')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_by_feature_set.png', dpi=160)
    plt.close(figure)


def plot_per_benchmark(out_directory):
    """Every method on every held out group, averaged over the five evidence counts."""
    methods = [OURS, OURS_TOTAL] + BASELINES
    grid = np.full((len(methods), len(GROUPS)), np.nan)
    minorities = []
    for column, group in enumerate(GROUPS):
        per_nv = [read_per_benchmark(table_path(nv, 'full')) for nv in EVIDENCE_COUNTS]
        minorities.append(per_nv[EVIDENCE_COUNTS.index(20)].get(
            (group, OURS, 'False', 'none'), (np.nan, 0))[1])
        for row, method in enumerate(methods):
            values = [rows[(group,) + method_key(method)][0]
                      for rows in per_nv if (group,) + method_key(method) in rows]
            if values:
                grid[row, column] = np.mean(values)

    # Ours on top, then the baselines from strongest to weakest.
    order = [0, 1] + sorted(range(2, len(methods)), key=lambda row: -np.nanmean(grid[row]))
    grid, methods = grid[order], [methods[row] for row in order]

    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    image = axis.imshow(grid, cmap='Blues', vmin=0.5, vmax=1.0, aspect='auto')
    for row in range(len(methods)):
        for column in range(len(GROUPS)):
            value = grid[row, column]
            axis.text(column, row, f'{value:.3f}', ha='center', va='center', fontsize=9,
                      color='white' if value > 0.78 else '#1a1a1a',
                      fontweight='bold' if row < 2 else 'normal')

    axis.axhline(1.5, color='white', linewidth=3)
    axis.set_xticks(range(len(GROUPS)))
    axis.set_xticklabels([f'{label}\n{minority} of the rarer class'
                          for label, minority in zip(GROUP_LABELS, minorities)], fontsize=8)
    axis.set_yticks(range(len(methods)))
    axis.set_yticklabels([NAMES[method] for method in methods], fontsize=8.5)
    for tick, method in zip(axis.get_yticklabels(), methods):
        tick.set_color(COLOURS[method])
        tick.set_fontweight('bold' if method in (OURS, OURS_TOTAL) else 'normal')
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    colourbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colourbar.set_label('ROC', fontsize=8)
    colourbar.outline.set_visible(False)
    axis.set_title('ROC on each held out group, averaged over the five evidence counts',
                   fontsize=11, loc='left', pad=10)
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_by_benchmark.png', dpi=160)
    plt.close(figure)


def plot_discrimination_against_calibration(out_directory):
    """Ranking and calibration are different questions; this shows both at once."""
    rows = read_main_average(table_path(20, 'full'))
    # The offset keeps labels off each other where two points sit close.  Only the
    # scores that already are probabilities can be placed: the four raw scores
    # have no ECE by design.
    points = [(OURS, (9, 3)), (OURS_TOTAL, (9, -12)), ('baseline self cons 0', (9, 3)),
              ('baseline self cons last', (9, 6)), ('baseline arith mean prob', (-9, -16))]

    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    for method, offset in points:
        if method_key(method) not in rows:
            continue
        roc, ece, _ = rows[method_key(method)]
        label, colour = NAMES[method], COLOURS[method]
        if ece == 'NaN':
            continue
        axis.scatter(float(ece), roc, s=70, color=colour, zorder=3)
        axis.annotate(label, (float(ece), roc), textcoords='offset points',
                      xytext=offset, fontsize=8.5, color='#333333',
                      ha='right' if offset[0] < 0 else 'left')

    axis.axhline(0.5, color='#c0c0c0', linewidth=1, linestyle=':')
    axis.set_xlim(0.0, 0.30)
    axis.set_ylim(0.45, 0.92)
    plain(axis, 'Ours wins on both axes: it ranks better and is better calibrated\n'
                'only scores that already are probabilities can appear here',
          'expected calibration error (lower is better)', 'ROC (higher is better)')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_against_ece.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    out_directory = f'{ABLATIONS}/{MODEL}/plots'
    os.makedirs(out_directory, exist_ok=True)

    plot_roc_against_evidence_count(out_directory)
    plot_feature_sets(out_directory)
    plot_per_benchmark(out_directory)
    plot_discrimination_against_calibration(out_directory)

    for name in sorted(os.listdir(out_directory)):
        print(f'wrote {out_directory}/{name}')
