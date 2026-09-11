"""Plots over the ablation tables, so the numbers can be read at a glance.

Reads the text files the training run writes and puts PNGs next to them.  Nothing
here touches the training code, the logs, or the tables themselves.  The main
average goes in plots/main_average/, and every held out dataset and domain gets its
own folder beside it holding the same plots for that one setting.

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

# Every held out setting that gets its own folder: (name in the tables, folder, label).
HOLD_OUTS = [('gpqa', 'gpqa', 'gpqa'),
             ('truthfulqa', 'truthfulqa', 'truthfulqa'),
             ('mmlu', 'mmlu', 'mmlu'),
             ('mmlu_pro', 'mmlu_pro', 'mmlu_pro'),
             ('gsm8k', 'gsm8k', 'gsm8k'),
             ('math500', 'math500', 'math500'),
             ('aime', 'aime', 'aime'),
             ('countdown', 'countdown', 'countdown'),
             ('mmlu,mmlu_pro', 'domain_multiple_choice_knowledge', 'multiple choice domain'),
             ('gsm8k,math500,aime', 'domain_mathematics', 'mathematics domain')]

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


def read_results(path, held_out = None, target = 'run'):
    """{(method, scaled, weight): (roc, ece, ece_minmax, minority)} for one setting.

    With no held out named this reads the MAIN AVERAGE block, which carries no
    minority count.  With one named it reads that setting's rows of the tables.
    A missing number comes back as nan.
    """
    rows, inside = {}, False
    for line in open(path):
        if held_out is None and line.startswith('MAIN AVERAGE'):
            inside = True
            continue

        fields = line.split()
        if held_out is None:
            # target, method, scaled, weight, roc, ece, ece_minmax
            if not inside or len(fields) < 7 or fields[0] != target:
                continue
            method, scaled, weight = ' '.join(fields[1:-5]), fields[-5], fields[-4]
            numbers, minority = fields[-3:], None
        else:
            # held out, target, method, scaled, weight, fit, train, test, minority, roc, roc sd, ece, ece_minmax
            if (len(fields) < 13 or fields[0] != held_out or fields[1] != target
                    or fields[-8] not in ('ok', 'STOP')):
                continue
            method, scaled, weight = ' '.join(fields[2:-10]), fields[-10], fields[-9]
            numbers, minority = [fields[-4], fields[-2], fields[-1]], fields[-5]

        try:
            roc, ece, minmax = (float(number) for number in numbers)
            rows[(method, scaled, weight)] = (roc, ece, minmax, None if minority is None else int(minority))
        except ValueError:
            continue

    return rows


def method_key(method):
    """How a method is keyed in the tables: trained rows unscaled with no weight."""
    return (method, 'False', 'none') if method in (OURS, OURS_TOTAL) else (method, '-', '-')


def describe(held_out, label):
    """What the y label says the plot covers, and the x label's count of wrong answers."""
    if held_out is None:
        return 'main average over four held out groups', ''

    minority = read_results(table_path(20, 'full'), held_out).get(method_key(OURS), (0, 0, 0, None))[3]
    note = f'\n[{label}: {minority} of the rarer class at 20 steps]' if minority is not None else ''
    return f'{label} held out', note


def roc_range(values, top):
    """The main average's familiar window, widened only when a held out goes outside it."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[~np.isnan(values)]
    if not len(values):
        return 0.45, top
    return max(0.0, min(0.45, values.min() - 0.05)), min(1.03, max(top, values.max() + 0.05))


def spread_labels(values, gap):
    """Nudge label heights apart so none sit closer than gap, keeping their order."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for position in range(1, len(order)):
        below, here = order[position - 1], order[position]
        placed[here] = max(placed[here], placed[below] + gap)
    return placed


def label_line_ends(axis, methods, series, gap, suffixes):
    """Each line's name at its last point, nudged apart, instead of a long legend."""
    anchors = []
    for values in series:
        values = np.asarray(values, dtype=float)
        last = np.where(~np.isnan(values))[0][-1]
        anchors.append((EVIDENCE_COUNTS[last], values[last]))

    heights = spread_labels([value for _, value in anchors], gap)
    for method, (x, value), height, suffix in zip(methods, anchors, heights, suffixes):
        axis.annotate(NAMES[method] + suffix, xy=(x, value), xytext=(26.2, height),
                      fontsize=8, color=COLOURS[method], va='center',
                      fontweight='bold' if method in (OURS, OURS_TOTAL) else 'normal',
                      arrowprops={'arrowstyle': '-', 'color': '#d0d0d0', 'linewidth': 0.6})


def plain(axis, xlabel, ylabel):
    """The look every plot here shares."""
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis='y', color='#e6e6e6', linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)


def plot_roc_against_evidence_count(out_directory, held_out = None, label = None):
    """Does giving the model more evidence steps buy anything?"""
    results = {nv: read_results(table_path(nv, 'full'), held_out) for nv in EVIDENCE_COUNTS}
    setting, note = describe(held_out, label)

    methods, series = [], []
    for method in [OURS, OURS_TOTAL] + BASELINES:
        values = [results[nv].get(method_key(method), (np.nan,))[0] for nv in EVIDENCE_COUNTS]
        if not np.all(np.isnan(values)):
            methods.append(method)
            series.append(values)

    figure, axis = plt.subplots(figsize=(9, 5.4))
    for method, values in zip(methods, series):
        ours = method in (OURS, OURS_TOTAL)
        axis.plot(EVIDENCE_COUNTS, values, '-' if ours else '--', color=COLOURS[method],
                  linewidth=2.4 if ours else 1.4, marker='o', markersize=5 if ours else 3.5)

    low, high = roc_range(series, 0.9)
    label_line_ends(axis, methods, series, (high - low) * 0.031, [''] * len(methods))

    axis.axhline(0.5, color='#7a7a7a', linewidth=1, linestyle=(0, (4, 3)))
    axis.text(4.6, 0.5 + (high - low) * 0.012, 'chance', fontsize=8, color='#7a7a7a')
    axis.set_xticks(EVIDENCE_COUNTS)
    axis.set_xlim(4, 31)
    axis.set_ylim(low, high)
    plain(axis, 'number of evidence steps' + note, f'ROC, {setting}')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_by_evidence_count.png', dpi=160)
    plt.close(figure)


def plot_ece_against_evidence_count(out_directory, held_out = None, label = None):
    """The same question as the ROC plot, asked of calibration.  Lower is better here.

    A raw log likelihood or entropy has no ECE of its own, so those four are drawn
    with the min max ECE instead: dotted and marked, because that number says more
    about where the score's range happens to land than about calibration.
    """
    results = {nv: read_results(table_path(nv, 'full'), held_out) for nv in EVIDENCE_COUNTS}
    setting, note = describe(held_out, label)

    methods, series, uses_minmax = [], [], []
    for method in [OURS, OURS_TOTAL] + BASELINES:
        rows = [results[nv].get(method_key(method)) for nv in EVIDENCE_COUNTS]
        ece = [row[1] if row else np.nan for row in rows]
        minmax = [row[2] if row else np.nan for row in rows]
        fallback = bool(np.all(np.isnan(ece)))
        values = minmax if fallback else ece
        if not np.all(np.isnan(values)):
            methods.append(method)
            series.append(values)
            uses_minmax.append(fallback)

    figure, axis = plt.subplots(figsize=(9, 5.4))
    for method, values, fallback in zip(methods, series, uses_minmax):
        ours = method in (OURS, OURS_TOTAL)
        style = ':' if fallback else ('-' if ours else '--')
        axis.plot(EVIDENCE_COUNTS, values, style, color=COLOURS[method],
                  linewidth=2.4 if ours else 1.4, marker='o', markersize=5 if ours else 3.5,
                  markerfacecolor='white' if fallback else COLOURS[method])

    everything = np.array(series, dtype=float)
    low, high = max(0.0, np.nanmin(everything) - 0.02), np.nanmax(everything) + 0.02
    label_line_ends(axis, methods, series, (high - low) * 0.035,
                    [' (minmax)' if fallback else '' for fallback in uses_minmax])

    axis.set_xticks(EVIDENCE_COUNTS)
    axis.set_xlim(4, 31)
    axis.set_ylim(low, high)
    plain(axis, 'number of evidence steps' + note, f'ECE (lower is better), {setting}')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/ece_by_evidence_count.png', dpi=160)
    plt.close(figure)


def plot_feature_sets(out_directory, held_out = None, label = None):
    """Which numbers actually carry the signal, and which are just length."""
    labels = ['full\n(loss +\nself_consistency)', 'self_consistency', 'agreeing rollout\nlength',
              'rollout\nlength', 'rollout length\n+ spread', 'published\nscalars', 'hidden state\n(4096 dim)']
    setting, note = describe(held_out, label)

    means, spreads = [], []
    for feature_set in FEATURE_SETS:
        key = method_key(OURS) if feature_set == 'full' else ('-', 'False', 'none')
        values = []
        for nv in EVIDENCE_COUNTS:
            rows = read_results(table_path(nv, feature_set), held_out)
            if key in rows and not np.isnan(rows[key][0]):
                values.append(rows[key][0])
        means.append(np.mean(values) if values else np.nan)
        spreads.append((np.max(values) - np.min(values)) / 2 if len(values) > 1 else 0.0)

    means, spreads = np.array(means), np.array(spreads)
    low, high = roc_range(np.concatenate([means - spreads, means + spreads]), 0.92)

    colours = [DARKBLUE, BLUE] + [GREY] * 3 + [ORANGE, ORANGE]
    figure, axis = plt.subplots(figsize=(9, 4.6))
    bars = axis.bar(range(len(labels)), means, yerr=spreads, capsize=3, zorder=2,
                    color=colours, width=0.66, error_kw={'elinewidth': 1, 'ecolor': '#666666'})
    for bar, value, spread in zip(bars, means, spreads):
        if not np.isnan(value):
            axis.text(bar.get_x() + bar.get_width() / 2, value + spread + (high - low) * 0.02,
                      f'{value:.3f}', ha='center', fontsize=8.5)

    axis.axhline(0.5, color='#7a7a7a', linewidth=1, linestyle=(0, (4, 3)), zorder=3)
    axis.text(len(labels) - 0.55, 0.5 + (high - low) * 0.015, 'chance', fontsize=8,
              color='#7a7a7a', ha='right', zorder=4)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylim(low, high)
    plain(axis, note.strip(), f'ROC averaged over the five evidence counts\n{setting}')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_by_feature_set.png', dpi=160)
    plt.close(figure)


def plot_per_benchmark(out_directory):
    """Every method on every held out group, averaged over the five evidence counts."""
    methods = [OURS, OURS_TOTAL] + BASELINES
    grid = np.full((len(methods), len(GROUPS)), np.nan)
    minorities = []
    for column, group in enumerate(GROUPS):
        per_nv = [read_results(table_path(nv, 'full'), group) for nv in EVIDENCE_COUNTS]
        minorities.append(per_nv[EVIDENCE_COUNTS.index(20)].get(method_key(OURS), (0, 0, 0, 0))[3])
        for row, method in enumerate(methods):
            values = [rows[method_key(method)][0] for rows in per_nv if method_key(method) in rows]
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
    colourbar.set_label('ROC, averaged over the five evidence counts', fontsize=8)
    colourbar.outline.set_visible(False)
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_by_benchmark.png', dpi=160)
    plt.close(figure)


def plot_discrimination_against_calibration(out_directory, held_out = None, label = None):
    """Ranking and calibration are different questions; this shows both at once."""
    rows = read_results(table_path(20, 'full'), held_out)
    setting, note = describe(held_out, label)

    # The offset keeps labels off each other where two points sit close.  Only the
    # scores that already are probabilities can be placed: the four raw scores
    # have no ECE by design.
    points = [(OURS, (9, 3)), (OURS_TOTAL, (9, -12)), ('baseline self cons 0', (9, 3)),
              ('baseline self cons last', (9, 6)), ('baseline arith mean prob', (-9, -16))]

    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    placed_ece, placed_roc = [], []
    for method, offset in points:
        if method_key(method) not in rows:
            continue
        roc, ece = rows[method_key(method)][:2]
        if np.isnan(roc) or np.isnan(ece):
            continue
        placed_ece.append(ece)
        placed_roc.append(roc)
        axis.scatter(ece, roc, s=70, color=COLOURS[method], zorder=3)
        axis.annotate(NAMES[method], (ece, roc), textcoords='offset points',
                      xytext=offset, fontsize=8.5, color='#333333',
                      ha='right' if offset[0] < 0 else 'left')

    axis.axhline(0.5, color='#c0c0c0', linewidth=1, linestyle=':')
    axis.set_xlim(0.0, max(0.30, max(placed_ece, default=0.0) + 0.05))
    axis.set_ylim(*roc_range(placed_roc, 0.92))
    plain(axis, 'expected calibration error (lower is better), 20 evidence steps' + note,
          f'ROC (higher is better), {setting}')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_against_ece.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    root = f'{ABLATIONS}/{MODEL}/plots'
    main_directory = f'{root}/main_average'
    os.makedirs(main_directory, exist_ok=True)

    plot_roc_against_evidence_count(main_directory)
    plot_ece_against_evidence_count(main_directory)
    plot_feature_sets(main_directory)
    plot_per_benchmark(main_directory)
    plot_discrimination_against_calibration(main_directory)
    print(f'wrote {main_directory}')

    # The same four plots for every single held out dataset and domain.  The
    # benchmark table is left out: it already puts every group side by side.
    for held_out, folder, label in HOLD_OUTS:
        out_directory = f'{root}/{folder}'
        os.makedirs(out_directory, exist_ok=True)
        plot_roc_against_evidence_count(out_directory, held_out, label)
        plot_ece_against_evidence_count(out_directory, held_out, label)
        plot_feature_sets(out_directory, held_out, label)
        plot_discrimination_against_calibration(out_directory, held_out, label)
        print(f'wrote {out_directory}')
