"""Plots over the ablation tables, so the numbers can be read at a glance.

Reads the text files the training run writes and puts PNGs next to them.  Nothing
here touches the training code, the logs, or the tables themselves.  The main
average goes in plots/main_average/, and every held out dataset and domain gets its
own folder beside it holding the same plots for that one setting.

    python -m src.diffusion_decision_model.ablation_plots
"""

import os
import functools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ABLATIONS = 'src/diffusion_decision_model/ablations'
MODEL = 'qwen-qwen3-8b'
EVIDENCE_COUNTS = [5, 10, 15, 20, 25]

# Every model that has ablation tables, with the evidence counts it was run at.  The
# main block draws each in turn by rebinding MODEL and EVIDENCE_COUNTS above.
PLOT_RUNS = [('qwen-qwen3-8b', [5, 10, 15, 20, 25]),
             ('deepseek-ai-deepseek-r1-distill-qwen-7b', [5, 10, 15, 20])]

# The four groups the main average covers, in the order they read best.
GROUPS = ['mmlu,mmlu_pro', 'gsm8k,math500,aime', 'gpqa', 'truthfulqa']
GROUP_LABELS = ['multiple choice\n(mmlu, mmlu_pro)', 'mathematics\n(gsm8k, math500, aime)', 'gpqa', 'truthfulqa']

# Every held out setting that gets its own folder: (name in the tables, folder).
HOLD_OUTS = [('gpqa', 'gpqa'), ('truthfulqa', 'truthfulqa'), ('mmlu', 'mmlu'), ('mmlu_pro', 'mmlu_pro'),
             ('gsm8k', 'gsm8k'), ('math500', 'math500'), ('aime', 'aime'), ('countdown', 'countdown'),
             ('mmlu,mmlu_pro', 'domain_multiple_choice_knowledge'),
             ('gsm8k,math500,aime', 'domain_mathematics')]

# Ours, read per token (the reported one) and summed.
OURS, OURS_TOTAL = 'CoT-EIG-Mean', 'CoT-EIG-Sum'

# The baselines, as the tables name them.  The last three are models trained on their
# own feature set and are read from that set's table; the rest need no training.
UNTRAINED = ['SC-10', 'SC-Budget', 'Sum-Loss', 'Mean-Loss', 'Sum-Ent', 'Mean-Ent', 'Mean-Prob']
TRAINED = ['CoT_Len', 'Logit_Feat', 'LastRep']
BASELINES = UNTRAINED + TRAINED

# The feature set bars: (feature set, the row that stands for it).
FEATURE_SET_BARS = [('CoT-EIG', OURS), ('EIG-SC', 'EIG-SC'), ('EIG-Loss', 'EIG-Loss-Mean'),
                    ('CoT_Len', 'CoT_Len'), ('Logit_Feat', 'Logit_Feat'), ('LastRep', 'LastRep')]

# Raw scores, not probabilities: their ECE is measured after a min max rescaling.
MINMAX_SCORED = ['Sum-Loss', 'Mean-Loss', 'Sum-Ent', 'Mean-Ent']

# One colour and one marker per method, the same in every plot.  A hue per family
# (blue ours, orange and red self consistency, violet loss, aqua entropy, magenta
# probability, ochre length, green the trained whole completion baselines), each pair
# in a family split by lightness.  Checked with the dataviz palette rules: every
# adjacent pair clears the normal vision floor of 15; the two colour blind pairs at
# 6 to 7 are held apart by their markers and the labels.
COLOURS = {OURS: '#1c5cab', OURS_TOTAL: '#5598e7', 'EIG-SC': '#4a95ea', 'EIG-Loss-Mean': '#9ec5f4',
           'SC-10': '#eb6834', 'SC-Budget': '#b02424', 'Sum-Loss': '#4a3aa7', 'Mean-Loss': '#9085e9',
           'Sum-Ent': '#0f7a54', 'Mean-Ent': '#1baf7a', 'Mean-Prob': '#e87ba4', 'CoT_Len': '#c98500',
           'Logit_Feat': '#008300', 'LastRep': '#6fbf3a'}
MARKERS = {OURS: 'o', OURS_TOTAL: 's', 'SC-10': '^', 'SC-Budget': 'v', 'Sum-Loss': 'D', 'Mean-Loss': 'd',
           'Sum-Ent': 'P', 'Mean-Ent': 'X', 'Mean-Prob': '*', 'CoT_Len': '<', 'Logit_Feat': '>', 'LastRep': 'h'}
INK, MUTED = '#333333', '#898781'

# A star draws smaller than the other shapes at the same size, so it gets more.
def marker_size(method, size):
    return size * 1.5 if MARKERS[method] == '*' else size


def table_path(evidence_count, feature_set):
    return f'{ABLATIONS}/{MODEL}/nv_{evidence_count}/ablation_{feature_set}.txt'


@functools.lru_cache(maxsize=None)
def read_results(path, held_out = None, target = 'run'):
    """{(method, scaled, weight): (roc, ece, minority)} for one setting.

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
            # target, method, scaled, weight, roc, roc sd, ece, ece sd
            if not inside or len(fields) < 8 or fields[0] != target:
                continue
            method, scaled, weight = ' '.join(fields[1:-6]), fields[-6], fields[-5]
            numbers, minority = [fields[-4], fields[-2]], None
        else:
            # held out, target, method, scaled, weight, fit, train, test, minority, roc, roc sd, ece, ece sd
            if (len(fields) < 13 or fields[0] != held_out or fields[1] != target
                    or fields[-8] not in ('ok', 'STOP')):
                continue
            method, scaled, weight = ' '.join(fields[2:-10]), fields[-10], fields[-9]
            numbers, minority = [fields[-4], fields[-2]], fields[-5]

        try:
            roc, ece = (float(number) for number in numbers)
            rows[(method, scaled, weight)] = (roc, ece, None if minority is None else int(minority))
        except ValueError:
            continue

    return rows


def method_key(method):
    """How a method is keyed in the tables: trained rows unscaled with no weight."""
    return (method, '-', '-') if method in UNTRAINED or method == 'SC-10-Last' else (method, 'False', 'none')


def lookup(evidence_count, method, held_out = None):
    """(roc, ece, minority) of one method, from the table that holds it, or None.

    A trained baseline or ablation lives in its own feature set's table; ours and
    the untrained baselines in CoT-EIG's.
    """
    table = next((feature_set for feature_set, row in FEATURE_SET_BARS if row == method), 'CoT-EIG')
    return read_results(table_path(evidence_count, table), held_out).get(method_key(method))


def value(evidence_count, method, metric, held_out = None):
    """One number of one method: metric 0 is ROC, 1 is ECE.  nan when missing."""
    row = lookup(evidence_count, method, held_out)
    return row[metric] if row else np.nan


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


def label_line_ends(axis, methods, series, gap, hollow):
    """Each line's name at its last point, nudged apart, instead of a long legend.

    The name is in ink; the method's own marker beside it carries the colour, hollow
    where the line's markers are.
    """
    anchors = []
    for values in series:
        values = np.asarray(values, dtype=float)
        last = np.where(~np.isnan(values))[0][-1]
        anchors.append((EVIDENCE_COUNTS[last], values[last]))

    right = max(EVIDENCE_COUNTS)
    heights = spread_labels([value for _, value in anchors], gap)
    for method, (x, value), height, open_marker in zip(methods, anchors, heights, hollow):
        axis.plot([x, right + 1.3], [value, height], color='#d8d8d8', linewidth=0.6, zorder=1)
        axis.plot(right + 1.6, height, MARKERS[method], color=COLOURS[method], markersize=marker_size(method, 5),
                  markerfacecolor='white' if open_marker else COLOURS[method], clip_on=False)
        axis.text(right + 2.1, height, method, fontsize=8, color=INK, va='center',
                  fontweight='bold' if method in (OURS, OURS_TOTAL) else 'normal')


def plain(axis, xlabel, ylabel):
    """The look every plot here shares."""
    axis.set_xlabel(xlabel, fontsize=9, color=INK)
    axis.set_ylabel(ylabel, fontsize=9, color=INK)
    axis.grid(axis='y', color='#e6e6e6', linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)


def line_style(method, minmax = False):
    """Ours solid, untrained baselines dashed, trained baselines dash dot, rescaled ECE dotted."""
    if minmax:
        return ':'
    return '-' if method in (OURS, OURS_TOTAL) else ('-.' if method in TRAINED else '--')


def plot_against_evidence_count(out_directory, metric, held_out = None):
    """Does giving the model more evidence steps buy anything?  metric 0 ROC, 1 ECE."""
    methods, series = [], []
    for method in [OURS, OURS_TOTAL] + BASELINES:
        values = [value(nv, method, metric, held_out) for nv in EVIDENCE_COUNTS]
        if not np.all(np.isnan(values)):
            methods.append(method)
            series.append(values)

    figure, axis = plt.subplots(figsize=(9, 5.4))
    for method, values in zip(methods, series):
        ours = method in (OURS, OURS_TOTAL)
        minmax = metric == 1 and method in MINMAX_SCORED
        axis.plot(EVIDENCE_COUNTS, values, line_style(method, minmax), color=COLOURS[method],
                  linewidth=2.4 if ours else 1.4, marker=MARKERS[method], markersize=marker_size(method, 6 if ours else 5),
                  markerfacecolor='white' if minmax else COLOURS[method])

    if metric == 0:
        low, high = roc_range(series, 0.9)
        axis.axhline(0.5, color='#7a7a7a', linewidth=1, linestyle=(0, (4, 3)))
        axis.text(min(EVIDENCE_COUNTS) - 0.4, 0.5 + (high - low) * 0.012, 'chance', fontsize=8, color=MUTED)
    else:
        everything = np.array(series, dtype=float)
        low, high = max(0.0, np.nanmin(everything) - 0.02), np.nanmax(everything) + 0.02
    label_line_ends(axis, methods, series, (high - low) * 0.034,
                    [metric == 1 and method in MINMAX_SCORED for method in methods])

    axis.set_xticks(EVIDENCE_COUNTS)
    axis.set_xlim(min(EVIDENCE_COUNTS) - 1, max(EVIDENCE_COUNTS) + 6)
    axis.set_ylim(low, high)
    plain(axis, 'number of evidence steps', ['ROC', 'ECE'][metric])
    figure.tight_layout()
    figure.savefig(f"{out_directory}/{['roc', 'ece'][metric]}_by_evidence_count.png", dpi=160)
    plt.close(figure)


def plot_feature_sets(out_directory, held_out = None):
    """Which numbers carry the signal: ROC and ECE of each feature set, side by side."""
    names = [feature_set for feature_set, _ in FEATURE_SET_BARS]
    rows = [row for _, row in FEATURE_SET_BARS]

    figure, axis = plt.subplots(figsize=(9, 4.6))
    width = 0.36
    for metric, shift in ((0, -width / 2 - 0.01), (1, width / 2 + 0.01)):
        for position, method in enumerate(rows):
            values = [value(nv, method, metric, held_out) for nv in EVIDENCE_COUNTS]
            values = [number for number in values if not np.isnan(number)]
            if not values:
                continue
            mean = np.mean(values)
            spread = (np.max(values) - np.min(values)) / 2 if len(values) > 1 else 0.0
            colour = COLOURS[method]
            # ROC a solid bar, ECE a hatched one in the same colour.
            axis.bar(position + shift, mean, width, yerr=spread, capsize=2.5, zorder=2,
                     color=colour if metric == 0 else colour + '40', edgecolor=colour,
                     hatch=None if metric == 0 else '////', linewidth=0 if metric == 0 else 1,
                     error_kw={'elinewidth': 0.9, 'ecolor': '#666666'})
            axis.text(position + shift, mean + spread + 0.015, f'{mean:.3f}', ha='center',
                      fontsize=7.5, color=INK)

    axis.legend(handles=[Patch(facecolor='#8a8a8a', label='ROC'),
                         Patch(facecolor='#8a8a8a40', edgecolor='#8a8a8a', hatch='////', label='ECE')],
                loc='upper right', fontsize=8.5, frameon=False)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, fontsize=9, color=INK)
    axis.set_ylim(0, 1.0)
    plain(axis, '', 'ROC / ECE')
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_ece_by_feature_set.png', dpi=160)
    plt.close(figure)


def plot_per_benchmark(out_directory):
    """Every method on every held out group, averaged over the evidence counts."""
    methods = [OURS, OURS_TOTAL] + BASELINES
    grid = np.full((len(methods), len(GROUPS)), np.nan)
    for column, group in enumerate(GROUPS):
        for row, method in enumerate(methods):
            values = [value(nv, method, 0, group) for nv in EVIDENCE_COUNTS]
            values = [number for number in values if not np.isnan(number)]
            if values:
                grid[row, column] = np.mean(values)

    # Ours on top, then the baselines from strongest to weakest.  A baseline this
    # model has no numbers for gets no row.
    measured = [row for row in range(2, len(methods)) if not np.all(np.isnan(grid[row]))]
    order = [0, 1] + sorted(measured, key=lambda row: -np.nanmean(grid[row]))
    grid, methods = grid[order], [methods[row] for row in order]

    figure, axis = plt.subplots(figsize=(8.8, 0.52 * len(methods) + 1.4))
    image = axis.imshow(grid, cmap='Blues', vmin=0.5, vmax=1.0, aspect='auto')
    for row in range(len(methods)):
        for column in range(len(GROUPS)):
            number = grid[row, column]
            axis.text(column, row, f'{number:.3f}', ha='center', va='center', fontsize=9,
                      color='white' if number > 0.78 else '#1a1a1a',
                      fontweight='bold' if row < 2 else 'normal')

    axis.axhline(1.5, color='white', linewidth=3)
    axis.set_xticks(range(len(GROUPS)))
    axis.set_xticklabels(GROUP_LABELS, fontsize=8, color=INK)
    axis.set_yticks(range(len(methods)))
    axis.set_yticklabels(methods, fontsize=8.5, color=INK)
    for tick, method in zip(axis.get_yticklabels(), methods):
        tick.set_fontweight('bold' if method in (OURS, OURS_TOTAL) else 'normal')
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    colourbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colourbar.set_label('ROC', fontsize=8)
    colourbar.outline.set_visible(False)
    figure.tight_layout()
    figure.savefig(f'{out_directory}/roc_by_benchmark.png', dpi=160)
    plt.close(figure)


def place_labels(figure, axis, placed):
    """Each point's name where it covers no other name, no marker and no edge.

    Every label tries the spots around its point, nearest first, and takes the
    first free one.  A label that had to move away gets a thin line to its point.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer)
    markers = [axis.transData.transform((ece, roc)) for _, ece, roc in placed]
    taken = []

    def free(box, own):
        if box.x0 < frame.x0 or box.x1 > frame.x1 or box.y0 < frame.y0 or box.y1 > frame.y1:
            return False
        if any(box.overlaps(other) for other in taken):
            return False
        return all(not (box.x0 - 6 < x < box.x1 + 6 and box.y0 - 6 < y < box.y1 + 6)
                   for index, (x, y) in enumerate(markers) if index != own)

    def measure(method, ece, roc, offset, ha, va, weight):
        probe = axis.annotate(method, (ece, roc), xytext=offset, textcoords='offset points', ha=ha, va=va,
                              fontsize=8.5, fontweight=weight)
        box = probe.get_window_extent(renderer)
        probe.remove()
        return box

    # Right, left, above, below, the four corners, then the same further out.
    directions = [(1, 0, 'left', 'center'), (-1, 0, 'right', 'center'), (0, 1, 'center', 'bottom'),
                  (0, -1, 'center', 'top'), (1, 1, 'left', 'bottom'), (1, -1, 'left', 'top'),
                  (-1, 1, 'right', 'bottom'), (-1, -1, 'right', 'top')]
    spots = [((dx * distance, dy * distance), ha, va, distance)
             for distance in (7, 14, 22, 32, 44) for dx, dy, ha, va in directions]
    for own, (method, ece, roc) in enumerate(placed):
        weight = 'bold' if method in (OURS, OURS_TOTAL) else 'normal'
        offset, ha, va, distance = next((spot for spot in spots
                                         if free(measure(method, ece, roc, spot[0], spot[1], spot[2], weight), own)),
                                        spots[0])
        line = {'arrowstyle': '-', 'color': '#b0b0b0', 'linewidth': 0.6, 'shrinkA': 1, 'shrinkB': 4}
        label = axis.annotate(method, (ece, roc), xytext=offset, textcoords='offset points', ha=ha, va=va,
                              fontsize=8.5, color=INK, fontweight=weight,
                              arrowprops=line if distance > 7 else None)
        taken.append(label.get_window_extent(renderer))


def plot_discrimination_against_calibration(out_directory, held_out = None):
    """Ranking and calibration are different questions; this shows both at once.

    Only the scores that already are probabilities are placed: the four raw scores'
    ECE is measured on a rescaled score, which is not the same calibration question.
    """
    points = [method for method in [OURS, OURS_TOTAL] + BASELINES if method not in MINMAX_SCORED]
    evidence_count = 20 if 20 in EVIDENCE_COUNTS else max(EVIDENCE_COUNTS)

    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    placed = []
    for method in points:
        roc, ece = value(evidence_count, method, 0, held_out), value(evidence_count, method, 1, held_out)
        if np.isnan(roc) or np.isnan(ece):
            continue
        placed.append((method, ece, roc))
        axis.scatter(ece, roc, s=marker_size(method, 60) * 1.5 if MARKERS[method] == '*' else 60,
                     color=COLOURS[method], marker=MARKERS[method], zorder=3,
                     edgecolors='white', linewidths=0.8)

    axis.axhline(0.5, color='#c0c0c0', linewidth=1, linestyle=':')
    axis.set_xlim(0.0, max(0.30, max((ece for _, ece, _ in placed), default=0.0) + 0.08))
    axis.set_ylim(*roc_range([roc for _, _, roc in placed], 0.92))
    plain(axis, 'ECE', 'ROC')
    figure.tight_layout()
    place_labels(figure, axis, placed)
    figure.savefig(f'{out_directory}/roc_against_ece.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    for MODEL, EVIDENCE_COUNTS in PLOT_RUNS:
        root = f'{ABLATIONS}/{MODEL}/plots'
        main_directory = f'{root}/main_average'
        os.makedirs(main_directory, exist_ok=True)

        plot_against_evidence_count(main_directory, 0)
        plot_against_evidence_count(main_directory, 1)
        plot_feature_sets(main_directory)
        plot_per_benchmark(main_directory)
        plot_discrimination_against_calibration(main_directory)
        print(f'wrote {main_directory}')

        # The same four plots for every single held out dataset and domain.  The
        # benchmark table is left out: it already puts every group side by side.
        for held_out, folder in HOLD_OUTS:
            out_directory = f'{root}/{folder}'
            os.makedirs(out_directory, exist_ok=True)
            plot_against_evidence_count(out_directory, 0, held_out)
            plot_against_evidence_count(out_directory, 1, held_out)
            plot_feature_sets(out_directory, held_out)
            plot_discrimination_against_calibration(out_directory, held_out)
            print(f'wrote {out_directory}')
