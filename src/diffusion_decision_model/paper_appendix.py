"""The paper's appendix of per held out results, built from the ablation tables and plots.

For each of the four held out groups the main average covers, writes a table in the
same format as the paper's Table 1 (nv 5, both models) to tables/holdout_<group>.tex
and copies the same four plots the main text shows to figures/appendix/.  The
appendix section that inputs them lives in the paper itself, so it is not written here.

    python -m src.diffusion_decision_model.paper_appendix
"""

import os
import shutil

ABLATIONS = 'src/diffusion_decision_model/ablations'
LATEX = 'src/diffusion_decision_model/LATEX/iclr2027'
EVIDENCE_COUNT = 5

# (folder under ablations, suffix the paper's figure files use, column title)
MODELS = [('qwen-qwen3-8b', 'qwen3', 'Qwen3-8B'),
          ('deepseek-ai-deepseek-r1-distill-qwen-7b', 'deepseek_r1', 'DeepSeek-R1-Distill-Qwen-7B')]

# (name in the tables, plot folder, short name for files and labels, how the text names it)
GROUPS = [('mmlu,mmlu_pro', 'domain_multiple_choice_knowledge', 'mcq',
           'the multiple-choice knowledge domain (MMLU and MMLU-Pro)', 'Multiple-Choice Knowledge Domain'),
          ('gsm8k,math500,aime', 'domain_mathematics', 'math',
           'the mathematics domain (GSM8K, MATH500 and AIME)', 'Mathematics Domain'),
          ('gpqa', 'gpqa', 'gpqa', 'GPQA', 'GPQA'),
          ('truthfulqa', 'truthfulqa', 'truthfulqa', 'TruthfulQA', 'TruthfulQA')]

# The rows of Table 1, in its order: (block title, [method names]).
BLOCKS = [('Token-level baselines', ['Mean-Ent', 'Sum-Ent', 'Mean-Loss', 'Sum-Loss', 'Mean-Prob']),
          ('Representation-based baseline', ['LastRep']),
          ('Self-consistency baselines', ['SC-10', 'SC-Budget']),
          ('Proposed method', ['CoT-EIG-Mean', 'CoT-EIG-Sum'])]
OURS = ['CoT-EIG-Mean', 'CoT-EIG-Sum']

PLOTS = ['roc_against_ece', 'roc_by_evidence_count', 'ece_by_evidence_count', 'roc_ece_by_feature_set']


def held_out_rows(model, held_out):
    """{method: (roc, roc sd, ece, ece sd)} for one held out group at the chosen nv.

    Ours and the untrained baselines come from the CoT-EIG table; LastRep, a model
    trained on the last layer's representation, from its own feature set's table.
    Trained rows are the unscaled, unweighted ones, as in Table 1.
    """
    rows = {}
    for feature_set, trained in (('CoT-EIG', OURS), ('LastRep', ['LastRep'])):
        path = f'{ABLATIONS}/{model}/nv_{EVIDENCE_COUNT}/ablation_{feature_set}.txt'
        for line in open(path):
            fields = line.split()
            # held out, target, method, scaled, weight, fit, train, test, minority, roc, roc sd, ece, ece sd
            if len(fields) < 13 or fields[0] != held_out or fields[1] != 'run' or fields[-8] not in ('ok', 'STOP'):
                continue
            method, scaled, weight = ' '.join(fields[2:-10]), fields[-10], fields[-9]
            if (scaled, weight) != (('False', 'none') if method in trained else ('-', '-')) or method in rows:
                continue
            rows[method] = tuple(float(number) for number in (fields[-4], fields[-3], fields[-2], fields[-1]))
    return rows


def cell(roc_or_ece, sd, best):
    """One table cell, bold when it is the column's best, as Table 1 marks it."""
    if roc_or_ece != roc_or_ece:  # nan: the model has no such run
        return '---'
    text = f'{roc_or_ece:.2f} \\pm {sd:.2f}'
    return f'$\\mathbf{{{text}}}$' if best else f'${text}$'


def table(held_out, short, name):
    columns = [held_out_rows(model, held_out) for model, _, _ in MODELS]
    methods = [method for _, block in BLOCKS for method in block]

    # The best of each column among the rows the table shows, highest ROC and lowest
    # ECE as printed, every method that ties with it included.
    best = []
    for rows in columns:
        present = [method for method in methods if method in rows and rows[method][0] == rows[method][0]]
        top_roc = max(round(rows[method][0], 2) for method in present)
        low_ece = min(round(rows[method][2], 2) for method in present)
        best.append(({method for method in present if round(rows[method][0], 2) == top_roc},
                     {method for method in present if round(rows[method][2], 2) == low_ece}))

    lines = ['\\begin{table}[h]', '    \\centering',
             f'    \\caption{{Same as Table~\\ref{{tab:confidence_results}}, with {name} held out.}}',
             f'    \\label{{tab:holdout_{short}}}',
             '    \\resizebox{\\linewidth}{!}{%', '    \\begin{tabular}{lcc|cc}', '        \\toprule',
             '        \\multirow{2}{*}{Method} &',
             f'        \\multicolumn{{2}}{{c|}}{{{MODELS[0][2]}}} &',
             f'        \\multicolumn{{2}}{{c}}{{{MODELS[1][2]}}} \\\\',
             '        & AUROC $\\uparrow$ & ECE $\\downarrow$', '        & AUROC $\\uparrow$ & ECE $\\downarrow$ \\\\',
             '        \\midrule']
    for index, (title, block) in enumerate(BLOCKS):
        if index:
            lines.append('        \\midrule')
        lines += [f'        \\multicolumn{{5}}{{c}}{{\\textit{{{title}}}}} \\\\', '        \\midrule']
        for method in block:
            cells = []
            for rows, (best_roc, best_ece) in zip(columns, best):
                roc, roc_sd, ece, ece_sd = rows.get(method, (float('nan'),) * 4)
                cells += [cell(roc, roc_sd, method in best_roc), cell(ece, ece_sd, method in best_ece)]
            lines.append(f'        {method} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['        \\bottomrule', '    \\end{tabular}%', '    }', '\\end{table}', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    os.makedirs(f'{LATEX}/figures/appendix', exist_ok=True)
    os.makedirs(f'{LATEX}/tables', exist_ok=True)

    for held_out, folder, short, name, title in GROUPS:
        for model, suffix, _ in MODELS:
            for plot in PLOTS:
                shutil.copyfile(f'{ABLATIONS}/{model}/plots/{folder}/{plot}.png',
                                f'{LATEX}/figures/appendix/{plot}_{suffix}_{short}.png')
        with open(f'{LATEX}/tables/holdout_{short}.tex', 'w') as output:
            output.write(table(held_out, short, name))
    print(f'wrote {len(GROUPS)} tables and {len(GROUPS) * len(MODELS) * len(PLOTS)} figures')
