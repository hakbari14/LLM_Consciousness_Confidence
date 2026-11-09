"""The ablation plots, drawn over the run_2 tables.

ablation_plots.py reads the submitted paper's tables, whose CoT-EIG carried the loss.
This points the same plots at ablations_run_2/ and at its feature sets, and changes
nothing else.

    python -m src.diffusion_decision_model.ablation_plots_run_2
"""

import os
from src.diffusion_decision_model import ablation_plots as plots

plots.ABLATIONS = 'src/diffusion_decision_model/ablations_run_2'
plots.OURS, plots.OURS_TOTAL = 'CoT-EIG', 'EIG-SC'
plots.FEATURE_SET_BARS = [('CoT-EIG', 'CoT-EIG'), ('EIG-SC', 'EIG-SC'), ('EIG-Prob', 'EIG-Prob'),
                          ('CoT_Len', 'CoT_Len'), ('Logit_Feat', 'Logit_Feat'), ('LastRep', 'LastRep')]
plots.COLOURS.update({'CoT-EIG': '#1c5cab', 'EIG-SC': '#5598e7', 'EIG-Prob': '#9ec5f4'})
plots.MARKERS.update({'CoT-EIG': 'o', 'EIG-SC': 's', 'EIG-Prob': 'p'})
plots.HOLD_OUTS = [hold_out for hold_out in plots.HOLD_OUTS if hold_out[0] != 'countdown']

MODELS = ['qwen-qwen3-8b', 'deepseek-ai-deepseek-r1-distill-qwen-7b', 'meta_llama_llama_3_1_8b_instruct',
          'mistralai-mistral-7b-instruct-v0.3', 'deepseek-ai-deepseek-r1-distill-qwen-1.5b']

for plots.MODEL in [model for model in MODELS if os.path.isdir(f'{plots.ABLATIONS}/{model}')]:
    root = f'{plots.ABLATIONS}/{plots.MODEL}/plots'
    for held_out, folder in [(None, 'main_average')] + plots.HOLD_OUTS:
        out_directory = f'{root}/{folder}'
        os.makedirs(out_directory, exist_ok=True)
        plots.plot_against_evidence_count(out_directory, 0, held_out)
        plots.plot_against_evidence_count(out_directory, 1, held_out)
        plots.plot_feature_sets(out_directory, held_out)
        plots.plot_discrimination_against_calibration(out_directory, held_out)
        if held_out is None:
            plots.plot_per_benchmark(out_directory)
        print(f'wrote {out_directory}')
