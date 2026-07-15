import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import re 
from sklearn.feature_selection import mutual_info_regression
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

class confidence_inference_analysis(object):

    @staticmethod
    def calculate_auroc_entropy_iit_reward(confidence_type: str) -> None:
        data_list = []
        dir, csv_paths = confidence_inference_analysis.get_filenames(confidence_type)
        for dataset, csv_dataset in csv_paths.items():
            file_paths = csv_dataset['file_paths']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            for file_path in file_paths: 
                for run_number in range(from_run_number, to_run_number):
                    try:
                        iit_type = confidence_inference_analysis.extract_iit_type(file_path)
                        if iit_type is None: continue

                        file_path_run_number = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path_run_number}')
                        required_cols = ["Accuracy", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss", "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]
                        confidence_inference_analysis.check_columns(df, required_cols)
                        
                        accuracy = df['Accuracy'].mean()
                        df = df[["Accuracy", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss" , "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]].dropna()

                        df['Accuracy_Reward'] = df['Accuracy'].map({True: 1, False: 0})        
                        accuracy_list = df['Accuracy_Reward'].tolist()

                        df['confidence_sum_probabilty'] = confidence_inference_analysis.convert_into_probability(df['Sequence_Probability'])
                        sum_probability_list = df['confidence_sum_probabilty'].tolist()
                        
                        norm_seq_prob_list = df['Length_Normalized_Sequence_Probability'].tolist()

                        df['confidence_entropy'] = confidence_inference_analysis.convert_into_probability(df['Entropy'], is_inverse=True)
                        confidence_entropy_list = df['confidence_entropy'].tolist()

                        df['confidence_loss'] = confidence_inference_analysis.convert_into_probability(df['Completion_Loss'], is_inverse=True)
                        confidence_loss_list = df['confidence_loss'].tolist()

                        df['confidence_iit_reward'] = confidence_inference_analysis.convert_into_probability(df['Phi_Reward_Raw'])
                        confidence_iit_reward_list = df['confidence_iit_reward'].tolist()

                        df['confidence_tpm_loss'] = confidence_inference_analysis.convert_into_probability(df['Tpm_Loss'], is_inverse=True)
                        df['confidence_iit_reward_tpm_loss'] = (df['confidence_iit_reward'] + df['confidence_tpm_loss']) / 2.0
                        iit_reward_tpm_loss_list = df['confidence_iit_reward_tpm_loss'].tolist()

                        df['confidence_tpm_entropy'] = confidence_inference_analysis.convert_into_probability(df['Tpm_Entropy'], is_inverse=True)
                        tpm_entropy_list = df['confidence_tpm_entropy'].tolist()
                        df['confidence_iit_reward_tpm_entropy'] = (df['confidence_iit_reward'] - df['confidence_tpm_entropy']) / 2.0
                        iit_reward_tpm_entropy_list = df['confidence_iit_reward_tpm_entropy'].tolist()

                        df['confidence_iit_reward_entropy'] = (df['confidence_iit_reward'] + df['confidence_entropy']) / 2.0
                        iit_reward_entropy_list = df['confidence_iit_reward_entropy'].tolist()
                        
                        correlation_entropy_tpm_entropy = df['confidence_entropy'].corr(df['confidence_tpm_entropy'])
                        y_true = np.array(accuracy_list)
                        accuracy = np.average(y_true)

                        con_A = np.array(sum_probability_list)
                        con_B = np.array(norm_seq_prob_list)
                        con_C = np.array(confidence_entropy_list)
                        con_D = np.array(confidence_loss_list)
                        con_E = np.array(confidence_iit_reward_list)
                        con_F = np.array(iit_reward_tpm_loss_list)
                        con_G = np.array(iit_reward_tpm_entropy_list)
                        con_H = np.array(iit_reward_entropy_list)

                        fpr1, tpr1, _ = roc_curve(y_true, con_A)
                        roc_auc1 = auc(fpr1, tpr1)

                        fpr2, tpr2, _ = roc_curve(y_true, con_B)
                        roc_auc2 = auc(fpr2, tpr2)

                        fpr3, tpr3, _ = roc_curve(y_true, con_C)
                        roc_auc3 = auc(fpr3, tpr3)

                        fpr4, tpr4, _ = roc_curve(y_true, con_D)
                        roc_auc4 = auc(fpr4, tpr4)
                        
                        fpr5, tpr5, _ = roc_curve(y_true, con_E)
                        roc_auc5 = auc(fpr5, tpr5)
                        
                        fpr6, tpr6, _ = roc_curve(y_true, con_F)
                        roc_auc6 = auc(fpr6, tpr6)

                        fpr7, tpr7, _ = roc_curve(y_true, con_G)
                        roc_auc7 = auc(fpr7, tpr7)

                        fpr8, tpr8, _ = roc_curve(y_true, con_H)
                        roc_auc8 = auc(fpr8, tpr8)
                        
                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "settings" : iit_type, 
                                        "accuracy": accuracy,
                                        "sum_prob": roc_auc1,
                                        "avg_prob": roc_auc2,
                                        "entropy": roc_auc3,
                                        "loss": roc_auc4,
                                        "iit": roc_auc5,
                                        "iit_tpm_loss": roc_auc6,
                                        "iit_tpm_entropy": roc_auc7,
                                        "iit_entropy": roc_auc8,
                                        "correlation_entropy_tpm_entropy": correlation_entropy_tpm_entropy,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")
        
        df_summary = pd.DataFrame(data_list)
        group_cols=['dataset', 'settings']        
        value_cols=['accuracy','sum_prob','avg_prob','entropy', 'loss', 'iit', 'iit_tpm_loss', 'iit_tpm_entropy', 'iit_entropy', 'correlation_entropy_tpm_entropy']
        df_summary = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['settings', 'dataset'])        
        print(f'{confidence_type} Settings')
        print(df_summary.to_string(index=False))        

        print()
        
        df_summary_dataset = pd.DataFrame(data_list)
        group_cols=['settings']        
        value_cols=['accuracy','sum_prob','avg_prob','entropy', 'loss', 'iit', 'iit_tpm_loss', 'iit_tpm_entropy', 'iit_entropy', 'correlation_entropy_tpm_entropy']
        df_summary_dataset = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary_dataset, group_cols, value_cols)
        df_summary_dataset = df_summary_dataset.sort_values(by=['settings'])        
        print(f'{confidence_type} Settings')
        print(df_summary_dataset.to_string(index=False))        


    @staticmethod
    def calculate_mutual_information_iit_reward(confidence_type: str) -> None:
        data_list = []
        dir, csv_paths = confidence_inference_analysis.get_filenames(confidence_type)
        for dataset, csv_dataset in csv_paths.items():
            file_paths = csv_dataset['file_paths']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            for file_path in file_paths: 
                for run_number in range(from_run_number, to_run_number):
                    try:
                        iit_type = confidence_inference_analysis.extract_iit_type(file_path)
                        if iit_type is None: continue

                        file_path_run_number = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path_run_number}')
                        required_cols = ["Accuracy", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss", "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]
                        confidence_inference_analysis.check_columns(df, required_cols)
                        
                        accuracy = df['Accuracy'].mean()
                        df = df[["Accuracy", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss" , "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]].dropna()

                        df['confidence_iit_reward'] = confidence_inference_analysis.convert_into_probability(df['Phi_Reward_Raw'])
                        x_list = df['confidence_iit_reward'].tolist()

                        df['confidence_tpm_entropy'] = confidence_inference_analysis.convert_into_probability(df['Tpm_Entropy'], is_inverse=True)
                        y_list = df['confidence_tpm_entropy'].tolist()

                        x = np.asarray(x_list, dtype=float)
                        y = np.asarray(y_list, dtype=float)

                        n_neighbors = 5
                        mi = mutual_info_regression(
                            x.reshape(-1, 1),
                            y,
                            discrete_features=False,
                            n_neighbors=n_neighbors,
                            random_state=42
                        )[0]

                        hx = confidence_inference_analysis.differential_entropy(x, n_neighbors)
                        hy = confidence_inference_analysis.differential_entropy(y, n_neighbors)

                        if hx <= 0 or hy <= 0:
                            nmi = np.nan
                        else:
                            nmi = mi / np.sqrt(hx * hy)

                       
                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "settings" : iit_type, 
                                        "accuracy": accuracy,
                                        "mutual_information": mi,
                                        "normalized_mutual_information": nmi,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")
        
        df_summary = pd.DataFrame(data_list)
        group_cols=['dataset', 'settings']        
        value_cols=['accuracy','mutual_information', 'normalized_mutual_information']
        df_summary = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['settings', 'dataset'])        
        print(f'{confidence_type} Settings')
        print(df_summary.to_string(index=False))        

        print()
        
        df_summary_dataset = pd.DataFrame(data_list)
        group_cols=['settings']        
        value_cols=['accuracy','mutual_information', 'normalized_mutual_information']
        df_summary_dataset = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary_dataset, group_cols, value_cols)
        df_summary_dataset = df_summary_dataset.sort_values(by=['settings'])        
        print(f'{confidence_type} Settings')
        print(df_summary_dataset.to_string(index=False))        

    @staticmethod
    def differential_entropy(x, k=5):
        x = np.asarray(x, dtype=float).reshape(-1, 1)

        n, d = x.shape

        nbrs = NearestNeighbors(
            n_neighbors=k + 1,
            metric="chebyshev"
        ).fit(x)

        distances, _ = nbrs.kneighbors(x)

        eps = distances[:, -1]

        volume_unit_ball = 2.0      # d=1

        H = (
            digamma(n)
            - digamma(k)
            + np.log(volume_unit_ball)
            + d * np.mean(np.log(eps + 1e-15))
        )

        return H

    @staticmethod
    def plot_auroc_entropy_iit_reward(confidence_type: str, from_run_number: int = 1, to_run_number: int = 1) -> None:
        data_list = []
        for run_number in range(from_run_number, to_run_number + 1):
            dir, csv_paths = confidence_inference_analysis.get_filenames(confidence_type)
            for dataset, csv_paths_dataset in csv_paths.items():
                fig, axes = plt.subplots(1, 3, figsize=(12, 6))
                axes = axes.flatten()

                plot_idx = 0
                for file_path in csv_paths_dataset: 
                    try:
                        iit_type = confidence_inference_analysis.extract_iit_type(file_path)
                        if iit_type is None: continue

                        file_path = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path}')
                        required_cols = ["Accuracy", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss", "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]
                        if 'Confidence_MultipleChoices' in df.columns:
                            required_cols.append('Confidence_MultipleChoices')
                        confidence_inference_analysis.check_columns(df, required_cols)
                        
                        accuracy = df['Accuracy'].mean()
                        if 'Confidence_MultipleChoices' in df.columns:
                            df = df[["Accuracy", "Confidence_MultipleChoices", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss" , "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]].dropna()
                        else:
                            df = df[["Accuracy", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss" , "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]].dropna()

                        df['Accuracy_Reward'] = df['Accuracy'].map({True: 1, False: 0})        
                        accuracy_list = df['Accuracy_Reward'].tolist()

                        min_sum_probabilty_val = df['Sequence_Probability'].min()
                        max_sum_probabilty_val = df['Sequence_Probability'].max()
                        df['confidence_sum_probabilty'] = df['Sequence_Probability'] / (max_sum_probabilty_val - min_sum_probabilty_val)
                        sum_probability_list = df['confidence_sum_probabilty'].tolist()
                        
                        norm_seq_prob_list = df['Length_Normalized_Sequence_Probability'].tolist()

                        min_entropy_val = df['Entropy'].min()
                        max_entropy_val = df['Entropy'].max()
                        df['confidence_entropy'] = (max_entropy_val - df['Entropy']) / (max_entropy_val - min_entropy_val)
                        confidence_entropy_list = df['confidence_entropy'].tolist()

                        min_loss_val = df['Completion_Loss'].min()
                        max_loss_val = df['Completion_Loss'].max()
                        df['confidence_loss'] = (max_entropy_val - df['Completion_Loss']) / (max_loss_val - min_loss_val)
                        confidence_loss_list = df['confidence_loss'].tolist()

                        min_iit_reward_val = df['Phi_Reward_Raw'].min()
                        max_iit_reward_val = df['Phi_Reward_Raw'].max()
                        df['confidence_iit_reward'] = (df['Phi_Reward_Raw']) / (max_iit_reward_val - min_iit_reward_val)
                        confidence_iit_reward_list = df['confidence_iit_reward'].tolist()

                        min_tpm_loss_val = df['Tpm_Loss'].min()
                        max_tpm_loss_val = df['Tpm_Loss'].max()
                        df['confidence_tpm_loss'] = (max_tpm_loss_val - df['Tpm_Loss']) / (max_tpm_loss_val - min_tpm_loss_val)

                        min_tpm_entropy_val = df['Tpm_Entropy'].min()
                        max_tpm_entropy_val = df['Tpm_Entropy'].max()
                        df['confidence_tpm_entropy'] = (max_tpm_entropy_val - df['Tpm_Entropy']) / (max_tpm_entropy_val - min_tpm_entropy_val)
                        
                        df['confidence_iit_reward_tpm_loss'] = (1 + df['confidence_iit_reward'] - df['confidence_tpm_loss']) / 2.0
                        iit_reward_tpm_loss_list = df['confidence_iit_reward_tpm_loss'].tolist()

                        df['confidence_iit_reward_tpm_entropy'] = (1 + df['confidence_iit_reward'] - df['confidence_tpm_entropy']) / 2.0
                        iit_reward_tpm_entropy_list = df['confidence_iit_reward_tpm_entropy'].tolist()

                        if 'Confidence_MultipleChoices' in df.columns:
                            confidence_list = df['Confidence_MultipleChoices'].tolist()
                        
                        y_true = np.array(accuracy_list)
                        accuracy = np.average(y_true)
                        con_A = np.array(sum_probability_list)
                        con_B = np.array(norm_seq_prob_list)
                        con_C = np.array(confidence_entropy_list)
                        con_D = np.array(confidence_loss_list)
                        con_E = np.array(confidence_iit_reward_list)
                        con_F = np.array(iit_reward_tpm_loss_list)
                        con_G = np.array(iit_reward_tpm_entropy_list)
                        if 'Confidence_MultipleChoices' in df.columns:
                            con_H = np.array(confidence_list)

                        fpr1, tpr1, _ = roc_curve(y_true, con_A)
                        roc_auc1 = auc(fpr1, tpr1)

                        fpr2, tpr2, _ = roc_curve(y_true, con_B)
                        roc_auc2 = auc(fpr2, tpr2)

                        fpr3, tpr3, _ = roc_curve(y_true, con_C)
                        roc_auc3 = auc(fpr3, tpr3)

                        fpr4, tpr4, _ = roc_curve(y_true, con_D)
                        roc_auc4 = auc(fpr4, tpr4)
                        
                        fpr5, tpr5, _ = roc_curve(y_true, con_E)
                        roc_auc5 = auc(fpr5, tpr5)
                        
                        fpr6, tpr6, _ = roc_curve(y_true, con_F)
                        roc_auc6 = auc(fpr6, tpr6)

                        fpr7, tpr7, _ = roc_curve(y_true, con_G)
                        roc_auc7 = auc(fpr7, tpr7)
                        
                        if 'Confidence_MultipleChoices' in df.columns:
                            fpr8, tpr8, _ = roc_curve(y_true, con_H)
                            roc_auc8 = auc(fpr8, tpr8)
                        else: 
                            roc_auc8 = 0

                        ax = axes[plot_idx]
                        plot_idx += 1
                        ax.plot(fpr1, tpr1, linewidth=2, label=f"Sum Prob({roc_auc1:.3f})")
                        ax.plot(fpr2, tpr2, linewidth=2, label=f"AVG_Prob({roc_auc2:.3f})")
                        ax.plot(fpr3, tpr3, linewidth=2, label=f"Entropy({roc_auc3:.3f})")
                        ax.plot(fpr4, tpr4, linewidth=2, label=f"Loss({roc_auc4:.3f})")
                        ax.plot(fpr5, tpr5, linewidth=2, label=f"IIT({roc_auc5:.3f})")
                        ax.plot(fpr6, tpr6, linewidth=2, label=f"IIT_Loss({roc_auc6:.3f})")
                        ax.plot(fpr7, tpr7, linewidth=2, label=f"IIT_Entropy({roc_auc7:.3f})")
                        if 'Confidence_MultipleChoices' in df.columns:
                            ax.plot(fpr8, tpr8, linewidth=2, label=f"Confidence MC({roc_auc8:.3f})")
                            
                        if 'Confidence_MultipleChoices' in df.columns:
                            ax.plot([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], linestyle="--", linewidth=1)
                        else:
                            ax.plot([0, 1], [0, 1], [0, 1], [0, 1], [0, 1],[0, 1], [0, 1], linestyle="--", linewidth=1)
                            
                        ax.set_title(f"{iit_type}, Accuracy= {accuracy:.2f}")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.legend()

                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "settings" : iit_type, 
                                        "accuracy": accuracy,
                                        "sum_prob": roc_auc1,
                                        "avg_prob": roc_auc2,
                                        "roc_entropy": roc_auc3,
                                        "roc_loss": roc_auc4,
                                        "roc_iit": roc_auc5,
                                        "roc_tpm_loss": roc_auc6,
                                        "roc_tpm_entropy": roc_auc7,
                                        "roc_multiplechoices": roc_auc8,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")

            
                plt.tight_layout()
                plt.plot()
                save_path = f'src/confidence/analysis/{dataset}/run_{run_number}'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{save_path}/{confidence_type}_auroc.png')
                plt.close(fig) 
        
        df_summary = pd.DataFrame(data_list)
        group_cols=['dataset', 'settings']        
        value_cols=['accuracy','sum_prob','avg_prob','roc_entropy', 'roc_multiplechoices', 'roc_loss', 'roc_iit', 'roc_tpm_loss', 'roc_tpm_entropy']
        df_summary = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['settings', 'dataset'])        
        print(f'{confidence_type} Settings')
        print(df_summary.to_string(index=False))        


    @staticmethod
    def scatterplot_entropy_iit_reward(confidence_type: str, run_number: int = 1) -> None:
        dir, csv_paths = confidence_inference_analysis.get_filenames(confidence_type)
        for dataset, csv_paths_dataset in csv_paths.items():
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            axes = axes.flatten()

            plot_idx = 0
            for file_path in csv_paths_dataset: 
                try:
                    iit_type = confidence_inference_analysis.extract_iit_type(file_path)
                    if iit_type is None: continue

                    file_path = file_path.replace('run_', f'run_{run_number}')
                    df = pd.read_csv(f'{dir}/{file_path}')
                    settings = confidence_inference_analysis.extract_first_number(file_path)                    

                    required_cols = ["Entropy", "Phi_Reward_Raw"]
                    if iit_type == 'ii':
                        required_cols = ["Entropy", "Phi_Reward_Raw_Actual"]
                    confidence_inference_analysis.check_columns(df, required_cols)
                    
                    if iit_type == 'ii':
                        df = df[["Phi_Reward_Raw_Actual", "Entropy"]].dropna()
                    else:
                        df = df[["Phi_Reward_Raw", "Entropy"]].dropna()
                    # Compute correlation
                    if iit_type == 'ii':
                        df_iit_reward = df["Phi_Reward_Raw_Actual"]
                    else:
                        df_iit_reward = df["Phi_Reward_Raw"]
                        
                    correlation = df_iit_reward.corr(df["Entropy"])

                    ax = axes[plot_idx]
                    plot_idx += 1

                    ax.scatter(df["Entropy"], df_iit_reward, alpha=0.7, edgecolors='k', color=f'C{plot_idx}')
                    ax.set_title(f'{settings}_{iit_type}, PC=({correlation:.3f})')
                    ax.set_xlabel("Entropy")
                    ax.set_ylabel("IIT Reward")
                    ax.grid(True, linestyle='--', alpha=0.6)
                except Exception as e:
                    print(f"[WARN] {e}")
        
            plt.tight_layout()
            plt.plot()
            save_path = f'src/confidence/iit/analysis/{dataset}/run_{run_number}'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{save_path}/{confidence_type}_scatterplot.png')
            plt.close(fig) 


    @staticmethod
    def check_columns(df, required_cols):
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV")
            
        return None    

    @staticmethod
    def get_filenames(confidence_type: str) -> None:
        dir = './src/confidence'
        csv_paths = {
            "truthfulqa": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/truthfulqa/{confidence_type}/run_/confidence_{confidence_type}_truthfulqa_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/truthfulqa/{confidence_type}/run_/confidence_{confidence_type}_truthfulqa_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/truthfulqa/{confidence_type}/run_/confidence_{confidence_type}_truthfulqa_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "mmlu": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/mmlu/{confidence_type}/run_/confidence_{confidence_type}_mmlu_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/mmlu/{confidence_type}/run_/confidence_{confidence_type}_mmlu_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/mmlu/{confidence_type}/run_/confidence_{confidence_type}_mmlu_Settings_65.csv",
                            ],
                            "from_run_number": 1,
                            "to_run_number": 6,
                        },
            "mmlu_pro": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/mmlu_pro/{confidence_type}/run_/confidence_{confidence_type}_mmlu_pro_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/mmlu_pro/{confidence_type}/run_/confidence_{confidence_type}_mmlu_pro_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/mmlu_pro/{confidence_type}/run_/confidence_{confidence_type}_mmlu_pro_Settings_65.csv",
                            ],
                            "from_run_number": 1,
                            "to_run_number": 6,
                        },
            "aime": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/aime/{confidence_type}/run_/confidence_{confidence_type}_aime_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/aime/{confidence_type}/run_/confidence_{confidence_type}_aime_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/aime/{confidence_type}/run_/confidence_{confidence_type}_aime_Settings_65.csv",
                            ],
                            "from_run_number": 1,
                            "to_run_number": 6,
                        },
            "countdown": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/countdown/{confidence_type}/run_/confidence_{confidence_type}_countdown_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/countdown/{confidence_type}/run_/confidence_{confidence_type}_countdown_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/countdown/{confidence_type}/run_/confidence_{confidence_type}_countdown_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "gsm8k": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/gsm8k/{confidence_type}/run_/confidence_{confidence_type}_gsm8k_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/gsm8k/{confidence_type}/run_/confidence_{confidence_type}_gsm8k_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/gsm8k/{confidence_type}/run_/confidence_{confidence_type}_gsm8k_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "gpqa": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/gpqa/{confidence_type}/run_/confidence_{confidence_type}_gpqa_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/gpqa/{confidence_type}/run_/confidence_{confidence_type}_gpqa_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/gpqa/{confidence_type}/run_/confidence_{confidence_type}_gpqa_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "math500": {
                            "file_paths" : [
                                    f"iit/deepseek_r1_distill_qwen_7b/math500/{confidence_type}/run_/confidence_{confidence_type}_math500_Settings_46.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/math500/{confidence_type}/run_/confidence_{confidence_type}_math500_Settings_64.csv", 
                                    f"iit/deepseek_r1_distill_qwen_7b/math500/{confidence_type}/run_/confidence_{confidence_type}_math500_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },

        }
        
        return dir, csv_paths

    @staticmethod
    def extract_iit_type(filename):
        match = re.search(r'Settings_(\d+)\.csv', filename)

        if not  match:
            return None
        
        return int(match.group(1))

    @staticmethod
    def convert_into_probability(x, is_inverse = False):
        min = x.min()
        max = x.max()
        if not is_inverse:
            x = (x - min) / (max - min)
        else:
            x = (max - x) / (max - min)

        return x

    @staticmethod
    def aggregate_mean_pandas_rounded(df, group_cols, value_cols) -> pd.DataFrame:
        result = df.groupby(group_cols)[value_cols].mean().reset_index()
        for col in value_cols:
            result[col] = result[col].round(3)
        return result


confidence_inference_analysis.calculate_mutual_information_iit_reward('whitebox')
print()
confidence_inference_analysis.calculate_mutual_information_iit_reward('blackbox')
