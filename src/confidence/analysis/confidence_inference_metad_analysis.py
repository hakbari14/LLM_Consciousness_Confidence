from scipy.stats import norm
import pandas as pd
import numpy as np 
import re 
from metadpy.utils import trials2counts
from metadpy.mle import metad
from tqdm import tqdm


class confidence_inference_metad_analysis(object):

    @staticmethod
    def calculate_metad_entropy_iit_reward(confidence_type: str) -> None:
        data_list = []
        dir, csv_paths = confidence_inference_metad_analysis.get_filenames(confidence_type)
        for dataset, csv_dataset in tqdm(csv_paths.items(), desc="Metad Prime Ratio", unit="step"): 
            file_paths = csv_dataset['file_paths']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            for file_path in file_paths: 
                for run_number in range(from_run_number, to_run_number):
                    try:
                        iit_type = confidence_inference_metad_analysis.extract_iit_type(file_path)
                        if iit_type is None: continue

                        base_model = confidence_inference_metad_analysis.extract_base_model(file_path)                    
                        if base_model is None: continue
                        
                        file_path_run_number = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path_run_number}')
                        required_cols = ["Accuracy", "Target", "Final_Answer", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss", "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]
                        if 'Confidence_MultipleChoices' in df.columns:
                            required_cols.append('Confidence_MultipleChoices')
                        confidence_inference_metad_analysis.check_columns(df, required_cols)
                        
                        accuracy = df['Accuracy'].mean()
                        if 'Confidence_MultipleChoices' in df.columns:
                            df = df[["Accuracy", "Target", "Final_Answer", "Confidence_MultipleChoices", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss" , "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]].dropna()
                        else:
                            df = df[["Accuracy", "Target", "Final_Answer", "Sequence_Probability", "Length_Normalized_Sequence_Probability", "Entropy", "Completion_Loss" , "Phi_Reward_Raw", "Tpm_Loss", "Tpm_Entropy"]].dropna()

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
                        
                        accuracy = np.average(accuracy_list)
                        
                        meta_d_prime_sum_prob = confidence_inference_metad_analysis.compute_meta_d_prime_ratio(accuracy_list, sum_probability_list)
                        meta_d_prime_avg_prob = confidence_inference_metad_analysis.compute_meta_d_prime_ratio(accuracy_list, norm_seq_prob_list)
                        meta_d_prime_entropy = confidence_inference_metad_analysis.compute_meta_d_prime_ratio(accuracy_list, confidence_entropy_list)
                        meta_d_prime_loss = confidence_inference_metad_analysis.compute_meta_d_prime_ratio(accuracy_list, confidence_loss_list)
                        meta_d_prime_iit = confidence_inference_metad_analysis.compute_meta_d_prime_ratio(accuracy_list, confidence_iit_reward_list)
                        meta_d_prime_tpm_loss = confidence_inference_metad_analysis.compute_meta_d_prime_ratio(accuracy_list, iit_reward_tpm_loss_list)
                        meta_d_prime_tpm_entropy = confidence_inference_metad_analysis.compute_meta_d_prime_ratio(accuracy_list, iit_reward_tpm_entropy_list)
                       

                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "model" : base_model, 
                                        "settings" : iit_type, 
                                        "accuracy": accuracy,
                                        "sum_prob": meta_d_prime_sum_prob,
                                        "avg_prob": meta_d_prime_avg_prob,
                                        "roc_entropy": meta_d_prime_entropy,
                                        "roc_loss": meta_d_prime_loss,
                                        "roc_iit": meta_d_prime_iit,
                                        "roc_tpm_loss": meta_d_prime_tpm_loss,
                                        "roc_tpm_entropy": meta_d_prime_tpm_entropy,
                                        "roc_multiplechoices": 0,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")
        
        df_summary = pd.DataFrame(data_list)
        group_cols=['dataset', 'model', 'settings']        
        value_cols=['accuracy','sum_prob','avg_prob','roc_entropy', 'roc_multiplechoices', 'roc_loss', 'roc_iit', 'roc_tpm_loss', 'roc_tpm_entropy']
        df_summary = confidence_inference_metad_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['settings', 'dataset', 'model'])        
        print(f'{confidence_type} Settings')
        print(df_summary.to_string(index=False))        

        print()
        
        df_summary_dataset = pd.DataFrame(data_list)
        group_cols=['settings', 'model']        
        value_cols=['accuracy','sum_prob','avg_prob','roc_entropy', 'roc_multiplechoices', 'roc_loss', 'roc_iit', 'roc_tpm_loss', 'roc_tpm_entropy']
        df_summary_dataset = confidence_inference_metad_analysis.aggregate_mean_pandas_rounded(df_summary_dataset, group_cols, value_cols)
        df_summary_dataset = df_summary_dataset.sort_values(by=['settings', 'model'])        
        print(f'{confidence_type} Settings')
        print(df_summary_dataset.to_string(index=False))        


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
                                    f"settings_0/truthfulqa/{confidence_type}/run_/confidence_{confidence_type}_truthfulqa_Settings_46.csv", 
                                    f"settings_0/truthfulqa/{confidence_type}/run_/confidence_{confidence_type}_truthfulqa_Settings_64.csv", 
                                    f"settings_0/truthfulqa/{confidence_type}/run_/confidence_{confidence_type}_truthfulqa_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "mmlu": {
                            "file_paths" : [
                                    f"settings_0/mmlu/{confidence_type}/run_/confidence_{confidence_type}_mmlu_Settings_46.csv", 
                                    f"settings_0/mmlu/{confidence_type}/run_/confidence_{confidence_type}_mmlu_Settings_64.csv", 
                                    f"settings_0/mmlu/{confidence_type}/run_/confidence_{confidence_type}_mmlu_Settings_65.csv",
                            ],
                            "from_run_number": 1,
                            "to_run_number": 6,
                        },
            "mmlu_pro": {
                            "file_paths" : [
                                    f"settings_0/mmlu_pro/{confidence_type}/run_/confidence_{confidence_type}_mmlu_pro_Settings_46.csv", 
                                    f"settings_0/mmlu_pro/{confidence_type}/run_/confidence_{confidence_type}_mmlu_pro_Settings_64.csv", 
                                    f"settings_0/mmlu_pro/{confidence_type}/run_/confidence_{confidence_type}_mmlu_pro_Settings_65.csv",
                            ],
                            "from_run_number": 1,
                            "to_run_number": 6,
                        },
            "aime": {
                            "file_paths" : [
                                    f"settings_0/aime/{confidence_type}/run_/confidence_{confidence_type}_aime_Settings_46.csv", 
                                    f"settings_0/aime/{confidence_type}/run_/confidence_{confidence_type}_aime_Settings_64.csv", 
                                    f"settings_0/aime/{confidence_type}/run_/confidence_{confidence_type}_aime_Settings_65.csv",
                            ],
                            "from_run_number": 1,
                            "to_run_number": 6,
                        },
            "countdown": {
                            "file_paths" : [
                                    f"settings_0/countdown/{confidence_type}/run_/confidence_{confidence_type}_countdown_Settings_46.csv", 
                                    f"settings_0/countdown/{confidence_type}/run_/confidence_{confidence_type}_countdown_Settings_64.csv", 
                                    f"settings_0/countdown/{confidence_type}/run_/confidence_{confidence_type}_countdown_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "gsm8k": {
                            "file_paths" : [
                                    f"settings_0/gsm8k/{confidence_type}/run_/confidence_{confidence_type}_gsm8k_Settings_46.csv", 
                                    f"settings_0/gsm8k/{confidence_type}/run_/confidence_{confidence_type}_gsm8k_Settings_64.csv", 
                                    f"settings_0/gsm8k/{confidence_type}/run_/confidence_{confidence_type}_gsm8k_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "gpqa": {
                            "file_paths" : [
                                    f"settings_0/gpqa/{confidence_type}/run_/confidence_{confidence_type}_gpqa_Settings_46.csv", 
                                    f"settings_0/gpqa/{confidence_type}/run_/confidence_{confidence_type}_gpqa_Settings_64.csv", 
                                    f"settings_0/gpqa/{confidence_type}/run_/confidence_{confidence_type}_gpqa_Settings_65.csv",
                            ],
                            "from_run_number": 6,
                            "to_run_number": 11,
                        },
            "math500": {
                            "file_paths" : [
                                    f"settings_0/math500/{confidence_type}/run_/confidence_{confidence_type}_math500_Settings_46.csv", 
                                    f"settings_0/math500/{confidence_type}/run_/confidence_{confidence_type}_math500_Settings_64.csv", 
                                    f"settings_0/math500/{confidence_type}/run_/confidence_{confidence_type}_math500_Settings_65.csv",
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
    def extract_base_model(filename):
        match = re.search(r'settings_(\d+)/', filename)

        if not  match:
            return None
        
        return int(match.group(1))

    @staticmethod
    def aggregate_mean_pandas_rounded(df, group_cols, value_cols) -> pd.DataFrame:
        result = df.groupby(group_cols)[value_cols].mean().reset_index()
        for col in value_cols:
            result[col] = result[col].round(3)
        return result

    @staticmethod
    def make_rating_bins(confidence, n_bins):
        edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            raise ValueError("Not enough variability in confidence for binning")

        ratings = np.digitize(confidence, edges[1:-1]) + 1
        return ratings, edges

    @staticmethod
    def compute_meta_d_prime_ratio(accuracy_list: list[int], confidence_list: list[float]) -> float:
        n_bins = 20
        ratings, _ = confidence_inference_metad_analysis.make_rating_bins(confidence_list, n_bins)

        response_list = accuracy_list.copy()
        nR_S1, nR_S2 = trials2counts(
            stimuli=accuracy_list,  
            responses=response_list,
            confidence=ratings,
            nRatings=len(np.unique(ratings))
        )

        # =========================
        # 3. Fit meta-d'
        # =========================
        fit = metad(
            nR_S1=nR_S1,
            nR_S2=nR_S2
        )

        return fit["m_ratio"].values[0]


confidence_inference_metad_analysis.calculate_metad_entropy_iit_reward('whitebox')
print()
confidence_inference_metad_analysis.calculate_metad_entropy_iit_reward('blackbox')
