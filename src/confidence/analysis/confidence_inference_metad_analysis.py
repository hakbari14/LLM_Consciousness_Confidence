from scipy.stats import norm
import pandas as pd
import numpy as np 
import re 
from tqdm import tqdm
import json

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

                        df['confidence_sum_probabilty'] = confidence_inference_metad_analysis.convert_into_probability(df['Sequence_Probability'])
                        sum_probability_list = df['confidence_sum_probabilty'].tolist()
                        norm_seq_prob_list = df['Length_Normalized_Sequence_Probability'].tolist()

                        df['confidence_entropy'] = confidence_inference_metad_analysis.convert_into_probability(df['Entropy'], is_inverse=True)
                        confidence_entropy_list = df['confidence_entropy'].tolist()

                        df['confidence_loss'] = confidence_inference_metad_analysis.convert_into_probability(df['Completion_Loss'], is_inverse=True)
                        confidence_loss_list = df['confidence_loss'].tolist()

                        df['confidence_iit_reward'] = confidence_inference_metad_analysis.convert_into_probability(df['Phi_Reward_Raw'])
                        confidence_iit_reward_list = df['confidence_iit_reward'].tolist()

                        df['confidence_tpm_loss'] = confidence_inference_metad_analysis.convert_into_probability(df['Tpm_Loss'], is_inverse=True)
                        df['confidence_tpm_entropy'] = confidence_inference_metad_analysis.convert_into_probability(df['Tpm_Entropy'], is_inverse=True)
                        
                        df['confidence_iit_reward_tpm_loss'] = (1 + df['confidence_iit_reward'] - df['confidence_tpm_loss']) / 2.0
                        iit_reward_tpm_loss_list = df['confidence_iit_reward_tpm_loss'].tolist()

                        df['confidence_iit_reward_tpm_entropy'] = (1 + df['confidence_iit_reward'] - df['confidence_tpm_entropy']) / 2.0
                        iit_reward_tpm_entropy_list = df['confidence_iit_reward_tpm_entropy'].tolist()

                        if 'Confidence_MultipleChoices' in df.columns:
                            confidence_list = df['Confidence_MultipleChoices'].tolist()
                        
                        accuracy = np.average(accuracy_list)
                        
                        meta_d_prime_sum_prob, _, _ = confidence_inference_metad_analysis.compute_meta_d_ratio(accuracy_list, sum_probability_list)
                        meta_d_prime_avg_prob, _, _ = confidence_inference_metad_analysis.compute_meta_d_ratio(accuracy_list, norm_seq_prob_list)
                        meta_d_prime_entropy, _, _ = confidence_inference_metad_analysis.compute_meta_d_ratio(accuracy_list, confidence_entropy_list)
                        meta_d_prime_loss, _, _ = confidence_inference_metad_analysis.compute_meta_d_ratio(accuracy_list, confidence_loss_list)
                        meta_d_prime_iit, _, _ = confidence_inference_metad_analysis.compute_meta_d_ratio(accuracy_list, confidence_iit_reward_list)
                        meta_d_prime_tpm_loss, _, _ = confidence_inference_metad_analysis.compute_meta_d_ratio(accuracy_list, iit_reward_tpm_loss_list)
                        meta_d_prime_tpm_entropy, _, _ = confidence_inference_metad_analysis.compute_meta_d_ratio(accuracy_list, iit_reward_tpm_entropy_list)
                       

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

        # print(json.dumps(data_list, indent=4, ensure_ascii=False))        

        # df_summary = pd.DataFrame(data_list)
        # group_cols=['dataset', 'model', 'settings']        
        # value_cols=['accuracy','sum_prob','avg_prob','roc_entropy', 'roc_multiplechoices', 'roc_loss', 'roc_iit', 'roc_tpm_loss', 'roc_tpm_entropy']
        # df_summary = confidence_inference_metad_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        # df_summary = df_summary.sort_values(by=['settings', 'dataset', 'model'])        
        # print(f'{confidence_type} Settings')
        # print(df_summary.to_string(index=False))        

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
    def convert_into_probability(x, is_inverse = False):
        min = x.min()
        max = x.max()
        if not is_inverse:
            x = x / (max - min)
        else:
            x = (max - x) / (max - min)

        return x

    @staticmethod
    def make_rating_bins(confidence, n_bins):
        edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            raise ValueError("Not enough variability in confidence for binning")

        ratings = np.digitize(confidence, edges[1:-1]) + 1
        return ratings, edges

    def compute_meta_d_ratio(accuracy_list: list[int], confidence_list: list[float]) -> float:
        hit_count: int = 0
        false_alaram_count: int = 0
        hit_conf_count: int = 0
        false_alaram_conf_count: int = 0
        total_conf: int = 0
       
        threashold = np.median(confidence_list)
        for accuracy, confidence in zip(accuracy_list, confidence_list):
            if accuracy == 1:
                hit_count += 1
            if accuracy == 0:
                false_alaram_count += 1

            if confidence < threashold: continue
            total_conf += 1
            if accuracy == 1:
                hit_conf_count += 1
            if accuracy == 0:
                false_alaram_conf_count += 1
        
        hit_prob:float = hit_count / len(accuracy_list)
        false_alaram_prob:float = false_alaram_count / len(accuracy_list)
        d_prime = confidence_inference_metad_analysis.compute_d_prime(hit_prob, false_alaram_prob)

        hit_conf_prob:float = hit_conf_count / total_conf
        false_alaram_conf_prob:float = false_alaram_conf_count / total_conf
        metad_prime = confidence_inference_metad_analysis.compute_d_prime(hit_conf_prob, false_alaram_conf_prob)
        
        m_ratio: float = None
        if metad_prime >= 0 and d_prime > 0:
            m_ratio = metad_prime / d_prime
        
        return m_ratio, metad_prime, d_prime

    def compute_d_prime(hit_rate: float, false_alarm_rate: float) -> float:
        eps = 1e-10
        hit_rate = min(max(hit_rate, eps), 1 - eps)
        false_alarm_rate = min(max(false_alarm_rate, eps), 1 - eps)

        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(false_alarm_rate)

        d_prime = z_hit - z_fa
        return d_prime


confidence_inference_metad_analysis.calculate_metad_entropy_iit_reward('whitebox')
print()
confidence_inference_metad_analysis.calculate_metad_entropy_iit_reward('blackbox')
