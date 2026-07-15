import pandas as pd
import re 

class confidence_inference_ECE_analysis(object):

    @staticmethod
    def calculate(confidence_type: str, n_bins: int = 10) -> None:
        data_list = []
        dir, csv_paths = confidence_inference_ECE_analysis.get_filenames(confidence_type)
        for dataset, csv_dataset in csv_paths.items():
            file_paths = csv_dataset['file_paths']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            for file_path in file_paths: 
                for run_number in range(from_run_number, to_run_number):
                    try:
                        file_path_run_number = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path_run_number}')
                        accuracy = df['Accuracy'].mean()
                        
                        df['sum_probabilty_confidence'] = confidence_inference_ECE_analysis.convert_into_probability(df['Sequence_Probability'])
                        ece_sum_prob, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'sum_probabilty_confidence', n_bins)
                        
                        ece_avg_prob, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'Length_Normalized_Sequence_Probability', n_bins)

                        df['entropy_confidence'] = confidence_inference_ECE_analysis.convert_into_probability(df['Entropy'], is_inverse=True)
                        ece_entropy, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'entropy_confidence', n_bins)

                        df['iit_reward_confidence'] = confidence_inference_ECE_analysis.convert_into_probability(df['Phi_Reward_Raw'])
                        ece_iit_reward, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'iit_reward_confidence', n_bins)
                        
                        df['confidence_tpm_loss'] = confidence_inference_ECE_analysis.convert_into_probability(df['Tpm_Loss'], is_inverse=True)
                        df['confidence_tpm_entropy'] = confidence_inference_ECE_analysis.convert_into_probability(df['Tpm_Entropy'], is_inverse=True)
                        
                        df['confidence_iit_reward_tpm_loss'] = (1 + df['iit_reward_confidence'] - df['confidence_tpm_loss']) / 2.0
                        ece_iit_reward_loss, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'confidence_iit_reward_tpm_loss', n_bins)
                        df['confidence_iit_reward_tpm_entropy'] = (1 + df['iit_reward_confidence'] - df['confidence_tpm_entropy']) / 2.0
                        ece_iit_reward_entropy, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'confidence_iit_reward_tpm_entropy', n_bins)
                        
                        base_model = confidence_inference_ECE_analysis.extract_base_model(file_path)
                        iit_type = confidence_inference_ECE_analysis.extract_iit_type(file_path)
                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "model" : base_model, 
                                        "settings" : iit_type, 
                                        "accuracy": accuracy,
                                        "ece_sum_prob": ece_sum_prob,
                                        "ece_avg_prob": ece_avg_prob,
                                        "ece_entropy": ece_entropy,
                                        "ece_iit_reward": ece_iit_reward,
                                        "ece_iit_reward_loss": ece_iit_reward_loss,
                                        "ece_iit_reward_entropy": ece_iit_reward_entropy,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")

        df_summary = pd.DataFrame(data_list)
        group_cols=['dataset', 'model', 'settings']        
        value_cols=['accuracy', 'ece_sum_prob', 'ece_avg_prob', 'ece_entropy', 'ece_iit_reward', 'ece_iit_reward_loss', 'ece_iit_reward_entropy']
        df_summary = confidence_inference_ECE_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['settings', 'dataset', 'model'])        
        print(f'{confidence_type} Settings')
        print(df_summary.to_string(index=False))        

        print()
        
        df_summary_dataset = pd.DataFrame(data_list)
        group_cols=['settings', 'model']        
        value_cols=['accuracy', 'ece_sum_prob', 'ece_avg_prob', 'ece_entropy', 'ece_iit_reward', 'ece_iit_reward_loss', 'ece_iit_reward_entropy']
        df_summary_dataset = confidence_inference_ECE_analysis.aggregate_mean_pandas_rounded(df_summary_dataset, group_cols, value_cols)
        df_summary_dataset = df_summary_dataset.sort_values(by=['settings', 'model'])        
        print(f'{confidence_type} Settings')
        print(df_summary_dataset.to_string(index=False))        


    @staticmethod
    def calculate_ECE_MCE(df, confidence_column_name ,n_bins = 10):
        df['accuracy_reward'] = df['Accuracy'].map({True: 1, False: 0})        
        df['binned_confidence'] = pd.qcut(df[confidence_column_name], q=n_bins, duplicates='drop')
        agg_perplexity = df.groupby('binned_confidence', observed=False)[confidence_column_name].agg(['mean'])
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
    def convert_into_probability(x, is_inverse = False):
        min = x.min()
        max = x.max()
        if not is_inverse:
            x = x / (max - min)
        else:
            x = (max - x) / (max - min)

        return x

    @staticmethod
    def aggregate_mean_pandas_rounded(df, group_cols, value_cols) -> pd.DataFrame:
        result = df.groupby(group_cols)[value_cols].mean().reset_index()
        for col in value_cols:
            result[col] = result[col].round(3)
        return result

confidence_inference_ECE_analysis.calculate('whitebox')
print()
confidence_inference_ECE_analysis.calculate('blackbox')