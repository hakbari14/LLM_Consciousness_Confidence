import pandas as pd
import re 

class confidence_inference_ECE_analysis(object):

    @staticmethod
    def calculate(to_run_number = 1, n_bins = 10):
        dir, csv_paths = confidence_inference_ECE_analysis.get_filenames()
        data_list = []
        for run_number in range(1, to_run_number + 1):
            for csv_path in csv_paths: 
                csv_path = csv_path.replace('run_', f'run_{run_number}')
                try:
                    df = pd.read_csv(f'{dir}/{csv_path}')
                    accuracy = df['Accuracy'].mean()
                    
                    min_entropy_val = df['Entropy'].min()
                    max_entropy_val = df['Entropy'].max()
                    df['entropy_confidence'] = (max_entropy_val - df['Entropy']) / (max_entropy_val - min_entropy_val)
                    ece_entropy, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'entropy_confidence', n_bins)

                    min_iit_reward_val = df['Phi_Reward_Raw'].min()
                    max_iit_reward_val = df['Phi_Reward_Raw'].max()
                    df['iit_reward_confidence'] = df['Phi_Reward_Raw'] / (max_iit_reward_val - min_iit_reward_val)
                    ece_iit_reward, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'iit_reward_confidence', n_bins)
                    
                    min_tpm_loss_val = df['Tpm_Loss'].min()
                    max_tpm_loss_val = df['Tpm_Loss'].max()
                    df['confidence_tpm_loss'] = (max_tpm_loss_val - df['Tpm_Loss']) / (max_tpm_loss_val - min_tpm_loss_val)

                    min_tpm_entropy_val = df['Tpm_Entropy'].min()
                    max_tpm_entropy_val = df['Tpm_Entropy'].max()
                    df['confidence_tpm_entropy'] = (max_tpm_entropy_val - df['Tpm_Entropy']) / (max_tpm_entropy_val - min_tpm_entropy_val)
                    
                    df['confidence_iit_reward_tpm_loss'] = (1 + df['iit_reward_confidence'] - df['confidence_tpm_loss']) / 2.0
                    ece_iit_reward_loss, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'confidence_iit_reward_tpm_loss', n_bins)
                    df['confidence_iit_reward_tpm_entropy'] = (1 + df['iit_reward_confidence'] - df['confidence_tpm_entropy']) / 2.0
                    ece_iit_reward_entropy, _ = confidence_inference_ECE_analysis.calculate_ECE_MCE(df, 'confidence_iit_reward_tpm_entropy', n_bins)
                    
                    base_model = confidence_inference_ECE_analysis.extract_base_model(csv_path)
                    iit_type = confidence_inference_ECE_analysis.extract_iit_type(csv_path)
                    benchmark = confidence_inference_ECE_analysis.extract_benchmark(csv_path)
                    data_item = {
                                    "run_number": run_number, 
                                    "dataset": benchmark , 
                                    "model" : base_model, 
                                    "settings" : iit_type, 
                                    "accuracy": accuracy,
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
        value_cols=['accuracy', 'ece_entropy', 'ece_iit_reward', 'ece_iit_reward_loss', 'ece_iit_reward_entropy']
        df_summary = confidence_inference_ECE_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['settings', 'dataset', 'model'])        
        print(df_summary.to_string(index=False))        


    @staticmethod
    def calculate_ECE_MCE(df, confidence_column_name ,n_bins = 10):
        df['accuracy_reward'] = df['Accuracy'].map({True: 1, False: 0})        
        df['binned_confidence'] = pd.qcut(df[confidence_column_name], q=n_bins)
        agg_perplexity = df.groupby('binned_confidence')[confidence_column_name].agg(['mean'])
        agg_accuracy = df.groupby('binned_confidence')['accuracy_reward'].agg(['mean'])

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
    def get_filenames():
        dir = './integrated_information_theory/inference/confidence/'
        csv_paths = [

                         "settings_0/aime/run_/confidence_aime_Settings_46.csv", 
                         "settings_0/aime/run_/confidence_aime_Settings_64.csv", 
                         "settings_0/aime/run_/confidence_aime_Settings_65.csv",

                         "settings_0/countdown/run_/confidence_countdown_Settings_46.csv", 
                         "settings_0/countdown/run_/confidence_countdown_Settings_64.csv", 
                         "settings_0/countdown/run_/confidence_countdown_Settings_65.csv",

                         "settings_0/gsm8k/run_/confidence_gsm8k_Settings_46.csv", 
                         "settings_0/gsm8k/run_/confidence_gsm8k_Settings_64.csv", 
                         "settings_0/gsm8k/run_/confidence_gsm8k_Settings_65.csv",

                         "settings_0/gpqa/run_/confidence_gpqa_Settings_46.csv", 
                         "settings_0/gpqa/run_/confidence_gpqa_Settings_64.csv", 
                         "settings_0/gpqa/run_/confidence_gpqa_Settings_65.csv",

                         "settings_0/math500/run_/confidence_math500_Settings_46.csv", 
                         "settings_0/math500/run_/confidence_math500_Settings_64.csv", 
                         "settings_0/math500/run_/confidence_math500_Settings_65.csv",

                    ]

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
    def extract_benchmark(filename):
        match = re.search(r'settings_\d+/([^/]+)/run_\d+/', filename)

        if not  match:
            return None
        
        return match.group(1)

    @staticmethod
    def aggregate_mean_pandas_rounded(df, group_cols, value_cols) -> pd.DataFrame:
        result = df.groupby(group_cols)[value_cols].mean().reset_index()
        for col in value_cols:
            result[col] = result[col].round(3)
        return result

confidence_inference_ECE_analysis.calculate(to_run_number=5)