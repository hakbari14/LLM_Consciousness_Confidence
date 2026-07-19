import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import roc_curve, auc
from metadpy.mle import metad

class confidence_inference_analysis(object):

    @staticmethod
    def calculate_auroc() -> None:
        data_list = []
        dir, csv_paths = confidence_inference_analysis.get_filenames()
        for dataset, csv_dataset in csv_paths.items():
            file_paths = csv_dataset['file_paths']
            confidence_type = csv_dataset['confidence_type']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            for file_path in file_paths: 
                for run_number in range(from_run_number, to_run_number):
                    try:
                        file_path_run_number = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path_run_number}')

                        required_cols = ["Accuracy", "Confidence_Level", "Confidence_Level_With_Solution", "Confidence_Level_Self_Criteria", "Confidence_Level_Self_Criteria_With_Solution"]
                        confidence_inference_analysis.check_columns(df, required_cols)
                        df = df.dropna(subset=["Accuracy", "Confidence_Level", "Confidence_Level_With_Solution" ,"Confidence_Level_Self_Criteria", "Confidence_Level_Self_Criteria_With_Solution"])                        

                        accuracy = df['Accuracy'].mean()
                        df['Accuracy_Reward'] = df['Accuracy'].map({True: 1, False: 0})        
                        accuracy_list = df['Accuracy_Reward'].tolist()

                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level', 'Confidence')
                        confidence_level_list = df['Confidence'].tolist()

                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level_With_Solution', 'Confidence_With_Solution')
                        confidence_level_with_solution_list = df['Confidence_With_Solution'].tolist()

                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level_Self_Criteria', 'Confidence_Self_Criteria')
                        confidence_level_self_criteria_list = df['Confidence_Self_Criteria'].tolist()

                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level_Self_Criteria_With_Solution', 'Confidence_Self_Criteria_With_Solution')
                        confidence_level_self_criteria_with_solution_list = df['Confidence_Self_Criteria_With_Solution'].tolist()

                        y_true = np.array(accuracy_list)
                        accuracy = np.average(y_true)

                        con_A = np.array(confidence_level_list)
                        con_B = np.array(confidence_level_with_solution_list)
                        con_C = np.array(confidence_level_self_criteria_list)
                        con_D = np.array(confidence_level_self_criteria_with_solution_list)

                        fpr1, tpr1, _ = roc_curve(y_true, con_A)
                        roc_auc1 = auc(fpr1, tpr1)

                        fpr2, tpr2, _ = roc_curve(y_true, con_B)
                        roc_auc2 = auc(fpr2, tpr2)

                        fpr3, tpr3, _ = roc_curve(y_true, con_C)
                        roc_auc3 = auc(fpr3, tpr3)

                        fpr4, tpr4, _ = roc_curve(y_true, con_D)
                        roc_auc4 = auc(fpr4, tpr4)

                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "confidence_type": confidence_type , 
                                        "accuracy": accuracy,
                                        "auc": roc_auc1,
                                        "auc_with_solution": roc_auc2,
                                        "auc_self_criteria": roc_auc3,
                                        "auc_self_criteria_with_solution": roc_auc4,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")
        
        df_summary = pd.DataFrame(data_list)
        group_cols=['confidence_type', 'dataset']        
        value_cols=['accuracy','auc', 'auc_with_solution', 'auc_self_criteria', 'auc_self_criteria_with_solution']
        df_summary = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['confidence_type', 'dataset'])        
        print(df_summary.to_string(index=False))        
        
    @staticmethod
    def calculate_ece(n_bins: int = 10) -> None:
        data_list = []
        dir, csv_paths = confidence_inference_analysis.get_filenames()
        for dataset, csv_dataset in csv_paths.items():
            file_paths = csv_dataset['file_paths']
            confidence_type = csv_dataset['confidence_type']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            for file_path in file_paths: 
                for run_number in range(from_run_number, to_run_number):
                    try:
                        file_path_run_number = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path_run_number}')

                        df = df[["Accuracy", "Confidence_Level", "Confidence_Level_With_Solution" ,"Confidence_Level_Self_Criteria", "Confidence_Level_Self_Criteria_With_Solution"]].dropna()
                        accuracy = df['Accuracy'].mean()
                        
                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level', 'Confidence')
                        confidence, _ = confidence_inference_analysis.calculate_ECE_MCE(df, 'Confidence', n_bins)

                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level_With_Solution', 'Confidence_With_Solution')
                        confidence_with_solution, _ = confidence_inference_analysis.calculate_ECE_MCE(df, 'Confidence_With_Solution', n_bins)
                        
                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level_Self_Criteria', 'Confidence_Self_Criteria')
                        confidence_self_criteria, _ = confidence_inference_analysis.calculate_ECE_MCE(df, 'Confidence_Self_Criteria', n_bins)

                        confidence_inference_analysis.calculate_confidence_level_map(df, confidence_type, 'Confidence_Level_Self_Criteria_With_Solution', 'Confidence_Self_Criteria_With_Solution')
                        confidence_self_criteria_with_solution, _ = confidence_inference_analysis.calculate_ECE_MCE(df, 'Confidence_Self_Criteria_With_Solution', n_bins)

                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "confidence_type": confidence_type , 
                                        "accuracy": accuracy,
                                        "ece": confidence,
                                        "ece_with_solution": confidence_with_solution,
                                        "ece_self_criteria": confidence_self_criteria,
                                        "ece_self_criteria_with_solution": confidence_self_criteria_with_solution,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")

        df_summary = pd.DataFrame(data_list)
        group_cols=['confidence_type', 'dataset']        
        value_cols=['accuracy', 'ece', 'ece_with_solution' ,'ece_self_criteria', 'ece_self_criteria_with_solution']
        df_summary = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['confidence_type','dataset'])        
        print(df_summary.to_string(index=False))        


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
    def calculate_m_ratio() -> None:
        data_list = []
        dir, csv_paths = confidence_inference_analysis.get_filenames()
        for dataset, csv_dataset in csv_paths.items():
            file_paths = csv_dataset['file_paths']
            confidence_type = csv_dataset['confidence_type']
            from_run_number = csv_dataset['from_run_number']
            to_run_number = csv_dataset['to_run_number']
            for file_path in file_paths: 
                for run_number in range(from_run_number, to_run_number):
                    try:
                        file_path_run_number = file_path.replace('run_', f'run_{run_number}')
                        df = pd.read_csv(f'{dir}/{file_path_run_number}')

                        df = df[["Target", "Final_Answer" , "Accuracy", "Confidence_Level", "Confidence_Level_With_Solution" , "Confidence_Level_Self_Criteria", "Confidence_Level_Self_Criteria_With_Solution"]].dropna()
                        accuracy = df['Accuracy'].mean()
                        
                        d_prime_c, meta_d_prime_c, m_ratio_c = confidence_inference_analysis.calculate_metad_dprime(df, 'Confidence_Level')
                        d_prime_c_ws, meta_d_prime_c_ws, m_ratio_c_ws = confidence_inference_analysis.calculate_metad_dprime(df, 'Confidence_Level_With_Solution')
                        
                        d_prime_csc, meta_d_prime_csc, m_ratio_csc = confidence_inference_analysis.calculate_metad_dprime(df, 'Confidence_Level_Self_Criteria')
                        d_prime_csc_ws, meta_d_prime_csc_ws, m_ratio_csc_ws = confidence_inference_analysis.calculate_metad_dprime(df, 'Confidence_Level_Self_Criteria_With_Solution')
            
                        data_item = {
                                        "run_number": run_number, 
                                        "dataset": dataset , 
                                        "accuracy": accuracy,
                                        "d_prime_confidence": d_prime_c,
                                        "meta_d_prime_confidence": meta_d_prime_c,
                                        "m_ratio_confidence": m_ratio_c,
                                        "d_prime_confidence_ws": d_prime_c_ws,
                                        "meta_d_prime_confidence_ws": meta_d_prime_c_ws,
                                        "m_ratio_confidence_ws": m_ratio_c_ws,
                                        "d_prime_confidence_self_criteria": d_prime_csc,
                                        "meta_d_prime_confidence_self_criteria": meta_d_prime_csc,
                                        "m_ratio_confidence_self_criteria": m_ratio_csc,
                                        "d_prime_confidence_self_criteria_with_Solution": d_prime_csc_ws,
                                        "meta_d_prime_confidence_self_criteria_with_Solution": meta_d_prime_csc_ws,
                                        "m_ratio_confidence_self_criteria_with_Solution": m_ratio_csc_ws,
                                    }
                        data_list.append(data_item)
                    except Exception as e:
                        print(f"[WARN] {e}")

        df_summary = pd.DataFrame(data_list)
        group_cols=['dataset']        
        value_cols=['accuracy', 'm_ratio_confidence', 'm_ratio_confidence_ws', 'm_ratio_confidence_self_criteria', 'm_ratio_confidence_self_criteria_with_Solution']
        df_summary = confidence_inference_analysis.aggregate_mean_pandas_rounded(df_summary, group_cols, value_cols)
        df_summary = df_summary.sort_values(by=['dataset'])        
        print(df_summary.to_string(index=False))        

    @staticmethod
    def calculate_metad_dprime(df, confidence_column_name):
        df['Stimuli'] = df['Target'].map({'Yes': 1, 'No': 0})        
        df['Response'] = (df['Target'] == df['Final_Answer']).astype(int)        
        df['Accuracy_Reward'] = df['Accuracy'].map({True: 1, False: 0}) 
        

        confidence_bin_column_name = f"{confidence_column_name}_bin"
        unique_values = sorted(df[confidence_column_name].dropna().unique())
        value_to_bin = {value: i + 1 for i, value in enumerate(unique_values)}
        df[confidence_bin_column_name] = df[confidence_column_name].map(value_to_bin)
        
        fit = metad(
            data=df,
            stimuli = "Stimuli",
            accuracy = "Accuracy_Reward",
            confidence = confidence_bin_column_name,
            nRatings = len(unique_values))
        
        d_prime = fit["dprime"][0]
        meta_d = fit["meta_d"][0]
        m_ratio = fit["m_ratio"][0]
        return d_prime, meta_d, m_ratio

    @staticmethod
    def check_columns(df, required_cols):
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV")
            
        return None    

    @staticmethod
    def get_filenames() -> None:
        dir = './src/confidence'
        csv_paths = {
            "deepseek_r1_7b_no_training_p": {
                            "file_paths" : [
                                    f"verbal/deepSeek_r1_distill_qwen_7b/no_training/run_/llm_generation_metacognitive_no_training.csv", 
                            ],
                            "confidence_type": "probability",
                            "from_run_number": 9,
                            "to_run_number": 13,
                        },
            "deepseek_r1_7b_no_training_l": {
                            "file_paths" : [
                                    f"verbal/deepSeek_r1_distill_qwen_7b/no_training/run_/llm_generation_metacognitive_no_training.csv", 
                            ],
                            "confidence_type": "level",
                            "from_run_number": 13,
                            "to_run_number": 17,
                        },
            "qwen3_8b_no_training_p": {
                            "file_paths" : [
                                    f"verbal/qwen3_8b/no_training/run_/llm_generation_metacognitive_no_training.csv", 
                            ],
                            "confidence_type": "probability",
                            "from_run_number": 9,
                            "to_run_number": 13,
                        },
            "qwen3_8b_no_training_l": {
                            "file_paths" : [
                                    f"verbal/qwen3_8b/no_training/run_/llm_generation_metacognitive_no_training.csv", 
                            ],
                            "confidence_type": "level",
                            "from_run_number": 13,
                            "to_run_number": 17,
                        },
            # "qwen3_8b_ar_p": {
            #                 "file_paths" : [
            #                         f"verbal/qwen3_8b/settings_0/run_/llm_generation_metacognitive_settings_0.csv", 
            #                 ],
            #                 "confidence_type": "probability",
            #                 "from_run_number": 1,
            #                 "to_run_number": 5,
            #             },
            # "qwen3_8b_ar_confidence_p": {
            #                 "file_paths" : [
            #                         f"verbal/qwen3_8b/settings_1/run_/llm_generation_metacognitive_settings_1.csv", 
            #                 ],
            #                 "confidence_type": "probability",
            #                 "from_run_number": 1,
            #                 "to_run_number": 5,
            #             },
            # "qwen3_8b_ar_confidence_wc_p": {
            #                 "file_paths" : [
            #                         f"verbal/qwen3_8b/settings_2/run_/llm_generation_metacognitive_settings_2.csv", 
            #                 ],
            #                 "confidence_type": "probability",
            #                 "from_run_number": 1,
            #                 "to_run_number": 5,
            #             },
            # "qwen3_8b_confidence_p": {
            #                 "file_paths" : [
            #                         f"verbal/qwen3_8b/settings_3/run_/llm_generation_metacognitive_settings_3.csv", 
            #                 ],
            #                 "confidence_type": "probability",
            #                 "from_run_number": 1,
            #                 "to_run_number": 5,
            #             },
            # "qwen3_8b_settings_8_l": {
            #                 "file_paths" : [
            #                         f"verbal/qwen3_8b/settings_8/run_/llm_generation_metacognitive_settings_8.csv", 
            #                 ],
            #                 "confidence_type": "level",
            #                 "from_run_number": 1,
            #                 "to_run_number": 5,
            #             },
            # "qwen3_8b_settings_10_l": {
            #                 "file_paths" : [
            #                         f"verbal/qwen3_8b/settings_10/run_/llm_generation_metacognitive_settings_10.csv", 
            #                 ],
            #                 "confidence_type": "level",
            #                 "from_run_number": 1,
            #                 "to_run_number": 5,
            #             },

        }
        
        return dir, csv_paths

    @staticmethod
    def aggregate_mean_pandas_rounded(df, group_cols, value_cols) -> pd.DataFrame:
        result = df.groupby(group_cols)[value_cols].mean().reset_index()
        for col in value_cols:
            result[col] = result[col].round(3)
        return result

    @staticmethod
    def calculate_confidence_level_map(df, confidence_type, confidence_column_name, confidence_value_column_name):
        if "probability" == confidence_type:
            df[confidence_value_column_name] = df[confidence_column_name] / 100
            return None
        
        elif "level" == confidence_type:
            confidence_values = {
                "very low": 0.1,
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very high": 0.9,
            }

            df[confidence_value_column_name] = (
                df[confidence_column_name]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map(confidence_values)
            )
            return None
        
    @staticmethod
    def plot_histogram(df, column_name, bins=100):

        plt.figure(figsize=(8, 5))
        plt.hist(df[column_name].dropna(), bins=bins, edgecolor='black')

        plt.title(f"Histogram of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")

        plt.grid(alpha=0.3)
        plt.savefig(
            "src/confidence/verbal/histogram.png",
            dpi=300,              
            bbox_inches="tight"   
        )

        plt.show()
        plt.close()

    @staticmethod
    def plot_histogram_per_group(df, column_name):
        counts = df[column_name].value_counts()

        plt.figure(figsize=(10, 6))
        counts.plot(kind="bar")

        plt.xlabel(column_name)
        plt.ylabel("Frequency")
        plt.title(f"Frequency of each value in '{column_name}'")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            "src/confidence/verbal/histogram.png",
            dpi=300,              
            bbox_inches="tight"   
        )
        plt.close()


confidence_inference_analysis.calculate_auroc()
print()
confidence_inference_analysis.calculate_ece()
print()
confidence_inference_analysis.calculate_m_ratio()


# df = pd.read_csv('src/confidence/verbal/deepSeek_r1_distill_qwen_7b/no_training/run_5/llm_generation_metacognitive_no_training.csv')
# confidence_inference_analysis.plot_histogram_per_group(df, 'Confidence_Level_Self_Criteria')