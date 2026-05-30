from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import pandas as pd
import torch.nn.functional as F
import glob
import numpy as np
import math

class my_utils(object):
    
    @staticmethod
    def has_add_bos_token(tokenizer):
        if tokenizer is None: 
            return False
        
        if hasattr(tokenizer, 'init_kwargs'):
            return tokenizer.init_kwargs.get('add_bos_token', 'Not found')
        
        return False

    @staticmethod
    def calculate_entropy(logits):
        entropies = []
        for step_logits in logits:
                probs = F.softmax(step_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum()
                entropies.append(entropy.item())

        return sum(entropies) / len(entropies)

    @staticmethod
    def calculate_perplexity(loss : float) -> float:
        if loss is None: 
            return None
        return math.exp(loss)

    @staticmethod
    def tensor_tostring(value):
        if value is None: 
            return None
        return value.item()

    @staticmethod
    def embedding_tostring(value):
        if value is None: 
            return None
        
        return value.shape

    @staticmethod
    def convert_model_into_hugginfacces(base_model_path, peft_model_path):
        base = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        model = PeftModel.from_pretrained(base, peft_model_path)
        model = model.merge_and_unload()

        # save final model
        peft_model_path_hf = peft_model_path + '-HF'
        model.save_pretrained(peft_model_path_hf)
        tokenizer.save_pretrained(peft_model_path_hf)

    @staticmethod
    def convert_parquet_to_csv(file_full_path):
        df = pd.read_parquet(file_full_path)
        csv_file_full_path = file_full_path.replace('parquet', 'csv')
        df.to_csv(csv_file_full_path, index=False)
        print(f"Conversion completed successfully! : {csv_file_full_path}")        

    @staticmethod
    def split_csv_file(file_full_path: str, num_parts: int) -> None: 
        df = pd.read_csv(file_full_path)
        indices = np.array_split(df.index, num_parts)
        for i, idx in enumerate(indices, start=1):
            chunk = df.loc[idx]
            output_file = file_full_path.replace(".csv", f"_part_{i}.csv")
            chunk.to_csv(output_file, index=False)

    @staticmethod
    def merge_csv_files(directory_full_path: str, file_name: str) -> None: 
        files = sorted(glob.glob(f"{directory_full_path}/*_part_*.csv"))

        if not files:
            print("files not found")
            exit()

        df_list = [pd.read_csv(file) for file in files]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(f'{directory_full_path}/{file_name}', index=False)

