from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from src.logger.diffusion_decision_model.diffusion_decision_model_log_entity import diffusion_decision_model_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_evidence_log_entity import diffusion_decision_model_evidence_log_entity
from src.logger.diffusion_decision_model.diffusion_decision_model_log_detail_entity import diffusion_decision_model_log_detail_entity
from src.datasets.dataset_handler import dataset_handler
from src.logger.diffusion_decision_model.diffusion_decision_model_logger import diffusion_decision_model_logger
from src.utils.utility import my_utils
from math import ceil
import math
import torch
import numpy as np 
import numpy as np
import traceback
from collections import Counter
import logging
import pandas as pd
import torch.nn.functional as F
import gc
import json 

logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)        


class diffusion_decision_model(ABC): 

    # A prompt is chat template text and already carries the model's start and turn tokens.
    # They are to be read as those tokens (tokenizer_mode 'hf'; vLLM's own Mistral tokenizer
    # reads "[INST]" as plain characters) and no second start token is to be put in front.
    TOKENIZATION = {'add_special_tokens': False}

    def __init__(self, modelname: str, number_of_evidence: int) -> None:
        self.modelname = modelname
        self.number_of_evidence = number_of_evidence

        if self.modelname is None:
            raise Exception('modelname is required')
        if self.number_of_evidence is None:
            raise Exception('number of evidence is required')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        

    def run(self, from_run_number: int , to_run_number: int) -> None:
        self.model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True, tokenizer_mode='hf')
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        
        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            
            log_list: list[diffusion_decision_model_log_entity] = self.generate_response()
            log_list = self.generate_self_consistency(log_list)
            
            logger = self.create_logger(run_number)
            logger.add_to_buffer_list(log_list)
            logger.write_to_log_file()
            
            print(f"{'*' * 210}")
            
        self.released_gpu_memory()

    def baseline_features_extractor(self, from_run_number: int , to_run_number: int) -> None:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
        bnb_4bit_use_double_quant = False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname, quantization_config = bnb_config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1        
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)

        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            logger = self.create_logger(run_number)
            df = pd.read_csv(logger.get_log_file_name())
            
            df = self.calculate_baseline_features(df)
            
            df.to_csv(logger.get_log_file_name(), index=False)            
            print(f"{'*' * 210}")
            
        self.released_gpu_memory()

    def evidence_features_extractor(self, from_run_number: int , to_run_number: int) -> None:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = getattr(torch, "bfloat16"),
        bnb_4bit_use_double_quant = False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname, quantization_config = bnb_config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1        
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)

        print(f"{'*' * 100}  {self.modelname}  {'*' * 100}")
        for run_number in range(from_run_number,to_run_number):
            print(f"{'*' * 100}  Run Number {run_number}  {'*' * 100}")
            logger = self.create_logger(run_number)
            df = pd.read_csv(logger.get_log_file_name())
            df_evidence = pd.read_csv(logger.get_evidence_log_file_name())
            
            df_evidence = self.calculate_evidence_features(df, df_evidence)
            
            df_evidence.to_csv(logger.get_evidence_log_file_name(), index=False)            
            print(f"{'*' * 210}")
            
        self.released_gpu_memory()

    def calculate_accracy(self, run_number: int) -> None:
        logger = self.create_logger(run_number)
        df = pd.read_csv(logger.get_log_file_name())
        percentage_true = df["Accuracy"].mean() * 100
        token_count = df["Token_Count"].mean()
        print(f"Records = {len(df)}, Accuracy = {percentage_true}, token count = {token_count}")        

    @torch.inference_mode()
    def generate_response(self, batch_size = 128) -> list[diffusion_decision_model_log_entity]: 
        print(f"{'*' * 100}  Generate Response {'*' * 100}")
        _, test_dataset = self.get_dataset().preprocess_dataset()

        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.2, 
                n = 1, 
                top_p= 0.9, 
                top_k=50,
                logprobs=1
            )
        
        log_list: list[diffusion_decision_model_log_entity] = []
        idx: int = 0
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing Batches", unit="step"):
            batch_dict = test_dataset[i : i + batch_size]

            batch = [
                {key: values[j] for key, values in batch_dict.items()}
                for j in range(len(batch_dict[next(iter(batch_dict))]))
            ]

            sample_ID_list = batch_dict['sample_id']
            problem_id_list = batch_dict['problem_id']
            split_list = batch_dict['split']
            question_list = batch_dict['question']
            prompt_list = batch_dict['prompt']
            target_list = batch_dict['target']
            try:
                outputs = self.model.generate(prompt_list, sampling_params, tokenization_kwargs=self.TOKENIZATION)
                for j, output in enumerate(outputs):
                    prompt = prompt_list[j]
                    sample_ID = sample_ID_list[j]
                    split = split_list[j]
                    target = target_list[j]
                    problem_id = problem_id_list[j]
                    question = question_list[j]
                    x = batch[j]
                    
                    log = diffusion_decision_model_log_entity()
                    log.ID = idx
                    log.x = x
                    log.sample_ID = sample_ID
                    log.problem_id = problem_id
                    log.split = split
                    log.question = question
                    log.prompt = prompt
                    log.target = target
                    
                    if output.outputs is None: continue
                    response = output.outputs[0]
                    # Stopped by the token limit: the answer was never finished, so whatever is read from the cut text is not the model's answer.
                    if response.finish_reason == 'length': continue
                    completion = response.text
                    
                    log.completion = completion
                    log.token_count = len(response.token_ids)
                    log.completion_loss = my_utils.get_loss_from_vllm_output(response)

                    try:
                        final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                        if final_answer is None or compared_final_answer is None: continue
                        log.final_answer = final_answer
                        log.compared_final_answer = compared_final_answer
                        log.accuracy = accuracy
                        log = self.add_evidence_log_list(log, response, list(output.prompt_token_ids))
                    except Exception as e:
                        logging.exception("An exception occurred")                        
                        print(f"[WARN]: {e}")
                        traceback.print_exc()                        

                    idx += 1
                    log_list.append(log)    
            except Exception as e:
                logging.exception("An exception occurred")                        
                print(f"[WARN]: {e}")
                traceback.print_exc()                        

        return log_list

    @torch.inference_mode()
    def generate_self_consistency(self, log_list: list[diffusion_decision_model_log_entity]) -> list[diffusion_decision_model_log_entity]: 
        print(f"{'*' * 100}  Generate Self Consistency {'*' * 100}")
        sc_sample_count = 10
        batch_size = ceil(120 / sc_sample_count)
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=1.0, 
                n = sc_sample_count, 
                top_p= 0.9, 
                top_k=50, 
                logprobs=1
            )

        evidence_log_list: list[diffusion_decision_model_evidence_log_entity] = []
        x_list: list[dict] = []
        original_prompt_list: list[str] = []
        final_answer_list: list[str] = []
        for log in log_list: 
            for evidence_log in log.evidence_list:
                evidence_log_list.append(evidence_log)
                x_list.append(log.x)
                original_prompt_list.append(log.prompt)
                final_answer_list.append(log.compared_final_answer)
        
        for i in tqdm(range(0, len(evidence_log_list), batch_size), desc="Processing Batches", unit="step"):
            batch: list[diffusion_decision_model_evidence_log_entity] = evidence_log_list[i : i + batch_size]        
            batch_partial_cot_list = list(map(lambda x: x.partial_cot, batch))
            
            batch_original_prompt_list: list[str] = original_prompt_list[i : i + batch_size]        
            batch_final_answer_list: list[str] = final_answer_list[i : i + batch_size]        

            try:
                # A rollout continues the very tokens of the original prompt and of the response so
                # far.  Building the prompt again from the question gave the rollouts another
                # instruction or another chat format than the response they are compared with.
                # prompt_list is the same prompt as text, for the log.
                prompt_list : list[str] = []
                for original_prompt, partial_cot in zip(batch_original_prompt_list, batch_partial_cot_list):
                    prompt_list.append(original_prompt + partial_cot)
                
                token_prompt_list = [{'prompt_token_ids': evidence_log.prompt_token_ids} for evidence_log in batch]
                outputs = self.model.generate(token_prompt_list, sampling_params, use_tqdm=False)
                for j, output in enumerate(outputs):
                    if output.outputs is None: continue
                    idx = i + j
                    evidence_log = batch[j]
                    prompt = prompt_list[j]
                    original_final_answer = batch_final_answer_list[j]
                    
                    for index in range(sc_sample_count):
                        response = output.outputs[index]
                        completion = response.text

                        log_detail = diffusion_decision_model_log_detail_entity()
                        log_detail.index = f'{idx}_{index}'
                        log_detail.prompt = prompt
                        log_detail.completion = completion
                        log_detail.token_count = len(response.token_ids)
                        log_detail.original_final_answer = original_final_answer
                        log_detail.loss = my_utils.get_loss_from_vllm_output(response)
                        
                        try:
                            final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), original_final_answer)
                            log_detail.final_answer = final_answer
                            log_detail.compared_final_answer = compared_final_answer
                            log_detail.accuracy = accuracy
                        except Exception as e:
                            log_detail.accuracy = False
                            logging.exception("An exception occurred")                        
                            print(f"[WARN]: {e}")
                            traceback.print_exc()                        
                            
                        evidence_log.add_consistency_list(log_detail)
                        
                    true_count = sum(x.accuracy for x in evidence_log.consistency_list)
                    evidence_log.evidence_accumulation_self_consistency = (true_count + 1.0) / (len(evidence_log.consistency_list) + 1.0)

                    losses = [
                        x.loss
                        for x in evidence_log.consistency_list
                        if x.accuracy and x.loss is not None
                    ]
                    losses.append(evidence_log.partial_cot_loss)
                    evidence_log.evidence_accumulation_loss = float(np.mean(losses))

            except Exception as e:
                logging.exception("An exception occurred")                        
                print(f"[WARN]: {e}")
                traceback.print_exc()                        

        
        for log in log_list: 
            for i in range(1, len(log.evidence_list)):
                current = log.evidence_list[i]
                previous = log.evidence_list[i - 1]
                current.delta_evidence_self_consistency = current.evidence_accumulation_self_consistency - previous.evidence_accumulation_self_consistency
                current.delta_evidence_loss = previous.evidence_accumulation_loss - current.evidence_accumulation_loss
                current.delta_evidence_avg_prob = current.evidence_accumulation_avg_prob - previous.evidence_accumulation_avg_prob
            
            self.calculate_self_consistency_log(log)
            self.calculate_self_consistency_completion_log(log)

        return log_list

    def calculate_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_columns_baseline_features(df)

        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt = df.loc[index, "Prompt"]
            completion = df.loc[index, "Completion"]
            
            # Named before the try, so the cleanup below can delete them even when the sample fails early.
            outputs = outputs_last_hidden_state = last_hidden_state = logits = representation = None
            inputs = input_ids = shift_logits = shift_labels = probs = log_probs = None

            with torch.inference_mode():
                try:
                    device = next(self.model.parameters()).device
                    
                    inputs = self.tokenizer(prompt + completion, return_tensors='pt', **self.TOKENIZATION)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    input_ids = inputs["input_ids"]

                    prompt_length = self.tokenizer(prompt, return_tensors='pt', **self.TOKENIZATION)["input_ids"].shape[1]

                    outputs_last_hidden_state = self.model.model(**inputs, labels=inputs["input_ids"], output_hidden_states=False)
                    last_hidden_state = outputs_last_hidden_state.last_hidden_state.float().squeeze(0).detach().cpu().numpy()

                    representation = np.mean(last_hidden_state[prompt_length:], axis=0)
                    df.at[index, "Last_Layer_Representations"] = json.dumps(representation.tolist())

                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    logits = outputs.logits

                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]

                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    probs = torch.exp(log_probs)

                    token_probs = []
                    token_entropy = []

                    # shift_labels[t] is input_ids[t + 1], so the answer's first
                    # label sits at t = prompt_length - 1.
                    for t in range(prompt_length - 1, shift_labels.shape[1]):
                        true_token_id = shift_labels[0, t].item()
                        prob = probs[0, t, true_token_id].item()
                        token_probs.append(prob)
                        entropy = -(probs[0, t] * log_probs[0, t]).sum().item()
                        token_entropy.append(entropy)

                    df.at[index, "Sequence_Probability"] = sum(token_probs)
                    df.at[index, "Length_Normalized_Sequence_Probability"] = sum(token_probs) / len(token_probs)
                    df.at[index, "Entropy"] = sum(token_entropy)
                    df.at[index, "Mean_Entropy"] = sum(token_entropy) / len(token_entropy)
                except Exception as e:
                    logging.exception("An exception occurred")                        
                    print(f"[WARN]: {e}")
                    traceback.print_exc()                        
                finally:            
                    del outputs, outputs_last_hidden_state, last_hidden_state, logits, representation
                    del inputs, input_ids
                    del shift_logits, shift_labels, probs, log_probs
                    gc.collect()
                    torch.cuda.empty_cache()

        return df 
    
    def calculate_evidence_features(self, df: pd.DataFrame, df_evidence: pd.DataFrame) -> pd.DataFrame:
        results = [] 
        for index, row in tqdm(df.iterrows(), total=len(df)):
            sample_ID = df.loc[index, "Sample_ID"]
            prompt = df.loc[index, "Prompt"]
            completion = df.loc[index, "Completion"]
            
            with torch.inference_mode():
                try:
                    device = next(self.model.parameters()).device
                    
                    inputs = self.tokenizer(prompt + completion, return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    input_ids = inputs["input_ids"]

                    prompt_length = self.tokenizer(prompt, return_tensors='pt')["input_ids"].shape[1]
                    prompt_completion_length = input_ids.shape[1]

                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    logits = outputs.logits

                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]

                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    probs = torch.exp(log_probs)

                    token_count = prompt_completion_length - prompt_length
                    base = token_count // (self.number_of_evidence + 1)
                    remainder = token_count % (self.number_of_evidence + 1)
                    
                    start = prompt_length - 1
                    old_evidence_accumulation_avg_prob = 0.0
                    for i in range(1, self.number_of_evidence):
                        group_size = base + (1 if i < remainder else 0)
                        evidence_index = i
                        evidence_token_count = group_size
                        token_probs = []
                        for t in range(start, start + evidence_token_count):
                            true_token_id = shift_labels[0, t].item()
                            prob = probs[0, t, true_token_id].item()
                            token_probs.append(prob)

                        evidence_accumulation_avg_prob = sum(token_probs) / len(token_probs) if len(token_probs) != 0 else 0.0
                        results.append({
                            "Sample_ID": sample_ID,
                            "Evidence_Index": evidence_index,
                            "Evidence_Token_Count": evidence_token_count,
                            "Evidence_Accumulation_Avg_Prob": evidence_accumulation_avg_prob,
                            "Delta_Evidence_Avg_Prob": evidence_accumulation_avg_prob - old_evidence_accumulation_avg_prob
                        })                        
                        start = start + evidence_token_count
                        old_evidence_accumulation_avg_prob = evidence_accumulation_avg_prob

                except Exception as e:
                    logging.exception("An exception occurred")                        
                    print(f"[WARN]: {e}")
                    traceback.print_exc()                        
                finally:            
                    del outputs, logits
                    del inputs, input_ids
                    del shift_logits, shift_labels, probs, log_probs
                    gc.collect()
                    torch.cuda.empty_cache()

        df_evidence['Evidence_Token_Count'] = np.nan
        df_evidence['Evidence_Token_Count'] = df_evidence['Evidence_Token_Count'].astype('Int64')
        df_evidence['Evidence_Accumulation_Avg_Prob'] = np.nan
        df_evidence['Evidence_Accumulation_Avg_Prob'] = df_evidence['Evidence_Accumulation_Avg_Prob'].astype('float64')
        df_evidence['Delta_Evidence_Avg_Prob'] = np.nan
        df_evidence['Delta_Evidence_Avg_Prob'] = df_evidence['Delta_Evidence_Avg_Prob'].astype('float64')
        for index, row in df_evidence.iterrows():
            sample_ID = df_evidence.loc[index, "Sample_ID"]
            evidence_index = df_evidence.loc[index, "Evidence_Index"]
            filter_list = list(filter(lambda x: x["Sample_ID"] == sample_ID and x["Evidence_Index"] == evidence_index , results))
            if len(filter_list) != 0: 
                item = filter_list[0]
                df_evidence.at[index, "Evidence_Token_Count"] = item["Evidence_Token_Count"]
                df_evidence.at[index, "Evidence_Accumulation_Avg_Prob"] = item["Evidence_Accumulation_Avg_Prob"]
                df_evidence.at[index, "Delta_Evidence_Avg_Prob"] = item["Delta_Evidence_Avg_Prob"]
            else: 
                df_evidence.at[index, "Evidence_Token_Count"] = 0
                df_evidence.at[index, "Evidence_Accumulation_Avg_Prob"] = 0.0
                df_evidence.at[index, "Delta_Evidence_Avg_Prob"] = 0.0
            

        return df_evidence 

    def create_columns_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Entropy'] = np.nan
        df['Entropy'] = df['Entropy'].astype('float64')
        df['Mean_Entropy'] = np.nan
        df['Mean_Entropy'] = df['Mean_Entropy'].astype('float64')
        
        df['Sequence_Probability'] = np.nan
        df['Sequence_Probability'] = df['Sequence_Probability'].astype('float64')
        df['Length_Normalized_Sequence_Probability'] = np.nan
        df['Length_Normalized_Sequence_Probability'] = df['Length_Normalized_Sequence_Probability'].astype('float64')

        df['Last_Layer_Representations'] = np.nan
        df['Last_Layer_Representations'] = df['Last_Layer_Representations'].astype('str')
        
        return df 
        
    def is_answer_present(self, answer) -> bool:
        """True when a rollout reached an answer that can take part in the vote.

        The test this replaces rejected None only. A failed extraction can also
        leave a not a number behind, and that passed, so the rollouts that
        found no answer grouped into one block, won the vote, and were written
        out as a confidence of one for an answer of 'nan'.
        """
        if answer is None:
            return False

        if isinstance(answer, float) and math.isnan(answer):
            return False

        return str(answer).strip().lower() not in ('', 'nan', 'none')

    def calculate_self_consistency_log(self, log: diffusion_decision_model_log_entity) -> diffusion_decision_model_log_entity:
        evidence_filtered_list = list(filter(lambda x: x.index == 0 , log.evidence_list))
        if len(evidence_filtered_list) == 0:
            return log
        
        evidence_0: diffusion_decision_model_evidence_log_entity = evidence_filtered_list[0]
        answers = [
            log_detail.compared_final_answer
            for log_detail in evidence_0.consistency_list
            if self.is_answer_present(log_detail.compared_final_answer)
        ]

        if not answers: 
            # Not one rollout reached a readable answer, so there is no vote to
            # report. Leave it empty rather than zero, which would be read back
            # as a confidence that was measured and found to be nothing.
            log.self_consistency_confidence = None
            return log
        
        answer_counts = Counter(answers)
        mv_compare_final_answer, mv_count = answer_counts.most_common(1)[0]
        log.self_consistency_confidence = mv_count / len(evidence_0.consistency_list)
        try:
            mv_accuracy, compared_final_answer = self.get_dataset().verify_final_answer(log.target, str(mv_compare_final_answer))
            log.self_consistency_accuracy = mv_accuracy
            log.self_consistency_final_answer = compared_final_answer
        except Exception as e:
            logging.exception("An exception occurred")                        
            print(f"[WARN]: {e}")
            traceback.print_exc()                        
            
        return log

    def calculate_self_consistency_completion_log(self, log: diffusion_decision_model_log_entity) -> diffusion_decision_model_log_entity:
        evidence_filtered_list = list(filter(lambda x: x.index == len(log.evidence_list) - 1 , log.evidence_list))
        if len(evidence_filtered_list) == 0:
            return log
        
        evidence_last: diffusion_decision_model_evidence_log_entity = evidence_filtered_list[0]
        answers = [
            log_detail.compared_final_answer
            for log_detail in evidence_last.consistency_list
            if self.is_answer_present(log_detail.compared_final_answer)
        ]

        if not answers: 
            # Same as above: no readable answer means no vote, not a vote that
            # came out at zero.
            log.self_consistency_completion_confidence = None
            return log
        
        answer_counts = Counter(answers)
        mv_compare_final_answer, mv_count = answer_counts.most_common(1)[0]
        log.self_consistency_completion_confidence = mv_count / len(evidence_last.consistency_list)
        try:
            mv_accuracy, compared_final_answer = self.get_dataset().verify_final_answer(log.target, str(mv_compare_final_answer))
            log.self_consistency_completion_accuracy = mv_accuracy
            log.self_consistency_completion_final_answer = compared_final_answer
        except Exception as e:
            logging.exception("An exception occurred")                        
            print(f"[WARN]: {e}")
            traceback.print_exc()                        
            
        return log
    
    def add_evidence_log_list(self, log: diffusion_decision_model_log_entity, response, prompt_token_ids: list) -> diffusion_decision_model_log_entity:
        token_count = len(response.logprobs)
        base = token_count // self.number_of_evidence
        remainder = token_count % self.number_of_evidence
        
        token_ids = response.token_ids 
        evidence_accumulation_avg_prob: float = my_utils.get_avg_prob_from_vllm_output(response, token_start = 0, token_end = token_count)
        evidence_log: diffusion_decision_model_evidence_log_entity = self.create_evidence_log(index = 0, evidence = '', evidence_token_count= token_count, partial_cot = '', partial_completion = log.completion, partial_cot_loss = log.completion_loss, evidence_accumulation_avg_prob = evidence_accumulation_avg_prob)
        evidence_log.prompt_token_ids = prompt_token_ids
        log.add_evidence_list(evidence_log)

        start = 0
        for i in range(1, self.number_of_evidence):
            group_size = base + (1 if i < remainder else 0)
            evidence = self.tokenizer.decode(token_ids[start:start + group_size], skip_special_tokens=True)
            evidence_token_count = group_size
            partial_cot = self.tokenizer.decode(token_ids[0:start + group_size], skip_special_tokens=True)
            partial_completion = self.tokenizer.decode(token_ids[start + group_size:], skip_special_tokens=True)
            partial_cot_loss: float = my_utils.get_loss_from_vllm_output(response, token_start = start + group_size)
            evidence_accumulation_avg_prob: float = my_utils.get_avg_prob_from_vllm_output(response, token_start = start, token_end = start + group_size)

            evidence_log: diffusion_decision_model_evidence_log_entity = self.create_evidence_log(index = i, evidence = evidence, evidence_token_count = evidence_token_count, partial_cot = partial_cot, partial_completion = partial_completion, partial_cot_loss = partial_cot_loss, evidence_accumulation_avg_prob = evidence_accumulation_avg_prob)
            evidence_log.prompt_token_ids = prompt_token_ids + list(token_ids[:start + group_size])
            log.add_evidence_list(evidence_log)

            start += group_size
           
                
        return log

    def create_evidence_log(self, index: int, evidence: str , evidence_token_count: int, partial_cot: str, partial_completion: str, partial_cot_loss: float, evidence_accumulation_avg_prob: float) -> int:
        evidence_log = diffusion_decision_model_evidence_log_entity()
        evidence_log.index = index
        evidence_log.evidence = evidence
        evidence_log.evidence_token_count = evidence_token_count
        evidence_log.partial_cot = partial_cot
        evidence_log.partial_completion = partial_completion
        evidence_log.partial_cot_loss = partial_cot_loss
        evidence_log.evidence_accumulation_avg_prob = evidence_accumulation_avg_prob
        return evidence_log

    def released_gpu_memory(self) -> None:
        del self.model
        del self.tokenizer

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        

    def get_modelname_dir(self) -> str:
        return self.modelname.replace('/', '-').lower()

    def get_number_of_evidence_dir(self) -> str:
        return f'_nv_{self.number_of_evidence}'

    @abstractmethod
    def get_dataset(self) -> dataset_handler:
        pass

    @abstractmethod
    def create_logger(self, run_number) -> diffusion_decision_model_logger:
        pass

    @abstractmethod
    def get_max_new_tokens(self) -> int:
        pass
    
