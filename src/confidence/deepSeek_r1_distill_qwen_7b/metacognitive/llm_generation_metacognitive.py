from src.logger.llm_response.llm_response_inference_logger import llm_response_inference_logger
from src.logger.llm_response.llm_response_log_entity import llm_response_log_entity
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.confidence.llm_generation import llm_generation
from src.datasets.confidence.metacognitive_dataset import metacognitive_dataset
from src.datasets.dataset_config import dataset_config
import torch
import re


class llm_generation_metacognitive(llm_generation): 

    def __init__(self, modelname):
        super().__init__(modelname)

    @torch.inference_mode()
    def generate_response(self, batch_size = 128, run_number = 0): 
        _, test_dataset = self.get_dataset().preprocess_dataset()

        print(f'{'*' * 90}  Generate Response Run Number {run_number} {'*' * 90}')
        model = LLM(model=self.modelname, tensor_parallel_size=1, trust_remote_code=True,)
        sampling_params = SamplingParams(
                max_tokens=self.get_max_new_tokens(), 
                temperature=0.7, 
                n = 1, 
                top_p= 0.9, 
                top_k=50
            )
        
        log_list: list[llm_response_log_entity] = []
        idx: int = 0
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing Batches", unit="step"):
            batch = test_dataset[i : i + batch_size]


            sample_ID_list = batch['sample_id']
            problem_id_list = batch['problem_id']
            split_list = batch['split']
            question_list = batch['question']

            correct_prompt_list = batch['correct_prompt']
            correct_answer_list = batch['correct_answer']
            correct_target_list = batch['correct_target']
            log_list.extend(self.generate(model, sampling_params, sample_ID_list, problem_id_list, split_list, question_list, correct_prompt_list, correct_answer_list, correct_target_list, idx))
            idx += len(log_list)
            
            incorrect_prompt_list = batch['incorrect_prompt']
            incorrect_answer_list = batch['incorrect_answer']
            incorrect_target_list = batch['incorrect_target']
            log_list.extend(self.generate(model, sampling_params, sample_ID_list, problem_id_list, split_list, question_list, incorrect_prompt_list, incorrect_answer_list, incorrect_target_list, idx))
            idx += len(log_list)

        logger = self.create_llm_response_logger(run_number)
        logger.add_to_buffer_list(log_list)
        logger.write_to_log_file()

    def generate(self, model, sampling_params, sample_ID_list, problem_id_list, split_list, question_list, prompt_list, answer_list, target_list, idx) -> list[llm_response_log_entity]: 
        log_list: list[llm_response_log_entity] = []
        try:
            outputs = model.generate(prompt_list, sampling_params)
            for j, output in enumerate(outputs):
                prompt = prompt_list[j]
                sample_ID = sample_ID_list[j]
                split = split_list[j]
                target = target_list[j]
                problem_id = problem_id_list[j]
                question = question_list[j]
                answer = answer_list[j]
                
                log = llm_response_log_entity()
                log.ID = idx
                log.sample_ID = sample_ID
                log.problem_id = problem_id
                log.split = split
                log.question = question
                log.answer = answer
                log.prompt = prompt
                log.target = target
                
                if output.outputs is None: continue
                response = output.outputs[0]
                completion = response.text
                
                log.completion = completion
                log.token_count = len(response.token_ids)

                try:
                    final_answer, accuracy, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, str(completion), target)
                    log.final_answer = final_answer
                    log.compared_final_answer = compared_final_answer
                    log.accuracy = accuracy
                except Exception as e:
                    print(f"[WARN] generate failed: {e}")

                idx += 1
                log_list.append(log)    
        except Exception as e:
            print(f"[WARN] generate failed: {e}")
            
        return log_list


    def generate_model_prompt_confidence(self, question_list, answer_list) -> list[str]:
        prompt_list : list[str] = []
        for question, answer in zip(question_list, answer_list):        
        
            prompt = f'Question: {question}\n\n'
            prompt += f'Answer: {answer}\n'
            prompt += 'Determine the confidence level for the above question and answer by performing a step-by-step evaluation.\n'
            prompt += 'Rate your confidence as an integer between 0 and 100, where 0 means no confidence at all and 100 means absolute certainty. Use only the following format:\n'
            prompt += 'Confidence:<integer between 0 and 100>\n'
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_self_criteria(self, question_list, answer_list, accuracy_list) -> list[str]:
        prompt_list : list[str] = []
        for question, answer, accuracy in zip(question_list, answer_list, accuracy_list):        
        
            prompt = f'Question: {question}\n\n'
            prompt += f'Answer: {answer}\n'
            
            correctness = 'correct' if accuracy else 'incorrect'
            prompt += f'Correctness Status: {correctness}\n'

            prompt += 'A question, its answer, and the correctness status of the answer (i.e., whether the answer is correct or incorrect) are provided.\n'
            prompt += '''Based on the question and answer, generate a numbered list of up to five criteria for assessing confidence in the answer's correctness.\n'''
            prompt += 'Only generate the evaluation criteria; do not evaluate the answer.\n'

            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_self_criteria_confidence(self, question_list, answer_list, self_criteria_list) -> list[str]:
        prompt_list : list[str] = []
        for question, answer, self_criteria in zip(question_list, answer_list, self_criteria_list):        
        
            prompt = f'Question: {question}\n\n'
            prompt += f'Answer: {answer}\n\n'
            prompt += f'Criteria:\n {self_criteria}\n\n'
            prompt += 'Determine the confidence level for the above question and answer based on the criteria by performing a step-by-step evaluation.\n'
            prompt += 'Rate your confidence as an integer between 0 and 100, where 0 means no confidence at all and 100 means absolute certainty. Use only the following format:\n'
            prompt += 'Confidence:<integer between 0 and 100>\n'

            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_self_criteria_with_solution(self, question_list, answer_list, completion_list, accuracy_list) -> list[str]:
        prompt_list : list[str] = []
        for question, answer, completion, accuracy in zip(question_list, answer_list, completion_list, accuracy_list):        
        
            prompt = f'[Question]: {question}\n\n'
            prompt += f'[Answer]: {answer}\n\n'
            prompt += f'[Reasoning Process]: {completion}\n\n'
            
            correctness = 'correct' if accuracy else 'incorrect'
            prompt += f'[Correctness Status]: {correctness}\n\n'

            prompt += 'A question, its answer, the reasoning process used to reach the answer, and the correctness status of the answer (i.e., whether the answer is correct or incorrect) are provided.\n'
            prompt += '''Based on the question, answer and reasoning process, generate a numbered list of up to five criteria for assessing confidence in the answer's correctness.\n'''
            prompt += 'Only generate the evaluation criteria; do not evaluate the answer, reasoning process, or correctness status.\n'

            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_self_criteria_with_solution_confidence(self, question_list, answer_list, completion_list, self_criteria_list) -> list[str]:
        prompt_list : list[str] = []
        for question, answer, completion, self_criteria in zip(question_list, answer_list, completion_list, self_criteria_list):        
        
            prompt = f'[Question]: {question}\n\n'
            prompt += f'[Answer]: {answer}\n\n'
            prompt += f'[Reasoning Process]: {completion}\n\n'
            prompt += f'[Criteria]:\n {self_criteria}\n\n'
            prompt += 'Determine the confidence level for the above question, answer and reasoning process based on the criteria by performing a step-by-step evaluation.\n'
            prompt += 'Rate your confidence as an integer between 0 and 100, where 0 means no confidence at all and 100 means absolute certainty. Use only the following format:\n'
            prompt += 'Confidence:<integer between 0 and 100>\n'

            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def extract_confidence(self, solution):
        patterns = [
            r"Confidence[\s*]*:[\s*]*(\d+(?:\.\d+)?)",
            r"Confidence\s*Score[\s*\n:]*([0-9]+(?:\.[0-9]+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue
            answer = float(match.group(1))
            if answer > 100 or answer < 0: continue
            return answer
        
        return None

    def extract_self_criteria(self, self_criteria_completion):
        parts = self_criteria_completion.split("</think>", 1)
        if len(parts) >= 2:
            content = parts[1]
        else:
            content = self_criteria_completion
            
        patterns = [
            r"^\s*\**\s*Step\s+\d+\s*[:\.\-]?\s*(.+)$", 
            r"^\s*(?:\d+\s*[\.:]?|-[\.:]?)\s*(.+)$",
            r"^\s*(?:\[\s*\d+\s*\]|\d+\s*[\.:]?|-[\.:]?)\s*(.+)$",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, flags=re.MULTILINE | re.IGNORECASE)
            if not matches or len(matches) == 0: continue
            result = "\n".join(
                f"{i}. {item.strip()}"
                for i, item in enumerate(matches, start=1)
            )
            return result

        return ""

    def get_dataset(self) -> metacognitive_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(5)
            self.dataset = metacognitive_dataset(config)
        return self.dataset

    def create_llm_response_logger(self, run_number) -> llm_response_inference_logger:
        return llm_response_inference_logger(log_file_name = f'src/confidence/deepSeek_r1_distill_qwen_7b/metacognitive/run_{run_number}/llm_generation_metacognitive.csv')

    def get_max_new_tokens(self) -> int:
        return 15000


t = llm_generation_metacognitive(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
# t.run(from_run_number=5, to_run_number=6)
# t.run(from_run_number=6, to_run_number=7)
# t.run(from_run_number=7, to_run_number=8)
t.run(from_run_number=8, to_run_number=9)