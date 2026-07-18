from src.logger.llm_response.llm_response_log_entity import llm_response_log_entity
from tqdm import tqdm
from vllm import LLM, SamplingParams
from src.confidence.llm_generation import llm_generation
from src.datasets.confidence.metacognitive_dataset import metacognitive_dataset
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import confidence_type_enum
import torch
import re


class llm_generation_metacognitive(llm_generation): 

    def __init__(self, modelname) -> None:
        super().__init__(modelname)

    @torch.inference_mode()
    def generate_response(self, batch_size = 128, run_number = 0): 
        _, test_dataset = self.get_dataset().preprocess_dataset()

        print(f"{'*' * 90}  Generate Response Run Number {run_number} {'*' * 90}")
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
            correct_proposed_answer_list = batch['correct_answer']
            correct_target_list = batch['correct_target']
            log_list.extend(self.generate(self.model, sampling_params, sample_ID_list, problem_id_list, split_list, question_list, correct_prompt_list, correct_proposed_answer_list, correct_target_list, idx))
            idx += len(log_list)
            
            incorrect_prompt_list = batch['incorrect_prompt']
            incorrect_proposed_answer_list = batch['incorrect_answer']
            incorrect_target_list = batch['incorrect_target']
            log_list.extend(self.generate(self.model, sampling_params, sample_ID_list, problem_id_list, split_list, question_list, incorrect_prompt_list, incorrect_proposed_answer_list, incorrect_target_list, idx))
            idx += len(log_list)

        logger = self.create_llm_response_logger(run_number)
        logger.add_to_buffer_list(log_list)
        logger.write_to_log_file()

    def generate(self, model, sampling_params, sample_ID_list, problem_id_list, split_list, question_list, prompt_list, proposed_answer_list, target_list, idx) -> list[llm_response_log_entity]: 
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
                proposed_answer = proposed_answer_list[j]
                
                log = llm_response_log_entity()
                log.ID = idx
                log.sample_ID = sample_ID
                log.problem_id = problem_id
                log.split = split
                log.question = question
                log.proposed_answer = proposed_answer
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


    def generate_model_prompt_confidence(self, confidence_type: confidence_type_enum, batch) -> list[str]:
        question_list = batch['Question']
        proposed_answer_list = batch['Proposed_Answer']
        final_answer_list = batch['Final_Answer']

        prompt_list : list[str] = []
        for question, proposed_answer, final_answer in zip(question_list, proposed_answer_list, final_answer_list):        
            
            if confidence_type_enum.PROBABILITY == confidence_type: 
                prompt = f'[Question]: {question}\n\n'
                prompt += f'[Proposed Answer]: {proposed_answer}\n\n'
                prompt += f'[Correctness Label]: {final_answer}\n\n'

                prompt += (
                    'Your only task is to evaluate how likely it is that the provided '
                    'Correctness Label (Yes/No) is accurate based solely on the Question '
                    'and the Proposed Answer.\n\n'
                    
                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence score or number in this section.\n'
                    '2. After you have completed your reasoning, on a new line, output your final confidence in the exact format: Confidence:<integer between 0 and 100>\n'
                    '3. Do not output anything else after the confidence value. Do not output the confidence before the reasoning.\n'
                )

                
            elif confidence_type_enum.LEVEL == confidence_type:

                prompt = f'[Question]: {question}\n\n'
                prompt += f'[Proposed Answer]: {proposed_answer}\n\n'
                prompt += f'[Correctness Label]: {final_answer}\n\n'

                prompt += (
                    'Your only task is to evaluate how likely it is that the provided '
                    'Correctness Label (Yes/No) is accurate based solely on the Question '
                    'and the Proposed Answer.\n\n'

                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence level or category during your reasoning.\n'
                    '2. After all reasoning is complete, output exactly one final line in the following format:\n'
                    '   Confidence:<Very Low | Low | Medium | High | Very High>\n'
                    '3. The confidence level must be exactly one of: Very Low, Low, Medium, High, or Very High.\n'
                    '4. Do not output anything before the reasoning or after the confidence line.\n'
                )
            
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_confidence_with_solution(self, confidence_type: confidence_type_enum, batch) -> list[str]:
        question_list = batch['Question']
        proposed_answer_list = batch['Proposed_Answer']
        final_answer_list = batch['Final_Answer']
        completion_list = batch['Completion']

        prompt_list : list[str] = []
        for question, proposed_answer, final_answer, completion in zip(question_list, proposed_answer_list, final_answer_list, completion_list):        
            if confidence_type_enum.PROBABILITY == confidence_type: 
                prompt = f'[Question]: {question}\n\n'
                prompt += f'[Proposed Answer]: {proposed_answer}\n\n'
                prompt += f'[Reasoning Process]: {completion}\n\n'
                prompt += f'[Correctness Label]: {final_answer}\n\n'

                prompt += (
                    'Your only task is to evaluate how likely it is that the provided '
                    'Correctness Label (Yes/No) is accurate based solely on the Question, '
                    'the Proposed Answer and Reasoning Process.\n\n'
                    
                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence score or number in this section.\n'
                    '2. After you have completed your reasoning, on a new line, output your final confidence in the exact format: Confidence:<integer between 0 and 100>\n'
                    '3. Do not output anything else after the confidence value. Do not output the confidence before the reasoning.\n'
                )
                
            elif confidence_type_enum.LEVEL == confidence_type:

                prompt = f'[Question]: {question}\n\n'
                prompt += f'[Proposed Answer]: {proposed_answer}\n\n'
                prompt += f'[Reasoning Process]: {completion}\n\n'
                prompt += f'[Correctness Label]: {final_answer}\n\n'

                prompt += (
                    'Your only task is to evaluate how likely it is that the provided '
                    'Correctness Label (Yes/No) is accurate based solely on the Question, '
                    'the Proposed Answer and Reasoning Process.\n\n'

                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence level or category during your reasoning.\n'
                    '2. After all reasoning is complete, output exactly one final line in the following format:\n'
                    '   Confidence:<Very Low | Low | Medium | High | Very High>\n'
                    '3. The confidence level must be exactly one of: Very Low, Low, Medium, High, or Very High.\n'
                    '4. Do not output anything before the reasoning or after the confidence line.\n'
                )

            
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_self_criteria(self, batch) -> list[str]:
        question_list = batch['Question']
        proposed_answer_list = batch['Proposed_Answer']
        final_answer_list = batch['Final_Answer']
        
        prompt_list : list[str] = []
        for question, proposed_answer, final_answer in zip(question_list, proposed_answer_list, final_answer_list):        
        
            prompt = f'[Question]: {question}\n\n'
            prompt += f'[Proposed Answer]: {proposed_answer}\n'
            prompt += f'[Correctness Label]: {final_answer}\n'

            prompt += (
                'A question, a proposed answer, and a correctness label (Yes or No) are provided. '
                'Based only on the question and the proposed answer, generate a numbered list '
                'of up to five criteria for assessing confidence in the accuracy of the '
                'provided correctness label. Generate only the evaluation criteria; '
                'do not evaluate the proposed answer or the correctness label.\n'
            )
            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_self_criteria_confidence(self, confidence_type: confidence_type_enum, batch) -> list[str]:
        question_list = batch['Question']
        proposed_answer_list = batch['Proposed_Answer']
        final_answer_list = batch['Final_Answer']
        self_criteria_list = batch['Self_Criteria']
        return self.generate_model_prompt_self_criteria_confidence_common(confidence_type, question_list, proposed_answer_list, final_answer_list, self_criteria_list)

    def generate_model_prompt_self_criteria_with_solution(self, batch) -> list[str]:
        question_list = batch['Question']
        proposed_answer_list = batch['Proposed_Answer']
        completion_list = batch['Completion']
        final_answer_list = batch['Final_Answer']
        
        prompt_list : list[str] = []
        for question, proposed_answer, completion, final_answer in zip(question_list, proposed_answer_list, completion_list, final_answer_list):        
        
            prompt = f'[Question]: {question}\n\n'
            prompt += f'[Proposed Answer]: {proposed_answer}\n'
            prompt += f'[Reasoning Process]: {completion}\n\n'
            prompt += f'[Correctness Label]: {final_answer}\n'

            prompt += (
                'A question, a proposed answer, a reasoning process, and a correctness label (Yes or No) are provided. '
                'Based only on the question, the proposed answer and reasoning process, generate a numbered list '
                'of up to five criteria for assessing confidence in the accuracy of the '
                'provided correctness label. Generate only the evaluation criteria; '
                'do not evaluate the proposed answer or the correctness label.\n'
            )

            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def generate_model_prompt_self_criteria_with_solution_confidence(self, confidence_type: confidence_type_enum, batch) -> list[str]:
        question_list = batch['Question']
        proposed_answer_list = batch['Proposed_Answer']
        final_answer_list = batch['Final_Answer']
        self_criteria_list = batch['Self_Criteria_With_Solution']
        return self.generate_model_prompt_self_criteria_confidence_common(confidence_type, question_list, proposed_answer_list, final_answer_list, self_criteria_list)

    def generate_model_prompt_self_criteria_confidence_common(self, confidence_type: confidence_type_enum, question_list, proposed_answer_list, final_answer_list, self_criteria_list) -> list[str]:
        prompt_list : list[str] = []
        for question, proposed_answer, final_answer, self_criteria in zip(question_list, proposed_answer_list, final_answer_list, self_criteria_list):        
        
            if confidence_type_enum.PROBABILITY == confidence_type: 
                prompt = f'[Question]: {question}\n\n'
                prompt += f'[Proposed Answer]: {proposed_answer}\n\n'
                prompt += f'[Evaluation Criteria]:\n{self_criteria}\n\n'
                prompt += f'[Correctness Label]: {final_answer}\n\n'

                prompt += (
                    'Your only task is to evaluate how likely it is that the provided '
                    'Correctness Label (Yes/No) is accurate based solely on the Question, '
                    'the Proposed Answer and Evaluation Criteria.\n\n'
                    
                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence score or number in this section.\n'
                    '2. After you have completed your reasoning, on a new line, output your final confidence in the exact format: Confidence:<integer between 0 and 100>\n'
                    '3. Do not output anything else after the confidence value. Do not output the confidence before the reasoning.\n'
                )
                
            elif confidence_type_enum.LEVEL == confidence_type:

                prompt = f'[Question]: {question}\n\n'
                prompt += f'[Proposed Answer]: {proposed_answer}\n\n'
                prompt += f'[Evaluation Criteria]:\n{self_criteria}\n\n'
                prompt += f'[Correctness Label]: {final_answer}\n\n'

                prompt += (
                    'Your only task is to evaluate how likely it is that the provided '
                    'Correctness Label (Yes/No) is accurate based solely on the Question, '
                    'the Proposed Answer and Evaluation Criteria.\n\n'

                    'Follow these instructions strictly:\n'
                    '1. First, provide your detailed step-by-step reasoning. Do not mention any confidence level or category during your reasoning.\n'
                    '2. After all reasoning is complete, output exactly one final line in the following format:\n'
                    '   Confidence:<Very Low | Low | Medium | High | Very High>\n'
                    '3. The confidence level must be exactly one of: Very Low, Low, Medium, High, or Very High.\n'
                    '4. Do not output anything before the reasoning or after the confidence line.\n'
                )

            prefix = [
                {"role": "user",
                    "content": prompt
                    },
            ]
            prompt_list.append(self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True))
        
        return prompt_list        

    def extract_confidence(self, confidence_type: confidence_type_enum, solution) -> str:
        if confidence_type_enum.PROBABILITY == confidence_type: 
            return self.extract_confidence_probability(solution)
        elif confidence_type_enum.LEVEL == confidence_type:
            return self.extract_confidence_level(solution)
        
        return None

    def extract_confidence_probability(self, solution):
        patterns = [
            r"Confidence\s*:\s*<\s*(\d{1,3})\s*>",
            r"\[\s*Confidence\s*\]\s*:\s*(\d+)",
            r"Confidence[\s*]*:[\s*]*(\d+(?:\.\d+)?)",
            r"Confidence\s*Score[\s*\n:]*([0-9]+(?:\.[0-9]+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue
            answer = float(match.group(1))
            if answer > 100 or answer < 0: continue
            return str(answer)
        
        return None

    def extract_confidence_level(self, solution):
        levels = [
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
        lookup = {item.lower(): item for item in levels}

        patterns = [
            r"confidence\s*[:=\-]?\s*(very\s+low|low|medium|high|very\s+high)\b",
            r"\b(very\s+low|low|medium|high|very\s+high)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if not match: continue

            confidence_level = match.group(1)
            if not confidence_level: continue
            
            return lookup.get(confidence_level.lower())            
        
        return None

    def extract_self_criteria(self, self_criteria_completion):
        parts = self_criteria_completion.split("</think>", 1)
        if len(parts) >= 2:
            content = parts[1]
        else:
            content = self_criteria_completion
            
        patterns = [
            r"^\s*\**\s*Step\s+\d+\s*[:\.\-]?\s*(.+)$", 
            r"^\s*(?:\[\s*\d+\s*\]|\d+\s*[\.:]?|-[\.:]?)\s*(.+)$",
            r"^\s*(?:\d+\s*[\.:]?|-[\.:]?)\s*(.+)$",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, flags=re.MULTILINE | re.IGNORECASE)
            if not matches or len(matches) == 0: continue
            matches = list(set(matches))            
            result = "\n".join(
                f"{i}. {item.strip()}"
                for i, item in enumerate(matches, start=1)
            )
            return result

        return ""

    def get_dataset(self) -> metacognitive_dataset:
        if self.dataset is None:
            config = dataset_config(self.modelname)
            config.set_max_test_dataset_size(10)
            self.dataset = metacognitive_dataset(config)
        return self.dataset

    def get_max_new_tokens(self) -> int:
        return 15000


