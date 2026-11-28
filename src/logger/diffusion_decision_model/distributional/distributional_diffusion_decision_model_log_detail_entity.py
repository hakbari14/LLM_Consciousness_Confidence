from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class distributional_diffusion_decision_model_log_detail_entity:

    index : Optional[int] = None
    log_index : Optional[int] = None
    evidence_index : Optional[int] = None
    prompt : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None
    accuracy : Optional[bool] = None
    loss : Optional[float] = None

    partial_prompt_list: List = field(default_factory=list)
    partial_prompt_tokens_list: List = field(default_factory=list)

    def validate(self): 
        if self.index is None:
            raise Exception('index required')
        
        if self.evidence_index is None:
            raise Exception('evidence index required')

        if self.log_index is None:
            raise Exception('evidence index required')

        if self.token_count is None:
            raise Exception('token count required')

        if self.accuracy is None:
            raise Exception('accuracy required')

    def add_partial_prompt_list(self, tokenizer, response, prompt, prompt_token_ids, number_of_evidence): 
        token_ids = response.token_ids 
        token_count = len(response.logprobs)
        base = token_count // number_of_evidence
        remainder = token_count % number_of_evidence
        
        self.partial_prompt_list.append(prompt)
        self.partial_prompt_tokens_list.append(prompt_token_ids)
        start = 0
        for i in range(1, number_of_evidence):
            group_size = base + (1 if i < remainder else 0)
            partial_prompt_tokens_list = prompt_token_ids + list(token_ids[:start + group_size])
            partial_prompt = tokenizer.decode(partial_prompt_tokens_list, skip_special_tokens=True)

            self.partial_prompt_list.append(partial_prompt)
            self.partial_prompt_tokens_list.append(partial_prompt_tokens_list)
            
            start += group_size

