from src.logger.inference.inference_log_entity import inference_log_entity

class self_consistency_log_entity(inference_log_entity):

    def __init__(self, ID, sample_ID, problem_id, split, prompt, target):
        super().__init__(ID, sample_ID, problem_id, split, prompt, target)
        self.pass_at_k = None
        self.consistency_list = []

    def validate(self): 
        super().validate()
        if self.consistency_list is None or len(self.consistency_list) == 0:
            raise Exception('self consistency list empty')

    def get_pass_at_k(self):
        return self.pass_at_k

    def set_pass_at_k(self, value):
        self.pass_at_k = value

    def get_consistency_list(self):
        return self.consistency_list

    def set_consistency_list(self, value):
        self.consistency_list = value

    def add_consistency_list(self, value):
        self.consistency_list.append(value)

    def clear_consistency_list(self):
        self.consistency_list = []

    def get_all_final_answers(self):
        if len(self.consistency_list) == 0:
            return None
        
        filtered_list = list(filter(lambda x: x.get_final_answer() is not None, self.consistency_list))
        return " ".join(map(lambda x: str(x.get_final_answer()), filtered_list))

