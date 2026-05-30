
class self_consistency_log_detail_entity:

    def __init__(self, index):
        self.index = index
        self.completion = None
        self.final_answer = None
        self.accuracy = None
        self.compared_final_answer = None
        self.token_count = None


    def get_index(self):
        return self.index

    def set_index(self, value):
        self.index = value

    def get_completion(self):
        return self.completion

    def set_completion(self, value):
        self.completion = value

    def get_final_answer(self):
        return self.final_answer

    def set_final_answer(self, value):
        self.final_answer = value

    def get_accuracy(self):
        return self.accuracy

    def set_accuracy(self, value):
        self.accuracy = value

    def get_compared_final_answer(self):
        return self.compared_final_answer

    def set_compared_final_answer(self, value):
        self.compared_final_answer = value

    def get_token_count(self):
        return self.token_count

    def set_token_count(self, value):
        self.token_count = value



