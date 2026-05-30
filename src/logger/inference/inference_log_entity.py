from src.logger.log_entity import log_entity

class inference_log_entity(log_entity):

    def __init__(self, ID, sample_ID, problem_id, split, prompt, target):
        super().__init__(sample_ID, problem_id, split, prompt, target)
        self.ID = ID

    def validate(self): 
        super().validate()
        if self.ID is None:
            raise Exception('ID is required')

    def equal(self, x):
        return self.ID == x.get_ID()

    def get_ID(self):
        return self.ID

    def set_ID(self, value):
        self.ID = value


