from src.logger.log_token_entity import log_token_entity

class log_entity:

    def __init__(self, sample_ID, problem_id, split, prompt, target):
        self.sample_ID = sample_ID
        self.problem_id = problem_id
        self.split = split
        self.prompt = prompt
        self.target = target
        self.completion = None
        self.token_count = None
        self.final_answer = None
        self.accuracy = None
        self.completion_embedding_shape = None
        self.completion_loss = None
        self.perplexity = None
        self.entropy = None
        self.token_count_for_reduced_dim = None
        self.reduced_dim = None
        self.phi_reward = None
        self.phi_reward_raw = None
        self.phi_reward_raw_actual = None
        self.tpm_loss = None
        self.tpm_entropy = None
        self.log_entity_list = []

    def validate(self): 
        if self.sample_ID is None:
            raise Exception('Sample ID is required')
        if self.prompt is None or len(self.prompt) == 0:
            raise Exception('prompt is required')
        if self.split is None:
            raise Exception('split is required')
        if self.target is None:
            raise Exception('target is required')

    def get_sample_ID(self):
        return self.sample_ID

    def set_sample_ID(self, value):
        self.sample_ID = value

    def get_problem_id(self):
        return self.problem_id

    def set_problem_id(self, value):
        self.problem_id = value

    def get_split(self):
        return self.split

    def set_split(self, value):
        self.split = value

    def get_prompt(self):
        return self.prompt

    def set_prompt(self, value):
        self.prompt = value

    def get_target(self):
        return self.target

    def set_target(self, value):
        self.target = value

    def get_completion(self):
        return self.completion

    def set_completion(self, value):
        self.completion = value

    def get_token_count(self):
        return self.token_count

    def set_token_count(self, value):
        self.token_count = value

    def get_final_answer(self):
        return self.final_answer

    def set_final_answer(self, value):
        self.final_answer = value

    def get_accuracy(self):
        return self.accuracy

    def set_accuracy(self, value):
        self.accuracy = value

    def get_completion_embedding_shape(self):
        return self.completion_embedding_shape

    def set_completion_embedding_shape(self, value):
        self.completion_embedding_shape = value

    def get_completion_loss(self):
        return self.completion_loss

    def set_completion_loss(self, value):
        self.completion_loss = value

    def get_perplexity(self):
        return self.perplexity

    def set_perplexity(self, value):
        self.perplexity = value

    def get_entropy(self):
        return self.entropy

    def set_entropy(self, value):
        self.entropy = value

    def get_token_count_for_reduced_dim(self):
        return self.token_count_for_reduced_dim

    def set_token_count_for_reduced_dim(self, value):
        self.token_count_for_reduced_dim = value

    def get_reduced_dim(self):
        return self.reduced_dim

    def set_reduced_dim(self, value):
        self.reduced_dim = value

    def get_phi_reward(self):
        return self.phi_reward

    def set_phi_reward(self, value):
        self.phi_reward = value

    def get_phi_reward_raw(self):
        return self.phi_reward_raw

    def set_phi_reward_raw(self, value):
        self.phi_reward_raw = value

    def get_phi_reward_raw_actual(self):
        return self.phi_reward_raw_actual

    def set_phi_reward_raw_actual(self, value):
        self.phi_reward_raw_actual = value

    def get_tpm_loss(self):
        return self.tpm_loss

    def set_tpm_loss(self, value):
        self.tpm_loss = value

    def get_tpm_entropy(self):
        return self.tpm_entropy

    def set_tpm_entropy(self, value):
        self.tpm_entropy = value

    def get_log_entity_list(self):
        return self.log_entity_list

    def set_log_entity_list(self, value):
        self.log_entity_list = value

    def add_log_token_list(self, value):
        self.log_entity_list.append(value)
