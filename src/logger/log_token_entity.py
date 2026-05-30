
class log_token_entity:
    
    def __init__(self, token_number, token):
        self.token_number = token_number
        self.token = token
        self.attended_embedding = None
        self.token_markov_chain_0_1_emb = None
        self.state_index = None
        self.iit_value = None
        self.iit_cause_state_index = None
        self.iit_effect_state_index = None

    def get_token_number(self):
        return self.token_number

    def set_token_number(self, value):
        self.token_number = value

    def get_token(self):
        return self.token

    def set_token(self, value):
        self.token = value

    def get_attended_embedding(self):
        return self.attended_embedding

    def set_attended_embedding(self, value):
        self.attended_embedding = value

    def get_token_markov_chain_0_1_emb(self):
        return self.token_markov_chain_0_1_emb

    def set_token_markov_chain_0_1_emb(self, value):
        self.token_markov_chain_0_1_emb = value

    def get_state_index(self):
        return self.state_index

    def set_state_index(self, value):
        self.state_index = value

    def get_iit_value(self):
        return self.iit_value

    def set_iit_value(self, value):
        self.iit_value = value

    def get_iit_cause_state_index(self):
        return self.iit_cause_state_index

    def set_iit_cause_state_index(self, value):
        self.iit_cause_state_index = value

    def get_iit_effect_state_index(self):
        return self.iit_effect_state_index

    def set_iit_effect_state_index(self, value):
        self.iit_effect_state_index = value

