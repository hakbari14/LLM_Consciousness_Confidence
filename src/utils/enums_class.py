from enum import StrEnum

class dataset_element_type_enum(StrEnum):
    TRAIN = 'train'
    EVAL = 'eval'

class iit_log_type_enum(StrEnum):
    TRAIN_TEST = 'train_test'
    TEST = 'test'

class ii_calculation_type_enum(StrEnum):
    SUM = 'sum'
    MAX = 'max'

class ii_phi_type_enum(StrEnum):
    SYSTEM_PHI = 'system_phi'
    BIG_PHI = 'big_phi'

class tpm_creation_type_enum(StrEnum):
    TRAJECTORY = 'trajectory'
    PROMPT = 'prompt'
    BATCH = 'batch'

class iit_layer_type_enum(StrEnum):
    ALL = 'all'
    LAST = 'last'
    SOME = 'some'

class iit_threashold_type_enum(StrEnum):
    AVERAGE = 'average'
    MEDIAN = 'median'

class last_layer_computation_type_enum(StrEnum):
    TANH = 'tanh'
    EXP = 'exp'
    IDENTITY = 'identity'

class granularity_enum(StrEnum):
    CHUNK = 'chunk'
    TOKEN = 'token'

class training_type_enum(StrEnum):
    ACCURACY_REWARD = 'accuracy_reward'
    CONFIDENCE = 'confidence'
    CONFIDENCE_WITH_CRITERAI = 'confidence_with_criteria'
    ACCURACY_REWARD_CONFIDENCE = 'accuracy_reward_confidence'
    ACCURACY_REWARD_CONFIDENCE_WITH_CRITERAI = 'accuracy_reward_confidence_with_criteria'

class llm_pipeline_type_enum(StrEnum):
    TRAINING = 'training'
    INFERENCE = 'inference'

class confidence_type_enum(StrEnum):
    PROBABILITY = 'probability'
    LEVEL = 'level'

