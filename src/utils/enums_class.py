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
    BASELINE = 'baseline'
    IIT = 'iit'
    ENTROPY = 'entropy'

class llm_pipeline_type_enum(StrEnum):
    TRAINING = 'training'
    INFERENCE = 'inference'

