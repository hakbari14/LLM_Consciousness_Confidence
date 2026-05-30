class dataset_config: 

    def __init__(self, model_name):
        self.model_name = model_name
        self.max_prompt_length = None
        self.max_completion_length = None
        self.max_test_dataset_size = None
        self.ratio_test_dataset_size = None
    
    def validate(self): 
        if self.get_model_name() is None:
            raise Exception('model name is required')

    def get_model_name(self): 
        return self.model_name

    def set_model_name(self, value): 
        self.model_name = value
    
    def get_pipeline_type(self): 
        return self.pipeline_type

    def set_pipeline_type(self, value): 
        self.pipeline_type = value
    
    def get_max_prompt_length(self): 
        return self.max_prompt_length

    def set_max_prompt_length(self, value): 
        self.max_prompt_length = value
    
    def get_max_completion_length(self): 
        return self.max_completion_length

    def set_max_completion_length(self, value): 
        self.max_completion_length = value
    
    def get_max_test_dataset_size(self): 
        return self.max_test_dataset_size

    def set_max_test_dataset_size(self, value): 
        self.max_test_dataset_size = value
    
    def get_ratio_test_dataset_size(self): 
        return self.ratio_test_dataset_size

    def set_ratio_test_dataset_size(self, value): 
        self.ratio_test_dataset_size = value
    

        



