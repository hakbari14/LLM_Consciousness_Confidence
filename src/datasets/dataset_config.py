class dataset_config: 

    def __init__(self, model_name : str) -> None :
        self.model_name : str = model_name
        self.max_prompt_length : int = None
        self.max_completion_length : int = None
        self.max_test_dataset_size : int = None
        self.ratio_test_dataset_size : float = None
        self.max_test_dataset_size_per_category : int = None
    
    def validate(self) -> None: 
        if self.get_model_name() is None:
            raise Exception('model name is required')

    def get_model_name(self) -> str: 
        return self.model_name

    def set_model_name(self, value : str) -> None: 
        self.model_name = value
    
    def get_max_prompt_length(self) -> int: 
        return self.max_prompt_length

    def set_max_prompt_length(self, value: int) -> None: 
        self.max_prompt_length = value
    
    def get_max_completion_length(self) -> int : 
        return self.max_completion_length

    def set_max_completion_length(self, value : int) -> None: 
        self.max_completion_length = value
    
    def get_max_test_dataset_size(self) -> int : 
        return self.max_test_dataset_size

    def set_max_test_dataset_size(self, value: int) -> None: 
        self.max_test_dataset_size = value
    
    def get_ratio_test_dataset_size(self) -> float : 
        return self.ratio_test_dataset_size

    def set_ratio_test_dataset_size(self, value: float) -> None: 
        self.ratio_test_dataset_size = value

    def get_max_test_dataset_size_per_category(self) -> int : 
        return self.max_test_dataset_size_per_category

    def set_max_test_dataset_size_per_category(self, value: int) -> None: 
        self.max_test_dataset_size_per_category = value
    

        



