from src.logger.logger import logger


class inference_logger(logger): 

    def __init__(self, log_file_name, has_token_details = False):
        super().__init__(log_file_name, has_token_details=has_token_details)
        
    def add_to_buffer(self, log): 
        if self.buffer is None or log is None:
            return
        
        self.validate_log(log)

        filtered_list = list(filter(lambda x: x.equal(log) , self.buffer))
        if filtered_list is None or len(filtered_list) == 0:
            self.buffer.append(log)
            return 
        if len(filtered_list) > 1:
            raise Exception(f'Duplicate log{log}')




