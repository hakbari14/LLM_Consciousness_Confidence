from abc import ABC, abstractmethod
import os
import csv


class logger(ABC): 

    def __init__(self, log_file_name: str) -> None:
        self.buffer = []
        self.log_file_name = log_file_name
        if self.log_file_name is None:
            raise Exception('logfile name is required')
        self.create_and_prepare(self.log_file_name, self.get_fieldnames())
        
    def write_to_log_file(self): 
        if len(self.buffer) == 0:
            return 
        
        try:
            with open(self.log_file_name, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = self.get_fieldnames())
                writer.writerows(self.convert_buffer())
                csvfile.close()
        except Exception as e:
            print(f"[WARN] Could not logs to CSV: {e}")

        self.write_attachments()    
        self.clear_buffer()
        return None


    def write_attachments(self) -> None: 
        return None
    
    def validate_log(self, log): 
        log.validate()

    def clear_buffer(self): 
        self.buffer = []

    def add_to_buffer_list(self, log_list): 
        for log in log_list:
            self.add_to_buffer(log)

    def create_and_prepare(self, filename, fieldnames): 
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        if not os.path.exists(filename):
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
                writer.writeheader()
                csvfile.close()

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

    @abstractmethod
    def convert_buffer(self): 
        pass

    @abstractmethod
    def get_fieldnames(self): 
        pass

    def get_log_file_name(self):
        return self.log_file_name

    def set_log_file_name(self, value):
        self.log_file_name = value
    