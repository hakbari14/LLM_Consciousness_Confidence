from abc import ABC, abstractmethod
import os
import csv


class logger(ABC): 

    def __init__(self, log_file_name, has_token_details = False):
        self.buffer = []

        self.log_file_name = log_file_name
        self.has_token_details = has_token_details
        if self.log_file_name is None:
            raise Exception('logfile name is required')
        self.token_log_file_name = log_file_name.replace('.csv', '_token.csv')

        self.create_and_prepare(self.log_file_name, self.get_fieldnames())
        if self.has_token_details: 
            self.create_and_prepare(self.token_log_file_name, self.get_token_fieldnames())

        
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

    def write_to_token_log_file(self): 
        if len(self.buffer) == 0:
            return 
        
        try:
            with open(self.token_log_file_name, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = self.get_token_fieldnames())
                writer.writerows(self.convert_token_buffer())
                csvfile.close()
        except Exception as e:
            print(f"[WARN] Could not logs to CSV: {e}")

    def write_attachments(self): 
        if self.has_token_details:
            self.write_to_token_log_file()

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

    @abstractmethod
    def add_to_buffer(self, log):
        pass

    @abstractmethod
    def convert_buffer(self): 
        pass

    @abstractmethod
    def get_fieldnames(self): 
        pass

    def convert_token_buffer(self): 
        list = []
        for log in self.buffer:
            for token_log in log.get_log_entity_list():
                b = { 
                    'ID': token_log.get_token_number(), 
                    'Sample_ID': log.get_sample_ID(), 
                    'Token': token_log.get_token(), 
                    'State': token_log.get_state_index(), 
                    'IIT_Value': token_log.get_iit_value(), 
                    }
                list.append(b)            
        return list

    def get_token_fieldnames(self): 
        return [ 
                'ID',
                'Sample_ID', 
                'Token', 
                'State', 
                'IIT_Value'
                ]

    def refine_log_filename(self, run_index):
        new_log_file_name = self.get_log_file_name().replace('/run/', f'/run{run_index}/')
        self.set_log_file_name(new_log_file_name)

    def get_log_file_name(self):
        return self.log_file_name

    def set_log_file_name(self, value):
        self.log_file_name = value
    