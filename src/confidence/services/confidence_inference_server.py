from confidence.services.iit_calculation_services_entity import self_consistency_log_api_entity, self_consistency_log_res_api_entity
from confidence.services.confidence_inference_calculation import confidence_inference_calculation
from fastapi import FastAPI
from typing import List

app = FastAPI(title="IIT Calculation API")
calculator = confidence_inference_calculation('/home/hr_akbari/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1')

@app.post("/calculate_iit", response_model=List[self_consistency_log_res_api_entity])
def calculate_iit(log: self_consistency_log_api_entity):
    return calculator.calculate_iit(log)

