from src.logger.llm_response.llm_response_inference_logger import llm_response_inference_logger
from src.confidence.verbal.qwen3_8b.llm_generation_metacognitive_qwen3_8b import llm_generation_metacognitive_qwen3_8b
from src.utils.enums_class import confidence_type_enum


class llm_generation_metacognitive_settings_8(llm_generation_metacognitive_qwen3_8b): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def create_llm_response_logger(self, run_number) -> llm_response_inference_logger:
        return llm_response_inference_logger(log_file_name = f'src/confidence/verbal/qwen3_8b/settings_3/run_{run_number}/llm_generation_metacognitive_settings_8.csv')


t = llm_generation_metacognitive_settings_8(modelname='/home/hr_akbari/research/LLM_Consciousness_Confidence/live_logs/settings_8/checkpoint-1000-HF')
t.run(from_run_number=1, to_run_number=2, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=2, to_run_number=3, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=3, to_run_number=4, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=4, to_run_number=5, confidence_type = confidence_type_enum.LEVEL)